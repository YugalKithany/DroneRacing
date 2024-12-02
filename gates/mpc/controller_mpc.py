import numpy as np
from scipy.optimize import minimize
import casadi as ca

class MPCController:
    def __init__(self, prediction_horizon=10, control_horizon=5, dt=0.1, setpoint=[0, 0, 0], output_limit=10):
        """
        Args:
            prediction_horizon (int): Number of future steps to predict
            control_horizon (int): Number of control inputs to optimize
            dt (float): Time step for discretization
            setpoint (list): Initial target position [x, y, z]
            output_limit (float): Maximum control output magnitude
        """
        self.N = prediction_horizon
        self.control_horizon = control_horizon
        self.dt = dt
        self.setpoint = np.array(setpoint)
        self.output_limit = output_limit
        
        # States: [x, y, z, vx, vy, vz]
        # Controls: [ax, ay, az]
        self.nx = 6  # num states
        self.nu = 3  # num controls
        
        # Setup optimization problem
        self.setup_mpc()
        
        # Store prev sol for warm/non=fresh start
        self.prev_solution = None
        self.current_state = np.zeros(self.nx)

    def setup_mpc(self):
        # ini CasADi symbolic var
        self.opti = ca.Opti()
        
        # Decision var
        self.X = self.opti.variable(self.nx, self.N + 1)  # states
        self.U = self.opti.variable(self.nu, self.N)      # controls
        
        # Param to update
        self.P = self.opti.parameter(self.nx)     # current state
        self.Ref = self.opti.parameter(3)         # reference position
        # Cost matrices
        Q = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])  # state cost
        R = np.diag([1.0, 1.0, 1.0])                     # control cost
        
        # Objective function -- undersan better
        obj = 0
        for k in range(self.N):
            state_error = self.X[:3, k] - self.Ref
            obj += ca.mtimes(state_error.T, state_error) * 10
            obj += ca.mtimes(self.U[:, k].T, self.U[:, k])
            
        self.opti.minimize(obj)
        
        # Dynamic constraints
        for k in range(self.N):
            # double integrator model
            x_next = self.X[:3, k] + self.X[3:, k] * self.dt
            v_next = self.X[3:, k] + self.U[:, k] * self.dt
            
            self.opti.subject_to(self.X[:3, k+1] == x_next)
            self.opti.subject_to(self.X[3:, k+1] == v_next)
        
        self.opti.subject_to(self.X[:, 0] == self.P) # Initial condition
        self.opti.subject_to(self.opti.bounded(-self.output_limit, self.U, self.output_limit)) # Control constraints
         
        # Solver options, init to 0, see if tthey can improve later TODO 
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100}
        self.opti.solver('ipopt', opts)

    def update(self, current_position, dt):
        # Update current state estimate
        current_velocity = (current_position - self.current_state[:3]) / max(dt, 0.001)
        self.current_state = np.hstack([current_position, current_velocity])
        
        try:
            self.opti.set_value(self.P, self.current_state)
            self.opti.set_value(self.Ref, self.setpoint)
            
            # Solve optimization problem using casadi
            if self.prev_solution is not None:
                self.opti.set_initial(self.X, self.prev_solution['x'])
                self.opti.set_initial(self.U, self.prev_solution['u'])
            
            sol = self.opti.solve()
            
            # Store solution for warm/not-fresh start
            self.prev_solution = {
                'x': sol.value(self.X),
                'u': sol.value(self.U)
            }
            
            # Return first control action (converted to velocity) so that ideally it follows the prev of next inst? 
            control = sol.value(self.U)[:, 0]
            
        except:
            print("MPC failed, using extrme vals")
            error = self.setpoint - current_position
            control = np.clip(error, -self.output_limit, self.output_limit)
        
        velocity_command = control * dt
        return velocity_command

    def update_setpoint(self, setpoint):
        """Match PID interface for setpoint updates"""
        self.setpoint = np.array(setpoint)