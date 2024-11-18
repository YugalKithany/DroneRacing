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
        
        # State space model matrices (simple double integrator model)
        # States: [x, y, z, vx, vy, vz]
        # Controls: [ax, ay, az]
        self.nx = 6  # number of states
        self.nu = 3  # number of controls
        
        # Setup optimization problem
        self.setup_mpc()
        
        # Store previous solution for warm start
        self.prev_solution = None
        self.current_state = np.zeros(self.nx)

    def setup_mpc(self):
        # CasADi symbolic variables
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.N + 1)  # states
        self.U = self.opti.variable(self.nu, self.N)      # controls
        
        # Parameters that can be updated
        self.P = self.opti.parameter(self.nx)     # current state
        self.Ref = self.opti.parameter(3)         # reference position
        
        # Cost matrices
        Q = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])  # state cost
        R = np.diag([1.0, 1.0, 1.0])                     # control cost
        
        # Objective function
        obj = 0
        for k in range(self.N):
            state_error = self.X[:3, k] - self.Ref
            obj += ca.mtimes(state_error.T, state_error) * 10
            obj += ca.mtimes(self.U[:, k].T, self.U[:, k])
            
        self.opti.minimize(obj)
        
        # Dynamic constraints
        for k in range(self.N):
            # Simple double integrator model
            x_next = self.X[:3, k] + self.X[3:, k] * self.dt
            v_next = self.X[3:, k] + self.U[:, k] * self.dt
            
            self.opti.subject_to(self.X[:3, k+1] == x_next)
            self.opti.subject_to(self.X[3:, k+1] == v_next)
        
        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.P)
        
        # Control constraints
        self.opti.subject_to(self.opti.bounded(-self.output_limit, self.U, self.output_limit))
        
        # Solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100}
        self.opti.solver('ipopt', opts)

    def update(self, current_position, dt):
        """
        Update method that matches PID interface
        Returns control signals (velocities) based on current position
        """
        # Update current state estimate
        current_velocity = (current_position - self.current_state[:3]) / max(dt, 0.001)
        self.current_state = np.hstack([current_position, current_velocity])
        
        try:
            # Set parameters
            self.opti.set_value(self.P, self.current_state)
            self.opti.set_value(self.Ref, self.setpoint)
            
            # Solve optimization problem
            if self.prev_solution is not None:
                self.opti.set_initial(self.X, self.prev_solution['x'])
                self.opti.set_initial(self.U, self.prev_solution['u'])
            
            sol = self.opti.solve()
            
            # Store solution for warm start
            self.prev_solution = {
                'x': sol.value(self.X),
                'u': sol.value(self.U)
            }
            
            # Return first control action (converted to velocity)
            control = sol.value(self.U)[:, 0]
            
        except:
            # Fallback in case optimization fails
            print("MPC optimization failed, using fallback control")
            error = self.setpoint - current_position
            control = np.clip(error, -self.output_limit, self.output_limit)
        
        # Convert acceleration commands to velocity commands to match PID interface
        velocity_command = control * dt
        return velocity_command

    def update_setpoint(self, setpoint):
        """Match PID interface for setpoint updates"""
        self.setpoint = np.array(setpoint)