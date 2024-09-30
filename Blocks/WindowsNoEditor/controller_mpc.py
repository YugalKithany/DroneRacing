import numpy as np
import scipy.optimize as opt

class MPCController:
    def __init__(self, dt, N, Q, R): # //q r not used for now
        self.dt = dt  # Time step
        self.N = N    # Prediction horizon
        self.Q = Q    # State cost matrix
        self.R = R    # Control cost matrix

    def state_update_nonlinear(self, x, u):
        """
            x Current state [pos_x, pos_y, vel_x, vel_y]
            u Control input [acc_x, acc_y]
        """
        x_next = np.zeros_like(x)
        x_next[0] = x[0] + self.dt * x[2]  # Position x update w velocity
        x_next[1] = x[1] + self.dt * x[3]  # Position y update w velocity
        x_next[2] = x[2] + self.dt * u[0]  # Velocity x update w acc input
        x_next[3] = x[3] + self.dt * u[1]  # Velocity y update w acc input
        return x_next

    def cost_function(self, u, x0, target):
        """
            u  Control inputs over the horizon, flattened (N, 2).
            x0  Initial state [pos_x, pos_y, vel_x, vel_y].
            target  Target state [pos_x, pos_y, 0, 0] (with zero velocity).
        """
        u = u.reshape(self.N, 2)  # reshape u to (N, 2) where N =horizon
        x_pred = np.zeros((4, self.N+1))
        x_pred[:, 0] = x0  # init state

        cost = 0
        for k in range(self.N):
            x_pred[:, k+1] = self.state_update_nonlinear(x_pred[:, k], u[k])
            # Penalize diff from a target state (position)
            cost += np.linalg.norm(x_pred[:, k+1] - target)**2
            # Penalize control input magnitude (effort), we want less control at all times
            cost += np.linalg.norm(u[k])**2

        return cost

    def nonlinear_constraints(self, u, x0, target):
        """
            u  Control inputs over the horizon, flattened (N, 2).
            x0  Initial state [pos_x, pos_y, vel_x, vel_y].
            target  Target state [pos_x, pos_y, 0, 0].
        """
        u = u.reshape(self.N, 2)  # Reshape u to (N, 2) where N is the horizon
        x_pred = np.zeros((4, self.N+1))
        x_pred[:, 0] = x0[:4]  # Initial state

        for k in range(self.N):
            x_pred[:, k+1] = self.state_update_nonlinear(x_pred[:, k], u[k])

        # Ensure that the final predicted state is close to the target state
        return x_pred[:, -1] - target[:4]

    def moveMPC(self, current_pos, current_vel, target_pos):

        # Current state [pos_x, pos_y, vel_x, vel_y]
        x0 = np.array([current_pos[0], current_pos[1], current_vel[0], current_vel[1]])
        
        # Target state [target_x, target_y, 0, 0] (target position with zero velocity)
        target = np.array([target_pos[0], target_pos[1], 0, 0])

        # Initialize control inputs (u) for the entire horizon
        u0 = np.zeros((self.N, 2))  # Initial guess: no control input

        # Set constraints and solve the optimization problem
        constraints = {'type': 'eq', 'fun': self.nonlinear_constraints, 'args': (x0, target)}

        # Solve the optimization problem
        result = opt.minimize(self.cost_function, u0.flatten(), args=(x0, target), 
                              constraints=constraints, method='SLSQP')

        # Extract the optimal control inputs from the optimization result
        u_opt = result.x.reshape((self.N, 2))

        # Return the first control input for the current time step
        return u_opt[0]





# import airsim
# import os
# import time
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# # from particle_main import RunParticle
# import traceback
# import random
# # from controller_m.gen_traj import Generate
# # from perception.perception import Perception
# # from simple_excitation import excitation

# from pyvista_visualiser import Perception_simulation
# import matplotlib.animation as animation


# def calculate_performance_metrics(position_history, settlingpoint, tolerance_percentage=2, timestep=None):
#     pos_adj = position_history - position_history[0]  # Adjust by init pos
#     settling_time=0
#     rise_time=0
#     # overshoot
#     peak = np.max(pos_adj)
#     Mp = 100 * (peak - settlingpoint) / settlingpoint  # Mp percentage

#     # Check if timestep is provided for time-based metrics
#     if timestep > -np.inf:
#     # settling time
#         tolerance = tolerance_percentage / 100 * settlingpoint
#         Ts_idx = np.where(np.abs(pos_adj - settlingpoint) <= tolerance)[0]
#         settling_time = Ts_idx[0] * timestep  # First time where pos is within tolerance

#     # rise time
#         Tr_idx = np.where(pos_adj >= 0.9 * settlingpoint)[0]
#         rise_time = Tr_idx[0] * timestep  # First time where 90% of settling point is reached
#     return Mp, settling_time, rise_time



# lead = "Drone_C"
# if __name__ == "__main__":


#     # connect to the AirSim simulator
#     client = airsim.MultirotorClient()
#     client.confirmConnection()
    

#     client.enableApiControl(True,lead)
#     client.armDisarm(True, lead)
#     client.takeoffAsync(20.0, lead).join()

#     total = client.getMultirotorState(lead).kinematics_estimated
#     current_pos = [client.getMultirotorState(lead).kinematics_estimated.position.x_val, client.getMultirotorState(lead).kinematics_estimated.position.y_val, client.getMultirotorState(lead).kinematics_estimated.position.z_val]
#     print("kinematics",total)

 
#     count = 0
#     totalcount = 1000
#     timestep = 0.01  # Time step
#     dt = 0.1
#     position_history = []

#     gain_x = [20,0,100,0]
#     gain_y = [20,0,100.0]
#     gain_z = [2,0,20.0]
#     setpoint = [1,1,35]
#     pid_controller = PIDController(gain_x=gain_x, gain_y=gain_y, gain_z=gain_z, setpoint=setpoint)
    
#     # Initialize PID controller with desired gains and setpoint
#     print("current pos",current_pos)
#     world = [client.simGetObjectPose(lead).position.x_val,client.simGetObjectPose(lead).position.y_val,client.simGetObjectPose(lead).position.z_val] 
#     print("world pose, ",world  )

#     step_target = [current_pos[0]+5,current_pos[1]+5,current_pos[2]+5]
#     # step_target = [10,10,10]
#     print("step target ",step_target)


#     try:
#         while True:
#             client = airsim.MultirotorClient()
#             client.confirmConnection()
#             current_velocity = [client.getMultirotorState(lead).kinematics_estimated.linear_velocity.x_val, client.getMultirotorState(lead).kinematics_estimated.linear_velocity.y_val,client.getMultirotorState(lead).kinematics_estimated.linear_velocity.z_val]
            
#             current_pos = np.array([client.getMultirotorState(lead).kinematics_estimated.position.x_val, client.getMultirotorState(lead).kinematics_estimated.position.y_val, client.getMultirotorState(lead).kinematics_estimated.position.z_val])
    
#             # Compute control signal using PID controller
#             control_signal = pid_controller.update(current_pos, dt)
            
#             # Update quadrotor velocity using control signal
#             current_velocity[0] += control_signal[0] * dt
#             current_velocity[1] += control_signal[1] * dt
#             current_velocity[2] += control_signal[2] * dt
        
#             # client.moveByVelocityZBodyFrameAsync(current_velocity[0],current_velocity[1],10, timestep, vehicle_name = lead)
#             client.moveByVelocityAsync(current_velocity[0],current_velocity[1],current_velocity[2], timestep, vehicle_name = lead)
            

#             count += 1

#             time.sleep(timestep)

#             if count == totalcount:
#                 break

#             position_history.append(current_pos)
        

#         client.reset()
#         client.armDisarm(False)
#         client.enableApiControl(False)

#         position_history=np.array(position_history)
#         times = np.arange(0,position_history.shape[0])*timestep
#         fig, (posx,posy,posz) = plt.subplots(3, 1, figsize=(14, 10))

#         posx.plot(times, position_history[:,0], label = "Pos x")
#         posx.legend()
#         posy.plot(times, position_history[:,1], label = "Pos y")
#         posy.legend()
#         posz.plot(times, position_history[:,2], label = "Pos z")    
#         posy.legend()

#         plt.show()



#     except Exception as e:
#         client = airsim.MultirotorClient()
#         client.confirmConnection()
#         print("Error Occured, Canceling: ",e)
#         traceback.print_exc()

#         client.reset()
#         client.armDisarm(False)

#         # that's enough fun for now. let's quit cleanly
#         client.enableApiControl(False)