import airsim
import numpy as np
import time
import os
import cv2
from controller_pid import PIDController
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import math
import json
from datetime import datetime

# Waypoints for drone navigation waypoints_qualifier1.yaml from rovery script TODO-Remake this, ensure it is center of gate
WAYPOINTS_Qualifier1 = [
    [10.388, 80.774, -43.580], [18.110, 76.260, -43.580], [25.434, 66.287, -43.580],
    [30.066, 56.550, -43.580], [32.801, 45.931, -43.580], [30.503, 38.200, -43.580],
    [3.264, 37.569, -43.580], [-17.863, 45.418, -46.580], [-15.494, 63.187, -52.080],
    [-6.321, 78.212, -55.780], [ 5.144, 82.385, -55.780], [14.559, 84.432, -55.180],
    [22.859, 82.832, -32.080], [38.259, 78.132, -31.380], [51.059, 52.132, -25.880],
    [44.959, 38.932, -25.880], [25.959, 26.332, -17.880], [11.659, 26.332, -13.780],
    [-10.141, 22.632, -6.380], [-23.641, 10.132, 2.120]
]

WAYPOINTS_ANGLES_Qualifier1 = [
    -15, -45, -60, -90 ,-115, 0, 0,  -45,  -90 , -135, -180, -180, -30, -45, -60, -135, -180, -180, -195, -25 ]


WAYPOINTS = [[0.0, 2.0, 2.0199999809265137], [1.5999999046325684, 10.800000190734863, 2.0199999809265137], 
             [8.887084007263184, 18.478761672973633, 2.0199999809265137], [18.74375343322754, 22.20650863647461, 2.0199999809265137],
             [30.04375457763672, 22.20648956298828, 2.0199999809265137], [39.04375457763672, 19.206478118896484, 2.0199999809265137],
            [45.74375534057617, 11.706478118896484, 2.0199999809265137], [45.74375534057617, 2.2064781188964844, 2.0199999809265137], 
            [40.343753814697266, -4.793521404266357, 2.0199999809265137], [30.74375343322754, -7.893521785736084,2.0199999809265137],
            [18.54375457763672, -7.893521785736084, 2.0199999809265137], [9.543754577636719, -5.093521595001221, 2.0199999809265137]]

WAYPOINTS_ANGLES = [
    15, 30, 45,60,75,90,120,135,150,165,180,195]

# WAYPOINTS = [
#     [-6.321, 78.212, -55.780], [5.144, 82.385, -55.780], [14.559, 84.432, -55.180],
#     [22.859, 82.832, -32.080], [38.259, 78.132, -31.380], [51.059, 52.132, -25.880],
#     [44.959, 38.932, -25.880], [25.959, 26.332, -19.880], [11.659, 26.332, -12.780],
#     [-10.141, 22.632, -6.380], [-24.641, 9.132, 2.120]
# ]



# 			"X":10.388,
# 			"Y": 80.774,
# 			"Z": -43.580,
# 4 cooked, 5 left too much, 
class simulation():
    def __init__(self, totalcount=50):
        self.lead = "Drone_L"
        self.client1 = airsim.MultirotorClient()
        self.client1.confirmConnection()
        self.client1.enableApiControl(True,self.lead)
        self.client1.armDisarm(True, self.lead)
        self.client1.takeoffAsync(30.0, self.lead).join()

        # Find Difference between global to NED coordinate frames, from last sem idk if still needed for agent/particale filter/yolo
        lead_pose = self.client1.simGetObjectPose(self.lead).position
        lead_global = [lead_pose.x_val, lead_pose.y_val,lead_pose.z_val]
        lead_pose = self.client1.simGetVehiclePose(self.lead).position
        lead_NED = [lead_pose.x_val, lead_pose.y_val,lead_pose.z_val]
        self.lead_coord_diff = np.array(lead_NED) - np.array(lead_global)
        # print(lead_pose)
        # self.mcl = RunParticle(starting_state=lead_global)   
        # Initialize mcl Position
        self.est_states = np.zeros((len(self.mcl.ref_traj) ,6)) # x y z vx vy vz
        self.gt_states  = np.zeros((len(self.mcl.ref_traj) ,16))
        self.PF_history_x = []
        self.PF_history_y = []
        self.PF_history_z = []
        self.PF_history_x.append(np.array(self.mcl.filter.particles['position'][:,0]).flatten())
        self.PF_history_y.append(np.array(self.mcl.filter.particles['position'][:,1]).flatten())
        self.PF_history_z.append(np.array(self.mcl.filter.particles['position'][:,2]).flatten())
        self.velocity_GT = []
        self.accel_GT = []
        self.global_state_history_L=[]
        self.global_state_history_C=[]
        self.particle_state_est=[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]

        # Assume constant time step between trajectory stepping
        self.timestep = 0.01
        self.totalcount = totalcount
        self.start_time = time.time()

# Coordinates of start and end gates
START_POS = [6.3, 81, -43]
END_POS = [-24.641, 9.132, 2.120]
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
# PID controller setup
gain_x = [3, 0, 8.0]  # Reduced from [5, 0, 10.0]
gain_y = [3, 0, 8.0]  # Reduced from [5, 0, 10.0]
gain_z = [1, 0, 5.0]  # Keep the same
pid = PIDController(gain_x=gain_x, gain_y=gain_y, gain_z=gain_z)

def move_by_waypoints():
    global drone_path
    drone_path = []
    airsim_waypoints = [airsim.Vector3r(wp[0], wp[1], wp[2]) for wp in WAYPOINTS]
    client.moveOnPathAsync(airsim_waypoints, velocity=5, drivetrain=airsim.DrivetrainType.ForwardOnly, 
                           yaw_mode=airsim.YawMode(False, 0), lookahead=-1, adaptive_lookahead=1)
    start_time = time.time()
    while time.time() - start_time < len(WAYPOINTS) * 5:
        pos = client.getMultirotorState().kinematics_estimated.position
        drone_path.append([pos.x_val, pos.y_val, pos.z_val])
        time.sleep(1)  # Save coordinates every second

    # for wp in WAYPOINTS:
    #     start_time = time.time()
    #     # client.moveToPositionAsync(wp[0], wp[1], wp[2], 5).join()
        # while time.time() - start_time < 5:  
        #     pos = client.getMultirotorState().kinematics_estimated.position
        #     drone_path.append([pos.x_val, pos.y_val, pos.z_val])
        #     time.sleep(1)  # Save coordinates every second


    print("Drone reached all waypoints.")


def calculate_yaw_angle(current_pos, target_pos):
    # Calculate the yaw angle based on the direction vector to the waypoint
    delta_x = target_pos[0] - current_pos[0]
    delta_y = target_pos[1] - current_pos[1]
    yaw = math.atan2(delta_y, delta_x)  # Calculate yaw in radians
    yaw_deg = math.degrees(yaw)         # Convert to degrees
    return yaw_deg




def state_based_pid_control(pidC):
    global drone_path
    drone_path = []
    gate_clearance_positions = []  # Store positions where gates were cleared
    collision_count = 0
    run_start_time = time.time()
    for i, wp in enumerate(WAYPOINTS):
        print(f"Target waypoint: {wp}")
        current_pos = client.getMultirotorState().kinematics_estimated.position
        pidC.update_setpoint(wp)
        
        final_approach_velocity = [0, 0, 0]
        stuck_timestamp = time.time()
        while not np.allclose([current_pos.x_val, current_pos.y_val, current_pos.z_val], wp, atol=1.5):
            current_coords = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])
            drone_path.append([current_pos.x_val, current_pos.y_val, current_pos.z_val])
            
            control_signal = pidC.update(current_coords, dt=1)
            control_signal = np.clip(control_signal, -5, 5)
            
            yaw_angle = calculate_yaw_angle(current_coords, wp)
            client.moveByVelocityAsync(
                control_signal[0]/2, 
                control_signal[1]/2, 
                control_signal[2]/5, 
                0.5,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, yaw_angle)
            ).join()
            
            collision_info = client.simGetCollisionInfo()
            if collision_info.has_collided:
                collision_count += 1 

            final_approach_velocity = [control_signal[0]/5, control_signal[1]/5, control_signal[2]/5]
            current_pos = client.getMultirotorState().kinematics_estimated.position
            start_time = time.time()
            if(start_time - stuck_timestamp > 10 ):
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(wp[0], wp[1], wp[2]), airsim.to_quaternion(0, 0, 0)), True)
                print("fixing stuck")

            print("TIME:" , start_time - run_start_time)
            if(start_time - run_start_time > 120): 
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(6.788, 81.6774, -43.380), 
                                    airsim.to_quaternion(0, 0, 0)), True)
                return np.array(gate_clearance_positions), collision_count, start_time - run_start_time

        if i < len(WAYPOINTS) - 1:
            print("Clearing gate...")
            # Store position at gate clearance
            current_pos = client.getMultirotorState().kinematics_estimated.position
            gate_clearance_positions.append([current_pos.x_val, current_pos.y_val, current_pos.z_val])
            
            clearance_time = 1.5
            # if(i == 12 ): clearance_time =4
            start_time = time.time()
            # print("TIME:" , start_time - run_start_time)
            if(start_time - run_start_time > 240): 
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(6.788, 81.6774, -43.380), 
                                    airsim.to_quaternion(0, 0, 0)), True)
                return np.array(gate_clearance_positions), collision_count, start_time - run_start_time

            while time.time() - start_time < clearance_time:
                current_pos = client.getMultirotorState().kinematics_estimated.position
                drone_path.append([current_pos.x_val, current_pos.y_val, current_pos.z_val])
                
                client.moveByVelocityAsync(
                    final_approach_velocity[0],
                    final_approach_velocity[1],
                    final_approach_velocity[2],
                    0.1,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(False, yaw_angle)
                ).join()

                collision_info = client.simGetCollisionInfo()
                if collision_info.has_collided:
                    collision_count += 1

    end_time = time.time()
    total_runtime = end_time - run_start_time
    print("Completed all waypoints")
    return np.array(gate_clearance_positions), collision_count, total_runtime

def plot_gate_errors(gate_positions, waypoints):
    gate_positions = np.array(gate_positions)
    waypoints = np.array(waypoints[:-1])  # Exclude last waypoint as it's not a gate
    
    # Calculate errors
    min_rows = min(gate_positions.shape[0], waypoints.shape[0])
    errors = np.sqrt(np.sum((gate_positions[:min_rows] - waypoints[:min_rows])**2, axis=1))
    percent_errors = (errors / np.sqrt(np.sum(waypoints**2, axis=1))) * 100
    
    # Create error plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(percent_errors)), percent_errors, 'bo-', linewidth=2)
    plt.xlabel('Gate Number')
    plt.ylabel('Percent Error (%)')
    plt.title('Gate Clearance Error vs Gate Number')
    plt.grid(True)
    plt.savefig('gate_errors.png')
    plt.show()

# 			"X":10.388,
# 			"Y": 80.774,
# 			"Z": -43.580,
def get_sim_picture():
    responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
    img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    return img_rgb

def get_coords_vision(image):
    # This is where some Agent processes the image, and returns me data @robert looking into what it will return, and how i can use it
    return WAYPOINTS[0]  # Replace this with real processing

def vision_based_navigation():
    for _ in range(len(WAYPOINTS)):  # Loop through as many gates as there are waypoints
        client.simPause(True)
        image = get_sim_picture()
        coords = get_coords_vision(image)
        print(f"Vision-based coords: {coords}")
        
        # Resume simulation and navigate to new coordinates
        client.simPause(False)
        client.moveByVelocityAsync(coords[0], coords[1], coords[2], 5).join()

# def plot_3d_path(drone_path, waypoints):
#     # Convert paths and waypoints to numpy arrays
#     drone_path = np.array(drone_path)
#     waypoints = np.array(waypoints)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter3d(
#         x=drone_path[:, 0],
#         y=drone_path[:, 1],
#         z=drone_path[:, 2],
#         mode='lines',
#         name='Drone Path',
#         line=dict(color='blue', width=5)
#     ))
#     fig.add_trace(go.Scatter3d(
#         x=waypoints[:, 0],
#         y=waypoints[:, 1],
#         z=waypoints[:, 2],
#         mode='markers',
#         name='Waypoints',
#         marker=dict(color='red', size=5)
#     ))
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z'
#         ),
#         title="3D Drone Path and Waypoints"
#     )
#     fig.write_html("3d_drone_path.html")
#     fig.show()
def plot_3d_path(drone_path, waypoints):
    # Convert paths and waypoints to numpy arrays
    drone_path = np.array(drone_path)
    waypoints = np.array(waypoints)
    
    fig = go.Figure()
    
    # Plot drone path
    fig.add_trace(go.Scatter3d(
        x=drone_path[:, 0],
        y=drone_path[:, 1],
        z=-drone_path[:, 2],  # Flip Z coordinate
        mode='lines',
        name='Drone Path',
        line=dict(color='blue', width=5)
    ))
    
    # Plot waypoints
    fig.add_trace(go.Scatter3d(
        x=waypoints[:, 0],
        y=waypoints[:, 1],
        z=-waypoints[:, 2],  # Flip Z coordinate
        mode='markers',
        name='Waypoints',
        marker=dict(color='red', size=5)
    ))
    
    # Add gates around waypoints
    gate_width_x = 2  # Thin in X direction
    gate_width_y = 5  # Width in Y direction
    gate_height = 5   # Height in Z direction
    
    for wp in waypoints:
        # Create the vertices of a box centered on the waypoint
        x = [wp[0] - gate_width_x/2, wp[0] + gate_width_x/2]
        y = [wp[1] - gate_width_y/2, wp[1] + gate_width_y/2]
        z = [-wp[2] - gate_height/2, -wp[2] + gate_height/2]  # Flip Z coordinate
        
        # Create the lines for the gate
        for i in range(2):
            for j in range(2):
                # Vertical lines
                fig.add_trace(go.Scatter3d(
                    x=[x[i], x[i]], y=[y[j], y[j]], z=[z[0], z[1]],
                    mode='lines',
                    line=dict(color='green', width=2),
                    showlegend=False
                ))
                # Horizontal lines at top and bottom
                for k in range(2):
                    fig.add_trace(go.Scatter3d(
                        x=[x[0], x[1]], y=[y[i], y[i]], z=[z[k], z[k]],
                        mode='lines',
                        line=dict(color='green', width=2),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=[x[j], x[j]], y=[y[0], y[1]], z=[z[k], z[k]],
                        mode='lines',
                        line=dict(color='green', width=2),
                        showlegend=False
                    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'  # This ensures the axes are scaled properly
        ),
        title="3D Drone Path and Waypoints"
    )
    
    fig.write_html("3d_drone_path.html")
    fig.show()





def run_pid_experiment(experiment_name, gain_configurations):
    """
    Run multiple PID experiments with different gains and save results
    
    Args:
        experiment_name (str): Name of the experiment
        gain_configurations (list): List of dictionaries containing gain configurations
                                  [{'gain_x': [kp, ki, kd], 'gain_y': [kp, ki, kd], 'gain_z': [kp, ki, kd]}]
    """
    results = []
    
    for i, gains in enumerate(gain_configurations):
        print(f"\nRunning configuration {i+1}/{len(gain_configurations)}")
        print(f"Gains: {gains}")
        # time.sleep(5)  # Give time for reset
        

        # Initialize PID controller with current gains
        pidC = PIDController(
            gain_x=gains['gain_x'],
            gain_y=gains['gain_y'],
            gain_z=gains['gain_z']
        )
        
        # Reset drone position
        client.reset()
        # time.sleep(5)  # Give time for reset
        client.enableApiControl(True)
        client.armDisarm(True)
        # client.takeoffAsync(5).join()
        target_x=6.788
        target_y=81.6774
        target_z =-43.380
        time.sleep(3)
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(target_x, target_y, target_z), airsim.to_quaternion(0, 0, 0)), True)
        print("Hey")
        client.takeoffAsync(5).join()

        # Run the controller
        gate_positions = state_based_pid_control(pidC)
        
        # Save the results
        experiment_data = {
            'config_id': i,
            'gains': gains,
            'drone_path': drone_path,
            'gate_positions': gate_positions.tolist(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        results.append(experiment_data)
        
        # Save after each run in case of crashes
        save_experiment_results(experiment_name, results)
        
        # Plot results for this configuration
        plot_3d_path(drone_path, WAYPOINTS)
        plot_gate_errors(gate_positions, WAYPOINTS)
        
        # Wait between runs
        time.sleep(5)
    
    return results

def save_experiment_results(experiment_name, results):
    """Save experiment results to a JSON file"""
    filename = f"pid_experiment_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def analyze_experiment_results(experiment_file):
    """Analyze and plot results from a saved experiment"""
    with open(experiment_file, 'r') as f:
        results = json.load(f)
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    for run in results:
        gate_positions = np.array(run['gate_positions'])
        waypoints = np.array(WAYPOINTS[:-1])
        errors = np.sqrt(np.sum((gate_positions - waypoints)**2, axis=1))
        percent_errors = (errors / np.sqrt(np.sum(waypoints**2, axis=1))) * 100
        
        label = f"Kp={run['gains']['gain_x'][0]}, Ki={run['gains']['gain_x'][1]}, Kd={run['gains']['gain_x'][2]}"
        plt.plot(range(len(percent_errors)), percent_errors, '-o', label=label)
    
    plt.xlabel('Gate Number')
    plt.ylabel('Percent Error (%)')
    plt.title('Gate Clearance Error Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('gain_comparison.png')
    plt.show()



def main():
    target_x=6.788
    target_y=81.6774
    target_z =-43.380
    time.sleep(3)
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(target_x, target_y, target_z), airsim.to_quaternion(0, 0, 0)), True)
    print("Hey")
    client.takeoffAsync(5).join()
    
    # print("Baseline Waypoint Navigation")
    # move_by_waypoints()
    # plot_3d_path(drone_path, WAYPOINTS)

    print("State-Based PID Control")
    gate_clearance_positions = state_based_pid_control()
    plot_3d_path(drone_path, WAYPOINTS)
    # plot_gate_errors(gate_clearance_positions, WAYPOINTS)

    # # Vision-based approach
    # print("Vision-Based Navigation")
    # vision_based_navigation()
    
    # Land after finishing
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)



# def plot_combined_gate_errors(all_gate_positions, gain_configurations, waypoints):
#     """Plot gate errors for all PID gains on the same graph"""
#     plt.figure(figsize=(12, 8))
    
#     for i, gate_positions in enumerate(all_gate_positions):
#         gate_positions = np.array(gate_positions)
#         waypoints_arr = np.array(waypoints[:-1])  # Exclude last waypoint
        
#         # Calculate errors
#         min_rows = min(gate_positions.shape[0], waypoints_arr.shape[0])
#         errors = np.sqrt(np.sum((gate_positions[:min_rows] - waypoints_arr[:min_rows])**2, axis=1))
#         percent_errors = (errors / np.sqrt(np.sum(waypoints_arr**2, axis=1))) * 100
        
#         # Create label from gain configuration
#         gains = gain_configurations[i]
#         label = f"Kp={gains['gain_x'][0]}, Ki={gains['gain_x'][1]}, Kd={gains['gain_x'][2]}"
        
#         plt.plot(range(len(percent_errors)), percent_errors, '-o', linewidth=2, label=label)
    
#     plt.xlabel('Gate Number')
#     plt.ylabel('Percent Error (%)')
#     plt.title('Gate Clearance Error vs Gate Number - Gain Comparison')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('combined_gate_errors.png')
#     plt.show()

# def plot_combined_3d_paths(all_drone_paths, all_gate_positions, gain_configurations, waypoints):
#     """Plot all 3D paths on the same graph"""
#     fig = go.Figure()
    
#     # Different colors for different gain configurations
#     # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'pink', 'purple']  # Add more colors if needed
#     colors = [
#     'blue', 'red', 'green', 'yellow', 'orange', 'pink', 'purple', 
#     'brown', 'black', 'white', 'gray', 'cyan', 'magenta', 'violet', 
#     'indigo', 'beige', 'salmon', 'turquoise', 'teal', 'lime', 'peach', 
#     'lavender', 'gold', 'silver', 'bronze', 'emerald', 'crimson', 'navy', 
#     'charcoal', 'mint', 'coral', 'fuchsia', 'amber', 'plum', 'chartreuse', 
#     'ivory', 'ruby', 'sapphire', 'topaz', 'tan', 'maroon', 'khaki', 'azure', 
#     'apricot', 'pearl', 'lavenderblush', 'orchid', 'periwinkle', 'rosy', 
#     'sepia', 'mauve', 'sage', 'lemon', 'honeydew', 'cobalt', 'magenta', 
#     'snow']
#     # Plot drone paths for each configuration
#     for i, drone_path in enumerate(all_drone_paths):
#         drone_path = np.array(drone_path)
#         gains = gain_configurations[i]
#         label = f"Kp={gains['gain_x'][0]}, Ki={gains['gain_x'][1]}, Kd={gains['gain_x'][2]}"
        
#         fig.add_trace(go.Scatter3d(
#             x=drone_path[:, 0],
#             y=drone_path[:, 1],
#             z=-drone_path[:, 2],
#             mode='lines',
#             name=label,
#             line=dict(color=colors[i%50], width=3)
#         ))
    
#     # Plot waypoints
#     waypoints = np.array(waypoints)
#     fig.add_trace(go.Scatter3d(
#         x=waypoints[:, 0],
#         y=waypoints[:, 1],
#         z=-waypoints[:, 2],
#         mode='markers',
#         name='Waypoints',
#         marker=dict(color='black', size=5)
#     ))
    
#     # Add gates around waypoints
#     gate_width_x = 2
#     gate_width_y = 5
#     gate_height = 5
    
#     for wp in waypoints:
#         x = [wp[0] - gate_width_x/2, wp[0] + gate_width_x/2]
#         y = [wp[1] - gate_width_y/2, wp[1] + gate_width_y/2]
#         z = [-wp[2] - gate_height/2, -wp[2] + gate_height/2]
        
#         for i in range(2):
#             for j in range(2):
#                 fig.add_trace(go.Scatter3d(
#                     x=[x[i], x[i]], y=[y[j], y[j]], z=[z[0], z[1]],
#                     mode='lines',
#                     line=dict(color='gray', width=2),
#                     showlegend=False
#                 ))
#                 for k in range(2):
#                     fig.add_trace(go.Scatter3d(
#                         x=[x[0], x[1]], y=[y[i], y[i]], z=[z[k], z[k]],
#                         mode='lines',
#                         line=dict(color='gray', width=2),
#                         showlegend=False
#                     ))
#                     fig.add_trace(go.Scatter3d(
#                         x=[x[j], x[j]], y=[y[0], y[1]], z=[z[k], z[k]],
#                         mode='lines',
#                         line=dict(color='gray', width=2),
#                         showlegend=False
#                     ))
    
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z',
#             aspectmode='data'
#         ),
#         title="3D Drone Paths Comparison - Multiple PID Gains"
#     )
    
#     fig.write_html("combined_3d_paths.html")
#     fig.show()




def plot_combined_3d_paths(all_drone_paths, all_gate_positions, gain_configurations, waypoints, waypoints_angles):
    """Plot all 3D paths on the same graph, with gates tilted based on waypoints_angles."""
    fig = go.Figure()
    
    # Colors for different gain configurations
    colors = [
    'black', 'blanchedalmond', 'blue',
    'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey',
    'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
    'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
    'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
    'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
    'lightgoldenrodyellow', 'lightgray', 'lightgrey',
    'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
    'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
    'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple',
    'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
    'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
    'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
    'plum', 'powderblue', 'purple', 'red', 'rosybrown',
    'royalblue', 'rebeccapurple', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver',
    'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
    'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
    'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
    'yellow', 'yellowgreen'
    ]

    # Plot drone paths for each configuration
    for i, drone_path in enumerate(all_drone_paths):
        drone_path = np.array(drone_path)
        gains = gain_configurations[i]
        label = f"Kp={gains['gain_x'][0]}, Ki={gains['gain_x'][1]}, Kd={gains['gain_x'][2]}"
        
        fig.add_trace(go.Scatter3d(
            x=drone_path[:, 0],
            y=drone_path[:, 1],
            z=-drone_path[:, 2],
            mode='lines',
            name=label,
            line=dict(color=colors[i % len(colors)], width=3)
        ))
    
    # Plot waypoints
    waypoints = np.array(waypoints)
    fig.add_trace(go.Scatter3d(
        x=waypoints[:, 0],
        y=waypoints[:, 1],
        z=-waypoints[:, 2],
        mode='markers',
        name='Waypoints',
        marker=dict(color='black', size=5)
    ))
    
    # Parameters for gates
    gate_width_x = 2
    gate_width_y = 5
    gate_height = 5
    
    # Add gates around waypoints
    for wp, angle in zip(waypoints, waypoints_angles):
        theta = np.deg2rad(angle)  # Convert angle to radians
        
        # Rotation matrix for Z-axis rotation
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Define gate corners
        corners = np.array([
            [-gate_width_x/2, -gate_width_y/2, -gate_height/2],
            [ gate_width_x/2, -gate_width_y/2, -gate_height/2],
            [-gate_width_x/2,  gate_width_y/2, -gate_height/2],
            [ gate_width_x/2,  gate_width_y/2, -gate_height/2],
            [-gate_width_x/2, -gate_width_y/2,  gate_height/2],
            [ gate_width_x/2, -gate_width_y/2,  gate_height/2],
            [-gate_width_x/2,  gate_width_y/2,  gate_height/2],
            [ gate_width_x/2,  gate_width_y/2,  gate_height/2]
        ])
        
        # Rotate and translate corners
        rotated_corners = (rotation_matrix @ corners.T).T + wp
        
        # Define lines between rotated corners
        line_indices = [
            (0, 1), (0, 2), (1, 3), (2, 3),  # Bottom face
            (4, 5), (4, 6), (5, 7), (6, 7),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges
        ]
        
        for start, end in line_indices:
            fig.add_trace(go.Scatter3d(
                x=[rotated_corners[start, 0], rotated_corners[end, 0]],
                y=[rotated_corners[start, 1], rotated_corners[end, 1]],
                z=[-rotated_corners[start, 2], -rotated_corners[end, 2]],  # Flip Z axis
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="3D Drone Paths Comparison - Multiple PID Gains with Tilted Gates"
    )
    
    fig.write_html("combined_3d_paths_with_tilted_gates.html")
    fig.show()


def convert_ndarray_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    else:
        return data



if __name__ == "__main__":
    # main()
    # move_by_waypoints() 
    

    gain_configurations = [
        {
            'gain_x': [5, 0, 8.0],
            'gain_y': [5, 0, 8.0],
            'gain_z': [6, 0, 5.0]
        },
        {
            'gain_x': [3, 0, 8.5],
            'gain_y': [3, 0, 8.5],
            'gain_z': [6, 0, 5.0]
        },
        {
            'gain_x': [3, 0, 7.5],
            'gain_y': [3, 0, 7.5],
            'gain_z': [6, 0, 5.0]
        },
        {
            'gain_x': [7, 0, 9],
            'gain_y': [7, 0, 9],
            'gain_z': [3, 0, 5.0]
        },
        {
            'gain_x': [2, 0, 5.5],
            'gain_y': [2, 0, 5.5],
            'gain_z': [6, 0, 5.0]
        },
    # {'gain_x': [3, 0, 6.0], 'gain_y': [3, 0, 6.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [4, 0, 7.5], 'gain_y': [4, 0, 7.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [5, 0, 7.0], 'gain_y': [5, 0, 7.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [6, 0, 7.5], 'gain_y': [6, 0, 7.5], 'gain_z': [6, 0, 5.0]},
    {'gain_x': [7, 0, 8.0], 'gain_y': [7, 0, 8.0], 'gain_z': [6, 0, 5.0]},
    {'gain_x': [3, 0, 9.0], 'gain_y': [3, 0, 9.0], 'gain_z': [6, 0, 5.0]},
    {'gain_x': [4, 0, 8.5], 'gain_y': [4, 0, 8.5], 'gain_z': [6, 0, 5.0]},
    {'gain_x': [5, 0, 8.0], 'gain_y': [5, 0, 8.0], 'gain_z': [6, 0, 5.0]}
    # {'gain_x': [6, 0, 9.5], 'gain_y': [6, 0, 9.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [7, 0, 10.0], 'gain_y': [7, 0, 10.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [3, 0, 5.5], 'gain_y': [3, 0, 5.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [4, 0, 6.0], 'gain_y': [4, 0, 6.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [5, 0, 6.5], 'gain_y': [5, 0, 6.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [6, 0, 7.0], 'gain_y': [6, 0, 7.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [7, 0, 8.0], 'gain_y': [7, 0, 8.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [4, 0, 10.0], 'gain_y': [4, 0, 10.0], 'gain_z': [6, 0, 4.0]},
    # {'gain_x': [3, 0, 5.0], 'gain_y': [3, 0, 5.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [4, 0, 6.0], 'gain_y': [4, 0, 6.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [5, 0, 7.5], 'gain_y': [5, 0, 7.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [6, 0, 8.0], 'gain_y': [6, 0, 8.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [7, 0, 9.0], 'gain_y': [7, 0, 9.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [8, 0, 6.0], 'gain_y': [8, 0, 6.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [6, 0, 6.5], 'gain_y': [6, 0, 6.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [7, 0, 5.5], 'gain_y': [7, 0, 5.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [8, 0, 9.0], 'gain_y': [8, 0, 9.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [4, 0, 6.0], 'gain_y': [4, 0, 6.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [5, 0, 8.5], 'gain_y': [5, 0, 8.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [7, 0, 10.0], 'gain_y': [7, 0, 10.0], 'gain_z': [6, 0, 4.5]},
    # {'gain_x': [6, 0, 9.5], 'gain_y': [6, 0, 9.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [5, 0, 7.0], 'gain_y': [5, 0, 7.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [4, 0, 8.5], 'gain_y': [4, 0, 8.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [3, 0, 5.5], 'gain_y': [3, 0, 5.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [4, 0, 6.5], 'gain_y': [4, 0, 6.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [6, 0, 8.5], 'gain_y': [6, 0, 8.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [5, 0, 6.0], 'gain_y': [5, 0, 6.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [7, 0, 7.0], 'gain_y': [7, 0, 7.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [8, 0, 5.0], 'gain_y': [8, 0, 5.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [5, 0, 6.0], 'gain_y': [5, 0, 6.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [6, 0, 8.0], 'gain_y': [6, 0, 8.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [7, 0, 6.5], 'gain_y': [7, 0, 6.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [8, 0, 7.0], 'gain_y': [8, 0, 7.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [4, 0, 9.5], 'gain_y': [4, 0, 9.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [6, 0, 5.0], 'gain_y': [6, 0, 5.0], 'gain_z': [6, 0, 5.0]}
    ]
    
    all_gate_positions = []
    all_drone_paths = []
    collision_counts = [] 
    runtimes = []  
    # target_x=6.788
    # target_y=81.6774
    # target_z =-43.380
    print("hey there")
    target_x=0
    target_y=0
    target_z =2.0199999809265137
    with open("experiment_results.json", "r") as infile:
        data = json.load(infile)

    # Assign each field to its respective array
    all_gate_positions = data.get("gate_positions", [])
    all_drone_paths = data.get("drone_paths", [])
    collision_counts = data.get("collision_counts", [])
    runtimes = data.get("runtimes", [])
    gain_configurations = data.get("gain_configurations", [])

    # # Run each configuration
    # for gains in gain_configurations:
    #     print(f"\nTesting gains: {gains}")
        
    #     # Reset drone position
    #     client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(target_x, target_y, target_z), 
    #                                         airsim.to_quaternion(0, 0, 0)), True)


    #     time.sleep(3)
    #     client.takeoffAsync().join()
        
    #     # Initialize PID controller with current gains
    #     pidC = PIDController(gain_x=gains['gain_x'], 
    #                       gain_y=gains['gain_y'], 
    #                       gain_z=gains['gain_z'])
        
    #     # Run controller and store results
    #     gate_positions, collision_count, total_runtime  = state_based_pid_control(pidC)

    #     data_to_save = {
    #         "gate_positions": convert_ndarray_to_list(all_gate_positions),
    #         "drone_paths": convert_ndarray_to_list(all_drone_paths),
    #         "collision_counts": collision_counts,
    #         "runtimes": runtimes,
    #         "gain_configurations": gain_configurations  # Save gain configurations as well for easy reference
    #     }

    #     # Write to a JSON file
    #     with open("experiment_results.json", "w") as outfile:
    #         json.dump(data_to_save, outfile, indent=4)
    #     print("Data saved to experiment_results.json")

    #     # if(collision_count != -1):
    #     all_gate_positions.append(gate_positions)
    #     all_drone_paths.append(drone_path.copy())  # Make sure to copy the drone_path
    #     collision_counts.append(collision_count)
    #     runtimes.append(total_runtime)
    #     # Land and reset
    #     client.landAsync().join()
    #     client.armDisarm(True)
    #     client.enableApiControl(True)
        # time.sleep(5)  # Wait between runs
    
    # Plot combined results
    # plot_combined_gate_errors(all_gate_positions, gain_configurations, WAYPOINTS)
    plot_combined_3d_paths(all_drone_paths, all_gate_positions, gain_configurations, WAYPOINTS, WAYPOINTS_ANGLES)

    gain_strings_x = [str(gains['gain_x']) for gains in gain_configurations]
    gain_strings_y = [str(gains['gain_y']) for gains in gain_configurations]
    gain_strings_z = [str(gains['gain_z']) for gains in gain_configurations]

    min_length = min(len(gain_strings_x), len(gain_strings_y), len(gain_strings_z), len(collision_counts), len(runtimes))
 
    # Truncate each array to the minimum length
    gain_strings_x = gain_strings_x[:min_length]
    gain_strings_y = gain_strings_y[:min_length]
    gain_strings_z = gain_strings_z[:min_length]
    collision_counts = collision_counts[:min_length]
    runtimes = runtimes[:min_length]


    # Convert the gain configurations to strings for the x-axis

    
    # Plot Collision Count vs Gain Values (strings)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot for gain_x vs collision count
    axes[0].scatter(gain_strings_x, collision_counts, label="Collision count vs gain_x", marker='o')
    axes[0].set_xlabel('gain_x')
    axes[0].set_ylabel('Collision Count')
    axes[0].set_title('Collision Count vs gain_x')
    axes[0].grid(True)
    axes[0].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

    # Plot for gain_y vs collision count
    axes[1].scatter(gain_strings_y, collision_counts, label="Collision count vs gain_y", marker='o')
    axes[1].set_xlabel('gain_y')
    axes[1].set_ylabel('Collision Count')
    axes[1].set_title('Collision Count vs gain_y')
    axes[1].grid(True)
    axes[1].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

    # Plot for gain_z vs collision count
    # axes[2].plot(gain_strings_z, collision_counts, label="Collision count vs gain_z", marker='o')
    # axes[2].set_xlabel('gain_z')
    # axes[2].set_ylabel('Collision Count')
    # axes[2].set_title('Collision Count vs gain_z')
    # axes[2].grid(True)
    # axes[2].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

    plt.tight_layout()
    plt.show()

    # Plot Runtime vs Gain Values (strings)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot for gain_x vs runtime
    axes[0].scatter(gain_strings_x, runtimes, label="Runtime vs gain_x", marker='o')
    axes[0].set_xlabel('gain_x')
    axes[0].set_ylabel('Total Runtime (seconds)')
    axes[0].set_title('Runtime vs gain_x')
    axes[0].grid(True)
    axes[0].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

    # Plot for gain_y vs runtime
    axes[1].scatter(gain_strings_y, runtimes, label="Runtime vs gain_y", marker='o')
    axes[1].set_xlabel('gain_y')
    axes[1].set_ylabel('Total Runtime (seconds)')
    axes[1].set_title('Runtime vs gain_y')
    axes[1].grid(True)
    axes[1].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

    # Plot for gain_z vs runtime
    # axes[2].plot(gain_strings_z, runtimes, label="Runtime vs gain_z", marker='o')
    # axes[2].set_xlabel('gain_z')
    # axes[2].set_ylabel('Total Runtime (seconds)')
    # axes[2].set_title('Runtime vs gain_z')
    # axes[2].grid(True)
    # axes[2].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

    plt.tight_layout()
    plt.show()

    # Access gain components from dictionaries
    gain_x_P = [config["gain_x"][0] for config in gain_configurations]  # P component of gain_x
    gain_x_D = [config["gain_x"][2] for config in gain_configurations]  # D component of gain_x
    gain_y_P = [config["gain_y"][0] for config in gain_configurations]  # P component of gain_y
    gain_y_D = [config["gain_y"][2] for config in gain_configurations]  # D component of gain_y

    gain_x_P = gain_x_P[:min_length]
    gain_x_D = gain_x_D[:min_length]
    gain_y_P = gain_y_P[:min_length]
    gain_y_D = gain_y_D[:min_length]

    # Create scatter plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Runtime vs PID Gain Components")

    # Plot runtime vs gain_x_P
    axs[0, 0].scatter(gain_x_P, runtimes, color="b")
    axs[0, 0].set_xlabel("gain_x_P")
    axs[0, 0].set_ylabel("Runtime")
    axs[0, 0].set_title("Runtime vs gain_x_P")

    # Plot runtime vs gain_x_D
    axs[0, 1].scatter(gain_x_D, runtimes, color="g")
    axs[0, 1].set_xlabel("gain_x_D")
    axs[0, 1].set_ylabel("Runtime")
    axs[0, 1].set_title("Runtime vs gain_x_D")

    # Plot runtime vs gain_y_P
    axs[1, 0].scatter(gain_y_P, runtimes, color="r")
    axs[1, 0].set_xlabel("gain_y_P")
    axs[1, 0].set_ylabel("Runtime")
    axs[1, 0].set_title("Runtime vs gain_y_P")

    # Plot runtime vs gain_y_D
    axs[1, 1].scatter(gain_y_D, runtimes, color="purple")
    axs[1, 1].set_xlabel("gain_y_D")
    axs[1, 1].set_ylabel("Runtime")
    axs[1, 1].set_title("Runtime vs gain_y_D")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()





    # identified_gates = [
    #     'Gate19', 'Gate18', 'Gate17', 'Gate16', 'Gate15',
    #     'Gate14', 'Gate13', 'Gate12', 'Gate11_23', 'Gate10_21',
    #     'Gate09', 'Gate08', 'Gate07', 'Gate06', 'Gate05',
    #     'Gate04', 'Gate03', 'Gate02', 'Gate01', 'Gate00'
    # ]

#     	"Vehicles": { 
# 		"Drone_L": {
# 			"VehicleType": "SimpleFlight",
# 			"X":10.388,
# 			"Y": 80.774,
# 			"Z": -43.580,
# 			"Yaw": 0
# 		}
# 	}
# }


# Traceback (most recent call last):
#   File "C:\Users\yello\Desktop\499\ADRL\gates.py", line 890, in <module>
#     plot_combined_3d_paths(all_drone_paths, all_gate_positions, gain_configurations, WAYPOINTS, WAYPOINTS)
#   File "C:\Users\yello\Desktop\499\ADRL\gates.py", line 694, in plot_combined_3d_paths
#     rotation_matrix = np.array([
#                       ^^^^^^^^^^
# ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (3, 3) + inhomogeneous part.