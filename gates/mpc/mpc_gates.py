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
from controller_mpc import MPCController

# Waypoints for drone navigation waypoints_qualifier1.yaml from rovery script TODO-Remake this, ensure it is center of gate
WAYPOINTS = [
    [10.388, 80.774, -43.580], [18.110, 76.260, -43.580], [25.434, 66.287, -43.580],
    [30.066, 56.550, -43.580], [32.801, 45.931, -43.580], [30.503, 38.200, -43.580],
    [3.264, 37.569, -43.580], [-17.863, 45.418, -46.580], [-15.494, 63.187, -52.080],
    [-6.321, 78.212, -55.780], [ 5.144, 82.385, -55.780], [14.559, 84.432, -55.180],
    [22.859, 82.832, -32.080], [38.259, 78.132, -31.380], [51.059, 52.132, -25.880],
    [44.959, 38.932, -25.880], [25.959, 26.332, -17.880], [11.659, 26.332, -13.780],
    [-10.141, 22.632, -6.380], [-23.641, 10.132, 2.120]
]

WAYPOINTS_ANGLES = [ -15, -45, -60, -90 ,-115, 0, 0,  -45,  -90 , -135, -180, -180, -30, -45, -60, -135, -180, -180, -195, -25 ]


# WAYPOINTS = [[0.0, 2.0, 2.0199999809265137], [1.5999999046325684, 10.800000190734863, 2.0199999809265137], 
#              [8.887084007263184, 18.478761672973633, 2.0199999809265137], [18.74375343322754, 22.20650863647461, 2.0199999809265137],
#              [30.04375457763672, 22.20648956298828, 2.0199999809265137], [39.04375457763672, 19.206478118896484, 2.0199999809265137],
#             [45.74375534057617, 11.706478118896484, 2.0199999809265137], [45.74375534057617, 2.2064781188964844, 2.0199999809265137], 
#             [40.343753814697266, -4.793521404266357, 2.0199999809265137], [30.74375343322754, -7.893521785736084,2.0199999809265137],
#             [18.54375457763672, -7.893521785736084, 2.0199999809265137], [9.543754577636719, -5.093521595001221, 2.0199999809265137]]
# WAYPOINTS_ANGLES = [
#     15, 30, 45,60,75,90,120,135,150,165,180,195]
# WAYPOINTS = [
#  [0.000, 0.000, 2.020 ], [0.251, 2.610, 2.020  ], [0.643, 5.583, 2.022  ], [1.086, 8.550, 2.023  ], [1.966, 11.395, 2.046 ], 
#  [3.191, 14.098, 2.160 ],  [5.537, 15.950, 2.106 ],  [7.986, 17.676, 2.033 ],  [9.934, 19.922, 2.176 ],  [12.016, 21.961, 2.381],
#  [14.991, 22.020, 2.241], [17.972, 22.051, 2.052], [20.895, 21.761, 2.019], [23.779, 21.025, 2.022], [26.747, 21.400, 2.021],
#  [29.657, 22.131, 2.020], [32.548, 22.169, 1.920], [35.457, 21.609, 1.901], [37.847, 19.808, 1.989], [37.465, 20.118, 2.011], [35.790, 17.531, 2.020],
#  [38.131, 15.716, 2.020], [40.683, 14.145, 2.020], [43.295, 12.679, 2.020], [44.962, 11.294, 2.020], [42.623, 9.430, 2.023 ], 
#  [42.555, 6.647, 2.005 ],  [44.334, 4.238, 2.002 ],  [45.085, 2.097, 1.954 ],  [42.367, 1.166, 1.634 ],  [41.107, -1.493, 1.438], [40.606, -4.367, 1.912],
#  [37.860, -5.364, 2.004], [35.025, -6.340, 2.018], [32.244, -7.464, 2.024], [29.497, -7.050, 2.346], [27.124, -5.416, 3.048], [24.147, -5.313, 3.103],
#  [21.259, -6.023, 2.809], [21.208, -7.796, 2.056], [18.500, -7.769, 2.028], [15.614, -6.966, 2.009], [12.783, -5.974, 2.015], [9.544, -5.094, 2.020 ]]


# WAYPOINTS_ANGLES = [15, 30, 45,60,75,90,120,135,150,165,180,195,15, 30, 45,60,75,90,120,135,150,165,180,195,15, 30, 45,60,75,90,120,135,150,165,180,195,15, 30, 45,60,75,90,120,135,150,165,180,195]
# WAYPOINTS_ANGLES = [15, 15, 15, 30, 30, 30, 45, 45, 45, 60, 60, 60, 75, 75, 75, 90, 90, 90, 90, 90, 90, 120, 120, 120, 135, 135, 135, 135, 150, 150,150,150, 150, 165, 165, 165, 180,180,180, 180, 180, 195, 195, 195]
# WAYPOINTS_ANGLES = [
#     15, 15, 15, 15, 30, 30, 30, 30, 45, 45, 45, 45,
#     60, 60, 60, 60, 75, 75, 75, 75, 90, 90, 90, 90,
#     120, 120, 120, 120, 135, 135, 135, 135, 150, 150, 150, 150,
#     165, 165, 165, 165, 180, 180, 180, 180, 195, 195, 195, 195
# ]


# 			"X":10.388,
# 			"Y": 80.774,
# 			"Z": -43.580,
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
# gain_x = [3, 0, 8.0]  # Reduced from [5, 0, 10.0]
# gain_y = [3, 0, 8.0]  # Reduced from [5, 0, 10.0]
# gain_z = [1, 0, 5.0]  # Keep the same
# pid = PIDController(gain_x=gain_x, gain_y=gain_y, gain_z=gain_z)
mpc = MPCController(
    prediction_horizon=10,
    control_horizon=5,
    dt=0.1,
    output_limit=10
)


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





def state_based_mpc_control(mpc_controller):
    """
    Modified version of state_based_pid_control to use MPC controller
    Args:
        mpc_controller: Instance of MPCController class
    Returns:
        tuple: (gate_clearance_positions, collision_count, total_runtime)
    """
    global drone_path
    drone_path = []
    gate_clearance_positions = []
    collision_count = 0
    run_start_time = time.time()

    for i, wp in enumerate(WAYPOINTS):
        print(f"Target waypoint: {wp}")
        current_pos = client.getMultirotorState().kinematics_estimated.position
        mpc_controller.update_setpoint(wp)
        
        final_approach_velocity = [0, 0, 0]
        stuck_timestamp = time.time()
        
        while not np.allclose([current_pos.x_val, current_pos.y_val, current_pos.z_val], wp, atol=1.5):
            current_coords = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])
            drone_path.append(list(current_coords))
            
            # Get control signal from MPC controller
            # Note: MPC already handles time step internally, but we pass it for state estimation
            dt = 0.5  # Matches the moveByVelocityAsync duration
            control_signal = mpc_controller.update(current_coords, dt)
            
            # Calculate yaw angle (same as before)
            yaw_angle = calculate_yaw_angle(current_coords, wp)
            
            # Apply control signal
            # Note: MPC already handles velocity scaling internally
            client.moveByVelocityAsync(
                control_signal[0], 
                control_signal[1], 
                control_signal[2], 
                dt,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, yaw_angle)
            ).join()
            
            # Check for collisions
            collision_info = client.simGetCollisionInfo()
            if collision_info.has_collided:
                collision_count += 1 

            # Store final approach velocity for gate clearance
            final_approach_velocity = control_signal * 0.2  # Reduce velocity for careful gate approach
            
            # Update current position
            current_pos = client.getMultirotorState().kinematics_estimated.position
            
            # Handle stuck condition
            current_time = time.time()
            if(current_time - stuck_timestamp > 20):
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(wp[0], wp[1], wp[2]), 
                                                    airsim.to_quaternion(0, 0, 0)), True)
                print("fixing stuck")
                stuck_timestamp = current_time

            # Check timeout condition
            if(current_time - run_start_time > 320): 
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(6.788, 81.6774, -43.380), 
                                    airsim.to_quaternion(0, 0, 0)), True)
                return np.array(gate_clearance_positions), collision_count, current_time - run_start_time

        # Handle gate clearance (same as before)
        if i < len(WAYPOINTS) - 1:
            print("Clearing gate...")
            current_pos = client.getMultirotorState().kinematics_estimated.position
            gate_clearance_positions.append([current_pos.x_val, current_pos.y_val, current_pos.z_val])
            
            clearance_time = 00
            start_time = time.time()
            
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



def main_mpc():

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    target_x=6.788
    target_y=81.6774
    target_z =-43.380
    # target_x=0
    # target_y=0    
    # target_z =2.0199999809265137
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(target_x, target_y, target_z), 
                                            airsim.to_quaternion(0, 0, 0)), True)

    # Take off
    client.takeoffAsync().join()
    
    # Create MPC controller with tuned parameters
    mpc = MPCController(
        prediction_horizon=10,  # Look ahead 10 steps
        control_horizon=5,      # Optimize 5 control inputs
        dt=0.1,                # 100ms time step for internal model
        output_limit=5         # Match original PID limits
    )
    
    try:
        # Run the control loop
        gate_positions, collisions, runtime = state_based_mpc_control(mpc)
        
        
        data_to_save = {
            "gate_positions": convert_ndarray_to_list(gate_positions),
            "drone_paths": convert_ndarray_to_list(drone_path),
            "collision_counts": collisions,
            "runtimes": runtime
            # "gain_configurations": gain_configurations  # Save gain configurations as well for easy reference
        }

        # Write to a JSON file
        with open("mpc_raw_qual1.json", "w") as outfile:
            json.dump(data_to_save, outfile, indent=4)
        print("Data saved to experiment_results.json")


        # Print results
        print(f"Flight completed:")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Collisions: {collisions}")
        print(f"Gates cleared: {len(gate_positions)}")
        


    except Exception as e:
        print(f"Flight failed with error: {str(e)}")
    
    finally:
        # Land and cleanup
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)



if __name__ == "__main__":
    # main_mpc()

    gain_configurations = [
    {'gain_x': [5, 0, 8.0], 'gain_y': [5, 0, 8.0], 'gain_z': [6, 0, 5.0]},
    {'gain_x': [3, 0, 8.5], 'gain_y': [3, 0, 8.5], 'gain_z': [6, 0, 5.0]}
    # {'gain_x': [3, 0, 7.5], 'gain_y': [3, 0, 7.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [7, 0, 9.0], 'gain_y': [7, 0, 9.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [2, 0, 5.5], 'gain_y': [2, 0, 5.5], 'gain_z': [6, 0, 5.0]}
    # {'gain_x': [3, 0, 6.0], 'gain_y': [3, 0, 6.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [4, 0, 7.5], 'gain_y': [4, 0, 7.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [5, 0, 7.0], 'gain_y': [5, 0, 7.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [6, 0, 7.5], 'gain_y': [6, 0, 7.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [7, 0, 8.0], 'gain_y': [7, 0, 8.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [3, 0, 9.0], 'gain_y': [3, 0, 9.0], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [4, 0, 8.5], 'gain_y': [4, 0, 8.5], 'gain_z': [6, 0, 5.0]},
    # {'gain_x': [5, 0, 8.0], 'gain_y': [5, 0, 8.0], 'gain_z': [6, 0, 5.0]}
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
    print("Starting Main Loop")
    # target_x=0
    # target_y=0    
    # target_z =2.0199999809265137
    target_x=6.788
    target_y=81.6774
    target_z =-43.380
    # client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(target_x, target_y, target_z), 
    #                                         airsim.to_quaternion(0, 0, 0)), True)
    with open("pid_raw_qual1.json", "r") as infile:
        data = json.load(infile)

    # Assign each field to its respective array
    # all_gate_positions = data.get("gate_positions", [])
    all_gate_positions = WAYPOINTS
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


    #     time.sleep(1)
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
    #     time.sleep(3)  # Wait between runs
    
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


    # Plot Collision Count vs Gain Values (strings) second 1 is # plots
    fig, axes = plt.subplots(1, 1, figsize=(18, 6))
    
    # Plot for gain_x vs collision count
    axes.scatter(gain_strings_x, collision_counts, label="Collision count vs gain", marker='o')
    axes.set_xlabel('gain')
    axes.set_ylabel('Collision Count')
    axes.set_title('Collision Count vs gain')
    axes.grid(True)
    axes.tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

    plt.tight_layout()
    plt.show()

    # Plot Runtime vs Gain Values (strings)
    fig, axes = plt.subplots(1, 1, figsize=(18, 6))
    
    # Plot for gain_x vs runtime
    axes.scatter(gain_strings_x, runtimes, label="Runtime vs gain", marker='o')
    axes.set_xlabel('gain')
    axes.set_ylabel('Total Runtime (seconds)')
    axes.set_title('Runtime vs gain')
    axes.grid(True)
    axes.tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

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
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle("Runtime vs PID Gain Components")

    # Plot runtime vs gain_x_P
    axs[0].scatter(gain_x_P, runtimes, color="b")
    axs[0].set_xlabel("gain_P")
    axs[0].set_ylabel("Runtime")
    axs[0].set_title("Runtime vs gain_P")

    # Plot runtime vs gain_x_D
    axs[1].scatter(gain_x_D, runtimes, color="g")
    axs[1].set_xlabel("gain_D")
    axs[1].set_ylabel("Runtime")
    axs[1].set_title("Runtime vs gain_D")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

