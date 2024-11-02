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
# Waypoints for drone navigation waypoints_qualifier1.yaml from rovery script TODO-Remake this, ensure it is center of gate
WAYPOINTS = [
    [10.388, 80.774, -43.580], [18.110, 76.260, -43.580], [25.434, 66.287, -43.580],
    [30.066, 56.550, -43.580], [32.301, 45.931, -43.880], [26.503, 38.200, -43.380],
    [3.264, 37.569, -43.580], [-17.863, 45.418, -46.580], [-15.494, 63.187, -52.080],
    [-6.321, 78.212, -55.780], [5.144, 82.385, -55.780]
]

# WAYPOINTS = [
#     [10.388, 80.774, -43.580], [18.110, 76.260, -43.580], [25.434, 66.287, -43.580],
#     [30.066, 56.550, -43.580], [32.301, 45.931, -43.880], [26.503, 38.200, -43.380],
#     [3.264, 37.569, -43.580], [-17.863, 45.418, -46.580], [-15.494, 63.187, -52.080],
#     [-6.321, 78.212, -55.780], [5.144, 82.385, -55.780], [14.559, 84.432, -55.180],
#     [22.859, 82.832, -32.080], [38.259, 78.132, -31.380], [51.059, 52.132, -25.880],
#     [44.959, 38.932, -25.880], [25.959, 26.332, -19.880], [11.659, 26.332, -12.780],
#     [-10.141, 22.632, -6.380], [-24.641, 9.132, 2.120]
# ]


# WAYPOINTS = [
#     [12.559, 82.432, -55.180],
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




def state_based_pid_control():
    for i, wp in enumerate(WAYPOINTS):
        print(f"Target waypoint: {wp}")
        current_pos = client.getMultirotorState().kinematics_estimated.position
        pid.update_setpoint(wp)
        
        # Store the final approach velocity
        final_approach_velocity = [0, 0, 0]
        
        # First phase: Approach the gate
        while not np.allclose([current_pos.x_val, current_pos.y_val, current_pos.z_val], wp, atol=1.5):
            current_coords = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])
            control_signal = pid.update(current_coords, dt=1)
            control_signal = np.clip(control_signal, -5, 5)
            
            # Calculate yaw but maintain it while approaching gate
            yaw_angle = calculate_yaw_angle(current_coords, wp)
            client.moveByVelocityAsync(
                control_signal[0]/2, 
                control_signal[1]/2, 
                control_signal[2]/5, 
                0.5,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, yaw_angle)
            ).join()
            
            # Store the current velocity for gate clearance
            final_approach_velocity = [control_signal[0]/5, control_signal[1]/5, control_signal[2]/5]
            current_pos = client.getMultirotorState().kinematics_estimated.position

        # Second phase: Clear the gate using the final approach velocity
        if i < len(WAYPOINTS) - 1:  # Don't do this for the last waypoint
            print("Clearing gate...")
            
            # Continue with the same velocity for about 1 second (adjust as needed)
            clearance_time = 1.0  # seconds
            start_time = time.time()
            
            while time.time() - start_time < clearance_time:
                client.moveByVelocityAsync(
                    final_approach_velocity[0],
                    final_approach_velocity[1],
                    final_approach_velocity[2],
                    0.1,  # Short duration for smooth movement
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(False, yaw_angle)
                ).join()
                
    print("Completed all waypoints")


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

def plot_3d_path(drone_path, waypoints):
    # Convert paths and waypoints to numpy arrays
    drone_path = np.array(drone_path)
    waypoints = np.array(waypoints)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=drone_path[:, 0],
        y=drone_path[:, 1],
        z=drone_path[:, 2],
        mode='lines',
        name='Drone Path',
        line=dict(color='blue', width=5)
    ))
    fig.add_trace(go.Scatter3d(
        x=waypoints[:, 0],
        y=waypoints[:, 1],
        z=waypoints[:, 2],
        mode='markers',
        name='Waypoints',
        marker=dict(color='red', size=5)
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="3D Drone Path and Waypoints"
    )
    fig.write_html("3d_drone_path.html")
    fig.show()


def main():
    target_x=6.788
    target_y=81.6774
    target_z =-43.380
    time.sleep(3)
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(target_x, target_y, target_z), airsim.to_quaternion(0, 0, 0)), True)
    print("Hey")
    client.takeoffAsync(5).join()
    
    # Baseline: Move by waypoints
    # print("Baseline Waypoint Navigation")
    # move_by_waypoints()
    # plot_3d_path(drone_path, WAYPOINTS)
    
    # # State-based PID control
# 			"X":10.388,
# 			"Y": 80.774,
# 			"Z": -43.580,
    print("INIT WAYPOINT")
    # initial_waypoint = [6.788,81.6774,-45.980]
    # client.moveToPositionAsync(initial_waypoint[0], initial_waypoint[1], initial_waypoint[2], 7).join()

    # print("START SLEEP")
    # time.sleep(10)
    # print("END SLEEP")
    # client.landAsync().join()
    # print("START SLEEP2")
    # time.sleep(5)
    # print("END SLEEP2")

    # client.takeoffAsync(-5).join()

    # client.landAsync().join()
    # time.sleep(5)
    # client.takeoffAsync().join()
    print("State-Based PID Control")
    state_based_pid_control()
    plot_3d_path(drone_path, WAYPOINTS)
    
    # # Vision-based approach
    # print("Vision-Based Navigation")
    # vision_based_navigation()
    
    # Land after finishing
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

if __name__ == "__main__":
    main()




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