import airsim
import time
import math
import matplotlib.pyplot as plt
import numpy as np

lead = "Drone_L"
chase = "Drone_C"

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
curr_state = client.simGetVehiclePose(lead) # print("lead state", curr_state)

# Initalize Lead and Chaser Drones
client.enableApiControl(True,lead)
client.armDisarm(True, lead)
client.takeoffAsync(3, lead).join()

client.enableApiControl(True,chase)
client.armDisarm(True, chase)
client.takeoffAsync(3, chase).join()



if __name__ == "__main__":
  clientL = airsim.MultirotorClient() 
  clientL.enableApiControl(True,chase)

  y=10
  z=7.27  

  # y_coords = []
  # z_coords = []
  # y_coords.append(client.simGetVehiclePose(lead).position.y_val-y)
  # z_coords.append(0-(client.simGetVehiclePose(lead).position.z_val-z))
  center = airsim.Vector3r(0, y, z) 
  def smooth_blacbox(client, waypoints, target_yaw, velocities):
    if len(waypoints) != len(velocities):
      raise ValueError("Number of waypoints and velocities must be equal")

    smoothed_waypoints = []
    for i in range(len(waypoints) - 1):
      start_waypoint = waypoints[i]
      end_waypoint = waypoints[i + 1]
      velocity = velocities[i]
      intermediate_points = 3  # Adjust for smoothness

      curr_pose_chase = airsim.Vector3r(client.simGetVehiclePose(chase).position.x_val,
                        client.simGetVehiclePose(chase).position.y_val,
                        client.simGetVehiclePose(chase).position.z_val)
    
      for j in range(1, intermediate_points + 1):
        weight = j / (intermediate_points + 1)
        intermediate_waypoint = start_waypoint + (end_waypoint - start_waypoint) * weight
        smoothed_waypoints.append(airsim.Vector3r(intermediate_waypoint.x_val, intermediate_waypoint.y_val, intermediate_waypoint.z_val))
        # print(intermediate_waypoint.x_val, " : ", intermediate_waypoint.y_val, " :", intermediate_waypoint.z_val)
        # new = []
        # new.append(airsim.Vector3r(curr_pose_chase))
        # new.append(smoothed_waypoints[-1])
        # print("ARRRIVED")
        print(smoothed_waypoints[-1])
        temp = [curr_pose_chase, smoothed_waypoints[-1] ]
        # clientL.moveOnPathAsync(smoothed_waypoints[-1], 5, 3, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1, 1, vehicle_name=lead)
        clientL.moveOnPathAsync([smoothed_waypoints[-1]], 5, 3, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,90), 1, 1, vehicle_name=lead)

        # print("HMMMM")
        time.sleep(3)

    # Move to the last waypoint with the target yaw
    clientL.moveOnPathAsync([waypoints[-1]], velocities[-1], 5, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1, 1, vehicle_name=lead)
    print(waypoints[-1])


  # Example usage
  # waypoints = [center, airsim.Vector3r(10, y, z), airsim.Vector3r(10, 20, z)]
  # baseline
  # clientL.moveOnPathAsync(waypoints, 5, 50 ,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1, 1, vehicle_name=lead)
  target_yaw = 90  # Yaw in degrees
  velocities = [5, 5, 3]  # Velocities for each waypoint (m/s)
  # smooh
  # smooth_blacbox(clientL, waypoints, target_yaw, velocities)

  # y=0
  # z=34.27  
  # center = airsim.Vector3r(-5, y, z) 
  # radius = 5 
  # num_waypoints = 30
  # waypoints = []
  # for i in range(num_waypoints):
  #     angle = 2 * np.pi * (i / (num_waypoints - 1))
  #     x = center.x_val + radius * np.cos(angle)
  #     y = center.y_val + radius * np.sin(angle)
  #     z = center.z_val  # Maintain same altitude
  #     waypoint = airsim.Vector3r(x, y, z)
  #     waypoints.append(waypoint)
  # client.moveOnPathAsync(waypoints, 5, 20 ,airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False,0), 1, 1, vehicle_name = lead)


  center = airsim.Vector3r(0, 0, 34.27 ) 
  waypoints = []
  waypoints.append(center)
  for cnt in range(300):
      period=100
      sizex=2
      sizey=5
      t = cnt / period * 2 * np.pi
      x = np.sqrt(2) * np.cos(t) / (1 + np.sin(t) ** 2)
      y = x * np.sin(t)
      x = sizex * x
      y = sizey * y
      # z = np.ones_like(x) * 1.5
      waypoint = airsim.Vector3r(x, y, 8.27) 
      waypoints.append(waypoint) 
  client.moveOnPathAsync(waypoints, 3, 60 ,airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False,0), 1, 1, vehicle_name = lead)




  # clientL.moveOnPathAsync(waypoints, 5, 50 ,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1, 1, vehicle_name=lead)









  for i in range (300):
    time.sleep(.1)  # Adjust sleep time as needed

  client.armDisarm(False)
  client.enableApiControl(False)

























  # waypoints = [center, airsim.Vector3r(0,y+5,z-5), airsim.Vector3r(0,y+5,z-15), airsim.Vector3r(0,y+35,z-10), 
  #              airsim.Vector3r(0,y+55,z-15), airsim.Vector3r(0,y+75,z-20), airsim.Vector3r(0,y+95,z-5)]
  


  # traj_tracker_gains = airsim.TrajectoryTrackerGains(kp_cross_track = 5.0, kd_cross_track = 0.0, 
  #                                                   kp_vel_cross_track = 3.0, kd_vel_cross_track = 0.0, 
  #                                                   kp_along_track = 0.4, kd_along_track = 0.0, 
  #                                                   kp_vel_along_track = 0.04, kd_vel_along_track = 0.0, 
  #                                                   kp_z_track = 2.0, kd_z_track = 0.0, 
  #                                                   kp_vel_z = 0.4, kd_vel_z = 0.0, 
  #                                                   kp_yaw = 3.0, kd_yaw = 0.1)

  # airsim.setTrajectoryTrackerGains(..., vehicle_name=lead)
  # time.sleep(0.2)

  # clientL.moveOnSplineAsync(center, vel_max=15.0, acc_max=5.0, add_position_constraint=True, add_velocity_constraint=False, 
  # add_acceleration_constraint=False, viz_traj=clientL.viz_traj, viz_traj_color_rgba=clientL.viz_traj_color_rgba, vehicle_name=lead)


  # center = airsim.Vector3r(0, y,z) 
  # radius = 10  # Adjust this value for the desired circle size (meters)
  # velocity = 5  # Adjust for desired speed
  # initial_circumference = 2 * math.pi * radius

  # num_loops = 45  # Adjust for desired number of loops (essentially time)
  # duration_per_step = initial_circumference / (velocity * num_loops)

  # for i in range(num_loops):
  #   angle = 2 * math.pi * (i / (num_loops - 1))
  #   vx = velocity * math.cos(angle)
  #   vy = velocity * math.sin(angle)
  #   client.moveByVelocityAsync( vx=vx, vy=vy, vz=0, duration=duration_per_step, vehicle_name=lead    )
  #   time.sleep(duration_per_step)

  # clientL.moveOnPathAsync(waypoints, 5, 50 ,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1, 1, vehicle_name=lead)
  # y_coords.append(client.simGetVehiclePose(lead).position.y_val-y)
  # z_coords.append(0-(client.simGetVehiclePose(lead).position.z_val-z))


  # for i in range (300):
    # Save Y and Z Coords
    # state = clientL.getMultirotorState()
    # current_position = state.kinematics_estimated.position

    # y_coords.append(client.simGetVehiclePose(lead).position.y_val-y)
    # z_coords.append(0-(client.simGetVehiclePose(lead).position.z_val-z))

    # time.sleep(.1)  # Adjust sleep time as needed

  # use matplot lib to save plot cords 
  # print("Finished")

  # plt.figure(figsize=(8, 6))  # Adjust figure size as desired
  # plt.plot(y_coords, z_coords, label="Drone Trajectory")
  # plt.xlabel("Y-Coordinate")
  # plt.ylabel("Z-Coordinate")
  # plt.title("Drone Flight Path")
  # plt.legend()
  # plt.grid(True)

  # # Optionally save the plot (replace 'plot.png' with desired filename)
  # plt.savefig('plot.png')

  # plt.show()



  # time.sleep(60)
  # client.armDisarm(False)
  # client.enableApiControl(False)




'''
  # Circle in XY Plane
  center = airsim.Vector3r(0, 10, 30) 
  radius = 10  
  num_waypoints = 30

  # waypoints = []
  # for i in range(num_waypoints):
  #     angle = 2 * math.pi * (i / (num_waypoints - 1))
  #     x = center.x_val + radius * math.cos(angle)
  #     y = center.y_val + radius * math.sin(angle)
  #     z = center.z_val  # Maintain same altitude
  #     waypoint = airsim.Vector3r(x, y, z)
  #     waypoints.append(waypoint)
  # clientL.moveOnPathAsync(waypoints, 5, 20 ,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1, 1, vehicle_name=lead)
  # time.sleep(1)

  
  # Triangle in XZ Plane
  base_length = 20
  height = 10
  center = airsim.Vector3r(0, 0, 34.27)
  waypoints = [center, 
                airsim.Vector3r(0, base_length, center.z_val), 
                airsim.Vector3r(0, 0, center.z_val-height), 
                airsim.Vector3r(0, -base_length, center.z_val),
                airsim.Vector3r(0,  base_length, center.z_val)]

  clientL.moveOnPathAsync(waypoints, 5, 30 ,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1, 1, vehicle_name=lead)
  time.sleep(30)
'''
  
  # for i in range(1):
  #     z=33
  #     clientL.moveOnPathAsync([ airsim.Vector3r(10,0,z), airsim.Vector3r(-10,0,z), airsim.Vector3r(0,10,z),
  #                                       airsim.Vector3r(0,-10,z)], 5, 20 ,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1, 1, vehicle_name=lead)
  #     time.sleep(1)
   
  # print("Finished")
  # time.sleep(3)
  # client.armDisarm(False)
  # client.enableApiControl(False)
