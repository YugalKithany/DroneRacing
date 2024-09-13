# ready to run example: PythonClient/multirotor/hello_drone.py
# note: async methods take a long time to execute, add join() to wait for it finish 
# NED coordinates: +x:forward +y:right +z:down

import airsim
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from particle_main import RunParticle
import traceback
import random
from controller_m.gen_traj import Generate
# from perception.perception import Perception # YK
from simple_excitation import excitation 
import threading
from pyvista_visualiser import Perception_simulation
from controller_pid import PIDController
import ctypes

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# client.enableApiControl(True,client)
# client.armDisarm(True, client)
client.takeoffAsync(30.0, client).join()




# Waypoints to visit
waypoints = [
    airsim.Vector3r(10, 10, -5),
    airsim.Vector3r(20, 10, -5),
    airsim.Vector3r(20, 20, -5),
    airsim.Vector3r(10, 20, -5),
    airsim.Vector3r(10, 10, -5),
    airsim.Vector3r(0, 0, -5),
    airsim.Vector3r(10, 0, -5),
    airsim.Vector3r(10, 10, -5),
    airsim.Vector3r(0, 10, -5),
    airsim.Vector3r(0, 0, -5)
]

# Take off
client.takeoff()

# Move to waypoints
for waypoint in waypoints:
    client.moveToPosition(waypoint.x, waypoint.y, waypoint.z, velocity=1)  # Adjust velocity as needed

# Land
client.land()