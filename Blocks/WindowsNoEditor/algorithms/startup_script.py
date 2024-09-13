import airsim
import os
import time
import numpy as np
import cv2
import math

from algorithms.gen_traj import Generate
from algorithms.perception import Perception
lead = "Drone_L"
chase = "Drone_C"

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
# client.reset()

curr_state = client.simGetVehiclePose(lead)
print("lead state", curr_state)

client.enableApiControl(True,lead)
client.armDisarm(True, lead)
client.takeoffAsync(10, lead)

client.enableApiControl(True,chase)
client.armDisarm(True, chase)
client.takeoffAsync(10, chase)