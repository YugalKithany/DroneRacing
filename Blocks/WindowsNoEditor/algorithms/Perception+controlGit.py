import airsim
import cv2
import time
import math
import sys
import numpy as np

# press "1" in the AirSim view to turn on the depth capture

client = airsim.MultirotorClient()
client.confirmConnection()

lead = "Drone_L"
chase = "Drone_C"
# client.enableApiControl(True)
# client.armDisarm(True)
# client.takeoffAsync().join()

client.enableApiControl(True,lead)
client.armDisarm(True, lead)
client.takeoffAsync(10.0, lead).join()

client.enableApiControl(True,chase)
client.armDisarm(True, chase)
client.takeoffAsync(10.0, chase).join()

# get depth image
yaw = 0
pi = math.pi
vx = 0
vy = 0
driving = 0

distance_data = []  # Create an empty array to store distance data
previous_distance = 255  # Initialize with an arbitrary value

# This perception Module is from https://github.com/microsoft/AirSim/blob/main/PythonClient/multirotor/navigate.py
while True:
    # this will return png width= 256, height= 144
    depthImage = client.simGetImage("0", airsim.ImageType.DepthVis) 

    rawImage = np.frombuffer(depthImage, np.int8)
    png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)

    top = np.vsplit(gray, 2)[0]

    bands = np.hsplit(top, [50,100,150,200])
    maxes = [np.max(x) for x in bands]
    min = np.argmin(maxes)    
    distance = 255 - maxes[min]


    # Calculate the difference between the current and previous distances
    distance_diff = abs(distance - previous_distance)

    # Store the current distance and difference in the array
    distance_data.append([distance, distance_diff])

    # Store the current distance for the next iteration
    previous_distance = distance





    # sanity check on what is directly in front of us (slot 2 in our hsplit)
    current = 255 - maxes[2]

    if (current < 20):
        client.hoverAsync().join()
        airsim.wait_key("whoops - we are about to crash, so stopping!")

    pitch, roll, yaw  = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)

    if (distance > current + 30):
    
        # we have a 90 degree field of view (pi/2), we've sliced that into 5 chunks, each chunk then represents
        # an angular delta of the following pi/10.
        change = 0
        driving = min
        if (min == 0):
            change = -2 * pi / 10
        elif (min == 1):
            change = -pi / 10
        elif (min == 2):
            change = 0 # center strip, go straight
        elif (min == 3):
            change = pi / 10
        else:
            change = 2*pi/10

        yaw = (yaw + change)
        vx = math.cos(yaw)
        vy = math.sin(yaw)
        print ("switching angle", math.degrees(yaw), vx, vy, min, distance, current)

    if (vx == 0 and vy == 0):
        vx = math.cos(yaw)
        vy = math.sin(yaw)

    print ("distance=", current)
    print("distance_diff=", distance_diff)  # Print the calculated difference


    client.moveByVelocityZAsync(vx, vy,-6, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()

    x = int(driving * 50)
    cv2.rectangle(png, (x,0), (x+50,50), (0,255,0), 2)
    cv2.imshow("Top", png)

    key = cv2.waitKey(1) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x')):
        break