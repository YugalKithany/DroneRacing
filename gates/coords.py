import airsim
import pkg_resources

def print_gate_info():
    # Create a VehicleClient instance
    client = airsim.VehicleClient()
    
    # Connect to the AirSim server
    try:
        client.confirmConnection()
        print("Successfully connected to AirSim server.")
    except Exception as e:
        print("Failed to connect to AirSim server.")
        print(f"Error: {e}")
        return
    
    # Get the list of all scene objects that start with "Gate"
    object_names = client.simListSceneObjects(name_regex='Gate.*')
    
    print(f"Found {len(object_names)} Gates:")
    
    for object_name in object_names:
        try:
            # Get the pose of each gate object
            pose = client.simGetObjectPose("Gate")
            
            # Print the object name and its coordinates
            print(f"Object: {object_name}")
            print(f"  Position: (x: {pose.position.x_val}, y: {pose.position.y_val}, z: {pose.position.z_val})")
            print(f"  Orientation: (roll: {pose.orientation.roll}, pitch: {pose.orientation.pitch}, yaw: {pose.orientation.yaw})")
        
        except Exception as e:
            print(f"Error retrieving pose for {object_name}: {e}")

if __name__ == "__main__":
    print_gate_info()
