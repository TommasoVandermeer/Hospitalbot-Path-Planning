"""
ROS 2 node to spawn a mobile robot with 180 degree LIDAR inside a hospital.

Author:
  - Tommaso Van Der Meer
  - tommaso.vandermeer@student.unisi.it
"""
import os # Operating system library
import sys # Python runtime environment library
import rclpy # ROS Client Library for Python

# Package Management Library
from ament_index_python.packages import get_package_share_directory 

# Gazebo's service to spawn a robot
from gazebo_msgs.srv import SpawnEntity

import math

def main():

    """ Main for spawning a robot node """
    # Get input arguments from user
    argv = sys.argv[1:]
    
    # Start node
    rclpy.init()
        
    # Create the node
    node = rclpy.create_node("entity_spawner")
    node.get_logger().info("HERE WE GOOO")

    # Show progress in the terminal window
    node.get_logger().info(
        'Creating Service client to connect to `/spawn_entity`')
    client = node.create_client(SpawnEntity, "/spawn_entity")

    # Get the spawn_entity service
    node.get_logger().info("Connecting to `/spawn_entity` service...")
    if not client.service_is_ready():
        client.wait_for_service()
        node.get_logger().info("...connected!")

    ## SPAWN ROBOT
    # Get path to the robot
    sdf_file_path = os.path.join(
        get_package_share_directory("hospital_robot_spawner"), "models",
        "pioneer3at", "model.sdf")

    # Show file path
    print(f"robot_sdf={sdf_file_path}")
    
    # Set data for request
    request = SpawnEntity.Request()
    request.name = argv[0]
    request.xml = open(sdf_file_path, 'r').read()
    request.robot_namespace = argv[1]
    request.initial_pose.position.x = float(argv[2])
    request.initial_pose.position.y = float(argv[3])
    request.initial_pose.position.z = float(argv[4])

    desired_angle = float(math.radians(-90))
    request.initial_pose.orientation.z = float(math.sin(desired_angle/2))
    request.initial_pose.orientation.w = float(math.cos(desired_angle/2))

    node.get_logger().info("Sending service request to `/spawn_entity`")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        print('response: %r' % future.result())
    else:
        raise RuntimeError(
            'exception while calling service: %r' % future.exception())

    ## SPAWN TARGET
    # Get path to the target
    target_sdf_file_path = os.path.join(
        get_package_share_directory("hospital_robot_spawner"), "models",
        "Target", "model.sdf")

    request = SpawnEntity.Request()
    request.name = "Target"
    request.xml = open(target_sdf_file_path, 'r').read()
    request.initial_pose.position.x = float(-10)
    request.initial_pose.position.y = float(18)
    request.initial_pose.position.z = float(0.01)

    node.get_logger().info("Sending service request to `/spawn_entity`")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        print('response: %r' % future.result())
    else:
        raise RuntimeError(
            'exception while calling service: %r' % future.exception())

    node.get_logger().info("Done! Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
