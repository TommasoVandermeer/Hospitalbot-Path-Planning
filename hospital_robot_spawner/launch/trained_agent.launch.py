import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ld = LaunchDescription()

    trained_agent = Node(
        package='hospital_robot_spawner',
        executable='trained_agent',
        #name='hospitalbot_training',
    )

    ld.add_action(trained_agent)

    return ld