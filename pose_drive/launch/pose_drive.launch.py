from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    pose_drive = Node(
        package="pose_drive",
        executable="pose_drive",
        name="pose_drive",
        remappings=[],
        parameters=[],
    )

    ld.add_action(pose_drive)

    return ld