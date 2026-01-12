from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory("bci_interface")
    params = os.path.join(pkg, "config", "bci.yaml")
    return LaunchDescription([
        Node(
            package="bci_interface",
            executable="bci_node",
            name="bci_node",
            output="screen",
            parameters=[params],
        )
    ])