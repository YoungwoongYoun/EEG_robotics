from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("env_filter")
    params = os.path.join(pkg_share, "config", "env_filter.yaml")

    return LaunchDescription([
        Node(
            package="env_filter",
            executable="env_filter",
            name="env_filter",
            output="screen",
            parameters=[params],
        )
    ])
