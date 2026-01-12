from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg = FindPackageShare("robot_description")
    xacro_file = PathJoinSubstitution([pkg, "urdf", "robot.urdf.xacro"])

    robot_description = ParameterValue(
        Command(["xacro ", xacro_file]),
        value_type=str
    )

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),

        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            output="screen",
        ),
    ])
