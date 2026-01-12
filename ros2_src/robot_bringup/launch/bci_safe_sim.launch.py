import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource, FrontendLaunchDescriptionSource


def generate_launch_description():
    # 1) Gazebo + spawn + bridge (XML)
    robot_gazebo_share = get_package_share_directory('robot_gazebo')
    sim_launch_xml = os.path.join(robot_gazebo_share, 'launch', 'launch_sim.launch.xml')

    # 2) env_filter (PY)
    env_filter_share = get_package_share_directory('env_filter')
    env_filter_launch_py = os.path.join(env_filter_share, 'launch', 'env_filter.launch.py')

    # 3) bci_interface (PY)
    bci_share = get_package_share_directory('bci_interface')
    bci_launch_py = os.path.join(bci_share, 'launch', 'bci.launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(sim_launch_xml)
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(env_filter_launch_py)
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bci_launch_py)
        ),
    ])
