from setuptools import find_packages, setup
from glob import glob

package_name = 'robot_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch_sim.launch.xml']),
        ('share/' + package_name + '/params', ['params/robot_bridge.yaml']),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ywy',
    maintainer_email='ywy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={'console_scripts': []},
)
