from setuptools import setup

package_name = 'env_filter'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/env_filter.yaml']),
        ('share/' + package_name + '/launch', ['launch/env_filter.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ywy',
    maintainer_email='TODO',
    description='Safety filter node: /cmd_vel_bci + /scan -> /cmd_vel',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'env_filter = env_filter.env_filter:main',
        ],
    },
)
