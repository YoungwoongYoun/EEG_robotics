from setuptools import setup
from glob import glob
import os

package_name = 'bci_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml') + glob('config/*.yml') + glob('config/*.pt') + glob('config/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'bci_node = bci_interface.bci_node:main',
        ],
    },
)
