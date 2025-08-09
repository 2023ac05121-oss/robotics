from setuptools import setup
import os
from glob import glob

package_name = 'mcl_localization'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='BITS Student',
    maintainer_email='student@example.com',
    description='Monte Carlo Localization implementation for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mcl_node = mcl_localization.mcl_node:main',
            'map_server = mcl_localization.map_server:main',
        ],
    },
)
