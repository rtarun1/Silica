import os
from glob import glob
from setuptools import setup

package_name = 'silica_ros'

setup(
    name=package_name,
    version='0.1.0',
    # List both packages
    packages=['silica_ros', 'silica_core', 'silica_core.utils'],
    # Only map silica_core, let setuptools find silica_ros automatically
    package_dir={
        'silica_core': 'src/silica_core'
    },
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'], 
    zip_safe=True,
    maintainer='Tarun R',
    maintainer_email='tarun.ramak@gmail.com',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'silica_node = silica_ros.silica_node:main'
        ],
    },
)