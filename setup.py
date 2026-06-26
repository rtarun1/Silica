import os
from glob import glob
from setuptools import setup

package_name = 'silica_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=['silica_core', 'silica_ros'],
    package_dir={
        'silica_core': 'src/silica_core',
        # Do NOT explicitly map silica_ros here; setuptools finds it automatically.
    },
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'], 
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='dev@todo.todo',
    description='Silica ML Core and ROS 2 Wrapper',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'silica_node = silica_ros.silica_node:main'
        ],
    },
)