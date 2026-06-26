import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    set_env_var = SetEnvironmentVariable(
        name='PYTORCH_ALLOC_CONF',
        value='expandable_segments:True'
    )

    realsense_launch_dir = os.path.join(get_package_share_directory('realsense2_camera'), 'launch')
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(realsense_launch_dir, 'rs_launch.py')),
        launch_arguments={
            'camera_name': 'camera1',
            'camera_namespace': 'camera1'
        }.items()
    )

    # Use the config from our new package installation
    params_file_path = os.path.join(
        get_package_share_directory('silica_ros'),
        'config',
        'params.yaml'
    )

    ml_node = Node(
        package='silica_ros',
        executable='silica_node',
        name='silica_node',
        output='screen',
        parameters=[params_file_path]
    )

    return LaunchDescription([
        set_env_var,
        realsense_node,
        ml_node,
    ])