from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('mcl_localization')
    
    # Define launch arguments
    map_yaml_path_arg = DeclareLaunchArgument(
        'map_yaml_path',
        default_value=os.path.join(pkg_dir, 'maps', 'map.yaml'),
        description='Path to the map YAML file'
    )
    
    map_topic_arg = DeclareLaunchArgument(
        'map_topic',
        default_value='/map',
        description='Topic on which to publish the map'
    )
    
    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/scan',
        description='Topic for laser scan data'
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/odom',
        description='Topic for odometry data'
    )
    
    global_localization_arg = DeclareLaunchArgument(
        'global_localization',
        default_value='false',
        description='Whether to initialize with global localization'
    )
    
    # Map server node
    map_server_node = Node(
        package='mcl_localization',
        executable='map_server',
        name='map_server',
        parameters=[{
            'map_yaml_path': LaunchConfiguration('map_yaml_path'),
            'map_topic': LaunchConfiguration('map_topic'),
            'publish_rate': 1.0
        }]
    )
    
    # MCL node
    mcl_node = Node(
        package='mcl_localization',
        executable='mcl_node',
        name='mcl_localization',
        parameters=[{
            'map_topic': LaunchConfiguration('map_topic'),
            'scan_topic': LaunchConfiguration('scan_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'num_particles': 1000,
            'motion_noise': [0.1, 0.1, 0.1],
            'measurement_noise': 0.1,
            'global_localization': LaunchConfiguration('global_localization'),
            'update_rate': 10.0,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'laser_frame': 'base_scan'
        }]
    )
    
    return LaunchDescription([
        map_yaml_path_arg,
        map_topic_arg,
        scan_topic_arg,
        odom_topic_arg,
        global_localization_arg,
        map_server_node,
        mcl_node
    ])
