from launch import LaunchDescription
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('wifi_com'),
        'config',
        'wifi_com.yaml'
    )

    return LaunchDescription([
        Node(
            package='wifi_com',
            executable='wifi_com_node',
            name='wifi_com_node',
            output='screen',
            parameters=[config]
        )
    ])
