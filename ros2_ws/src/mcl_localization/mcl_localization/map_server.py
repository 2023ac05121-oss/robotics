#!/usr/bin/env python3

"""
Map server for MCL localization
"""

import os
import yaml
import cv2
import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion

class MapServer(Node):
    """ROS2 node for serving map data."""
    
    def __init__(self):
        super().__init__('map_server')
        
        # Declare parameters
        self.declare_parameter('map_yaml_path', '')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('publish_rate', 1.0)  # Hz
        
        # Get parameters
        self.map_yaml_path = self.get_parameter('map_yaml_path').value
        self.map_topic = self.get_parameter('map_topic').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Create map publisher
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            self.map_topic,
            10
        )
        
        # Load map
        self.load_map()
        
        # Create timer for publishing
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_map)
        
        self.get_logger().info(f'Map server initialized, publishing to {self.map_topic}')
    
    def load_map(self):
        """Load map from YAML file."""
        if not self.map_yaml_path:
            self.get_logger().error('No map YAML path specified')
            return
        
        if not os.path.exists(self.map_yaml_path):
            self.get_logger().error(f'Map YAML file not found: {self.map_yaml_path}')
            return
        
        try:
            with open(self.map_yaml_path, 'r') as f:
                map_data = yaml.safe_load(f)
            
            # Get map image path
            map_image_path = map_data.get('image')
            if not map_image_path:
                self.get_logger().error('No image path specified in map YAML')
                return
            
            # If path is relative, make it absolute
            if not os.path.isabs(map_image_path):
                map_dir = os.path.dirname(self.map_yaml_path)
                map_image_path = os.path.join(map_dir, map_image_path)
            
            if not os.path.exists(map_image_path):
                self.get_logger().error(f'Map image file not found: {map_image_path}')
                return
            
            # Load image
            img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Try with PIL
                img = np.array(Image.open(map_image_path).convert('L'))
            
            if img is None:
                self.get_logger().error(f'Failed to load map image: {map_image_path}')
                return
            
            # Get map metadata
            resolution = map_data.get('resolution', 0.05)
            origin = map_data.get('origin', [0, 0, 0])
            occupied_thresh = map_data.get('occupied_thresh', 0.65)
            free_thresh = map_data.get('free_thresh', 0.196)
            
            # Convert image to occupancy grid
            # In ROS, 0 = free, 100 = occupied, -1 = unknown
            grid = np.zeros(img.shape, dtype=np.int8)
            grid[img >= (occupied_thresh * 255)] = 100  # Occupied
            grid[img < (free_thresh * 255)] = 0  # Free
            grid[(img >= (free_thresh * 255)) & (img < (occupied_thresh * 255))] = -1  # Unknown
            
            # Create message
            self.map_msg = OccupancyGrid()
            self.map_msg.info.resolution = resolution
            self.map_msg.info.width = img.shape[1]
            self.map_msg.info.height = img.shape[0]
            
            # Set origin (ROS uses x-right, y-up, while images use x-right, y-down)
            self.map_msg.info.origin.position.x = origin[0]
            self.map_msg.info.origin.position.y = origin[1]
            self.map_msg.info.origin.position.z = 0.0
            
            # Flatten grid for ROS message (row-major order)
            self.map_msg.data = grid.flatten().tolist()
            
            self.get_logger().info(f'Map loaded: {map_image_path}, {img.shape[1]}x{img.shape[0]} at {resolution}m/px')
            
        except Exception as e:
            self.get_logger().error(f'Error loading map: {str(e)}')
    
    def publish_map(self):
        """Publish the map."""
        if hasattr(self, 'map_msg'):
            self.map_msg.header.stamp = self.get_clock().now().to_msg()
            self.map_msg.header.frame_id = 'map'
            self.map_pub.publish(self.map_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MapServer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
