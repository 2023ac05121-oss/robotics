#!/usr/bin/env python3

"""
Monte Carlo Localization ROS2 Node
"""

import math
import numpy as np
import time
import os
import yaml
import cv2
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, PoseWithCovarianceStamped, TransformStamped, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA, Header
from std_srvs.srv import Empty

import tf_transformations

class Particle:
    """Represents a single particle in the particle filter."""
    
    def __init__(self, x: float, y: float, theta: float, weight: float = 1.0):
        """
        Initialize a particle.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            theta: Orientation (in radians)
            weight: Particle weight (default 1.0)
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
    
    def __repr__(self):
        return f"Particle(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f}, weight={self.weight:.4f})"
    
    def to_pose(self) -> Pose:
        """Convert particle to ROS Pose message."""
        pose = Pose()
        pose.position.x = self.x
        pose.position.y = self.y
        pose.position.z = 0.0
        
        # Convert theta to quaternion
        q = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        
        return pose

class MCLNode(Node):
    """ROS2 Node for Monte Carlo Localization."""
    
    def __init__(self):
        super().__init__('mcl_localization')
        
        # Declare parameters
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('initial_pose_topic', '/initialpose')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('num_particles', 1000)
        self.declare_parameter('motion_noise', [0.1, 0.1, 0.1])  # x, y, theta
        self.declare_parameter('measurement_noise', 0.1)
        self.declare_parameter('global_localization', False)
        self.declare_parameter('update_rate', 10.0)  # Hz
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('laser_frame', 'base_scan')
        
        # Get parameters
        self.map_topic = self.get_parameter('map_topic').value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.num_particles = self.get_parameter('num_particles').value
        self.motion_noise = tuple(self.get_parameter('motion_noise').value)
        self.measurement_noise = self.get_parameter('measurement_noise').value
        self.global_localization = self.get_parameter('global_localization').value
        self.update_rate = self.get_parameter('update_rate').value
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.laser_frame = self.get_parameter('laser_frame').value
        
        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            QoSProfile(depth=1)
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            sensor_qos
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            sensor_qos
        )
        
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.initial_pose_callback,
            QoSProfile(depth=1)
        )
        
        # Create publishers
        self.particle_pub = self.create_publisher(
            PoseArray,
            'particlecloud',
            QoSProfile(depth=1)
        )
        
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            'amcl_pose',
            QoSProfile(depth=1)
        )
        
        self.particle_marker_pub = self.create_publisher(
            MarkerArray,
            'particle_markers',
            QoSProfile(depth=1)
        )
        
        # Services
        self.global_loc_srv = self.create_service(
            Empty,
            'global_localization',
            self.global_localization_callback
        )
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # State variables
        self.map_data = None
        self.map_info = None
        self.particles = []
        self.last_odom = None
        self.current_odom = None
        self.last_scan = None
        self.initialized = False
        self.last_update_time = self.get_clock().now()
        
        # Create timer
        self.timer = self.create_timer(1.0 / self.update_rate, self.update_callback)
        
        self.get_logger().info('MCL node initialized')
    
    def map_callback(self, msg: OccupancyGrid):
        """Process incoming map message."""
        self.get_logger().info('Received map')
        
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        
        if not self.initialized and self.global_localization:
            self.initialize_global()
    
    def scan_callback(self, msg: LaserScan):
        """Process incoming laser scan."""
        self.last_scan = msg
    
    def odom_callback(self, msg: Odometry):
        """Process incoming odometry."""
        if self.current_odom is None:
            self.current_odom = msg
            self.last_odom = msg
            return
        
        self.last_odom = self.current_odom
        self.current_odom = msg
    
    def initial_pose_callback(self, msg: PoseWithCovarianceStamped):
        """Process initial pose estimate."""
        self.get_logger().info('Received initial pose')
        
        if self.map_data is None:
            self.get_logger().warn('No map received yet, cannot initialize pose')
            return
        
        # Extract pose
        pose = msg.pose.pose
        position = pose.position
        orientation = pose.orientation
        
        # Convert quaternion to Euler angles
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(q)
        
        # Initialize particles around the given pose
        self.initialize_particles(position.x, position.y, yaw)
        
        self.initialized = True
    
    def global_localization_callback(self, request, response):
        """Perform global localization."""
        if self.map_data is None:
            self.get_logger().warn('No map received yet, cannot perform global localization')
            return response
        
        self.initialize_global()
        return response
    
    def initialize_global(self):
        """Initialize particles across the entire map."""
        if self.map_data is None:
            self.get_logger().warn('No map received yet, cannot initialize particles')
            return
        
        self.particles = []
        
        # Find free space in the map
        free_space = np.where(self.map_data < 50)
        
        if len(free_space[0]) == 0:
            self.get_logger().error('No free space found in the map')
            return
        
        # Randomly sample points in free space
        for _ in range(self.num_particles):
            idx = np.random.randint(0, len(free_space[0]))
            y, x = free_space[0][idx], free_space[1][idx]
            
            # Convert from grid to world coordinates
            world_x = x * self.map_info.resolution + self.map_info.origin.position.x
            world_y = y * self.map_info.resolution + self.map_info.origin.position.y
            theta = np.random.uniform(0, 2 * math.pi)
            
            self.particles.append(Particle(world_x, world_y, theta))
        
        self.initialized = True
        self.get_logger().info(f'Initialized {len(self.particles)} particles for global localization')
    
    def initialize_particles(self, x: float, y: float, theta: float):
        """Initialize particles around a given pose."""
        self.particles = []
        
        for _ in range(self.num_particles):
            # Add noise to position and orientation
            noisy_x = x + np.random.normal(0, self.motion_noise[0])
            noisy_y = y + np.random.normal(0, self.motion_noise[1])
            noisy_theta = theta + np.random.normal(0, self.motion_noise[2])
            
            # Normalize angle
            noisy_theta = (noisy_theta + math.pi) % (2 * math.pi) - math.pi
            
            self.particles.append(Particle(noisy_x, noisy_y, noisy_theta))
        
        self.get_logger().info(f'Initialized {len(self.particles)} particles around ({x}, {y}, {theta})')
    
    def update_callback(self):
        """Main update loop for MCL algorithm."""
        if not self.initialized or self.map_data is None or self.last_scan is None:
            return
        
        # Check if we have odometry data
        if self.last_odom is None or self.current_odom is None:
            return
        
        # Calculate time difference
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        
        if dt < 0.01:  # Minimum update interval
            return
        
        # Motion update
        self.motion_update()
        
        # Measurement update
        self.measurement_update()
        
        # Resample particles
        self.resample_particles()
        
        # Publish results
        self.publish_particles()
        self.publish_pose_estimate()
        self.publish_tf()
        
        self.last_update_time = current_time
    
    def motion_update(self):
        """Update particles based on odometry."""
        if len(self.particles) == 0:
            return
        
        # Extract poses
        last_pos = self.last_odom.pose.pose.position
        last_ori = self.last_odom.pose.pose.orientation
        curr_pos = self.current_odom.pose.pose.position
        curr_ori = self.current_odom.pose.pose.orientation
        
        # Convert to Euler angles
        last_q = [last_ori.x, last_ori.y, last_ori.z, last_ori.w]
        curr_q = [curr_ori.x, curr_ori.y, curr_ori.z, curr_ori.w]
        _, _, last_yaw = tf_transformations.euler_from_quaternion(last_q)
        _, _, curr_yaw = tf_transformations.euler_from_quaternion(curr_q)
        
        # Calculate motion in robot frame
        dx = curr_pos.x - last_pos.x
        dy = curr_pos.y - last_pos.y
        dtheta = curr_yaw - last_yaw
        
        # Normalize angle
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
        
        # Update each particle
        for particle in self.particles:
            # Add noise to motion
            noisy_dx = dx + np.random.normal(0, self.motion_noise[0])
            noisy_dy = dy + np.random.normal(0, self.motion_noise[1])
            noisy_dtheta = dtheta + np.random.normal(0, self.motion_noise[2])
            
            # Update particle position in map frame
            particle.x += noisy_dx * math.cos(particle.theta) - noisy_dy * math.sin(particle.theta)
            particle.y += noisy_dx * math.sin(particle.theta) + noisy_dy * math.cos(particle.theta)
            particle.theta += noisy_dtheta
            
            # Normalize angle
            particle.theta = (particle.theta + math.pi) % (2 * math.pi) - math.pi
    
    def measurement_update(self):
        """Update particle weights based on sensor measurements."""
        if len(self.particles) == 0 or self.last_scan is None:
            return
        
        # Extract scan data
        ranges = np.array(self.last_scan.ranges)
        angle_min = self.last_scan.angle_min
        angle_increment = self.last_scan.angle_increment
        max_range = self.last_scan.range_max
        
        # Generate angles for each beam
        angles = np.arange(len(ranges)) * angle_increment + angle_min
        
        # Update weights for each particle
        max_weight = 0.0
        
        for particle in self.particles:
            # Calculate expected measurements for this particle
            expected_ranges = self.get_expected_ranges(particle, angles, max_range)
            
            # Compare with actual measurements
            valid_indices = ~np.isnan(ranges) & ~np.isnan(expected_ranges)
            
            if np.sum(valid_indices) == 0:
                particle.weight = 1e-10
                continue
            
            # Calculate likelihood
            errors = ranges[valid_indices] - expected_ranges[valid_indices]
            likelihood = np.exp(-0.5 * (errors**2) / (self.measurement_noise**2))
            likelihood = np.mean(likelihood)  # Average likelihood across all beams
            
            # Update particle weight
            particle.weight *= likelihood
            
            # Track maximum weight
            max_weight = max(max_weight, particle.weight)
        
        # Normalize weights
        if max_weight > 0:
            for particle in self.particles:
                particle.weight /= max_weight
    
    def get_expected_ranges(self, particle: Particle, angles: np.ndarray, max_range: float) -> np.ndarray:
        """
        Calculate expected range measurements for a particle.
        
        Args:
            particle: Particle to calculate measurements for
            angles: Array of beam angles
            max_range: Maximum sensor range
            
        Returns:
            Array of expected range measurements
        """
        if self.map_data is None:
            return np.full_like(angles, max_range)
        
        expected_ranges = np.zeros_like(angles)
        
        for i, angle in enumerate(angles):
            # Calculate global angle
            global_angle = particle.theta + angle
            
            # Cast ray
            expected_ranges[i] = self.cast_ray(
                particle.x, particle.y, global_angle, max_range
            )
        
        return expected_ranges
    
    def cast_ray(self, x: float, y: float, angle: float, max_range: float) -> float:
        """
        Cast a ray from (x, y) in the direction of angle and return the distance to the nearest obstacle.
        
        Args:
            x: Starting x-coordinate in world frame
            y: Starting y-coordinate in world frame
            angle: Direction angle (in radians)
            max_range: Maximum range to check
            
        Returns:
            Distance to the nearest obstacle, or max_range if no obstacle is found
        """
        if self.map_data is None or self.map_info is None:
            return max_range
        
        # Convert world coordinates to map coordinates
        map_x = (x - self.map_info.origin.position.x) / self.map_info.resolution
        map_y = (y - self.map_info.origin.position.y) / self.map_info.resolution
        
        # Calculate ray direction
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Initialize distance
        distance = 0.0
        
        # Step size for ray casting (in world units)
        step_size = self.map_info.resolution * 0.5
        
        while distance < max_range:
            # Update position
            curr_x = x + distance * dx
            curr_y = y + distance * dy
            
            # Convert to map coordinates
            curr_map_x = (curr_x - self.map_info.origin.position.x) / self.map_info.resolution
            curr_map_y = (curr_y - self.map_info.origin.position.y) / self.map_info.resolution
            
            # Check if position is valid
            map_i, map_j = int(curr_map_y), int(curr_map_x)
            
            # Check if ray has hit an obstacle or gone outside the map
            if (map_j < 0 or map_j >= self.map_info.width or 
                map_i < 0 or map_i >= self.map_info.height or 
                self.map_data[map_i, map_j] > 50):  # > 50 means occupied in OccupancyGrid
                return distance
            
            # Increment distance
            distance += step_size
        
        # If no obstacle found within max_range, return max_range
        return max_range
    
    def resample_particles(self):
        """Resample particles based on their weights."""
        if len(self.particles) == 0:
            return
        
        # Normalize weights
        total_weight = sum(p.weight for p in self.particles)
        
        if total_weight <= 0:
            self.get_logger().warn('Total particle weight is zero or negative, skipping resampling')
            return
        
        for particle in self.particles:
            particle.weight /= total_weight
        
        # Create cumulative distribution
        cumulative_weights = np.zeros(len(self.particles))
        cumulative_sum = 0
        
        for i, particle in enumerate(self.particles):
            cumulative_sum += particle.weight
            cumulative_weights[i] = cumulative_sum
        
        # Resample using low variance sampler
        new_particles = []
        step = 1.0 / len(self.particles)
        r = np.random.uniform(0, step)
        c_idx = 0
        
        for i in range(len(self.particles)):
            u = r + i * step
            
            while u > cumulative_weights[c_idx]:
                c_idx += 1
                if c_idx >= len(self.particles):
                    c_idx = len(self.particles) - 1
                    break
            
            # Copy particle (with some noise to avoid sample impoverishment)
            old_particle = self.particles[c_idx]
            new_particle = Particle(
                old_particle.x + np.random.normal(0, self.motion_noise[0] * 0.1),
                old_particle.y + np.random.normal(0, self.motion_noise[1] * 0.1),
                old_particle.theta + np.random.normal(0, self.motion_noise[2] * 0.1),
                1.0 / len(self.particles)  # Equal weights after resampling
            )
            
            new_particles.append(new_particle)
        
        self.particles = new_particles
    
    def get_pose_estimate(self) -> Tuple[float, float, float]:
        """
        Calculate estimated pose as the weighted average of all particles.
        
        Returns:
            Tuple of (x, y, theta) representing the estimated position
        """
        if len(self.particles) == 0:
            return (0.0, 0.0, 0.0)
        
        x_sum = 0.0
        y_sum = 0.0
        cos_sum = 0.0
        sin_sum = 0.0
        
        for particle in self.particles:
            x_sum += particle.x * particle.weight
            y_sum += particle.y * particle.weight
            cos_sum += math.cos(particle.theta) * particle.weight
            sin_sum += math.sin(particle.theta) * particle.weight
        
        # Calculate average orientation using atan2 to handle angle wrapping
        avg_theta = math.atan2(sin_sum, cos_sum)
        
        return (x_sum, y_sum, avg_theta)
    
    def publish_particles(self):
        """Publish particle cloud."""
        if len(self.particles) == 0:
            return
        
        # Create PoseArray message
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        
        # Add particles
        for particle in self.particles:
            msg.poses.append(particle.to_pose())
        
        self.particle_pub.publish(msg)
        
        # Also publish as markers with weights visualized
        self.publish_particle_markers()
    
    def publish_particle_markers(self):
        """Publish particles as markers with weights visualized."""
        if len(self.particles) == 0:
            return
        
        # Create MarkerArray message
        marker_array = MarkerArray()
        
        # Add markers
        for i, particle in enumerate(self.particles):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "particles"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Position and orientation
            marker.pose = particle.to_pose()
            
            # Scale (arrow size)
            marker.scale.x = 0.3 * particle.weight * len(self.particles)  # Length
            marker.scale.y = 0.05  # Width
            marker.scale.z = 0.05  # Height
            
            # Color (blue with alpha based on weight)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = max(0.1, min(1.0, particle.weight * len(self.particles)))
            
            marker_array.markers.append(marker)
        
        self.particle_marker_pub.publish(marker_array)
    
    def publish_pose_estimate(self):
        """Publish the estimated pose."""
        if len(self.particles) == 0:
            return
        
        # Get pose estimate
        x, y, theta = self.get_pose_estimate()
        
        # Create PoseWithCovarianceStamped message
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        
        # Set position
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        
        # Set orientation (convert theta to quaternion)
        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        
        # Set covariance (simplified)
        cov = [0.0] * 36
        cov[0] = 0.1  # x variance
        cov[7] = 0.1  # y variance
        cov[35] = 0.1  # theta variance
        msg.pose.covariance = cov
        
        self.pose_pub.publish(msg)
    
    def publish_tf(self):
        """Publish the transform from map to base_footprint."""
        if len(self.particles) == 0:
            return
        
        # Get pose estimate
        x, y, theta = self.get_pose_estimate()
        
        # Create transform message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.base_frame
        
        # Set translation
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        
        # Set rotation
        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        # Publish transform
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
