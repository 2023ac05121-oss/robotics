#!/usr/bin/env python3
"""
Utility functions for robotics algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import random
from typing import List, Tuple, Optional

def create_random_map(width: int, height: int, obstacle_prob: float = 0.2, 
                     min_obstacle_size: int = 1, max_obstacle_size: int = 5,
                     random_seed: Optional[int] = None) -> np.ndarray:
    """
    Create a random occupancy grid map.
    
    Args:
        width: Width of the map
        height: Height of the map
        obstacle_prob: Probability of placing an obstacle
        min_obstacle_size: Minimum size of obstacles
        max_obstacle_size: Maximum size of obstacles
        random_seed: Optional random seed for reproducibility
    
    Returns:
        2D numpy array representing the map (0 = free, 1 = obstacle)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    # Initialize empty map
    map_data = np.zeros((height, width))
    
    # Add random obstacles
    for y in range(0, height, max_obstacle_size):
        for x in range(0, width, max_obstacle_size):
            if random.random() < obstacle_prob:
                # Random obstacle size
                size_x = random.randint(min_obstacle_size, max_obstacle_size)
                size_y = random.randint(min_obstacle_size, max_obstacle_size)
                
                # Ensure obstacle fits within map
                end_x = min(x + size_x, width)
                end_y = min(y + size_y, height)
                
                # Place obstacle
                map_data[y:end_y, x:end_x] = 1
    
    return map_data

def add_border_to_map(map_data: np.ndarray, border_width: int = 1) -> np.ndarray:
    """
    Add a border of obstacles around the map.
    
    Args:
        map_data: Input map
        border_width: Width of the border
    
    Returns:
        Map with border added
    """
    height, width = map_data.shape
    
    # Copy the map
    bordered_map = map_data.copy()
    
    # Add borders
    bordered_map[:border_width, :] = 1  # Top
    bordered_map[-border_width:, :] = 1  # Bottom
    bordered_map[:, :border_width] = 1  # Left
    bordered_map[:, -border_width:] = 1  # Right
    
    return bordered_map

def visualize_map(map_data: np.ndarray, title: str = "Map", save_path: Optional[str] = None):
    """
    Visualize a map.
    
    Args:
        map_data: 2D numpy array representing the map
        title: Plot title
        save_path: Optional path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(map_data, cmap='gray_r', origin='lower')
    plt.title(title)
    plt.colorbar(label='Occupancy')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def calculate_path_length(path: List[Tuple[float, float]]) -> float:
    """
    Calculate the length of a path.
    
    Args:
        path: List of (x, y) coordinates
    
    Returns:
        Total path length
    """
    if not path or len(path) < 2:
        return 0.0
    
    length = 0.0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return length

def generate_noisy_odometry(true_motion: Tuple[float, float, float], 
                           noise_level: float = 0.1) -> Tuple[float, float, float]:
    """
    Generate noisy odometry readings.
    
    Args:
        true_motion: True motion as (dx, dy, dtheta)
        noise_level: Standard deviation of noise as fraction of motion
    
    Returns:
        Noisy odometry as (dx, dy, dtheta)
    """
    dx, dy, dtheta = true_motion
    
    # Add noise proportional to motion
    noisy_dx = dx + random.gauss(0, abs(dx) * noise_level)
    noisy_dy = dy + random.gauss(0, abs(dy) * noise_level)
    noisy_dtheta = dtheta + random.gauss(0, abs(dtheta) * noise_level + 0.01)  # Add small constant to avoid zero noise for zero rotation
    
    return (noisy_dx, noisy_dy, noisy_dtheta)

def generate_noisy_sensor_readings(true_readings: List[float], 
                                  noise_level: float = 0.1,
                                  max_range: float = float('inf')) -> List[float]:
    """
    Generate noisy sensor readings.
    
    Args:
        true_readings: List of true distance readings
        noise_level: Standard deviation of noise as fraction of reading
        max_range: Maximum sensor range
    
    Returns:
        List of noisy readings
    """
    noisy_readings = []
    
    for reading in true_readings:
        # Add noise proportional to reading
        noise = random.gauss(0, reading * noise_level + 0.01)  # Add small constant to avoid zero noise for zero readings
        noisy_reading = max(0.0, min(reading + noise, max_range))  # Clamp to valid range
        noisy_readings.append(noisy_reading)
    
    return noisy_readings

def transform_pose(pose: Tuple[float, float, float], 
                  transform: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Apply a transform to a pose.
    
    Args:
        pose: Original pose as (x, y, theta)
        transform: Transform to apply as (dx, dy, dtheta) in the global frame
    
    Returns:
        Transformed pose as (x, y, theta)
    """
    x, y, theta = pose
    dx, dy, dtheta = transform
    
    new_x = x + dx
    new_y = y + dy
    new_theta = (theta + dtheta) % (2 * math.pi)
    
    return (new_x, new_y, new_theta)

def transform_local_to_global(local_pose: Tuple[float, float, float], 
                             reference_pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Transform a pose from local to global coordinates.
    
    Args:
        local_pose: Local pose as (x, y, theta) relative to reference
        reference_pose: Reference pose as (x, y, theta) in global frame
    
    Returns:
        Global pose as (x, y, theta)
    """
    local_x, local_y, local_theta = local_pose
    ref_x, ref_y, ref_theta = reference_pose
    
    # Rotate local coordinates
    global_x = ref_x + local_x * math.cos(ref_theta) - local_y * math.sin(ref_theta)
    global_y = ref_y + local_x * math.sin(ref_theta) + local_y * math.cos(ref_theta)
    global_theta = (ref_theta + local_theta) % (2 * math.pi)
    
    return (global_x, global_y, global_theta)

def transform_global_to_local(global_pose: Tuple[float, float, float], 
                             reference_pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Transform a pose from global to local coordinates.
    
    Args:
        global_pose: Global pose as (x, y, theta)
        reference_pose: Reference pose as (x, y, theta) in global frame
    
    Returns:
        Local pose as (x, y, theta) relative to reference
    """
    global_x, global_y, global_theta = global_pose
    ref_x, ref_y, ref_theta = reference_pose
    
    # Translate and rotate to local frame
    dx = global_x - ref_x
    dy = global_y - ref_y
    
    local_x = dx * math.cos(-ref_theta) - dy * math.sin(-ref_theta)
    local_y = dx * math.sin(-ref_theta) + dy * math.cos(-ref_theta)
    local_theta = (global_theta - ref_theta) % (2 * math.pi)
    
    return (local_x, local_y, local_theta)

def calculate_pose_error(estimated_pose: Tuple[float, float, float], 
                        true_pose: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Calculate position and orientation error between estimated and true poses.
    
    Args:
        estimated_pose: Estimated pose as (x, y, theta)
        true_pose: True pose as (x, y, theta)
    
    Returns:
        Tuple of (position_error, orientation_error)
    """
    est_x, est_y, est_theta = estimated_pose
    true_x, true_y, true_theta = true_pose
    
    # Position error (Euclidean distance)
    position_error = math.sqrt((est_x - true_x)**2 + (est_y - true_y)**2)
    
    # Orientation error (normalize to [-pi, pi])
    orientation_error = abs((est_theta - true_theta + math.pi) % (2 * math.pi) - math.pi)
    
    return (position_error, orientation_error)

def cast_ray(map_data: np.ndarray, x: float, y: float, angle: float, 
            max_range: float = float('inf')) -> float:
    """
    Cast a ray from (x, y) in the direction of angle and return the distance to the nearest obstacle.
    
    Args:
        map_data: 2D numpy array representing the map (0 = free, 1 = obstacle)
        x: Starting x-coordinate
        y: Starting y-coordinate
        angle: Direction angle (in radians)
        max_range: Maximum range to check
        
    Returns:
        Distance to the nearest obstacle, or max_range if no obstacle is found
    """
    height, width = map_data.shape
    
    # Calculate ray direction
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    # Initialize distance
    distance = 0.0
    
    # Step size for ray casting
    step_size = 0.1
    
    while distance < max_range:
        # Update position
        curr_x = x + distance * dx
        curr_y = y + distance * dy
        
        # Check if position is valid
        map_x, map_y = int(curr_x), int(curr_y)
        
        # Check if ray has hit an obstacle or gone outside the map
        if (map_x < 0 or map_x >= width or 
            map_y < 0 or map_y >= height or 
            map_data[map_y, map_x] > 0.5):
            return distance
        
        # Increment distance
        distance += step_size
    
    # If no obstacle found within max_range, return max_range
    return max_range

def simulate_range_sensor(map_data: np.ndarray, pose: Tuple[float, float, float], 
                         sensor_angles: List[float], max_range: float = float('inf'),
                         noise_level: float = 0.0) -> List[float]:
    """
    Simulate range sensor readings for a given pose.
    
    Args:
        map_data: 2D numpy array representing the map
        pose: Robot pose as (x, y, theta)
        sensor_angles: List of sensor angles relative to the robot's orientation
        max_range: Maximum sensor range
        noise_level: Standard deviation of noise as fraction of reading
    
    Returns:
        List of distance readings
    """
    x, y, theta = pose
    readings = []
    
    for angle in sensor_angles:
        # Calculate absolute angle
        abs_angle = theta + angle
        
        # Cast a ray
        distance = cast_ray(map_data, x, y, abs_angle, max_range)
        
        # Add noise if specified
        if noise_level > 0:
            noise = random.gauss(0, distance * noise_level)
            distance = max(0.0, distance + noise)
        
        readings.append(min(distance, max_range))
    
    return readings

# Example usage
if __name__ == "__main__":
    # Create a random map
    random_map = create_random_map(50, 50, obstacle_prob=0.2, random_seed=42)
    random_map = add_border_to_map(random_map)
    
    # Visualize the map
    visualize_map(random_map, "Random Map", "random_map.png")
    
    # Test ray casting
    start_pos = (25, 25)
    angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
    max_range = 20.0
    
    # Get sensor readings
    pose = (start_pos[0], start_pos[1], 0.0)
    readings = simulate_range_sensor(random_map, pose, angles, max_range)
    
    # Visualize sensor readings
    plt.figure(figsize=(10, 8))
    plt.imshow(random_map, cmap='gray_r', origin='lower')
    
    # Plot the robot position
    plt.scatter([pose[0]], [pose[1]], c='r', s=100, marker='o')
    
    # Plot sensor rays
    for i, angle in enumerate(angles):
        end_x = pose[0] + readings[i] * math.cos(angle)
        end_y = pose[1] + readings[i] * math.sin(angle)
        plt.plot([pose[0], end_x], [pose[1], end_y], 'r-', alpha=0.5)
    
    plt.title("Simulated Range Sensor")
    plt.savefig("sensor_simulation.png")
    plt.close()
    
    print("Utility functions test complete.")

def create_artificial_map(map_size: Tuple[int, int], obstacle_count: int) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
    """
    Create an artificial map with randomly placed rectangular obstacles.
    
    Args:
        map_size: Tuple of (width, height)
        obstacle_count: Number of obstacles to place
        
    Returns:
        Tuple of (map_data, obstacle_list)
        - map_data: 2D numpy array (0 = free, 1 = obstacle)
        - obstacle_list: List of (x, y, width, height) for each obstacle
    """
    width, height = map_size
    map_data = np.zeros((height, width))
    
    # Add border
    map_data = add_border_to_map(map_data, border_width=1)
    
    # Create random obstacles
    obstacles = []
    for _ in range(obstacle_count):
        # Random position and size for obstacle
        obs_width = random.randint(3, 10)
        obs_height = random.randint(3, 10)
        obs_x = random.randint(5, width - obs_width - 5)
        obs_y = random.randint(5, height - obs_height - 5)
        
        # Add obstacle to map
        map_data[obs_y:obs_y+obs_height, obs_x:obs_x+obs_width] = 1
        
        # Store obstacle info
        obstacles.append((obs_x, obs_y, obs_width, obs_height))
    
    return map_data, obstacles

def simulate_robot_motion(position: np.ndarray, control: np.ndarray, 
                         noise: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                         dt: float = 1.0) -> np.ndarray:
    """
    Simulate robot motion using a simple motion model with noise.
    
    Args:
        position: Current position array [x, y, theta]
        control: Control input [linear_velocity, angular_velocity]
        noise: Motion noise as standard deviations (x, y, theta)
        dt: Time step
        
    Returns:
        New position array [x, y, theta]
    """
    x, y, theta = position
    v, omega = control  # Linear and angular velocity
    
    # Apply motion model with noise
    if abs(omega) < 1e-6:  # Straight line motion
        x_new = x + v * dt * math.cos(theta)
        y_new = y + v * dt * math.sin(theta)
        theta_new = theta
    else:  # Circular motion
        radius = v / omega
        x_new = x + radius * (-math.sin(theta) + math.sin(theta + omega * dt))
        y_new = y + radius * (math.cos(theta) - math.cos(theta + omega * dt))
        theta_new = theta + omega * dt
    
    # Add noise
    if noise[0] > 0:
        x_new += np.random.normal(0, noise[0])
    if noise[1] > 0:
        y_new += np.random.normal(0, noise[1])
    if noise[2] > 0:
        theta_new += np.random.normal(0, noise[2])
    
    # Normalize angle
    theta_new = (theta_new + math.pi) % (2 * math.pi) - math.pi
    
    return np.array([x_new, y_new, theta_new])

def is_position_occupied(map_data: np.ndarray, position: Tuple[float, float]) -> bool:
    """
    Check if a position is occupied in the map.
    
    Args:
        map_data: 2D map array
        position: Position to check (x, y)
        
    Returns:
        True if position is occupied, False otherwise
    """
    x, y = position
    height, width = map_data.shape
    
    # Check boundaries
    if (x < 0 or x >= width or y < 0 or y >= height):
        return True  # Out of bounds is considered occupied
    
    # Check cell occupancy
    cell_x, cell_y = int(x), int(y)
    return map_data[cell_y, cell_x] == 1

def cast_ray(map_data: np.ndarray, start: Tuple[float, float], angle: float, 
            max_range: float) -> float:
    """
    Cast a ray in the map and return distance to the nearest obstacle.
    Uses Bresenham's line algorithm for ray casting.
    
    Args:
        map_data: 2D map array
        start: Starting point (x, y)
        angle: Ray angle in radians
        max_range: Maximum ray length
        
    Returns:
        Distance to the nearest obstacle or max_range if none found
    """
    height, width = map_data.shape
    start_x, start_y = start
    
    # Check boundaries
    if (start_x < 0 or start_x >= width or 
        start_y < 0 or start_y >= height):
        return 0.0  # Out of bounds
    
    # Calculate end point
    end_x = start_x + max_range * math.cos(angle)
    end_y = start_y + max_range * math.sin(angle)
    
    # Bresenham's line algorithm
    x0, y0 = int(start_x), int(start_y)
    x1, y1 = int(end_x), int(end_y)
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    current_distance = 0.0
    
    while True:
        # Check if current point is an obstacle
        if (x0 < 0 or x0 >= width or y0 < 0 or y0 >= height or map_data[y0, x0] == 1):
            # Calculate distance from start to obstacle
            dx_to_obs = x0 - start_x
            dy_to_obs = y0 - start_y
            return math.sqrt(dx_to_obs**2 + dy_to_obs**2)
        
        # Check if we've reached max range
        current_distance = math.sqrt((x0 - start_x)**2 + (y0 - start_y)**2)
        if current_distance >= max_range:
            return max_range
        
        # Move to next point
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def simulate_sensor_readings(position: np.ndarray, map_data: np.ndarray, 
                            num_rays: int = 36, max_range: float = 50.0,
                            noise: float = 0.0) -> np.ndarray:
    """
    Simulate robot sensor readings (laser scan) in a map.
    
    Args:
        position: Robot position [x, y, theta]
        map_data: 2D numpy array representing the map (0 = free, 1 = obstacle)
        num_rays: Number of sensor rays
        max_range: Maximum sensor range
        noise: Standard deviation of measurement noise
        
    Returns:
        Array of distance measurements
    """
    x, y, theta = position
    height, width = map_data.shape
    
    # Check if robot is inside an obstacle
    if is_position_occupied(map_data, (x, y)):
        return np.full(num_rays, 0.0)  # All readings are 0 if robot is inside obstacle
    
    # Initialize array for distance measurements
    measurements = np.zeros(num_rays)
    
    # Calculate angles for each ray
    angles = np.linspace(theta - math.pi, theta + math.pi, num_rays, endpoint=False)
    
    # Cast rays
    for i, angle in enumerate(angles):
        # Ray casting with Bresenham's line algorithm
        distance = cast_ray(map_data, (x, y), angle, max_range)
        
        # Add noise
        if noise > 0:
            distance += np.random.normal(0, noise)
            distance = max(0, distance)  # Ensure non-negative
        
        measurements[i] = distance
    
    return measurements
