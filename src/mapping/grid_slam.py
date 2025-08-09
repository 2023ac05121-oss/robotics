#!/usr/bin/env python3
"""
Grid-based SLAM (Simultaneous Localization and Mapping) implementation.
This algorithm builds a map of the environment while simultaneously localizing the robot.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from typing import List, Tuple, Optional
import time

class GridCell:
    """Represents a single cell in the occupancy grid."""
    
    def __init__(self, log_odds: float = 0.0):
        """
        Initialize a grid cell.
        
        Args:
            log_odds: Log odds ratio of the cell being occupied (default 0.0 = 0.5 probability)
        """
        self.log_odds = log_odds
    
    @property
    def probability(self) -> float:
        """Convert log odds to probability."""
        return 1.0 - (1.0 / (1.0 + math.exp(self.log_odds)))
    
    def update(self, measurement: float, free_update: float = -0.4, occupied_update: float = 0.9):
        """
        Update the cell's log odds based on measurement.
        
        Args:
            measurement: 1.0 if cell is measured as occupied, 0.0 if free
            free_update: Value to add to log_odds when cell is measured as free
            occupied_update: Value to add to log_odds when cell is measured as occupied
        """
        if measurement > 0.5:  # Occupied
            self.log_odds += occupied_update
        else:  # Free
            self.log_odds += free_update
        
        # Clamp to reasonable range to prevent overflow
        self.log_odds = max(-100.0, min(100.0, self.log_odds))

class GridSLAM:
    """Implementation of Grid-based SLAM algorithm."""
    
    def __init__(self, grid_size: Tuple[int, int] = (100, 100), cell_size: float = 0.1,
                 initial_position: Tuple[float, float, float] = (50.0, 50.0, 0.0),
                 motion_noise: Tuple[float, float, float] = (0.1, 0.1, 0.05),
                 measurement_noise: float = 0.1):
        """
        Initialize the Grid SLAM algorithm.
        
        Args:
            grid_size: Tuple of (width, height) in grid cells
            cell_size: Size of each grid cell in meters
            initial_position: Initial robot position (x, y, theta)
            motion_noise: Tuple of (x, y, theta) standard deviations for motion model
            measurement_noise: Standard deviation for measurement model
        """
        self.width, self.height = grid_size
        self.cell_size = cell_size
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        
        # Initialize occupancy grid
        self.grid = [[GridCell() for _ in range(self.width)] for _ in range(self.height)]
        
        # Robot pose (in grid coordinates)
        self.x, self.y, self.theta = initial_position
        
        # Path history
        self.path = [(self.x, self.y, self.theta)]
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid cell indices.
        
        Args:
            x: X-coordinate in world frame
            y: Y-coordinate in world frame
            
        Returns:
            Tuple of (grid_x, grid_y) indices
        """
        grid_x = int(x)
        grid_y = int(y)
        return grid_x, grid_y
    
    def is_valid_cell(self, x: int, y: int) -> bool:
        """
        Check if grid cell indices are valid.
        
        Args:
            x: Grid x-index
            y: Grid y-index
            
        Returns:
            True if cell indices are within grid bounds
        """
        return 0 <= x < self.width and 0 <= y < self.height
    
    def update_map(self, measurements: List[float], sensor_angles: List[float], max_range: float):
        """
        Update the occupancy grid based on sensor measurements.
        
        Args:
            measurements: List of distance measurements from sensors
            sensor_angles: List of angles (in radians) corresponding to each measurement
            max_range: Maximum range of the sensors
        """
        if len(measurements) != len(sensor_angles):
            raise ValueError("Number of measurements must match number of sensor angles")
        
        # For each measurement
        for i, (measurement, angle) in enumerate(zip(measurements, sensor_angles)):
            # Calculate absolute angle of this sensor ray
            abs_angle = self.theta + angle
            
            # Ray-cast from robot position to the measured point
            self.update_cells_along_ray(self.x, self.y, abs_angle, measurement, max_range)
    
    def update_cells_along_ray(self, x: float, y: float, angle: float, 
                              measurement: float, max_range: float):
        """
        Update occupancy probabilities for cells along a sensor ray.
        
        Args:
            x: Starting x-coordinate
            y: Starting y-coordinate
            angle: Direction angle (in radians)
            measurement: Measured distance
            max_range: Maximum sensor range
        """
        # Calculate ray direction
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Step size for ray casting (in grid cells)
        step_size = 0.5
        
        # Current distance along ray
        distance = 0.0
        
        # Mark cells along the ray as free
        while distance < min(measurement, max_range):
            # Calculate world coordinates
            curr_x = x + distance * dx
            curr_y = y + distance * dy
            
            # Convert to grid coordinates
            grid_x, grid_y = self.world_to_grid(curr_x, curr_y)
            
            # Update cell if valid
            if self.is_valid_cell(grid_x, grid_y):
                self.grid[grid_y][grid_x].update(0.0)  # Mark as free
            
            # Increment distance
            distance += step_size
        
        # If we have a valid measurement (not max_range), mark the endpoint as occupied
        if measurement < max_range:
            # Calculate endpoint
            end_x = x + measurement * dx
            end_y = y + measurement * dy
            
            # Add some noise to endpoint (to account for uncertainty)
            end_x += random.gauss(0, self.measurement_noise)
            end_y += random.gauss(0, self.measurement_noise)
            
            # Convert to grid coordinates
            grid_x, grid_y = self.world_to_grid(end_x, end_y)
            
            # Update cell if valid
            if self.is_valid_cell(grid_x, grid_y):
                self.grid[grid_y][grid_x].update(1.0)  # Mark as occupied
    
    def move_robot(self, dx: float, dy: float, dtheta: float):
        """
        Update robot pose based on motion command.
        
        Args:
            dx: Change in x-coordinate (in local robot frame)
            dy: Change in y-coordinate (in local robot frame)
            dtheta: Change in orientation (in radians)
        """
        # Add noise to motion
        noisy_dx = dx + random.gauss(0, self.motion_noise[0])
        noisy_dy = dy + random.gauss(0, self.motion_noise[1])
        noisy_dtheta = dtheta + random.gauss(0, self.motion_noise[2])
        
        # Update robot pose
        self.x += noisy_dx * math.cos(self.theta) - noisy_dy * math.sin(self.theta)
        self.y += noisy_dx * math.sin(self.theta) + noisy_dy * math.cos(self.theta)
        self.theta += noisy_dtheta
        
        # Normalize angle
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        
        # Record path
        self.path.append((self.x, self.y, self.theta))
    
    def get_map_as_array(self) -> np.ndarray:
        """
        Convert the occupancy grid to a numpy array for visualization.
        
        Returns:
            2D numpy array with occupancy probabilities
        """
        map_array = np.zeros((self.height, self.width))
        
        for y in range(self.height):
            for x in range(self.width):
                map_array[y, x] = self.grid[y][x].probability
        
        return map_array
    
    def visualize(self, show_path: bool = True):
        """
        Visualize the current state of the SLAM algorithm.
        
        Args:
            show_path: Whether to show the robot's path
        """
        plt.figure(figsize=(10, 8))
        
        # Convert grid to numpy array for visualization
        map_array = self.get_map_as_array()
        
        # Plot the map
        plt.imshow(map_array, cmap='gray_r', origin='lower', vmin=0.0, vmax=1.0)
        
        # Plot the robot position
        plt.scatter([self.x], [self.y], c='r', s=200, marker='o', label='Robot Position')
        
        # Draw direction arrow for robot
        arrow_length = 2.0
        plt.arrow(self.x, self.y, 
                  arrow_length * math.cos(self.theta), 
                  arrow_length * math.sin(self.theta),
                  head_width=1.0, head_length=1.5, fc='r', ec='r')
        
        # Plot the path if requested
        if show_path and len(self.path) > 1:
            path_x = [p[0] for p in self.path]
            path_y = [p[1] for p in self.path]
            plt.plot(path_x, path_y, 'b-', linewidth=1, alpha=0.7, label='Robot Path')
        
        plt.title('Grid-based SLAM')
        plt.legend()
        plt.colorbar(label='Occupancy Probability')
        plt.savefig('grid_slam_visualization.png')
        plt.close()
    
    def run_slam(self, motion_commands, sensor_readings, sensor_angles, max_range, visualize_steps=False):
        """
        Run the complete SLAM algorithm over a sequence of motions and measurements.
        
        Args:
            motion_commands: List of (dx, dy, dtheta) motion commands
            sensor_readings: List of lists containing sensor readings at each step
            sensor_angles: List of angles (in radians) corresponding to each sensor
            max_range: Maximum sensor range
            visualize_steps: Whether to visualize each step
        
        Returns:
            Final occupancy grid as a numpy array
        """
        for step, (motion, readings) in enumerate(zip(motion_commands, sensor_readings)):
            dx, dy, dtheta = motion
            
            # Apply motion model
            self.move_robot(dx, dy, dtheta)
            
            # Update map with measurements
            self.update_map(readings, sensor_angles, max_range)
            
            # Visualize if requested
            if visualize_steps and step % 5 == 0:  # Visualize every 5 steps to reduce output
                self.visualize()
        
        # Final visualization
        self.visualize()
        
        return self.get_map_as_array()

# Simulated environment for testing
class Environment:
    """Simulated environment for testing SLAM."""
    
    def __init__(self, width: int, height: int):
        """
        Initialize a simulated environment.
        
        Args:
            width: Width of the environment in cells
            height: Height of the environment in cells
        """
        self.width = width
        self.height = height
        self.map = np.zeros((height, width))
        self.robot_x = width // 2
        self.robot_y = height // 2
        self.robot_theta = 0.0
    
    def add_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        """
        Add a rectangular obstacle to the environment.
        
        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
        """
        self.map[y1:y2, x1:x2] = 1.0
    
    def add_circle(self, x: int, y: int, radius: int):
        """
        Add a circular obstacle to the environment.
        
        Args:
            x, y: Center coordinates
            radius: Circle radius
        """
        y_indices, x_indices = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
        mask = dist_from_center <= radius
        self.map[mask] = 1.0
    
    def is_occupied(self, x: int, y: int) -> bool:
        """
        Check if a cell is occupied.
        
        Args:
            x, y: Cell coordinates
            
        Returns:
            True if cell is occupied
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.map[y, x] > 0.5
        return True  # Consider out-of-bounds as occupied
    
    def get_sensor_reading(self, angle: float, max_range: float) -> float:
        """
        Get simulated sensor reading.
        
        Args:
            angle: Sensor angle (in global frame)
            max_range: Maximum sensor range
            
        Returns:
            Distance to nearest obstacle, or max_range if none found
        """
        # Calculate ray direction
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Current position
        x = self.robot_x
        y = self.robot_y
        
        # Step size for ray casting
        step_size = 0.1
        
        # Current distance along ray
        distance = 0.0
        
        while distance < max_range:
            # Calculate current position
            curr_x = x + distance * dx
            curr_y = y + distance * dy
            
            # Check if position is occupied
            grid_x = int(curr_x)
            grid_y = int(curr_y)
            
            if self.is_occupied(grid_x, grid_y):
                return distance
            
            distance += step_size
        
        return max_range
    
    def get_sensor_readings(self, sensor_angles: List[float], max_range: float) -> List[float]:
        """
        Get readings from all sensors.
        
        Args:
            sensor_angles: List of sensor angles (relative to robot orientation)
            max_range: Maximum sensor range
            
        Returns:
            List of distance readings
        """
        readings = []
        
        for angle in sensor_angles:
            # Calculate absolute angle
            abs_angle = self.robot_theta + angle
            
            # Get reading and add noise
            reading = self.get_sensor_reading(abs_angle, max_range)
            noisy_reading = reading + random.gauss(0, 0.1)
            readings.append(max(0, noisy_reading))
        
        return readings
    
    def move_robot(self, dx: float, dy: float, dtheta: float) -> bool:
        """
        Move the robot in the environment.
        
        Args:
            dx: Change in x-coordinate (in local robot frame)
            dy: Change in y-coordinate (in local robot frame)
            dtheta: Change in orientation (in radians)
            
        Returns:
            True if move is valid, False if it would collide with an obstacle
        """
        # Calculate new position
        new_x = self.robot_x + dx * math.cos(self.robot_theta) - dy * math.sin(self.robot_theta)
        new_y = self.robot_y + dx * math.sin(self.robot_theta) + dy * math.cos(self.robot_theta)
        
        # Check if new position is valid
        if self.is_occupied(int(new_x), int(new_y)):
            return False
        
        # Update position and orientation
        self.robot_x = new_x
        self.robot_y = new_y
        self.robot_theta += dtheta
        
        # Normalize angle
        self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi
        
        return True
    
    def visualize(self):
        """Visualize the environment."""
        plt.figure(figsize=(10, 8))
        
        # Plot the map
        plt.imshow(self.map, cmap='gray_r', origin='lower')
        
        # Plot the robot position
        plt.scatter([self.robot_x], [self.robot_y], c='r', s=200, marker='o', label='Robot')
        
        # Draw direction arrow for robot
        arrow_length = 2.0
        plt.arrow(self.robot_x, self.robot_y, 
                  arrow_length * math.cos(self.robot_theta), 
                  arrow_length * math.sin(self.robot_theta),
                  head_width=1.0, head_length=1.5, fc='r', ec='r')
        
        plt.title('Simulated Environment')
        plt.legend()
        plt.colorbar(label='Obstacle')
        plt.savefig('environment_visualization.png')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Create a simulated environment
    env = Environment(100, 100)
    
    # Add some obstacles
    env.add_rectangle(20, 20, 25, 80)  # Vertical wall
    env.add_rectangle(20, 20, 80, 25)  # Horizontal wall
    env.add_rectangle(20, 75, 80, 80)  # Horizontal wall
    env.add_rectangle(75, 20, 80, 80)  # Vertical wall
    env.add_circle(50, 50, 10)  # Central obstacle
    
    # Visualize the environment
    env.visualize()
    
    # Define sensor angles (e.g., 8 sensors around the robot)
    sensor_angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
    
    # Maximum sensor range
    max_range = 30.0
    
    # Initialize Grid SLAM
    slam = GridSLAM(grid_size=(100, 100), initial_position=(env.robot_x, env.robot_y, env.robot_theta))
    
    # Get initial sensor readings
    initial_readings = env.get_sensor_readings(sensor_angles, max_range)
    
    # Update map with initial readings
    slam.update_map(initial_readings, sensor_angles, max_range)
    
    # Visualize initial state
    slam.visualize()
    
    # Define a sequence of motions to explore the environment
    motion_commands = []
    num_steps = 100
    
    # Generate a sequence of random motions
    for _ in range(num_steps):
        # Random motion: forward/backward, slight turn
        dx = random.uniform(0.5, 1.5)  # Move forward
        dy = 0.0  # No lateral movement
        dtheta = random.uniform(-0.2, 0.2)  # Small random turn
        
        motion_commands.append((dx, dy, dtheta))
    
    # Run the simulation
    sensor_readings = []
    
    for dx, dy, dtheta in motion_commands:
        # Try to move the robot
        if env.move_robot(dx, dy, dtheta):
            # Get sensor readings at new position
            readings = env.get_sensor_readings(sensor_angles, max_range)
            sensor_readings.append(readings)
        else:
            # If move fails, try turning in place
            env.move_robot(0.0, 0.0, dtheta)
            readings = env.get_sensor_readings(sensor_angles, max_range)
            sensor_readings.append(readings)
            # Update motion command to reflect what actually happened
            motion_commands[-1] = (0.0, 0.0, dtheta)
    
    # Visualize final environment state
    env.visualize()
    
    # Run the SLAM algorithm
    start_time = time.time()
    final_map = slam.run_slam(motion_commands, sensor_readings, sensor_angles, max_range, visualize_steps=True)
    end_time = time.time()
    
    print(f"SLAM completed in {end_time - start_time:.2f} seconds")
    
    # Compare maps
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(env.map, cmap='gray_r', origin='lower')
    plt.title('True Environment Map')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(final_map, cmap='gray_r', origin='lower')
    plt.title('SLAM Generated Map')
    plt.colorbar()
    
    plt.savefig('map_comparison.png')
    plt.close()
    
    print("Simulation complete.")
