#!/usr/bin/env python3
"""
Monte Carlo Localization (MCL) implementation.
This algorithm uses particle filters to estimate the robot's position in a known map.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import random
from typing import List, Tuple, Optional

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

class MonteCarloLocalization:
    """Implementation of Monte Carlo Localization algorithm."""
    
    def __init__(self, map_data: np.ndarray, num_particles: int = 1000, 
                 motion_noise: Tuple[float, float, float] = (0.1, 0.1, 0.05),
                 measurement_noise: float = 0.1):
        """
        Initialize the MCL algorithm.
        
        Args:
            map_data: 2D numpy array representing the environment (1 = obstacle, 0 = free space)
            num_particles: Number of particles to use
            motion_noise: Tuple of (x, y, theta) standard deviations for motion model
            measurement_noise: Standard deviation for measurement model
        """
        self.map = map_data
        self.height, self.width = map_data.shape
        self.num_particles = num_particles
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.particles = []
        self.initialize_particles()
    
    def initialize_particles(self):
        """Initialize particles randomly across the free space in the map."""
        self.particles = []
        
        # Find all free space cells in the map
        free_cells = np.argwhere(self.map == 0)
        
        if len(free_cells) == 0:
            raise ValueError("No free space found in the map")
        
        # Generate random particles
        for _ in range(self.num_particles):
            # Randomly choose a free cell
            idx = random.randint(0, len(free_cells) - 1)
            y, x = free_cells[idx]
            
            # Add some noise to the position to distribute particles
            x += random.uniform(-0.45, 0.45)
            y += random.uniform(-0.45, 0.45)
            
            # Random orientation
            theta = random.uniform(0, 2 * math.pi)
            
            self.particles.append(Particle(x, y, theta))
    
    def motion_update(self, dx: float, dy: float, dtheta: float):
        """
        Update particles based on robot motion.
        
        Args:
            dx: Change in x-coordinate
            dy: Change in y-coordinate
            dtheta: Change in orientation (in radians)
        """
        for i, p in enumerate(self.particles):
            # Add noise to motion
            noisy_dx = dx + random.gauss(0, self.motion_noise[0])
            noisy_dy = dy + random.gauss(0, self.motion_noise[1])
            noisy_dtheta = dtheta + random.gauss(0, self.motion_noise[2])
            
            # Update particle position based on motion model
            p.x += noisy_dx * math.cos(p.theta) - noisy_dy * math.sin(p.theta)
            p.y += noisy_dx * math.sin(p.theta) + noisy_dy * math.cos(p.theta)
            p.theta += noisy_dtheta
            
            # Normalize angle
            p.theta = (p.theta + math.pi) % (2 * math.pi) - math.pi
            
            # Check if the particle is still in free space
            map_x, map_y = int(p.x), int(p.y)
            
            # If particle goes outside the map or into an obstacle, give it a very low weight
            if (map_x < 0 or map_x >= self.width or 
                map_y < 0 or map_y >= self.height or 
                self.map[map_y, map_x] == 1):
                p.weight = 1e-10
            
            self.particles[i] = p
    
    def measurement_update(self, measurements: List[float], sensor_angles: List[float], max_range: float):
        """
        Update particle weights based on sensor measurements.
        
        Args:
            measurements: List of distance measurements from sensors
            sensor_angles: List of angles (in radians) corresponding to each measurement
            max_range: Maximum range of the sensors
        """
        if len(measurements) != len(sensor_angles):
            raise ValueError("Number of measurements must match number of sensor angles")
        
        # Update weights for each particle
        for i, p in enumerate(self.particles):
            # Predict what this particle should measure
            predicted_measurements = self.get_expected_measurements(p, sensor_angles, max_range)
            
            # Calculate likelihood of measurements
            likelihood = 1.0
            for actual, predicted in zip(measurements, predicted_measurements):
                # Calculate the probability of getting the actual measurement given the predicted one
                # Using a Gaussian model for the sensor noise
                error = actual - predicted
                likelihood *= math.exp(-(error ** 2) / (2 * self.measurement_noise ** 2))
            
            # Update particle weight
            self.particles[i].weight *= likelihood
        
        # Normalize weights
        self.normalize_weights()
    
    def get_expected_measurements(self, particle: Particle, sensor_angles: List[float], max_range: float) -> List[float]:
        """
        Calculate expected measurements for a particle.
        
        Args:
            particle: The particle to calculate measurements for
            sensor_angles: List of angles (in radians) corresponding to each measurement
            max_range: Maximum range of the sensors
            
        Returns:
            List of expected distance measurements
        """
        expected_measurements = []
        
        for angle in sensor_angles:
            # Calculate absolute angle for this measurement
            abs_angle = particle.theta + angle
            
            # Cast a ray from the particle position
            distance = self.cast_ray(particle.x, particle.y, abs_angle, max_range)
            expected_measurements.append(distance)
        
        return expected_measurements
    
    def cast_ray(self, x: float, y: float, angle: float, max_range: float) -> float:
        """
        Cast a ray from (x, y) in the direction of angle and return the distance to the nearest obstacle.
        
        Args:
            x: Starting x-coordinate
            y: Starting y-coordinate
            angle: Direction angle (in radians)
            max_range: Maximum range to check
            
        Returns:
            Distance to the nearest obstacle, or max_range if no obstacle is found
        """
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
            if (map_x < 0 or map_x >= self.width or 
                map_y < 0 or map_y >= self.height or 
                self.map[map_y, map_x] == 1):
                return distance
            
            # Increment distance
            distance += step_size
        
        # If no obstacle found within max_range, return max_range
        return max_range
    
    def normalize_weights(self):
        """Normalize particle weights so they sum to 1."""
        total_weight = sum(p.weight for p in self.particles)
        
        # Avoid division by zero
        if total_weight > 0:
            for i in range(len(self.particles)):
                self.particles[i].weight /= total_weight
        else:
            # If all weights are zero, reinitialize with equal weights
            for i in range(len(self.particles)):
                self.particles[i].weight = 1.0 / self.num_particles
    
    def resample(self):
        """Resample particles based on their weights."""
        # Create cumulative weights array
        cumulative_weights = np.zeros(self.num_particles)
        cumulative_sum = 0
        
        for i, p in enumerate(self.particles):
            cumulative_sum += p.weight
            cumulative_weights[i] = cumulative_sum
        
        # Generate random starting point and step size
        step = 1.0 / self.num_particles
        start = random.uniform(0, step)
        
        # Create new particle set
        new_particles = []
        
        # Low variance resampling
        current_idx = 0
        for i in range(self.num_particles):
            # Find the particle corresponding to this weight
            u = start + i * step
            
            # Move up the cumulative weights until we find the right particle
            while u > cumulative_weights[current_idx]:
                current_idx += 1
                if current_idx >= self.num_particles:
                    current_idx = self.num_particles - 1
                    break
            
            # Copy the selected particle (with some noise to avoid sample impoverishment)
            p = self.particles[current_idx]
            new_p = Particle(
                p.x + random.gauss(0, self.motion_noise[0] * 0.1),
                p.y + random.gauss(0, self.motion_noise[1] * 0.1),
                p.theta + random.gauss(0, self.motion_noise[2] * 0.1),
                1.0 / self.num_particles  # Reset weights
            )
            
            new_particles.append(new_p)
        
        self.particles = new_particles
    
    def get_position_estimate(self) -> Tuple[float, float, float]:
        """
        Calculate estimated position as the weighted average of all particles.
        
        Returns:
            Tuple of (x, y, theta) representing the estimated position
        """
        if not self.particles:
            return (0, 0, 0)
        
        x_sum = 0
        y_sum = 0
        cos_sum = 0
        sin_sum = 0
        
        for p in self.particles:
            x_sum += p.x * p.weight
            y_sum += p.y * p.weight
            cos_sum += math.cos(p.theta) * p.weight
            sin_sum += math.sin(p.theta) * p.weight
        
        # Calculate average orientation using atan2 to handle angle wrapping
        avg_theta = math.atan2(sin_sum, cos_sum)
        
        return (x_sum, y_sum, avg_theta)
    
    def visualize(self, robot_pose: Optional[Tuple[float, float, float]] = None, show_particles: bool = True):
        """
        Visualize the current state of the MCL algorithm.
        
        Args:
            robot_pose: Optional tuple of (x, y, theta) representing the true robot pose
            show_particles: Whether to show all particles or just the position estimate
        """
        plt.figure(figsize=(10, 8))
        
        # Plot the map
        plt.imshow(self.map, cmap='gray_r', origin='lower')
        
        if show_particles:
            # Plot all particles
            particle_xs = [p.x for p in self.particles]
            particle_ys = [p.y for p in self.particles]
            particle_weights = [p.weight * 500 for p in self.particles]  # Scale weights for visibility
            
            plt.scatter(particle_xs, particle_ys, c='b', alpha=0.3, s=particle_weights)
        
        # Plot the estimated position
        est_x, est_y, est_theta = self.get_position_estimate()
        plt.scatter([est_x], [est_y], c='g', s=200, marker='*', label='Estimated Position')
        
        # Draw direction line for estimated position
        arrow_length = 1.0
        plt.arrow(est_x, est_y, 
                  arrow_length * math.cos(est_theta), 
                  arrow_length * math.sin(est_theta),
                  head_width=0.3, head_length=0.5, fc='g', ec='g')
        
        # Plot true robot position if provided
        if robot_pose:
            true_x, true_y, true_theta = robot_pose
            plt.scatter([true_x], [true_y], c='r', s=200, marker='o', label='True Position')
            
            # Draw direction line for true position
            plt.arrow(true_x, true_y, 
                      arrow_length * math.cos(true_theta), 
                      arrow_length * math.sin(true_theta),
                      head_width=0.3, head_length=0.5, fc='r', ec='r')
        
        plt.title('Monte Carlo Localization')
        plt.legend()
        plt.grid(True)
        plt.colorbar(label='Obstacle Probability')
        plt.savefig('mcl_visualization.png')
        plt.close()
    
    def run_localization(self, motion_commands, sensor_readings, sensor_angles, max_range, 
                        true_poses=None, visualize_steps=False):
        """
        Run the complete localization algorithm over a sequence of motions and measurements.
        
        Args:
            motion_commands: List of (dx, dy, dtheta) motion commands
            sensor_readings: List of lists containing sensor readings at each step
            sensor_angles: List of angles (in radians) corresponding to each sensor
            max_range: Maximum sensor range
            true_poses: Optional list of true robot poses for visualization
            visualize_steps: Whether to visualize each step
        
        Returns:
            List of estimated poses at each step
        """
        estimated_poses = []
        
        for step, (motion, readings) in enumerate(zip(motion_commands, sensor_readings)):
            dx, dy, dtheta = motion
            
            # Apply motion model
            self.motion_update(dx, dy, dtheta)
            
            # Apply measurement model
            self.measurement_update(readings, sensor_angles, max_range)
            
            # Resample particles
            self.resample()
            
            # Get current position estimate
            estimated_pose = self.get_position_estimate()
            estimated_poses.append(estimated_pose)
            
            # Visualize if requested
            if visualize_steps:
                true_pose = true_poses[step] if true_poses and step < len(true_poses) else None
                self.visualize(true_pose)
        
        return estimated_poses

# Example usage
if __name__ == "__main__":
    # Create a simple map (20x20 grid with some obstacles)
    map_data = np.zeros((20, 20))
    # Add some obstacles
    map_data[5:15, 5] = 1  # Vertical wall
    map_data[5, 5:15] = 1  # Horizontal wall
    map_data[15, 5:15] = 1  # Horizontal wall
    map_data[5:15, 15] = 1  # Vertical wall
    
    # Initialize MCL with 1000 particles
    mcl = MonteCarloLocalization(map_data, num_particles=1000)
    
    # Define sensor angles (e.g., 8 sensors around the robot)
    sensor_angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
    
    # Maximum sensor range
    max_range = 10.0
    
    # True robot pose (for simulation)
    true_pose = (10.0, 10.0, 0.0)  # (x, y, theta)
    
    # Generate simulated sensor readings
    readings = []
    for angle in sensor_angles:
        # Calculate absolute angle
        abs_angle = true_pose[2] + angle
        
        # Cast a ray and add some noise
        distance = mcl.cast_ray(true_pose[0], true_pose[1], abs_angle, max_range)
        noisy_distance = distance + random.gauss(0, mcl.measurement_noise)
        readings.append(max(0, noisy_distance))  # Ensure non-negative distance
    
    # Update MCL with the measurements
    mcl.measurement_update(readings, sensor_angles, max_range)
    
    # Visualize
    mcl.visualize(true_pose)
    
    print(f"True position: {true_pose}")
    print(f"Estimated position: {mcl.get_position_estimate()}")
    
    # Example of running a sequence of steps
    motion_commands = [
        (0.5, 0.0, 0.0),   # Move forward 0.5 units
        (0.0, 0.0, math.pi/4),  # Turn 45 degrees
        (0.5, 0.0, 0.0),   # Move forward 0.5 units
    ]
    
    # Generate simulated sensor readings for each step
    simulated_readings = []
    true_poses = [true_pose]
    
    for dx, dy, dtheta in motion_commands:
        # Update true pose
        x, y, theta = true_poses[-1]
        new_x = x + dx * math.cos(theta) - dy * math.sin(theta)
        new_y = y + dx * math.sin(theta) + dy * math.cos(theta)
        new_theta = (theta + dtheta) % (2 * math.pi)
        true_poses.append((new_x, new_y, new_theta))
        
        # Generate readings for this pose
        step_readings = []
        for angle in sensor_angles:
            abs_angle = new_theta + angle
            distance = mcl.cast_ray(new_x, new_y, abs_angle, max_range)
            noisy_distance = distance + random.gauss(0, mcl.measurement_noise)
            step_readings.append(max(0, noisy_distance))
        
        simulated_readings.append(step_readings)
    
    # Run the full localization algorithm
    estimated_poses = mcl.run_localization(
        motion_commands, simulated_readings, sensor_angles, max_range, 
        true_poses=true_poses, visualize_steps=True
    )
    
    print("Motion sequence complete.")
    print(f"Final true position: {true_poses[-1]}")
    print(f"Final estimated position: {estimated_poses[-1]}")
