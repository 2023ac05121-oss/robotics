#!/usr/bin/env python3
"""
Integration script to demonstrate the combined use of all robotics components.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import os
import sys
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from .localization.monte_carlo_localization import MonteCarloLocalization
from .mapping.grid_slam import GridSLAM, Environment
from .planning.dstar_planner import DStar
from .deep_learning.deep_mcl import DeepMCL
from .utils.robot_utils import (
    create_random_map, add_border_to_map, visualize_map,
    calculate_path_length, generate_noisy_odometry, 
    generate_noisy_sensor_readings, simulate_range_sensor
)

class IntegratedRobot:
    """
    Integrated robot that combines localization, mapping, and planning.
    """
    
    def __init__(self, map_size: Tuple[int, int] = (100, 100), 
                 use_deep_learning: bool = True,
                 num_particles: int = 1000,
                 grid_size: float = 1.0,
                 num_sensors: int = 8,
                 max_sensor_range: float = 10.0,
                 sensor_range: float = 20.0,
                 motion_noise: Tuple[float, float, float] = (0.1, 0.1, 0.05),
                 measurement_noise: float = 0.1,
                 model_path: Optional[str] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize the integrated robot.
        
        Args:
            map_size: Tuple of (width, height) for the map
            use_deep_learning: Whether to use deep learning enhanced localization
            num_particles: Number of particles for MCL
            grid_size: Size of each grid cell for SLAM
            num_sensors: Number of sensors
            max_sensor_range: Maximum sensor range
            sensor_range: Range for the sensor (used in test scripts)
            motion_noise: Tuple of (x, y, theta) standard deviations for motion model
            measurement_noise: Standard deviation for measurement model
            model_path: Path to the deep learning model
            random_seed: Optional random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        self.width, self.height = map_size
        self.use_deep_learning = use_deep_learning
        self.num_particles = num_particles
        self.grid_size = grid_size
        self.num_sensors = num_sensors
        self.max_sensor_range = max_sensor_range
        self.sensor_range = sensor_range
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.model_path = model_path
        
        # Initialize empty map
        self.true_map = None
        self.estimated_map = None
        
        # Robot pose (x, y, theta)
        self.true_pose = None
        self.estimated_pose = None
        
        # Sensor angles
        self.sensor_angles = [i * 2 * math.pi / num_sensors for i in range(num_sensors)]
        
        # Path planning
        self.planned_path = []
        self.executed_path = []
        
        # Components
        self.mcl = None
        self.slam = None
        self.planner = None
        
        # Visualization directory
        self.vis_dir = "visualization"
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def initialize(self, true_map: Optional[np.ndarray] = None,
                  start_pos: Optional[Tuple[int, int]] = None,
                  goal_pos: Optional[Tuple[int, int]] = None):
        """
        Initialize the robot with a map and position.
        
        Args:
            true_map: Optional true map (if None, a random map will be created)
            start_pos: Optional start position (if None, a random position will be chosen)
            goal_pos: Optional goal position (if None, a random position will be chosen)
        """
        # Create or use provided map
        if true_map is None:
            self.true_map = create_random_map(self.width, self.height, obstacle_prob=0.2)
            self.true_map = add_border_to_map(self.true_map)
        else:
            self.true_map = true_map
        
        # Initialize estimated map (for SLAM)
        self.estimated_map = np.zeros_like(self.true_map)
        
        # Find start position if not provided
        if start_pos is None:
            free_cells = np.argwhere(self.true_map == 0)
            if len(free_cells) == 0:
                raise ValueError("No free space found in the map")
            
            idx = random.randint(0, len(free_cells) - 1)
            start_y, start_x = free_cells[idx]
            start_pos = (start_x, start_y)
        
        # Initial orientation
        start_theta = random.uniform(0, 2 * math.pi)
        
        # Set true pose
        self.true_pose = (start_pos[0], start_pos[1], start_theta)
        self.executed_path = [self.true_pose]
        
        # Find goal position if not provided
        if goal_pos is None:
            while True:
                free_cells = np.argwhere(self.true_map == 0)
                idx = random.randint(0, len(free_cells) - 1)
                goal_y, goal_x = free_cells[idx]
                goal_pos = (goal_x, goal_y)
                
                # Ensure goal is not too close to start
                dist = math.sqrt((goal_pos[0] - start_pos[0])**2 + (goal_pos[1] - start_pos[1])**2)
                if dist > min(self.width, self.height) / 3:  # At least 1/3 of map size away
                    break
        
        self.goal_pos = goal_pos
        
        # Initialize localization
        if self.use_deep_learning:
            self.mcl = DeepMCL(
                self.true_map, 
                num_particles=self.num_particles,
                motion_noise=self.motion_noise,
                measurement_noise=self.measurement_noise
            )
            
            # Train the sensor model
            self.mcl.train_sensor_model(
                num_samples=5000,
                num_sensors=self.num_sensors,
                max_range=self.max_sensor_range,
                noise_level=0.2,
                batch_size=64,
                num_epochs=5
            )
        else:
            self.mcl = MonteCarloLocalization(
                self.true_map, 
                num_particles=self.num_particles,
                motion_noise=self.motion_noise,
                measurement_noise=self.measurement_noise
            )
        
        # Initialize SLAM
        self.slam = GridSLAM(
            grid_size=(self.width, self.height),
            initial_position=self.true_pose,
            motion_noise=self.motion_noise,
            measurement_noise=self.measurement_noise
        )
        
        # Initialize planner with true map (in real world, would use SLAM-generated map)
        self.planner = DStar(self.true_map)
        
        # Initial sensor readings
        initial_readings = simulate_range_sensor(
            self.true_map, 
            self.true_pose, 
            self.sensor_angles, 
            self.max_sensor_range,
            self.measurement_noise
        )
        
        # Update localization with initial readings
        self.mcl.measurement_update(initial_readings, self.sensor_angles, self.max_sensor_range)
        
        # Get initial position estimate
        self.estimated_pose = self.mcl.get_position_estimate()
        
        # Update SLAM with initial readings
        self.slam.update_map(initial_readings, self.sensor_angles, self.max_sensor_range)
        
        # Plan initial path
        self.planner.compute_shortest_path(
            int(self.true_pose[0]), int(self.true_pose[1]),
            self.goal_pos[0], self.goal_pos[1]
        )
        
        self.planned_path = self.planner.get_path(
            int(self.true_pose[0]), int(self.true_pose[1]),
            self.goal_pos[0], self.goal_pos[1]
        )
        
        # Visualize initial state
        self.visualize(0)
    
    def step(self, step_idx: int):
        """
        Execute one step of the robot's operation.
        
        Args:
            step_idx: Current step index
        
        Returns:
            True if goal reached, False otherwise
        """
        # Check if goal reached
        if self.true_pose[0] == self.goal_pos[0] and self.true_pose[1] == self.goal_pos[1]:
            print("Goal reached!")
            return True
        
        # Check if we have a valid path
        if not self.planned_path or len(self.planned_path) < 2:
            print("No valid path to goal!")
            return True
        
        # Get next waypoint from path
        next_waypoint = self.planned_path[1]  # Skip current position
        
        # Calculate motion command to reach the waypoint
        dx = next_waypoint[0] - self.true_pose[0]
        dy = next_waypoint[1] - self.true_pose[1]
        
        # Calculate angle to waypoint
        target_angle = math.atan2(dy, dx)
        
        # Calculate angular difference
        dtheta = target_angle - self.true_pose[2]
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]
        
        # Define maximum step size
        max_step = 1.0
        
        # Limit motion to max_step
        distance = math.sqrt(dx**2 + dy**2)
        if distance > max_step:
            dx = max_step * math.cos(target_angle)
            dy = max_step * math.sin(target_angle)
        
        # Prepare motion command
        motion_command = (dx, dy, dtheta)
        
        # Generate noisy odometry
        noisy_motion = generate_noisy_odometry(motion_command, noise_level=0.1)
        
        # Move the robot (true motion)
        new_x = self.true_pose[0] + dx
        new_y = self.true_pose[1] + dy
        new_theta = (self.true_pose[2] + dtheta) % (2 * math.pi)
        
        # Check if new position is valid (not in obstacle)
        map_x, map_y = int(new_x), int(new_y)
        if (0 <= map_x < self.width and 0 <= map_y < self.height and 
            self.true_map[map_y, map_x] == 0):
            self.true_pose = (new_x, new_y, new_theta)
        else:
            # If invalid, just turn in place
            self.true_pose = (self.true_pose[0], self.true_pose[1], new_theta)
            motion_command = (0, 0, dtheta)
            noisy_motion = (0, 0, dtheta + random.gauss(0, 0.05))
        
        # Record executed path
        self.executed_path.append(self.true_pose)
        
        # Get sensor readings
        sensor_readings = simulate_range_sensor(
            self.true_map, 
            self.true_pose, 
            self.sensor_angles, 
            self.max_sensor_range,
            self.measurement_noise
        )
        
        # Update localization with noisy motion and sensor readings
        self.mcl.motion_update(*noisy_motion)
        self.mcl.measurement_update(sensor_readings, self.sensor_angles, self.max_sensor_range)
        self.mcl.resample()
        
        # Update position estimate
        self.estimated_pose = self.mcl.get_position_estimate()
        
        # Update SLAM
        self.slam.move_robot(*noisy_motion)
        self.slam.update_map(sensor_readings, self.sensor_angles, self.max_sensor_range)
        
        # Get updated map from SLAM
        self.estimated_map = self.slam.get_map_as_array()
        
        # Check if replanning is needed
        # 1. If we've reached the next waypoint
        # 2. If we've discovered new obstacles
        reached_waypoint = (abs(self.true_pose[0] - next_waypoint[0]) < 0.5 and 
                           abs(self.true_pose[1] - next_waypoint[1]) < 0.5)
        
        # Use estimated position for replanning
        est_x, est_y = int(self.estimated_pose[0]), int(self.estimated_pose[1])
        
        if reached_waypoint or step_idx % 10 == 0:  # Replan periodically
            # Use SLAM-generated map for planning in a real robot
            # For simplicity, we'll use the true map here
            self.planner = DStar(self.true_map)
            
            # Recompute path from current estimated position
            self.planner.compute_shortest_path(
                est_x, est_y,
                self.goal_pos[0], self.goal_pos[1]
            )
            
            self.planned_path = self.planner.get_path(
                est_x, est_y,
                self.goal_pos[0], self.goal_pos[1]
            )
        
        # Visualize current state
        self.visualize(step_idx + 1)
        
        # Check if goal reached
        goal_reached = (abs(self.true_pose[0] - self.goal_pos[0]) < 1.0 and 
                        abs(self.true_pose[1] - self.goal_pos[1]) < 1.0)
        
        return goal_reached
    
    def run_simulation(self, max_steps: int = 100):
        """
        Run the complete simulation.
        
        Args:
            max_steps: Maximum number of steps to run
        """
        for step_idx in range(max_steps):
            goal_reached = self.step(step_idx)
            
            if goal_reached:
                print(f"Simulation completed in {step_idx + 1} steps")
                break
        
        self.finalize()
    
    def finalize(self):
        """Finalize the simulation and show results."""
        # Final visualization
        self.visualize(final=True)
        
        # Calculate path length
        true_path_length = calculate_path_length([(p[0], p[1]) for p in self.executed_path])
        
        # Calculate localization error
        position_errors = []
        for step, true_pos in enumerate(self.executed_path):
            if step >= len(self.mcl.path):
                break
            
            est_pos = self.mcl.path[step]
            error = math.sqrt((true_pos[0] - est_pos[0])**2 + (true_pos[1] - est_pos[1])**2)
            position_errors.append(error)
        
        avg_error = sum(position_errors) / len(position_errors) if position_errors else 0
        
        # Print results
        print(f"Simulation Results:")
        print(f"- Total path length: {true_path_length:.2f}")
        print(f"- Average localization error: {avg_error:.2f}")
        print(f"- Final position: {self.true_pose}")
        print(f"- Final estimated position: {self.estimated_pose}")
        
        # Plot error over time
        plt.figure(figsize=(10, 6))
        plt.plot(position_errors)
        plt.xlabel('Step')
        plt.ylabel('Position Error')
        plt.title('Localization Error Over Time')
        plt.grid(True)
        plt.savefig(os.path.join(self.vis_dir, 'localization_error.png'))
        plt.close()
    
    def visualize(self, step_idx: int = 0, final: bool = False):
        """
        Visualize the current state of the robot.
        
        Args:
            step_idx: Current step index
            final: Whether this is the final visualization
        """
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot true map and robot position
        axs[0, 0].imshow(self.true_map, cmap='gray_r', origin='lower')
        axs[0, 0].scatter([self.true_pose[0]], [self.true_pose[1]], c='r', s=200, marker='o')
        axs[0, 0].scatter([self.goal_pos[0]], [self.goal_pos[1]], c='g', s=200, marker='*')
        
        # Draw true path
        path_x = [p[0] for p in self.executed_path]
        path_y = [p[1] for p in self.executed_path]
        axs[0, 0].plot(path_x, path_y, 'r-', linewidth=2)
        
        # Draw direction arrow for robot
        arrow_length = 2.0
        axs[0, 0].arrow(self.true_pose[0], self.true_pose[1], 
                  arrow_length * math.cos(self.true_pose[2]), 
                  arrow_length * math.sin(self.true_pose[2]),
                  head_width=1.0, head_length=1.5, fc='r', ec='r')
        
        axs[0, 0].set_title('True Map and Path')
        
        # Plot estimated map from SLAM
        axs[0, 1].imshow(self.estimated_map, cmap='gray_r', origin='lower', vmin=0.0, vmax=1.0)
        axs[0, 1].scatter([self.estimated_pose[0]], [self.estimated_pose[1]], c='b', s=200, marker='o')
        axs[0, 1].scatter([self.goal_pos[0]], [self.goal_pos[1]], c='g', s=200, marker='*')
        
        # Draw direction arrow for estimated position
        axs[0, 1].arrow(self.estimated_pose[0], self.estimated_pose[1], 
                  arrow_length * math.cos(self.estimated_pose[2]), 
                  arrow_length * math.sin(self.estimated_pose[2]),
                  head_width=1.0, head_length=1.5, fc='b', ec='b')
        
        axs[0, 1].set_title('SLAM Map and Estimated Position')
        
        # Plot particles
        if self.mcl:
            particle_xs = [p.x for p in self.mcl.particles]
            particle_ys = [p.y for p in self.mcl.particles]
            particle_weights = [p.weight * 1000 for p in self.mcl.particles]
            
            axs[1, 0].imshow(self.true_map, cmap='gray_r', origin='lower')
            axs[1, 0].scatter(particle_xs, particle_ys, c='b', alpha=0.3, s=particle_weights)
            axs[1, 0].scatter([self.estimated_pose[0]], [self.estimated_pose[1]], c='b', s=200, marker='o')
            axs[1, 0].scatter([self.true_pose[0]], [self.true_pose[1]], c='r', s=200, marker='x')
            axs[1, 0].set_title('Particle Distribution')
        
        # Plot planned path
        axs[1, 1].imshow(self.true_map, cmap='gray_r', origin='lower')
        axs[1, 1].scatter([self.true_pose[0]], [self.true_pose[1]], c='r', s=200, marker='o')
        axs[1, 1].scatter([self.goal_pos[0]], [self.goal_pos[1]], c='g', s=200, marker='*')
        
        if self.planned_path:
            planned_x = [p[0] for p in self.planned_path]
            planned_y = [p[1] for p in self.planned_path]
            axs[1, 1].plot(planned_x, planned_y, 'b-', linewidth=2)
        
        axs[1, 1].set_title('Planned Path')
        
        # Add step number to title
        if final:
            fig.suptitle(f'Final State', fontsize=16)
        else:
            fig.suptitle(f'Step {step_idx}', fontsize=16)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f'step_{step_idx:03d}.png'))
        plt.close()

# Example usage
if __name__ == "__main__":
    # Create a robot
    robot = IntegratedRobot(
        map_size=(50, 50),
        use_deep_learning=True,
        num_particles=1000,
        num_sensors=8,
        max_sensor_range=10.0,
        random_seed=42
    )
    
    # Initialize
    print("Initializing robot...")
    robot.initialize()
    
    # Run simulation
    print("Running simulation...")
    robot.run_simulation(max_steps=100)
    
    print("Simulation complete!")
