#!/usr/bin/env python3

"""
Deep Learning Enhanced MCL Test Script

This script demonstrates the deep learning enhanced Monte Carlo Localization implementation
by running a simulation with a robot in a known environment and comparing the results
with the standard MCL implementation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from matplotlib.animation import FuncAnimation

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the MCL implementation
from src.localization.monte_carlo_localization import MonteCarloLocalization as StandardMCL
from src.deep_learning.deep_mcl import DeepMCL
from src.utils.robot_utils import create_artificial_map, simulate_robot_motion, simulate_sensor_readings

class DeepMCLTest:
    def __init__(self, map_size=(100, 100), num_particles=500, motion_noise=(0.1, 0.1, 0.05), 
                 measurement_noise=0.1, max_steps=100, obstacle_count=10, use_trained_model=False,
                 model_path=None, train_epochs=50):
        self.map_size = map_size
        self.num_particles = num_particles
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.max_steps = max_steps
        self.obstacle_count = obstacle_count
        self.use_trained_model = use_trained_model
        self.model_path = model_path
        self.train_epochs = train_epochs
        
        # Create artificial map with obstacles
        self.map_data, self.obstacles = create_artificial_map(map_size, obstacle_count)
        
        # Initialize Standard Monte Carlo Localization
        self.standard_mcl = StandardMCL(num_particles=num_particles, 
                                       map_data=self.map_data,
                                       motion_noise=motion_noise,
                                       measurement_noise=measurement_noise)
        
        # Initialize Deep Learning Enhanced Monte Carlo Localization
        self.deep_mcl = DeepMCL(num_particles=num_particles, 
                               map_data=self.map_data,
                               motion_noise=motion_noise,
                               measurement_noise=measurement_noise,
                               use_trained_model=use_trained_model,
                               model_path=model_path)
        
        # Robot's true position (x, y, theta)
        valid_position = False
        while not valid_position:
            self.true_position = np.array([
                np.random.uniform(0, map_size[0]),
                np.random.uniform(0, map_size[1]),
                np.random.uniform(-np.pi, np.pi)
            ])
            # Check if position is valid (not inside an obstacle)
            valid_position = self.is_valid_position(self.true_position[:2])
        
        # Storage for trajectory and particle data
        self.trajectory = [self.true_position.copy()]
        self.standard_particle_history = [self.standard_mcl.particles.copy()]
        self.deep_particle_history = [self.deep_mcl.particles.copy()]
        
        # Error tracking
        self.standard_errors = []
        self.deep_errors = []
        
        # Set up visualization
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.canvas.set_window_title('Deep Learning MCL Comparison')
    
    def is_valid_position(self, position):
        """Check if a position is valid (not inside an obstacle)"""
        x, y = position
        
        # Check if position is within map bounds
        if x < 0 or x >= self.map_size[0] or y < 0 or y >= self.map_size[1]:
            return False
        
        # Check if position is inside any obstacle
        for obstacle in self.obstacles:
            ox, oy, radius = obstacle
            distance = np.sqrt((x - ox)**2 + (y - oy)**2)
            if distance < radius:
                return False
        
        return True
    
    def train_model(self, num_samples=1000):
        """Train the deep learning model with simulated data"""
        print("Training deep learning model...")
        
        # Generate training data
        train_inputs = []
        train_targets = []
        
        for _ in range(num_samples):
            # Generate random position
            valid_position = False
            while not valid_position:
                position = np.array([
                    np.random.uniform(0, self.map_size[0]),
                    np.random.uniform(0, self.map_size[1]),
                    np.random.uniform(-np.pi, np.pi)
                ])
                valid_position = self.is_valid_position(position[:2])
            
            # Simulate sensor readings (raw)
            raw_measurements = simulate_sensor_readings(position, self.map_data, 
                                                      self.obstacles, noise=0.0)  # No noise for ground truth
            
            # Simulate noisy sensor readings
            noisy_measurements = simulate_sensor_readings(position, self.map_data, 
                                                         self.obstacles, noise=self.measurement_noise)
            
            train_inputs.append(noisy_measurements)
            train_targets.append(raw_measurements)
        
        # Convert to PyTorch tensors
        inputs = torch.tensor(train_inputs, dtype=torch.float32)
        targets = torch.tensor(train_targets, dtype=torch.float32)
        
        # Train the model
        self.deep_mcl.train_model(inputs, targets, epochs=self.train_epochs)
        
        # Save the model if path is provided
        if self.model_path:
            self.deep_mcl.save_model(self.model_path)
            print(f"Model saved to {self.model_path}")
    
    def run_simulation(self):
        """Run the MCL simulation for the specified number of steps"""
        # First train the model if needed
        if not self.use_trained_model:
            self.train_model()
        
        # Define a set of waypoints for the robot to follow
        waypoints = []
        for _ in range(5):  # Generate 5 random waypoints
            valid_waypoint = False
            while not valid_waypoint:
                waypoint = np.array([
                    np.random.uniform(0, self.map_size[0]),
                    np.random.uniform(0, self.map_size[1])
                ])
                valid_waypoint = self.is_valid_position(waypoint)
            waypoints.append(waypoint)
        
        current_waypoint_idx = 0
        steps_without_progress = 0
        
        for step in range(self.max_steps):
            # Get the current waypoint
            target = waypoints[current_waypoint_idx]
            
            # Calculate direction to waypoint
            dx = target[0] - self.true_position[0]
            dy = target[1] - self.true_position[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # If we're close to the waypoint, move to the next one
            if distance < 2.0:
                current_waypoint_idx = (current_waypoint_idx + 1) % len(waypoints)
                steps_without_progress = 0
                continue
            
            # Calculate desired heading
            desired_theta = np.arctan2(dy, dx)
            
            # Calculate difference in heading
            dtheta = desired_theta - self.true_position[2]
            # Normalize to [-pi, pi]
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            
            # If heading is off by more than 0.1 radians, rotate first
            if abs(dtheta) > 0.1:
                # Rotate the robot
                rotation = 0.1 * np.sign(dtheta)
                control = np.array([0.0, 0.0, rotation])
            else:
                # Move forward
                speed = min(1.0, distance / 5.0)  # Slow down when close to waypoint
                control = np.array([speed, 0.0, 0.0])
            
            # Simulate robot motion with the control input
            new_position = simulate_robot_motion(self.true_position, control, noise=self.motion_noise)
            
            # Check if the new position is valid
            if self.is_valid_position(new_position[:2]):
                self.true_position = new_position
                steps_without_progress = 0
            else:
                # If not valid, try a random direction
                steps_without_progress += 1
                if steps_without_progress > 10:
                    # If we're stuck for too long, pick a new random waypoint
                    current_waypoint_idx = np.random.randint(0, len(waypoints))
                    steps_without_progress = 0
                continue
            
            # Simulate sensor readings
            measurements = simulate_sensor_readings(self.true_position, self.map_data, 
                                                  self.obstacles, noise=self.measurement_noise)
            
            # Update both MCL implementations with control and measurements
            self.standard_mcl.update(control, measurements)
            self.deep_mcl.update(control, measurements)
            
            # Store trajectory and particles
            self.trajectory.append(self.true_position.copy())
            self.standard_particle_history.append(self.standard_mcl.particles.copy())
            self.deep_particle_history.append(self.deep_mcl.particles.copy())
            
            # Get the estimated positions
            standard_pos = self.standard_mcl.get_estimated_position()
            deep_pos = self.deep_mcl.get_estimated_position()
            
            # Calculate errors
            standard_error = np.sqrt((standard_pos[0] - self.true_position[0])**2 + 
                                    (standard_pos[1] - self.true_position[1])**2)
            deep_error = np.sqrt((deep_pos[0] - self.true_position[0])**2 + 
                               (deep_pos[1] - self.true_position[1])**2)
            
            self.standard_errors.append(standard_error)
            self.deep_errors.append(deep_error)
            
            print(f"Step {step+1}/{self.max_steps}: Standard Error = {standard_error:.2f}, Deep Error = {deep_error:.2f}")
        
        # Calculate average errors
        avg_standard_error = np.mean(self.standard_errors)
        avg_deep_error = np.mean(self.deep_errors)
        
        print(f"Average Error - Standard MCL: {avg_standard_error:.2f}, Deep MCL: {avg_deep_error:.2f}")
        print(f"Error Reduction: {(1 - avg_deep_error/avg_standard_error) * 100:.2f}%")
        
        return self.trajectory, self.standard_particle_history, self.deep_particle_history
    
    def visualize_static(self):
        """Create a static visualization of the final state"""
        # Left subplot - Standard MCL
        ax1 = self.axes[0]
        ax1.imshow(self.map_data, cmap='gray', origin='lower')
        
        # Plot the trajectory
        trajectory = np.array(self.trajectory)
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Robot Path')
        
        # Plot the final particles
        particles = np.array(self.standard_particle_history[-1])
        ax1.scatter(particles[:, 0], particles[:, 1], s=3, c='b', alpha=0.5, label='Particles')
        
        # Get the final estimated position
        estimated_position = self.standard_mcl.get_estimated_position()
        ax1.plot(estimated_position[0], estimated_position[1], 'yo', markersize=10, label='Estimated Position')
        
        # Plot true final position
        ax1.plot(self.trajectory[-1][0], self.trajectory[-1][1], 'go', markersize=10, label='True Position')
        
        ax1.set_title(f'Standard MCL\nFinal Error: {self.standard_errors[-1]:.2f}')
        ax1.legend()
        
        # Right subplot - Deep MCL
        ax2 = self.axes[1]
        ax2.imshow(self.map_data, cmap='gray', origin='lower')
        
        # Plot the trajectory
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Robot Path')
        
        # Plot the final particles
        particles = np.array(self.deep_particle_history[-1])
        ax2.scatter(particles[:, 0], particles[:, 1], s=3, c='b', alpha=0.5, label='Particles')
        
        # Get the final estimated position
        estimated_position = self.deep_mcl.get_estimated_position()
        ax2.plot(estimated_position[0], estimated_position[1], 'yo', markersize=10, label='Estimated Position')
        
        # Plot true final position
        ax2.plot(self.trajectory[-1][0], self.trajectory[-1][1], 'go', markersize=10, label='True Position')
        
        ax2.set_title(f'Deep Learning Enhanced MCL\nFinal Error: {self.deep_errors[-1]:.2f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_animation(self):
        """Create an animated visualization of both MCL implementations over time"""
        # Left subplot - Standard MCL
        ax1 = self.axes[0]
        ax1.imshow(self.map_data, cmap='gray', origin='lower')
        
        # Initialize trajectory line
        line1, = ax1.plot([], [], 'r-', linewidth=2, label='Robot Path')
        
        # Initialize particles scatter plot
        particles_scatter1 = ax1.scatter([], [], s=3, c='b', alpha=0.5, label='Particles')
        
        # Initialize robot true position
        robot_true1, = ax1.plot([], [], 'go', markersize=10, label='True Position')
        
        # Initialize robot estimated position
        robot_est1, = ax1.plot([], [], 'yo', markersize=10, label='Estimated Position')
        
        ax1.set_title('Standard MCL')
        ax1.legend()
        
        # Right subplot - Deep MCL
        ax2 = self.axes[1]
        ax2.imshow(self.map_data, cmap='gray', origin='lower')
        
        # Initialize trajectory line
        line2, = ax2.plot([], [], 'r-', linewidth=2, label='Robot Path')
        
        # Initialize particles scatter plot
        particles_scatter2 = ax2.scatter([], [], s=3, c='b', alpha=0.5, label='Particles')
        
        # Initialize robot true position
        robot_true2, = ax2.plot([], [], 'go', markersize=10, label='True Position')
        
        # Initialize robot estimated position
        robot_est2, = ax2.plot([], [], 'yo', markersize=10, label='Estimated Position')
        
        ax2.set_title('Deep Learning Enhanced MCL')
        ax2.legend()
        
        def init():
            line1.set_data([], [])
            particles_scatter1.set_offsets(np.empty((0, 2)))
            robot_true1.set_data([], [])
            robot_est1.set_data([], [])
            
            line2.set_data([], [])
            particles_scatter2.set_offsets(np.empty((0, 2)))
            robot_true2.set_data([], [])
            robot_est2.set_data([], [])
            
            return (line1, particles_scatter1, robot_true1, robot_est1,
                   line2, particles_scatter2, robot_true2, robot_est2)
        
        def update(frame):
            # Update trajectory
            trajectory = np.array(self.trajectory[:frame+1])
            line1.set_data(trajectory[:, 0], trajectory[:, 1])
            line2.set_data(trajectory[:, 0], trajectory[:, 1])
            
            # Update robot true position
            robot_true1.set_data(self.trajectory[frame][0], self.trajectory[frame][1])
            robot_true2.set_data(self.trajectory[frame][0], self.trajectory[frame][1])
            
            # Update standard MCL
            std_particles = np.array(self.standard_particle_history[frame])
            particles_scatter1.set_offsets(std_particles[:, :2])
            
            std_weights = std_particles[:, 3]
            std_est_pos = np.average(std_particles[:, :3], axis=0, weights=std_weights)
            robot_est1.set_data(std_est_pos[0], std_est_pos[1])
            
            # Update deep MCL
            deep_particles = np.array(self.deep_particle_history[frame])
            particles_scatter2.set_offsets(deep_particles[:, :2])
            
            deep_weights = deep_particles[:, 3]
            deep_est_pos = np.average(deep_particles[:, :3], axis=0, weights=deep_weights)
            robot_est2.set_data(deep_est_pos[0], deep_est_pos[1])
            
            # Update titles with errors
            if frame < len(self.standard_errors):
                ax1.set_title(f'Standard MCL (Step {frame+1})\nError: {self.standard_errors[frame]:.2f}')
                ax2.set_title(f'Deep Learning Enhanced MCL (Step {frame+1})\nError: {self.deep_errors[frame]:.2f}')
            
            return (line1, particles_scatter1, robot_true1, robot_est1,
                   line2, particles_scatter2, robot_true2, robot_est2)
        
        anim = FuncAnimation(self.fig, update, frames=len(self.trajectory),
                             init_func=init, blit=True, interval=100)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def plot_error_comparison(self):
        """Plot the error comparison between standard and deep learning enhanced MCL"""
        plt.figure(figsize=(10, 6))
        
        steps = range(1, len(self.standard_errors) + 1)
        plt.plot(steps, self.standard_errors, 'b-', label='Standard MCL')
        plt.plot(steps, self.deep_errors, 'r-', label='Deep Learning Enhanced MCL')
        
        plt.xlabel('Simulation Step')
        plt.ylabel('Position Error')
        plt.title('Error Comparison: Standard vs Deep Learning Enhanced MCL')
        plt.legend()
        plt.grid(True)
        
        # Calculate average errors
        avg_standard_error = np.mean(self.standard_errors)
        avg_deep_error = np.mean(self.deep_errors)
        
        # Add average error lines
        plt.axhline(y=avg_standard_error, color='b', linestyle='--', 
                   label=f'Avg Standard Error: {avg_standard_error:.2f}')
        plt.axhline(y=avg_deep_error, color='r', linestyle='--',
                   label=f'Avg Deep Error: {avg_deep_error:.2f}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def save_data(self, trajectory_path, std_particles_path, deep_particles_path):
        """Save trajectory and particle data to files"""
        # Save trajectory
        np.save(trajectory_path, np.array(self.trajectory))
        
        # Save standard particles
        std_particles_array = np.array([np.array(p) for p in self.standard_particle_history])
        np.save(std_particles_path, std_particles_array)
        
        # Save deep particles
        deep_particles_array = np.array([np.array(p) for p in self.deep_particle_history])
        np.save(deep_particles_path, deep_particles_array)
        
        print(f"Trajectory saved to {trajectory_path}")
        print(f"Standard particles saved to {std_particles_path}")
        print(f"Deep particles saved to {deep_particles_path}")

def main():
    parser = argparse.ArgumentParser(description='Deep Learning Enhanced MCL Test Script')
    parser.add_argument('--map_size', type=int, nargs=2, default=[100, 100], help='Map size (width, height)')
    parser.add_argument('--num_particles', type=int, default=500, help='Number of particles')
    parser.add_argument('--motion_noise', type=float, nargs=3, default=[0.1, 0.1, 0.05], 
                        help='Motion noise parameters (x, y, theta)')
    parser.add_argument('--measurement_noise', type=float, default=0.1, help='Measurement noise parameter')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum simulation steps')
    parser.add_argument('--obstacle_count', type=int, default=10, help='Number of obstacles in the map')
    parser.add_argument('--use_trained_model', action='store_true', help='Use a pre-trained model')
    parser.add_argument('--model_path', type=str, default='deep_mcl_model.pth', help='Path to save/load the model')
    parser.add_argument('--train_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--animate', action='store_true', help='Create an animation instead of a static plot')
    parser.add_argument('--plot_errors', action='store_true', help='Plot error comparison')
    parser.add_argument('--save_trajectory', type=str, help='Path to save the trajectory data (.npy)')
    parser.add_argument('--save_std_particles', type=str, help='Path to save the standard particles data (.npy)')
    parser.add_argument('--save_deep_particles', type=str, help='Path to save the deep particles data (.npy)')
    
    args = parser.parse_args()
    
    # Create and run the test
    test = DeepMCLTest(map_size=args.map_size, 
                       num_particles=args.num_particles,
                       motion_noise=tuple(args.motion_noise),
                       measurement_noise=args.measurement_noise,
                       max_steps=args.max_steps,
                       obstacle_count=args.obstacle_count,
                       use_trained_model=args.use_trained_model,
                       model_path=args.model_path,
                       train_epochs=args.train_epochs)
    
    # Run the simulation
    test.run_simulation()
    
    # Save data if requested
    if args.save_trajectory:
        test.save_data(args.save_trajectory, args.save_std_particles, args.save_deep_particles)
    
    # Plot error comparison if requested
    if args.plot_errors:
        test.plot_error_comparison()
    
    # Visualize the results
    if args.animate:
        test.visualize_animation()
    else:
        test.visualize_static()

if __name__ == '__main__':
    main()
