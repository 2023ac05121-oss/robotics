#!/usr/bin/env python3

"""
MCL Test Script

This script demonstrates the standalone Monte Carlo Localization implementation
by running a simulation with a robot in a known environment.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from matplotlib.animation import FuncAnimation

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the MCL implementation
from src.localization.monte_carlo_localization import MonteCarloLocalization, Particle
from src.utils.robot_utils import create_artificial_map, simulate_robot_motion, simulate_sensor_readings

class MCLTest:
    def __init__(self, map_size=(100, 100), num_particles=500, motion_noise=(0.1, 0.1, 0.05), 
                 measurement_noise=0.1, max_steps=100, obstacle_count=10):
        self.map_size = map_size
        self.num_particles = num_particles
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.max_steps = max_steps
        self.obstacle_count = obstacle_count
        
        # Create artificial map with obstacles
        self.map_data, self.obstacles = create_artificial_map(map_size, obstacle_count)
        
        # Initialize Monte Carlo Localization
        self.mcl = MonteCarloLocalization(num_particles=num_particles, 
                              map_data=self.map_data,
                              motion_noise=motion_noise,
                              measurement_noise=measurement_noise)
        
        # Robot's true position (x, y, theta)
        valid_position = False
        while not valid_position:
            self.true_position = np.array([
                np.random.uniform(0, map_size[0]),
                np.random.uniform(0, map_size[1]),
                np.random.uniform(-np.pi, np.pi)
            ])
            valid_position = self.is_valid_position(self.true_position[:2])
        
        # History for visualization
        self.position_history = [self.true_position.copy()]
        self.estimated_position_history = []
        self.particle_history = []
        
        # Performance metrics
        self.position_errors = []
        self.orientation_errors = []
    
    def is_valid_position(self, position):
        """Check if a position is valid (inside the map and not inside an obstacle)."""
        x, y = position
        
        # Check if position is inside map boundaries
        if x < 0 or x >= self.map_size[0] or y < 0 or y >= self.map_size[1]:
            return False
        
        # Check if position is inside any obstacle
        for obstacle in self.obstacles:
            ox, oy, width, height = obstacle
            # Check if point is inside the rectangle
            if (ox <= x <= ox + width) and (oy <= y <= oy + height):
                return False
        
        return True
    
    def run_simulation(self):
        """Run the MCL simulation."""
        print("Running Monte Carlo Localization simulation...")
        
        # Initialize animation data if needed
        self.particle_history.append(self.mcl.particles.copy())
        
        # Initial measurement to update particle weights
        measurements = simulate_sensor_readings(
            self.true_position, self.map_data, 
            num_rays=36, 
            max_range=50.0, 
            noise=self.measurement_noise
        )
        
        # Update weights based on measurements
        sensor_angles = np.linspace(-np.pi, np.pi, 36, endpoint=False)
        self.mcl.measurement_update(measurements, sensor_angles, max_range=50.0)
        
        # Resample particles
        self.mcl.resample()
        
        # Get estimated position
        x_weighted_avg = np.sum([p.x * p.weight for p in self.mcl.particles])
        y_weighted_avg = np.sum([p.y * p.weight for p in self.mcl.particles])
        theta_weighted_avg = np.sum([p.theta * p.weight for p in self.mcl.particles])
        estimated_position = np.array([x_weighted_avg, y_weighted_avg, theta_weighted_avg])
        self.estimated_position_history.append(estimated_position)
        
        # Calculate initial error
        position_error = np.sqrt(
            (estimated_position[0] - self.true_position[0])**2 + 
            (estimated_position[1] - self.true_position[1])**2
        )
        orientation_error = abs(
            (estimated_position[2] - self.true_position[2] + np.pi) % (2 * np.pi) - np.pi
        )
        self.position_errors.append(position_error)
        self.orientation_errors.append(orientation_error)
        
        # Main simulation loop
        for step in range(self.max_steps):
            # Generate random control input (linear velocity, angular velocity)
            control = np.array([
                np.random.uniform(0.5, 2.0),         # Linear velocity
                np.random.uniform(-0.3, 0.3)         # Angular velocity
            ])
            
            # Update true position using motion model
            new_position = simulate_robot_motion(self.true_position, control, noise=self.motion_noise)
            
            # Check if new position is valid
            if not self.is_valid_position(new_position[:2]):
                # If not valid, try a different control
                continue
            
            # Update true position
            self.true_position = new_position
            self.position_history.append(self.true_position.copy())
            
            # Move particles according to motion model
            # Convert control input to motion deltas for the MCL algorithm
            dt = 1.0  # time step
            dx = control[0] * dt * np.cos(self.true_position[2])
            dy = control[0] * dt * np.sin(self.true_position[2])
            dtheta = control[1] * dt
            self.mcl.motion_update(dx, dy, dtheta)
            
            # Get measurements from true position
            measurements = simulate_sensor_readings(
                self.true_position, self.map_data, 
                num_rays=36, 
                max_range=50.0, 
                noise=self.measurement_noise
            )
            
            # Update weights based on measurements
            sensor_angles = np.linspace(-np.pi, np.pi, 36, endpoint=False)
            self.mcl.measurement_update(measurements, sensor_angles, max_range=50.0)
            
            # Resample particles if needed
            if step % 5 == 0:
                self.mcl.resample()
            
            # Save particles for visualization
            self.particle_history.append(self.mcl.particles.copy())
            
            # Get estimated position
            x_weighted_avg = np.sum([p.x * p.weight for p in self.mcl.particles])
            y_weighted_avg = np.sum([p.y * p.weight for p in self.mcl.particles])
            theta_weighted_avg = np.sum([p.theta * p.weight for p in self.mcl.particles])
            estimated_position = np.array([x_weighted_avg, y_weighted_avg, theta_weighted_avg])
            self.estimated_position_history.append(estimated_position)
            
            # Calculate error
            position_error = np.sqrt(
                (estimated_position[0] - self.true_position[0])**2 + 
                (estimated_position[1] - self.true_position[1])**2
            )
            orientation_error = abs(
                (estimated_position[2] - self.true_position[2] + np.pi) % (2 * np.pi) - np.pi
            )
            self.position_errors.append(position_error)
            self.orientation_errors.append(orientation_error)
            
            # Print progress
            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}/{self.max_steps} | Position Error: {position_error:.2f} | Orientation Error: {orientation_error:.2f}")
        
        print("Simulation complete!")
        
        # Calculate average errors
        avg_position_error = np.mean(self.position_errors)
        avg_orientation_error = np.mean(self.orientation_errors)
        
        print(f"Average Position Error: {avg_position_error:.2f}")
        print(f"Average Orientation Error: {avg_orientation_error:.2f}")
        
        return {
            'position_history': self.position_history,
            'estimated_position_history': self.estimated_position_history,
            'particle_history': self.particle_history,
            'position_errors': self.position_errors,
            'orientation_errors': self.orientation_errors,
            'map_data': self.map_data,
            'obstacles': self.obstacles
        }
    
    def visualize_results(self, animate=False, save_path=None):
        """Visualize the simulation results."""
        if not self.position_history or not self.estimated_position_history:
            print("No simulation data to visualize. Run the simulation first.")
            return
        
        if animate:
            self.animate_results(save_path)
        else:
            self.plot_results(save_path)
    
    def plot_results(self, save_path=None):
        """Plot the final results of the simulation."""
        # Convert histories to numpy arrays for easier plotting
        true_positions = np.array(self.position_history)
        estimated_positions = np.array(self.estimated_position_history)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot map and trajectories
        ax1.imshow(self.map_data, cmap='gray_r', origin='lower', 
                 extent=[0, self.map_size[0], 0, self.map_size[1]])
        
        # Plot obstacles
        for ox, oy, width, height in self.obstacles:
            rect = plt.Rectangle((ox, oy), width, height, fill=True, color='red', alpha=0.5)
            ax1.add_patch(rect)
        
        # Plot true trajectory
        ax1.plot(true_positions[:, 0], true_positions[:, 1], 'b-', label='True Path')
        
        # Plot estimated trajectory
        ax1.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'r--', label='Estimated Path')
        
        # Plot start and end positions
        ax1.plot(true_positions[0, 0], true_positions[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(true_positions[-1, 0], true_positions[-1, 1], 'mo', markersize=10, label='End')
        
        ax1.set_title('Robot Trajectory')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid(True)
        
        # Plot errors
        steps = np.arange(len(self.position_errors))
        ax2.plot(steps, self.position_errors, 'b-', label='Position Error')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Position Error', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.grid(True)
        
        # Create second y-axis for orientation errors
        ax3 = ax2.twinx()
        ax3.plot(steps, self.orientation_errors, 'r-', label='Orientation Error')
        ax3.set_ylabel('Orientation Error (rad)', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        
        # Add combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2.set_title('Localization Errors')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def animate_results(self, save_path=None):
        """Create an animation of the simulation."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show the map
        ax.imshow(self.map_data, cmap='gray_r', origin='lower', 
                extent=[0, self.map_size[0], 0, self.map_size[1]])
        
        # Plot obstacles
        for ox, oy, width, height in self.obstacles:
            rect = plt.Rectangle((ox, oy), width, height, fill=True, color='red', alpha=0.5)
            ax.add_patch(rect)
        
        # Initialize plot elements
        true_path, = ax.plot([], [], 'b-', label='True Path')
        estimated_path, = ax.plot([], [], 'r--', label='Estimated Path')
        true_pos = ax.scatter([], [], c='blue', s=100, marker='o')
        est_pos = ax.scatter([], [], c='red', s=100, marker='o')
        particles = ax.scatter([], [], c='green', s=10, alpha=0.5, marker='.')
        
        # Set plot limits and labels
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        ax.set_title('Monte Carlo Localization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        # Ensure histories are converted to lists of numpy arrays
        true_positions = [pos[:2] for pos in self.position_history]
        estimated_positions = [pos[:2] for pos in self.estimated_position_history]
        
        def init():
            true_path.set_data([], [])
            estimated_path.set_data([], [])
            true_pos.set_offsets(np.empty((0, 2)))
            est_pos.set_offsets(np.empty((0, 2)))
            particles.set_offsets(np.empty((0, 2)))
            return true_path, estimated_path, true_pos, est_pos, particles
        
        def update(frame):
            # Update true path
            true_x = [pos[0] for pos in true_positions[:frame+1]]
            true_y = [pos[1] for pos in true_positions[:frame+1]]
            true_path.set_data(true_x, true_y)
            
            # Update estimated path
            if frame < len(estimated_positions):
                est_x = [pos[0] for pos in estimated_positions[:frame+1]]
                est_y = [pos[1] for pos in estimated_positions[:frame+1]]
                estimated_path.set_data(est_x, est_y)
                
                # Update current positions
                true_pos.set_offsets([true_positions[frame]])
                est_pos.set_offsets([estimated_positions[frame]])
                
                # Update particles
                if frame < len(self.particle_history):
                    current_particles = self.particle_history[frame]
                    particle_positions = np.array([[p.x, p.y] for p in current_particles])
                    particles.set_offsets(particle_positions)
            
            return true_path, estimated_path, true_pos, est_pos, particles
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=len(true_positions),
            init_func=init, blit=True, interval=100
        )
        
        if save_path:
            # Save animation as gif or mp4
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=10)
            else:
                anim.save(save_path, writer='ffmpeg', fps=10)
            plt.close()
        else:
            plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Monte Carlo Localization Test')
    parser.add_argument('--map_size', type=int, nargs=2, default=[100, 100], help='Map size as width height')
    parser.add_argument('--num_particles', type=int, default=500, help='Number of particles')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum simulation steps')
    parser.add_argument('--obstacle_count', type=int, default=10, help='Number of obstacles')
    parser.add_argument('--animate', action='store_true', help='Animate results')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--save_data', action='store_true', help='Save simulation data to JSON')
    args = parser.parse_args()
    
    # Create and run test
    test = MCLTest(map_size=args.map_size, 
                   num_particles=args.num_particles,
                   max_steps=args.max_steps,
                   obstacle_count=args.obstacle_count)
    
    results = test.run_simulation()
    
    # Visualize results
    test.visualize_results(animate=args.animate, save_path=args.save_path)
    
    # Save data if requested
    if args.save_data:
        # Convert numpy arrays to lists for JSON serialization
        save_data = {
            'position_history': [pos.tolist() for pos in results['position_history']],
            'estimated_position_history': [pos.tolist() for pos in results['estimated_position_history']],
            'position_errors': results['position_errors'],
            'orientation_errors': results['orientation_errors'],
            'obstacles': results['obstacles']
        }
        
        # Save as JSON
        with open('mcl_simulation_data.json', 'w') as f:
            json.dump(save_data, f)
        
        print("Simulation data saved to mcl_simulation_data.json")

if __name__ == '__main__':
    main()
