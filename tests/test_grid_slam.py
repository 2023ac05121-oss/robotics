#!/usr/bin/env python3

"""
Grid-based SLAM Test Script

This script demonstrates the Grid-based SLAM implementation
by running a simulation with a robot in an unknown environment.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
from matplotlib.animation import FuncAnimation

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the SLAM implementation
from src.mapping.grid_slam import GridSLAM
from src.utils.robot_utils import create_artificial_map, simulate_robot_motion, simulate_sensor_readings

class GridSLAMTest:
    def __init__(self, map_size=(100, 100), grid_size=1.0, motion_noise=(0.1, 0.1, 0.05), 
                 measurement_noise=0.1, max_steps=100, obstacle_count=10):
        self.map_size = map_size
        self.grid_size = grid_size
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.max_steps = max_steps
        self.obstacle_count = obstacle_count
        
        # Create artificial map with obstacles
        self.true_map, self.obstacles = create_artificial_map(map_size, obstacle_count)
        
        # Initialize Grid-based SLAM
        self.slam = GridSLAM(
            grid_size=map_size, 
            cell_size=grid_size,
            initial_position=(map_size[0]/2, map_size[1]/2, 0.0),
            motion_noise=motion_noise,
            measurement_noise=measurement_noise
        )
        
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
        
        # Storage for trajectory and map data
        self.trajectory = [self.true_position.copy()]
        self.estimated_trajectory = [self.true_position.copy()]  # Initially, we know the position
        self.map_history = [self.slam.get_map_as_array().copy()]
        
        # Error tracking
        self.position_errors = [0.0]  # Initially, we know the position exactly
        self.map_errors = [self.calculate_map_error()]
        
        # Set up visualization
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.suptitle('Grid-based SLAM Test')
    
    def is_valid_position(self, position):
        """Check if a position is valid (not inside an obstacle)"""
        x, y = position
        
        # Check if position is within map bounds
        if x < 0 or x >= self.map_size[0] or y < 0 or y >= self.map_size[1]:
            return False
        
        # Check if position is inside any obstacle
        for obstacle in self.obstacles:
            ox, oy, width, height = obstacle
            # Check if point is inside the rectangle
            if (ox <= x <= ox + width) and (oy <= y <= oy + height):
                return False
        
        return True
    
    def calculate_map_error(self):
        """Calculate the error between the estimated map and the true map"""
        estimated_map = self.slam.get_map_as_array()
        
        # Convert true map to occupancy grid format
        # In true_map: 0 = occupied, 255 = free
        # In occupancy grid: 0-0.5 = free, 0.5-1 = occupied
        true_occupancy = np.zeros_like(self.true_map, dtype=float)
        true_occupancy[self.true_map > 127] = 0.0  # Free
        true_occupancy[self.true_map <= 127] = 1.0  # Occupied
        
        # Calculate error (mean absolute difference)
        error = np.mean(np.abs(estimated_map - true_occupancy))
        
        return error
    
    def run_simulation(self):
        """Run the SLAM simulation for the specified number of steps"""
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
                control = np.array([0.0, 0.0, rotation])  # [dx, dy, dtheta]
            else:
                # Move forward
                speed = min(1.0, distance / 5.0)  # Slow down when close to waypoint
                dx = speed * math.cos(self.true_position[2])
                dy = speed * math.sin(self.true_position[2])
                control = np.array([dx, dy, 0.0])  # [dx, dy, dtheta]
            
            # Simulate robot motion with the control input
            dx, dy, dtheta = control
            # Create a motion vector for simulation [linear_velocity, angular_velocity]
            # We'll use a simplified approach: linear velocity is the magnitude of [dx, dy]
            # and angular velocity is just dtheta
            v = math.sqrt(dx*dx + dy*dy)
            omega = dtheta
            motion_control = np.array([v, omega])
            new_position = simulate_robot_motion(self.true_position, motion_control, noise=self.motion_noise)
            
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
            measurements = simulate_sensor_readings(
                self.true_position, 
                self.true_map, 
                num_rays=36, 
                max_range=50.0, 
                noise=self.measurement_noise
            )
            
            # Update SLAM with control and measurements
            dx, dy, dtheta = control
            self.slam.move_robot(dx, dy, dtheta)
            
            # Calculate sensor angles
            sensor_angles = np.linspace(-np.pi, np.pi, 36, endpoint=False)
            
            # Update map with measurements
            self.slam.update_map(measurements, sensor_angles, max_range=50.0)
            
            # Store trajectory and map data
            self.trajectory.append(self.true_position.copy())
            estimated_pos = self.slam.get_position() if hasattr(self.slam, 'get_position') else (self.slam.x, self.slam.y, self.slam.theta)
            self.estimated_trajectory.append(estimated_pos)
            self.map_history.append(self.slam.get_map_as_array().copy())
            
            # Calculate errors
            position_error = np.sqrt((estimated_pos[0] - self.true_position[0])**2 + 
                                    (estimated_pos[1] - self.true_position[1])**2)
            map_error = self.calculate_map_error()
            
            self.position_errors.append(position_error)
            self.map_errors.append(map_error)
            
            print(f"Step {step+1}/{self.max_steps}: Position Error = {position_error:.2f}, Map Error = {map_error:.4f}")
        
        # Calculate average errors
        avg_position_error = np.mean(self.position_errors)
        avg_map_error = np.mean(self.map_errors)
        
        print(f"Average Position Error: {avg_position_error:.2f}")
        print(f"Average Map Error: {avg_map_error:.4f}")
        
        return self.trajectory, self.estimated_trajectory, self.map_history
    
    def visualize_static(self):
        """Create a static visualization of the final state"""
        # Left subplot - True map and trajectory
        ax1 = self.axes[0]
        ax1.imshow(self.true_map, cmap='gray', origin='lower')
        
        # Plot the true trajectory
        trajectory = np.array(self.trajectory)
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='True Path')
        
        # Plot start and end positions
        ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'bo', markersize=10, label='End')
        
        ax1.set_title('True Map and Trajectory')
        ax1.legend()
        
        # Middle subplot - Estimated map
        ax2 = self.axes[1]
        estimated_map = self.map_history[-1]
        
        # Convert occupancy grid to visualization format
        # In occupancy grid: 0-0.5 = free, 0.5-1 = occupied, 0.5 = unknown
        viz_map = np.ones_like(estimated_map) * 0.5  # Initialize as unknown
        viz_map[estimated_map < 0.45] = 1.0  # Free (white)
        viz_map[estimated_map > 0.55] = 0.0  # Occupied (black)
        
        ax2.imshow(viz_map, cmap='gray', origin='lower')
        
        # Plot the estimated trajectory
        est_trajectory = np.array(self.estimated_trajectory)
        ax2.plot(est_trajectory[:, 0], est_trajectory[:, 1], 'r-', linewidth=2, label='Estimated Path')
        
        # Plot start and end positions
        ax2.plot(est_trajectory[0, 0], est_trajectory[0, 1], 'go', markersize=10, label='Start')
        ax2.plot(est_trajectory[-1, 0], est_trajectory[-1, 1], 'bo', markersize=10, label='End')
        
        ax2.set_title('Estimated Map and Trajectory')
        ax2.legend()
        
        # Right subplot - Error metrics
        ax3 = self.axes[2]
        steps = range(0, len(self.position_errors))
        
        # Plot position error
        ax3.plot(steps, self.position_errors, 'r-', label='Position Error')
        
        # Plot map error
        ax3_twin = ax3.twinx()
        ax3_twin.plot(steps, self.map_errors, 'b-', label='Map Error')
        
        ax3.set_xlabel('Simulation Step')
        ax3.set_ylabel('Position Error', color='r')
        ax3_twin.set_ylabel('Map Error', color='b')
        
        ax3.tick_params(axis='y', labelcolor='r')
        ax3_twin.tick_params(axis='y', labelcolor='b')
        
        ax3.set_title('Error Metrics')
        
        # Add a legend
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_animation(self):
        """Create an animated visualization of the SLAM process over time"""
        # Left subplot - True map and trajectory
        ax1 = self.axes[0]
        ax1.imshow(self.true_map, cmap='gray', origin='lower')
        
        # Initialize true trajectory line
        true_line, = ax1.plot([], [], 'r-', linewidth=2, label='True Path')
        
        # Initialize robot true position
        robot_true, = ax1.plot([], [], 'go', markersize=10, label='Robot')
        
        ax1.set_title('True Map and Trajectory')
        ax1.legend()
        
        # Middle subplot - Estimated map
        ax2 = self.axes[1]
        
        # Initialize estimated map
        estimated_map_img = ax2.imshow(np.ones((self.map_size[0], self.map_size[1])) * 0.5, 
                                      cmap='gray', origin='lower')
        
        # Initialize estimated trajectory line
        est_line, = ax2.plot([], [], 'r-', linewidth=2, label='Estimated Path')
        
        # Initialize robot estimated position
        robot_est, = ax2.plot([], [], 'go', markersize=10, label='Robot')
        
        ax2.set_title('Estimated Map and Trajectory')
        ax2.legend()
        
        # Right subplot - Error metrics
        ax3 = self.axes[2]
        
        # Initialize position error line
        pos_error_line, = ax3.plot([], [], 'r-', label='Position Error')
        
        # Create twin axis for map error
        ax3_twin = ax3.twinx()
        
        # Initialize map error line
        map_error_line, = ax3_twin.plot([], [], 'b-', label='Map Error')
        
        ax3.set_xlabel('Simulation Step')
        ax3.set_ylabel('Position Error', color='r')
        ax3_twin.set_ylabel('Map Error', color='b')
        
        ax3.tick_params(axis='y', labelcolor='r')
        ax3_twin.tick_params(axis='y', labelcolor='b')
        
        ax3.set_title('Error Metrics')
        
        # Add a legend
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Set initial axis limits for error plot
        ax3.set_xlim(0, self.max_steps)
        ax3.set_ylim(0, 10)  # Adjust based on expected error range
        ax3_twin.set_ylim(0, 0.5)  # Adjust based on expected error range
        
        def init():
            true_line.set_data([], [])
            robot_true.set_data([], [])
            
            est_line.set_data([], [])
            robot_est.set_data([], [])
            
            pos_error_line.set_data([], [])
            map_error_line.set_data([], [])
            
            return (true_line, robot_true, estimated_map_img, est_line, robot_est, 
                   pos_error_line, map_error_line)
        
        def update(frame):
            # Update true trajectory and position
            true_trajectory = np.array(self.trajectory[:frame+1])
            true_line.set_data(true_trajectory[:, 0], true_trajectory[:, 1])
            robot_true.set_data(self.trajectory[frame][0], self.trajectory[frame][1])
            
            # Update estimated map
            estimated_map = self.map_history[frame]
            
            # Convert occupancy grid to visualization format
            viz_map = np.ones_like(estimated_map) * 0.5  # Initialize as unknown
            viz_map[estimated_map < 0.45] = 1.0  # Free (white)
            viz_map[estimated_map > 0.55] = 0.0  # Occupied (black)
            
            estimated_map_img.set_array(viz_map)
            
            # Update estimated trajectory and position
            est_trajectory = np.array(self.estimated_trajectory[:frame+1])
            est_line.set_data(est_trajectory[:, 0], est_trajectory[:, 1])
            robot_est.set_data(self.estimated_trajectory[frame][0], self.estimated_trajectory[frame][1])
            
            # Update error metrics
            steps = range(0, frame+1)
            pos_error_line.set_data(steps, self.position_errors[:frame+1])
            map_error_line.set_data(steps, self.map_errors[:frame+1])
            
            # Update titles with current errors
            ax1.set_title(f'True Map and Trajectory (Step {frame+1})')
            ax2.set_title(f'Estimated Map and Trajectory (Step {frame+1})')
            ax3.set_title(f'Error Metrics (Position: {self.position_errors[frame]:.2f}, Map: {self.map_errors[frame]:.4f})')
            
            return (true_line, robot_true, estimated_map_img, est_line, robot_est, 
                   pos_error_line, map_error_line)
        
        anim = FuncAnimation(self.fig, update, frames=len(self.trajectory),
                             init_func=init, blit=True, interval=100)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def save_data(self, true_trajectory_path, est_trajectory_path, map_history_path):
        """Save trajectory and map data to files"""
        # Save true trajectory
        np.save(true_trajectory_path, np.array(self.trajectory))
        
        # Save estimated trajectory
        np.save(est_trajectory_path, np.array(self.estimated_trajectory))
        
        # Save map history
        np.save(map_history_path, np.array(self.map_history))
        
        print(f"True trajectory saved to {true_trajectory_path}")
        print(f"Estimated trajectory saved to {est_trajectory_path}")
        print(f"Map history saved to {map_history_path}")

def main():
    parser = argparse.ArgumentParser(description='Grid-based SLAM Test Script')
    parser.add_argument('--map_size', type=int, nargs=2, default=[100, 100], help='Map size (width, height)')
    parser.add_argument('--grid_size', type=float, default=1.0, help='Grid cell size')
    parser.add_argument('--motion_noise', type=float, nargs=3, default=[0.1, 0.1, 0.05], 
                        help='Motion noise parameters (x, y, theta)')
    parser.add_argument('--measurement_noise', type=float, default=0.1, help='Measurement noise parameter')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum simulation steps')
    parser.add_argument('--obstacle_count', type=int, default=10, help='Number of obstacles in the map')
    parser.add_argument('--animate', action='store_true', help='Create an animation instead of a static plot')
    parser.add_argument('--save_true_trajectory', type=str, help='Path to save the true trajectory data (.npy)')
    parser.add_argument('--save_est_trajectory', type=str, help='Path to save the estimated trajectory data (.npy)')
    parser.add_argument('--save_map_history', type=str, help='Path to save the map history data (.npy)')
    
    args = parser.parse_args()
    
    # Create and run the test
    test = GridSLAMTest(map_size=args.map_size, 
                       grid_size=args.grid_size,
                       motion_noise=tuple(args.motion_noise),
                       measurement_noise=args.measurement_noise,
                       max_steps=args.max_steps,
                       obstacle_count=args.obstacle_count)
    
    # Run the simulation
    test.run_simulation()
    
    # Save data if requested
    if args.save_true_trajectory:
        test.save_data(args.save_true_trajectory, args.save_est_trajectory, args.save_map_history)
    
    # Visualize the results
    if args.animate:
        test.visualize_animation()
    else:
        test.visualize_static()

if __name__ == '__main__':
    main()
