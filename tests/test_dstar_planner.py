#!/usr/bin/env python3

"""
D* Planner Test Script

This script demonstrates the D* Path Planning algorithm implementation
by running a simulation with a robot navigating in a dynamic environment.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.animation import FuncAnimation
import time

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the D* Planner implementation
from src.planning.dstar_planner import DStar
from src.utils.robot_utils import create_artificial_map, simulate_robot_motion

class DStarTest:
    def __init__(self, map_size=(100, 100), motion_noise=(0.1, 0.1, 0.05), 
                 max_steps=100, obstacle_count=10, dynamic_obstacles=True,
                 new_obstacle_interval=10, sensor_range=20):
        self.map_size = map_size
        self.motion_noise = motion_noise
        self.max_steps = max_steps
        self.obstacle_count = obstacle_count
        self.dynamic_obstacles = dynamic_obstacles
        self.new_obstacle_interval = new_obstacle_interval
        self.sensor_range = sensor_range
        
        # Create artificial map with obstacles
        self.true_map, self.obstacles = create_artificial_map(map_size, obstacle_count)
        
        # Create binary map for D* (0 = free, 1 = obstacle)
        self.binary_map = np.ones(map_size, dtype=int)  # Initialize as all obstacle
        self.binary_map[self.true_map > 127] = 0  # Free space
        
        # The robot's known map (initially incomplete)
        self.known_map = np.ones(map_size, dtype=int)  # Initialize as all unknown
        
        # Initialize D* Planner
        self.dstar = DStar(self.known_map)
        
        # Generate start and goal positions
        self.start_pos, self.goal_pos = self.generate_start_goal()
        
        # Robot's current position (x, y, theta)
        self.current_pos = np.array([self.start_pos[0], self.start_pos[1], 0.0])
        
        # Update the known map with initial sensor information
        self.update_known_map()
        
        # Plan initial path
        self.dstar.init(self.start_pos[0], self.start_pos[1], self.goal_pos[0], self.goal_pos[1])
        self.path = self.dstar.plan()
        
        # Dynamic obstacles list
        self.dynamic_obstacles = []
        
        # Storage for trajectory and path data
        self.trajectory = [self.current_pos.copy()]
        self.path_history = [self.path.copy() if self.path else []]
        self.known_map_history = [self.known_map.copy()]
        
        # Set up visualization
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.canvas.set_window_title('D* Path Planning Test')
    
    def generate_start_goal(self):
        """Generate valid start and goal positions"""
        # Generate start position
        valid_start = False
        while not valid_start:
            start = (np.random.randint(0, self.map_size[0]), 
                     np.random.randint(0, self.map_size[1]))
            if self.binary_map[start[0], start[1]] == 0:  # Free space
                valid_start = True
        
        # Generate goal position (ensure it's far enough from start)
        valid_goal = False
        min_distance = max(self.map_size) / 3  # At least 1/3 of the map size apart
        
        while not valid_goal:
            goal = (np.random.randint(0, self.map_size[0]), 
                    np.random.randint(0, self.map_size[1]))
            
            # Check if it's free space
            if self.binary_map[goal[0], goal[1]] == 0:
                # Check distance from start
                distance = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
                if distance >= min_distance:
                    valid_goal = True
        
        return start, goal
    
    def update_known_map(self):
        """Update the robot's known map based on its current position and sensor range"""
        x, y = int(self.current_pos[0]), int(self.current_pos[1])
        
        # Update cells within sensor range
        for i in range(max(0, x - self.sensor_range), min(self.map_size[0], x + self.sensor_range + 1)):
            for j in range(max(0, y - self.sensor_range), min(self.map_size[1], y + self.sensor_range + 1)):
                # Calculate distance to cell
                distance = np.sqrt((i - x)**2 + (j - y)**2)
                
                if distance <= self.sensor_range:
                    # Update the known map with the actual value from the binary map
                    self.known_map[i, j] = self.binary_map[i, j]
    
    def add_dynamic_obstacle(self):
        """Add a new random obstacle to the environment"""
        # Generate a position for the new obstacle
        valid_pos = False
        while not valid_pos:
            pos = (np.random.randint(0, self.map_size[0]), 
                   np.random.randint(0, self.map_size[1]))
            
            # Ensure it's not too close to the robot or goal
            robot_distance = np.sqrt((pos[0] - self.current_pos[0])**2 + (pos[1] - self.current_pos[1])**2)
            goal_distance = np.sqrt((pos[0] - self.goal_pos[0])**2 + (pos[1] - self.goal_pos[1])**2)
            
            if robot_distance > 10 and goal_distance > 10 and self.binary_map[pos[0], pos[1]] == 0:
                valid_pos = True
        
        # Add the obstacle
        obstacle_size = np.random.randint(3, 8)  # Random size between 3x3 and 7x7
        
        for i in range(max(0, pos[0] - obstacle_size//2), min(self.map_size[0], pos[0] + obstacle_size//2 + 1)):
            for j in range(max(0, pos[1] - obstacle_size//2), min(self.map_size[1], pos[1] + obstacle_size//2 + 1)):
                self.binary_map[i, j] = 1  # Set as obstacle
                
                # If within sensor range, update the known map too
                distance = np.sqrt((i - self.current_pos[0])**2 + (j - self.current_pos[1])**2)
                if distance <= self.sensor_range:
                    self.known_map[i, j] = 1
        
        print(f"Added new obstacle at position {pos} with size {obstacle_size}x{obstacle_size}")
        
        # Return the affected area
        return (max(0, pos[0] - obstacle_size//2), max(0, pos[1] - obstacle_size//2), 
                min(self.map_size[0], pos[0] + obstacle_size//2 + 1), 
                min(self.map_size[1], pos[1] + obstacle_size//2 + 1))
    
    def run_simulation(self):
        """Run the D* planning simulation for the specified number of steps"""
        steps_since_last_obstacle = 0
        
        for step in range(self.max_steps):
            # Check if we've reached the goal
            current_cell = (int(self.current_pos[0]), int(self.current_pos[1]))
            if current_cell[0] == self.goal_pos[0] and current_cell[1] == self.goal_pos[1]:
                print(f"Goal reached in {step} steps!")
                break
            
            # Add dynamic obstacle periodically
            if self.dynamic_obstacles and steps_since_last_obstacle >= self.new_obstacle_interval:
                affected_area = self.add_dynamic_obstacle()
                steps_since_last_obstacle = 0
                
                # Update D* with the new obstacle information
                # This will trigger replanning if the obstacle affects the current path
                self.dstar.update_cell(affected_area)
                self.path = self.dstar.plan()
            else:
                steps_since_last_obstacle += 1
            
            # Get next waypoint from the path
            if self.path and len(self.path) > 1:
                next_waypoint = self.path[1]  # path[0] is current position
            else:
                print("No valid path to goal found!")
                break
            
            # Calculate direction to next waypoint
            dx = next_waypoint[0] - self.current_pos[0]
            dy = next_waypoint[1] - self.current_pos[1]
            
            # Calculate desired heading
            desired_theta = np.arctan2(dy, dx)
            
            # Calculate difference in heading
            dtheta = desired_theta - self.current_pos[2]
            # Normalize to [-pi, pi]
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            
            # If heading is off by more than 0.1 radians, rotate first
            if abs(dtheta) > 0.1:
                # Rotate the robot
                rotation = 0.1 * np.sign(dtheta)
                control = np.array([0.0, 0.0, rotation])
            else:
                # Move forward
                distance = np.sqrt(dx**2 + dy**2)
                speed = min(1.0, distance)  # Limit speed
                control = np.array([speed, 0.0, 0.0])
            
            # Simulate robot motion with the control input
            new_position = simulate_robot_motion(self.current_pos, control, noise=self.motion_noise)
            
            # Check if the new position is valid
            new_cell = (int(new_position[0]), int(new_position[1]))
            if (new_cell[0] >= 0 and new_cell[0] < self.map_size[0] and
                new_cell[1] >= 0 and new_cell[1] < self.map_size[1] and
                self.binary_map[new_cell[0], new_cell[1]] == 0):  # Free space
                self.current_pos = new_position
            else:
                # If not valid, stay in place but update heading
                self.current_pos[2] = new_position[2]
            
            # Update the known map with new sensor information
            self.update_known_map()
            
            # Check if the robot needs to replan due to new map information
            if step % 5 == 0:  # Replan periodically to handle map updates
                self.dstar.update_known_map(self.known_map)
                self.path = self.dstar.plan()
            
            # Store trajectory and path data
            self.trajectory.append(self.current_pos.copy())
            self.path_history.append(self.path.copy() if self.path else [])
            self.known_map_history.append(self.known_map.copy())
            
            print(f"Step {step+1}/{self.max_steps}: Position = ({self.current_pos[0]:.1f}, {self.current_pos[1]:.1f}), "
                  f"Distance to Goal = {np.sqrt((self.current_pos[0] - self.goal_pos[0])**2 + (self.current_pos[1] - self.goal_pos[1])**2):.1f}")
        
        return self.trajectory, self.path_history, self.known_map_history
    
    def visualize_static(self):
        """Create a static visualization of the final state"""
        # Left subplot - True map and trajectory
        ax1 = self.axes[0]
        
        # Create a visualization map
        viz_map = np.zeros((self.map_size[0], self.map_size[1], 3))  # RGB
        viz_map[self.binary_map == 0] = [1, 1, 1]  # Free space (white)
        viz_map[self.binary_map == 1] = [0, 0, 0]  # Obstacles (black)
        
        # Mark the start and goal positions
        viz_map[self.start_pos[0], self.start_pos[1]] = [0, 1, 0]  # Start (green)
        viz_map[self.goal_pos[0], self.goal_pos[1]] = [1, 0, 0]  # Goal (red)
        
        ax1.imshow(viz_map, origin='lower')
        
        # Plot the true trajectory
        trajectory = np.array(self.trajectory)
        ax1.plot(trajectory[:, 1], trajectory[:, 0], 'b-', linewidth=2, label='Robot Path')
        
        ax1.set_title('True Map and Trajectory')
        ax1.legend()
        
        # Right subplot - Known map and planned path
        ax2 = self.axes[1]
        
        # Create a visualization of the known map
        known_viz_map = np.zeros((self.map_size[0], self.map_size[1], 3))  # RGB
        known_viz_map[self.known_map == 0] = [1, 1, 1]  # Free space (white)
        known_viz_map[self.known_map == 1] = [0, 0, 0]  # Obstacles (black)
        
        # Mark unexplored areas
        unexplored = (self.known_map != 0) & (self.known_map != 1)
        known_viz_map[unexplored] = [0.7, 0.7, 0.7]  # Unexplored (gray)
        
        # Mark the start and goal positions
        known_viz_map[self.start_pos[0], self.start_pos[1]] = [0, 1, 0]  # Start (green)
        known_viz_map[self.goal_pos[0], self.goal_pos[1]] = [1, 0, 0]  # Goal (red)
        
        ax2.imshow(known_viz_map, origin='lower')
        
        # Plot the final planned path
        if self.path and len(self.path) > 1:
            path = np.array(self.path)
            ax2.plot(path[:, 1], path[:, 0], 'y-', linewidth=2, label='Planned Path')
        
        ax2.set_title('Known Map and Planned Path')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_animation(self):
        """Create an animated visualization of the D* planning process over time"""
        # Left subplot - True map and trajectory
        ax1 = self.axes[0]
        
        # Create a visualization map
        viz_map = np.zeros((self.map_size[0], self.map_size[1], 3))  # RGB
        viz_map[self.binary_map == 0] = [1, 1, 1]  # Free space (white)
        viz_map[self.binary_map == 1] = [0, 0, 0]  # Obstacles (black)
        
        # Mark the start and goal positions
        viz_map[self.start_pos[0], self.start_pos[1]] = [0, 1, 0]  # Start (green)
        viz_map[self.goal_pos[0], self.goal_pos[1]] = [1, 0, 0]  # Goal (red)
        
        # Display the true map
        true_map_img = ax1.imshow(viz_map, origin='lower')
        
        # Initialize trajectory line
        traj_line, = ax1.plot([], [], 'b-', linewidth=2, label='Robot Path')
        
        # Initialize robot position
        robot_pos, = ax1.plot([], [], 'yo', markersize=10, label='Robot')
        
        ax1.set_title('True Map and Trajectory')
        ax1.legend()
        
        # Right subplot - Known map and planned path
        ax2 = self.axes[1]
        
        # Initialize known map image
        known_map_img = ax2.imshow(np.zeros((self.map_size[0], self.map_size[1], 3)), origin='lower')
        
        # Initialize planned path line
        path_line, = ax2.plot([], [], 'y-', linewidth=2, label='Planned Path')
        
        ax2.set_title('Known Map and Planned Path')
        ax2.legend()
        
        def init():
            traj_line.set_data([], [])
            robot_pos.set_data([], [])
            path_line.set_data([], [])
            return true_map_img, traj_line, robot_pos, known_map_img, path_line
        
        def update(frame):
            # Update trajectory
            trajectory = np.array(self.trajectory[:frame+1])
            traj_line.set_data(trajectory[:, 1], trajectory[:, 0])
            
            # Update robot position
            robot_pos.set_data(self.trajectory[frame][1], self.trajectory[frame][0])
            
            # Update known map
            known_map = self.known_map_history[frame]
            
            # Create a visualization of the known map
            known_viz_map = np.zeros((self.map_size[0], self.map_size[1], 3))  # RGB
            known_viz_map[known_map == 0] = [1, 1, 1]  # Free space (white)
            known_viz_map[known_map == 1] = [0, 0, 0]  # Obstacles (black)
            
            # Mark unexplored areas
            unexplored = (known_map != 0) & (known_map != 1)
            known_viz_map[unexplored] = [0.7, 0.7, 0.7]  # Unexplored (gray)
            
            # Mark the start and goal positions
            known_viz_map[self.start_pos[0], self.start_pos[1]] = [0, 1, 0]  # Start (green)
            known_viz_map[self.goal_pos[0], self.goal_pos[1]] = [1, 0, 0]  # Goal (red)
            
            known_map_img.set_array(known_viz_map)
            
            # Update planned path
            path = self.path_history[frame]
            if path and len(path) > 1:
                path = np.array(path)
                path_line.set_data(path[:, 1], path[:, 0])
            else:
                path_line.set_data([], [])
            
            # Update titles
            ax1.set_title(f'True Map and Trajectory (Step {frame+1})')
            ax2.set_title(f'Known Map and Planned Path (Step {frame+1})')
            
            return true_map_img, traj_line, robot_pos, known_map_img, path_line
        
        anim = FuncAnimation(self.fig, update, frames=len(self.trajectory),
                             init_func=init, blit=True, interval=100)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def save_data(self, trajectory_path, path_history_path, known_map_history_path):
        """Save trajectory and path data to files"""
        # Save trajectory
        np.save(trajectory_path, np.array(self.trajectory))
        
        # Save path history (convert to numpy array, padded to max length)
        max_path_length = max(len(path) for path in self.path_history) if self.path_history else 0
        path_array = np.zeros((len(self.path_history), max_path_length, 2), dtype=int)
        
        for i, path in enumerate(self.path_history):
            if path:
                path_array[i, :len(path)] = path
        
        np.save(path_history_path, path_array)
        
        # Save known map history
        np.save(known_map_history_path, np.array(self.known_map_history))
        
        print(f"Trajectory saved to {trajectory_path}")
        print(f"Path history saved to {path_history_path}")
        print(f"Known map history saved to {known_map_history_path}")

def main():
    parser = argparse.ArgumentParser(description='D* Path Planning Test Script')
    parser.add_argument('--map_size', type=int, nargs=2, default=[100, 100], help='Map size (width, height)')
    parser.add_argument('--motion_noise', type=float, nargs=3, default=[0.1, 0.1, 0.05], 
                        help='Motion noise parameters (x, y, theta)')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum simulation steps')
    parser.add_argument('--obstacle_count', type=int, default=10, help='Number of obstacles in the map')
    parser.add_argument('--dynamic_obstacles', action='store_true', help='Enable dynamic obstacles')
    parser.add_argument('--new_obstacle_interval', type=int, default=10, 
                        help='Interval (in steps) between adding new obstacles')
    parser.add_argument('--sensor_range', type=int, default=20, help='Robot sensor range')
    parser.add_argument('--animate', action='store_true', help='Create an animation instead of a static plot')
    parser.add_argument('--save_trajectory', type=str, help='Path to save the trajectory data (.npy)')
    parser.add_argument('--save_path_history', type=str, help='Path to save the path history data (.npy)')
    parser.add_argument('--save_map_history', type=str, help='Path to save the known map history data (.npy)')
    
    args = parser.parse_args()
    
    # Create and run the test
    test = DStarTest(map_size=args.map_size, 
                    motion_noise=tuple(args.motion_noise),
                    max_steps=args.max_steps,
                    obstacle_count=args.obstacle_count,
                    dynamic_obstacles=args.dynamic_obstacles,
                    new_obstacle_interval=args.new_obstacle_interval,
                    sensor_range=args.sensor_range)
    
    # Run the simulation
    start_time = time.time()
    test.run_simulation()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Save data if requested
    if args.save_trajectory:
        test.save_data(args.save_trajectory, args.save_path_history, args.save_map_history)
    
    # Visualize the results
    if args.animate:
        test.visualize_animation()
    else:
        test.visualize_static()

if __name__ == '__main__':
    main()
