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
        
        # Try to plan initial path
        self.path = self.initialize_planning()
        
        # Dynamic obstacles list
        self.dynamic_obstacles_list = []
        
        # Storage for trajectory and path data
        self.trajectory = [self.current_pos.copy()]
        self.path_history = [self.path.copy() if self.path else []]
        self.known_map_history = [self.known_map.copy()]
        
        # Set up visualization
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        plt.suptitle('D* Path Planning Test')
    
    def initialize_planning(self):
        """Initialize the D* planner and get the initial path"""
        try:
            # Plan initial path
            print("Calling dstar.init...")
            self.dstar.init(self.start_pos[0], self.start_pos[1], self.goal_pos[0], self.goal_pos[1])
            print("Calling dstar.plan...")
            path = self.dstar.plan()
            print(f"Initial path: {path[:5] if path and len(path) > 5 else path}")
            return path
        except Exception as e:
            print(f"Error in path planning: {str(e)}")
            return []
    
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
        
        # Track the affected area
        min_x = max(0, pos[0] - obstacle_size//2)
        min_y = max(0, pos[1] - obstacle_size//2)
        max_x = min(self.map_size[0], pos[0] + obstacle_size//2 + 1)
        max_y = min(self.map_size[1], pos[1] + obstacle_size//2 + 1)
        
        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                self.binary_map[i, j] = 1  # Set as obstacle
                
                # If within sensor range, update the known map too
                distance = np.sqrt((i - self.current_pos[0])**2 + (j - self.current_pos[1])**2)
                if distance <= self.sensor_range:
                    self.known_map[i, j] = 1
        
        print(f"Added new obstacle at position {pos} with size {obstacle_size}x{obstacle_size}")
        
        # Add to the list of dynamic obstacles
        self.dynamic_obstacles_list.append((pos, obstacle_size))
        
        # Return the affected area
        return (min_x, min_y, max_x, max_y)
    
    def run_simulation(self):
        """Run the D* planning simulation for the specified number of steps"""
        steps_since_last_obstacle = 0
        
        # Limit steps for faster animation
        actual_max_steps = min(self.max_steps, 50)
        print(f"Running simulation for {actual_max_steps} steps (max was {self.max_steps})")
        print(f"Start position: {self.start_pos}, Goal position: {self.goal_pos}")
        print(f"Current position: {self.current_pos}")
        print("Initializing path...")
        
        # Make sure we have a valid path
        if not self.path or len(self.path) <= 1:
            print("Error: Initial path is empty or too short!")
            print(f"Path: {self.path}")
            # Add a minimum path to ensure we have something to visualize
            self.path = [self.start_pos]
            if self.start_pos != self.goal_pos:
                self.path.append(self.goal_pos)
            
            # Add the current position and path to the history
            self.trajectory = [self.current_pos.copy()]
            self.path_history = [self.path.copy()]
            self.known_map_history = [self.known_map.copy()]
            return self.trajectory, self.path_history, self.known_map_history
        
        return self._run_simulation_steps(actual_max_steps, steps_since_last_obstacle)
    
    def _run_simulation_steps(self, actual_max_steps, steps_since_last_obstacle):
        """Helper method to run the simulation steps"""
        for step in range(actual_max_steps):
            # Check if we've reached the goal
            current_cell = (int(self.current_pos[0]), int(self.current_pos[1]))
            if current_cell[0] == self.goal_pos[0] and current_cell[1] == self.goal_pos[1]:
                print(f"Goal reached in {step} steps!")
                break
            
            # Add dynamic obstacle periodically
            if self.dynamic_obstacles and steps_since_last_obstacle >= self.new_obstacle_interval:
                steps_since_last_obstacle = self._handle_dynamic_obstacle()
            else:
                steps_since_last_obstacle += 1
            
            # Get next waypoint and move robot
            success = self._move_robot_step()
            if not success:
                break
            
            # Update the known map with new sensor information
            self.update_known_map()
            
            # Check if the robot needs to replan due to new map information
            if step % 5 == 0:  # Replan periodically to handle map updates
                try:
                    self.dstar.update_known_map(self.known_map)
                    self.path = self.dstar.plan()
                except Exception as e:
                    print(f"Error during periodic replanning: {e}")
                    # Continue with the current path
            
            # Store trajectory and path data
            self.trajectory.append(self.current_pos.copy())
            self.path_history.append(self.path.copy() if self.path else [])
            self.known_map_history.append(self.known_map.copy())
            
            print(f"Step {step+1}/{self.max_steps}: Position = ({self.current_pos[0]:.1f}, {self.current_pos[1]:.1f}), "
                  f"Distance to Goal = {np.sqrt((self.current_pos[0] - self.goal_pos[0])**2 + (self.current_pos[1] - self.goal_pos[1])**2):.1f}")
        
        return self.trajectory, self.path_history, self.known_map_history
    
    def _handle_dynamic_obstacle(self):
        """Add a dynamic obstacle and handle replanning"""
        affected_area = self.add_dynamic_obstacle()
        
        # Update D* with the new obstacle information
        # This will trigger replanning if the obstacle affects the current path
        try:
            self.dstar.update_cell(affected_area)
            self.path = self.dstar.plan()
        except Exception as e:
            print(f"Error during replanning: {e}")
            # Continue with the current path
        
        return 0  # Reset counter
    
    def _move_robot_step(self):
        """Move the robot one step toward the next waypoint"""
        # Get next waypoint from the path
        if self.path and len(self.path) > 1:
            next_waypoint = self.path[1]  # path[0] is current position
        else:
            print("No valid path to goal found!")
            return False
        
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
        
        return True
    
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
        print(f"Creating animation with {len(self.trajectory)} frames...")
        
        if len(self.trajectory) <= 1:
            print("Error: Not enough trajectory points for animation. Run the simulation first.")
            return None
            
        print("Trajectory data:")
        for i, pos in enumerate(self.trajectory[:5]):
            print(f"  Frame {i}: {pos}")
            
        # Set up visualization
        self._setup_animation_axes()
        
        # Create animation with a faster interval for better performance
        print("Creating FuncAnimation object...")
        anim = FuncAnimation(self.fig, self._update_animation_frame, 
                            frames=min(len(self.trajectory), 30),  # Limit to 30 frames max for performance
                            init_func=self._init_animation, 
                            blit=True, interval=300)  # Slower interval for better performance
        
        # Add a print statement to show we're about to display the animation
        print(f"Animation created with {min(len(self.trajectory), 30)} frames. Displaying now...")
        
        plt.tight_layout()
        
        # Try different ways to display the animation based on the platform
        try:
            # Method 1: Interactive mode with pause
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)
            plt.pause(0.5)  # Give it time to render
            plt.ioff()  # Turn off interactive mode
            
            # Keep the figure open
            print("Animation displayed. Close the window to continue.")
            plt.show(block=True)
        except Exception as e:
            print(f"Error with interactive mode: {e}")
            try:
                # Method 2: Simple blocking show
                plt.show()
            except Exception as e2:
                print(f"Error displaying animation: {e2}")
        
        print("Animation display completed.")
        return anim
    
    def _setup_animation_axes(self):
        """Setup the axes for animation"""
        # Left subplot - True map and trajectory
        ax1 = self.axes[0]
        
        # Create a visualization map
        viz_map = np.zeros((self.map_size[0], self.map_size[1], 3))  # RGB
        viz_map[self.binary_map == 0] = [1, 1, 1]  # Free space (white)
        viz_map[self.binary_map == 1] = [0, 0, 0]  # Obstacles (black)
        
        # Mark the start and goal positions
        start_y, start_x = self.start_pos[0], self.start_pos[1]
        goal_y, goal_x = self.goal_pos[0], self.goal_pos[1]
        
        # Make sure coordinates are within bounds
        if 0 <= start_y < viz_map.shape[0] and 0 <= start_x < viz_map.shape[1]:
            viz_map[start_y, start_x] = [0, 1, 0]  # Start (green)
            
        if 0 <= goal_y < viz_map.shape[0] and 0 <= goal_x < viz_map.shape[1]:
            viz_map[goal_y, goal_x] = [1, 0, 0]  # Goal (red)
        
        # Display the true map
        self.true_map_img = ax1.imshow(viz_map, origin='lower')
        
        # Initialize trajectory line
        self.traj_line, = ax1.plot([], [], 'b-', linewidth=2, label='Robot Path')
        
        # Initialize robot position
        self.robot_pos, = ax1.plot([], [], 'yo', markersize=10, label='Robot')
        
        ax1.set_title('True Map and Trajectory')
        ax1.legend()
        
        # Right subplot - Known map and planned path
        ax2 = self.axes[1]
        
        # Initialize known map image
        self.known_map_img = ax2.imshow(np.zeros((self.map_size[0], self.map_size[1], 3)), origin='lower')
        
        # Initialize planned path line
        self.path_line, = ax2.plot([], [], 'y-', linewidth=2, label='Planned Path')
        
        ax2.set_title('Known Map and Planned Path')
        ax2.legend()
    
    def _init_animation(self):
        """Initialize the animation"""
        self.traj_line.set_data([], [])
        self.robot_pos.set_data([], [])
        self.path_line.set_data([], [])
        return self.true_map_img, self.traj_line, self.robot_pos, self.known_map_img, self.path_line
    
    def _update_animation_frame(self, frame):
        """Update a single frame of the animation"""
        # Debug print to track animation progress
        if frame % 5 == 0:
            print(f"Rendering animation frame {frame}/{len(self.trajectory)-1}")
        
        try:
            # Check if frame index is valid
            if frame >= len(self.trajectory):
                print(f"Warning: Frame index {frame} exceeds trajectory length {len(self.trajectory)}")
                frame = len(self.trajectory) - 1
            
            # Update trajectory
            if frame > 0:  # Only plot if we have at least 2 points
                trajectory = np.array(self.trajectory[:frame+1])
                # Note: Swapping coordinates for plotting (x,y on plot vs y,x in array)
                self.traj_line.set_data(trajectory[:, 1], trajectory[:, 0])
                
                # Update robot position
                self.robot_pos.set_data([self.trajectory[frame][1]], [self.trajectory[frame][0]])
            else:
                # Just plot the first point
                self.traj_line.set_data([self.trajectory[0][1]], [self.trajectory[0][0]])
                self.robot_pos.set_data([self.trajectory[0][1]], [self.trajectory[0][0]])
            
            # Make sure we have enough map history
            if frame < len(self.known_map_history):
                # Update known map visualization
                known_viz_map = self._get_known_map_visualization(frame)
                self.known_map_img.set_array(known_viz_map)
            
                # Update planned path
                self._update_path_visualization(frame)
                
                # Update titles
                self.axes[0].set_title(f'True Map and Trajectory (Step {frame+1})')
                self.axes[1].set_title(f'Known Map and Planned Path (Step {frame+1})')
            else:
                print(f"Warning: Not enough map history for frame {frame}")
            
            return self.true_map_img, self.traj_line, self.robot_pos, self.known_map_img, self.path_line
        except Exception as e:
            print(f"Error in animation frame {frame}: {e}")
            import traceback
            traceback.print_exc()
            return self.true_map_img, self.traj_line, self.robot_pos, self.known_map_img, self.path_line
    
    def _get_known_map_visualization(self, frame):
        """Create visualization for the known map at a specific frame"""
        known_map = self.known_map_history[frame]
        
        # Create a visualization of the known map
        known_viz_map = np.zeros((self.map_size[0], self.map_size[1], 3))  # RGB
        known_viz_map[known_map == 0] = [1, 1, 1]  # Free space (white)
        known_viz_map[known_map == 1] = [0, 0, 0]  # Obstacles (black)
        
        # Mark unexplored areas
        unexplored = (known_map != 0) & (known_map != 1)
        known_viz_map[unexplored] = [0.7, 0.7, 0.7]  # Unexplored (gray)
        
        # Mark the start and goal positions
        start_y, start_x = self.start_pos[0], self.start_pos[1]
        goal_y, goal_x = self.goal_pos[0], self.goal_pos[1]
        
        if 0 <= start_y < known_viz_map.shape[0] and 0 <= start_x < known_viz_map.shape[1]:
            known_viz_map[start_y, start_x] = [0, 1, 0]  # Start (green)
            
        if 0 <= goal_y < known_viz_map.shape[0] and 0 <= goal_x < known_viz_map.shape[1]:
            known_viz_map[goal_y, goal_x] = [1, 0, 0]  # Goal (red)
        
        return known_viz_map
    
    def _update_path_visualization(self, frame):
        """Update the path visualization for a specific frame"""
        try:
            # Make sure we have enough path history data
            if frame < len(self.path_history):
                path = self.path_history[frame]
                if path and len(path) > 1:
                    path = np.array(path)
                    # Note: Swapping coordinates for plotting
                    self.path_line.set_data(path[:, 1], path[:, 0])
                else:
                    self.path_line.set_data([], [])
            else:
                print(f"Warning: Not enough path history for frame {frame}")
                self.path_line.set_data([], [])
        except Exception as e:
            print(f"Error in path visualization for frame {frame}: {e}")
            self.path_line.set_data([], [])
    
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

def setup_arg_parser():
    """Set up and return the argument parser"""
    parser = argparse.ArgumentParser(description='D* Path Planning Test Script')
    parser.add_argument('--map_size', type=int, nargs=2, default=[50, 50], help='Map size (width, height)')
    parser.add_argument('--motion_noise', type=float, nargs=3, default=[0.1, 0.1, 0.05], 
                        help='Motion noise parameters (x, y, theta)')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum simulation steps')
    parser.add_argument('--obstacle_count', type=int, default=5, help='Number of obstacles in the map')
    parser.add_argument('--dynamic_obstacles', action='store_true', help='Enable dynamic obstacles')
    parser.add_argument('--new_obstacle_interval', type=int, default=10, 
                        help='Interval (in steps) between adding new obstacles')
    parser.add_argument('--sensor_range', type=int, default=20, help='Robot sensor range')
    parser.add_argument('--animate', action='store_true', help='Create an animation instead of a static plot')
    parser.add_argument('--save_trajectory', type=str, help='Path to save the trajectory data (.npy)')
    parser.add_argument('--save_path_history', type=str, help='Path to save the path history data (.npy)')
    parser.add_argument('--save_map_history', type=str, help='Path to save the known map history data (.npy)')
    parser.add_argument('--smaller_map', action='store_true', help='Use a smaller map for quicker testing (25x25)')
    return parser

def setup_matplotlib():
    """Configure matplotlib backend"""
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        print("Using TkAgg backend for matplotlib")
    except Exception as e:
        print(f"Could not set matplotlib backend: {e}")

def handle_invalid_trajectory(test, trajectory, path_history):
    """Handle cases where the simulation did not produce a valid trajectory"""
    print("Warning: Simulation did not produce a valid trajectory.")
    
    # Add minimal trajectory for visualization
    if not trajectory:
        trajectory = [test.current_pos.copy()]
        test.trajectory = trajectory
    if len(trajectory) == 1:
        # Add a second point to make the animation work
        second_point = trajectory[0].copy()
        second_point[0] += 1  # Move one step
        trajectory.append(second_point)
        test.trajectory = trajectory
    
    # Make sure we have path history
    if not path_history or not path_history[0]:
        path_history = [[test.start_pos, test.goal_pos]]
        test.path_history = path_history
    
    print("Created minimal trajectory for visualization.")
    return trajectory, path_history

def visualize_results(test, args):
    """Visualize the simulation results"""
    if args.animate:
        print("Starting animation...")
        try:
            test.visualize_animation()
        except Exception as e:
            print(f"Error during animation: {e}")
            # Fall back to static visualization
            print("Falling back to static visualization...")
            test.visualize_static()
    else:
        test.visualize_static()

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Override map size if smaller_map option is selected
    if args.smaller_map:
        args.map_size = [25, 25]
        args.obstacle_count = 3
        args.sensor_range = 10
        print("Using smaller map (25x25) for quicker testing")
    
    print("D* Path Planning Test")
    print(f"Map size: {args.map_size}")
    print(f"Obstacle count: {args.obstacle_count}")
    print(f"Animation: {'Enabled' if args.animate else 'Disabled'}")
    
    # Set up matplotlib backend
    setup_matplotlib()
    
    # Create and run the test
    test = DStarTest(map_size=tuple(args.map_size), 
                    motion_noise=tuple(args.motion_noise),
                    max_steps=args.max_steps,
                    obstacle_count=args.obstacle_count,
                    dynamic_obstacles=args.dynamic_obstacles,
                    new_obstacle_interval=args.new_obstacle_interval,
                    sensor_range=args.sensor_range)
    
    # Run the simulation
    start_time = time.time()
    trajectory, path_history, _ = test.run_simulation()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Handle invalid trajectory if needed
    if not trajectory or len(trajectory) <= 1:
        trajectory, path_history = handle_invalid_trajectory(test, trajectory, path_history)
    
    # Save data if requested
    if args.save_trajectory and trajectory:
        test.save_data(args.save_trajectory, 
                      args.save_path_history or "path_history.npy", 
                      args.save_map_history or "map_history.npy")
    
    # Visualize the results
    visualize_results(test, args)

if __name__ == '__main__':
    main()
