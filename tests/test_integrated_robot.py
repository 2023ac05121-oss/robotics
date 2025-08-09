#!/usr/bin/env python3

"""
Integrated Robot Test Script

This script demonstrates the integration of Monte Carlo Localization,
Grid-based SLAM, and D* Planner in a complete robot navigation system.
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

# Import the integrated robot implementation
from src.integrated_robot import IntegratedRobot
from src.utils.robot_utils import create_artificial_map, simulate_robot_motion, simulate_sensor_readings

class IntegratedRobotTest:
    def __init__(self, map_size=(100, 100), num_particles=500, grid_size=1.0, 
                 motion_noise=(0.1, 0.1, 0.05), measurement_noise=0.1, 
                 max_steps=100, obstacle_count=10, dynamic_obstacles=True,
                 new_obstacle_interval=20, sensor_range=20, use_deep_learning=False,
                 model_path=None):
        self.map_size = map_size
        self.num_particles = num_particles
        self.grid_size = grid_size
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.max_steps = max_steps
        self.obstacle_count = obstacle_count
        self.dynamic_obstacles = dynamic_obstacles
        self.new_obstacle_interval = new_obstacle_interval
        self.sensor_range = sensor_range
        self.use_deep_learning = use_deep_learning
        self.model_path = model_path
        
        # Create artificial map with obstacles
        self.true_map, self.obstacles = create_artificial_map(map_size, obstacle_count)
        
        # Create binary map for D* (0 = free, 1 = obstacle)
        self.binary_map = np.ones(map_size, dtype=int)  # Initialize as all obstacle
        self.binary_map[self.true_map > 127] = 0  # Free space
        
        # Generate start and goal positions
        self.start_pos, self.goal_pos = self.generate_start_goal()
        
        # Robot's current position (x, y, theta)
        self.current_pos = np.array([self.start_pos[0], self.start_pos[1], 0.0])
        
        # Initialize the integrated robot
        self.robot = IntegratedRobot(
            map_size=map_size,
            num_particles=num_particles,
            grid_size=grid_size,
            motion_noise=motion_noise,
            measurement_noise=measurement_noise,
            sensor_range=sensor_range,
            use_deep_learning=use_deep_learning,
            model_path=model_path
        )
        
        # Set the robot's initial position and goal
        self.robot.set_position(self.current_pos)
        self.robot.set_goal(self.goal_pos)
        
        # Storage for trajectory and visualization data
        self.trajectory = [self.current_pos.copy()]
        self.estimated_trajectory = [self.current_pos.copy()]
        self.planned_paths = [self.robot.get_planned_path()]
        self.slam_maps = [self.robot.get_slam_map()]
        
        # Set up visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.suptitle('Integrated Robot Test')
    
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
        
        print(f"Added new obstacle at position {pos} with size {obstacle_size}x{obstacle_size}")
    
    def run_simulation(self):
        """Run the integrated robot simulation for the specified number of steps"""
        steps_since_last_obstacle = 0
        
        for step in range(self.max_steps):
            # Check if we've reached the goal
            current_cell = (int(self.current_pos[0]), int(self.current_pos[1]))
            goal_cell = (int(self.goal_pos[0]), int(self.goal_pos[1]))
            
            distance_to_goal = np.sqrt((current_cell[0] - goal_cell[0])**2 + 
                                       (current_cell[1] - goal_cell[1])**2)
            
            if distance_to_goal < 2.0:
                print(f"Goal reached in {step} steps!")
                break
            
            # Add dynamic obstacle periodically
            if self.dynamic_obstacles and steps_since_last_obstacle >= self.new_obstacle_interval:
                self.add_dynamic_obstacle()
                steps_since_last_obstacle = 0
            else:
                steps_since_last_obstacle += 1
            
            # Simulate sensor readings
            measurements = simulate_sensor_readings(self.current_pos, self.true_map, 
                                                  num_rays=36, max_range=50.0, noise=self.measurement_noise)
            
            # Get control command from the robot
            control = self.robot.get_control(measurements)
            
            # Apply the control command to the robot
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
            
            # Update the robot's state with the new position and measurements
            self.robot.update(control, measurements)
            
            # Store trajectory and visualization data
            self.trajectory.append(self.current_pos.copy())
            self.estimated_trajectory.append(self.robot.get_estimated_position())
            self.planned_paths.append(self.robot.get_planned_path())
            self.slam_maps.append(self.robot.get_slam_map())
            
            # Calculate position error
            est_pos = self.robot.get_estimated_position()
            position_error = np.sqrt((est_pos[0] - self.current_pos[0])**2 + 
                                    (est_pos[1] - self.current_pos[1])**2)
            
            print(f"Step {step+1}/{self.max_steps}: "
                  f"Position = ({self.current_pos[0]:.1f}, {self.current_pos[1]:.1f}), "
                  f"Estimated = ({est_pos[0]:.1f}, {est_pos[1]:.1f}), "
                  f"Error = {position_error:.2f}, "
                  f"Distance to Goal = {distance_to_goal:.1f}")
        
        return self.trajectory, self.estimated_trajectory, self.planned_paths, self.slam_maps
    
    def visualize_static(self):
        """Create a static visualization of the final state"""
        # Top-left subplot - True map and trajectory
        ax1 = self.axes[0, 0]
        
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
        ax1.plot(trajectory[:, 1], trajectory[:, 0], 'b-', linewidth=2, label='True Path')
        
        ax1.set_title('True Map and Trajectory')
        ax1.legend()
        
        # Top-right subplot - SLAM map
        ax2 = self.axes[0, 1]
        
        # Get the final SLAM map
        slam_map = self.slam_maps[-1]
        
        # Convert to visualization format
        slam_viz_map = np.ones((self.map_size[0], self.map_size[1], 3)) * 0.5  # Initialize as unknown (gray)
        slam_viz_map[slam_map < 0.45] = [1, 1, 1]  # Free space (white)
        slam_viz_map[slam_map > 0.55] = [0, 0, 0]  # Obstacles (black)
        
        # Mark the start and goal positions
        slam_viz_map[self.start_pos[0], self.start_pos[1]] = [0, 1, 0]  # Start (green)
        slam_viz_map[self.goal_pos[0], self.goal_pos[1]] = [1, 0, 0]  # Goal (red)
        
        ax2.imshow(slam_viz_map, origin='lower')
        
        # Plot the estimated trajectory
        est_trajectory = np.array(self.estimated_trajectory)
        ax2.plot(est_trajectory[:, 1], est_trajectory[:, 0], 'y-', linewidth=2, label='Estimated Path')
        
        ax2.set_title('SLAM Map and Estimated Trajectory')
        ax2.legend()
        
        # Bottom-left subplot - Localization particles
        ax3 = self.axes[1, 0]
        
        # Display the true map
        ax3.imshow(viz_map, origin='lower')
        
        # Get the final particles
        particles = self.robot.get_particles()
        
        # Plot the particles
        particle_positions = particles[:, :2]
        particle_weights = particles[:, 3]
        
        # Normalize weights for visualization
        max_weight = np.max(particle_weights)
        norm_weights = particle_weights / max_weight if max_weight > 0 else particle_weights
        
        # Use weights for point sizes
        sizes = 5 + 30 * norm_weights
        ax3.scatter(particle_positions[:, 1], particle_positions[:, 0], s=sizes, c='b', alpha=0.5, label='Particles')
        
        # Plot the estimated position
        est_pos = self.robot.get_estimated_position()
        ax3.plot(est_pos[1], est_pos[0], 'yo', markersize=10, label='Estimated Position')
        
        # Plot the true position
        ax3.plot(self.current_pos[1], self.current_pos[0], 'go', markersize=10, label='True Position')
        
        ax3.set_title('Monte Carlo Localization')
        ax3.legend()
        
        # Bottom-right subplot - Path planning
        ax4 = self.axes[1, 1]
        
        # Display the SLAM map
        ax4.imshow(slam_viz_map, origin='lower')
        
        # Get the final planned path
        planned_path = self.planned_paths[-1]
        
        # Plot the planned path
        if planned_path and len(planned_path) > 1:
            planned_path = np.array(planned_path)
            ax4.plot(planned_path[:, 1], planned_path[:, 0], 'y-', linewidth=2, label='Planned Path')
        
        # Plot the estimated position
        ax4.plot(est_pos[1], est_pos[0], 'yo', markersize=10, label='Current Position')
        
        # Plot the goal position
        ax4.plot(self.goal_pos[1], self.goal_pos[0], 'ro', markersize=10, label='Goal Position')
        
        ax4.set_title('D* Path Planning')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_animation(self):
        """Create an animated visualization of the integrated robot over time"""
        # Top-left subplot - True map and trajectory
        ax1 = self.axes[0, 0]
        
        # Create a visualization map
        viz_map = np.zeros((self.map_size[0], self.map_size[1], 3))  # RGB
        viz_map[self.binary_map == 0] = [1, 1, 1]  # Free space (white)
        viz_map[self.binary_map == 1] = [0, 0, 0]  # Obstacles (black)
        
        # Mark the start and goal positions
        viz_map[self.start_pos[0], self.start_pos[1]] = [0, 1, 0]  # Start (green)
        viz_map[self.goal_pos[0], self.goal_pos[1]] = [1, 0, 0]  # Goal (red)
        
        # Display the true map
        true_map_img = ax1.imshow(viz_map, origin='lower')
        
        # Initialize true trajectory line
        true_traj_line, = ax1.plot([], [], 'b-', linewidth=2, label='True Path')
        
        # Initialize robot true position
        robot_true, = ax1.plot([], [], 'go', markersize=10, label='Robot')
        
        ax1.set_title('True Map and Trajectory')
        ax1.legend()
        
        # Top-right subplot - SLAM map
        ax2 = self.axes[0, 1]
        
        # Initialize SLAM map image
        slam_map_img = ax2.imshow(np.ones((self.map_size[0], self.map_size[1], 3)) * 0.5, origin='lower')
        
        # Initialize estimated trajectory line
        est_traj_line, = ax2.plot([], [], 'y-', linewidth=2, label='Estimated Path')
        
        # Initialize robot estimated position
        robot_est, = ax2.plot([], [], 'yo', markersize=10, label='Estimated Position')
        
        ax2.set_title('SLAM Map and Estimated Trajectory')
        ax2.legend()
        
        # Bottom-left subplot - Localization particles
        ax3 = self.axes[1, 0]
        
        # Display the true map
        ax3.imshow(viz_map, origin='lower')
        
        # Initialize particles scatter plot
        particles_scatter = ax3.scatter([], [], s=5, c='b', alpha=0.5, label='Particles')
        
        # Initialize robot true and estimated positions
        robot_true2, = ax3.plot([], [], 'go', markersize=10, label='True Position')
        robot_est2, = ax3.plot([], [], 'yo', markersize=10, label='Estimated Position')
        
        ax3.set_title('Monte Carlo Localization')
        ax3.legend()
        
        # Bottom-right subplot - Path planning
        ax4 = self.axes[1, 1]
        
        # Initialize SLAM map image
        slam_map_img2 = ax4.imshow(np.ones((self.map_size[0], self.map_size[1], 3)) * 0.5, origin='lower')
        
        # Initialize planned path line
        planned_path_line, = ax4.plot([], [], 'y-', linewidth=2, label='Planned Path')
        
        # Initialize robot and goal positions
        robot_pos, = ax4.plot([], [], 'yo', markersize=10, label='Current Position')
        goal_pos, = ax4.plot(self.goal_pos[1], self.goal_pos[0], 'ro', markersize=10, label='Goal Position')
        
        ax4.set_title('D* Path Planning')
        ax4.legend()
        
        def init():
            true_traj_line.set_data([], [])
            robot_true.set_data([], [])
            
            est_traj_line.set_data([], [])
            robot_est.set_data([], [])
            
            particles_scatter.set_offsets(np.empty((0, 2)))
            robot_true2.set_data([], [])
            robot_est2.set_data([], [])
            
            planned_path_line.set_data([], [])
            robot_pos.set_data([], [])
            
            return (true_map_img, true_traj_line, robot_true, 
                   slam_map_img, est_traj_line, robot_est,
                   particles_scatter, robot_true2, robot_est2,
                   slam_map_img2, planned_path_line, robot_pos, goal_pos)
        
        def update(frame):
            # Update true trajectory and position
            true_trajectory = np.array(self.trajectory[:frame+1])
            true_traj_line.set_data(true_trajectory[:, 1], true_trajectory[:, 0])
            robot_true.set_data(self.trajectory[frame][1], self.trajectory[frame][0])
            
            # Update SLAM map
            slam_map = self.slam_maps[frame]
            
            # Convert to visualization format
            slam_viz_map = np.ones((self.map_size[0], self.map_size[1], 3)) * 0.5  # Initialize as unknown (gray)
            slam_viz_map[slam_map < 0.45] = [1, 1, 1]  # Free space (white)
            slam_viz_map[slam_map > 0.55] = [0, 0, 0]  # Obstacles (black)
            
            # Mark the start and goal positions
            slam_viz_map[self.start_pos[0], self.start_pos[1]] = [0, 1, 0]  # Start (green)
            slam_viz_map[self.goal_pos[0], self.goal_pos[1]] = [1, 0, 0]  # Goal (red)
            
            slam_map_img.set_array(slam_viz_map)
            slam_map_img2.set_array(slam_viz_map)
            
            # Update estimated trajectory and position
            est_trajectory = np.array(self.estimated_trajectory[:frame+1])
            est_traj_line.set_data(est_trajectory[:, 1], est_trajectory[:, 0])
            robot_est.set_data(self.estimated_trajectory[frame][1], self.estimated_trajectory[frame][0])
            
            # Update particles
            if frame < len(self.trajectory):
                # Get the particles at this frame
                self.robot.set_position(self.trajectory[frame])
                particles = self.robot.get_particles()
                
                # Update particles scatter plot
                particle_positions = particles[:, :2]
                particle_weights = particles[:, 3]
                
                # Normalize weights for visualization
                max_weight = np.max(particle_weights)
                norm_weights = particle_weights / max_weight if max_weight > 0 else particle_weights
                
                # Use weights for point sizes
                sizes = 5 + 30 * norm_weights
                
                # Convert to xy format for scatter plot
                xy_particles = np.column_stack((particle_positions[:, 1], particle_positions[:, 0]))
                
                particles_scatter.set_offsets(xy_particles)
                particles_scatter.set_sizes(sizes)
                
                # Update robot positions in MCL subplot
                robot_true2.set_data(self.trajectory[frame][1], self.trajectory[frame][0])
                robot_est2.set_data(self.estimated_trajectory[frame][1], self.estimated_trajectory[frame][0])
                
                # Update planned path
                planned_path = self.planned_paths[frame]
                if planned_path and len(planned_path) > 1:
                    planned_path = np.array(planned_path)
                    planned_path_line.set_data(planned_path[:, 1], planned_path[:, 0])
                else:
                    planned_path_line.set_data([], [])
                
                # Update robot position in planning subplot
                robot_pos.set_data(self.estimated_trajectory[frame][1], self.estimated_trajectory[frame][0])
            
            # Update titles
            ax1.set_title(f'True Map and Trajectory (Step {frame+1})')
            ax2.set_title(f'SLAM Map and Estimated Trajectory (Step {frame+1})')
            
            # Calculate position error
            if frame < len(self.trajectory):
                position_error = np.sqrt((self.estimated_trajectory[frame][0] - self.trajectory[frame][0])**2 + 
                                       (self.estimated_trajectory[frame][1] - self.trajectory[frame][1])**2)
                
                ax3.set_title(f'Monte Carlo Localization (Error: {position_error:.2f})')
            
            ax4.set_title(f'D* Path Planning (Step {frame+1})')
            
            return (true_map_img, true_traj_line, robot_true, 
                   slam_map_img, est_traj_line, robot_est,
                   particles_scatter, robot_true2, robot_est2,
                   slam_map_img2, planned_path_line, robot_pos, goal_pos)
        
        anim = FuncAnimation(self.fig, update, frames=len(self.trajectory),
                             init_func=init, blit=True, interval=100)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def save_data(self, true_trajectory_path, est_trajectory_path, 
                 planned_paths_path, slam_maps_path):
        """Save trajectory and visualization data to files"""
        # Save true trajectory
        np.save(true_trajectory_path, np.array(self.trajectory))
        
        # Save estimated trajectory
        np.save(est_trajectory_path, np.array(self.estimated_trajectory))
        
        # Save planned paths (convert to numpy array, padded to max length)
        max_path_length = max(len(path) for path in self.planned_paths if path) if self.planned_paths else 0
        path_array = np.zeros((len(self.planned_paths), max_path_length, 2), dtype=int)
        
        for i, path in enumerate(self.planned_paths):
            if path and len(path) > 0:
                path_array[i, :len(path)] = path
        
        np.save(planned_paths_path, path_array)
        
        # Save SLAM maps
        np.save(slam_maps_path, np.array(self.slam_maps))
        
        print(f"True trajectory saved to {true_trajectory_path}")
        print(f"Estimated trajectory saved to {est_trajectory_path}")
        print(f"Planned paths saved to {planned_paths_path}")
        print(f"SLAM maps saved to {slam_maps_path}")

def main():
    parser = argparse.ArgumentParser(description='Integrated Robot Test Script')
    parser.add_argument('--map_size', type=int, nargs=2, default=[100, 100], help='Map size (width, height)')
    parser.add_argument('--num_particles', type=int, default=500, help='Number of particles for MCL')
    parser.add_argument('--grid_size', type=float, default=1.0, help='Grid cell size for SLAM')
    parser.add_argument('--motion_noise', type=float, nargs=3, default=[0.1, 0.1, 0.05], 
                        help='Motion noise parameters (x, y, theta)')
    parser.add_argument('--measurement_noise', type=float, default=0.1, help='Measurement noise parameter')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum simulation steps')
    parser.add_argument('--obstacle_count', type=int, default=10, help='Number of obstacles in the map')
    parser.add_argument('--dynamic_obstacles', action='store_true', help='Enable dynamic obstacles')
    parser.add_argument('--new_obstacle_interval', type=int, default=20, 
                        help='Interval (in steps) between adding new obstacles')
    parser.add_argument('--sensor_range', type=int, default=20, help='Robot sensor range')
    parser.add_argument('--use_deep_learning', action='store_true', help='Use deep learning enhanced MCL')
    parser.add_argument('--model_path', type=str, help='Path to pre-trained deep learning model')
    parser.add_argument('--animate', action='store_true', help='Create an animation instead of a static plot')
    parser.add_argument('--save_true_trajectory', type=str, help='Path to save the true trajectory data (.npy)')
    parser.add_argument('--save_est_trajectory', type=str, help='Path to save the estimated trajectory data (.npy)')
    parser.add_argument('--save_planned_paths', type=str, help='Path to save the planned paths data (.npy)')
    parser.add_argument('--save_slam_maps', type=str, help='Path to save the SLAM maps data (.npy)')
    
    args = parser.parse_args()
    
    # Create and run the test
    test = IntegratedRobotTest(
        map_size=args.map_size, 
        num_particles=args.num_particles,
        grid_size=args.grid_size,
        motion_noise=tuple(args.motion_noise),
        measurement_noise=args.measurement_noise,
        max_steps=args.max_steps,
        obstacle_count=args.obstacle_count,
        dynamic_obstacles=args.dynamic_obstacles,
        new_obstacle_interval=args.new_obstacle_interval,
        sensor_range=args.sensor_range,
        use_deep_learning=args.use_deep_learning,
        model_path=args.model_path
    )
    
    # Run the simulation
    start_time = time.time()
    test.run_simulation()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Save data if requested
    if args.save_true_trajectory:
        test.save_data(
            args.save_true_trajectory,
            args.save_est_trajectory,
            args.save_planned_paths,
            args.save_slam_maps
        )
    
    # Visualize the results
    if args.animate:
        test.visualize_animation()
    else:
        test.visualize_static()

if __name__ == '__main__':
    main()
