#!/usr/bin/env python3

"""
MCL Visualization Tool

This script provides visualization capabilities for the Monte Carlo Localization algorithm.
It can be used to visualize both the Python implementation and the ROS2 implementation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import json
import os
import sys

class MCLVisualizer:
    def __init__(self, map_path, trajectory_path=None, particles_path=None):
        self.map_path = map_path
        self.trajectory_path = trajectory_path
        self.particles_path = particles_path
        
        # Load map data
        self.map_data = self.load_map()
        
        # Load trajectory if available
        self.trajectory = None
        if trajectory_path:
            self.trajectory = self.load_trajectory()
        
        # Load particles if available
        self.particles_data = None
        if particles_path:
            self.particles_data = self.load_particles()
        
        # Setup visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.set_window_title('Monte Carlo Localization Visualization')
        
    def load_map(self):
        """Load map data from file"""
        # Check file extension
        _, ext = os.path.splitext(self.map_path)
        
        if ext == '.npy':
            # Load NumPy binary file
            return np.load(self.map_path)
        elif ext == '.pgm':
            # Load PGM image file (common ROS map format)
            from PIL import Image
            img = Image.open(self.map_path)
            return np.array(img)
        elif ext == '.yaml':
            # Load ROS map YAML file
            import yaml
            with open(self.map_path, 'r') as f:
                map_data = yaml.safe_load(f)
            
            # The YAML file references a PGM file
            map_pgm_path = os.path.join(os.path.dirname(self.map_path), map_data['image'])
            from PIL import Image
            img = Image.open(map_pgm_path)
            return np.array(img)
        else:
            # Try to load as text file with numbers
            try:
                return np.loadtxt(self.map_path)
            except:
                raise ValueError(f"Unsupported map file format: {ext}")
    
    def load_trajectory(self):
        """Load robot trajectory data"""
        _, ext = os.path.splitext(self.trajectory_path)
        
        if ext == '.npy':
            return np.load(self.trajectory_path)
        elif ext == '.json':
            with open(self.trajectory_path, 'r') as f:
                data = json.load(f)
            
            # Convert to numpy array
            if isinstance(data, list):
                # Check if it's a list of positions or a list of dictionaries
                if data and isinstance(data[0], dict):
                    # Convert list of dicts to numpy array
                    return np.array([[pos.get('x', 0), pos.get('y', 0), pos.get('theta', 0)] 
                                     for pos in data])
                else:
                    return np.array(data)
            else:
                raise ValueError("JSON file must contain a list of positions")
        else:
            # Try to load as text file with numbers
            try:
                return np.loadtxt(self.trajectory_path)
            except:
                raise ValueError(f"Unsupported trajectory file format: {ext}")
    
    def load_particles(self):
        """Load particle data"""
        _, ext = os.path.splitext(self.particles_path)
        
        if ext == '.npy':
            return np.load(self.particles_path)
        elif ext == '.json':
            with open(self.particles_path, 'r') as f:
                data = json.load(f)
            
            # Convert to numpy array
            if isinstance(data, list) and isinstance(data[0], list):
                # It's a list of time steps, each containing a list of particles
                return [np.array([[p.get('x', 0), p.get('y', 0), p.get('theta', 0), p.get('weight', 1.0)] 
                                 if isinstance(p, dict) else p for p in step]) 
                        for step in data]
            else:
                raise ValueError("JSON file must contain a list of time steps with particles")
        else:
            # Try to load as text file with numbers
            try:
                data = np.loadtxt(self.particles_path)
                # Reshape if it's a single time step
                if len(data.shape) == 2:
                    return [data]
                else:
                    # It's multiple time steps
                    return [data[i] for i in range(data.shape[0])]
            except:
                raise ValueError(f"Unsupported particles file format: {ext}")
    
    def visualize_static(self):
        """Create a static visualization of the MCL results"""
        # Display the map
        self.ax.imshow(self.map_data, cmap='gray')
        
        # Plot the trajectory if available
        if self.trajectory is not None:
            self.ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], 'r-', linewidth=2, label='Robot Path')
            
            # Plot start and end positions
            self.ax.plot(self.trajectory[0, 0], self.trajectory[0, 1], 'go', markersize=10, label='Start')
            self.ax.plot(self.trajectory[-1, 0], self.trajectory[-1, 1], 'bo', markersize=10, label='End')
        
        # Plot the particles if available (last time step)
        if self.particles_data is not None:
            particles = self.particles_data[-1]
            
            # Extract positions and weights
            if particles.shape[1] >= 4:  # Has weights
                positions = particles[:, :2]
                weights = particles[:, 3]
                
                # Normalize weights for visualization
                max_weight = np.max(weights)
                norm_weights = weights / max_weight if max_weight > 0 else weights
                
                # Use weights for point sizes
                sizes = 5 + 30 * norm_weights
                self.ax.scatter(positions[:, 0], positions[:, 1], s=sizes, c='b', alpha=0.5, label='Particles')
            else:  # No weights
                positions = particles[:, :2]
                self.ax.scatter(positions[:, 0], positions[:, 1], s=5, c='b', alpha=0.5, label='Particles')
        
        self.ax.set_title('Monte Carlo Localization Results')
        self.ax.legend()
        plt.tight_layout()
        plt.show()
    
    def visualize_animation(self):
        """Create an animated visualization of the MCL results over time"""
        # This requires both trajectory and particles data over time
        if self.trajectory is None or self.particles_data is None:
            print("Animation requires both trajectory and particles data over time")
            return self.visualize_static()
        
        # Ensure we have matching time steps
        num_steps = min(len(self.trajectory), len(self.particles_data))
        
        # Display the map
        map_img = self.ax.imshow(self.map_data, cmap='gray')
        
        # Initialize trajectory line
        line, = self.ax.plot([], [], 'r-', linewidth=2, label='Robot Path')
        
        # Initialize particles scatter plot
        particles_scatter = self.ax.scatter([], [], s=5, c='b', alpha=0.5, label='Particles')
        
        # Initialize robot position
        robot_pos, = self.ax.plot([], [], 'go', markersize=10, label='Robot')
        
        # Set up the plot
        self.ax.set_title('Monte Carlo Localization Animation')
        self.ax.legend()
        
        def init():
            line.set_data([], [])
            particles_scatter.set_offsets(np.empty((0, 2)))
            robot_pos.set_data([], [])
            return line, particles_scatter, robot_pos
        
        def update(frame):
            # Update trajectory
            line.set_data(self.trajectory[:frame+1, 0], self.trajectory[:frame+1, 1])
            
            # Update robot position
            robot_pos.set_data(self.trajectory[frame, 0], self.trajectory[frame, 1])
            
            # Update particles
            particles = self.particles_data[frame]
            positions = particles[:, :2]
            
            if particles.shape[1] >= 4:  # Has weights
                weights = particles[:, 3]
                max_weight = np.max(weights)
                norm_weights = weights / max_weight if max_weight > 0 else weights
                sizes = 5 + 30 * norm_weights
                particles_scatter.set_sizes(sizes)
            
            particles_scatter.set_offsets(positions)
            
            return line, particles_scatter, robot_pos
        
        anim = FuncAnimation(self.fig, update, frames=num_steps, 
                             init_func=init, blit=True, interval=100)
        
        plt.tight_layout()
        plt.show()
    
    def save_animation(self, output_path):
        """Save the animation to a file"""
        # This requires both trajectory and particles data over time
        if self.trajectory is None or self.particles_data is None:
            print("Animation requires both trajectory and particles data over time")
            return
        
        # Ensure we have matching time steps
        num_steps = min(len(self.trajectory), len(self.particles_data))
        
        # Display the map
        map_img = self.ax.imshow(self.map_data, cmap='gray')
        
        # Initialize trajectory line
        line, = self.ax.plot([], [], 'r-', linewidth=2, label='Robot Path')
        
        # Initialize particles scatter plot
        particles_scatter = self.ax.scatter([], [], s=5, c='b', alpha=0.5, label='Particles')
        
        # Initialize robot position
        robot_pos, = self.ax.plot([], [], 'go', markersize=10, label='Robot')
        
        # Set up the plot
        self.ax.set_title('Monte Carlo Localization Animation')
        self.ax.legend()
        
        def init():
            line.set_data([], [])
            particles_scatter.set_offsets(np.empty((0, 2)))
            robot_pos.set_data([], [])
            return line, particles_scatter, robot_pos
        
        def update(frame):
            # Update trajectory
            line.set_data(self.trajectory[:frame+1, 0], self.trajectory[:frame+1, 1])
            
            # Update robot position
            robot_pos.set_data(self.trajectory[frame, 0], self.trajectory[frame, 1])
            
            # Update particles
            particles = self.particles_data[frame]
            positions = particles[:, :2]
            
            if particles.shape[1] >= 4:  # Has weights
                weights = particles[:, 3]
                max_weight = np.max(weights)
                norm_weights = weights / max_weight if max_weight > 0 else weights
                sizes = 5 + 30 * norm_weights
                particles_scatter.set_sizes(sizes)
            
            particles_scatter.set_offsets(positions)
            
            return line, particles_scatter, robot_pos
        
        anim = FuncAnimation(self.fig, update, frames=num_steps, 
                             init_func=init, blit=True, interval=100)
        
        # Save animation
        anim.save(output_path, writer='ffmpeg', fps=10, dpi=100)
        print(f"Animation saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='MCL Visualization Tool')
    parser.add_argument('--map', required=True, help='Path to the map file (.npy, .pgm, or .yaml)')
    parser.add_argument('--trajectory', help='Path to the trajectory data file (.npy or .json)')
    parser.add_argument('--particles', help='Path to the particles data file (.npy or .json)')
    parser.add_argument('--animate', action='store_true', help='Create an animation instead of a static plot')
    parser.add_argument('--output', help='Path to save the animation (requires --animate)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = MCLVisualizer(args.map, args.trajectory, args.particles)
    
    if args.animate:
        if args.output:
            visualizer.save_animation(args.output)
        else:
            visualizer.visualize_animation()
    else:
        visualizer.visualize_static()

if __name__ == '__main__':
    main()
