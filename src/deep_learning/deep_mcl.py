#!/usr/bin/env python3
"""
Deep Learning enhanced Monte Carlo Localization.
This implementation uses a neural network to improve the sensor model.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple, Optional

# Import base MCL implementation
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from localization.monte_carlo_localization import MonteCarloLocalization, Particle

class SensorModelNN(nn.Module):
    """Neural network for improving sensor measurements in MCL."""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        Initialize the sensor model neural network.
        
        Args:
            input_size: Number of input features (raw sensor readings)
            hidden_size: Size of hidden layers
        """
        super(SensorModelNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

class SensorDataset(Dataset):
    """Dataset for training the sensor model neural network."""
    
    def __init__(self, raw_readings: np.ndarray, true_readings: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            raw_readings: Array of raw sensor readings with noise
            true_readings: Array of ground truth sensor readings
        """
        self.raw_readings = torch.tensor(raw_readings, dtype=torch.float32)
        self.true_readings = torch.tensor(true_readings, dtype=torch.float32)
    
    def __len__(self):
        return len(self.raw_readings)
    
    def __getitem__(self, idx):
        return self.raw_readings[idx], self.true_readings[idx]

class DeepMCL(MonteCarloLocalization):
    """Monte Carlo Localization with deep learning enhanced sensor model."""
    
    def __init__(self, map_data: np.ndarray, num_particles: int = 1000, 
                 motion_noise: Tuple[float, float, float] = (0.1, 0.1, 0.05),
                 measurement_noise: float = 0.1, model_path: Optional[str] = None):
        """
        Initialize the Deep MCL algorithm.
        
        Args:
            map_data: 2D numpy array representing the environment (1 = obstacle, 0 = free space)
            num_particles: Number of particles to use
            motion_noise: Tuple of (x, y, theta) standard deviations for motion model
            measurement_noise: Standard deviation for measurement model
            model_path: Path to pre-trained sensor model (if None, will create a new one)
        """
        super(DeepMCL, self).__init__(map_data, num_particles, motion_noise, measurement_noise)
        
        # Neural network for sensor model
        self.sensor_nn = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and os.path.exists(model_path):
            self.load_sensor_model(model_path)
    
    def create_sensor_model(self, num_sensors: int):
        """
        Create a new sensor model neural network.
        
        Args:
            num_sensors: Number of sensors (input/output size)
        """
        self.sensor_nn = SensorModelNN(num_sensors).to(self.device)
    
    def load_sensor_model(self, model_path: str):
        """
        Load a pre-trained sensor model.
        
        Args:
            model_path: Path to model file
        """
        # Get number of sensors from model filename
        # Assuming filename format: "sensor_model_<num_sensors>.pth"
        try:
            num_sensors = int(model_path.split('_')[-1].split('.')[0])
        except:
            num_sensors = 8  # Default
        
        self.create_sensor_model(num_sensors)
        self.sensor_nn.load_state_dict(torch.load(model_path, map_location=self.device))
        self.sensor_nn.eval()
    
    def save_sensor_model(self, model_path: str):
        """
        Save the trained sensor model.
        
        Args:
            model_path: Path to save model file
        """
        if self.sensor_nn:
            torch.save(self.sensor_nn.state_dict(), model_path)
            
    def save_model(self, model_path: str):
        """
        Save the trained sensor model (alias for save_sensor_model for compatibility).
        
        Args:
            model_path: Path to save model file
        """
        self.save_sensor_model(model_path)
    
    def generate_training_data(self, num_samples: int, num_sensors: int, 
                              max_range: float, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for the sensor model.
        
        Args:
            num_samples: Number of samples to generate
            num_sensors: Number of sensors
            max_range: Maximum sensor range
            noise_level: Standard deviation of noise to add
            
        Returns:
            Tuple of (noisy_readings, true_readings) arrays
        """
        # Arrays to store the data
        true_readings = np.zeros((num_samples, num_sensors))
        noisy_readings = np.zeros((num_samples, num_sensors))
        
        # Generate sensor angles
        sensor_angles = [i * 2 * math.pi / num_sensors for i in range(num_sensors)]
        
        # Find all free space cells in the map
        free_cells = np.argwhere(self.map == 0)
        
        if len(free_cells) == 0:
            raise ValueError("No free space found in the map")
        
        # Generate samples
        for i in range(num_samples):
            # Randomly choose a free cell
            idx = random.randint(0, len(free_cells) - 1)
            y, x = free_cells[idx]
            
            # Add some noise to the position
            x += random.uniform(-0.45, 0.45)
            y += random.uniform(-0.45, 0.45)
            
            # Random orientation
            theta = random.uniform(0, 2 * math.pi)
            
            # Get true sensor readings
            for j, angle in enumerate(sensor_angles):
                # Calculate absolute angle
                abs_angle = theta + angle
                
                # Cast a ray and get true distance
                distance = self.cast_ray(x, y, abs_angle, max_range)
                true_readings[i, j] = distance
                
                # Add noise to create raw reading
                noisy_distance = distance + random.gauss(0, noise_level * max_range)
                noisy_readings[i, j] = max(0, min(noisy_distance, max_range))
        
        return noisy_readings, true_readings
    
    def train_sensor_model(self, num_samples: int = 10000, num_sensors: int = 8, 
                          max_range: float = 10.0, noise_level: float = 0.1, 
                          batch_size: int = 64, num_epochs: int = 10,
                          learning_rate: float = 0.001):
        """
        Train the sensor model neural network.
        
        Args:
            num_samples: Number of training samples
            num_sensors: Number of sensors
            max_range: Maximum sensor range
            noise_level: Noise level for training data
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        # Create model if it doesn't exist
        if self.sensor_nn is None:
            self.create_sensor_model(num_sensors)
        
        # Generate training data
        print("Generating training data...")
        noisy_readings, true_readings = self.generate_training_data(
            num_samples, num_sensors, max_range, noise_level
        )
        
        # Create dataset and dataloader
        dataset = SensorDataset(noisy_readings, true_readings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.sensor_nn.parameters(), lr=learning_rate)
        
        # Training loop
        print("Training sensor model...")
        self.sensor_nn.train()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for batch_idx, (raw_batch, true_batch) in enumerate(dataloader):
                # Transfer to device
                raw_batch = raw_batch.to(self.device)
                true_batch = true_batch.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.sensor_nn(raw_batch)
                loss = criterion(outputs, true_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 20 == 19:  # Print every 20 batches
                    print(f"[{epoch+1}, {batch_idx+1}] loss: {running_loss/20:.4f}")
                    running_loss = 0.0
        
        print("Training complete!")
        self.sensor_nn.eval()
    
    def train_model(self, inputs, targets, epochs=10, batch_size=64, learning_rate=0.001):
        """
        Train the sensor model using pre-prepared data.
        
        Args:
            inputs: Tensor of input (noisy) measurements
            targets: Tensor of target (true) measurements
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        # Create model if it doesn't exist
        if self.sensor_nn is None:
            self.create_sensor_model(inputs.shape[1])
        
        # Create dataset and dataloader
        dataset = SensorDataset(inputs.numpy(), targets.numpy())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.sensor_nn.parameters(), lr=learning_rate)
        
        # Training loop
        print(f"Training sensor model for {epochs} epochs...")
        self.sensor_nn.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch_idx, (raw_batch, true_batch) in enumerate(dataloader):
                # Transfer to device
                raw_batch = raw_batch.to(self.device)
                true_batch = true_batch.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.sensor_nn(raw_batch)
                loss = criterion(outputs, true_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
        
        print("Training complete!")
        self.sensor_nn.eval()
    
    def enhance_measurements(self, measurements: List[float]) -> List[float]:
        """
        Enhance raw sensor measurements using the neural network.
        
        Args:
            measurements: List of raw distance measurements from sensors
            
        Returns:
            List of enhanced distance measurements
        """
        if self.sensor_nn is None:
            return measurements  # No model available, return raw measurements
        
        # Convert to tensor
        raw_tensor = torch.tensor(measurements, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Pass through neural network
        with torch.no_grad():
            enhanced_tensor = self.sensor_nn(raw_tensor)
        
        # Convert back to list
        enhanced_measurements = enhanced_tensor.squeeze(0).cpu().numpy().tolist()
        
        return enhanced_measurements
    
    def measurement_update(self, measurements: List[float], sensor_angles: List[float], max_range: float):
        """
        Override the measurement update to use the enhanced measurements.
        
        Args:
            measurements: List of distance measurements from sensors
            sensor_angles: List of angles (in radians) corresponding to each measurement
            max_range: Maximum range of the sensors
        """
        # Enhance measurements if sensor model exists
        enhanced_measurements = self.enhance_measurements(measurements)
        
        # Call the parent class method with enhanced measurements
        super(DeepMCL, self).measurement_update(enhanced_measurements, sensor_angles, max_range)
    
    def visualize_sensor_model(self, num_test_samples: int = 5, num_sensors: int = 8, 
                              max_range: float = 10.0, noise_level: float = 0.2):
        """
        Visualize the performance of the sensor model.
        
        Args:
            num_test_samples: Number of test samples to visualize
            num_sensors: Number of sensors
            max_range: Maximum sensor range
            noise_level: Noise level for test data
        """
        if self.sensor_nn is None:
            print("No sensor model available. Please train or load a model first.")
            return
        
        # Generate test data
        noisy_readings, true_readings = self.generate_training_data(
            num_test_samples, num_sensors, max_range, noise_level
        )
        
        # Enhanced readings using the neural network
        enhanced_readings = []
        for i in range(num_test_samples):
            enhanced = self.enhance_measurements(noisy_readings[i])
            enhanced_readings.append(enhanced)
        
        enhanced_readings = np.array(enhanced_readings)
        
        # Visualize
        plt.figure(figsize=(15, 5 * num_test_samples))
        angles = np.linspace(0, 2*np.pi, num_sensors, endpoint=False)
        
        for i in range(num_test_samples):
            plt.subplot(num_test_samples, 1, i+1, polar=True)
            
            plt.plot(angles, true_readings[i], 'g-', linewidth=2, label='Ground Truth')
            plt.plot(angles, noisy_readings[i], 'r--', linewidth=1, label='Noisy Measurements')
            plt.plot(angles, enhanced_readings[i], 'b-', linewidth=1.5, label='Enhanced Measurements')
            
            plt.title(f'Sample {i+1}')
            plt.legend(loc='lower right')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('sensor_model_visualization.png')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Create a simple map (20x20 grid with some obstacles)
    map_data = np.zeros((20, 20))
    # Add some obstacles
    map_data[5:15, 5] = 1  # Vertical wall
    map_data[5, 5:15] = 1  # Horizontal wall
    map_data[15, 5:15] = 1  # Horizontal wall
    map_data[5:15, 15] = 1  # Vertical wall
    
    # Number of sensors and max range
    num_sensors = 8
    max_range = 10.0
    
    # Define sensor angles
    sensor_angles = [i * 2 * math.pi / num_sensors for i in range(num_sensors)]
    
    # Initialize Deep MCL
    deep_mcl = DeepMCL(map_data, num_particles=1000)
    
    # Train the sensor model
    deep_mcl.train_sensor_model(
        num_samples=5000,
        num_sensors=num_sensors,
        max_range=max_range,
        noise_level=0.2,
        batch_size=64,
        num_epochs=5
    )
    
    # Save the trained model
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sensor_model_{num_sensors}.pth")
    deep_mcl.save_sensor_model(model_save_path)
    
    # Visualize the sensor model performance
    deep_mcl.visualize_sensor_model(num_test_samples=3, num_sensors=num_sensors, max_range=max_range)
    
    # True robot pose (for simulation)
    true_pose = (10.0, 10.0, 0.0)  # (x, y, theta)
    
    # Generate simulated sensor readings
    readings = []
    for angle in sensor_angles:
        # Calculate absolute angle
        abs_angle = true_pose[2] + angle
        
        # Cast a ray and add some noise
        distance = deep_mcl.cast_ray(true_pose[0], true_pose[1], abs_angle, max_range)
        noisy_distance = distance + random.gauss(0, 0.2 * max_range)
        readings.append(max(0, noisy_distance))  # Ensure non-negative distance
    
    # Update with the measurements
    deep_mcl.measurement_update(readings, sensor_angles, max_range)
    
    # Visualize
    deep_mcl.visualize(true_pose)
    
    print(f"True position: {true_pose}")
    print(f"Estimated position: {deep_mcl.get_position_estimate()}")
    
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
            distance = deep_mcl.cast_ray(new_x, new_y, abs_angle, max_range)
            noisy_distance = distance + random.gauss(0, 0.2 * max_range)
            step_readings.append(max(0, noisy_distance))
        
        simulated_readings.append(step_readings)
    
    # Run the full localization algorithm
    estimated_poses = deep_mcl.run_localization(
        motion_commands, simulated_readings, sensor_angles, max_range, 
        true_poses=true_poses, visualize_steps=True
    )
    
    print("Motion sequence complete.")
    print(f"Final true position: {true_poses[-1]}")
    print(f"Final estimated position: {estimated_poses[-1]}")
    
    # Compare with standard MCL
    standard_mcl = MonteCarloLocalization(map_data, num_particles=1000)
    standard_estimates = standard_mcl.run_localization(
        motion_commands, simulated_readings, sensor_angles, max_range, 
        true_poses=true_poses, visualize_steps=False
    )
    
    # Calculate errors
    deep_errors = []
    standard_errors = []
    
    for i, true_pos in enumerate(true_poses[1:]):  # Skip initial position
        deep_est = estimated_poses[i]
        standard_est = standard_estimates[i]
        
        deep_error = math.sqrt((true_pos[0] - deep_est[0])**2 + (true_pos[1] - deep_est[1])**2)
        standard_error = math.sqrt((true_pos[0] - standard_est[0])**2 + (true_pos[1] - standard_est[1])**2)
        
        deep_errors.append(deep_error)
        standard_errors.append(standard_error)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(deep_errors, 'b-', linewidth=2, label='Deep MCL')
    plt.plot(standard_errors, 'r--', linewidth=2, label='Standard MCL')
    plt.xlabel('Step')
    plt.ylabel('Position Error')
    plt.title('Error Comparison: Deep MCL vs Standard MCL')
    plt.legend()
    plt.grid(True)
    plt.savefig('mcl_comparison.png')
    plt.close()
    
    print(f"Average Deep MCL error: {sum(deep_errors)/len(deep_errors):.4f}")
    print(f"Average Standard MCL error: {sum(standard_errors)/len(standard_errors):.4f}")
