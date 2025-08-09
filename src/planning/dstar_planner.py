#!/usr/bin/env python3
"""
D* Planner implementation.
This algorithm performs efficient path planning in dynamic environments.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
from typing import List, Tuple, Dict, Set, Optional
import time

class State:
    """Represents a state (grid cell) in the D* algorithm."""
    
    def __init__(self, x: int, y: int):
        """
        Initialize a state.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
        """
        self.x = x
        self.y = y
        self.tag = "NEW"  # Tag: NEW, OPEN, CLOSED
        self.h = float('inf')  # Cost-to-goal estimate
        self.k = float('inf')  # Minimum of h values during previous expansions
        self.back_ptr = None  # Back pointer for path reconstruction
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.x == other.x and self.y == other.y
        return False
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __lt__(self, other):
        # For priority queue comparison
        if self.k == other.k:
            return self.h < other.h
        return self.k < other.k
    
    def __repr__(self):
        return f"State({self.x}, {self.y}, {self.tag}, h={self.h:.2f}, k={self.k:.2f})"

class DStar:
    """Implementation of the D* algorithm for path planning."""
    
    def __init__(self, grid_map: np.ndarray):
        """
        Initialize the D* planner.
        
        Args:
            grid_map: 2D numpy array (0 = free, 1 = obstacle)
        """
        self.map = grid_map
        self.height, self.width = grid_map.shape
        self.states = {}  # Dictionary mapping (x, y) to State objects
        self.open_list = []  # Priority queue for OPEN states
        self.k_m = 0  # Accumulated cost (for moving robot)
        
        # Possible movements (8-connected grid)
        self.movements = [
            (1, 0),   # Right
            (1, 1),   # Up-Right
            (0, 1),   # Up
            (-1, 1),  # Up-Left
            (-1, 0),  # Left
            (-1, -1), # Down-Left
            (0, -1),  # Down
            (1, -1),  # Down-Right
        ]
        
        # Cost of movements (diagonal movements cost more)
        self.movement_costs = [
            1.0,      # Right
            math.sqrt(2),  # Up-Right
            1.0,      # Up
            math.sqrt(2),  # Up-Left
            1.0,      # Left
            math.sqrt(2),  # Down-Left
            1.0,      # Down
            math.sqrt(2),  # Down-Right
        ]
    
    def get_state(self, x: int, y: int) -> State:
        """
        Get the State object for the given coordinates, creating it if necessary.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            State object for the given coordinates
        """
        if (x, y) not in self.states:
            self.states[(x, y)] = State(x, y)
        return self.states[(x, y)]
    
    def is_valid_state(self, x: int, y: int) -> bool:
        """
        Check if coordinates represent a valid, obstacle-free state.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            True if the state is valid and obstacle-free
        """
        # Check if within bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # Check if obstacle-free
        return self.map[y, x] == 0
    
    def get_neighbors(self, state: State) -> List[Tuple[State, float]]:
        """
        Get neighbors of a state with their edge costs.
        
        Args:
            state: State object
            
        Returns:
            List of (neighbor_state, cost) tuples
        """
        neighbors = []
        
        for i, (dx, dy) in enumerate(self.movements):
            new_x, new_y = state.x + dx, state.y + dy
            
            if self.is_valid_state(new_x, new_y):
                neighbor = self.get_state(new_x, new_y)
                cost = self.movement_costs[i]
                neighbors.append((neighbor, cost))
        
        return neighbors
    
    def get_predecessors(self, state: State) -> List[Tuple[State, float]]:
        """
        Get predecessors of a state (states that can reach this state).
        
        Args:
            state: State object
            
        Returns:
            List of (predecessor_state, cost) tuples
        """
        predecessors = []
        
        for i, (dx, dy) in enumerate(self.movements):
            pred_x, pred_y = state.x - dx, state.y - dy
            
            if self.is_valid_state(pred_x, pred_y):
                predecessor = self.get_state(pred_x, pred_y)
                cost = self.movement_costs[i]
                predecessors.append((predecessor, cost))
        
        return predecessors
    
    def insert(self, state: State, h_new: float):
        """
        Insert or update a state in the OPEN list.
        
        Args:
            state: State to insert/update
            h_new: New h value for the state
        """
        if state.tag == "NEW":
            state.k = h_new
        elif state.tag == "OPEN":
            state.k = min(state.k, h_new)
        elif state.tag == "CLOSED":
            state.k = min(state.k, h_new)
        
        state.h = h_new
        state.tag = "OPEN"
        
        # Add to priority queue
        heapq.heappush(self.open_list, (state.k, state))
    
    def remove_min(self) -> Optional[State]:
        """
        Remove and return the state with minimum k value from the OPEN list.
        
        Returns:
            State with minimum k value, or None if the OPEN list is empty
        """
        if not self.open_list:
            return None
        
        # Get the state with minimum k value
        _, state = heapq.heappop(self.open_list)
        
        if state.tag == "CLOSED":  # If already CLOSED, skip
            return self.remove_min()
        
        state.tag = "CLOSED"
        return state
    
    def modify_cost(self, x: int, y: int, new_cost: float = float('inf')):
        """
        Modify the cost of a cell (e.g., when an obstacle is detected).
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            new_cost: New cost value (inf for obstacle)
        """
        # Update the map
        if new_cost == float('inf'):
            self.map[y, x] = 1  # Mark as obstacle
        else:
            self.map[y, x] = 0  # Mark as free
        
        # Get the state
        state = self.get_state(x, y)
        
        # If the state is closed, we need to recompute paths through it
        if state.tag == "CLOSED":
            self.insert(state, state.h)
        
        # Process the OPEN list to update affected states
        self.process_replan()
    
    def process_replan(self):
        """Process the OPEN list to propagate changes and replan."""
        while self.open_list:
            # Get the state with minimum k value
            state = self.remove_min()
            
            if state is None:
                break
            
            # If k is less than h, state needs updating
            if state.k < state.h:
                # For each predecessor
                for pred, cost in self.get_predecessors(state):
                    if (self.is_valid_state(pred.x, pred.y) and 
                        pred.h > state.k + cost):
                        pred.h = state.k + cost
                        pred.back_ptr = state
                        self.insert(pred, pred.h)
            
            # If k equals h, state is correct
            elif state.k == state.h:
                # For each predecessor
                for pred, cost in self.get_predecessors(state):
                    if (pred.tag == "NEW" or 
                        (pred.back_ptr == state and pred.h != state.h + cost) or
                        (pred.back_ptr != state and pred.h > state.h + cost)):
                        pred.back_ptr = state
                        self.insert(pred, state.h + cost)
            
            else:  # k > h, state needs updating
                # Update the state itself
                state.h = float('inf')
                
                # Update predecessors
                for pred, cost in self.get_predecessors(state):
                    if (pred.back_ptr == state or 
                        (pred.tag != "NEW" and pred.h > state.h + cost)):
                        pred.back_ptr = state
                        self.insert(pred, state.h + cost)
    
    def compute_shortest_path(self, start_x: int, start_y: int, goal_x: int, goal_y: int):
        """
        Compute the shortest path from start to goal.
        
        Args:
            start_x, start_y: Start coordinates
            goal_x, goal_y: Goal coordinates
        """
        if not self.is_valid_state(start_x, start_y) or not self.is_valid_state(goal_x, goal_y):
            raise ValueError("Start or goal position is invalid or in an obstacle")
        
        # Get start and goal states
        start_state = self.get_state(start_x, start_y)
        goal_state = self.get_state(goal_x, goal_y)
        
        # Initialize goal state
        goal_state.h = 0
        goal_state.k = 0
        goal_state.tag = "OPEN"
        
        # Add goal to OPEN list
        self.open_list = [(goal_state.k, goal_state)]
        
        # Process the OPEN list
        self.process_replan()
    
    def get_path(self, start_x: int, start_y: int, goal_x: int, goal_y: int) -> List[Tuple[int, int]]:
        """
        Get the path from start to goal.
        
        Args:
            start_x, start_y: Start coordinates
            goal_x, goal_y: Goal coordinates
            
        Returns:
            List of (x, y) coordinates representing the path
        """
        # Check if path exists
        start_state = self.get_state(start_x, start_y)
        goal_state = self.get_state(goal_x, goal_y)
        
        if start_state.h == float('inf'):
            return []  # No path exists
        
        # Reconstruct path
        path = [(start_x, start_y)]
        current = start_state
        
        while current != goal_state:
            if current.back_ptr is None:
                return []  # No path exists
            
            current = current.back_ptr
            path.append((current.x, current.y))
        
        return path
    
    def replan(self, start_x: int, start_y: int, goal_x: int, goal_y: int, 
              changed_cells: List[Tuple[int, int, float]]) -> List[Tuple[int, int]]:
        """
        Replan the path after map changes.
        
        Args:
            start_x, start_y: Start coordinates
            goal_x, goal_y: Goal coordinates
            changed_cells: List of (x, y, new_cost) tuples for cells that have changed
            
        Returns:
            New path as a list of (x, y) coordinates
        """
        # Update map with changed cells
        for x, y, cost in changed_cells:
            self.modify_cost(x, y, cost)
        
        # Recompute shortest path
        self.compute_shortest_path(start_x, start_y, goal_x, goal_y)
        
        # Get new path
        return self.get_path(start_x, start_y, goal_x, goal_y)
    
    def visualize(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 path: List[Tuple[int, int]] = None, robot_pos: Tuple[int, int] = None):
        """
        Visualize the map, path, and robot position.
        
        Args:
            start: Start coordinates (x, y)
            goal: Goal coordinates (x, y)
            path: List of (x, y) coordinates representing the path
            robot_pos: Current robot position (x, y)
        """
        plt.figure(figsize=(10, 8))
        
        # Create a colored map for visualization
        vis_map = np.zeros((self.height, self.width, 3))
        
        # White = free space, Black = obstacle
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y, x] == 0:
                    vis_map[y, x] = [1, 1, 1]  # White (free)
                else:
                    vis_map[y, x] = [0, 0, 0]  # Black (obstacle)
        
        # Plot the map
        plt.imshow(vis_map, origin='lower')
        
        # Plot the start and goal
        plt.scatter([start[0]], [start[1]], c='g', s=200, marker='o', label='Start')
        plt.scatter([goal[0]], [goal[1]], c='r', s=200, marker='*', label='Goal')
        
        # Plot the path
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
        
        # Plot the current robot position
        if robot_pos:
            plt.scatter([robot_pos[0]], [robot_pos[1]], c='c', s=150, marker='^', label='Robot')
        
        plt.title('D* Path Planning')
        plt.legend()
        plt.grid(True)
        plt.savefig('dstar_visualization.png')
        plt.close()
    
    def run_simulation(self, start: Tuple[int, int], goal: Tuple[int, int], 
                      sensor_range: int = 3, visualize_steps: bool = True):
        """
        Run a simulation of a robot navigating using D*.
        
        Args:
            start: Start coordinates (x, y)
            goal: Goal coordinates (x, y)
            sensor_range: How far the robot can "see" obstacles
            visualize_steps: Whether to visualize each step
            
        Returns:
            List of (x, y) coordinates representing the actual path taken
        """
        # Initialize
        self.compute_shortest_path(start[0], start[1], goal[0], goal[1])
        
        # Get initial path
        path = self.get_path(start[0], start[1], goal[0], goal[1])
        
        if not path:
            print("No path found from start to goal!")
            return []
        
        # Robot position starts at the beginning of the path
        robot_pos = list(start)
        
        # Actual path taken
        actual_path = [tuple(robot_pos)]
        
        # Simulate robot movement along the path
        while tuple(robot_pos) != goal:
            if visualize_steps:
                self.visualize(start, goal, path, tuple(robot_pos))
            
            # Get next position from path
            next_pos_idx = 1  # Skip current position
            if next_pos_idx >= len(path):
                print("Reached end of path but not at goal!")
                break
                
            next_pos = path[next_pos_idx]
            
            # Check for nearby obstacles (simulate sensor)
            changed_cells = []
            for y in range(robot_pos[1] - sensor_range, robot_pos[1] + sensor_range + 1):
                for x in range(robot_pos[0] - sensor_range, robot_pos[0] + sensor_range + 1):
                    # If within map bounds
                    if 0 <= x < self.width and 0 <= y < self.height:
                        # If this is an obstacle that hasn't been detected yet
                        actual_cell_value = self.map[y, x]
                        if actual_cell_value == 1 and self.is_valid_state(x, y):
                            # Found a new obstacle
                            changed_cells.append((x, y, float('inf')))
            
            # If obstacles found, replan
            if changed_cells:
                print(f"Found {len(changed_cells)} new obstacles, replanning...")
                path = self.replan(robot_pos[0], robot_pos[1], goal[0], goal[1], changed_cells)
                
                if not path:
                    print("No path found after replanning!")
                    break
                
                # Next position is now the first in the new path
                next_pos = path[1] if len(path) > 1 else tuple(robot_pos)
            
            # Move robot to next position
            robot_pos[0], robot_pos[1] = next_pos
            actual_path.append(tuple(robot_pos))
            
            # Remove current position from path
            path = path[1:]
        
        # Final visualization
        if visualize_steps:
            self.visualize(start, goal, path, tuple(robot_pos))
        
        if tuple(robot_pos) == goal:
            print("Goal reached!")
        else:
            print("Failed to reach goal.")
        
        return actual_path

# Example usage
if __name__ == "__main__":
    # Create a simple map (grid)
    grid_size = (50, 50)
    grid_map = np.zeros(grid_size)
    
    # Add some obstacles
    # Vertical wall with a gap
    grid_map[10:40, 20] = 1
    grid_map[23:27, 20] = 0  # Gap in the wall
    
    # Horizontal wall with a gap
    grid_map[30, 10:40] = 1
    grid_map[30, 23:27] = 0  # Gap in the wall
    
    # Some random obstacles
    np.random.seed(42)  # For reproducibility
    random_obstacles = np.random.rand(grid_size[0], grid_size[1]) < 0.1
    random_obstacles[5:45, 5:45] = random_obstacles[5:45, 5:45] & (np.random.rand(40, 40) < 0.1)
    grid_map = np.logical_or(grid_map, random_obstacles).astype(float)
    
    # Clear start and goal areas
    start = (5, 5)
    goal = (45, 45)
    grid_map[start[1]-2:start[1]+3, start[0]-2:start[0]+3] = 0
    grid_map[goal[1]-2:goal[1]+3, goal[0]-2:goal[0]+3] = 0
    
    # Initialize D* planner
    dstar = DStar(grid_map)
    
    # Add a "hidden" obstacle that will be discovered during navigation
    # This won't be in the original grid_map but will be detected by the robot
    true_grid_map = grid_map.copy()
    true_grid_map[35, 25:35] = 1  # Hidden horizontal wall
    dstar.map = true_grid_map  # Set the true map for sensor simulation
    
    # Run the simulation
    start_time = time.time()
    actual_path = dstar.run_simulation(start, goal, sensor_range=3, visualize_steps=True)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Path length: {len(actual_path)}")
    
    # Visualize the final result
    plt.figure(figsize=(12, 10))
    
    # Plot the map
    plt.imshow(true_grid_map, cmap='gray_r', origin='lower')
    
    # Plot the start and goal
    plt.scatter([start[0]], [start[1]], c='g', s=200, marker='o', label='Start')
    plt.scatter([goal[0]], [goal[1]], c='r', s=200, marker='*', label='Goal')
    
    # Plot the actual path taken
    if actual_path:
        path_x = [p[0] for p in actual_path]
        path_y = [p[1] for p in actual_path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='Actual Path')
    
    plt.title('D* Path Planning - Final Result')
    plt.legend()
    plt.grid(True)
    plt.savefig('dstar_final_result.png')
    plt.close()
    
    print("Simulation complete.")
