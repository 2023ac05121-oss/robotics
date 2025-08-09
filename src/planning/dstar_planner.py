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
import argparse
import os

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
        
        # Check if obstacle-free (0 is free, 1 is obstacle)
        # Important: Map is indexed as [y, x] (row, column)
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
        print(f"Computing shortest path from ({start_x},{start_y}) to ({goal_x},{goal_y})")
        
        # Input validation
        if not self.is_valid_state(start_x, start_y):
            print(f"Warning: Start position ({start_x},{start_y}) is invalid or in an obstacle")
            # Try to find a nearby valid state
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    new_x, new_y = start_x + dx, start_y + dy
                    if self.is_valid_state(new_x, new_y):
                        print(f"Using alternate start position: ({new_x},{new_y})")
                        start_x, start_y = new_x, new_y
                        break
                else:
                    continue
                break
            else:
                raise ValueError("Could not find a valid start position")
                
        if not self.is_valid_state(goal_x, goal_y):
            print(f"Warning: Goal position ({goal_x},{goal_y}) is invalid or in an obstacle")
            # Try to find a nearby valid state
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    new_x, new_y = goal_x + dx, goal_y + dy
                    if self.is_valid_state(new_x, new_y):
                        print(f"Using alternate goal position: ({new_x},{new_y})")
                        goal_x, goal_y = new_x, new_y
                        break
                else:
                    continue
                break
            else:
                raise ValueError("Could not find a valid goal position")
        
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
    
    def visualize(self, start, goal, path, current_pos=None, changed_cells=None):
        """
        Visualize the map, path, and robot position.
        
        Args:
            start: Start coordinates (x, y)
            goal: Goal coordinates (x, y)
            path: List of (x, y) coordinates representing the path
            current_pos: Current robot position (x, y)
            changed_cells: List of cells that have changed (for highlighting)
        """
        plt.figure(figsize=(10, 8))
        
        # Create a visualization map
        viz_map = np.zeros((self.height, self.width, 3))  # RGB
        
        # Mark free space as white
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y, x] == 0:
                    viz_map[y, x] = [1, 1, 1]  # White
                else:
                    viz_map[y, x] = [0, 0, 0]  # Black (obstacle)
        
        # Mark changed cells (if any)
        if changed_cells:
            for x, y, _ in changed_cells:
                if 0 <= x < self.width and 0 <= y < self.height:
                    viz_map[y, x] = [1, 0, 0]  # Red (changed)
        
        # Display the map
        plt.imshow(viz_map, origin='lower')
        
        # Mark start and goal
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'r*', markersize=12, label='Goal')
        
        # Plot the path
        if path and len(path) > 1:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')
        
        # Mark current position
        if current_pos:
            plt.plot(current_pos[0], current_pos[1], 'yo', markersize=8, label='Current Position')
        
        plt.title('D* Path Planning')
        plt.legend()
        plt.grid(True)
        
        # Display but don't block
        plt.draw()
        plt.pause(0.1)
        
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
        print(f"Initializing D* planner from {start} to {goal}...")
        self.compute_shortest_path(start[0], start[1], goal[0], goal[1])
        
        # Get initial path
        path = self.get_path(start[0], start[1], goal[0], goal[1])
        
        if not path:
            print("No path found from start to goal!")
            return []
        
        print(f"Initial path found with {len(path)} steps")
        
        # Robot position starts at the beginning of the path
        robot_pos = list(start)
        
        # Actual path taken
        actual_path = [tuple(robot_pos)]
        
        # Simulation step counter
        step_count = 0
        
        # Create a directory for visualization frames if needed
        if visualize_steps:
            import os
            frames_dir = "dstar_frames"
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
            print(f"Visualization frames will be saved in {frames_dir}/")
        
        # Initial visualization
        if visualize_steps:
            print("Showing initial map and path...")
            self.visualize(start, goal, path, tuple(robot_pos))
            plt.savefig(f'dstar_frames/dstar_step_000.png')
            plt.close()
        
        # Simulate robot movement along the path
        while tuple(robot_pos) != goal:
            step_count += 1
            
            # Progress indication
            distance_to_goal = np.sqrt((robot_pos[0]-goal[0])**2 + (robot_pos[1]-goal[1])**2)
            print(f"Step {step_count}: Robot at {tuple(robot_pos)}, distance to goal: {distance_to_goal:.2f}")
            
            # Get next position from path
            if len(path) <= 1:
                print("Path is empty or just contains the current position!")
                break
                
            next_pos = path[1]  # Skip current position (path[0])
            
            # Check for nearby obstacles (simulate sensor)
            changed_cells = []
            for y in range(robot_pos[1] - sensor_range, robot_pos[1] + sensor_range + 1):
                for x in range(robot_pos[0] - sensor_range, robot_pos[0] + sensor_range + 1):
                    # Skip out-of-bounds cells
                    if x < 0 or x >= self.width or y < 0 or y >= self.height:
                        continue
                        
                    # If this is an obstacle that we didn't know about before
                    if self.map[y, x] == 1 and self.is_valid_state(x, y):
                        changed_cells.append((x, y, float('inf')))
            
            # If we detected any changes, update the map and replan
            if changed_cells:
                print(f"  Detected {len(changed_cells)} new obstacles, replanning...")
                
                # Update the map and replan
                path = self.replan(robot_pos[0], robot_pos[1], goal[0], goal[1], changed_cells)
                
                if not path:
                    print("  No path found after replanning!")
                    break
                    
                print(f"  New path found with {len(path)} steps")
                
                # Visualize the changes and new path
                if visualize_steps:
                    print(f"  Visualizing after replanning...")
                    self.visualize(start, goal, path, tuple(robot_pos), changed_cells)
                    plt.savefig(f'dstar_frames/dstar_step_{step_count:03d}_replan.png')
                    plt.close()
                    
                # Continue to next iteration (skip movement for this step)
                continue
            
            # Move the robot to the next position
            robot_pos[0] = next_pos[0]
            robot_pos[1] = next_pos[1]
            
            # Add to actual path
            actual_path.append(tuple(robot_pos))
            
            # Update path to remove the position we just moved to
            path = path[1:]
            
            # Visualize after movement
            if visualize_steps and step_count % 5 == 0:
                print(f"  Visualizing step {step_count}...")
                self.visualize(start, goal, path, tuple(robot_pos))
                plt.savefig(f'dstar_frames/dstar_step_{step_count:03d}.png')
                plt.close()
            
            # Check if we've reached the goal
            if tuple(robot_pos) == goal:
                print(f"Goal reached in {step_count} steps!")
                break
                
            # Check for maximum steps to prevent infinite loops
            if step_count >= 1000:
                print("Maximum steps reached, stopping simulation.")
                break
        
        # Final visualization
        if visualize_steps:
            print("Creating final visualization...")
            self.visualize(start, goal, path, tuple(robot_pos))
            plt.savefig(f'dstar_frames/dstar_step_final.png')
            plt.close()
        
        if tuple(robot_pos) == goal:
            print("Simulation completed successfully! Robot reached the goal.")
        else:
            print("Simulation ended without reaching the goal.")
        
        return actual_path
    
    def init(self, start_x: int, start_y: int, goal_x: int, goal_y: int):
        """
        Initialize the D* algorithm with start and goal positions.
        This is an alias for compute_shortest_path for compatibility with the test code.
        
        Args:
            start_x, start_y: Start coordinates
            goal_x, goal_y: Goal coordinates
        """
        self.compute_shortest_path(start_x, start_y, goal_x, goal_y)
    
    def plan(self) -> List[Tuple[int, int]]:
        """
        Plan a path from the last initialized start to goal.
        This is an alias for get_path for compatibility with the test code.
        
        Returns:
            List of (x, y) coordinates representing the path
        """
        # Get the start and goal states (assuming they've been initialized)
        start_states = [s for s in self.states.values() if s.tag == "CLOSED"]
        goal_states = [s for s in self.states.values() if s.h == 0]
        
        if not start_states:
            print("Error: No closed states found. D* not properly initialized. Call init() first.")
            return []
        
        if not goal_states:
            print("Error: No goal state found with h=0. D* not properly initialized. Call init() first.")
            return []
        
        # Find the start state - we have multiple strategies to ensure we get a valid one
        try:
            # Strategy 1: Find state with highest h-value (should be furthest from goal)
            start_state = max(start_states, key=lambda s: s.h)
            
            # Validate that we have a reasonable start state
            if start_state.h == float('inf'):
                print("Warning: Start state has h=inf, searching for better start state...")
                # Strategy 2: Find any closed state with finite h value
                valid_states = [s for s in start_states if s.h < float('inf')]
                if valid_states:
                    start_state = valid_states[0]
                    print(f"Using alternate start state: {start_state.x},{start_state.y} with h={start_state.h}")
            
        except Exception as e:
            print(f"Error finding start state: {e}")
            # Strategy 3: Find any processed state with finite h value
            for state in self.states.values():
                if state.tag == "CLOSED" and state.h < float('inf'):
                    start_state = state
                    print(f"Using backup start state: {start_state.x},{start_state.y}")
                    break
            else:
                print("Critical error: Could not find any valid start state")
                return []
        
        goal_state = goal_states[0]
        print(f"Planning path from ({start_state.x},{start_state.y}) to ({goal_state.x},{goal_state.y})")
        
        # Get the path
        path = self.get_path(start_state.x, start_state.y, goal_state.x, goal_state.y)
        
        # Check if path is valid
        if not path:
            print("Warning: No path found from start to goal")
        else:
            print(f"Path found with {len(path)} steps")
            
        return path
    
    def update_cell(self, affected_area):
        """
        Update D* when cells in the affected area have changed.
        
        Args:
            affected_area: Tuple (min_x, min_y, max_x, max_y) of the affected area
        """
        min_x, min_y, max_x, max_y = affected_area
        
        # Get all states in the affected area
        changed_cells = []
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Check if the cell is an obstacle in the map
                    if self.map[y, x] == 1:
                        changed_cells.append((x, y, float('inf')))
        
        # If we found any changed cells, update them
        if changed_cells:
            print(f"Updating {len(changed_cells)} cells in D* map")
            # Get the start and goal states (assuming they've been initialized)
            start_states = [s for s in self.states.values() if s.tag == "CLOSED"]
            goal_states = [s for s in self.states.values() if s.h == 0]
            
            if not start_states or not goal_states:
                print("Error: D* not properly initialized. Call init() first.")
                return
            
            # Find the start state (should be the one with the highest h value)
            start_state = max(start_states, key=lambda s: s.h)
            goal_state = goal_states[0]
            
            # Update the map and replan
            for x, y, cost in changed_cells:
                self.modify_cost(x, y, cost)
            
            self.compute_shortest_path(start_state.x, start_state.y, goal_state.x, goal_state.y)
    
    def update_known_map(self, known_map):
        """
        Update the D* planner with a new map.
        
        Args:
            known_map: New map to use for planning
        """
        # Update the map
        self.map = known_map
        
        # Get the start and goal states (assuming they've been initialized)
        start_states = [s for s in self.states.values() if s.tag == "CLOSED"]
        goal_states = [s for s in self.states.values() if s.h == 0]
        
        if not start_states or not goal_states:
            print("Error: D* not properly initialized. Call init() first.")
            return
        
        # Find the start state (should be the one with the highest h value)
        start_state = max(start_states, key=lambda s: s.h)
        goal_state = goal_states[0]
        
        # Recompute the shortest path
        self.compute_shortest_path(start_state.x, start_state.y, goal_state.x, goal_state.y)
        
    def test_planning(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Test if the planner can find a path without simulation or visualization.
        
        Args:
            start: Start coordinates (x, y)
            goal: Goal coordinates (x, y)
            
        Returns:
            True if a path is found, False otherwise
        """
        print(f"Testing D* planner from {start} to {goal}...")
        
        # Initialize
        self.compute_shortest_path(start[0], start[1], goal[0], goal[1])
        
        # Get initial path
        path = self.get_path(start[0], start[1], goal[0], goal[1])
        
        if not path:
            print("No path found from start to goal!")
            return False
        
        print(f"Path found with {len(path)} steps")
        print(f"First few steps: {path[:5]}")
        print(f"Last few steps: {path[-5:]}")
        
        return True

# Example usage
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='D* Path Planning Algorithm')
    parser.add_argument('--test_only', action='store_true', help='Run in test mode only (no visualization)')
    parser.add_argument('--map_size', type=int, default=50, help='Size of the map (default: 50)')
    parser.add_argument('--obstacle_density', type=float, default=0.1, help='Density of random obstacles (default: 0.1)')
    args = parser.parse_args()
    
    print("D* Planner Test")
    print(f"Map size: {args.map_size}x{args.map_size}")
    print(f"Obstacle density: {args.obstacle_density}")
    
    # Create a simple map (grid)
    grid_size = (args.map_size, args.map_size)
    grid_map = np.zeros(grid_size)
    
    # Add some obstacles
    # Vertical wall with a gap
    wall_pos = args.map_size // 2
    grid_map[10:args.map_size-10, wall_pos] = 1
    gap_start = args.map_size // 2 - 2
    grid_map[gap_start:gap_start+4, wall_pos] = 0  # Gap in the wall
    
    # Some random obstacles
    np.random.seed(42)  # For reproducibility
    random_obstacles = np.random.rand(grid_size[0], grid_size[1]) < args.obstacle_density
    random_obstacles[5:args.map_size-5, 5:args.map_size-5] = random_obstacles[5:args.map_size-5, 5:args.map_size-5] & (np.random.rand(args.map_size-10, args.map_size-10) < args.obstacle_density)
    grid_map = np.logical_or(grid_map, random_obstacles).astype(float)
    
    # Clear start and goal areas
    start = (5, 5)
    goal = (args.map_size-5, args.map_size-5)
    grid_map[start[1]-2:start[1]+3, start[0]-2:start[0]+3] = 0
    grid_map[goal[1]-2:goal[1]+3, goal[0]-2:goal[0]+3] = 0
    
    # Initialize D* planner
    dstar = DStar(grid_map)
    
    if args.test_only:
        # Just test if planning works
        if dstar.test_planning(start, goal):
            print("Planning test successful! A path was found.")
            
            # Visualize just the initial path
            plt.figure(figsize=(12, 10))
            plt.imshow(grid_map, cmap='gray_r', origin='lower')
            plt.scatter([start[0]], [start[1]], c='g', s=200, marker='o', label='Start')
            plt.scatter([goal[0]], [goal[1]], c='r', s=200, marker='*', label='Goal')
            
            # Get the path
            path = dstar.get_path(start[0], start[1], goal[0], goal[1])
            if path:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                plt.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
            
            plt.title('D* Path Planning - Test Result')
            plt.legend()
            plt.grid(True)
            plt.savefig('dstar_test_result.png')
            print("Test visualization saved to dstar_test_result.png")
    else:
        # Run the full simulation
        print("Running full simulation (this may take a while)...")
        # Add a "hidden" obstacle that will be discovered during navigation
        true_grid_map = grid_map.copy()
        true_grid_map[args.map_size-15, args.map_size//4:args.map_size//4*3] = 1  # Hidden horizontal wall
        dstar.map = true_grid_map  # Set the true map for sensor simulation
        
        # Run the simulation
        start_time = time.time()
        actual_path = dstar.run_simulation(start, goal, sensor_range=3, visualize_steps=False)
        end_time = time.time()
        
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        print(f"Path length: {len(actual_path)}")
        
        # Visualize the final result
        plt.figure(figsize=(12, 10))
        plt.imshow(true_grid_map, cmap='gray_r', origin='lower')
        plt.scatter([start[0]], [start[1]], c='g', s=200, marker='o', label='Start')
        plt.scatter([goal[0]], [goal[1]], c='r', s=200, marker='*', label='Goal')
        
        if actual_path:
            path_x = [p[0] for p in actual_path]
            path_y = [p[1] for p in actual_path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='Actual Path')
        
        plt.title('D* Path Planning - Final Result')
        plt.legend()
        plt.grid(True)
        plt.savefig('dstar_final_result.png')
        print("Final visualization saved to dstar_final_result.png")
    
    print("Simulation complete.")
