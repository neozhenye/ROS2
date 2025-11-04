#!/usr/bin/env python3

"""
Maze Solver Node for ROS 2 Humble
Combines SLAM, Nav2, and Frontier Exploration to solve unknown mazes
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
from scipy.ndimage import binary_dilation
import math

class MazeSolver(Node):
    def __init__(self):
        super().__init__('maze_solver')
        
        # Parameters
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('frontier_min_size', 5)
        self.declare_parameter('exploration_interval', 2.0)
        self.declare_parameter('maze_exit_threshold', 3.0)  # Distance from walls indicating exit
        
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.frontier_min_size = self.get_parameter('frontier_min_size').value
        self.exploration_interval = self.get_parameter('exploration_interval').value
        self.exit_threshold = self.get_parameter('maze_exit_threshold').value
        
        # State variables
        self.current_pose = None
        self.map_data = None
        self.map_info = None
        self.exploring = True
        self.exit_found = False
        self.exit_position = None
        self.start_position = None
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # Action client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Timer for exploration
        self.exploration_timer = self.create_timer(
            self.exploration_interval,
            self.exploration_callback
        )
        
        self.get_logger().info('Maze Solver Node initialized')
        self.get_logger().info('Starting maze exploration with SLAM...')
    
    def odom_callback(self, msg):
        """Update current robot position"""
        self.current_pose = msg.pose.pose
        
        if self.start_position is None:
            self.start_position = (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y
            )
            self.get_logger().info(f'Start position recorded: {self.start_position}')
    
    def map_callback(self, msg):
        """Process updated map from SLAM"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        
        # Check if we've explored enough to find the exit
        if not self.exit_found and self.current_pose is not None:
            self.check_for_exit()
    
    def scan_callback(self, msg):
        """Process laser scan data"""
        # Can be used for additional exit detection logic
        pass
    
    def check_for_exit(self):
        """
        Detect maze exit by looking for large open spaces far from start
        Exit criteria: Open area with significant distance from starting position
        """
        if self.map_data is None or self.map_info is None:
            return
        
        # Find free spaces
        free_spaces = np.where(self.map_data == 0)
        
        if len(free_spaces[0]) == 0:
            return
        
        # Convert to world coordinates and check distance from start
        for i in range(len(free_spaces[0])):
            grid_y = free_spaces[0][i]
            grid_x = free_spaces[1][i]
            
            world_x = grid_x * self.map_info.resolution + self.map_info.origin.position.x
            world_y = grid_y * self.map_info.resolution + self.map_info.origin.position.y
            
            # Check if this point is far from start and near edge of known map
            if self.start_position:
                distance_from_start = math.sqrt(
                    (world_x - self.start_position[0])**2 + 
                    (world_y - self.start_position[1])**2
                )
                
                # Check if near edge of explored area (potential exit)
                is_near_edge = (
                    grid_x < 5 or grid_x > self.map_info.width - 5 or
                    grid_y < 5 or grid_y > self.map_info.height - 5
                )
                
                if distance_from_start > 3.0 and is_near_edge:
                    # Check if there's open space around it (not a dead end)
                    if self.is_open_area(grid_x, grid_y):
                        self.exit_found = True
                        self.exit_position = (world_x, world_y)
                        self.get_logger().info(
                            f'EXIT FOUND at ({world_x:.2f}, {world_y:.2f})!'
                        )
                        self.navigate_to_exit()
                        return
    
    def is_open_area(self, x, y, radius=3):
        """Check if area around point is open (not surrounded by walls)"""
        if self.map_data is None:
            return False
        
        h, w = self.map_data.shape
        count_free = 0
        count_total = 0
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    count_total += 1
                    if self.map_data[ny, nx] == 0:
                        count_free += 1
        
        return count_free / count_total > 0.6 if count_total > 0 else False
    
    def exploration_callback(self):
        """Main exploration loop - find and navigate to frontiers"""
        if self.exit_found:
            return  # Stop exploring if exit found
        
        if self.map_data is None or self.current_pose is None:
            return
        
        # Find frontiers
        frontiers = self.find_frontiers()
        
        if len(frontiers) == 0:
            self.get_logger().warn('No frontiers found - maze may be fully explored')
            # Try to find exit in fully explored maze
            self.find_exit_in_explored_maze()
            return
        
        # Select best frontier
        target_frontier = self.select_best_frontier(frontiers)
        
        if target_frontier is not None:
            self.navigate_to_frontier(target_frontier)
    
    def find_frontiers(self):
        """
        Find frontier cells (boundary between known free space and unknown space)
        Returns list of frontier centroids in world coordinates
        """
        if self.map_data is None:
            return []
        
        # Identify free cells (0), unknown cells (-1), and occupied cells (100)
        free_cells = (self.map_data == 0)
        unknown_cells = (self.map_data == -1)
        
        # Dilate free cells to find boundaries
        dilated_free = binary_dilation(free_cells, iterations=1)
        
        # Frontiers are unknown cells adjacent to free cells
        frontiers = unknown_cells & dilated_free
        
        # Label connected frontier regions
        from scipy.ndimage import label
        labeled_frontiers, num_features = label(frontiers)
        
        frontier_centroids = []
        
        for i in range(1, num_features + 1):
            region = (labeled_frontiers == i)
            size = np.sum(region)
            
            if size >= self.frontier_min_size:
                # Calculate centroid
                coords = np.where(region)
                centroid_y = int(np.mean(coords[0]))
                centroid_x = int(np.mean(coords[1]))
                
                # Convert to world coordinates
                world_x = centroid_x * self.map_info.resolution + self.map_info.origin.position.x
                world_y = centroid_y * self.map_info.resolution + self.map_info.origin.position.y
                
                frontier_centroids.append((world_x, world_y, size))
        
        return frontier_centroids
    
    def select_best_frontier(self, frontiers):
        """
        Select the best frontier to explore
        Prioritize: distance from current position and frontier size
        """
        if not frontiers or self.current_pose is None:
            return None
        
        best_frontier = None
        best_score = float('-inf')
        
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        for fx, fy, size in frontiers:
            distance = math.sqrt((fx - current_x)**2 + (fy - current_y)**2)
            
            # Score: balance between size and proximity
            # Prefer larger frontiers that aren't too far
            score = size / (distance + 1.0)
            
            if score > best_score:
                best_score = score
                best_frontier = (fx, fy)
        
        return best_frontier
    
    def navigate_to_frontier(self, frontier):
        """Send navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = frontier[0]
        goal_msg.pose.pose.position.y = frontier[1]
        goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info(f'Navigating to frontier at ({frontier[0]:.2f}, {frontier[1]:.2f})')
        
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal_msg)
    
    def find_exit_in_explored_maze(self):
        """
        Find exit in a fully explored maze
        Look for the point farthest from start with open space
        """
        if self.map_data is None or self.start_position is None:
            return
        
        free_spaces = np.where(self.map_data == 0)
        max_distance = 0
        exit_candidate = None
        
        for i in range(len(free_spaces[0])):
            grid_y = free_spaces[0][i]
            grid_x = free_spaces[1][i]
            
            world_x = grid_x * self.map_info.resolution + self.map_info.origin.position.x
            world_y = grid_y * self.map_info.resolution + self.map_info.origin.position.y
            
            distance = math.sqrt(
                (world_x - self.start_position[0])**2 + 
                (world_y - self.start_position[1])**2
            )
            
            if distance > max_distance:
                max_distance = distance
                exit_candidate = (world_x, world_y)
        
        if exit_candidate and max_distance > 2.0:
            self.exit_found = True
            self.exit_position = exit_candidate
            self.get_logger().info(
                f'Exit identified at ({exit_candidate[0]:.2f}, {exit_candidate[1]:.2f})'
            )
            self.navigate_to_exit()
    
    def navigate_to_exit(self):
        """Navigate to the exit using Nav2"""
        if self.exit_position is None:
            return
        
        self.exploring = False
        self.exploration_timer.cancel()
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = self.exit_position[0]
        goal_msg.pose.pose.position.y = self.exit_position[1]
        goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('NAVIGATING TO EXIT!')
        self.get_logger().info(f'Exit position: ({self.exit_position[0]:.2f}, {self.exit_position[1]:.2f})')
        self.get_logger().info('=' * 50)
        
        self.nav_client.wait_for_server()
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return
        
        self.get_logger().info('Goal accepted, navigating to exit...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        """Handle goal result"""
        result = future.result().result
        self.get_logger().info('=' * 50)
        self.get_logger().info('MAZE SOLVED! EXIT REACHED!')
        self.get_logger().info('=' * 50)


def main(args=None):
    rclpy.init(args=args)
    maze_solver = MazeSolver()
    
    try:
        rclpy.spin(maze_solver)
    except KeyboardInterrupt:
        pass
    finally:
        maze_solver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
