#!/usr/bin/env python3

"""
Maze Solver Node for ROS 2 Humble
Combines SLAM, Nav2, and Frontier Exploration to solve unknown mazes
With TIME-BASED stuck detection for accurate movement tracking
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Point, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
from scipy.ndimage import binary_dilation
import math
import random
import time

class MazeSolver(Node):
    def __init__(self):
        super().__init__('maze_solver')
        
        # DON'T declare use_sim_time - it's already passed by launch file
        
        # Parameters
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('frontier_min_size', 5)
        self.declare_parameter('exploration_interval', 5.0)
        self.declare_parameter('maze_exit_threshold', 3.0)
        self.declare_parameter('initial_rotation_time', 20.0)
        self.declare_parameter('min_frontier_distance', 0.5)
        self.declare_parameter('backup_distance', 0.2)
        self.declare_parameter('backup_speed', 0.05)
        self.declare_parameter('spin_angle', 1.57)
        self.declare_parameter('min_obstacle_distance', 0.3)
        self.declare_parameter('stuck_check_duration', 2.5)  # NEW: Time window for stuck detection
        self.declare_parameter('stuck_velocity_threshold', 0.04)  # NEW: 4 cm/s threshold
        self.declare_parameter('stuck_displacement_threshold', 0.15)  # NEW: 15cm minimum movement
        
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.frontier_min_size = self.get_parameter('frontier_min_size').value
        self.exploration_interval = self.get_parameter('exploration_interval').value
        self.exit_threshold = self.get_parameter('maze_exit_threshold').value
        self.initial_rotation_time = self.get_parameter('initial_rotation_time').value
        self.min_frontier_distance = self.get_parameter('min_frontier_distance').value
        self.backup_distance = self.get_parameter('backup_distance').value
        self.backup_speed = self.get_parameter('backup_speed').value
        self.spin_angle = self.get_parameter('spin_angle').value
        self.min_obstacle_distance = self.get_parameter('min_obstacle_distance').value
        self.stuck_check_duration = self.get_parameter('stuck_check_duration').value
        self.stuck_velocity_threshold = self.get_parameter('stuck_velocity_threshold').value
        self.stuck_displacement_threshold = self.get_parameter('stuck_displacement_threshold').value
        
        # State variables
        self.current_pose = None
        self.map_data = None
        self.map_info = None
        self.exploring = True
        self.exit_found = False
        self.exit_position = None
        self.start_position = None
        
        # Laser scan data
        self.latest_scan = None
        self.scan_ranges = None
        
        # Initial mapping state
        self.initial_mapping_done = False
        self.initial_mapping_start_time = None
        self.last_log_time = 0
        
        # Stuck detection - TIME-BASED
        self.last_frontier_target = None
        self.same_frontier_count = 0
        self.stuck_threshold = 3
        self.position_history = []  # Stores (position, timestamp) tuples
        self.last_recovery_time = None
        self.min_recovery_interval = 12.0  # Increased from 10.0
        
        # Failed frontiers tracking
        self.failed_frontiers = []
        self.max_failed_attempts = 2
        
        # Goal handle for cancellation
        self._current_goal_handle = None
        
        # Goal tracking for stuck detection
        self.current_goal_sent_time = None
        self.min_time_before_stuck_check = 3.0  # Don't check stuck for 3s after goal sent
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
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
        
        # Timers
        self.initial_mapping_timer = self.create_timer(
            0.1,
            self.initial_mapping_callback
        )
        
        self.exploration_timer = self.create_timer(
            self.exploration_interval,
            self.exploration_callback
        )
        
        self.get_logger().info('Maze Solver Node initialized')
        self.get_logger().info('=' * 60)
        self.get_logger().info('PHASE 1: Initial 360° rotation for map building')
        self.get_logger().info(f'Duration: {self.initial_rotation_time} seconds')
        self.get_logger().info('=' * 60)
    
    def initial_mapping_callback(self):
        """Rotate robot initially to build a basic map"""
        if self.initial_mapping_done:
            self.initial_mapping_timer.cancel()
            return
        
        if self.initial_mapping_start_time is None:
            self.initial_mapping_start_time = self.get_clock().now()
        
        elapsed = (self.get_clock().now() - self.initial_mapping_start_time).nanoseconds / 1e9
        
        if elapsed < self.initial_rotation_time:
            twist = Twist()
            twist.angular.z = 0.5
            self.cmd_vel_pub.publish(twist)
            
            if int(elapsed) >= self.last_log_time + 5:
                self.last_log_time = int(elapsed)
                progress = (elapsed / self.initial_rotation_time) * 100
                self.get_logger().info(
                    f'Initial mapping progress: {int(elapsed)}s / {int(self.initial_rotation_time)}s '
                    f'({progress:.0f}%)'
                )
        else:
            twist = Twist()
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            
            self.initial_mapping_done = True
            self.get_logger().info('=' * 60)
            self.get_logger().info('✓ PHASE 1 COMPLETE: Initial mapping done!')
            self.get_logger().info('=' * 60)
            self.get_logger().info('PHASE 2: Starting frontier-based maze exploration...')
            self.get_logger().info('=' * 60)
    
    def odom_callback(self, msg):
        """Update current robot position with timestamp - FIXED for startup"""
        self.current_pose = msg.pose.pose
    
        if self.start_position is None:
            self.start_position = (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y
            )
            self.get_logger().info(
                f'Start position recorded: ({self.start_position[0]:.3f}, '
                f'{self.start_position[1]:.3f})'
            )
    
        # Store position with timestamp
        current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        current_time = self.get_clock().now()
    
        # Add to history
        self.position_history.append((current_pos, current_time))
    
        # Clean old data only if we have enough samples
        # This prevents negative time errors at simulation start
        if len(self.position_history) > 20:
            try:
                # Calculate cutoff time
                cutoff_duration = Duration(seconds=self.stuck_check_duration + 1.0)
                cutoff_time = current_time - cutoff_duration
            
                # Remove positions older than cutoff
                self.position_history = [
                    (pos, t) for pos, t in self.position_history 
                    if t > cutoff_time
                ]
            except ValueError:
                # At very start of simulation, time might be too early
                # Just keep all history for now
                pass
    
        # Safety limit: keep max 200 samples (4 seconds at 50Hz)
        if len(self.position_history) > 200:
            self.position_history = self.position_history[-200:]
    
    def map_callback(self, msg):
        """Process updated map from SLAM"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        
        if not self.exit_found and self.current_pose is not None and self.initial_mapping_done:
            self.check_for_exit()
    
    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        self.latest_scan = msg
        self.scan_ranges = np.array(msg.ranges)
        self.scan_ranges[np.isinf(self.scan_ranges)] = msg.range_max
    
    def is_goal_in_map_bounds(self, goal_x, goal_y):
        """
        Check if a goal position is within the map bounds
        FIXED: Off-by-one error corrected
        """
        if self.map_info is None:
            return False
        
        # Convert world coordinates to map coordinates
        mx = int((goal_x - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((goal_y - self.map_info.origin.position.y) / self.map_info.resolution)
        
        # Check bounds with safety margin
        # CRITICAL FIX: Valid indices are 0 to (width-1) and 0 to (height-1)
        margin = 3
        is_valid = (margin <= mx < (self.map_info.width - margin) and 
                   margin <= my < (self.map_info.height - margin))
        
        if not is_valid:
            self.get_logger().debug(
                f'Goal ({goal_x:.2f}, {goal_y:.2f}) -> map({mx}, {my}) '
                f'is outside bounds (0-{self.map_info.width-1}, 0-{self.map_info.height-1})'
            )
        
        return is_valid
    
    def is_frontier_failed(self, frontier_x, frontier_y):
        """Check if this frontier has failed too many times"""
        for (fx, fy, count) in self.failed_frontiers:
            distance = math.sqrt((fx - frontier_x)**2 + (fy - frontier_y)**2)
            if distance < 0.3:
                return count >= self.max_failed_attempts
        return False
    
    def mark_frontier_failed(self, frontier_x, frontier_y):
        """Mark a frontier as failed"""
        for i, (fx, fy, count) in enumerate(self.failed_frontiers):
            distance = math.sqrt((fx - frontier_x)**2 + (fy - frontier_y)**2)
            if distance < 0.3:
                self.failed_frontiers[i] = (fx, fy, count + 1)
                self.get_logger().warn(
                    f'Frontier ({fx:.2f}, {fy:.2f}) failed {count + 1}/{self.max_failed_attempts} times'
                )
                return
        
        # New failure
        self.failed_frontiers.append((frontier_x, frontier_y, 1))
        self.get_logger().info(
            f'Marked frontier ({frontier_x:.2f}, {frontier_y:.2f}) as failed (1/{self.max_failed_attempts})'
        )
    
    def check_for_exit(self):
        """Detect maze exit"""
        if self.map_data is None or self.map_info is None:
            return
        
        free_spaces = np.where(self.map_data == 0)
        
        if len(free_spaces[0]) == 0:
            return
        
        for i in range(0, len(free_spaces[0]), 10):
            grid_y = free_spaces[0][i]
            grid_x = free_spaces[1][i]
            
            world_x = grid_x * self.map_info.resolution + self.map_info.origin.position.x
            world_y = grid_y * self.map_info.resolution + self.map_info.origin.position.y
            
            if self.start_position:
                distance_from_start = math.sqrt(
                    (world_x - self.start_position[0])**2 + 
                    (world_y - self.start_position[1])**2
                )
                
                is_near_edge = (
                    grid_x < 5 or grid_x > self.map_info.width - 5 or
                    grid_y < 5 or grid_y > self.map_info.height - 5
                )
                
                if distance_from_start > self.exit_threshold and is_near_edge:
                    if self.is_open_area(grid_x, grid_y):
                        self.exit_found = True
                        self.exit_position = (world_x, world_y)
                        self.get_logger().info('=' * 60)
                        self.get_logger().info('🎯 EXIT FOUND!')
                        self.get_logger().info(f'Exit location: ({world_x:.2f}, {world_y:.2f})')
                        self.get_logger().info(f'Distance from start: {distance_from_start:.2f}m')
                        self.get_logger().info('=' * 60)
                        self.navigate_to_exit()
                        return
    
    def is_open_area(self, x, y, radius=3):
        """Check if area around point is open"""
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
    
    def is_robot_stuck(self):
        """
        TIME-BASED stuck detection
        Checks actual movement over specified time duration
        Much more accurate than sample-based approach
        """
        # Need at least 20 position samples
        if len(self.position_history) < 20:
            return False
        
        # Don't check stuck immediately after sending a goal
        if self.current_goal_sent_time is not None:
            time_since_goal = (self.get_clock().now() - self.current_goal_sent_time).nanoseconds / 1e9
            if time_since_goal < self.min_time_before_stuck_check:
                return False
        
        # Get oldest and newest positions within time window
        current_time = self.get_clock().now()
        cutoff_time = current_time - Duration(seconds=self.stuck_check_duration)
        
        # Filter positions within time window
        recent_positions = [
            (pos, t) for pos, t in self.position_history
            if t > cutoff_time
        ]
        
        if len(recent_positions) < 10:
            return False
        
        # Get first and last position
        oldest_pos, oldest_time = recent_positions[0]
        newest_pos, newest_time = recent_positions[-1]
        
        # Calculate total displacement (straight-line distance)
        total_displacement = math.sqrt(
            (newest_pos[0] - oldest_pos[0])**2 +
            (newest_pos[1] - oldest_pos[1])**2
        )
        
        # Calculate actual time elapsed
        time_elapsed = (newest_time - oldest_time).nanoseconds / 1e9
        
        # Avoid division by zero
        if time_elapsed < 0.1:
            return False
        
        # Calculate average velocity
        velocity = total_displacement / time_elapsed
        
        # Calculate path length (sum of all segments)
        path_length = 0.0
        for i in range(len(recent_positions) - 1):
            pos1, _ = recent_positions[i]
            pos2, _ = recent_positions[i + 1]
            segment_length = math.sqrt(
                (pos2[0] - pos1[0])**2 +
                (pos2[1] - pos1[1])**2
            )
            path_length += segment_length
        
        # Robot is stuck if ALL conditions are met:
        # 1. Have monitored for at least 80% of check duration
        # 2. Velocity is very low
        # 3. Total displacement is small
        # 4. Path length is also small (not just spinning)
        min_monitoring_time = self.stuck_check_duration * 0.8
        
        is_stuck = (
            time_elapsed >= min_monitoring_time and
            velocity < self.stuck_velocity_threshold and
            total_displacement < self.stuck_displacement_threshold and
            path_length < (self.stuck_displacement_threshold * 1.5)
        )
        
        if is_stuck:
            self.get_logger().warn(
                f'🚫 Robot stuck: displacement={total_displacement:.3f}m, '
                f'path={path_length:.3f}m, velocity={velocity:.3f}m/s '
                f'over {time_elapsed:.1f}s (threshold: {self.stuck_velocity_threshold}m/s, '
                f'{self.stuck_displacement_threshold}m)'
            )
        
        return is_stuck
    
    def should_attempt_recovery(self):
        """
        Check if enough time has passed since last recovery
        Prevents recovery spam
        """
        if self.last_recovery_time is None:
            return True
        
        current_time = self.get_clock().now()
        time_since_recovery = (current_time - self.last_recovery_time).nanoseconds / 1e9
        
        if time_since_recovery < self.min_recovery_interval:
            self.get_logger().debug(
                f'Recovery cooldown: {time_since_recovery:.1f}s / {self.min_recovery_interval}s'
            )
            return False
        
        return True
    
    def exploration_callback(self):
        """Main exploration loop"""
        if not self.initial_mapping_done:
            return
        
        if self.exit_found:
            return
        
        if self.map_data is None or self.current_pose is None:
            return
        
        # Check if robot is stuck AND enough time since last recovery
        if self.is_robot_stuck() and self.should_attempt_recovery():
            self.get_logger().warn('⚠️ Robot stuck! Executing recovery...')
            
            # Mark current frontier as failed
            if self.last_frontier_target is not None:
                self.mark_frontier_failed(
                    self.last_frontier_target[0],
                    self.last_frontier_target[1]
                )
                self.last_frontier_target = None
            
            self.recovery_behavior()
            return
        
        # Find frontiers
        frontiers = self.find_frontiers()
        
        if len(frontiers) == 0:
            self.get_logger().warn('No frontiers found - random exploration')
            self.random_exploration()
            return
        
        # Select best valid frontier
        target_frontier = self.select_best_frontier_with_filter(frontiers)
        
        if target_frontier is None:
            self.get_logger().warn('No valid frontiers - random exploration')
            self.random_exploration()
            return
        
        # Check if targeting same frontier repeatedly
        if self.last_frontier_target is not None:
            distance_to_last = math.sqrt(
                (target_frontier[0] - self.last_frontier_target[0])**2 +
                (target_frontier[1] - self.last_frontier_target[1])**2
            )
            
            if distance_to_last < 0.3:
                self.same_frontier_count += 1
                self.get_logger().warn(
                    f'⚠️ Same frontier {self.same_frontier_count}/{self.stuck_threshold} times: '
                    f'({target_frontier[0]:.2f}, {target_frontier[1]:.2f})'
                )
                
                if self.same_frontier_count >= self.stuck_threshold:
                    self.get_logger().warn('❌ Frontier unreachable! Marking as failed.')
                    self.mark_frontier_failed(target_frontier[0], target_frontier[1])
                    self.same_frontier_count = 0
                    self.last_frontier_target = None
                    return
            else:
                self.same_frontier_count = 0
        
        self.last_frontier_target = target_frontier
        self.navigate_to_frontier(target_frontier)
    
    def find_frontiers(self):
        """Find frontier cells"""
        if self.map_data is None:
            return []
        
        free_cells = (self.map_data == 0)
        unknown_cells = (self.map_data == -1)
        dilated_free = binary_dilation(free_cells, iterations=1)
        frontiers = unknown_cells & dilated_free
        
        from scipy.ndimage import label
        labeled_frontiers, num_features = label(frontiers)
        
        frontier_centroids = []
        
        for i in range(1, num_features + 1):
            region = (labeled_frontiers == i)
            size = np.sum(region)
            
            if size >= self.frontier_min_size:
                coords = np.where(region)
                centroid_y = int(np.mean(coords[0]))
                centroid_x = int(np.mean(coords[1]))
                
                world_x = centroid_x * self.map_info.resolution + self.map_info.origin.position.x
                world_y = centroid_y * self.map_info.resolution + self.map_info.origin.position.y
                
                frontier_centroids.append((world_x, world_y, size))
        
        if len(frontier_centroids) > 0:
            self.get_logger().info(f'Found {len(frontier_centroids)} frontiers')
        
        return frontier_centroids
    
    def select_best_frontier_with_filter(self, frontiers):
        """
        Select best frontier
        Filters out: too close, failed, outside bounds
        """
        if not frontiers or self.current_pose is None:
            return None
        
        best_frontier = None
        best_score = float('-inf')
        
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        valid_frontiers = []
        
        for fx, fy, size in frontiers:
            # Filter 1: Check if previously failed
            if self.is_frontier_failed(fx, fy):
                self.get_logger().debug(f'Skipping failed frontier ({fx:.2f}, {fy:.2f})')
                continue
            
            # Filter 2: Check distance
            distance = math.sqrt((fx - current_x)**2 + (fy - current_y)**2)
            if distance < self.min_frontier_distance:
                continue
            
            # Filter 3: Check if within map bounds
            if not self.is_goal_in_map_bounds(fx, fy):
                continue
            
            valid_frontiers.append((fx, fy, size, distance))
        
        if not valid_frontiers:
            self.get_logger().warn(
                f'All {len(frontiers)} frontiers filtered out '
                f'(failed/close/out-of-bounds)'
            )
            return None
        
        # Score valid frontiers
        for fx, fy, size, distance in valid_frontiers:
            score = size / (distance + 1.0)
            
            if score > best_score:
                best_score = score
                best_frontier = (fx, fy)
        
        return best_frontier
    
    def navigate_to_frontier(self, frontier):
        """Send navigation goal with fresh timestamp"""
        # Cancel previous goal
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = frontier[0]
        goal_msg.pose.pose.position.y = frontier[1]
        goal_msg.pose.pose.orientation.w = 1.0
        
        # Record when goal was sent (for stuck detection)
        self.current_goal_sent_time = self.get_clock().now()
        
        self.get_logger().info(f'→ Navigating to frontier at ({frontier[0]:.2f}, {frontier[1]:.2f})')
        
        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_sent_callback)
    
    def _goal_sent_callback(self, future):
        """Store goal handle"""
        self._current_goal_handle = future.result()
        if not self._current_goal_handle.accepted:
            self.get_logger().warn('Goal rejected!')
            self._current_goal_handle = None
    
    def random_exploration(self):
        """
        Move to random nearby point
        Only generates goals within map bounds AND on free space
        """
        if self.current_pose is None or self.map_info is None:
            return
        
        max_attempts = 20
        for attempt in range(max_attempts):
            # Generate random angle
            angle = random.uniform(0, 2 * math.pi)
            # Shorter distance for tight mazes
            distance = random.uniform(0.3, 0.8)
            
            goal_x = self.current_pose.position.x + distance * math.cos(angle)
            goal_y = self.current_pose.position.y + distance * math.sin(angle)
            
            # Check if within bounds AND on free space
            if self.is_goal_in_map_bounds(goal_x, goal_y):
                # Convert to map coordinates to check if free
                mx = int((goal_x - self.map_info.origin.position.x) / self.map_info.resolution)
                my = int((goal_y - self.map_info.origin.position.y) / self.map_info.resolution)
                
                # Check if free space
                if (0 <= mx < self.map_info.width and 
                    0 <= my < self.map_info.height and
                    self.map_data[my, mx] == 0):
                    
                    self.get_logger().info(
                        f'🔀 Random exploration to ({goal_x:.2f}, {goal_y:.2f}) '
                        f'[attempt {attempt+1}]'
                    )
                    
                    # Cancel previous goal
                    if self._current_goal_handle is not None:
                        self._current_goal_handle.cancel_goal_async()
                    
                    goal_msg = NavigateToPose.Goal()
                    goal_msg.pose.header.frame_id = 'map'
                    goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
                    goal_msg.pose.pose.position.x = goal_x
                    goal_msg.pose.pose.position.y = goal_y
                    goal_msg.pose.pose.orientation.w = 1.0
                    
                    # Record when goal was sent
                    self.current_goal_sent_time = self.get_clock().now()
                    
                    self.nav_client.wait_for_server()
                    future = self.nav_client.send_goal_async(goal_msg)
                    future.add_done_callback(self._goal_sent_callback)
                    return
        
        self.get_logger().error(
            f'❌ Failed to find valid random goal after {max_attempts} attempts!'
        )
        
        # Last resort: just spin
        self.get_logger().info('Last resort: Spinning 360°...')
        self.safe_spin(angle=2 * math.pi)
    
    def recovery_behavior(self):
        """
        Enhanced recovery behavior for tight spaces
        """
        self.get_logger().info('🔄 Executing enhanced recovery behavior...')
        
        # Record recovery time
        self.last_recovery_time = self.get_clock().now()
        
        # Reset counters
        self.same_frontier_count = 0
        self.position_history.clear()
        
        # Cancel current goal
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
        
        # Stop robot
        self.stop_robot()
        time.sleep(0.5)
        
        # Check space around robot
        if self.scan_ranges is not None and len(self.scan_ranges) > 0:
            # Find direction with most space
            num_readings = len(self.scan_ranges)
            best_direction_idx = 0
            max_space = 0
            
            # Check 8 directions (every 45 degrees)
            for i in range(8):
                idx = int((i / 8.0) * num_readings)
                # Sample 5 readings around this direction
                start = max(0, idx - 2)
                end = min(num_readings, idx + 3)
                space = np.mean(self.scan_ranges[start:end])
                
                if space > max_space:
                    max_space = space
                    best_direction_idx = idx
            
            # Calculate angle to turn
            angle_to_turn = (best_direction_idx / num_readings) * 2 * math.pi
            # Normalize to [-pi, pi]
            if angle_to_turn > math.pi:
                angle_to_turn -= 2 * math.pi
            
            self.get_logger().info(
                f'  🔄 Turning {math.degrees(angle_to_turn):.0f}° toward open space '
                f'({max_space:.2f}m clearance)'
            )
            
            # Turn toward most open direction
            self.safe_spin(angle=angle_to_turn)
            time.sleep(0.5)
            
            # Try to back up if stuck and space behind
            if max_space < 0.5 and self.is_path_clear_behind(0.25):
                self.get_logger().info('  ⬅️ Backing up 0.2m...')
                self.safe_backup(distance=0.2)
        else:
            # No scan data, just spin
            self.get_logger().info('  🔄 Spinning 180°...')
            self.safe_spin(angle=math.pi)
        
        self.stop_robot()
        self.get_logger().info('✓ Recovery complete')
    
    def stop_robot(self):
        """Stop all movement"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
    
    def safe_spin(self, angle=1.57):
        """Spin robot by specified angle"""
        angular_speed = 0.4
        duration = abs(angle / angular_speed)
        
        twist = Twist()
        twist.angular.z = angular_speed if angle > 0 else -angular_speed
        
        start_time = time.time()
        while time.time() - start_time < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)
        
        self.stop_robot()
    
    def is_path_clear_behind(self, distance=0.2):
        """Check if path behind robot is clear"""
        if self.latest_scan is None or self.scan_ranges is None:
            return False
        
        num_readings = len(self.scan_ranges)
        # Check rear 90° arc
        rear_start = int(num_readings * 0.375)
        rear_end = int(num_readings * 0.625)
        
        rear_readings = self.scan_ranges[rear_start:rear_end]
        min_rear_distance = np.min(rear_readings)
        
        return min_rear_distance > distance
    
    def safe_backup(self, distance=0.2):
        """Back up robot safely"""
        if not self.is_path_clear_behind(distance + 0.05):
            self.get_logger().warn('Cannot backup - obstacle behind!')
            return
        
        duration = distance / self.backup_speed
        
        twist = Twist()
        twist.linear.x = -self.backup_speed
        
        start_time = time.time()
        while time.time() - start_time < duration:
            if not self.is_path_clear_behind(0.15):
                self.get_logger().warn('⚠️ Obstacle during backup! Stopping.')
                break
            
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)
        
        self.stop_robot()
    
    def navigate_to_exit(self):
        """Navigate to exit"""
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
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('PHASE 3: NAVIGATING TO EXIT!')
        self.get_logger().info(f'Exit: ({self.exit_position[0]:.2f}, {self.exit_position[1]:.2f})')
        self.get_logger().info('=' * 60)
        
        self.nav_client.wait_for_server()
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejected')
            return
        
        self.get_logger().info('✓ Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        """Handle goal result"""
        self.get_logger().info('=' * 60)
        self.get_logger().info('🎉 MAZE SOLVED! 🎉')
        self.get_logger().info('=' * 60)


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
