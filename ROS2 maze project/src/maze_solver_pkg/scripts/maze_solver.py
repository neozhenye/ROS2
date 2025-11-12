#!/usr/bin/env python3

"""
Maze Solver Algorithm - FIXED VERSION
==============================================
Creator : Neo Zhen Ye
Admin ID : 2402759 

FIXES:
- Properly handles coordinate frame conversion for exit alignment
- Robot aligns to face exit with laser -20° to +20° pointing at free space
- Then drives forward 1m through exit

what code does in general:
1. Continuously seeks map edges/boundaries
2. Exit detection: ≥60° opening, all readings ≥3.5m clear
3. When exit found, align to exit and drive forward 1m through exit
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from action_msgs.msg import GoalStatus
import numpy as np
import math
import time

class MazeSolver(Node):
    def __init__(self):
        super().__init__('maze_solver')
        
        #parameters
        
        #Edge seeking
        self.declare_parameter('exploration_interval', 6.0)
        self.declare_parameter('goal_timeout', 25.0)
        self.declare_parameter('map_ready_threshold', 100)
        self.declare_parameter('min_goal_distance', 0.3)
        self.declare_parameter('edge_search_directions', 12)
        self.declare_parameter('edge_goal_distance', 1.3)
        
        #exit detection
        self.declare_parameter('exit_min_depth', 3.5)  #depth needed for a goal
        self.declare_parameter('exit_min_angle', 60.0)  #how many degrees of depth clearance : 60deg
        
        #exit approach - distance to drive forward through exit
        self.declare_parameter('exit_drive_distance', 1.0)  #drive 1.0m forward through exit
        self.declare_parameter('exit_align_tolerance', 20.0)  #±20 degrees alignment tolerance
        
        #load parameters
        self.exploration_interval = self.get_parameter('exploration_interval').value
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.map_ready_threshold = self.get_parameter('map_ready_threshold').value
        self.min_goal_distance = self.get_parameter('min_goal_distance').value
        self.edge_search_directions = self.get_parameter('edge_search_directions').value
        self.edge_goal_distance = self.get_parameter('edge_goal_distance').value
        
        self.exit_min_depth = self.get_parameter('exit_min_depth').value
        self.exit_min_angle = self.get_parameter('exit_min_angle').value
        self.exit_drive_distance = self.get_parameter('exit_drive_distance').value
        self.exit_align_tolerance = self.get_parameter('exit_align_tolerance').value
        
        #robot state
        self.current_pose = None
        self.start_position = None
        
        #sensor data
        self.latestScan = None
        self.scanRanges = None
        self.mapData = None
        self.mapInfo = None
        self.mapReady = False
        
        #phase tracking
        self.exit_found = False
        self.exit_approach_goal_sent = False
        self.exit_direction_rad = None  # Exit direction in robot's laser frame (relative)
        self.exit_target_yaw = None  # Target yaw in global map frame (absolute)
        self.exit_width = None
        self.exit_depth = None
        self.maze_completed = False
        self.driving_through_exit = False
        self.aligning_to_exit = False
        self.alignment_start_time = None  # Track when alignment phase actually starts
        self.drive_start_time = None
        self.drive_start_position = None
        
        #exploration state
        self.current_direction_index = 0
        self.exploration_directions = [i * (360.0 / self.edge_search_directions) 
                                       for i in range(self.edge_search_directions)]
        self.failed_goal_attempts = 0
        
        #goal tracking
        self._goal_active = False
        self._goal_start_time = None
        self._current_goal_handle = None
        self.last_goal_position = None
        
        #publishers, subscriber, and action client
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        #timers 
        self.exploration_timer = self.create_timer(self.exploration_interval, self.exploration_callback)
        self.exit_check_timer = self.create_timer(0.5, self.check_for_exit)
        self.exit_drive_timer = self.create_timer(0.1, self.exit_drive_callback)
        
        #startup msg
        
        self.get_logger().info('=' * 80)
        self.get_logger().info('MAZE SOLVER - DIRECT EXIT DRIVING VERSION')
        self.get_logger().info('Strategy: Edge-seeking, then align and drive through exit')
        self.get_logger().info('Starting exploration immediately!')
        self.get_logger().info('=' * 80)

    #callbacks
    
    def map_callback(self, msg):
        """Process incoming map from SLAM"""
        self.mapData = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.mapInfo = msg.info
        
        free_cells = np.sum(self.mapData == 0)
        unknown_cells = np.sum(self.mapData == -1)
        
        was_ready = self.mapReady
        self.mapReady = free_cells >= self.map_ready_threshold
        
        if self.mapReady and not was_ready:
            self.get_logger().info(f'Map ready! Free: {free_cells} | Unknown: {unknown_cells}')
    
    def odom_callback(self, msg):
        """Track robot position"""
        self.current_pose = msg.pose.pose
        
        if self.start_position is None:
            self.start_position = (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y
            )
            self.get_logger().info(f'Start: ({self.start_position[0]:.3f}, {self.start_position[1]:.3f})')
    
    def scan_callback(self, msg):
        """Process laser scan data"""
        self.latestScan = msg
        self.scanRanges = np.array(msg.ranges)
        self.scanRanges[np.isinf(self.scanRanges)] = msg.range_max
        self.scanRanges[np.isnan(self.scanRanges)] = 0.0
    
    #exploration with edge seeking
    
    def exploration_callback(self):
        """continuously seek map edges"""
        
        if not self.mapReady:
            return
        
        if self.exit_found or self.maze_completed:
            return
        
        if self.current_pose is None or self.start_position is None:
            return
        
        if self._goal_active:
            if self._goal_start_time is not None:
                elapsed = (self.get_clock().now() - self._goal_start_time).nanoseconds / 1e9
                if elapsed > self.goal_timeout:
                    self.get_logger().warn(f'⏱️  Goal timeout - cancelling')
                    if self._current_goal_handle is not None:
                        self._current_goal_handle.cancel_goal_async()
                    self._goal_active = False
                else:
                    return
            else:
                return
        
        target = self.find_edge_goal()
        
        if target is None:
            self.failed_goal_attempts += 1
            self.get_logger().warn(f'No valid edge goals found (attempt {self.failed_goal_attempts})')
            
            if self.failed_goal_attempts >= 3:
                target = self.generate_random_goal()
                if target is not None:
                    self.get_logger().info('Using random exploration goal')
                    self.failed_goal_attempts = 0
                else:
                    return
            else:
                return
        else:
            self.failed_goal_attempts = 0
        
        self.navigate_to_target(target)
    
    def find_edge_goal(self):
        """find a goal position heading toward map edges"""
        
        attempts = 0
        max_attempts = self.edge_search_directions * 2
        
        while attempts < max_attempts:
            angle_deg = self.exploration_directions[self.current_direction_index]
            self.current_direction_index = (self.current_direction_index + 1) % len(self.exploration_directions)
            attempts += 1
            
            angle_rad = math.radians(angle_deg)
            distance = self.edge_goal_distance
            
            for dist_attempt in range(10):
                test_distance = distance - (dist_attempt * 0.15)
                if test_distance < self.min_goal_distance:
                    break
                
                target_x = self.current_pose.position.x + test_distance * math.cos(angle_rad)
                target_y = self.current_pose.position.y + test_distance * math.sin(angle_rad)
                
                if self.is_valid_edge_goal(target_x, target_y):
                    dist_from_robot = math.sqrt(
                        (target_x - self.current_pose.position.x)**2 +
                        (target_y - self.current_pose.position.y)**2
                    )
                    
                    if dist_from_robot >= self.min_goal_distance:
                        if self.last_goal_position is not None:
                            dist_from_last = math.sqrt(
                                (target_x - self.last_goal_position[0])**2 +
                                (target_y - self.last_goal_position[1])**2
                            )
                            if dist_from_last < 0.25:
                                continue
                        
                        return (target_x, target_y)
        
        return None
    
    def is_valid_edge_goal(self, x, y):
        """check if position is valid - lenient to embrace unknown space"""
        if not self.mapReady or self.mapInfo is None or self.mapData is None:
            return False
        # turn meters into cells
        mx = int((x - self.mapInfo.origin.position.x) / self.mapInfo.resolution)
        my = int((y - self.mapInfo.origin.position.y) / self.mapInfo.resolution)
        #prevent goal from being set outside the map
        margin = 1
        if not (margin <= mx < (self.mapInfo.width - margin) and margin <= my < (self.mapInfo.height - margin)):
            return False
        #check if target cell is occupied, less than 70% chance of obstacle
        cell_value = self.mapData[my, mx]
        if cell_value > 70:
            return False
        #below is to check nearby cells 3x3 grid, check if nearby cells are occupied and make sure at least 60% of surrounding cells are free
        check_radius = 1
        occupied_count = 0
        total_checked = 0
        
        for dy in range(-check_radius, check_radius + 1):
            for dx in range(-check_radius, check_radius + 1):
                cx, cy = mx + dx, my + dy
                if (0 <= cx < self.mapInfo.width and 0 <= cy < self.mapInfo.height):
                    total_checked += 1
                    if self.mapData[cy, cx] > 70:
                        occupied_count += 1
        
        if total_checked > 0 and (occupied_count / total_checked) > 0.4:
            return False
        
        return True
    
    def generate_random_goal(self):
        """generate a random goal for exploration fallback"""
        if self.current_pose is None:
            return None
        
        for _ in range(10):
            angle_rad = np.random.uniform(0, 2 * math.pi)
            distance = np.random.uniform(0.5, 1.2)
            
            target_x = self.current_pose.position.x + distance * math.cos(angle_rad)
            target_y = self.current_pose.position.y + distance * math.sin(angle_rad)
            
            if self.is_valid_edge_goal(target_x, target_y):
                return (target_x, target_y)
        
        return None
    #send goal via action server
    def navigate_to_target(self, target):
        """Send navigation goal to Nav2"""
        target_x, target_y = target
        
        self.get_logger().info(
            f'Sending goal to position: ({target_x:.2f}, {target_y:.2f})'
        )
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = target_x
        goal_msg.pose.pose.position.y = target_y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0
        
        self.last_goal_position = (target_x, target_y)
        self._goal_active = True
        self._goal_start_time = self.get_clock().now()
        
        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)

    # exit detection and final goal
    
    def check_for_exit(self):
        """Detect exits and drive forward through exit"""
        
        if not self.mapReady or self.exit_found or self.maze_completed:
            return
        
        if self.latestScan is None or self.current_pose is None or self.start_position is None:
            return
        
        num_readings = len(self.scanRanges) #360
        angle_increment = self.latestScan.angle_increment
        
        best_exit = None
        best_score = 0
        
        i = 0
        while i < num_readings:
            if self.scanRanges[i] < self.exit_min_depth:
                i += 1 #tracks how much of the laser reading is less than 3.5m
                continue

            opening_start = i
            opening_readings = []
            
            while i < num_readings and self.scanRanges[i] >= self.exit_min_depth:
                opening_readings.append(self.scanRanges[i])
                i += 1 #tracks number of laser reading above 3.5m
            
            opening_end = i - 1 #Identify which part of the 360 scan is the exit
            
            if len(opening_readings) < 5:
                continue
            
            opening_angle_rad = (opening_end - opening_start) * angle_increment
            opening_angle_deg = math.degrees(opening_angle_rad)
            
            if opening_angle_deg >= self.exit_min_angle: # if it pass the 65degs test, try to get the center of the exit
                min_depth = np.min(opening_readings)
                avg_depth = np.mean(opening_readings)
                max_depth = np.max(opening_readings)
                
                score = opening_angle_deg * 0.6 + avg_depth * 0.4
                
                if score > best_score:
                    best_score = score
                    center_idx = (opening_start + opening_end) // 2
                    center_angle_rad = self.latestScan.angle_min + center_idx * angle_increment
                    best_exit = {
                        'direction_rad': center_angle_rad,
                        'angle_span': opening_angle_deg,
                        'min_depth': min_depth,
                        'avg_depth': avg_depth,
                        'max_depth': max_depth,
                        'num_readings': len(opening_readings)
                    }
    
        if best_exit is not None:
            self.exit_found = True
            # Exit direction in robot's laser frame (relative to robot's forward direction)
            self.exit_direction_rad = best_exit['direction_rad']
            self.exit_width = best_exit['angle_span']
            self.exit_depth = best_exit['avg_depth']
            
            # calculate target yaw in global map frame
            # Get current robot orientation in global frame
            current_yaw = self.get_yaw_from_pose(self.current_pose)
            # Add the relative exit direction to get absolute target yaw
            self.exit_target_yaw = self.normalize_angle(current_yaw + self.exit_direction_rad)
            
            self.get_logger().info('EXIT DETECTED (STRICT CRITERIA MET)!')
            
            # cancel timers FIRST to prevent new goals from being sent
            self.exploration_timer.cancel()
            self.exit_check_timer.cancel()
            
            # Then cancel any active navigation goals
            self.cancel_all_goals()
            
            self.start_driving_through_exit()

    def cancel_all_goals(self):
        """Cancel all active goals and stop Nav2 completely"""
        self.get_logger().info('🛑 STOPPING ALL NAVIGATION - Taking manual control!')
        
        # Cancel our navigation goal if active
        if self._current_goal_handle is not None:
            self.get_logger().info('  ↳ Cancelling maze solver navigation goal...')
            self._current_goal_handle.cancel_goal_async()
            self._goal_active = False
            self._current_goal_handle = None
        
        # STOP the robot completely before switching to manual control
        # This prevents Nav2 from continuing to send commands
        stop_twist = Twist()
        for _ in range(10):  # Publish stop command multiple times to ensure it's received
            self.cmd_vel_pub.publish(stop_twist)
            time.sleep(0.05)  # 50ms delay
        
        self.get_logger().info('  ↳ Robot stopped, ready for manual control')
        self.get_logger().info('  ↳ Nav2 will no longer interfere with alignment/driving')

    def start_driving_through_exit(self):
        """Start the process of driving through the exit"""
        self.driving_through_exit = True
        self.aligning_to_exit = True
        self.drive_start_time = self.get_clock().now()
        self.drive_start_position = (self.current_pose.position.x, self.current_pose.position.y)
        self.alignment_start_time = None  # Track when alignment actually starts
        
        self.get_logger().info('=' * 80)
        self.get_logger().info('EXIT DRIVE SEQUENCE STARTED')
        self.get_logger().info('Phase 0: Waiting for Nav2 to fully stop (0.5s)...')
        self.get_logger().info('Phase 1: Aligning robot to face exit direction')
        self.get_logger().info('Phase 2: Driving forward 1.0m through exit')
        self.get_logger().info('=' * 80)

    def exit_drive_callback(self):
        """Handle driving through exit - called frequently during exit drive"""
        if not self.driving_through_exit or self.maze_completed:
            return
        
        if self.current_pose is None:
            return
        
        # Check if robot has exited the map bounds (SUCCESS!)
        if self.is_outside_map_bounds():
            self.get_logger().info('=' * 80)
            self.get_logger().info('🎉 ROBOT HAS EXITED THE MAP! 🎉')
            self.get_logger().info('Detected robot outside original map boundaries')
            self.get_logger().info('=' * 80)
            self.complete_mission()
            return
        
        twist = Twist()
        
        # wait for Nav2 to fully stop (0.5 second settling period)
        if self.aligning_to_exit and self.alignment_start_time is None:
            elapsed = (self.get_clock().now() - self.drive_start_time).nanoseconds / 1e9
            if elapsed < 0.5:
                # Keep publishing stop command during settling period
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                return
            else:
                # Settling period complete, start alignment
                self.alignment_start_time = self.get_clock().now()
                self.get_logger().info('✓ Nav2 settled. Starting alignment now!')
        
        #align to face the exit direction
        if self.aligning_to_exit:
            # Get current robot orientation in global frame
            current_yaw = self.get_yaw_from_pose(self.current_pose)
            
            # Calculate angle difference to target yaw (both in global frame now!)
            angle_diff = self.normalize_angle(self.exit_target_yaw - current_yaw)
            
            # Check if aligned (within tolerance)
            if abs(angle_diff) <= math.radians(self.exit_align_tolerance):
                self.get_logger().info('=' * 80)
                self.get_logger().info('✓ ALIGNMENT COMPLETE!')
                self.get_logger().info(f'  Current yaw: {math.degrees(current_yaw):.1f}°')
                self.get_logger().info(f'  Target yaw: {math.degrees(self.exit_target_yaw):.1f}°')
                self.get_logger().info(f'  Error: {math.degrees(angle_diff):.1f}° (within ±{self.exit_align_tolerance}°)')
                self.get_logger().info('  Laser -20° to +20° now facing exit!')
                self.get_logger().info('  Starting forward drive...')
                self.get_logger().info('=' * 80)
                
                # Stop rotation
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                
                # Switch to driving phase
                self.aligning_to_exit = False
                self.drive_start_position = (self.current_pose.position.x, self.current_pose.position.y)
                return
            
            # Rotate towards target yaw
            # Use proportional control for smoother rotation
            rotation_speed = 0.3
            if abs(angle_diff) < math.radians(45):  # Slow down near target
                rotation_speed = 0.15
            
            twist.linear.x = 0.0
            twist.angular.z = rotation_speed if angle_diff > 0 else -rotation_speed
            
            self.get_logger().info(
                f'Aligning: current={math.degrees(current_yaw):.1f}° → target={math.degrees(self.exit_target_yaw):.1f}° (diff={math.degrees(angle_diff):.1f}°)',
                throttle_duration_sec=1.0
            )
        
        # Phase 2: Drive forward after alignment
        else:
            current_x = self.current_pose.position.x
            current_y = self.current_pose.position.y
            distance_traveled = math.sqrt(
                (current_x - self.drive_start_position[0])**2 + 
                (current_y - self.drive_start_position[1])**2
            )
            
            if distance_traveled >= self.exit_drive_distance:
                self.get_logger().info('=' * 80)
                self.get_logger().info(f'✓ DRIVE COMPLETE! Traveled {distance_traveled:.2f}m')
                self.get_logger().info('=' * 80)
                self.complete_mission()
                return
            
            # Drive straight forward
            twist.linear.x = 0.2
            twist.angular.z = 0.0
            
            # Log progress every 0.25m
            if int(distance_traveled * 4) != int((distance_traveled - 0.05) * 4):
                self.get_logger().info(f'📍 Driving through exit... {distance_traveled:.2f}/{self.exit_drive_distance}m')
        
        self.cmd_vel_pub.publish(twist)

    def is_outside_map_bounds(self):
        """Check if robot has driven outside the original mapped area"""
        if self.current_pose is None or self.mapInfo is None:
            return False
        
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        
        # Get map bounds
        map_min_x = self.mapInfo.origin.position.x
        map_min_y = self.mapInfo.origin.position.y
        map_max_x = map_min_x + self.mapInfo.width * self.mapInfo.resolution
        map_max_y = map_min_y + self.mapInfo.height * self.mapInfo.resolution
        
        # Add a small margin (0.2m) to avoid false positives from being near edge
        margin = 0.2
        
        if (x < (map_min_x - margin) or x > (map_max_x + margin) or
            y < (map_min_y - margin) or y > (map_max_y + margin)):
            return True
        
        return False

    def get_yaw_from_pose(self, pose):
        """Extract yaw angle from pose orientation quaternion"""
        x = pose.orientation.x
        y = pose.orientation.y
        z = pose.orientation.z
        w = pose.orientation.w
        
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return yaw

    def normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def complete_mission(self):
        """Final mission completion routine"""
        if self.maze_completed:  # Prevent duplicate completion
            return
            
        self.maze_completed = True
        self.driving_through_exit = False
        self.aligning_to_exit = False
        
        # Stop the robot
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        
        # Cancel the exit drive timer to stop callbacks
        if hasattr(self, 'exit_drive_timer'):
            self.exit_drive_timer.cancel()
        
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        distance_from_start = math.sqrt(
            (x - self.start_position[0])**2 + 
            (y - self.start_position[1])**2
        )
        
        self.get_logger().info('=' * 80)
        self.get_logger().info('║' + ' ' * 78 + '║')
        self.get_logger().info('║' + ' ' * 20 + '🎉 MAZE COMPLETED! 🎉' + ' ' * 20 + '║')
        self.get_logger().info('║' + ' ' * 10 + 'SUCCESSFULLY EXITED THE MAZE!' + ' ' * 10 + '║')
        self.get_logger().info('║' + ' ' * 78 + '║')
        self.get_logger().info('=' * 80)
        self.get_logger().info('')
        self.get_logger().info('📍 FINAL STATISTICS:')
        self.get_logger().info(f'   • Final position: ({x:.3f}, {y:.3f})')
        self.get_logger().info(f'   • Distance from start: {distance_from_start:.3f}m')
        if self.exit_width is not None and self.exit_depth is not None:
            self.get_logger().info(f'   • Exit detected: {self.exit_width:.1f}° opening, {self.exit_depth:.2f}m deep')
        self.get_logger().info(f'   • Status: Robot safely outside maze boundaries ✓')
        self.get_logger().info('')
        self.get_logger().info('=' * 80)
        self.get_logger().info('Note: "Robot out of bounds" warnings are expected and normal.')
        self.get_logger().info('They confirm the robot successfully exited the mapped area!')
        self.get_logger().info('=' * 80)
    
    # goal callbacks
    
    def _goal_response_callback(self, future):
        """Handle goal acceptance/rejection"""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            self._goal_active = False
            self._current_goal_handle = None
            return
        
        self._current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)
    
    def _goal_result_callback(self, future):
        """Handle goal completion"""
        result = future.result()
        status = result.status
        
        self._goal_active = False
        self._current_goal_handle = None
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('✓ Goal reached')
        else:
            self.get_logger().warn('✗ Goal failed')


def main(args=None):
    rclpy.init(args=args)
    maze_solver = MazeSolver()
    
    try:
        rclpy.spin(maze_solver)
    except KeyboardInterrupt:
        pass
    finally:
        twist = Twist()
        maze_solver.cmd_vel_pub.publish(twist)
        maze_solver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
