import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
    RegisterEventHandler
)
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # Paths
    world_file = os.path.join(
        os.path.expanduser('~'),
        'maze_ws',
        'src',
        'maze_solver_pkg',
        'worlds',
        'tb3_maze.world'
    )
    
    slam_params_file = os.path.join(
        os.path.expanduser('~'),
        'maze_ws',
        'src',
        'maze_solver_pkg',
        'config',
        'slam_params.yaml'
    )
    
    nav2_params_file = os.path.join(
        os.path.expanduser('~'),
        'maze_ws',
        'src',
        'maze_solver_pkg',
        'config',
        'nav2_params.yaml'
    )
    
    rviz_config_file = os.path.join(
        os.path.expanduser('~'),
        'maze_ws',
        'src',
        'maze_solver_pkg',
        'config',
        'maze_solver.rviz'
    )
    
    # Validate world file
    if not os.path.exists(world_file):
        print(f"\n{'='*60}")
        print(f"ERROR: World file not found at: {world_file}")
        print(f"{'='*60}\n")
        return LaunchDescription()
    
    # Get package directories
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    bringup_dir = get_package_share_directory('nav2_bringup')
    
    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Declare arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )
    
    #############################################################################
    # Gazebo Launch
    #############################################################################
    
    start_gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file, 'verbose': 'false'}.items()
    )
    
    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )
    
    #############################################################################
    # Robot State Publisher - START EARLY!
    #############################################################################
    
    urdf_file = os.path.join(
        pkg_turtlebot3_gazebo,
        'urdf',
        'turtlebot3_burger.urdf'
    )
    
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()
    
    # Start robot_state_publisher IMMEDIATELY with Gazebo
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_desc
        }]
    )
    
    #############################################################################
    # RViz
    #############################################################################
    
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    #############################################################################
    # Spawn Robot
    #############################################################################
    
    spawn_turtlebot_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot3_burger',
            '-file', os.path.join(pkg_turtlebot3_gazebo, 'models', 'turtlebot3_burger', 'model.sdf'),
            '-x', '-0.25',
            '-y', '-0.25',
            '-z', '0.01',
            '-timeout', '60.0'
        ],
        output='screen'
    )
    
    # Delay spawn to let Gazebo fully initialize
    spawn_turtlebot_cmd = TimerAction(
        period=10.0,  # Longer delay for stability
        actions=[
            LogInfo(msg='Spawning TurtleBot3...'),
            spawn_turtlebot_node
        ]
    )
    
    #############################################################################
    # SLAM Toolbox - Wait for robot to be fully spawned and publishing TF
    #############################################################################
    
    start_slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            slam_params_file,
            {'use_sim_time': use_sim_time}
        ]
    )
    
    # Wait for spawn to complete, then add extra delay for TF to stabilize
    start_slam_after_spawn = RegisterEventHandler(
        OnProcessStart(
            target_action=spawn_turtlebot_node,
            on_start=[
                TimerAction(
                    period=8.0,  # Extra time for TF tree to stabilize
                    actions=[
                        LogInfo(msg='Starting SLAM Toolbox (TF should be ready)...'),
                        start_slam_toolbox_node
                    ]
                )
            ]
        )
    )
    
    #############################################################################
    # Nav2 - Wait for SLAM to initialize
    #############################################################################
    
    start_nav2_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': nav2_params_file
        }.items()
    )
    
    # Wait for SLAM to start, then delay Nav2
    start_nav2_after_slam = RegisterEventHandler(
        OnProcessStart(
            target_action=start_slam_toolbox_node,
            on_start=[
                TimerAction(
                    period=8.0,  # Give SLAM time to initialize
                    actions=[
                        LogInfo(msg='Starting Nav2 navigation stack...'),
                        start_nav2_node
                    ]
                )
            ]
        )
    )
    
    #############################################################################
    # Maze Solver Node - Start last
    #############################################################################
    
    start_maze_solver_cmd = TimerAction(
        period=32.0,  # 10 (gazebo) + 8 (spawn settle) + 8 (SLAM) + 6 (Nav2)
        actions=[
            LogInfo(msg='Starting Maze Solver...'),
            Node(
                package='maze_solver_pkg',
                executable='maze_solver',
                name='maze_solver',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}]
            )
        ]
    )
    
    #############################################################################
    # Build Launch Description
    #############################################################################
    
    ld = LaunchDescription()
    
    # Arguments
    ld.add_action(declare_use_sim_time_cmd)
    
    # CRITICAL: Start robot_state_publisher WITH Gazebo (not after)
    ld.add_action(LogInfo(msg='Starting Gazebo...'))
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)
    ld.add_action(robot_state_publisher_cmd)  # Start early!
    ld.add_action(start_rviz_cmd)
    
    # Spawn robot after Gazebo is ready
    ld.add_action(spawn_turtlebot_cmd)
    
    # Chain the rest with event handlers
    ld.add_action(start_slam_after_spawn)
    ld.add_action(start_nav2_after_slam)
    ld.add_action(start_maze_solver_cmd)
    
    return ld
