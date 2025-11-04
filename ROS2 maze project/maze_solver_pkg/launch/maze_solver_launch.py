import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    #world = LaunchConfiguration('world', default='tb3_maze.world')  # Your custom maze world
    
    # Path to your maze world file
    world_file = os.path.join(
        os.path.expanduser('~'),
        'maze_ws',
        'src',
        'maze_solver_pkg',
        'worlds',
        'tb3_maze.world'
    )
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_dir, 'launch', 'turtlebot3_world.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )
    
    # SLAM Toolbox launch (Online Async mode for real-time mapping)
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items()
    )
    
    # Nav2 launch
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': os.path.join(
                os.path.expanduser('~'),
                'maze_ws',
                'src',
                'maze_solver_pkg',
                'config',
                'nav2_params.yaml'
            )
        }.items()
    )
    
    # Maze solver node
    maze_solver_node = Node(
        package='maze_solver_pkg',
        executable='maze_solver',
        name='maze_solver',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'frontier_threshold': 0.5,
            'min_frontier_size': 10,
            'exploration_radius': 5.0,
            'goal_tolerance': 0.3,
            'wall_following_distance': 0.3,
            'linear_speed': 0.15,
            'angular_speed': 0.5,
        }]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'world',
            default_value=world_file,
            description='Full path to world file'
        ),
        gazebo,
        slam_toolbox,
        nav2,
        maze_solver_node,
    ])
