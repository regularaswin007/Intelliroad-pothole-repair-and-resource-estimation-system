"""
Launch file for the IntelliRoad pothole point cloud node.

NOT RUN: never executed with a ROS 2 install.
The project is not an ament package, so this starts the node script directly
with ExecuteProcess (cwd = project root so ./cv_models resolves). Unset string
arguments use the sentinel 'none'.

The node needs open3d / torch / ultralytics. `ros2` itself runs under the system
Python, so the node is started with the ACTIVE VIRTUAL ENV's python when
VIRTUAL_ENV is set (override with python_executable:=/path/to/python).

  ros2 launch ros2/launch/pothole_pointcloud.launch.py image_path:=/path/to/road.jpg camera_height_m:=1.4
  ros2 launch ros2/launch/pothole_pointcloud.launch.py image_topic:=/camera/image_raw rviz:=true
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_python() -> str:
    """Prefer the active venv's python (it has the project's pip packages)"""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        for rel in ("bin/python", "Scripts/python.exe"):
            cand = os.path.join(venv, rel)
            if os.path.exists(cand):
                return cand
    return sys.executable


# Passed to the node as ROS parameters
NODE_ARGS = [
    ("image_topic", "/camera/image_raw", "Image topic to subscribe to (ignored if image_path is set)"),
    ("camera_info_topic", "none", "CameraInfo topic for real intrinsics ('none' = assume hfov_deg)"),
    ("cloud_topic", "/intelliroad/pothole_cloud", "PointCloud2 output topic"),
    ("frame_id", "none", "Cloud frame_id ('none' = image header frame, else 'camera_optical_frame')"),
    ("image_path", "none", "Publish a cloud for one still image instead of subscribing ('none' = off)"),
    ("model_dir", "./cv_models", "Folder with YOLO weights"),
    ("hfov_deg", "60.0", "Assumed horizontal FOV when no CameraInfo is given"),
    ("camera_height_m", "0.0", "Camera height above road in m (0 = unknown -> relative units)"),
    ("sensitivity", "0.5", "Detection sensitivity 0..1"),
    ("min_period_s", "1.0", "Minimum seconds between processed frames"),
]

# Launch-only arguments
EXTRA_ARGS = [
    ("python_executable", _default_python(), "Python used to run the node (default: active venv, else system)"),
    ("rviz", "false", "Also start RViz2 (you still add the PointCloud2 display once)"),
    ("rviz_fixed_frame", "camera_optical_frame", "RViz Fixed Frame - must match the cloud's frame_id"),
]


def generate_launch_description():
    ros2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(ros2_dir)
    node_script = os.path.join(ros2_dir, "pothole_pointcloud_node.py")

    cmd = [LaunchConfiguration("python_executable"), node_script, "--ros-args"]
    for name, _, _ in NODE_ARGS:
        cmd += ["-p", [name + ":=", LaunchConfiguration(name)]]

    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in NODE_ARGS + EXTRA_ARGS]
        + [
            ExecuteProcess(cmd=cmd, cwd=project_root, output="screen", name="pothole_pointcloud"),
            Node(package="rviz2", executable="rviz2", name="rviz2", output="screen",
                 arguments=["-f", LaunchConfiguration("rviz_fixed_frame")],
                 condition=IfCondition(LaunchConfiguration("rviz"))),
        ]
    )
