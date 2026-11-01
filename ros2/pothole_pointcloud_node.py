#!/usr/bin/env python3
"""
ROS 2 Node - IntelliRoad Pothole Point Cloud
Subscribes to a camera image, runs YOLOv8 + MiDaS + Open3D road-plane analysis,
and publishes a colored sensor_msgs/PointCloud2:
  grey = other points, green = fitted road plane, yellow->red = pothole points (depth below plane)

NOT RUN: written against the rclpy API but never executed on a ROS 2 install.
MiDaS depth is RELATIVE - the cloud is only in meters if camera_height_m > 0
(approximate, one-number calibration). Otherwise it is in relative units.

Run (from the project root, ROS 2 sourced):
  ros2 launch ros2/launch/pothole_pointcloud.launch.py image_path:=/path/to/road.jpg
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField

from cv_detection import RobustPotholeDetector
from pointcloud_analysis import (
    CameraIntrinsics, PointCloudPotholeAnalyzer, inverse_depth_to_depth,
    annotate_detections, colorize_cloud, pack_xyzrgb,
)


def image_msg_to_bgr(msg) -> np.ndarray:
    """sensor_msgs/Image -> BGR uint8 array (no cv_bridge dependency)"""
    ch = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}.get(msg.encoding)
    if ch is None:
        raise ValueError(f"Unsupported image encoding '{msg.encoding}' (use bgr8, rgb8, bgra8, rgba8, mono8)")
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    arr = buf[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
    convert = {"bgr8": None, "rgb8": cv2.COLOR_RGB2BGR, "bgra8": cv2.COLOR_BGRA2BGR,
               "rgba8": cv2.COLOR_RGBA2BGR, "mono8": cv2.COLOR_GRAY2BGR}[msg.encoding]
    return np.ascontiguousarray(arr if convert is None else cv2.cvtColor(arr, convert))


def _unset(value: str) -> str:
    """Launch files pass 'none' for unset string params (empty overrides are awkward)"""
    return "" if value.strip().lower() in ("", "none") else value


class PotholePointCloudNode(Node):
    def __init__(self):
        super().__init__("pothole_pointcloud")

        # Numeric params use dynamic typing where available so '-p x:=1' and 'x:=1.0' both work
        try:
            numeric = ParameterDescriptor(dynamic_typing=True)
        except (AssertionError, TypeError):
            numeric = ParameterDescriptor()

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "")
        self.declare_parameter("cloud_topic", "/intelliroad/pothole_cloud")
        self.declare_parameter("frame_id", "")
        self.declare_parameter("image_path", "")
        self.declare_parameter("model_dir", "./cv_models")
        self.declare_parameter("hfov_deg", 60.0, numeric)
        self.declare_parameter("camera_height_m", 0.0, numeric)
        self.declare_parameter("sensitivity", 0.5, numeric)
        self.declare_parameter("min_period_s", 1.0, numeric)

        p = self.get_parameter
        self.image_path = _unset(p("image_path").value)
        self.frame_id = _unset(p("frame_id").value)
        camera_info_topic = _unset(p("camera_info_topic").value)
        self.sensitivity = float(p("sensitivity").value)
        self.min_period_s = float(p("min_period_s").value)
        camera_height_m = float(p("camera_height_m").value)
        hfov_deg = float(p("hfov_deg").value)

        self.intr = None                      # from CameraInfo if provided, else assumed from hfov_deg
        self._last_t = None
        self._warned_relative = False

        self.get_logger().info("Loading detector (YOLO + MiDaS)...")
        self.detector = RobustPotholeDetector(model_dir=_unset(p("model_dir").value) or "./cv_models")
        self.analyzer = PointCloudPotholeAnalyzer(hfov_deg=hfov_deg, camera_height_m=camera_height_m)
        self.pub = self.create_publisher(PointCloud2, p("cloud_topic").value, 1)

        if camera_info_topic:
            self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 1)

        if self.image_path:
            self._static_img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if self._static_img is None:
                raise RuntimeError(f"Could not read image_path '{self.image_path}'")
            self.create_timer(max(self.min_period_s, 0.1), self.timer_callback)
            self.get_logger().info(f"Publishing cloud for static image {self.image_path}")
        else:
            self.create_subscription(Image, p("image_topic").value, self.image_callback, qos_profile_sensor_data)
            self.get_logger().info(f"Subscribed to {p('image_topic').value}")

    # ---------------- callbacks ----------------
    def camera_info_callback(self, msg: CameraInfo):
        if self.intr is None:
            self.intr = CameraIntrinsics.from_camera_info_k(msg.k, msg.width, msg.height)
            self.get_logger().info("Using intrinsics from CameraInfo")

    def image_callback(self, msg: Image):
        now = self.get_clock().now()
        if self._last_t is not None and (now - self._last_t).nanoseconds * 1e-9 < self.min_period_s:
            return                              # YOLO + MiDaS + Open3D are heavy: throttle
        self._last_t = now
        try:
            bgr = image_msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            return
        self.process(bgr, msg.header.stamp, msg.header.frame_id)

    def timer_callback(self):
        self.process(self._static_img, self.get_clock().now().to_msg(), "")

    # ---------------- processing ----------------
    def process(self, bgr: np.ndarray, stamp, header_frame: str):
        detections = self.detector.detect(bgr, sensitivity=self.sensitivity)
        inv_depth = self.detector.estimate_depth(bgr)
        if inv_depth is None:
            self.get_logger().warn("MiDaS unavailable - cannot build a point cloud", throttle_duration_sec=10.0)
            return

        result = self.analyzer.analyze(inverse_depth_to_depth(inv_depth), detections, intrinsics=self.intr)
        annotate_detections(detections, result)

        if not result["metric"] and not self._warned_relative:
            self.get_logger().warn("camera_height_m not set: cloud is in RELATIVE units (MiDaS depth is not metric)")
            self._warned_relative = True
        if result["status"] != "ok":
            self.get_logger().warn(f"Road plane not fitted: {result['status']}", throttle_duration_sec=5.0)

        for i, m in enumerate(result["potholes"]):
            if m["valid"]:
                self.get_logger().info(
                    f"pothole {i}: depth={m['depth']:.4f} area={m['area']:.4f} ({result['units']}, {m['n_pixels']} px)")

        frame = self.frame_id or header_frame or "camera_optical_frame"
        self.pub.publish(self.cloud_to_msg(result["cloud"], stamp, frame))

    @staticmethod
    def cloud_to_msg(cloud: dict, stamp, frame_id: str) -> PointCloud2:
        pts = cloud["points"].astype(np.float32)
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id            # optical frame: x right, y down, z forward
        msg.height, msg.width = 1, len(pts)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(pts)
        msg.is_dense = True
        msg.data = pack_xyzrgb(pts, colorize_cloud(cloud))
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PotholePointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
