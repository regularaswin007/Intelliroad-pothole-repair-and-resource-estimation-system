"""
Point Cloud Pothole Analysis (Open3D)
Measures pothole depth and area relative to a RANSAC-fitted road plane

Pipeline:
  1. Depth map -> 3D point cloud (pinhole back-projection with camera intrinsics)
  2. Road candidates = points OUTSIDE the YOLO boxes (lower part of the image only)
  3. Voxel downsample + statistical outlier removal
  4. Road plane via Open3D segment_plane (RANSAC)
  5. Per pothole: depth / area of the points lying BELOW the plane

IMPORTANT - MiDaS depth is RELATIVE (affine-invariant inverse depth):
  It has no metric scale or shift, so a point cloud built from it is only
  geometrically meaningful up to an unknown scale (and some distortion).
  - Without calibration, every length is reported in "relative units"
    (metric=False) and must NOT be read as cm / m.
  - If camera_height_m is given, the cloud is rescaled so the fitted road plane
    sits camera_height_m below the camera (metric=True). This is a one-number
    calibration, not true metric depth.
  - If you have real metric depth (RGB-D, stereo, LiDAR, metric depth model),
    pass it with depth_is_metric=True and no calibration is needed.
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass


# ========================================
# CAMERA MODEL
# ========================================
@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics (pixels)"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 60.0) -> "CameraIntrinsics":
        """Assumed intrinsics from a horizontal field of view (use real calibration if you have it)"""
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(fx, fx, width / 2.0, height / 2.0, width, height)

    @classmethod
    def from_camera_info_k(cls, k, width: int, height: int) -> "CameraIntrinsics":
        """Build from the row-major 3x3 K matrix of a ROS sensor_msgs/CameraInfo"""
        return cls(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]),
                   width=int(width), height=int(height))

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Rescale intrinsics when the image was resized relative to the calibration"""
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                                int(width), int(height))


# ========================================
# DEPTH HELPERS
# ========================================
def inverse_depth_to_depth(inv_depth: np.ndarray, inv_floor: float = 0.1) -> np.ndarray:
    """
    Heuristic conversion of normalized MiDaS inverse depth (0..1, larger = nearer)
    to a depth-like map in RELATIVE units (range 1 .. 1/inv_floor).

    The floor stands in for the unknown shift of MiDaS output. This is NOT a
    recovery of true depth - it only gives a consistent, monotonic mapping.
    """
    inv = np.clip(np.asarray(inv_depth, dtype=np.float32), 0.0, 1.0)
    return 1.0 / (inv_floor + (1.0 - inv_floor) * inv)


def backproject(depth: np.ndarray, intr: CameraIntrinsics, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Back-project pixels to camera-frame points (x right, y down, z forward = ROS optical frame)"""
    z = depth[rows, cols].astype(np.float32)
    x = (cols - intr.cx) * z / intr.fx
    y = (rows - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def severity_from_measurements(depth_cm: float, area_m2: float) -> int:
    """Severity 1-4 - mirrors the thresholds in cv_detection.RobustPotholeDetector.detect"""
    if depth_cm > 12 or area_m2 > 2.0:
        return 4
    elif depth_cm > 8 or area_m2 > 1.2:
        return 3
    elif depth_cm > 5 or area_m2 > 0.7:
        return 2
    return 1


# ========================================
# ANALYZER
# ========================================
class PointCloudPotholeAnalyzer:
    """
    Fits the road plane from points outside the detection boxes and measures
    each detected pothole relative to that plane.
    """

    def __init__(
        self,
        hfov_deg: float = 60.0,
        camera_height_m: float | None = None,
        voxel_frac: float = 0.01,
        outlier_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
        ransac_thresh_frac: float = 0.005,
        ransac_iters: int = 1000,
        box_margin_frac: float = 0.25,
        road_roi_top_frac: float = 0.35,
        min_inlier_ratio: float = 0.2,
        min_pothole_pixels: int = 40,
        depth_percentile: float = 95.0,
        max_cloud_points: int = 300000,
        max_pothole_cloud_points: int = 50000,
        seed: int = 0,
    ):
        """
        voxel_frac / ransac_thresh_frac are fractions of the median road depth, so
        they work in any unit (relative or metric). Smaller ransac_thresh_frac
        detects shallower potholes but needs cleaner depth.
        road_roi_top_frac: only pixels below this fraction of image height are
        used for the road plane (keeps sky / buildings out of the RANSAC fit).
        """
        self.hfov_deg = hfov_deg
        self.camera_height_m = camera_height_m if camera_height_m and camera_height_m > 0 else None
        self.voxel_frac = voxel_frac
        self.outlier_neighbors = outlier_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.ransac_thresh_frac = ransac_thresh_frac
        self.ransac_iters = ransac_iters
        self.box_margin_frac = box_margin_frac
        self.road_roi_top_frac = road_roi_top_frac
        self.min_inlier_ratio = min_inlier_ratio
        self.min_pothole_pixels = min_pothole_pixels
        self.depth_percentile = depth_percentile
        self.max_cloud_points = max_cloud_points
        self.max_pothole_cloud_points = max_pothole_cloud_points
        self.seed = seed

    # =================================
    # Public interface
    # =================================
    def analyze(
        self,
        depth: np.ndarray,
        detections: list[dict],
        intrinsics: CameraIntrinsics | None = None,
        depth_is_metric: bool = False,
    ) -> dict:
        """
        depth: (H, W) depth-like map. Use inverse_depth_to_depth() on MiDaS output,
               or real metric depth with depth_is_metric=True.
        detections: dicts with "bbox" = [x, y, w, h] in pixels (cv_detection format).

        Returns a dict:
          status    'ok' or a failure reason (potholes are then all unmeasured)
          metric    True if lengths are in meters (metric depth or camera-height calibrated)
          units     'm' or 'relative'
          plane     normal / offset / inlier_ratio / camera_height / distance_threshold
          potholes  one dict per detection, same order as `detections`
          cloud     points (M,3), labels (M,) [0 other, 1 road plane, 2 pothole], pothole_depth (M,)
        """
        depth = np.asarray(depth, dtype=np.float32)
        H, W = depth.shape
        intr = intrinsics if intrinsics is not None else CameraIntrinsics.from_fov(W, H, self.hfov_deg)
        if (intr.width, intr.height) != (W, H):
            intr = intr.scaled_to(W, H)

        valid_depth = np.isfinite(depth) & (depth > 0)

        # ---- 1. Strided grid -> road candidate cloud (outside boxes) ----
        stride = max(1, int(np.ceil(np.sqrt(H * W / self.max_cloud_points))))
        rr, cc = np.mgrid[0:H:stride, 0:W:stride]
        rr, cc = rr.ravel(), cc.ravel()

        excluded = np.zeros((H, W), dtype=bool)
        for det in detections:
            x, y, w, h = [int(v) for v in det["bbox"]]
            mx, my = int(w * self.box_margin_frac), int(h * self.box_margin_frac)
            excluded[max(0, y - my):min(H, y + h + my), max(0, x - mx):min(W, x + w + mx)] = True

        roi_top = int(self.road_roi_top_frac * H)
        keep = (~excluded[rr, cc]) & valid_depth[rr, cc] & (rr >= roi_top)
        if keep.sum() < 500:
            return self._failed("too_few_road_points", detections)

        road_pts = backproject(depth, intr, rr[keep], cc[keep])
        scene_scale = float(np.median(road_pts[:, 2]))

        # ---- 2. Voxel downsample + statistical outlier removal ----
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(road_pts.astype(np.float64))
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_frac * scene_scale)
        if len(pcd.points) < 100:
            return self._failed("too_few_points_after_voxel", detections)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.outlier_neighbors, std_ratio=self.outlier_std_ratio
        )
        if len(pcd.points) < 100:
            return self._failed("too_few_points_after_outlier_removal", detections)

        # ---- 3. RANSAC road plane ----
        if hasattr(o3d.utility, "random"):
            o3d.utility.random.seed(self.seed)
        dist_thr = self.ransac_thresh_frac * scene_scale
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thr, ransac_n=3, num_iterations=self.ransac_iters
        )
        plane = np.asarray(plane_model, dtype=np.float64).reshape(4)
        plane = plane / np.linalg.norm(plane[:3])
        if plane[3] < 0:        # orient so the camera (origin) is on the positive side
            plane = -plane
        normal, offset = plane[:3], float(plane[3])
        inlier_ratio = len(inliers) / len(pcd.points)

        if inlier_ratio < self.min_inlier_ratio:
            return self._failed("weak_plane_fit", detections, inlier_ratio=inlier_ratio)
        if offset < 1e-6 * scene_scale:
            return self._failed("degenerate_plane", detections, inlier_ratio=inlier_ratio)

        # ---- 4. Scale: metric depth, camera-height calibration, or relative ----
        notes = []
        if depth_is_metric:
            scale, metric = 1.0, True
            if self.camera_height_m:
                notes.append("camera_height_m ignored (depth_is_metric=True)")
        elif self.camera_height_m:
            scale, metric = self.camera_height_m / offset, True
        else:
            scale, metric = 1.0, False

        # ---- 5. Per-pothole measurement (full-resolution pixels inside each box) ----
        potholes, pothole_pts, pothole_dep = [], [], []
        for det in detections:
            m, pts, dep = self._measure(depth, valid_depth, det, intr, normal, offset, dist_thr)
            potholes.append(m)
            if pts is not None:
                pothole_pts.append(pts)
                pothole_dep.append(dep)

        # ---- 6. Apply scale to outputs ----
        for m in potholes:
            if m["valid"]:
                m["depth"] *= scale
                m["depth_max"] *= scale
                m["area"] *= scale ** 2
                m["depth_cm"] = m["depth"] * 100.0 if metric else None
                m["area_m2"] = m["area"] if metric else None

        cloud = self._build_cloud(pcd, inliers, pothole_pts, pothole_dep, scale)

        return {
            "status": "ok",
            "metric": metric,
            "units": "m" if metric else "relative",
            "scale": scale,
            "notes": notes,
            "plane": {
                "normal": normal,
                "offset": offset * scale,
                "inlier_ratio": inlier_ratio,
                "camera_height": offset * scale,   # = measured camera height above the plane
                "distance_threshold": dist_thr * scale,
            },
            "potholes": potholes,
            "cloud": cloud,
        }

    # =================================
    # Internals
    # =================================
    def _measure(self, depth, valid_depth, det, intr, normal, offset, thr):
        """Depth / area of one detection relative to the plane (input units, unscaled)"""
        H, W = depth.shape
        x, y, w, h = [int(v) for v in det["bbox"]]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        out = {"valid": False, "reason": "empty_box", "depth": None, "depth_max": None, "area": None,
               "depth_cm": None, "area_m2": None, "n_pixels": 0, "fill_ratio": 0.0}
        if x1 <= x0 or y1 <= y0:
            return out, None, None

        rows, cols = np.mgrid[y0:y1, x0:x1]
        rows, cols = rows.ravel(), cols.ravel()
        ok = valid_depth[rows, cols]
        rows, cols = rows[ok], cols[ok]
        if len(rows) == 0:
            out["reason"] = "no_valid_depth"
            return out, None, None

        pts = backproject(depth, intr, rows, cols)
        depression = -(pts @ normal + offset)     # >0 = below the road plane (farther than it)
        mask = depression > thr
        n_px = int(mask.sum())
        out["n_pixels"] = n_px
        out["fill_ratio"] = n_px / len(rows)
        if n_px < self.min_pothole_pixels:
            out["reason"] = "no_depression_below_plane"
            return out, None, None

        pm, dm = pts[mask], depression[mask]

        # Area: sum of per-pixel surface footprints. A pixel at depth z with ray angle
        # alpha (to the optical axis) hitting a surface whose normal makes angle theta
        # with the ray covers  (z/fx)(z/fy) * cos(alpha) / cos(theta).
        ray = pm / np.linalg.norm(pm, axis=1, keepdims=True)
        cos_alpha = ray[:, 2]
        cos_theta = np.clip(np.abs(ray @ normal), 0.1, 1.0)
        footprint = (pm[:, 2] / intr.fx) * (pm[:, 2] / intr.fy) * cos_alpha / cos_theta

        out.update({
            "valid": True,
            "reason": "ok",
            "depth": float(np.percentile(dm, self.depth_percentile)),
            "depth_max": float(dm.max()),
            "area": float(footprint.sum()),
        })
        return out, pm, dm

    def _build_cloud(self, pcd, inliers, pothole_pts, pothole_dep, scale):
        """Downsampled labelled cloud for visualization / ROS (scaled units)"""
        road = np.asarray(pcd.points, dtype=np.float32)
        labels = np.zeros(len(road), dtype=np.uint8)
        labels[np.asarray(inliers, dtype=np.int64)] = 1
        pothole_depth = np.zeros(len(road), dtype=np.float32)

        points, all_labels, all_depth = [road], [labels], [pothole_depth]
        if pothole_pts:
            pp = np.concatenate(pothole_pts, axis=0)
            pd = np.concatenate(pothole_dep, axis=0).astype(np.float32)
            if len(pp) > self.max_pothole_cloud_points:
                idx = np.random.default_rng(self.seed).choice(len(pp), self.max_pothole_cloud_points, replace=False)
                pp, pd = pp[idx], pd[idx]
            points.append(pp)
            all_labels.append(np.full(len(pp), 2, dtype=np.uint8))
            all_depth.append(pd)

        return {
            "points": np.concatenate(points, axis=0) * np.float32(scale),
            "labels": np.concatenate(all_labels, axis=0),
            "pothole_depth": np.concatenate(all_depth, axis=0) * np.float32(scale),
        }

    def _failed(self, reason, detections, inlier_ratio=None):
        potholes = [{"valid": False, "reason": reason, "depth": None, "depth_max": None, "area": None,
                     "depth_cm": None, "area_m2": None, "n_pixels": 0, "fill_ratio": 0.0}
                    for _ in detections]
        return {
            "status": reason,
            "metric": False,
            "units": "relative",
            "scale": 1.0,
            "notes": [],
            "plane": None if inlier_ratio is None else {"inlier_ratio": inlier_ratio},
            "potholes": potholes,
            "cloud": {"points": np.zeros((0, 3), np.float32),
                      "labels": np.zeros(0, np.uint8),
                      "pothole_depth": np.zeros(0, np.float32)},
        }


# ========================================
# FEED RESULTS INTO THE EXISTING PIPELINE
# ========================================
def annotate_detections(detections: list[dict], result: dict) -> list[dict]:
    """
    Write plane-based measurements into the detection dicts (in place).

    Only when the measurement is metric AND valid are 'depth' (cm), 'area' (m2) and
    'severity' overwritten, so the existing Pothole / PPO / SAC code receives real
    units. Otherwise the original heuristic values are left untouched and the
    relative measurement is stored under plane_depth / plane_area.
    """
    for det, m in zip(detections, result["potholes"]):
        det.setdefault("heuristic_depth", det.get("depth"))
        det.setdefault("heuristic_area", det.get("area"))
        det.setdefault("heuristic_severity", det.get("severity"))

        det["plane_valid"] = m["valid"]
        det["plane_reason"] = m["reason"]
        det["plane_units"] = result["units"]
        det["plane_depth"] = m["depth"]
        det["plane_area"] = m["area"]
        det["depth_source"] = "heuristic (MiDaS-relative)"

        if m["valid"] and result["metric"]:
            det["depth"] = m["depth_cm"]
            det["area"] = m["area_m2"]
            det["severity"] = severity_from_measurements(m["depth_cm"], m["area_m2"])
            det["depth_source"] = "plane"
    return detections


# ========================================
# POINTCLOUD2 HELPERS (pure numpy, no ROS needed)
# ========================================
LABEL_COLORS = {0: (150, 150, 150), 1: (60, 200, 80)}   # other = grey, road plane = green


def colorize_cloud(cloud: dict) -> np.ndarray:
    """(M,3) uint8 colors: grey other, green road plane, yellow->red pothole by depth below plane"""
    labels = cloud["labels"]
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for lab, rgb in LABEL_COLORS.items():
        colors[labels == lab] = rgb
    pot = labels == 2
    if pot.any():
        d = cloud["pothole_depth"][pot]
        t = np.clip(d / max(float(d.max()), 1e-9), 0.0, 1.0)
        colors[pot, 0] = 255
        colors[pot, 1] = (255 * (1.0 - t)).astype(np.uint8)
        colors[pot, 2] = 0
    return colors


def pack_xyzrgb(points: np.ndarray, colors: np.ndarray) -> bytes:
    """
    Pack to the sensor_msgs/PointCloud2 layout used with RViz:
    x,y,z float32 at offsets 0/4/8, rgb at offset 12 (uint32 0x00RRGGBB stored in a
    FLOAT32 field, PCL convention). point_step = 16 bytes.
    """
    n = len(points)
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    rec["x"], rec["y"], rec["z"] = points[:, 0], points[:, 1], points[:, 2]
    c = colors.astype(np.uint32)
    rec["rgb"] = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    return rec.tobytes()
