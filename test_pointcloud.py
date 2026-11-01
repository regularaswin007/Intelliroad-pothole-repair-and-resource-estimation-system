"""
Synthetic test for pointcloud_analysis.py
Renders a road plane with bowl-shaped potholes of KNOWN depth/area, runs the
analyzer, and prints measured vs ground-truth values.

No YOLO / MiDaS / ROS needed - run standalone:  python test_pointcloud.py
NOTE: this checks the geometry only. It says nothing about real MiDaS output
or real photos.
"""

import numpy as np

from pointcloud_analysis import (
    CameraIntrinsics, PointCloudPotholeAnalyzer, inverse_depth_to_depth,
    annotate_detections, colorize_cloud, pack_xyzrgb,
)

W, H, HFOV = 960, 720, 60.0
CAM_HEIGHT = 1.4      # m
PITCH = np.radians(15.0)
INTR = CameraIntrinsics.from_fov(W, H, HFOV)

# Camera frame: x right, y down, z forward. "down" vector for a camera pitched down.
G = np.array([0.0, np.cos(PITCH), np.sin(PITCH)])
A = np.array([1.0, 0.0, 0.0])                                  # ground axis (right)
B = np.array([0.0, -np.sin(PITCH), np.cos(PITCH)])             # ground axis (forward)
P0 = CAM_HEIGHT * G                                            # foot point on plane

# (X0 right, Y0 forward, radius, depth) in meters
POTHOLES = [(0.3, 4.5, 0.30, 0.08), (-0.6, 3.0, 0.20, 0.04)]


def bowl(X, Y):
    """Total depression below the plane at ground coordinates (X, Y)"""
    d = np.zeros_like(X)
    for X0, Y0, R, D in POTHOLES:
        r2 = ((X - X0) ** 2 + (Y - Y0) ** 2) / R ** 2
        d = np.maximum(d, np.where(r2 < 1.0, D * (1.0 - r2), 0.0))
    return d


def render_depth(noise_frac=0.002, seed=1):
    """Ray-march the surface (plane + bowls). Returns z-depth map (NaN above horizon)."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:H, 0:W]
    ray = np.stack([(u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy, np.ones_like(u, dtype=float)], axis=-1)
    gr = ray @ G
    below = gr > 1e-3
    t0 = np.where(below, CAM_HEIGHT / np.where(below, gr, 1.0), np.nan)   # plane hit (z-depth)

    t = t0.copy()
    P = t0[..., None] * ray
    X, Y = (P - P0) @ A, (P - P0) @ B
    near = below & (bowl(X, Y) > 0)
    # refine only rays whose plane hit lies inside a bowl: march along the ray
    idx = np.argwhere(near)
    steps = np.linspace(0.0, 1.0, 400)
    for i, j in idx:
        g = gr[i, j]
        ts = t0[i, j] + steps * (0.3 / g)          # window behind the plane hit
        Pm = ts[:, None] * ray[i, j][None, :]
        Xm, Ym = (Pm - P0) @ A, (Pm - P0) @ B
        f = Pm @ G - CAM_HEIGHT - bowl(Xm, Ym)     # <0 above surface, >0 below
        k = np.argmax(f >= 0)
        if f[k] >= 0 and k > 0:
            w = -f[k - 1] / (f[k] - f[k - 1])
            t[i, j] = ts[k - 1] + w * (ts[k] - ts[k - 1])
        else:
            t[i, j] = t0[i, j]
    depth = t + rng.normal(0.0, noise_frac, t.shape) * t
    return depth, near


def boxes_from_mask(near, margin=12):
    """Per-pothole detection boxes (like YOLO would give), slightly larger than the bowl"""
    dets = []
    for X0, Y0, R, D in POTHOLES:
        c = CAM_HEIGHT * G + X0 * A + Y0 * B
        # project bowl rim points to find the pixel box
        ang = np.linspace(0, 2 * np.pi, 72)
        rim = c[None, :] + R * np.cos(ang)[:, None] * A + R * np.sin(ang)[:, None] * B
        px = rim[:, 0] / rim[:, 2] * INTR.fx + INTR.cx
        py = rim[:, 1] / rim[:, 2] * INTR.fy + INTR.cy
        x0, x1 = int(px.min()) - margin, int(px.max()) + margin
        y0, y1 = int(py.min()) - margin, int(py.max()) + margin
        dets.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "confidence": 0.9, "depth": 6.0, "area": 0.4,
                     "severity": 2, "method": "synthetic"})
    return dets


def report(title, result, thr_note=""):
    print(f"\n--- {title} ---")
    print(f"status={result['status']}  metric={result['metric']}  units={result['units']}  scale={result['scale']:.3f}")
    if result["plane"] and "normal" in result["plane"]:
        p = result["plane"]
        ang = np.degrees(np.arccos(np.clip(np.abs(p["normal"] @ -G), -1, 1)))
        print(f"plane: inlier_ratio={p['inlier_ratio']:.2f}  camera_height_est={p['camera_height']:.3f}"
              f"  normal_error={ang:.2f} deg  dist_thr={p['distance_threshold']:.4f}")
    for (X0, Y0, R, D), m in zip(POTHOLES, result["potholes"]):
        thr = result["plane"]["distance_threshold"] if result["plane"] and "distance_threshold" in result["plane"] else 0
        exp_area = np.pi * R ** 2 * max(0.0, 1.0 - thr / D)
        exp_p95 = thr + 0.95 * (D - thr)
        if m["valid"]:
            print(f"pothole R={R:.2f} D={D*100:.0f}cm: depth_p95={m['depth']*100:.1f}cm (expected ~{exp_p95*100:.1f}), "
                  f"depth_max={m['depth_max']*100:.1f}cm (true {D*100:.0f}), area={m['area']:.4f} "
                  f"(expected ~{exp_area:.4f}, full bowl {np.pi*R**2:.4f}) px={m['n_pixels']}")
        else:
            print(f"pothole R={R:.2f} D={D*100:.0f}cm: NOT MEASURED ({m['reason']})")


if __name__ == "__main__":
    depth_m, near = render_depth()
    dets = boxes_from_mask(near)
    print(f"Rendered {W}x{H} scene, {int(near.sum())} pixels inside bowls; boxes: {[d['bbox'] for d in dets]}")

    # A) Metric depth (e.g. RGB-D): no calibration needed
    an = PointCloudPotholeAnalyzer(hfov_deg=HFOV)
    res_a = an.analyze(depth_m, dets, intrinsics=INTR, depth_is_metric=True)
    report("A) metric depth input", res_a)

    # B) Same scene but depth is in an UNKNOWN scale (x3.7); recover scale from camera height
    an_b = PointCloudPotholeAnalyzer(hfov_deg=HFOV, camera_height_m=CAM_HEIGHT)
    res_b = an_b.analyze(depth_m * 3.7, dets, intrinsics=INTR)
    report("B) unknown-scale depth + camera_height_m calibration", res_b)

    # C) MiDaS-like relative inverse depth (idealised affine inverse depth, min-max normalised)
    inv = np.where(np.isfinite(depth_m), 1.0 / depth_m, 0.0)
    inv = (inv - inv.min()) / (inv.max() - inv.min())
    rel = inverse_depth_to_depth(inv)
    res_c = PointCloudPotholeAnalyzer(hfov_deg=HFOV, camera_height_m=CAM_HEIGHT).analyze(rel, dets, intrinsics=INTR)
    report("C) relative inverse depth -> heuristic depth -> camera-height calibration", res_c)
    res_c2 = PointCloudPotholeAnalyzer(hfov_deg=HFOV).analyze(rel, dets, intrinsics=INTR)
    report("C2) relative inverse depth, NO calibration (relative units only)", res_c2)

    # D) No detections: plane should still fit
    res_d = PointCloudPotholeAnalyzer(hfov_deg=HFOV).analyze(depth_m, [], intrinsics=INTR, depth_is_metric=True)
    print(f"\n--- D) no detections --- status={res_d['status']} potholes={len(res_d['potholes'])} "
          f"cloud_points={len(res_d['cloud']['points'])}")

    # E) Garbage depth map: expect a failure status, not a confident measurement
    rng = np.random.default_rng(3)
    res_e = PointCloudPotholeAnalyzer(hfov_deg=HFOV).analyze(rng.uniform(1, 10, (H, W)).astype(np.float32), dets,
                                                             intrinsics=INTR, depth_is_metric=True)
    print(f"\n--- E) random depth --- status={res_e['status']} plane={res_e['plane'] and {k: (round(float(v),3) if np.isscalar(v) else '...') for k, v in res_e['plane'].items()}}"
          f" measured={[m['valid'] for m in res_e['potholes']]}")

    # F) annotate_detections feeds the existing pipeline
    det_a = [dict(d) for d in dets]
    annotate_detections(det_a, res_a)
    print("\n--- F) annotate_detections (metric result) ---")
    for d in det_a:
        print({k: d[k] for k in ("depth", "area", "severity", "depth_source", "heuristic_depth", "heuristic_severity")})
    det_c2 = [dict(d) for d in dets]
    annotate_detections(det_c2, res_c2)
    print("annotate with relative-only result leaves depth/area untouched:",
          all(d["depth"] == 6.0 and d["area"] == 0.4 and d["depth_source"].startswith("heuristic") for d in det_c2))

    # G) PointCloud2 byte layout round-trip
    cloud = res_a["cloud"]
    col = colorize_cloud(cloud)
    raw = pack_xyzrgb(cloud["points"], col)
    n = len(cloud["points"])
    back = np.frombuffer(raw, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]))
    ok_xyz = np.allclose(back["x"], cloud["points"][:, 0]) and np.allclose(back["z"], cloud["points"][:, 2])
    r, g, b = (back["rgb"] >> 16) & 255, (back["rgb"] >> 8) & 255, back["rgb"] & 255
    ok_rgb = np.array_equal(np.stack([r, g, b], 1).astype(np.uint8), col)
    print(f"\n--- G) pack_xyzrgb --- points={n} bytes={len(raw)} (16/pt -> {16*n}) xyz_ok={ok_xyz} rgb_ok={ok_rgb}")
