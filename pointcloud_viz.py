"""
Point Cloud Visualization
Plane vs pothole points - Plotly figure (Streamlit), PLY export, Open3D window (local)
"""

import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from pointcloud_analysis import colorize_cloud


def _to_display(points: np.ndarray) -> np.ndarray:
    """Camera optical frame (x right, y down, z forward) -> plot frame (x right, y forward, z up)"""
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)


def _subsample(arr_len: int, max_points: int, seed: int = 0) -> np.ndarray:
    if arr_len <= max_points:
        return np.arange(arr_len)
    return np.sort(np.random.default_rng(seed).choice(arr_len, max_points, replace=False))


def plane_patch(result: dict) -> np.ndarray | None:
    """4 corner points (camera frame) of a rectangle on the fitted plane covering the road inliers"""
    plane, cloud = result.get("plane"), result["cloud"]
    if not plane or "normal" not in plane:
        return None
    inl = cloud["points"][cloud["labels"] == 1]
    if len(inl) < 10:
        return None
    n = plane["normal"]
    p0 = -plane["offset"] * n                          # foot of the perpendicular from the camera
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    coords = np.stack([(inl - p0) @ u, (inl - p0) @ v], axis=1)
    (u0, v0), (u1, v1) = np.percentile(coords, 2, axis=0), np.percentile(coords, 98, axis=0)
    return np.array([p0 + a * u + b * v for a, b in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]])


def build_plotly_figure(result: dict, max_points: int = 20000) -> go.Figure:
    """3D scatter: road-plane inliers (green), other points (grey, hidden by default),
    pothole points colored by depth below the plane, and the fitted plane as a surface."""
    cloud = result["cloud"]
    units = "m" if result["metric"] else "relative units"
    fig = go.Figure()

    if len(cloud["points"]) == 0:
        fig.update_layout(title=f"No point cloud ({result['status']})")
        return fig

    pts, labels, pdepth = cloud["points"], cloud["labels"], cloud["pothole_depth"]

    for lab, name, color, size, op, vis in [
        (1, "Road plane inliers", "rgb(60,200,80)", 1.5, 0.35, True),
        (0, "Other road-candidate points", "rgb(150,150,150)", 1.5, 0.3, "legendonly"),
    ]:
        sel = np.where(labels == lab)[0]
        sel = sel[_subsample(len(sel), max_points)]
        if len(sel):
            d = _to_display(pts[sel])
            fig.add_trace(go.Scatter3d(x=d[:, 0], y=d[:, 1], z=d[:, 2], mode="markers", name=name,
                                       marker=dict(size=size, color=color, opacity=op), visible=vis))

    sel = np.where(labels == 2)[0]
    sel = sel[_subsample(len(sel), max_points)]
    if len(sel):
        d = _to_display(pts[sel])
        fig.add_trace(go.Scatter3d(
            x=d[:, 0], y=d[:, 1], z=d[:, 2], mode="markers", name="Pothole points",
            marker=dict(size=2.5, color=pdepth[sel], colorscale="YlOrRd", showscale=True,
                        colorbar=dict(title=f"Depth below plane ({units})"))))

    patch = plane_patch(result)
    if patch is not None:
        d = _to_display(patch.astype(np.float32))
        fig.add_trace(go.Mesh3d(x=d[:, 0], y=d[:, 1], z=d[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                                color="royalblue", opacity=0.2, name="Fitted road plane", showlegend=True))

    fig.update_layout(
        title=f"Road plane vs pothole points ({units})",
        scene=dict(xaxis_title="x (right)", yaxis_title="z (forward)", zaxis_title="up",
                   aspectmode="data", camera=dict(eye=dict(x=0.0, y=-1.8, z=0.9))),
        height=650, margin=dict(l=0, r=0, t=40, b=0), legend=dict(itemsizing="constant"),
    )
    return fig


def to_open3d(result: dict) -> o3d.geometry.PointCloud:
    """Colored Open3D point cloud of the analysis result (camera frame)"""
    cloud = result["cloud"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud["points"].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colorize_cloud(cloud).astype(np.float64) / 255.0)
    return pcd


def save_ply(result: dict, path: str) -> bool:
    """Write the colored cloud to a .ply file (viewable in MeshLab / CloudCompare / Open3D)"""
    return bool(o3d.io.write_point_cloud(path, to_open3d(result)))


def show_open3d_window(result: dict):
    """Open an interactive Open3D viewer (needs a display - local use only)"""
    geoms = [to_open3d(result)]
    patch = plane_patch(result)
    if patch is not None:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(patch.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        mesh.paint_uniform_color([0.25, 0.4, 0.9])
        geoms.append(mesh)
    o3d.visualization.draw_geometries(geoms, window_name="IntelliRoad - road plane vs potholes")
