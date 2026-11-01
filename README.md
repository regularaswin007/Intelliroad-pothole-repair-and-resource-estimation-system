# IntelliRoad

Road-pothole detection, repair prioritization and material estimation, with a point-cloud measurement
module and a ROS 2 node. Detection uses YOLOv8 plus classical CV, depth comes from MiDaS, repair
prioritization uses PPO, material estimation uses SAC, repair routes use A*, and everything is shown in a
Streamlit app.

## Scope: what is real and what is synthetic

- **Detection** runs on real images. The detector loads YOLO weights from `cv_models/`
  (`rdd2020_best.pt` or `pothole_yolo.pt`; otherwise generic `yolov8n.pt`, which is not pothole-trained).
  The weights are pre-trained for the RDD2020 road-damage dataset and were downloaded from the web.
- **PPO and SAC agents are trained and evaluated only in a synthetic simulator** (`data/pothole_generator.py`,
  `data/road_network.py`). No real road data is used for reinforcement learning, so the numbers below say
  nothing about real-world performance.
- `data/synthetic_data.py` generates **synthetic** municipal-style, UK-style and damage-class-mix records.
  The class proportions in it are hand-entered approximations, not computed from any dataset.
- The point-cloud module has been validated on synthetic scenes only (see below). The ROS 2 node has not been
  run on a ROS 2 install yet.

## How it fits together

```
road image --> YOLOv8 + contour/texture ensemble --> pothole boxes, severity
          \--> MiDaS (relative inverse depth) --> point cloud --> RANSAC road plane --> depth/area per pothole
                                                                                         |
boxes + measurements --> PPO priority agent (repair order) --> SAC material estimator --> A* route animation
                                                                                         |
                                                                         Streamlit app / ROS 2 PointCloud2
```

## Repository layout

```
├── config.py
├── complete_pipeline.py        main Streamlit app
├── cv_detection.py             YOLO + classical-CV ensemble, MiDaS depth
├── llm_analyzer.py             rule-based analysis text (Phi-2 branch is optional)
├── pointcloud_analysis.py      Open3D point cloud, RANSAC plane, pothole depth/area
├── pointcloud_viz.py           Plotly plane-vs-pothole figure, .ply export
├── test_pointcloud.py          synthetic ground-truth test for the point-cloud module
├── train_priority_drl.py       trains models/priority_drl.zip
├── train_material_sac.py       trains models/material_estimator_sac.zip
├── evaluate_agents.py          agents vs baselines -> reports/evaluation_results.json
├── verify_priority_agent.py    inspects the trained priority agent
├── main.py, train_improved.py, analyze_results.py   yearly budget-scheduling branch (optional)
├── train_moving_agent.py       moving-truck agent (optional)
├── integrate_synthetic_data.py synthetic-record initialization demo (optional)
├── data/                       pothole_generator.py, road_network.py, synthetic_data.py
├── environment/                priority_repair_env.py, repair_env.py, moving_agent_env.py
├── models/                     drl_agent.py, sac_material_estimator.py (+ trained .zip files)
├── cv_models/                  your YOLO weights (not in git)
└── ros2/                       pothole_pointcloud_node.py, launch/pothole_pointcloud.launch.py
```

## Setup

Use Python 3.9-3.11 (the pinned `numpy==1.24.3` has no 3.12 wheels).

```bash
git clone https://github.com/regularaswin007/IntelliRoad.git
cd IntelliRoad
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

- YOLO weights go in `cv_models/` (see Scope above). They are not committed.
- MiDaS (`MiDaS_small`) is downloaded on first use through `torch.hub`, so the first run needs internet.

## Run

```bash
python train_priority_drl.py          # trains the PPO priority agent
python train_material_sac.py          # trains the SAC material estimator
python evaluate_agents.py             # compares both against baselines; feeds the Analytics page
streamlit run complete_pipeline.py    # app: Upload & Detect / Simulate Repairs / Analytics
python test_pointcloud.py             # point-cloud geometry check (no models needed)
```

Upload & Detect works without trained agents. Simulate Repairs needs the two `.zip` models.

## Optional: Phi-2 branch

The Streamlit app uses the rule-based analyzer, so none of this is needed to run it. `train_llm.py` fine-tunes
Phi-2 on synthetic, template-generated inspection reports, and `llm_analyzer.PotholeAnalyzer` can load a local
Phi-2 download. Install the extra packages first:

```bash
pip install -r requirements-llm.txt
```

This branch is not wired into the app, and no evaluation results for it are included.

## Results (synthetic simulator)

Produced by `python evaluate_agents.py` (50 seeded episodes per policy for priority, 1000 seeded potholes for
material). Every policy sees the same episodes.

| Priority policy | Mean episode reward | Critical potholes fixed (of ~36) | Travel distance |
|---|---|---|---|
| random | 1913 ± 335 | 9.3 | 203 |
| urgency-sort baseline | 5089 ± 163 | 29.7 | 50 |

The urgency-sort baseline is strong, so a trained PPO agent only means something if it beats it. Results for the
trained PPO and SAC agents are not included here; run `python evaluate_agents.py` after training to add them to
the Analytics page.

| Material policy | MAE vs target multiplier | Within 0.05 of target |
|---|---|---|
| fixed 1.0 | 0.254 | 10% |

The SAC target is the rule-based multiplier in `MaterialEstimationEnv`, so this measures how well SAC
reproduces that rule on synthetic potholes, not real material usage.

> ## MiDaS depth is relative, not metric
> MiDaS predicts *relative inverse depth*: it is only defined up to an unknown scale and shift, and its
> geometry is not guaranteed to be consistent with a real pinhole camera. A point cloud built from it is
> **not a metric reconstruction**.
>
> - With no calibration, every length the point-cloud module reports is in **relative units**
>   (`metric=False`) and must not be read as cm or m. These values are **not** passed to the repair planner.
> - If you give the camera height above the road, the cloud is rescaled so the fitted road plane sits that
>   far below the camera (`metric=True`). That is a one-number calibration that makes values *approximately*
>   metric. It does not fix MiDaS's non-affine errors.
> - For real measurements use real depth (RGB-D, stereo, LiDAR or a metric depth model) and call
>   `analyze(..., depth_is_metric=True)`. The geometry code is the same.
>
> The detector's original `depth` field (`MiDaS max x 20`, shown as "cm") is also a relative heuristic, not a
> measurement. It is left unchanged unless the plane-based measurement is calibrated.

## Point-cloud module

`pointcloud_analysis.py` (Open3D + numpy):

1. Back-project the depth map to a point cloud with camera intrinsics (assumed from a horizontal FOV, or from
   ROS `CameraInfo`).
2. Road candidates = points **outside the YOLO boxes** (expanded by 25%) in the lower 65% of the image.
3. Voxel-downsample (1% of median road depth), then `remove_statistical_outlier` (20 neighbors, 2 sigma).
4. Fit the road plane with `segment_plane` (RANSAC, 1000 iterations, seeded, threshold 0.5% of median road depth).
5. Per detection: full-resolution pixels in its box that lie **below the plane** by more than the threshold.
   Depth = 95th percentile of distance below the plane (max also reported). Area = sum of per-pixel surface
   footprints, `(z/fx)(z/fy)cos(alpha)/cos(theta)`.
6. Failures return a status instead of a number: too few road points, weak plane fit (inlier ratio < 20%), or no
   depression below the plane.

`annotate_detections()` writes results into the detection dicts. Only when the result is **metric and valid**
are `depth` (cm), `area` (m2) and `severity` overwritten, which then flows into `Pothole(...)`, the PPO
priority observation and the SAC material estimator. Otherwise the old heuristic values are kept and the
relative measurement is stored as `plane_depth` / `plane_area`. In the app, use the sidebar section
"Point-Cloud Measurement (Open3D)" to enter the camera height.

### Validation on synthetic scenes (`test_pointcloud.py`)
A 960x720 road plane (camera 1.4 m up, pitched 15 deg, 0.2% depth noise) with two bowl-shaped potholes of known
size:

| Case | Camera height est. | Pothole 1 (R 0.30 m, 8 cm deep) | Pothole 2 (R 0.20 m, 4 cm deep) |
|---|---|---|---|
| Metric depth input | 1.400 m (true 1.4) | depth p95 7.8 cm, max 8.8 cm; area 0.226 m2 | depth p95 4.0 cm, max 4.8 cm; area 0.073 m2 |
| Unknown-scale depth (x3.7) + `camera_height_m` | 1.400 m | same as above | same as above |
| Idealized MiDaS-like relative inverse depth + heuristic + `camera_height_m` | 1.400 m (by construction) | depth p95 6.2 cm, area 0.118 m2 | depth p95 3.4 cm, area 0.047 m2 |

- With true depth the plane normal error was 0.00 deg and depth/area matched the expected values within about 3%.
  Max depth reads high because of noise.
- Measured area is smaller than the full bowl (0.226 vs 0.283 m2; 0.073 vs 0.126 m2) because pixels within the
  1.7 cm plane threshold are not counted. Shallow potholes lose proportionally more area.
- In the idealized relative-depth case (affine inverse depth, no noise) the heuristic depth conversion alone cut
  area by roughly 40-50% and depth by 13-20%, with a 4 deg plane error. Real MiDaS output has additional error
  that has not been measured here.
- Random depth gives `weak_plane_fit` instead of a measurement; no detections still fits the plane.

No real photo has been evaluated with this module, so there are no accuracy numbers on real data.

## ROS 2 node (`ros2/`) - not yet run on a ROS 2 install

`pothole_pointcloud_node.py` subscribes to a `sensor_msgs/Image` (bgr8, rgb8, bgra8, rgba8, mono8; no cv_bridge),
runs detector + MiDaS + the analyzer, and publishes a colored `sensor_msgs/PointCloud2` (grey = other, green =
road plane, yellow-to-red = pothole points by depth below plane) in the camera optical frame (x right, y down,
z forward), throttled by `min_period_s`.

The node needs the **same Python minor version ROS 2 was built for** (Humble on Ubuntu 22.04 = 3.10) and
`numpy<2`:

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages ~/intelliroad_venv
source ~/intelliroad_venv/bin/activate
pip install -r requirements.txt
python -c "import rclpy, open3d, torch, ultralytics; print('imports ok')"

# from the project root, ROS sourced and the venv active
ros2 launch ros2/launch/pothole_pointcloud.launch.py image_path:=/path/to/road.jpg camera_height_m:=1.4 rviz:=true
```

In RViz set **Fixed Frame** to `camera_optical_frame`, add a PointCloud2 display on
`/intelliroad/pothole_cloud`, and set Color Transformer to RGB8. The launch file starts the node with the active
venv's python (override: `python_executable:=...`). Launch arguments: `image_topic`, `camera_info_topic`,
`cloud_topic`, `frame_id`, `image_path`, `model_dir`, `hfov_deg`, `camera_height_m`, `sensitivity`,
`min_period_s`, `rviz`, `rviz_fixed_frame`. Unset strings use `none`. The project is not an ament package, so the
launch file starts the script with `ExecuteProcess`.

## Known limitations
- One dominant plane: crowned roads, slopes and curbs break the single-plane assumption.
- The plane threshold limits the smallest measurable depth; water-filled holes, shadows and boxes that do not
  enclose the pothole give wrong or missing measurements.
- Camera-height calibration assumes the road plane is the ground and the height is accurate.
- The detector runs MiDaS once inside `detect()` and again in `estimate_depth()` (extra latency).
- Agent results come from a simulator with synthetic potholes and a hand-designed reward.

## Author
Developed individually by [@regularaswin007](https://github.com/regularaswin007).

## Third-party software and data
Ultralytics YOLOv8 (AGPL-3.0, review its terms before redistributing), MiDaS (Intel ISL, via `torch.hub`),
Open3D, Stable-Baselines3, Gymnasium, PyTorch, OpenCV, NetworkX, Streamlit, Plotly, Pygame (optional),
Phi-2 (optional). Dataset: RDD2020 - D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, Y. Sekimoto,
"RDD2020: An annotated image dataset for automatic road damage detection using deep learning", Data in Brief, vol. 36, 107133, 2021.
Check each project's license and the dataset's terms.

## License
No license has been specified yet (all rights reserved).
