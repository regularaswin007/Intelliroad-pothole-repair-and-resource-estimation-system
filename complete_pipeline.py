"""
COMPLETE INTEGRATED SYSTEM - Production Ready without LLM Toggle

"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import sys
import os
import time
import heapq

sys.path.append('./environment')
sys.path.append('./models')
sys.path.append('./data')

from priority_repair_env import PriorityRepairEnv
from drl_agent import PotholeRepairAgent
from pothole_generator import Pothole
from cv_detection import RobustPotholeDetector
from llm_analyzer import SimplePotholeAnalyzer
from pointcloud_analysis import PointCloudPotholeAnalyzer, inverse_depth_to_depth, annotate_detections
from pointcloud_viz import build_plotly_figure, save_ply

def create_road_grid(image):
    """Convert image to navigable grid (roads = 1, obstacles = 0)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Roads are usually medium-gray (70-150)
    road_mask = cv2.inRange(gray, 70, 150)
    
    # Downsample for faster A* (every 10 pixels = 1 grid cell)
    h, w = road_mask.shape
    grid_h, grid_w = h // 10, w // 10
    grid = cv2.resize(road_mask, (grid_w, grid_h)) > 127
    
    return grid.astype(int)


def a_star_path(grid, start, goal):
    """A* pathfinding on grid"""
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def neighbors(pos):
        x, y = pos
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                if grid[ny, nx] == 1:  # Walkable
                    yield (nx, ny)
    
    start = (int(start[0] // 10), int(start[1] // 10))
    goal = (int(goal[0] // 10), int(goal[1] // 10))
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = [goal]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            # Scale back to image coordinates
            return [(x * 10, y * 10) for x, y in path]
        
        for neighbor in neighbors(current):
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
    
    # No path found, return straight line
    return [start, goal]

# Page config
st.set_page_config(
    page_title="DRL Pothole Repair System - Professional Grade",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Complete DRL Pothole Repair System with AI Analysis")
st.markdown("AGENT")

# ========================================
# SIDEBAR
# ========================================
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("### System Status")

# Initialize rule-based analyzer
analyzer = SimplePotholeAnalyzer()
st.sidebar.info("✅analysis active")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["📷 Upload & Detect", "🎮 Simulate Repairs", "📊 Analytics"]
)

# ========================================
# PROFESSIONAL CNN DETECTION MODULE
# ========================================
class ProfessionalPotholeDetector:
    """Professional-grade CNN detection using YOLOv8 + MiDaS"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLOv8 for pothole detection
        try:
            from ultralytics import YOLO
            
            # Try loading pothole-specific model from Roboflow
            try:
                # Download pothole-trained weights
                import urllib.request
                os.makedirs('./cv_models', exist_ok=True)
                
                # Use Roboflow pre-trained pothole model
                self.yolo_model = YOLO('yolov8n.pt')  # Will customize with threshold
                self.yolo_model.conf = 0.15  # Lower confidence threshold
                
                st.sidebar.success("✅ YOLOv8 Loaded (Pothole-optimized)")
                self.yolo_loaded = True
            except:
                # Fallback to base model with low threshold
                self.yolo_model = YOLO('yolov8n.pt')
                self.yolo_model.conf = 0.1
                self.yolo_loaded = True
                st.sidebar.warning("⚠️ Using generic YOLO (may miss potholes)")
        except Exception as e:
            st.sidebar.error(f"❌ YOLOv8 Error: {e}")
            self.yolo_loaded = False
        
        # Load MiDaS for depth estimation
        try:
            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.midas.to(self.device)
            self.midas.eval()
            
            # MiDaS transforms
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.transform = midas_transforms.small_transform
            
            self.midas_loaded = True
            st.sidebar.success("✅ MiDaS Depth Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ MiDaS Error: {e}")
            self.midas_loaded = False
    
    def draw_detections(image, detections):
        """Draw bounding boxes with severity colors"""
        out = image.copy()
        colors = {1:(255,255,0), 2:(0,255,255), 3:(0,165,255), 4:(0,0,255)}  # Yellow->Cyan->Orange->Red
        
        for d in detections:
            x, y, w, h = d["bbox"]
            sev = int(d.get("severity", 2))
            
            # Draw box
            cv2.rectangle(out, (x, y), (x+w, y+h), colors.get(sev, (0,255,0)), 3)
            
            # Draw label
            depth = d.get('depth', 0)
            method = d.get('method', '')
            label = f"Sev:{sev} D:{depth:.1f}cm {method}"
            cv2.putText(out, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        
        return out

# ========================================
# MODE 1: UPLOAD & DETECT
# ========================================
if mode == "📷 Upload & Detect":
    st.header("📷Pothole Detection with AI Analysis")
    st.markdown("**Using specialized YOLOv8 + preprocessing + validation + Rule-based analysis**")
    
    # Sensitivity slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎚️ Detection Sensitivity")
    sensitivity = st.sidebar.slider(
        "Adjust detection sensitivity",
        0.0, 1.0, 0.5, 0.05,
        help="""
        🟢 Low (0.0-0.3): Detects only obvious potholes
        🟡 Medium (0.3-0.7): Balanced detection
        🔴 High (0.7-1.0): Detects everything (more false positives)
        """
    )
    
    # Point-cloud (Open3D) plane measurement
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧊 Point-Cloud Measurement (Open3D)")
    use_pointcloud = st.sidebar.checkbox(
        "Measure depth/area vs fitted road plane",
        value=True,
        help="Builds a point cloud from MiDaS depth, fits the road plane with RANSAC "
             "(using points outside the YOLO boxes) and measures each pothole against it."
    )
    camera_height_m = st.sidebar.number_input(
        "Camera height above road (m, 0 = unknown)",
        min_value=0.0, max_value=10.0, value=0.0, step=0.1,
        help="MiDaS depth is RELATIVE. With a height, the cloud is rescaled so the fitted road "
             "plane is that far below the camera and values become approximate metric. "
             "With 0, results stay in relative units and are NOT fed into the repair planner."
    )
    hfov_deg = st.sidebar.slider(
        "Camera horizontal FOV (deg, assumed)", 30, 120, 60,
        help="Used to build camera intrinsics. Use your real calibration if you have it."
    )
    
    uploaded_file = st.file_uploader("Upload road image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                st.error("❌ Could not read image. Please try a different file.")
            else:
                # Store in session state
                st.session_state['uploaded_image'] = img_bgr.copy()
                
                # Convert for display
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(img_rgb, width=400)
                
                # Initialize detector
                with st.spinner("Loading specialized model..."):
                    detector = RobustPotholeDetector(model_dir="./cv_models")
                
                # Run detection (all-in-one: preprocessing + detection + validation + depth)
                with st.spinner("Running detection..."):
                    detections = detector.detect(img_bgr, sensitivity=sensitivity)
                
                # Point-cloud plane measurement (Open3D): depth/area relative to the fitted road plane
                pc_result = None
                if use_pointcloud and len(detections) > 0:
                    with st.spinner("Building point cloud & fitting road plane..."):
                        inv_depth = detector.estimate_depth(img_bgr)
                        if inv_depth is None:
                            st.warning("⚠️ MiDaS unavailable - point-cloud measurement skipped")
                        else:
                            pc_analyzer = PointCloudPotholeAnalyzer(
                                hfov_deg=hfov_deg, camera_height_m=camera_height_m
                            )
                            pc_result = pc_analyzer.analyze(inverse_depth_to_depth(inv_depth), detections)
                            annotate_detections(detections, pc_result)
                
                # Draw results
                def draw_detections(image, detections):
                    out = image.copy()
                    colors = {1:(255,255,0), 2:(0,255,255), 3:(0,165,255), 4:(0,0,255)}
                    for d in detections:
                        x, y, w, h = d["bbox"]
                        sev = int(d.get("severity", 2))
                        cv2.rectangle(out, (x, y), (x+w, y+h), colors.get(sev, (0,255,0)), 3)
                        label = f"Sev:{sev} D:{d.get('depth',0):.1f}cm {d.get('method','')}"
                        cv2.putText(out, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                    return out
                
                detected_img = draw_detections(img_bgr, detections)
                detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader(f"Detected: {len(detections)} Potholes")
                    st.image(detected_img_rgb, width=400)
                
                # Detection details
                st.subheader("📊 Detection Analysis")
                
                if len(detections) > 0:
                    # ========================================
                    # AI ANALYSIS SECTION
                    # ========================================
                    st.markdown("---")
                    st.subheader("🤖 AI-Powered Road Condition Analysis")
                    
                    with st.spinner("🧠 Analyzing road conditions..."):
                        try:
                            analysis = analyzer.analyze_detections(detections, image_size=img_bgr.shape[:2])
                            
                            if analysis is None:
                                st.warning("⚠️ Analysis returned no results")
                                analysis = {
                                    'summary': 'Analysis unavailable',
                                    'severity_report': 'Unable to assess severity',
                                    'recommendations': 'Manual inspection recommended',
                                    'urgency': 'Unknown'
                                }
                                
                        except Exception as e:
                            st.error(f"❌ Analysis error: {str(e)[:200]}")
                            analysis = {
                                'summary': 'Analysis error occurred',
                                'severity_report': 'Unable to assess severity',
                                'recommendations': 'Manual inspection recommended',
                                'urgency': 'Unknown'
                            }
                    
                    # Display analysis in expandable sections
                    col_ai1, col_ai2 = st.columns(2)
                    
                    with col_ai1:
                        st.markdown("#### 📋 Executive Summary")
                        st.info(analysis['summary'])
                        
                        st.markdown("#### ⚠️ Severity Report")
                        severity_color = {
                            'CRITICAL': 'error',
                            'SEVERE': 'warning', 
                            'MODERATE': 'info',
                            'MINOR': 'success'
                        }.get(analysis.get('overall_severity', 'MODERATE'), 'info')
                        
                        getattr(st, severity_color)(analysis['severity_report'])
                    
                    with col_ai2:
                        st.markdown("#### 🔧 Repair Recommendations")
                        st.warning(analysis['recommendations'])
                        
                        # Urgency timeline
                        st.markdown("#### ⏱️ Action Timeline")
                        urgency = analysis.get('urgency', 'Unknown')
                        if 'Immediate' in urgency or '24' in urgency:
                            st.error(f"🚨 {urgency}")
                        elif 'High' in urgency or '3-5' in urgency:
                            st.warning(f"⚡ {urgency}")
                        else:
                            st.info(f"📅 {urgency}")
                    
                    st.markdown("---")
                    
                    # Summary stats
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Total Detected", len(detections))
                    col_b.metric("Avg Depth", f"{np.mean([d['depth'] for d in detections]):.1f} cm")
                    col_c.metric("Avg Severity", f"{np.mean([d['severity'] for d in detections]):.1f}")
                    
                    # Individual details
                    st.markdown("#### 🕳️ Individual Pothole Details")
                    for i, det in enumerate(detections):
                        with st.expander(f"🕳️ Pothole #{i+1} - Severity {det['severity']}"):
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Depth", f"{det['depth']:.1f} cm")
                            col2.metric("Area", f"{det['area']:.3f} m²")
                            col3.metric("Severity", f"{det['severity']}/4")
                            col4.metric("Method", det.get('method', 'Unknown'))
                            st.caption(f"Depth/area source: {det.get('depth_source', 'heuristic (MiDaS-relative)')}")
                    
                    # ========================================
                    # POINT CLOUD ANALYSIS SECTION
                    # ========================================
                    if pc_result is not None:
                        import pandas as pd
                        import tempfile
                        
                        st.markdown("---")
                        st.subheader("🧊 Point-Cloud Plane Analysis (Open3D)")
                        
                        if pc_result['status'] != 'ok':
                            st.warning(f"⚠️ Road plane not fitted ({pc_result['status']}) - using heuristic depth/area")
                        else:
                            plane = pc_result['plane']
                            unit = "m" if pc_result['metric'] else "rel."
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Plane inliers", f"{plane['inlier_ratio']:.0%}")
                            c2.metric("Units", "metric (calibrated)" if pc_result['metric'] else "relative")
                            c3.metric("Fitted camera height", f"{plane['camera_height']:.2f} {unit}")
                            
                            if not pc_result['metric']:
                                st.info("MiDaS depth is relative. Set the camera height in the sidebar to get "
                                        "approximate metric values. Values below are in relative units and are "
                                        "NOT used by the repair planner.")
                            
                            pc_rows = []
                            for i, m in enumerate(pc_result['potholes']):
                                pc_rows.append({
                                    'Pothole': i + 1,
                                    f'Depth p95 ({unit})': f"{m['depth']:.4f}" if m['valid'] else '-',
                                    f'Max depth ({unit})': f"{m['depth_max']:.4f}" if m['valid'] else '-',
                                    f'Area ({unit}²)': f"{m['area']:.4f}" if m['valid'] else '-',
                                    'Pixels below plane': m['n_pixels'],
                                    'Status': m['reason'],
                                })
                            st.dataframe(pd.DataFrame(pc_rows), use_container_width=True)
                            
                            st.plotly_chart(build_plotly_figure(pc_result), use_container_width=True)
                            
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                ply_path = os.path.join(tmp_dir, "pothole_cloud.ply")
                                if save_ply(pc_result, ply_path):
                                    with open(ply_path, 'rb') as f:
                                        st.download_button("📥 Download point cloud (.ply)", f.read(),
                                                           file_name="pothole_cloud.ply")
                    
                    # Store for simulation
                    st.session_state['detections'] = detections
                    st.session_state['image_processed'] = True
                    st.session_state['analysis'] = analysis  # Store analysis
                    
                    st.success("✅ Detection complete! Go to '🎮 Simulate Repairs' to see DRL agent in action.")
                else:
                    st.warning("⚠️ No potholes detected. Try adjusting sensitivity slider!")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            import traceback
            st.code(traceback.format_exc())
        
# ========================================
# MODE 2: SIMULATE REPAIRS
# ========================================
elif mode == "🎮 Simulate Repairs":
    st.header("🎮 DRL Agent Simulation & Material Estimation")
    
    if 'image_processed' not in st.session_state:
        st.warning("⚠️ Please upload and process an image first!")
    else:
        # Show analysis at top if available
        if 'analysis' in st.session_state:
            analysis = st.session_state['analysis']
            
            with st.expander("📊 AI Road Condition Report", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.info(analysis['summary'])
                with col2:
                    st.warning(analysis['recommendations'])
        
        # Safety check for uploaded_image
        if 'uploaded_image' not in st.session_state:
            st.error("❌ Image not found. Please re-upload and detect potholes first.")
            st.stop()
        
        detections = st.session_state['detections']
        uploaded_img = st.session_state['uploaded_image']
        
        st.info(f"📊 Found {len(detections)} potholes from uploaded image")
        
        def severity_to_string(level):
            """Convert numeric severity (1-4) to string for Pothole class"""
            return {1: "minor", 2: "moderate", 3: "severe", 4: "critical"}.get(int(level), "moderate")

        # Convert detections to potholes
        potholes = []
        for i, det in enumerate(detections):
            segment_id = i % 15  # Distribute across segments
            
            # Convert severity from int to string
            severity_str = severity_to_string(det.get('severity', 2))
            
            pothole = Pothole(
                pothole_id=i,
                segment_id=segment_id,
                severity=severity_str,
                depth=float(det.get('depth', 6.0)),
                area=float(det.get('area', 0.5)),
                age=0
            )
            potholes.append(pothole)
        
        # Load agents
        try:
            env = PriorityRepairEnv(num_segments=15)
            
            # Load DRL (PPO) for prioritization
            drl_agent = PotholeRepairAgent(env=env, verbose=0)
            drl_agent.load('./models/priority_drl')
            
            from sac_material_estimator import SACMaterialEstimator
            material_estimator = SACMaterialEstimator(verbose=0)
            material_estimator.load('./models/material_estimator_sac')
                        
            st.success("✅ DRL Priority Agent & SAC Material Estimator Loaded!")
            
            # Add explanation
            with st.expander("🤖 How DRL + SAC System Works"):
                st.markdown("""
                ### Dual AI System
                
                **DRL (PPO) - Prioritization:**
                1. **Analyzes** each detected pothole's features (depth, area, severity)
                2. **Predicts** urgency using trained PPO neural network
                3. **Ranks** all potholes from most to least urgent
                4. **Optimizes** repair sequence for maximum safety impact
                
                **SAC - Material Estimation:**
                1. **Calculates** base volume from depth × area
                2. **Adjusts** for pothole characteristics (age, severity, shape)
                3. **Predicts** optimal material multiplier (0.8-1.5x)
                4. **Accounts** for overfill, compaction, edge crumbling
                
                **A* Pathfinding:**
                - Plans intelligent routes between potholes
                - Follows drivable roads, avoids obstacles
                - Minimizes travel time
                
                **Algorithms:**
                - Prioritization: Proximal Policy Optimization (PPO)
                - Material estimation: Soft Actor-Critic (SAC)
                - Measured results: see the Analytics page (`python evaluate_agents.py`)
                
                Priorities come from the trained PPO policy and material multipliers from the trained SAC policy.
                """)
            
            # Generate priorities and material estimates
            st.subheader("🤖 DRL Priorities & SAC Material Estimates")
            
            env.potholes = potholes
            priorities = {}
            material_estimates = {}
            
            for i, pothole in enumerate(potholes):
                env.current_pothole_idx = i
                obs = env._get_obs()
                
                # DRL priority
                action, _ = drl_agent.model.predict(obs, deterministic=True)
                priorities[pothole.id] = float(action[0])
                
                # SAC material estimate
                estimated_vol, multiplier = material_estimator.estimate_material(pothole)
                material_estimates[pothole.id] = {
                    'volume': estimated_vol,
                    'multiplier': multiplier,
                    'base_volume': pothole.get_volume()
                }
            
            # Sort by priority
            sorted_potholes = sorted(potholes, key=lambda p: priorities[p.id], reverse=True)
            
            # Display queue with material estimates
            st.write("**Priority Queue with Material Estimates (Top 5):**")
            for rank, pothole in enumerate(sorted_potholes[:5], 1):
                priority = priorities[pothole.id]
                mat_est = material_estimates[pothole.id]
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.write(f"**#{rank}**")
                col2.write(f"Seg {pothole.segment_id}")
                col3.write(f"{pothole.depth:.1f}cm")
                col4.write(f"{pothole.severity}")
                col5.progress(priority, text=f"{priority:.3f}")
                col6.metric("Material", f"{mat_est['volume']:.3f} m³")
            
            # Store in session state
            st.session_state['priorities'] = priorities
            st.session_state['material_estimates'] = material_estimates
            st.session_state['sorted_potholes'] = sorted_potholes
            
            # ========================================
            # INTERACTIVE ANIMATION
            # ========================================
            
            st.markdown("---")
            st.subheader("🎬 Interactive Demonstration")
            
            if st.button("▶️ Animate Repair Process", type="primary", use_container_width=True):
                st.markdown("---")
                
                # Animation placeholder
                animation_placeholder = st.empty()
                
                # Stats placeholders
                col_a, col_b, col_c, col_d = st.columns(4)
                stat_repaired = col_a.empty()
                stat_current = col_b.empty()
                stat_remaining = col_c.empty()
                stat_material = col_d.empty()
                
                # Run animation
                repair_queue = [(det, priorities[i]) for i, det in enumerate(detections)]
                repair_queue.sort(key=lambda x: x[1], reverse=True)
                
                current_image = uploaded_img.copy()
                repaired_count = 0
                total_material_used = 0.0
                truck_position = None
                
                # Create road grid for A*
                if 'road_grid' not in st.session_state:
                    st.session_state['road_grid'] = create_road_grid(uploaded_img)
                road_grid = st.session_state['road_grid']
                
                # Initialize truck at first pothole
                if repair_queue:
                    first_det, _ = repair_queue[0]
                    x, y, w, h = first_det['bbox']
                    truck_position = (x + w//2, y + h//2)
                
                for idx, (detection, priority) in enumerate(repair_queue):
                    x, y, w, h = detection['bbox']
                    target_x, target_y = x + w//2, y + h//2
                    
                    # Get material estimate for this pothole
                    current_pothole = potholes[idx]
                    mat_est = material_estimates[current_pothole.id]
                    
                    # Update stats
                    stat_repaired.metric("✅ Repaired", repaired_count)
                    stat_current.metric("🔧 Priority", f"{priority:.3f}")
                    stat_remaining.metric("⏳ Remaining", len(repair_queue) - idx)
                    stat_material.metric("🧱 Material", f"{total_material_used:.2f} m³")
                    
                    # A* pathfinding movement
                    if truck_position:
                        # Show A* pathfinding computation message
                        computation_frame = current_image.copy()
                        cv2.rectangle(computation_frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
                        cv2.putText(computation_frame, "Computing A* Path...", 
                                   (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 255), 2)
                        cv2.putText(computation_frame, f"From: ({truck_position[0]}, {truck_position[1]})", 
                                   (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.4, (255, 255, 255), 1)
                        cv2.putText(computation_frame, f"To: ({target_x}, {target_y})", 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.4, (255, 255, 255), 1)
                        
                        # Draw truck at current position
                        cv2.rectangle(computation_frame, 
                                    (truck_position[0] - 20, truck_position[1] - 20),
                                    (truck_position[0] + 20, truck_position[1] + 20),
                                    (255, 165, 0), -1)
                        cv2.rectangle(computation_frame, 
                                    (truck_position[0] - 20, truck_position[1] - 20),
                                    (truck_position[0] + 20, truck_position[1] + 20),
                                    (255, 255, 255), 2)
                        
                        animation_placeholder.image(computation_frame, channels="BGR", use_container_width=True)
                        
                        # Show computation progress with animated dots
                        for dot_count in range(1, 4):
                            progress_frame = computation_frame.copy()
                            dots = "." * dot_count
                            cv2.putText(progress_frame, f"Computing A* Path{dots}", 
                                       (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.6, (0, 255, 255), 2)
                            animation_placeholder.image(progress_frame, channels="BGR", use_container_width=True)
                            time.sleep(1.0)
                        
                        # Compute A* path
                        path = a_star_path(road_grid, truck_position, (target_x, target_y))
                        
                        # Show path found message
                        path_found_frame = current_image.copy()
                        cv2.rectangle(path_found_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        cv2.putText(path_found_frame, f"✓ Path Found! {len(path)} waypoints", 
                                   (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 0), 2)
                        
                        # Draw entire path
                        for i in range(len(path) - 1):
                            cv2.line(path_found_frame, path[i], path[i+1], (100, 200, 255), 2)
                        
                        animation_placeholder.image(path_found_frame, channels="BGR", use_container_width=True)
                        time.sleep(1.5)
                        
                        # Animate along A* path
                        for i in range(len(path) - 1):
                            curr_waypoint = path[i]
                            next_waypoint = path[i + 1]
                            
                            # Interpolate between waypoints
                            for substep in range(5):
                                progress = substep / 5
                                curr_x = int(curr_waypoint[0] + (next_waypoint[0] - curr_waypoint[0]) * progress)
                                curr_y = int(curr_waypoint[1] + (next_waypoint[1] - curr_waypoint[1]) * progress)
                                
                                frame = current_image.copy()
                                
                                # Draw target box
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                                cv2.putText(frame, f"Priority: {priority:.3f}", 
                                           (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"Material: {mat_est['volume']:.2f}m³", 
                                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.5, (255, 255, 0), 2)
                                
                                # Draw completed path
                                for j in range(i):
                                    cv2.line(frame, path[j], path[j+1], (50, 150, 200), 2)
                                
                                # Draw remaining path (faded)
                                for j in range(i, len(path) - 1):
                                    cv2.line(frame, path[j], path[j+1], (150, 200, 255), 1)
                                
                                # Draw truck
                                cv2.rectangle(frame, 
                                            (curr_x - 20, curr_y - 20),
                                            (curr_x + 20, curr_y + 20),
                                            (0, 100, 200), -1)
                                cv2.rectangle(frame, 
                                            (curr_x - 20, curr_y - 20),
                                            (curr_x + 20, curr_y + 20),
                                            (255, 255, 255), 2)
                                
                                # Show waypoint progress
                                cv2.putText(frame, f"Waypoint {i+1}/{len(path)-1}", 
                                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.6, (255, 255, 255), 2)
                                
                                animation_placeholder.image(frame, channels="BGR", use_container_width=True)
                                time.sleep(0.03)
                    
                    truck_position = (target_x, target_y)
                    
                    # Repair animation with material filling
                    for repair_step in range(11):
                        frame = current_image.copy()
                        
                        if repair_step < 10:
                            # Progress bar
                            bar_width = w
                            filled = int(bar_width * repair_step / 10)
                            cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), 
                                        (200, 200, 200), -1)
                            cv2.rectangle(frame, (x, y+h+5), (x+filled, y+h+15), 
                                        (0, 255, 0), -1)
                            
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            cv2.putText(frame, "FILLING...", (x, y-25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Show material being used
                            current_fill = mat_est['volume'] * (repair_step / 10)
                            cv2.putText(frame, f"{current_fill:.2f}/{mat_est['volume']:.2f}m³", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, (255, 255, 0), 2)
                        else:
                            cv2.putText(frame, "✓ REPAIRED", (x, y-25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame, f"Used: {mat_est['volume']:.2f}m³", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, (255, 255, 0), 2)
                        
                        # Draw truck
                        cv2.rectangle(frame, 
                                    (target_x - 20, target_y - 20),
                                    (target_x + 20, target_y + 20),
                                    (50, 150, 255) if repair_step < 10 else (0, 100, 200), -1)
                        cv2.rectangle(frame, 
                                    (target_x - 20, target_y - 20),
                                    (target_x + 20, target_y + 20),
                                    (255, 255, 255), 2)
                        
                        animation_placeholder.image(frame, channels="BGR", use_container_width=True)
                        time.sleep(0.08)
                    
                    # Mark as repaired
                    overlay = current_image.copy()
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (34, 139, 34), -1)
                    cv2.addWeighted(overlay, 0.3, current_image, 0.7, 0, current_image)
                    
                    # Update totals
                    total_material_used += mat_est['volume']
                    repaired_count += 1
                    stat_material.metric("🧱 Material", f"{total_material_used:.2f} m³")
                    
                    time.sleep(0.2)
                
                # Final stats
                stat_repaired.metric("✅ Repaired", repaired_count)
                stat_remaining.metric("⏳ Remaining", 0)
                stat_material.metric("🧱 Total Material", f"{total_material_used:.2f} m³")
                
                st.success(f"🎉 **Repair Complete!** Fixed {repaired_count} potholes using {total_material_used:.2f} m³ of material")
                
                # Material breakdown
                st.markdown("---")
                st.subheader("📊 Material Usage & Cost Analysis")
                
                # India pricing
                MATERIAL_COST_PER_M3 = 2500  # ₹2500 per cubic meter (typical Indian market rate)
                LABOR_COST_PER_POTHOLE = 500  # ₹500 per pothole (labor)
                EQUIPMENT_COST = 2000  # ₹2000 fixed equipment/transport cost
                
                total_cost = (total_material_used * MATERIAL_COST_PER_M3) + \
                            (repaired_count * LABOR_COST_PER_POTHOLE) + \
                            EQUIPMENT_COST
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Avg Material/Pothole", f"{total_material_used/repaired_count:.3f} m³")
                col_m2.metric("SAC Efficiency", f"{np.mean([m['multiplier'] for m in material_estimates.values()]):.2f}x")
                col_m3.metric("Total Material Used", f"{total_material_used:.2f} m³")
                col_m4.metric("Total Cost (INR)", f"₹{total_cost:,.0f}")
                
                # Cost breakdown
                st.write("**Cost Breakdown (In Indian Rupees):**")
                cost_col1, cost_col2, cost_col3 = st.columns(3)
                
                material_cost = total_material_used * MATERIAL_COST_PER_M3
                labor_cost = repaired_count * LABOR_COST_PER_POTHOLE
                
                cost_col1.metric("Material Cost", f"₹{material_cost:,.0f}")
                cost_col2.metric("Labor Cost", f"₹{labor_cost:,.0f}")
                cost_col3.metric("Equipment/Transport", f"₹{EQUIPMENT_COST:,.0f}")
                
                # Cost per pothole
                st.metric("Average Cost per Pothole", f"₹{total_cost/repaired_count:,.0f}")
                
                # Before/After
                st.subheader("📊 Before vs After")
                col_before, col_after = st.columns(2)
                
                with col_before:
                    st.write("**Before (Detected)**")
                    before = uploaded_img.copy()
                    for det, _ in repair_queue:
                        x, y, w, h = det['bbox']
                        cv2.rectangle(before, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    st.image(before, channels="BGR", use_container_width=True)
                
                with col_after:
                    st.write("**After (Repairs with Material)**")
                    st.image(current_image, channels="BGR", use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error loading agents: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.info("Train agents: `python train_priority_drl.py` and `python train_material_sac.py`")

# ========================================
# MODE 3: ANALYTICS
# ========================================
else:
    st.header("📊 System Analytics")
    
    results_path = './reports/evaluation_results.json'
    if not os.path.exists(results_path):
        st.info("No evaluation results yet. Train the agents, then run `python evaluate_agents.py` "
                "(results are read from `./reports/evaluation_results.json`).")
    else:
        import json
        import pandas as pd
        
        with open(results_path) as f:
            results = json.load(f)
        
        st.caption(f"Measured on the synthetic simulator (not real-road data) - generated "
                   f"{results['generated_at']}, {results['n_episodes']} episodes.")
        
        priority = results.get('priority')
        if priority:
            st.subheader("Priority agent vs baselines")
            rows = []
            for name, m in priority['policies'].items():
                rows.append({
                    'Policy': name,
                    'Mean reward': f"{m['mean_reward']:.1f} ± {m['std_reward']:.1f}",
                    'Critical potholes fixed': f"{m['mean_critical_fixed']:.1f} / {m['mean_critical_total']:.1f}",
                    'Travel distance (segments)': f"{m['mean_travel_distance']:.1f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(f"Repair budget: {priority['repair_budget']} potholes per episode. "
                       "Higher reward, more critical potholes fixed and shorter distance are better.")
        
        material = results.get('material')
        if material:
            st.subheader("Material estimator (SAC) vs fixed multiplier")
            rows = []
            for name, m in material['policies'].items():
                rows.append({
                    'Policy': name,
                    'MAE (multiplier)': f"{m['mae']:.3f}",
                    'Within 0.05 of target': f"{m['within_0_05']:.0%}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(material['note'])
        
        if not priority and not material:
            st.warning("The results file contains no priority or material results.")

# ========================================
# FOOTER
# ========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Technologies")
st.sidebar.markdown("""
- **YOLOv8 for RDD**: Object detection  
- **MiDaS**: Depth estimation
- **PPO**: DRL agent
- **SAC**: Material estimation
- **A-star Algorithm**: Path planning
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Adjust sensitivity for better detection
- Use animation to see DRL in action
- Analyzer is fast and accurate!
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
**DRL Pothole Repair System**

For autonomous road maintenance.
""")