# cv_detection.py
# Combined Strategy 2 + 3 + 4 for robust pothole detection

import os
import cv2
import numpy as np
import torch

# Optional: use a specialized free model if present (RDD2020/Roboflow or any YOLO .pt you place)
# Fallback to generic YOLO if custom weights not found
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class RobustPotholeDetector:
    """
    Ensemble detector with:
      - Specialized YOLO weights if available (Strategy 2)
      - Image preprocessing (Strategy 3)
      - Post-validation filters (Strategy 4)
    """

    def __init__(self, model_dir: str = "./cv_models", device: str | None = None):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.yolo_model = None
        self.yolo_loaded = False

        if YOLO_AVAILABLE:
            # Priority: user-provided specialized weights
            # 1) ./cv_models/rdd2020_best.pt  (download from Roboflow RDD or your trained weights)
            # 2) ./cv_models/pothole_yolo.pt  (any pothole-trained weights)
            # 3) generic yolov8n.pt as fallback (will rely more on classical pipeline)
            weight_candidates = [
                os.path.join(self.model_dir, "rdd2020_best.pt"),
                os.path.join(self.model_dir, "pothole_yolo.pt"),
            ]

            path = None
            for wc in weight_candidates:
                if os.path.exists(wc):
                    path = wc
                    break

            try:
                if path is not None:
                    self.yolo_model = YOLO(path)
                    self.yolo_loaded = True
                else:
                    # Fallback for structure only; generic COCO is not specialized
                    self.yolo_model = YOLO("yolov8n.pt")
                    self.yolo_loaded = True
            except Exception:
                self.yolo_model = None
                self.yolo_loaded = False

        # MiDaS depth (optional for depth-based validation; won’t block if fails)
        self.midas = None
        self.midas_transform = None
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.to(self.device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.midas_transform = transforms.small_transform
        except Exception:
            self.midas = None
            self.midas_transform = None

    # =========================
    # Strategy 3: Preprocessing
    # =========================
    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        # 1) CLAHE (contrast)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2) Denoise
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, h=10)

        # 3) Shadow mitigation
        rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(rgb)
        bright = np.maximum(np.maximum(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bright_dil = cv2.dilate(bright, kernel)
        bright_dil = np.clip(bright_dil.astype(np.float32), 1.0, 255.0)
        r2 = np.uint8(np.clip(r.astype(np.float32) * 255.0 / bright_dil, 0, 255))
        g2 = np.uint8(np.clip(g.astype(np.float32) * 255.0 / bright_dil, 0, 255))
        b2 = np.uint8(np.clip(b.astype(np.float32) * 255.0 / bright_dil, 0, 255))
        shadow_fixed = cv2.merge([b2, g2, r2])  # back to BGR order

        return shadow_fixed

    def _get_depth_map(self, bgr: np.ndarray) -> np.ndarray | None:
        if self.midas is None or self.midas_transform is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            inp = self.midas_transform(rgb).to(self.device)
            pred = self.midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        d = pred.cpu().numpy()
        # normalize 0..1
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        return d

    def estimate_depth(self, bgr: np.ndarray) -> np.ndarray | None:
        """
        Public wrapper around MiDaS. Returns RELATIVE inverse depth min-max normalized
        to 0..1 (larger = nearer, no metric scale), run on the raw image (no preprocessing).
        Returns None if MiDaS is unavailable.
        """
        return self._get_depth_map(bgr)

    # =================================
    # Strategy 2: Specialized detector
    # =================================
    def _yolo_detect(self, bgr: np.ndarray, conf: float, iou: float):
        dets = []
        if not self.yolo_loaded:
            return dets
        try:
            res = self.yolo_model(bgr, conf=conf, iou=iou)
            for r in res:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = int(x2 - x1), int(y2 - y1)
                    dets.append(
                        dict(
                            bbox=[int(x1), int(y1), w, h],
                            confidence=float(box.conf[0]),
                            area=(w * h) / 10000.0,
                            severity=None,
                            depth=None,
                            method="YOLO",
                        )
                    )
        except Exception:
            pass
        return dets

    # ==========================================
    # Strategy 3 helpers: contour + texture cues
    # ==========================================
    def _contour_detect(self, bgr: np.ndarray, thresh_val: int, min_area: int, max_area: int):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            area_px = cv2.contourArea(c)
            if min_area <= area_px <= max_area:
                x, y, w, h = cv2.boundingRect(c)
                ar = w / (h + 1e-6)
                if 0.25 <= ar <= 4.0:
                    dets.append(
                        dict(
                            bbox=[x, y, w, h],
                            confidence=0.70,
                            area=(w * h) / 10000.0,
                            severity=None,
                            depth=None,
                            method="Contour",
                        )
                    )
        return dets

    def _texture_detect(self, bgr: np.ndarray, pct: float, min_area: int, max_area: int):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_abs = np.abs(lap)
        thr = np.percentile(lap_abs, pct)  # 80–92 suggested
        _, th = cv2.threshold(lap_abs.astype(np.uint8), int(thr), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            area_px = cv2.contourArea(c)
            if min_area <= area_px <= max_area:
                x, y, w, h = cv2.boundingRect(c)
                dets.append(
                    dict(
                        bbox=[x, y, w, h],
                        confidence=0.65,
                        area=(w * h) / 10000.0,
                        severity=None,
                        depth=None,
                        method="Texture",
                    )
                )
        return dets

    # =================================
    # Strategy 4: post-validation rules
    # =================================
    def validate(self, det: dict, bgr: np.ndarray, depth_map: np.ndarray | None) -> bool:
        x, y, w, h = det["bbox"]
        roi = bgr[y : y + h, x : x + w]
        if roi.size == 0:
            return False

        # 1) Brightness (potholes typically darker)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if brightness > 165:  # too bright
            return False

        # 2) Texture variance (surface roughness)
        var = float(np.var(gray))
        if var < 90:  # too smooth
            return False

        # 3) Edge density (defined boundaries)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / (edges.size + 1e-6)
        if edge_ratio < 0.04:
            return False

        # 4) Depth consistency (depression)
        if depth_map is not None:
            dm = depth_map[y : y + h, x : x + w]
            if dm.size > 0:
                # pothole area should contain a local max (or min depending on scaling)
                local_contrast = float(dm.max() - dm.min())
                if local_contrast < 0.08:  # insufficient depth contrast
                    return False

        return True

    def merge_overlaps(self, dets: list, iou_thresh: float = 0.4) -> list:
        if not dets:
            return dets
        boxes = np.array([d["bbox"] for d in dets])
        confs = np.array([d.get("confidence", 0.5) for d in dets])
        idxs = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=confs.tolist(),
            score_threshold=0.01,
            nms_threshold=iou_thresh,
        )
        idxs = idxs.flatten().tolist() if len(idxs) else []
        return [dets[i] for i in idxs]

    # ==========================
    # Public detection interface
    # ==========================
    def detect(self, bgr: np.ndarray, sensitivity: float = 0.5) -> list[dict]:
        """
        sensitivity: 0..1
          Low -> conservative
          High -> aggressive
        """
        pre = self.preprocess(bgr)

        # Map sensitivity to params
        yolo_conf = 0.15 + 0.25 * sensitivity     # 0.15–0.40
        yolo_iou = 0.5                            # stable
        th_val = int(70 + 25 * sensitivity)       # 70–95
        min_area = int(1200 - 600 * (1 - sensitivity))  # 600–1200
        min_area = max(min_area, 600)
        max_area = 450000

        # Strategy 2: YOLO (specialized if present)
        yolo_dets = self._yolo_detect(pre, yolo_conf, yolo_iou)

        # Strategy 3: classical cues
        cnt_dets = self._contour_detect(pre, th_val, min_area, max_area)
        txt_dets = self._texture_detect(pre, 85 + 10 * sensitivity, min_area, max_area)

        # Merge raw
        raw = yolo_dets + cnt_dets + txt_dets
        raw = self.merge_overlaps(raw, iou_thresh=0.5)

        # Depth map for validation (optional)
        depth_map = self._get_depth_map(pre)

        # Strategy 4: validate
        valid = [d for d in raw if self.validate(d, pre, depth_map)]

        # Severity heuristic (depth + area proxy)
        for d in valid:
            # rough severity based on bbox area proxy if depth not computed later
            area_m2 = d["area"]
            # if depth map exists, assign a pseudo-depth for ranking
            if depth_map is not None:
                x, y, w, h = d["bbox"]
                dm = depth_map[y : y + h, x : x + w]
                if dm.size > 0:
                    est_depth_cm = float(dm.max() * 20.0)  # 0–20 cm
                else:
                    est_depth_cm = 6.0
            else:
                # fallback estimate from area and darkness
                est_depth_cm = 5.0 + 10.0 * min(1.0, area_m2 / 2.0)

            d["depth"] = est_depth_cm
            # severity: 1..4
            if est_depth_cm > 12 or area_m2 > 2.0:
                d["severity"] = 4
            elif est_depth_cm > 8 or area_m2 > 1.2:
                d["severity"] = 3
            elif est_depth_cm > 5 or area_m2 > 0.7:
                d["severity"] = 2
            else:
                d["severity"] = 1

        # Sort by confidence then depth
        valid.sort(key=lambda x: (x.get("confidence", 0.6), x["depth"]), reverse=True)

        # Cap output
        return valid[:20]
