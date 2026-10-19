"""
Car Detection + Segmentation + Color Web App  -  YOLOv11
=========================================================

What it does
  * Detection      : your custom YOLOv11 detector (Bus / Hatchback / SUV / Sedan / Van / Auto)
  * Segmentation   : YOLOv11-seg instance masks, matched to each detection (GrabCut fallback)
  * Color          : HSV color analysis - runs on the *mask pixels* when a mask exists,
                     so road / background / neighbouring cars no longer pollute the result
  * Video          : background jobs with progress, ByteTrack IDs -> unique vehicle counts,
                     per-vehicle thumbnails, timeline, CSV/JSON export, browser-playable H.264
  * Live           : webcam frames from the browser -> /predict_frame
  * Color Lab      : run only the color analyser on a car crop and see every internal ratio

Environment overrides (optional)
  DET_MODEL   path to the detection weights        (default: your D:/ paths, then yolo11n.pt)
  SEG_MODEL   path to the segmentation weights     (default: yolo11s-seg.pt, auto-downloads)
              If you later train a *segmentation* model on your own 6 classes and point
              SEG_MODEL at it, the app detects that and uses it alone in segment mode.
  PORT        default 5000
  FLASK_DEBUG 1 to enable debug
"""

import io
import os
import re
import csv
import json
import time
import uuid
import base64
import shutil
import zipfile
import threading
import subprocess
from collections import Counter

import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

try:
    from flask_cors import CORS
except ImportError:  # CORS is optional - the GUI is served from the same origin
    CORS = None

try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else 'cpu'
except Exception:  # pragma: no cover
    DEVICE = 'cpu'
HALF = DEVICE != 'cpu'
HALF_KW = {'half': True} if HALF else {}   # newer ultralytics warns when half is passed at all

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
if CORS:
    CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(STATIC_DIR, 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
IMAGE_EXT = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
VIDEO_EXT = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.70
DEFAULTS = {
    'conf': CONFIDENCE_THRESHOLD,  # detection confidence
    'iou': 0.60,                   # NMS IoU
    'imgsz': 640,                  # inference size
    'mode': 'segment',             # 'detect' | 'segment'
    'color': True,                 # run color analysis
    'refine': 'auto',              # 'auto' = GrabCut when the seg model has no mask for a car | 'off'
    'classes': None,               # None = all vehicle classes, else list of class names
    'show_boxes': True,
    'show_masks': True,
    'show_labels': True,
    'mask_alpha': 0.45,
    'frame_skip': 1,               # video: run the model on every Nth frame
    'track': True,                 # video: ByteTrack IDs for unique counting
}
SEG_CONF = 0.25          # seg model only proposes masks; the detector decides what is a vehicle
MATCH_IOU = 0.45         # box IoU needed to attach a seg mask to a detection
TRACKER_CFG = 'bytetrack.yaml'
DOMINANT_HUE_MIN = 0.30      # color step 1b: a non-blue hue covering >=30% of body pixels ...
DOMINANT_HUE_VS_BLACK = 1.5  # ... and 1.5x the dark-pixel share beats the black check
DOMINANT_LIGHT_MIN = 0.40    # color step 1c: white/silver covering >=40% and 1.5x black beats black
VEHICLE_NAMES = {'car', 'bus', 'truck', 'motorcycle', 'motorbike', 'van', 'auto', 'rickshaw',
                 'hatchback', 'suv', 'sedan', 'autorickshaw', 'auto-rickshaw'}

# ============================================================
# MODEL LOADING
# ============================================================
DET_CANDIDATES = [
    os.environ.get('DET_MODEL'),
    "D:/carmodel2/weights/best.pt",
    "D:\\MULTICAR\\runs3\\multicar_train\\weights\\best.pt",
]


def _pick_det_path():
    for p in DET_CANDIDATES:
        if p and os.path.exists(p):
            return p, True
    return 'yolo11n.pt', False


DET_PATH, DET_IS_CUSTOM = _pick_det_path()
print(f"🔍 Loading detector from: {DET_PATH}")
det_model = YOLO(DET_PATH)
if DET_IS_CUSTOM:
    print(f"✅ Custom detector loaded: {det_model.names}")
else:
    print("⚠️ Custom weights not found - using pretrained YOLOv11n (COCO vehicle classes only)")

SEG_PATH = os.environ.get('SEG_MODEL', 'yolo11s-seg.pt')
seg_model = None
SEG_ERROR = None
try:
    print(f"🔍 Loading segmentation model from: {SEG_PATH}")
    seg_model = YOLO(SEG_PATH)
    if seg_model.task != 'segment':
        SEG_ERROR = f"{SEG_PATH} is a '{seg_model.task}' model, not a segmentation model"
        seg_model = None
    else:
        print(f"✅ Segmentation model loaded ({len(seg_model.names)} classes)")
except Exception as e:  # no internet / bad path -> GrabCut only
    SEG_ERROR = str(e)
    seg_model = None
if SEG_ERROR:
    print(f"⚠️ Segmentation model unavailable ({SEG_ERROR}) - masks will use GrabCut")


def _lower_names(model):
    return {str(n).lower() for n in model.names.values()}


# A seg model trained on *your* classes can do detection + masks in one pass
SEG_IS_CUSTOM = bool(seg_model is not None and DET_IS_CUSTOM
                     and _lower_names(seg_model) == _lower_names(det_model))


def vehicle_ids(model, custom):
    """Class ids treated as vehicles for a model (all ids for a custom vehicle model)."""
    if custom:
        return sorted(int(k) for k in model.names)
    return sorted(int(k) for k, n in model.names.items() if str(n).lower() in VEHICLE_NAMES)


DET_VEHICLE_IDS = vehicle_ids(det_model, DET_IS_CUSTOM)
SEG_VEHICLE_IDS = vehicle_ids(seg_model, SEG_IS_CUSTOM) if seg_model is not None else []

INFER_LOCK = threading.Lock()      # ultralytics predictors are not thread-safe
VIDEO_SLOTS = threading.Semaphore(1)


def _warmup():
    dummy = np.zeros((480, 640, 3), np.uint8)
    for m in (det_model, seg_model):
        if m is None:
            continue
        try:
            m.predict(dummy, imgsz=640, verbose=False, device=DEVICE, **HALF_KW)
        except Exception as e:
            print(f"⚠️ warmup failed: {e}")


_warmup()

# ============================================================
# CLASS NAMES & COLORS
# ============================================================
class_names = {0: "Bus", 1: "Hatchback", 2: "SUV", 3: "Sedan", 4: "Van", 5: "Auto"}

html_colors = {
    'Bus': '#ef4444', 'Hatchback': '#f97316', 'SUV': '#3b82f6',
    'Sedan': '#22c55e', 'Van': '#a855f7', 'Auto': '#facc15',
    # COCO fallback names
    'car': '#22c55e', 'bus': '#ef4444', 'truck': '#3b82f6', 'motorcycle': '#facc15',
}
_EXTRA_PALETTE = ['#06b6d4', '#ec4899', '#84cc16', '#f43f5e', '#14b8a6', '#eab308', '#6366f1']

# Reference swatches for color names (used when a measured hex is not available)
COLOR_SWATCH = {
    'Black': '#151515', 'White': '#f4f4f4', 'Silver/Gray': '#9ca3af', 'Red': '#dc2626',
    'Orange': '#f97316', 'Yellow': '#facc15', 'Green': '#16a34a', 'Blue': '#2563eb',
    'Purple': '#7c3aed', 'Brown': '#7c4a1e', 'Unknown': '#525252',
}


def class_hex(name):
    if name in html_colors:
        return html_colors[name]
    return _EXTRA_PALETTE[sum(map(ord, str(name))) % len(_EXTRA_PALETTE)]


def hex_to_bgr(h):
    h = h.lstrip('#')
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def bgr_to_hex(bgr):
    b, g, r = [int(np.clip(round(float(c)), 0, 255)) for c in bgr]
    return f'#{r:02x}{g:02x}{b:02x}'


# ============================================================
# MASK HELPERS
# A mask is stored as (x0, y0, bool_array) - local to its region, so a 4K frame with
# 30 cars does not allocate 30 full-frame masks.
# ============================================================
def clip_box(box, W, H):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def expand_box(box, W, H, frac):
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * frac, (y2 - y1) * frac
    return clip_box((x1 - pw, y1 - ph, x2 + pw, y2 + ph), W, H)


def poly_to_mask(poly, region):
    x0, y0, x1, y1 = region
    if x1 <= x0 or y1 <= y0 or poly is None or len(poly) < 3:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    pts = np.round(np.asarray(poly, np.float32) - [x0, y0]).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return (x0, y0, m.astype(bool)) if m.any() else None


def mask_in_box(mask, box):
    """Return the part of a local mask that falls inside box, as a bool array of box size."""
    bx1, by1, bx2, by2 = box
    out = np.zeros((by2 - by1, bx2 - bx1), bool)
    if mask is None:
        return out
    x0, y0, m = mask
    ix1, iy1 = max(bx1, x0), max(by1, y0)
    ix2, iy2 = min(bx2, x0 + m.shape[1]), min(by2, y0 + m.shape[0])
    if ix2 > ix1 and iy2 > iy1:
        out[iy1 - by1:iy2 - by1, ix1 - bx1:ix2 - bx1] = m[iy1 - y0:iy2 - y0, ix1 - x0:ix2 - x0]
    return out


def mask_to_polygon(mask, eps_frac=0.003):
    if mask is None:
        return None
    x0, y0, m = mask
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, max(0.8, eps_frac * cv2.arcLength(c, True)), True)
    if len(c) < 3:
        return None
    return [[int(p[0][0]) + x0, int(p[0][1]) + y0] for p in c]


def largest_component(m):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lbl == k


def grabcut_mask(frame, bbox):
    """Class-agnostic foreground mask inside a box - used when the seg model misses a car."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < 16 or bh < 16:
        return None
    rx1, ry1, rx2, ry2 = expand_box(bbox, W, H, 0.12)
    roi = frame[ry1:ry2, rx1:rx2]
    scale = min(1.0, 320.0 / max(roi.shape[:2]))
    small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else roi
    rect = (max(1, int((x1 - rx1) * scale)), max(1, int((y1 - ry1) * scale)),
            max(2, int(bw * scale) - 2), max(2, int(bh * scale) - 2))
    gm = np.zeros(small.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gm, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = ((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if scale < 1:
        fg = cv2.resize(fg, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    fg = largest_component(fg)
    if fg.sum() < 0.15 * bw * bh:      # GrabCut failed - not worth keeping
        return None
    return (rx1, ry1, fg)


def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


# ============================================================
# ⭐ CAR COLOR DETECTION
# Same tuned decision order as before (blue-reflection and white-glare fixes kept),
# but it now looks at mask pixels when a segmentation mask is available.
# ============================================================
def analyze_vehicle_color(frame, bbox, mask=None):
    """
    Returns {'name', 'hex', 'coverage', 'source', 'pixels', 'debug'}.

    Strategy (order matters!):
    1. Bright saturated blue check (real blue car, not reflection)
    1b. Dominant non-blue hue (red bus with black tyres/windows is still red)
    1c. Dominant white/silver (white bus with a dark rear window is still white)
    2. Black check (wide thresholds, 22% area)
    3. White check (loose thresholds, 20% area)
    4. Silver/Gray check
    5. Colored pixels (Red, Orange, Yellow, Green, Blue, Purple, Brown)
    6. Fallback using mean V/S values
    """
    unknown = {'name': 'Unknown', 'hex': COLOR_SWATCH['Unknown'], 'coverage': 0.0,
               'source': 'none', 'pixels': 0, 'debug': {}}
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return unknown
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if ch < 20 or cw < 20:
        return unknown

    # Blur to reduce noise, reflections
    blurred = cv2.GaussianBlur(crop, (5, 5), 0)

    # Skip windshield (top 25%) and road (bottom 15%)
    top, bottom = int(ch * 0.25), int(ch * 0.85)
    band = np.zeros((ch, cw), bool)
    band[top:bottom, :] = True

    sel, source = None, 'box'
    if mask is not None:
        m = mask_in_box(mask, (x1, y1, x2, y2)).astype(np.uint8)
        k = max(3, int(min(ch, cw) * 0.04)) | 1          # erode -> drop edge / background bleed
        m = cv2.erode(m, np.ones((k, k), np.uint8)).astype(bool)
        cand = m & band
        if np.count_nonzero(cand) < max(150, 0.05 * ch * cw):
            cand = m
        if np.count_nonzero(cand) >= 150:
            sel, source = cand, 'mask'
    if sel is None:
        # No mask: skip side edges (15%) like the original rectangle method
        sel = band.copy()
        sel[:, :int(cw * 0.15)] = False
        sel[:, int(cw * 0.85):] = False
        if not sel.any():
            return unknown

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    px_hsv = hsv[sel].astype(np.int32)
    px_bgr = blurred[sel]
    h, s, v = px_hsv[:, 0], px_hsv[:, 1], px_hsv[:, 2]
    total = len(h)

    def result(name, support, coverage):
        if support is not None and np.count_nonzero(support) > 0:
            hx = bgr_to_hex(np.median(px_bgr[support], axis=0))
        else:
            hx = bgr_to_hex(np.median(px_bgr, axis=0))
        return {'name': name, 'hex': hx, 'coverage': round(float(coverage), 3),
                'source': source, 'pixels': int(total), 'debug': dbg}

    dbg = {'mean_h': round(float(np.mean(h)), 1), 'mean_s': round(float(np.mean(s)), 1),
           'mean_v': round(float(np.mean(v)), 1)}

    # STEP 1: BRIGHT SATURATED BLUE (black cars reflect *dark* blue sky)
    bright_blue = (h >= 95) & (h <= 125) & (s >= 120) & (v >= 110)
    bb = np.count_nonzero(bright_blue) / total
    dbg['bright_blue'] = round(bb, 3)

    # STEP 2..4 masks (computed up front so the Color Lab can show every ratio)
    black_mask = (v < 90) & (s < 110)
    white_mask = (s < 50) & (v >= 175)
    silver_mask = (s < 55) & (v >= 90) & (v < 175)
    dbg['black'] = round(np.count_nonzero(black_mask) / total, 3)
    dbg['white'] = round(np.count_nonzero(white_mask) / total, 3)
    dbg['silver'] = round(np.count_nonzero(silver_mask) / total, 3)

    colored = (s >= 60) & (v >= 70)
    hue_masks = {
        'Red': colored & ((h <= 10) | (h >= 170)),
        'Orange': colored & (h >= 11) & (h <= 20),
        'Yellow': colored & (h >= 21) & (h <= 33),
        'Green': colored & (h >= 34) & (h <= 85),
        'Blue': colored & (h >= 86) & (h <= 128),
        'Purple': colored & (h >= 129) & (h <= 160),
        'Brown': colored & (h >= 5) & (h <= 22) & (v < 150) & (s >= 70),
    }
    scores = {k: int(np.count_nonzero(mk)) for k, mk in hue_masks.items()}
    dbg['colored'] = {k: round(c / total, 3) for k, c in scores.items()}

    if bb > 0.20:
        return result('Blue', bright_blue, bb)

    # STEP 1b: DOMINANT SATURATED HUE. Tyres, windows and bumpers are dark, so a red / green /
    # yellow vehicle can cross the 22% black threshold. A clearly dominant non-blue hue wins first.
    # (Blue / Purple are excluded so the black-car-with-sky-reflection fix still holds.)
    top_hue, top_px = max(((k, c) for k, c in scores.items() if k not in ('Blue', 'Purple')), key=lambda x: x[1])
    top_ratio = top_px / total
    if top_ratio >= DOMINANT_HUE_MIN and top_ratio >= DOMINANT_HUE_VS_BLACK * dbg['black']:
        dbg['hue_override'] = top_hue
        return result(top_hue, hue_masks[top_hue], top_ratio)

    # STEP 1c: DOMINANT LIGHT NEUTRAL. Same idea for white / silver bodies with a dark rear window
    # or black ad panel: if light pixels clearly outnumber dark ones, don't call it Black.
    # White keeps priority over silver exactly like the original order (white > 20% -> White).
    light_ratio = dbg['white'] + dbg['silver']          # the two masks are disjoint in V
    if light_ratio >= DOMINANT_LIGHT_MIN and light_ratio >= DOMINANT_HUE_VS_BLACK * dbg['black']:
        light_name = 'White' if dbg['white'] > 0.20 else 'Silver/Gray'
        dbg['neutral_override'] = light_name
        return result(light_name, white_mask if light_name == 'White' else (white_mask | silver_mask), light_ratio)

    if dbg['black'] > 0.22:
        return result('Black', black_mask, dbg['black'])
    if dbg['white'] > 0.20:
        return result('White', white_mask, dbg['white'])
    if dbg['silver'] > 0.30:
        return result('Silver/Gray', silver_mask, dbg['silver'])

    color, pixels = max(scores.items(), key=lambda x: x[1])
    if pixels < max(15, int(total * 0.15)):
        mean_v, mean_s = float(np.mean(v)), float(np.mean(s))
        dbg['fallback'] = True
        if mean_v < 80:
            return result('Black', v < 80, np.count_nonzero(v < 80) / total)
        if mean_s < 50:
            if mean_v >= 190:
                return result('White', s < 50, np.count_nonzero(s < 50) / total)
            return result('Silver/Gray', s < 50, np.count_nonzero(s < 50) / total)
        out = result('Unknown', None, 0.0)
        return out
    return result(color, hue_masks[color], pixels / total)


def estimate_vehicle_color(frame, bbox, mask=None):
    """Backwards-compatible: returns only the color name."""
    return analyze_vehicle_color(frame, bbox, mask)['name']


# ============================================================
# CORE PIPELINE
# ============================================================
def selected_class_ids(model, custom, wanted):
    base = vehicle_ids(model, custom)
    if not wanted:
        return base
    want = {str(w).lower() for w in wanted}
    return [i for i in base if str(model.names[i]).lower() in want]


def run_segmenter(frame, o):
    kw = dict(conf=min(o['conf'], SEG_CONF), iou=0.6, imgsz=o['imgsz'], classes=SEG_VEHICLE_IDS or None,
              verbose=False, device=DEVICE, **HALF_KW)
    with INFER_LOCK:
        r = seg_model.predict(frame, **kw)[0]
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = r.boxes.xyxy.cpu().numpy()
    return [(boxes[i], r.masks.xy[i]) for i in range(len(boxes)) if len(r.masks.xy[i]) >= 3]


def analyze_frame(frame, o, tracker=None, allow_grabcut=True, poly_eps=0.003):
    """
    Detect -> (segment) -> color for one BGR frame.
    Returns (detections, timings). Each detection keeps its mask under '_mask'.
    """
    H, W = frame.shape[:2]
    timings = {}
    t0 = time.perf_counter()

    seg_primary = o['mode'] == 'segment' and SEG_IS_CUSTOM
    primary = tracker or (seg_model if seg_primary else det_model)
    custom = SEG_IS_CUSTOM if seg_primary else DET_IS_CUSTOM
    ids = selected_class_ids(primary, custom, o['classes'])
    if not ids:
        return [], {'detect_ms': 0.0}

    kw = dict(conf=o['conf'], iou=o['iou'], imgsz=o['imgsz'], classes=ids,
              verbose=False, device=DEVICE, **HALF_KW)
    with INFER_LOCK:
        if tracker is not None:
            r = tracker.track(frame, persist=True, tracker=TRACKER_CFG, **kw)[0]
        else:
            r = primary.predict(frame, **kw)[0]
    timings['detect_ms'] = (time.perf_counter() - t0) * 1000

    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets, timings
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    tids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None
    polys = r.masks.xy if getattr(r, 'masks', None) is not None else None

    for i in range(len(xyxy)):
        box = clip_box(xyxy[i], W, H)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        cls = int(clss[i])
        d = {
            'class_id': cls,
            'class_name': str(primary.names.get(cls, f'Class_{cls}')),
            'confidence': float(confs[i]),
            'bbox': list(box),
            'track_id': int(tids[i]) if tids is not None else None,
            '_mask': None,
            'mask_source': None,
        }
        if polys is not None and o['mode'] == 'segment' and len(polys[i]) >= 3:
            p = polys[i]
            region = clip_box((p[:, 0].min(), p[:, 1].min(), p[:, 0].max() + 1, p[:, 1].max() + 1), W, H)
            d['_mask'] = poly_to_mask(p, region)
            d['mask_source'] = 'model' if d['_mask'] is not None else None
        dets.append(d)

    # ---- segmentation with the separate (COCO) seg model, matched by box IoU ----
    if o['mode'] == 'segment' and not seg_primary:
        t1 = time.perf_counter()
        if seg_model is not None and dets:
            items = run_segmenter(frame, o)
            if items:
                iou = box_iou([d['bbox'] for d in dets], [it[0] for it in items])
                used_d, used_s = set(), set()
                for flat in np.argsort(-iou, axis=None):
                    di, si = np.unravel_index(flat, iou.shape)
                    if iou[di, si] < MATCH_IOU:
                        break
                    if di in used_d or si in used_s:
                        continue
                    used_d.add(di)
                    used_s.add(si)
                    region = expand_box(dets[di]['bbox'], W, H, 0.08)   # stop bleed into neighbours
                    mk = poly_to_mask(items[si][1], region)
                    if mk is not None:
                        dets[di]['_mask'] = mk
                        dets[di]['mask_source'] = 'yolo-seg'
        if allow_grabcut and o['refine'] == 'auto':
            for d in dets:
                if d['_mask'] is None:
                    d['_mask'] = grabcut_mask(frame, d['bbox'])
                    d['mask_source'] = 'grabcut' if d['_mask'] is not None else None
        timings['segment_ms'] = (time.perf_counter() - t1) * 1000

    # ---- color ----
    t2 = time.perf_counter()
    for d in dets:
        if o['color']:
            d['color'] = analyze_vehicle_color(frame, d['bbox'], d['_mask'])
            d['vehicle_color'] = d['color']['name']
        else:
            d['color'] = None
            d['vehicle_color'] = None
    if o['color']:
        timings['color_ms'] = (time.perf_counter() - t2) * 1000

    # ---- geometry ----
    for idx, d in enumerate(dets, 1):
        d['id'] = idx
        x1, y1, x2, y2 = d['bbox']
        if d['_mask'] is not None:
            d['polygon'] = mask_to_polygon(d['_mask'], poly_eps)
            d['mask_area'] = int(d['_mask'][2].sum())
        else:
            d['polygon'] = None
            d['mask_area'] = None
        d['box_area'] = int((x2 - x1) * (y2 - y1))
        d['area_ratio'] = round((d['mask_area'] or d['box_area']) / float(W * H), 5)
    return dets, timings


def public(d):
    """Detection dict without private (numpy) fields - safe for jsonify."""
    out = {k: v for k, v in d.items() if not k.startswith('_')}
    out['class_hex'] = class_hex(d['class_name'])
    return out


def compute_stats(dets, W, H):
    by_class = Counter(d['class_name'] for d in dets)
    by_color, matrix = {}, {}
    for d in dets:
        c = d.get('vehicle_color')
        if c:
            e = by_color.setdefault(c, {'count': 0, 'hex': COLOR_SWATCH.get(c, '#888888')})
            e['count'] += 1
            matrix.setdefault(d['class_name'], Counter())[c] += 1
    covered = sum(d['mask_area'] or 0 for d in dets)
    return {
        'by_class': dict(by_class),
        'by_color': by_color,
        'matrix': {k: dict(v) for k, v in matrix.items()},
        'avg_confidence': round(float(np.mean([d['confidence'] for d in dets])), 4) if dets else 0.0,
        'segmented': sum(1 for d in dets if d['_mask'] is not None),
        'mask_coverage': round(covered / float(W * H), 5),
    }


# ============================================================
# RENDERING
# ============================================================
def draw_annotations(img, dets, show_boxes=True, show_masks=True, show_labels=True, alpha=0.45):
    out = img.copy()
    H, W = out.shape[:2]
    th = max(2, int(round(min(H, W) / 360)))
    fs = max(0.45, min(H, W) / 1100)
    ft = max(1, th - 1)

    if show_masks:
        for d in dets:
            if d.get('_mask') is None:
                continue
            x0, y0, m = d['_mask']
            col = np.array(hex_to_bgr(class_hex(d['class_name'])), np.float32)
            roi = out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
            roi[m] = (roi[m].astype(np.float32) * (1 - alpha) + col * alpha).astype(np.uint8)
            if d.get('polygon'):
                cv2.polylines(out, [np.array(d['polygon'], np.int32)], True,
                              tuple(int(c) for c in col), ft, cv2.LINE_AA)

    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        col = hex_to_bgr(class_hex(d['class_name']))
        if show_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)
        if show_labels:
            tag = f"#{d['track_id']} " if d.get('track_id') is not None else ''
            label = f"{tag}{d['class_name']} {d['confidence']:.2f}"
            if d.get('vehicle_color'):
                label += f" | {d['vehicle_color']}"
            (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
            sw = tht + 4 if d.get('color') else 0              # color swatch square
            pad = 4
            ly2 = y1 if y1 - tht - 2 * pad >= 0 else y1 + tht + 2 * pad
            ly1 = ly2 - tht - 2 * pad
            lx2 = min(W - 1, x1 + tw + sw + 3 * pad)
            cv2.rectangle(out, (x1, ly1), (lx2, ly2), col, -1)
            tx = x1 + pad
            if sw:
                cv2.rectangle(out, (tx, ly1 + 3), (tx + sw - 4, ly2 - 3),
                              hex_to_bgr(d['color']['hex']), -1)
                cv2.rectangle(out, (tx, ly1 + 3), (tx + sw - 4, ly2 - 3), (255, 255, 255), 1)
                tx += sw + pad - 2
            cv2.putText(out, label, (tx, ly2 - pad - 1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), ft, cv2.LINE_AA)
    return out


INSTANCE_PALETTE = ['#f43f5e', '#22d3ee', '#a3e635', '#fbbf24', '#c084fc', '#fb923c', '#34d399',
                    '#60a5fa', '#f472b6', '#facc15', '#2dd4bf', '#a78bfa']


def instance_map(shape, dets):
    out = np.zeros((shape[0], shape[1], 3), np.uint8)
    for i, d in enumerate(dets):
        if d.get('_mask') is None:
            continue
        x0, y0, m = d['_mask']
        out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m] = hex_to_bgr(INSTANCE_PALETTE[i % len(INSTANCE_PALETTE)])
    return out


def cutouts_zip(img, dets, path):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            crop = img[y1:y2, x1:x2]
            alpha = mask_in_box(d['_mask'], d['bbox']).astype(np.uint8) * 255 \
                if d.get('_mask') is not None else np.full(crop.shape[:2], 255, np.uint8)
            bgra = np.dstack([crop, alpha])
            ok, buf = cv2.imencode('.png', bgra)
            if ok:
                name = f"{d['id']:02d}_{d['class_name']}_{(d.get('vehicle_color') or 'na').replace('/', '-')}.png"
                zf.writestr(name, buf.tobytes())


def save_thumb(img, path, width=480):
    h, w = img.shape[:2]
    s = width / float(w)
    cv2.imwrite(path, cv2.resize(img, (width, max(1, int(h * s))), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 82])


# ============================================================
# HELPERS
# ============================================================
ID_RE = re.compile(r'^[A-Za-z0-9_-]+$')


def new_id(prefix):
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def static_url(abs_path):
    rel = os.path.relpath(abs_path, STATIC_DIR).replace('\\', '/')
    return '/static/' + rel


def ext_of(filename):
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


def allowed_file(filename, kinds=None):
    kinds = kinds or (IMAGE_EXT | VIDEO_EXT)
    return ext_of(filename) in kinds


def read_image(path_or_bytes):
    """cv2 cannot read GIF - fall back to Pillow."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path_or_bytes, cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image
            src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
            img = cv2.cvtColor(np.array(Image.open(src).convert('RGB')), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


def _bool(v, default):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


def _num(v, default, lo, hi, cast=float):
    try:
        return max(lo, min(hi, cast(v)))
    except (TypeError, ValueError):
        return default


def parse_opts(src):
    src = src or {}
    o = dict(DEFAULTS)
    o['conf'] = _num(src.get('conf'), o['conf'], 0.01, 0.99)
    o['iou'] = _num(src.get('iou'), o['iou'], 0.05, 0.95)
    o['imgsz'] = int(_num(src.get('imgsz'), o['imgsz'], 160, 1920, int)) // 32 * 32
    o['mode'] = src.get('mode') if src.get('mode') in ('detect', 'segment') else o['mode']
    o['refine'] = src.get('refine') if src.get('refine') in ('auto', 'off') else o['refine']
    o['color'] = _bool(src.get('color'), o['color'])
    o['track'] = _bool(src.get('track'), o['track'])
    o['frame_skip'] = int(_num(src.get('frame_skip'), o['frame_skip'], 1, 10, int))
    o['mask_alpha'] = _num(src.get('mask_alpha'), o['mask_alpha'], 0.0, 1.0)
    for k in ('show_boxes', 'show_masks', 'show_labels'):
        o[k] = _bool(src.get(k), o[k])
    cl = src.get('classes')
    if isinstance(cl, str):
        cl = [c.strip() for c in cl.split(',') if c.strip()]
    o['classes'] = cl or None
    if o['mode'] == 'detect':
        o['show_masks'] = False
    return o


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)


def find_ffmpeg():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = find_ffmpeg()


def transcode_h264(src, dst):
    """OpenCV writes mp4v which browsers will not play - re-encode to H.264."""
    if not FFMPEG:
        return False
    cmd = [FFMPEG, '-y', '-loglevel', 'error', '-i', src,
           '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264', '-preset', 'veryfast',
           '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', dst]
    try:
        subprocess.run(cmd, check=True, timeout=3600)
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception as e:
        print(f"⚠️ ffmpeg transcode failed: {e}")
        return False


def models_info():
    return {
        'detector': {'path': DET_PATH, 'custom': DET_IS_CUSTOM, 'task': det_model.task,
                     'classes': {int(k): str(v) for k, v in det_model.names.items()},
                     'vehicle_classes': [str(det_model.names[i]) for i in DET_VEHICLE_IDS]},
        'segmenter': {'path': SEG_PATH, 'available': seg_model is not None, 'error': SEG_ERROR,
                      'custom': SEG_IS_CUSTOM,
                      'vehicle_classes': [str(seg_model.names[i]) for i in SEG_VEHICLE_IDS] if seg_model else []},
        'device': 'cuda:0' if DEVICE != 'cpu' else 'cpu',
        'half': HALF,
        'ffmpeg': bool(FFMPEG),
    }


# ============================================================
# ROUTES - pages & info
# ============================================================
@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204  # icon is inlined in the page


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': det_model is not None,
        'segmentation_loaded': seg_model is not None,
        'model_classes': list(det_model.names.values()),
        'class_count': len(det_model.names),
        'model_path': DET_PATH,
        'device': models_info()['device'],
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'active_jobs': sum(1 for j in JOBS.values() if j['status'] in ('queued', 'processing', 'encoding')),
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'classes': det_model.names,
        'class_count': len(det_model.names),
        'model_type': 'YOLOv11',
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        **models_info(),
    })


@app.route('/config', methods=['GET'])
def config():
    names = [str(det_model.names[i]) for i in DET_VEHICLE_IDS]
    return jsonify({
        'defaults': DEFAULTS,
        'classes': [{'name': n, 'hex': class_hex(n)} for n in names],
        'color_swatches': COLOR_SWATCH,
        'models': models_info(),
        'limits': {'max_upload_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
                   'image_ext': sorted(IMAGE_EXT), 'video_ext': sorted(VIDEO_EXT)},
        'color_rules': {
            'bright_blue': 'H 95-125, S>=120, V>=110  -> Blue if >20%',
            'dominant_hue': 'non-blue hue >=30% and >=1.5x black share -> that hue',
            'dominant_light': 'white+silver >=40% and >=1.5x black share -> White (if white>20%) else Silver',
            'black': 'V<90, S<110  -> Black if >22%',
            'white': 'S<50, V>=175 -> White if >20%',
            'silver': 'S<55, 90<=V<175 -> Silver/Gray if >30%',
            'colored': 'S>=60, V>=70 -> hue bins, winner needs >=15%',
        },
    })


# ============================================================
# ROUTES - image
# ============================================================
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        file = request.files.get('image')
        if file is None:
            return jsonify({'error': 'No image uploaded'}), 400
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        if not allowed_file(file.filename, IMAGE_EXT):
            return jsonify({'error': f'File type not allowed (use {", ".join(sorted(IMAGE_EXT))})'}), 400

        o = parse_opts(request.form)
        rid = new_id('img')
        out_dir = os.path.join(OUTPUT_FOLDER, rid)
        os.makedirs(out_dir, exist_ok=True)
        filename = secure_filename(file.filename) or f'image.{ext_of(file.filename) or "jpg"}'
        input_path = os.path.join(UPLOAD_FOLDER, f'{rid}_{filename}')
        file.save(input_path)

        img = read_image(input_path)
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
        H, W = img.shape[:2]

        t0 = time.perf_counter()
        dets, timings = analyze_frame(img, o)

        annotated = draw_annotations(img, dets, o['show_boxes'], o['show_masks'], o['show_labels'], o['mask_alpha'])
        out_path = os.path.join(out_dir, 'annotated.jpg')
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        save_thumb(annotated, os.path.join(out_dir, 'thumb.jpg'))

        mask_url = None
        if any(d['_mask'] is not None for d in dets):
            mpath = os.path.join(out_dir, 'mask.png')
            cv2.imwrite(mpath, instance_map(img.shape, dets))
            mask_url = static_url(mpath)

        cutouts_url = None
        if dets:
            zpath = os.path.join(out_dir, 'cutouts.zip')
            cutouts_zip(img, dets, zpath)
            cutouts_url = static_url(zpath)
        timings['total_ms'] = (time.perf_counter() - t0) * 1000

        report = {
            'success': True,
            'kind': 'image',
            'id': rid,
            'created': time.time(),
            'filename': file.filename,
            'width': W, 'height': H,
            'count': len(dets),
            'detections': [public(d) for d in dets],
            'stats': compute_stats(dets, W, H),
            'timings': {k: round(v, 1) for k, v in timings.items()},
            'settings': o,
            'threshold': o['conf'],
            'input_url': static_url(input_path),
            'output_url': static_url(out_path),
            'mask_url': mask_url,
            'cutouts_url': cutouts_url,
            'thumb_url': static_url(os.path.join(out_dir, 'thumb.jpg')),
            'json_url': static_url(os.path.join(out_dir, 'report.json')),
        }
        write_json(os.path.join(out_dir, 'report.json'), report)
        return jsonify(report)

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# ROUTES - live webcam frames
# ============================================================
@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        if request.files.get('frame'):
            raw = request.files['frame'].read()
            o = parse_opts(request.form)
        else:
            body = request.get_json(silent=True) or {}
            data = body.get('image', '')
            if ',' in data:
                data = data.split(',', 1)[1]
            raw = base64.b64decode(data) if data else b''
            o = parse_opts(body)
        img = read_image(raw) if raw else None
        if img is None:
            return jsonify({'error': 'No frame / undecodable frame'}), 400
        H, W = img.shape[:2]
        t0 = time.perf_counter()
        # GrabCut is too slow for live video - YOLO-seg masks only
        dets, timings = analyze_frame(img, o, allow_grabcut=False, poly_eps=0.006)
        timings['total_ms'] = (time.perf_counter() - t0) * 1000
        return jsonify({
            'success': True, 'width': W, 'height': H, 'count': len(dets),
            'detections': [public(d) for d in dets],
            'stats': compute_stats(dets, W, H),
            'timings': {k: round(v, 1) for k, v in timings.items()},
        })
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# ROUTES - color lab (color analyser only)
# ============================================================
@app.route('/analyze_color', methods=['POST'])
def analyze_color():
    """
    Run only the color analyser. Upload a car photo / crop.
    use_mask=true -> segment the largest vehicle first and analyse its mask pixels.
    Otherwise the whole image is treated as the vehicle's bounding box.
    """
    try:
        file = request.files.get('image')
        if file is None or file.filename == '':
            return jsonify({'error': 'No image uploaded'}), 400
        img = read_image(file.read())
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
        H, W = img.shape[:2]
        use_mask = _bool(request.form.get('use_mask'), True)
        bbox, mask, src = (0, 0, W, H), None, 'whole image'
        if use_mask:
            if seg_model is not None:
                items = run_segmenter(img, {**DEFAULTS, 'conf': SEG_CONF})
                if items:
                    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b, _ in items]
                    b, poly = items[int(np.argmax(areas))]
                    bbox = clip_box(b, W, H)
                    mask = poly_to_mask(poly, expand_box(bbox, W, H, 0.05))
                    src = 'yolo-seg'
            if mask is None:
                bb = (int(W * 0.04), int(H * 0.04), int(W * 0.96), int(H * 0.96))
                mask = grabcut_mask(img, bb)
                if mask is not None:
                    bbox, src = bb, 'grabcut'
        t0 = time.perf_counter()
        col = analyze_vehicle_color(img, bbox, mask)
        return jsonify({
            'success': True, 'width': W, 'height': H, 'bbox': list(bbox),
            'mask_source': src, 'polygon': mask_to_polygon(mask),
            'color': col, 'swatch': COLOR_SWATCH.get(col['name']),
            'ms': round((time.perf_counter() - t0) * 1000, 1),
        })
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# ROUTES - video (background jobs)
# ============================================================
JOBS = {}


def _vote(counter):
    return counter.most_common(1)[0][0] if counter else None


def _video_worker(job_id, input_path, o):
    job = JOBS[job_id]
    out_dir = job['out_dir']
    with VIDEO_SLOTS:
        if job['cancel']:
            job['status'] = 'cancelled'
            return
        job['status'] = 'processing'
        job['started'] = time.time()
        try:
            _process_video(job, input_path, out_dir, o)
        except Exception as e:
            print(f"❌ Video job {job_id} failed: {e}")
            job['status'] = 'error'
            job['error'] = str(e)


def _process_video(job, input_path, out_dir, o):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError('Could not open video')
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    fps = fps if 1 <= fps <= 240 else 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    job['total_frames'] = total
    job['fps'] = fps

    # Fresh model instance per job: tracker state lives on the model object
    tracker = None
    if o['track']:
        try:
            path = SEG_PATH if (o['mode'] == 'segment' and SEG_IS_CUSTOM) else DET_PATH
            tracker = YOLO(path)
        except Exception as e:
            job['warnings'].append(f'Tracking disabled: {e}')

    raw_path = os.path.join(out_dir, 'raw.mp4')
    writer = None
    tracks = {}
    grouped = {}
    timeline = []
    best_frame, best_count = None, -1
    total_dets = 0
    last = []
    idx = 0
    t_start = time.time()

    while True:
        if job['cancel']:
            break
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
            job['width'], job['height'] = W, H

        processed = idx % o['frame_skip'] == 0
        if processed:
            try:
                last, _ = analyze_frame(frame, o, tracker=tracker, allow_grabcut=False)
            except Exception as e:
                if tracker is not None:   # e.g. 'lap' missing -> continue untracked
                    job['warnings'].append(f'Tracking disabled: {e}')
                    tracker = None
                    last, _ = analyze_frame(frame, o, allow_grabcut=False)
                else:
                    raise
            dets = last
            total_dets += len(dets)
            tl = {'f': idx, 't': round(idx / fps, 3), 'n': len(dets)}
            if dets:
                tl['c'] = dict(Counter(d['class_name'] for d in dets))
            timeline.append(tl)

            for d in dets:
                key = (d['class_name'], d.get('vehicle_color') or 'n/a')
                g = grouped.setdefault(key, {'class_name': key[0], 'vehicle_color': d.get('vehicle_color'),
                                             'count': 0, 'first_frame': idx, 'last_frame': idx,
                                             'max_confidence': 0.0})
                g['count'] += 1
                g['last_frame'] = idx
                g['max_confidence'] = max(g['max_confidence'], d['confidence'])

                tid = d.get('track_id')
                if tid is None:
                    continue
                t = tracks.setdefault(tid, {'track_id': tid, 'classes': Counter(), 'colors': Counter(),
                                            'hex': {}, 'first_frame': idx, 'last_frame': idx,
                                            'frames': 0, 'max_confidence': 0.0, '_thumb': None})
                t['classes'][d['class_name']] += 1
                if d.get('color'):
                    t['colors'][d['color']['name']] += 1
                    t['hex'][d['color']['name']] = d['color']['hex']
                t['frames'] += 1
                t['last_frame'] = idx
                if d['confidence'] > t['max_confidence']:
                    t['max_confidence'] = d['confidence']
                    x1, y1, x2, y2 = expand_box(d['bbox'], W, H, 0.06)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        s = 160.0 / max(crop.shape[:2])
                        t['_thumb'] = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop.copy()
                        t['best_frame'] = idx
        else:
            dets = last  # reuse last result on skipped frames

        annotated = draw_annotations(frame, dets, o['show_boxes'], o['show_masks'], o['show_labels'], o['mask_alpha'])
        writer.write(annotated)
        if processed and len(dets) > best_count:
            best_count, best_frame = len(dets), annotated

        idx += 1
        el = time.time() - t_start
        job['frame'] = idx
        job['proc_fps'] = round(idx / el, 2) if el > 0 else 0
        if total and job['proc_fps']:
            job['progress'] = min(0.99, idx / total)
            job['eta_s'] = round((total - idx) / job['proc_fps'], 1)

    cap.release()
    if writer is not None:
        writer.release()
    if writer is None:
        raise RuntimeError('Video has no readable frames')

    job['status'] = 'encoding'
    final_path = os.path.join(out_dir, 'annotated.mp4')
    playable = transcode_h264(raw_path, final_path)
    if playable:
        os.remove(raw_path)
    else:
        os.replace(raw_path, final_path)
        job['warnings'].append('ffmpeg not found - video is mp4v and may not play in the browser '
                               '(download it, or `pip install imageio-ffmpeg`)')

    if best_frame is not None:
        save_thumb(best_frame, os.path.join(out_dir, 'thumb.jpg'))

    # unique vehicles from tracks (drop flicker tracks seen < 3 processed frames)
    min_frames = min(3, max(1, len(timeline)))
    track_list = []
    os.makedirs(os.path.join(out_dir, 'tracks'), exist_ok=True)
    for tid, t in sorted(tracks.items()):
        if t['frames'] < min_frames:
            continue
        cname = _vote(t['colors'])
        thumb_url = None
        if t['_thumb'] is not None:
            p = os.path.join(out_dir, 'tracks', f'{tid}.jpg')
            cv2.imwrite(p, t['_thumb'], [cv2.IMWRITE_JPEG_QUALITY, 85])
            thumb_url = static_url(p)
        track_list.append({
            'track_id': tid,
            'class_name': _vote(t['classes']),
            'class_hex': class_hex(_vote(t['classes'])),
            'vehicle_color': cname,
            'color_hex': t['hex'].get(cname) if cname else None,
            'class_votes': dict(t['classes']),
            'color_votes': dict(t['colors']),
            'first_frame': t['first_frame'], 'last_frame': t['last_frame'],
            'first_time': round(t['first_frame'] / fps, 2), 'last_time': round(t['last_frame'] / fps, 2),
            'frames_seen': t['frames'],
            'max_confidence': round(t['max_confidence'], 4),
            'thumb_url': thumb_url,
        })

    csv_path = os.path.join(out_dir, 'vehicles.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if track_list:
            w.writerow(['track_id', 'class', 'color', 'color_hex', 'first_frame', 'last_frame',
                        'first_time_s', 'last_time_s', 'frames_seen', 'max_confidence'])
            for t in track_list:
                w.writerow([t['track_id'], t['class_name'], t['vehicle_color'], t['color_hex'], t['first_frame'],
                            t['last_frame'], t['first_time'], t['last_time'], t['frames_seen'], t['max_confidence']])
        else:
            w.writerow(['class', 'color', 'detections', 'first_frame', 'last_frame', 'max_confidence'])
            for g in grouped.values():
                w.writerow([g['class_name'], g['vehicle_color'], g['count'], g['first_frame'],
                            g['last_frame'], round(g['max_confidence'], 4)])

    # timeline: cap payload size
    step = max(1, len(timeline) // 1500)
    if track_list:   # unique vehicles
        by_class = dict(Counter(t['class_name'] for t in track_list))
    else:            # untracked: per-class detection totals
        by_class = dict(sum((Counter({g['class_name']: g['count']}) for g in grouped.values()), Counter()))
    by_color = {}
    for t in track_list:
        if t['vehicle_color']:
            e = by_color.setdefault(t['vehicle_color'], {'count': 0, 'hex': COLOR_SWATCH.get(t['vehicle_color'])})
            e['count'] += 1

    report = {
        'success': True,
        'kind': 'video',
        'id': job['id'],
        'created': time.time(),
        'filename': job['filename'],
        'cancelled': bool(job['cancel']),
        'width': job.get('width'), 'height': job.get('height'),
        'fps': round(fps, 3),
        'total_frames': total or idx,
        'processed_frames': len(timeline),
        'frames_read': idx,
        'duration_s': round(idx / fps, 2),
        'processing_s': round(time.time() - t_start, 1),
        'tracking': tracker is not None,
        'unique_vehicles': len(track_list),
        'total_detections': total_dets,
        'peak_count': max([t['n'] for t in timeline], default=0),
        'tracks': track_list,
        'by_class': by_class,
        'by_color': by_color,
        'vehicle_summary': [{**g, 'max_confidence': round(g['max_confidence'], 4)} for g in grouped.values()],
        'timeline': timeline[::step],
        'settings': o,
        'threshold': o['conf'],
        'warnings': job['warnings'],
        'browser_playable': playable,
        'input_url': static_url(input_path),
        'output_url': static_url(final_path),
        'csv_url': static_url(csv_path),
        'thumb_url': static_url(os.path.join(out_dir, 'thumb.jpg')) if best_frame is not None else None,
        'json_url': static_url(os.path.join(out_dir, 'report.json')),
    }
    write_json(os.path.join(out_dir, 'report.json'), report)
    job['result'] = report
    job['progress'] = 1.0
    job['status'] = 'cancelled' if job['cancel'] else 'done'


@app.route('/predict_video', methods=['POST'])
def predict_video():
    """
    Starts a background job and returns {job_id} (HTTP 202).
    Poll GET /jobs/<job_id>. Send form field sync=1 to block and get the full result instead.
    """
    try:
        file = request.files.get('video')
        if file is None:
            return jsonify({'error': 'No video uploaded'}), 400
        if file.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        if not allowed_file(file.filename, VIDEO_EXT):
            return jsonify({'error': f'File type not allowed (use {", ".join(sorted(VIDEO_EXT))})'}), 400

        o = parse_opts(request.form)
        rid = new_id('vid')
        out_dir = os.path.join(OUTPUT_FOLDER, rid)
        os.makedirs(out_dir, exist_ok=True)
        filename = secure_filename(file.filename) or f'video.{ext_of(file.filename)}'
        input_path = os.path.join(UPLOAD_FOLDER, f'{rid}_{filename}')
        file.save(input_path)

        JOBS[rid] = {'id': rid, 'filename': file.filename, 'status': 'queued', 'progress': 0.0,
                     'frame': 0, 'total_frames': 0, 'proc_fps': 0, 'eta_s': None, 'cancel': False,
                     'warnings': [], 'error': None, 'result': None, 'out_dir': out_dir,
                     'created': time.time(), 'settings': o}
        if _bool(request.form.get('sync'), False):
            _video_worker(rid, input_path, o)
            job = JOBS[rid]
            if job['status'] == 'error':
                return jsonify({'error': job['error']}), 500
            return jsonify(job['result'])
        threading.Thread(target=_video_worker, args=(rid, input_path, o), daemon=True).start()
        return jsonify({'success': True, 'job_id': rid, 'status_url': f'/jobs/{rid}'}), 202

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


def _job_view(job):
    return {k: v for k, v in job.items() if k not in ('out_dir', 'cancel')} | {'cancel_requested': job['cancel']}


@app.route('/jobs', methods=['GET'])
def list_jobs():
    return jsonify([{k: v for k, v in _job_view(j).items() if k != 'result'}
                    for j in sorted(JOBS.values(), key=lambda j: -j['created'])])


@app.route('/jobs/<job_id>', methods=['GET'])
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job'}), 404
    return jsonify(_job_view(job))


@app.route('/jobs/<job_id>/cancel', methods=['POST'])
def job_cancel(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job'}), 404
    job['cancel'] = True
    return jsonify({'success': True})


# ============================================================
# ROUTES - history
# ============================================================
@app.route('/history', methods=['GET'])
def history():
    items = []
    for name in os.listdir(OUTPUT_FOLDER):
        rp = os.path.join(OUTPUT_FOLDER, name, 'report.json')
        if not os.path.isfile(rp):
            continue
        try:
            with open(rp, encoding='utf-8') as f:
                r = json.load(f)
        except Exception:
            continue
        items.append({
            'id': r.get('id'), 'kind': r.get('kind'), 'created': r.get('created'),
            'filename': r.get('filename'), 'thumb_url': r.get('thumb_url'), 'json_url': r.get('json_url'),
            'count': r.get('count') if r.get('kind') == 'image' else r.get('unique_vehicles'),
            'by_class': r.get('stats', {}).get('by_class') if r.get('kind') == 'image' else r.get('by_class'),
            'mode': (r.get('settings') or {}).get('mode'),
        })
    items.sort(key=lambda x: -(x['created'] or 0))
    return jsonify(items[:200])


def _delete_result(rid):
    shutil.rmtree(os.path.join(OUTPUT_FOLDER, rid), ignore_errors=True)
    for fn in os.listdir(UPLOAD_FOLDER):
        if fn.startswith(rid + '_'):
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, fn))
            except OSError:
                pass
    JOBS.pop(rid, None)


@app.route('/history/<rid>', methods=['DELETE'])
def history_delete(rid):
    if not ID_RE.match(rid):
        return jsonify({'error': 'Bad id'}), 400
    job = JOBS.get(rid)
    if job and job['status'] in ('queued', 'processing', 'encoding'):
        return jsonify({'error': 'Job still running - cancel it first'}), 409
    _delete_result(rid)
    return jsonify({'success': True})


@app.route('/history', methods=['DELETE'])
def history_clear():
    running = {j['id'] for j in JOBS.values() if j['status'] in ('queued', 'processing', 'encoding')}
    n = 0
    for name in os.listdir(OUTPUT_FOLDER):
        if name not in running and ID_RE.match(name):
            _delete_result(name)
            n += 1
    return jsonify({'success': True, 'deleted': n})


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': f'File too large (max {app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)} MB)'}), 413


if __name__ == '__main__':
    info = models_info()
    print("\n" + "=" * 64)
    print("🚗 Car Detection + Segmentation + Color - YOLOv11")
    print("=" * 64)
    print(f"📊 Detector     : {DET_PATH}  ({'custom' if DET_IS_CUSTOM else 'COCO fallback'})")
    print(f"   Classes      : {info['detector']['vehicle_classes']}")
    print(f"🧩 Segmentation : {SEG_PATH if seg_model else 'GrabCut fallback only'}"
          f"{'  (custom classes - single pass)' if SEG_IS_CUSTOM else ''}")
    print(f"⚙️  Device       : {info['device']}   ffmpeg: {'yes' if FFMPEG else 'NO'}")
    print(f"🎯 Confidence   : {CONFIDENCE_THRESHOLD * 100:.0f}% (adjustable in the GUI)")
    port = int(os.environ.get('PORT', 5000))
    print(f"📍 http://127.0.0.1:{port}")
    print("=" * 64)
    app.run(debug=_bool(os.environ.get('FLASK_DEBUG'), False), host='0.0.0.0', port=port,
            threaded=True, use_reloader=False)
