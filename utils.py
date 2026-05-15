import cv2
import numpy as np
import matplotlib.pyplot as plt
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together, visualize_sample
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

from huggingface_hub import login
import os

hf_token = os.environ["HF_TOKEN"]
login(token=hf_token)

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

skeleton_visualizer = SkeletonVisualizer(line_width=2, radius=5)
skeleton_visualizer.set_pose_meta(mhr70_pose_info)

# Load estimator
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

def process_image(img_path:str):
    """
    Simple processing of an image.
    
    Args:
    img_path:

    Returns:
    rendered image
    """
    img_bgr = cv2.imread(img_path)
    outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    # Visualize and save results
    rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
    cv2.imwrite("output.jpg", rend_img.astype(np.uint8))
    
    return rend_img

def save_obj(outputs, faces, output_path="output.obj"):
    """
    create a 3D object file of the player's mesh

    Args:
    outputs
    faces
    output_path

    Returns:
    3D object file
    """
    all_vertices = []
    for pid, person_output in enumerate(outputs):
        verts = person_output["pred_vertices"] + person_output["pred_cam_t"]
        all_vertices.append(verts)
    all_vertices = np.concatenate(all_vertices, axis=0)
    all_vertices[:, 1] *= -1  # flip Y
    all_vertices[:, 2] *= -1  # flip Z

    with open(output_path, "w") as f:
        for v in all_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # .obj faces are 1-indexed
        for pid in range(len(outputs)):
            offset = len(person_output["pred_vertices"]) * pid
            for face in faces:
                f1, f2, f3 = face + offset + 1
                f.write(f"f {f1} {f2} {f3}\n")

    return output_path

def process_video(video_path: str, output_path: str = "output_video.mp4"):
    """
    Read a video frame by frame, run the SAM 3D body estimator on each frame,
    and write the visualised results to a new video file.

    Args:
        video_path:  Path to the input video file.
        output_path: Path for the output video (default: 'output_video.mp4').
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input  : {video_path}")
    print(f"Frames : {total}  |  FPS : {fps:.2f}  |  Size : {width}x{height}")

    # We don't know the output frame size until we process the first frame,
    # so defer writer creation until after the first successful render.
    writer = None
    out_w, out_h = None, None

    try:
        for frame_idx in range(total if total > 0 else int(1e9)):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # --- run estimator ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            outputs   = estimator.process_one_image(frame_rgb)

            # --- render overlay ---
            rend = visualize_sample_together(frame_bgr, outputs, estimator.faces)
            rend_bgr = rend.astype(np.uint8)   # already BGR from visualize_sample_together

            # --- init writer on first frame ---
            if writer is None:
                out_h, out_w = rend_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
                print(f"Output : {output_path}  |  Size : {out_w}x{out_h}")

            writer.write(rend_bgr)

            if (frame_idx + 1) % 10 == 0:
                print(f"  processed {frame_idx + 1}/{total} frames…")

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Done – saved to {output_path}")
    return output_path

# ── MHR70 joint indices (Multi-HMR / SMPL-X convention) ─────────────────────
# ── MHR70 joint indices ──────────────────────────────────────────────────────
J = {
    "pelvis":       0,
    "l_hip":        1,   "r_hip":        2,
    "spine1":       3,
    "l_knee":       4,   "r_knee":        5,
    "spine2":       6,
    "l_ankle":      7,   "r_ankle":       8,
    "spine3":       9,
    "l_foot":      10,   "r_foot":       11,
    "neck":        12,
    "l_collar":    13,   "r_collar":     14,
    "head":        15,
    "l_shoulder":  16,   "r_shoulder":   17,
    "l_elbow":     18,   "r_elbow":      19,
    "l_wrist":     20,   "r_wrist":      21,
}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def angle_between(a, b, c):
    """Angle at joint b formed by segments b→a and b→c, in degrees."""
    v1 = a - b; v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def valgus_angle(hip, knee, ankle):
    """Medial knee deviation from hip-ankle line in the frontal plane (XY)."""
    ref2 = (ankle - hip)[:2]; dev2 = (knee - hip)[:2]
    if np.linalg.norm(ref2) < 1e-6 or np.linalg.norm(dev2) < 1e-6:
        return 0.0
    cos_a = np.dot(ref2, dev2) / (np.linalg.norm(ref2) * np.linalg.norm(dev2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def vec_angle_from_vertical(vec, use_3d=True):
    """Angle of a vector from the vertical Y axis, in degrees."""
    vertical = np.array([0., 1., 0.]) if use_3d else np.array([0., 1.])
    v = vec[:3] if use_3d else vec[:2]
    cos_a = np.dot(v, vertical) / (np.linalg.norm(v) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def asymmetry_index(left, right):
    """Standard asymmetry index as a percentage."""
    mean = (abs(left) + abs(right)) / 2
    return float(abs(left - right) / (mean + 1e-8) * 100)


# ── Temporal buffer ───────────────────────────────────────────────────────────

class TemporalBuffer:
    """
    Maintains a rolling window of per-frame metric dicts.
    Pass the same instance across all frames of a video.
    For single-image use, pass None to compute_biomechanics.
    """
    def __init__(self, fps=30.0, window=10):
        self.fps    = fps
        self.window = window
        self.frames = []   # list of metric dicts

    def push(self, metrics):
        self.frames.append(metrics)
        if len(self.frames) > self.window:
            self.frames.pop(0)

    def last(self, key, n=2):
        vals = [f[key] for f in self.frames[-n:] if key in f]
        return vals

    def delta(self, key):
        vals = self.last(key, 2)
        return (vals[-1] - vals[0]) if len(vals) == 2 else 0.0

    def range_of_motion(self, key):
        vals = [f[key] for f in self.frames if key in f]
        return (max(vals) - min(vals)) if vals else 0.0


# ── Main biomechanics function ────────────────────────────────────────────────

def compute_biomechanics(person_output, buf=None, fps=30.0):
    """
    Compute all biomechanical metrics for a single person.

    Args:
        person_output : dict from estimator.process_one_image()
        buf           : TemporalBuffer instance (None for single-image)
        fps           : video frame rate, used for velocity calculations

    Returns a flat dict with all metrics.
    """
    kp      = person_output.get("pred_keypoints_3d", person_output.get("pred_keypoints_2d"))
    use_3d  = "pred_keypoints_3d" in person_output
    verts   = person_output.get("pred_vertices", None)

    def pt(name):
        return kp[J[name]]

    # ── Centre of mass ────────────────────────────────────────────────────────
    com = np.mean(verts, axis=0) if verts is not None else pt("pelvis")

    # ── Lower body ───────────────────────────────────────────────────────────
    l_knee_deg = angle_between(pt("l_hip"),  pt("l_knee"),  pt("l_ankle"))
    r_knee_deg = angle_between(pt("r_hip"),  pt("r_knee"),  pt("r_ankle"))
    l_hip_deg  = angle_between(pt("spine1"), pt("l_hip"),   pt("l_knee"))
    r_hip_deg  = angle_between(pt("spine1"), pt("r_hip"),   pt("r_knee"))
    l_ankle_deg = angle_between(pt("l_knee"), pt("l_ankle"), pt("l_foot"))
    r_ankle_deg = angle_between(pt("r_knee"), pt("r_ankle"), pt("r_foot"))
    l_val = valgus_angle(pt("l_hip"), pt("l_knee"), pt("l_ankle"))
    r_val = valgus_angle(pt("r_hip"), pt("r_knee"), pt("r_ankle"))

    knee_w = np.linalg.norm(pt("l_knee")[:2] - pt("r_knee")[:2])
    hip_w  = np.linalg.norm(pt("l_hip")[:2]  - pt("r_hip")[:2])
    kwr    = float(knee_w / (hip_w + 1e-8))
    hip_drop = float((pt("l_hip")[1] - pt("r_hip")[1]) * 100)

    # Stride: horizontal ankle separation (proxy for stride length)
    stride_len = float(np.linalg.norm((pt("l_ankle") - pt("r_ankle"))[:2]))

    # Ground contact: foot closest to ground (min Y)
    l_foot_h = float(pt("l_foot")[1])
    r_foot_h = float(pt("r_foot")[1])
    ground_contact = "left" if l_foot_h <= r_foot_h else "right"

    # ── Trunk / posture ──────────────────────────────────────────────────────
    trunk_vec   = pt("neck") - pt("pelvis")
    trunk_lean  = vec_angle_from_vertical(trunk_vec, use_3d)

    body_vec    = pt("head") - pt("l_ankle") * 0.5 - pt("r_ankle") * 0.5
    forward_lean = vec_angle_from_vertical(body_vec, use_3d)

    head_vec    = pt("head") - pt("neck")
    head_tilt   = vec_angle_from_vertical(head_vec, use_3d)

    # ── Upper body ───────────────────────────────────────────────────────────
    l_elbow_deg = angle_between(pt("l_shoulder"), pt("l_elbow"), pt("l_wrist"))
    r_elbow_deg = angle_between(pt("r_shoulder"), pt("r_elbow"), pt("r_wrist"))

    # Shoulder rotation: angle of shoulder line vs hip line in the transverse plane
    shoulder_vec = (pt("l_shoulder") - pt("r_shoulder"))[:2]
    hip_vec_2d   = (pt("l_hip")      - pt("r_hip"))[:2]
    cos_sh = np.dot(shoulder_vec, hip_vec_2d) / (
        np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec_2d) + 1e-8)
    shoulder_rot = float(np.degrees(np.arccos(np.clip(cos_sh, -1, 1))))

    # ── Asymmetry indices ────────────────────────────────────────────────────
    asym_knee   = asymmetry_index(l_knee_deg, r_knee_deg)
    asym_hip    = asymmetry_index(l_hip_deg,  r_hip_deg)
    asym_ankle  = asymmetry_index(l_ankle_deg, r_ankle_deg)
    asym_elbow  = asymmetry_index(l_elbow_deg, r_elbow_deg)
    asym_valgus = asymmetry_index(l_val, r_val)

    # ── Risk scores ──────────────────────────────────────────────────────────
    # Dynamic Valgus Score (0-10): combines knee valgus + hip drop + trunk lean
    dvs = min(10.0, (
        max(l_val, r_val) / 3.0 +
        abs(hip_drop) / 2.0 +
        trunk_lean / 10.0
    ))

    # LESS proxy: knee flexion < 30deg at near-ground-contact + valgus > 10deg
    near_ground  = min(l_foot_h, r_foot_h) < 0.05
    less_flag    = near_ground and (
        (l_knee_deg < 30 or r_knee_deg < 30) or
        (l_val > 10 or r_val > 10)
    )

    low_confidence = (l_knee_deg > 170 and r_knee_deg > 170)

    # ── Build base metrics dict ───────────────────────────────────────────────
    m = {
        # Lower body
        "l_knee_deg":    l_knee_deg,
        "r_knee_deg":    r_knee_deg,
        "l_hip_deg":     l_hip_deg,
        "r_hip_deg":     r_hip_deg,
        "l_ankle_deg":   l_ankle_deg,
        "r_ankle_deg":   r_ankle_deg,
        "l_valgus":      l_val,
        "r_valgus":      r_val,
        "kwr":           kwr,
        "hip_drop_cm":   hip_drop,
        "stride_len":    stride_len,
        "ground_contact": ground_contact,
        # Trunk / posture
        "trunk_lean":    trunk_lean,
        "forward_lean":  forward_lean,
        "head_tilt":     head_tilt,
        # Upper body
        "l_elbow_deg":   l_elbow_deg,
        "r_elbow_deg":   r_elbow_deg,
        "shoulder_rot":  shoulder_rot,
        # Asymmetry
        "asym_knee":     asym_knee,
        "asym_hip":      asym_hip,
        "asym_ankle":    asym_ankle,
        "asym_elbow":    asym_elbow,
        "asym_valgus":   asym_valgus,
        # Risk
        "dvs":           dvs,
        "less_flag":     less_flag,
        # CoM
        "com":           com,
        # Meta
        "low_confidence": low_confidence,
        # Temporal (defaults — overwritten below if buffer available)
        "knee_angular_vel_l": 0.0,
        "knee_angular_vel_r": 0.0,
        "hip_angular_vel_l":  0.0,
        "hip_angular_vel_r":  0.0,
        "com_sway_lateral":   0.0,
        "com_accel":          0.0,
        "cadence":            0.0,
        "rom_knee_l":         0.0,
        "rom_knee_r":         0.0,
        "rom_hip_l":          0.0,
        "rom_hip_r":          0.0,
    }

    # ── Temporal metrics (only when buffer is provided) ───────────────────────
    if buf is not None:
        buf.push(m)  # push BEFORE reading so we have at least 1 frame

        dt = 1.0 / fps

        # Angular velocities (deg/s)
        m["knee_angular_vel_l"] = abs(buf.delta("l_knee_deg")) / dt if len(buf.frames) >= 2 else 0.0
        m["knee_angular_vel_r"] = abs(buf.delta("r_knee_deg")) / dt if len(buf.frames) >= 2 else 0.0
        m["hip_angular_vel_l"]  = abs(buf.delta("l_hip_deg"))  / dt if len(buf.frames) >= 2 else 0.0
        m["hip_angular_vel_r"]  = abs(buf.delta("r_hip_deg"))  / dt if len(buf.frames) >= 2 else 0.0

        # CoM lateral sway (units match keypoint space)
        prev_com_vals = buf.last("com", 2)
        if len(prev_com_vals) == 2:
            com_delta = prev_com_vals[-1] - prev_com_vals[0]
            m["com_sway_lateral"] = float(abs(com_delta[0]))
            m["com_accel"]        = float(np.linalg.norm(com_delta) / dt)
        
        # Cadence: count ankle height zero-crossings (foot-strike events)
        ankle_h = buf.last("ground_contact", len(buf.frames))
        contacts = sum(1 for i in range(1, len(ankle_h)) if ankle_h[i] != ankle_h[i-1])
        m["cadence"] = float(contacts / (len(buf.frames) / fps)) if buf.frames else 0.0

        # Range of motion over buffer window
        m["rom_knee_l"] = buf.range_of_motion("l_knee_deg")
        m["rom_knee_r"] = buf.range_of_motion("r_knee_deg")
        m["rom_hip_l"]  = buf.range_of_motion("l_hip_deg")
        m["rom_hip_r"]  = buf.range_of_motion("r_hip_deg")

    return m
# ── Overlay renderer ─────────────────────────────────────────────────────────

def draw_metrics_overlay(frame, metrics, pos=(15, 30)):
    """
    Draw all biomechanical metrics as a structured HUD split into five panels:
    Posture | Lower Body | Upper Body | Risk | Temporal
    Colour codes: green = OK, orange = mild, red = concerning.
    """
    out  = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = 0.52
    fw   = 1
    lh   = 22  # line height
    pw   = 210 # panel width

    def col(val, warn, bad):
        if abs(val) >= bad:  return (0, 0, 220)
        if abs(val) >= warn: return (0, 140, 255)
        return (0, 200, 0)

    def put(img, text, x, y, color=(200, 200, 200), scale=None, weight=None):
        cv2.putText(img, text, (x, y), font,
                    scale or fs, color, weight or fw, cv2.LINE_AA)

    def panel_bg(img, x, y, w, h, alpha=0.45):
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if metrics.get("low_confidence"):
        panel_bg(out, pos[0]-5, pos[1]-20, 220, 35)
        put(out, "LOW CONFIDENCE", pos[0], pos[1], (0, 0, 220), 0.6, 2)
        return out

    panels = [
        ("POSTURE", [
            ("Trunk lean",    f"{metrics['trunk_lean']:.1f}deg",    col(metrics["trunk_lean"],    15, 30)),
            ("Fwd lean",      f"{metrics['forward_lean']:.1f}deg",  col(metrics["forward_lean"],  20, 40)),
            ("Head tilt",     f"{metrics['head_tilt']:.1f}deg",     col(metrics["head_tilt"],     10, 20)),
            ("Hip drop",      f"{metrics['hip_drop_cm']:.1f}cm",    col(abs(metrics["hip_drop_cm"]), 3, 6)),
            ("Shoulder rot",  f"{metrics['shoulder_rot']:.1f}deg",  col(metrics["shoulder_rot"],  15, 30)),
        ]),
        ("LOWER BODY", [
            ("L-Knee",        f"{metrics['l_knee_deg']:.1f}deg",    col(180-metrics["l_knee_deg"],  20, 40)),
            ("R-Knee",        f"{metrics['r_knee_deg']:.1f}deg",    col(180-metrics["r_knee_deg"],  20, 40)),
            ("L-Hip",         f"{metrics['l_hip_deg']:.1f}deg",     col(180-metrics["l_hip_deg"],   25, 50)),
            ("R-Hip",         f"{metrics['r_hip_deg']:.1f}deg",     col(180-metrics["r_hip_deg"],   25, 50)),
            ("L-Ankle",       f"{metrics['l_ankle_deg']:.1f}deg",   col(abs(metrics["l_ankle_deg"]-90), 15, 30)),
            ("R-Ankle",       f"{metrics['r_ankle_deg']:.1f}deg",   col(abs(metrics["r_ankle_deg"]-90), 15, 30)),
            ("L-Valgus",      f"{metrics['l_valgus']:.1f}deg",      col(metrics["l_valgus"],  10, 20)),
            ("R-Valgus",      f"{metrics['r_valgus']:.1f}deg",      col(metrics["r_valgus"],  10, 20)),
            ("KWR",           f"{metrics['kwr']:.2f}",              col(abs(metrics["kwr"]-1.0), 0.3, 0.6)),
            ("Stride",        f"{metrics['stride_len']:.2f}u",      (200, 200, 200)),
            ("Contact",       metrics["ground_contact"],             (200, 200, 200)),
        ]),
        ("UPPER BODY", [
            ("L-Elbow",       f"{metrics['l_elbow_deg']:.1f}deg",   (200, 200, 200)),
            ("R-Elbow",       f"{metrics['r_elbow_deg']:.1f}deg",   (200, 200, 200)),
        ]),
        ("RISK", [
            ("Dyn Valgus",    f"{metrics['dvs']:.1f}/10",           col(metrics["dvs"], 4, 7)),
            ("LESS flag",     "YES" if metrics["less_flag"] else "no",
                              (0, 0, 220) if metrics["less_flag"] else (0, 200, 0)),
            ("Asym Knee",     f"{metrics['asym_knee']:.1f}%",       col(metrics["asym_knee"],  15, 30)),
            ("Asym Hip",      f"{metrics['asym_hip']:.1f}%",        col(metrics["asym_hip"],   15, 30)),
            ("Asym Ankle",    f"{metrics['asym_ankle']:.1f}%",      col(metrics["asym_ankle"], 15, 30)),
            ("Asym Elbow",    f"{metrics['asym_elbow']:.1f}%",      col(metrics["asym_elbow"], 15, 30)),
            ("Asym Valgus",   f"{metrics['asym_valgus']:.1f}%",     col(metrics["asym_valgus"],15, 30)),
        ]),
        ("TEMPORAL", [
            ("Knee vel L",    f"{metrics['knee_angular_vel_l']:.0f}d/s", col(metrics["knee_angular_vel_l"], 200, 500)),
            ("Knee vel R",    f"{metrics['knee_angular_vel_r']:.0f}d/s", col(metrics["knee_angular_vel_r"], 200, 500)),
            ("Hip vel L",     f"{metrics['hip_angular_vel_l']:.0f}d/s",  col(metrics["hip_angular_vel_l"],  200, 500)),
            ("Hip vel R",     f"{metrics['hip_angular_vel_r']:.0f}d/s",  col(metrics["hip_angular_vel_r"],  200, 500)),
            ("CoM sway",      f"{metrics['com_sway_lateral']:.3f}u",     col(metrics["com_sway_lateral"], 0.05, 0.15)),
            ("CoM accel",     f"{metrics['com_accel']:.2f}u/s",          (200, 200, 200)),
            ("Cadence",       f"{metrics['cadence']:.1f}s/s",            (200, 200, 200)),
            ("ROM Knee L",    f"{metrics['rom_knee_l']:.1f}deg",         (200, 200, 200)),
            ("ROM Knee R",    f"{metrics['rom_knee_r']:.1f}deg",         (200, 200, 200)),
            ("ROM Hip L",     f"{metrics['rom_hip_l']:.1f}deg",          (200, 200, 200)),
            ("ROM Hip R",     f"{metrics['rom_hip_r']:.1f}deg",          (200, 200, 200)),
        ]),
    ]

    # Layout: stack panels left to right, wrap if needed
    img_h, img_w = out.shape[:2]
    col_x   = pos[0]
    col_y   = pos[1]
    max_col_h = img_h - 20

    for panel_title, rows in panels:
        panel_h = lh * (len(rows) + 1) + 10
        if col_y + panel_h > max_col_h:
            col_x += pw
            col_y  = pos[1]
        # Background
        panel_bg(out, col_x - 5, col_y - 18, pw - 5, panel_h)
        # Title
        put(out, panel_title, col_x, col_y, (255, 255, 255), 0.50, 2)
        col_y += lh
        for label, value, color in rows:
            put(out, f"{label}: {value}", col_x + 4, col_y, color)
            col_y += lh
        col_y += 6  # gap between panels

    return out

def render_mesh_only(outputs, faces, img_h, img_w, buf=None, fps=30.0):
    """
    Render mesh(es) onto a black canvas sorted by depth,
    then draw skeleton + full biomechanics HUD.

    Args:
        buf : TemporalBuffer instance (None for single-image use)
        fps : video fps for temporal metric calculation
    """
    all_depths = np.stack([p["pred_cam_t"] for p in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[i] for i in np.argsort(-all_depths)]

    all_pred_vertices, all_faces = [], []
    for pid, p in enumerate(outputs_sorted):
        all_pred_vertices.append(p["pred_vertices"] + p["pred_cam_t"])
        all_faces.append(faces + len(p["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces         = np.concatenate(all_faces, axis=0)

    fake_cam_t = (
        np.max(all_pred_vertices[-2*18439:], axis=0) +
        np.min(all_pred_vertices[-2*18439:], axis=0)
    ) / 2
    all_pred_vertices -= fake_cam_t

    black_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    renderer  = Renderer(focal_length=outputs_sorted[-1]["focal_length"], faces=all_faces)
    rend = (renderer(
        all_pred_vertices, fake_cam_t, black_img,
        mesh_base_color=LIGHT_BLUE, scene_bg_color=(0, 0, 0),
    ) * 255).astype(np.uint8)

    # Per-person skeleton + HUD
    for person_output in outputs_sorted:
        kp2d = person_output["pred_keypoints_2d"]
        kp2d = np.concatenate([kp2d, np.ones((kp2d.shape[0], 1))], axis=-1)
        rend = skeleton_visualizer.draw_skeleton(rend, kp2d)
        metrics = compute_biomechanics(person_output, buf=buf, fps=fps)
        rend    = draw_metrics_overlay(rend, metrics)

    return rend


def process_video_mesh(video_path, output_path="output_video.mp4"):
    """Process all detected players per frame with full biomechanics overlay."""
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    buf    = TemporalBuffer(fps=fps, window=15)

    print(f"Input: {video_path}  |  {total} frames @ {fps:.1f}fps  |  {width}x{height}")
    try:
        for frame_idx in range(total if total > 0 else int(1e9)):
            ret, frame_bgr = cap.read()
            if not ret: break
            outputs = estimator.process_one_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            rend    = render_mesh_only(outputs, estimator.faces, height, width, buf=buf, fps=fps) if outputs else np.zeros((height, width, 3), dtype=np.uint8)
            writer.write(rend)
            if (frame_idx + 1) % 10 == 0:
                print(f"  {frame_idx+1}/{total} frames")
    finally:
        cap.release()
        writer.release()
    print(f"Done -> {output_path}")
    return output_path

# ── Colour-based jersey fingerprint ─────────────────────────────────────────

def get_jersey_color(frame_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Return the mean HSV colour of the torso region of a bounding box.
    We sample the middle third vertically (avoids head and legs).
    """
    x1, y1, x2, y2 = bbox[:4].astype(int)
    h = y2 - y1
    torso = frame_bgr[y1 + h // 3 : y1 + 2 * h // 3, x1:x2]
    if torso.size == 0:
        return np.zeros(3)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    return hsv.mean(axis=(0, 1))   # (H, S, V)


def color_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """Weighted HSV distance — hue counts more than value."""
    weights = np.array([2.0, 1.0, 0.5])
    return float(np.linalg.norm((c1 - c2) * weights))


# ── Tracker helpers ──────────────────────────────────────────────────────────

def outputs_to_bytetrack(outputs) -> np.ndarray:
    """
    Convert estimator outputs to ByteTrack input format:
    [[x1, y1, x2, y2, confidence], ...]
    """
    if not outputs:
        return np.empty((0, 6), dtype=np.float32)
    rows = []
    for p in outputs:
        x1, y1, x2, y2 = p["bbox"][:4]
        conf = float(p.get("bbox_score", p.get("score", 1.0)))
        rows.append([x1, y1, x2, y2, conf, 0])
    return np.array(rows, dtype=np.float32)


def match_output_to_track(person_output, tracks) -> int | None:
    """Return the track ID whose bbox best overlaps this person's bbox, or None."""
    if tracks is None or len(tracks) == 0:
        return None
    px1, py1, px2, py2 = person_output["bbox"][:4]
    best_iou, best_id = 0.0, None
    for t in tracks:
        tx1, ty1, tx2, ty2, tid = t[:5]
        ix1, iy1 = max(px1, tx1), max(py1, ty1)
        ix2, iy2 = min(px2, tx2), min(py2, ty2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            continue
        union = (px2-px1)*(py2-py1) + (tx2-tx1)*(ty2-ty1) - inter
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou, best_id = iou, int(tid)
    return best_id if best_iou > 0.3 else None


# ── Mode 2 – pick-up mid-video by jersey colour ──────────────────────────────

def process_video_track_color(
    video_path: str,
    reference_frame_idx: int,
    reference_bbox: tuple,           # (x1, y1, x2, y2) drawn around target
    output_path: str = "output_tracked_color.mp4",
    color_threshold: float = 30.0,   # lower = stricter match
):
    """
    Lets you specify a reference frame + bounding box for the target player.
    Their jersey colour is fingerprinted and then matched every frame via
    ByteTrack ID + colour distance fallback.

    Args:
        reference_frame_idx : frame number where you identify the player.
        reference_bbox      : (x1, y1, x2, y2) tight box around them.
        color_threshold     : max HSV distance to still count as a match.
    """
    tracker = ByteTrack()
    cap     = cv2.VideoCapture(video_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer  = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    ref_color  = None
    target_id  = None

    # Pre-extract reference jersey colour from the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_idx)
    ret, ref_frame = cap.read()
    if ret:
        ref_bbox   = np.array(reference_bbox, dtype=float)
        ref_color  = get_jersey_color(ref_frame, ref_bbox)
        print(f"Reference colour (HSV): {ref_color.round(1)}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind

    print(f"Mode: colour-match  |  {total} frames  |  {fps:.1f} fps")

    try:
        for frame_idx in range(total if total > 0 else int(1e9)):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            outputs = estimator.process_one_image(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            )
            dets   = outputs_to_bytetrack(outputs)
            tracks = tracker.update(dets, frame_bgr)

            # Try to find the target by colour each frame
            best_person, best_dist = None, float("inf")
            for p in (outputs or []):
                tid = match_output_to_track(p, tracks)
                color = get_jersey_color(frame_bgr, np.array(p["bbox"]))
                dist  = color_distance(color, ref_color) if ref_color is not None else 0

                # Prefer a known track ID, fall back to colour distance
                if tid == target_id:
                    best_person, best_dist = p, -1   # guaranteed best
                    break
                if dist < best_dist and dist < color_threshold:
                    best_person, best_dist = p, dist

            # Update target_id if we found someone
            if best_person is not None:
                tid = match_output_to_track(best_person, tracks)
                if tid is not None:
                    target_id = tid

            if best_person:
                rend = render_mesh_only([best_person], estimator.faces, height, width)
            else:
                rend = np.zeros((height, width, 3), dtype=np.uint8)

            writer.write(rend)

            if (frame_idx + 1) % 10 == 0:
                print(f"  {frame_idx + 1}/{total} frames…")
    finally:
        cap.release()
        writer.release()

    print(f"Done → {output_path}")