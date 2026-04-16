import cv2
import numpy as np
import matplotlib.pyplot as plt
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together, visualize_sample
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from boxmot import ByteTrack

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
# These are the indices into pred_keypoints_2d / pred_keypoints_3d
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

# ── Geometry helpers ─────────────────────────────────────────────────────────

def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at joint b formed by segments b→a and b→c, in degrees."""
    v1 = a - b;  v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def valgus_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """
    Knee valgus: medial deviation of the knee from the hip-ankle line.
    Computed in the frontal plane (XY). Positive = valgus (knee caves in).
    """
    ref = ankle - hip
    dev = knee  - hip
    # Project onto frontal plane (ignore Z / depth)
    ref2 = ref[:2];  dev2 = dev[:2]
    if np.linalg.norm(ref2) < 1e-6 or np.linalg.norm(dev2) < 1e-6:
        return 0.0
    cos_a = np.dot(ref2, dev2) / (np.linalg.norm(ref2) * np.linalg.norm(dev2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


# ── Per-person biomechanics ──────────────────────────────────────────────────

def compute_biomechanics(person_output: dict) -> dict:
    """
    Compute biomechanical metrics from a single person's 3D keypoints.
    Uses pred_keypoints_3d (camera-space 3D) when available,
    falls back to pred_keypoints_2d.

    Returns a dict with:
        left_knee_deg, right_knee_deg   – knee flexion angles
        l_valgus, r_valgus              – knee valgus angles (deg)
        knee_width_ratio                – knee-width / hip-width
        hip_drop_cm                     – pelvic drop (L vs R hip height, cm)
        trunk_lean_deg                  – trunk lean from vertical
        asym                            – |left_knee - right_knee|
        low_confidence                  – True if keypoints look unreliable
    """
    # Prefer 3D keypoints; fall back to 2D
    kp = person_output.get("pred_keypoints_3d", person_output.get("pred_keypoints_2d"))
    use_3d = "pred_keypoints_3d" in person_output
    scale = 100.0  # arbitrary → cm-like units for hip_drop when using 3D

    def pt(name):
        return kp[J[name]]

    # ── Knee flexion (hip-knee-ankle angle) ──────────────────────────────────
    l_knee_deg = angle_between(pt("l_hip"),  pt("l_knee"),  pt("l_ankle"))
    r_knee_deg = angle_between(pt("r_hip"),  pt("r_knee"),  pt("r_ankle"))

    # ── Valgus ───────────────────────────────────────────────────────────────
    l_val = valgus_angle(pt("l_hip"), pt("l_knee"), pt("l_ankle"))
    r_val = valgus_angle(pt("r_hip"), pt("r_knee"), pt("r_ankle"))

    # ── Knee-width ratio ─────────────────────────────────────────────────────
    knee_w = np.linalg.norm(pt("l_knee")[:2] - pt("r_knee")[:2])
    hip_w  = np.linalg.norm(pt("l_hip")[:2]  - pt("r_hip")[:2])
    kwr    = float(knee_w / (hip_w + 1e-8))

    # ── Hip drop (pelvic tilt in frontal plane) ───────────────────────────────
    # Difference in Y (vertical) between left and right hip — positive = L drops
    hip_drop = float((pt("l_hip")[1] - pt("r_hip")[1]) * scale)

    # ── Trunk lean from vertical ─────────────────────────────────────────────
    # Vector from pelvis to neck; angle with the vertical axis
    trunk_vec = pt("neck") - pt("pelvis")
    vertical  = np.array([0.0, 1.0, 0.0]) if use_3d else np.array([0.0, 1.0])
    trunk_vec_n = trunk_vec[:len(vertical)]
    cos_a = np.dot(trunk_vec_n, vertical) / (np.linalg.norm(trunk_vec_n) + 1e-8)
    trunk_lean = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    # ── Confidence gate ───────────────────────────────────────────────────────
    # Flag frames where both knees are nearly fully extended (likely occluded / bad fit)
    low_conf = (l_knee_deg > 170 and r_knee_deg > 170)

    return {
        "left_knee_deg":  l_knee_deg,
        "right_knee_deg": r_knee_deg,
        "l_valgus":       l_val,
        "r_valgus":       r_val,
        "knee_width_ratio": kwr,
        "hip_drop_cm":    hip_drop,
        "trunk_lean_deg": trunk_lean,
        "asym":           abs(l_knee_deg - r_knee_deg),
        "low_confidence": low_conf,
    }


# ── Overlay renderer ─────────────────────────────────────────────────────────

def draw_metrics_overlay(frame: np.ndarray, metrics: dict, pos: tuple = (20, 40)) -> np.ndarray:
    """
    Draw biomechanical metrics as a HUD overlay on a frame (in-place).
    Colour-codes values: green = OK, orange = mild, red = concerning.
    """
    out   = frame.copy()
    x, y  = pos
    lh    = 30   # line height px
    fs    = 0.65 # font scale
    fw    = 1    # font weight
    font  = cv2.FONT_HERSHEY_SIMPLEX

    def colour(val, warn, bad):
        if abs(val) >= bad:   return (0, 0, 220)    # red
        if abs(val) >= warn:  return (0, 140, 255)   # orange
        return (0, 210, 0)                           # green

    if metrics.get("low_confidence"):
        cv2.putText(out, "LOW CONFIDENCE", (x, y), font, fs, (0,0,220), fw, cv2.LINE_AA)
        return out

    lines = [
        ("L-Knee",   f"{metrics['left_knee_deg']:.1f}deg",  colour(180-metrics['left_knee_deg'],  20, 40)),
        ("R-Knee",   f"{metrics['right_knee_deg']:.1f}deg", colour(180-metrics['right_knee_deg'], 20, 40)),
        ("L-Val",    f"{metrics['l_valgus']:.1f}deg",       colour(metrics['l_valgus'],  10, 20)),
        ("R-Val",    f"{metrics['r_valgus']:.1f}deg",       colour(metrics['r_valgus'],  10, 20)),
        ("KWR",      f"{metrics['knee_width_ratio']:.2f}",  colour(abs(metrics['knee_width_ratio']-1.0), 0.3, 0.6)),
        ("HipDrop",  f"{metrics['hip_drop_cm']:.1f}cm",     colour(abs(metrics['hip_drop_cm']),  3, 6)),
        ("Trunk",    f"{metrics['trunk_lean_deg']:.1f}deg", colour(metrics['trunk_lean_deg'],    20, 40)),
        ("Asym",     f"{metrics['asym']:.1f}deg",           colour(metrics['asym'],              15, 30)),
    ]

    for label, value, col in lines:
        text = f"{label}: {value}"
        cv2.putText(out, text, (x, y), font, fs, col, fw, cv2.LINE_AA)
        y += lh

    return out

def render_mesh_only(outputs, faces, img_h: int, img_w: int) -> np.ndarray:
    """
    Render the reconstructed 3D body mesh(es) onto a black canvas,
    then draw 2D joint skeletons and biomechanical metrics on top.
    All coordinates must be in the same space as img_h/img_w.
    """
    all_depths = np.stack([p["pred_cam_t"] for p in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    all_pred_vertices, all_faces = [], []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces         = np.concatenate(all_faces, axis=0)

    fake_pred_cam_t = (
        np.max(all_pred_vertices[-2 * 18439:], axis=0)
        + np.min(all_pred_vertices[-2 * 18439:], axis=0)
    ) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t

    black_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    renderer  = Renderer(focal_length=outputs_sorted[-1]["focal_length"], faces=all_faces)
    rend = (
        renderer(
            all_pred_vertices, fake_pred_cam_t, black_img,
            mesh_base_color=LIGHT_BLUE, scene_bg_color=(0, 0, 0),
        ) * 255
    ).astype(np.uint8)

    # Skeleton + metrics — keypoints must already be in img_h/img_w space
    for person_output in outputs_sorted:
        kp2d = person_output["pred_keypoints_2d"]
        kp2d = np.concatenate([kp2d, np.ones((kp2d.shape[0], 1))], axis=-1)
        rend = skeleton_visualizer.draw_skeleton(rend, kp2d)
        metrics = compute_biomechanics(person_output)
        rend    = draw_metrics_overlay(rend, metrics)

    return rend


def process_video_mesh(video_path: str, output_path: str = "output_video.mp4"):
    """
    Read a video frame by frame, run the SAM 3D body estimator on each frame,
    render the reconstructed mesh with joint skeleton and biomechanical metrics
    overlay onto a black background, and write the result to a new video file.
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Output : {output_path}")

    try:
        for frame_idx in range(total if total > 0 else int(1e9)):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            outputs   = estimator.process_one_image(frame_rgb)

            if outputs:
                rend_bgr = render_mesh_only(outputs, estimator.faces, height, width)
            else:
                rend_bgr = np.zeros((height, width, 3), dtype=np.uint8)

            writer.write(rend_bgr)

            if (frame_idx + 1) % 10 == 0:
                print(f"  processed {frame_idx + 1}/{total} frames…")

    finally:
        cap.release()
        writer.release()

    print(f"Done – saved to {output_path}")
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