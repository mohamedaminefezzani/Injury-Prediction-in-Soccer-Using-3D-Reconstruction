import gradio as gr
import tempfile
import os
from utils import *

def run_process_image(img_array):
    """Process a single uploaded image and return the rendered result."""
    if img_array is None:
        return None, "No image provided."
    outputs = estimator.process_one_image(img_array)
    if not outputs:
        return None, "No person detected."
    h, w = img_array.shape[:2]
    rend = render_mesh_only(outputs, estimator.faces, h, w)
    metrics = compute_biomechanics(outputs[0])
    lines = [
        f"L-Knee : {metrics['left_knee_deg']:.1f}°",
        f"R-Knee : {metrics['right_knee_deg']:.1f}°",
        f"L-Val  : {metrics['l_valgus']:.1f}°",
        f"R-Val  : {metrics['r_valgus']:.1f}°",
        f"KWR    : {metrics['knee_width_ratio']:.2f}",
        f"HipDrop: {metrics['hip_drop_cm']:.1f} cm",
        f"Trunk  : {metrics['trunk_lean_deg']:.1f}°",
        f"Asym   : {metrics['asym']:.1f}°",
        f"Low confidence: {metrics['low_confidence']}",
    ]
    return cv2.cvtColor(rend, cv2.COLOR_BGR2RGB), "\n".join(lines)


def run_process_video_mesh(video_path):
    if video_path is None:
        return None, "No video provided."
    out_path = os.path.join(tempfile.mkdtemp(), "output_mesh.mp4")
    process_video_mesh(video_path, out_path)
    return out_path, f"Done → {out_path}"


def run_save_obj(img_array):
    if img_array is None:
        return None, "No image provided."
    outputs = estimator.process_one_image(img_array)
    if not outputs:
        return None, "No person detected."
    out_path = os.path.join(tempfile.mkdtemp(), "mesh.obj")
    save_obj(outputs, estimator.faces, out_path)
    return out_path, f"Saved → {out_path}"


# ── Track-player helpers ──────────────────────────────────────────────────────

# Shared state between preview and tracking steps
_preview_state = {"video_path": None, "detections": [], "frame": None}


def preview_first_frame(video_path):
    """
    Extract frame 0, run the detector, draw numbered bboxes, and return
    the annotated frame so the user can identify which player to track.
    """
    if video_path is None:
        return None, "No video provided.", gr.update(maximum=0, value=0, visible=False)

    cap = cv2.VideoCapture(video_path)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        return None, "Could not read video.", gr.update(visible=False)

    # Run detector on frame 0
    boxes = estimator.detector.run_human_detection(
        frame_bgr, det_cat_id=0, bbox_thr=0.5, nms_thr=0.3,
        default_to_full_image=False,
    )

    # Draw numbered bboxes
    preview = frame_bgr.copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4].astype(int)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview, str(idx),
            (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2, cv2.LINE_AA
        )

    # Store state for the tracking step
    _preview_state["video_path"] = video_path
    _preview_state["detections"] = boxes
    _preview_state["frame"]      = frame_bgr

    n = len(boxes)
    status = f"{n} player(s) detected. Enter the number of the player you want to track."
    slider = gr.update(maximum=max(n - 1, 0), value=0, visible=n > 0)

    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), status, slider


def run_track_selected(player_idx):
    """
    Run process_video_track_first with the bbox of the selected player
    injected as the initial lock-on target.
    """
    video_path = _preview_state["video_path"]
    boxes      = _preview_state["detections"]

    if video_path is None:
        return None, "Please preview a video first."
    if len(boxes) == 0:
        return None, "No detections from preview step."

    player_idx = int(player_idx)
    if player_idx >= len(boxes):
        return None, f"Invalid player index. Choose between 0 and {len(boxes)-1}."

    selected_bbox = boxes[player_idx][:4]
    out_path = os.path.join(tempfile.mkdtemp(), "output_tracked.mp4")

    # Run tracking with the selected player's bbox as the seed
    _process_video_track_with_bbox(video_path, selected_bbox, out_path)

    return out_path, f"Done → {out_path}"


def _process_video_track_with_bbox(video_path, seed_bbox, output_path):
    """
    Same as process_video_track_first but seeds the tracker with a
    specific bbox from frame 0 instead of picking the most-central player.
    """
    from boxmot import ByteTrack

    tracker      = ByteTrack()
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    target_id = None

    try:
        for frame_idx in range(total if total > 0 else int(1e9)):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Run detector to get all current bboxes
            all_boxes = estimator.detector.run_human_detection(
                frame_bgr, det_cat_id=0, bbox_thr=0.5, nms_thr=0.3,
                default_to_full_image=False,
            )

            if len(all_boxes) == 0:
                writer.write(np.zeros((height, width, 3), dtype=np.uint8))
                continue

            dets = np.hstack([
                all_boxes,
                np.ones((len(all_boxes), 1), dtype=np.float32),
                np.zeros((len(all_boxes), 1), dtype=np.float32),
            ])
            tracks = tracker.update(dets, frame_bgr)

            # On frame 0, lock onto the track whose bbox best overlaps seed_bbox
            if target_id is None and tracks is not None and len(tracks):
                if frame_idx == 0:
                    # Match seed_bbox via IoU
                    best_iou, best_id = 0.0, None
                    sx1, sy1, sx2, sy2 = seed_bbox
                    for t in tracks:
                        tx1, ty1, tx2, ty2, tid = t[:5]
                        ix1, iy1 = max(sx1, tx1), max(sy1, ty1)
                        ix2, iy2 = min(sx2, tx2), min(sy2, ty2)
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        if inter == 0: continue
                        union = (sx2-sx1)*(sy2-sy1) + (tx2-tx1)*(ty2-ty1) - inter
                        iou = inter / (union + 1e-8)
                        if iou > best_iou:
                            best_iou, best_id = iou, int(tid)
                    target_id = best_id
                    print(f"  Locked onto track ID {target_id} (IoU={best_iou:.2f})")

            # Find target bbox from ByteTrack
            target_bbox = None
            if tracks is not None:
                for t in tracks:
                    if int(t[4]) == target_id:
                        target_bbox = t[:4]
                        break

            if target_bbox is None:
                writer.write(np.zeros((height, width, 3), dtype=np.uint8))
                continue

            bbox_input = np.array(target_bbox, dtype=np.float32).reshape(1, 4)
            outputs    = estimator.process_one_image(frame_rgb, bboxes=bbox_input)

            if outputs:
                rend_bgr = render_mesh_only([outputs[0]], estimator.faces, height, width)
            else:
                rend_bgr = np.zeros((height, width, 3), dtype=np.uint8)

            writer.write(rend_bgr)

            if (frame_idx + 1) % 10 == 0:
                print(f"  {frame_idx + 1}/{total} frames…")

    finally:
        cap.release()
        writer.release()