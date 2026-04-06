"""
SAM 3D Body — FastAPI interface
Run from the notebook:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import asyncio
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Import your existing pipeline ────────────────────────────────────────────
from utils import (
    estimator, render_mesh_only, process_video_mesh,
    save_obj, LIGHT_BLUE, skeleton_visualizer,
)
from gradio_utils import _process_video_track_with_bbox

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="SAM 3D Body")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

WORK_DIR = Path(tempfile.mkdtemp(prefix="sam3d_"))
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(WORK_DIR)), name="outputs")

# ── Job state ─────────────────────────────────────────────────────────────────
jobs: dict[str, dict] = {}   # job_id → {status, result, error}

def _set(job_id, **kw):
    jobs.setdefault(job_id, {}).update(kw)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/preview")
async def preview(video: UploadFile = File(...)):
    """
    Save uploaded video, extract frame 0, run detector,
    return annotated JPEG + list of bbox coordinates.
    """
    video_path = WORK_DIR / f"{uuid.uuid4()}.mp4"
    video_path.write_bytes(await video.read())

    cap = cv2.VideoCapture(str(video_path))
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(400, "Could not read video")

    boxes = estimator.detector.run_human_detection(
        frame_bgr, det_cat_id=0, bbox_thr=0.5, nms_thr=0.3,
        default_to_full_image=False,
    )

    preview_img = frame_bgr.copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4].astype(int)
        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 230, 100), 2)
        cv2.putText(preview_img, str(idx), (x1 + 6, y1 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 100), 2, cv2.LINE_AA)

    preview_path = WORK_DIR / f"{video_path.stem}_preview.jpg"
    cv2.imwrite(str(preview_path), preview_img)

    return JSONResponse({
        "video_id":    video_path.stem,
        "preview_url": f"/outputs/{preview_path.name}",
        "n_players":   len(boxes),
        "boxes":       boxes.tolist() if len(boxes) else [],
    })


@app.post("/track")
async def track(
    background_tasks: BackgroundTasks,
    video_id: str,
    player_idx: int,
):
    """
    Start a background tracking job for the selected player.
    Returns a job_id to poll.
    """
    video_path = WORK_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        raise HTTPException(404, "Video not found — run /preview first")

    # Retrieve boxes from a fresh detection on frame 0
    cap = cv2.VideoCapture(str(video_path))
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(400, "Could not read video")

    boxes = estimator.detector.run_human_detection(
        frame_bgr, det_cat_id=0, bbox_thr=0.5, nms_thr=0.3,
        default_to_full_image=False,
    )

    if player_idx >= len(boxes):
        raise HTTPException(400, f"player_idx {player_idx} out of range (found {len(boxes)})")

    seed_bbox  = boxes[player_idx][:4]
    job_id     = str(uuid.uuid4())
    output_path = WORK_DIR / f"{job_id}_tracked.mp4"

    _set(job_id, status="running", result=None, error=None)

    def _run():
        try:
            _process_video_track_with_bbox(str(video_path), seed_bbox, str(output_path))
            _set(job_id, status="done", result=f"/outputs/{output_path.name}")
        except Exception as e:
            _set(job_id, status="error", error=str(e))

    background_tasks.add_task(_run)
    return {"job_id": job_id}


@app.get("/job/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/download/{filename}")
def download(filename: str):
    path = WORK_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), media_type="video/mp4",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})
