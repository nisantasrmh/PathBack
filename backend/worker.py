"""
PathBack Async Background Worker and Job Queue.

Handles asynchronous video surveillance scanning, FAISS vector index rebuilding,
and batch feature extraction jobs in worker threads without blocking API responses.
"""

from __future__ import annotations

import os
import time
import uuid
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from sqlmodel import Session, select

from pages.helper import db_queries
from pages.helper.data_models import RegisteredCases
from pages.helper.model_cache import (
    get_cached_face_detector,
    get_cached_face_recognizer,
    get_mog2_subtractor,
    probe_hardware_backend,
)
from pages.helper.match_algo import (
    parse_or_extract_embedding,
    cosine_sim_to_confidence,
)
from pages.helper.utils import (
    l2_normalize_vector,
    detect_motion_mog2,
    crop_face_thumbnail,
    SimpleFaceTracker,
)
from pages.helper.train_model import BiometricVectorIndex, get_train_data

logger = logging.getLogger("pathback.worker")
logging.basicConfig(level=logging.INFO)


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    progress_pct: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BackgroundWorkerQueue:
    """
    In-memory background job manager with thread pool execution.
    """
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PathBackWorker")
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            jobs_sorted = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [
                {
                    "job_id": j.job_id,
                    "job_type": j.job_type,
                    "status": j.status.value,
                    "progress_pct": round(j.progress_pct, 1),
                    "created_at": j.created_at.isoformat(),
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                    "error": j.error,
                }
                for j in jobs_sorted[:limit]
            ]

    def submit_job(self, job_type: str, fn: Callable[..., Dict[str, Any]], *args: Any, **kwargs: Any) -> str:
        job_id = str(uuid.uuid4())
        job = JobRecord(job_id=job_id, job_type=job_type, status=JobStatus.PENDING)
        with self._lock:
            self._jobs[job_id] = job

        def _runner():
            with self._lock:
                job.status = JobStatus.RUNNING
            try:
                def progress_cb(pct: float):
                    with self._lock:
                        job.progress_pct = max(0.0, min(100.0, pct))

                kwargs["progress_callback"] = progress_cb
                res = fn(*args, **kwargs)
                with self._lock:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now(timezone.utc)
                    job.progress_pct = 100.0
                    job.result = res
            except Exception as e:
                logger.exception(f"Background job {job_id} failed: {e}")
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now(timezone.utc)
                    job.error = str(e)

        self.executor.submit(_runner)
        return job_id


# Global worker instance
worker_queue = BackgroundWorkerQueue(max_workers=3)


# -------------------------------------------------------------
# Background Tasks
# -------------------------------------------------------------
def run_video_surveillance_job(
    video_path: str,
    confidence_threshold: float = 60.0,
    frame_skip: int = 5,
    enable_motion_gating: bool = True,
    min_motion_pct: float = 0.5,
    save_thumbnails: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """
    Executes surveillance video analysis in worker thread.
    """
    t0 = time.time()
    _, _, hw_backend = probe_hardware_backend()

    # Calculate cosine threshold
    if confidence_threshold <= 60.0:
        sim_threshold = (confidence_threshold / 60.0) * 0.363
    else:
        sim_threshold = 0.363 + ((confidence_threshold - 60.0) / 40.0) * (1.0 - 0.363)

    # 1. Load active registered missing person targets
    db_queries.create_db()
    with Session(db_queries.engine) as session:
        reg_cases = session.exec(
            select(RegisteredCases).where(RegisteredCases.status == "NF")
        ).all()

    targets: List[Dict[str, Any]] = []
    for c in reg_cases:
        emb = parse_or_extract_embedding(c.id, c.face_mesh)
        if emb is not None:
            targets.append({
                "id": c.id,
                "name": c.name,
                "age": c.age or "N/A",
                "mobile": c.complainant_mobile or "N/A",
                "embedding": emb,
            })

    if not targets:
        return {
            "status": "error",
            "message": "No active registered cases with valid facial profiles found for comparison.",
            "alerts": [],
        }

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 25.0

    recognizer = get_cached_face_recognizer()
    tracker = SimpleFaceTracker(iou_threshold=0.3, max_lost_frames=int(fps_source * 1.5))
    mog2_subtractor = get_mog2_subtractor() if enable_motion_gating else None

    frame_idx = 0
    processed_count = 0
    alerts: List[Dict[str, Any]] = []

    resources_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")
    os.makedirs(resources_dir, exist_ok=True)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            processed_count += 1
            h, w = frame.shape[:2]

            # Resize if > 960 width
            if w > 960:
                scale = 960.0 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                h, w = frame.shape[:2]

            # Motion detection gating
            has_motion = True
            if enable_motion_gating and mog2_subtractor is not None:
                has_motion, _ = detect_motion_mog2(
                    mog2_subtractor, frame, min_motion_percent=(min_motion_pct / 100.0)
                )

            detected_faces_for_tracker = []
            if has_motion:
                detector = get_cached_face_detector(input_size=(w, h), conf_threshold=0.5)
                if detector is not None:
                    detector.setInputSize((w, h))
                    _, faces = detector.detect(frame)

                    if faces is not None and len(faces) > 0 and recognizer is not None:
                        for face in faces:
                            bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
                            aligned = recognizer.alignCrop(frame, face)
                            feat = recognizer.feature(aligned)[0]
                            norm_feat = l2_normalize_vector(feat)

                            detected_faces_for_tracker.append({
                                "bbox": bbox,
                                "embedding": norm_feat,
                                "raw_face": face,
                            })

            active_tracks = tracker.update(detected_faces_for_tracker, frame_idx)

            for track in active_tracks:
                best_target = None
                highest_sim = -1.0
                highest_conf = 0.0

                for target in targets:
                    cos_sim = float(np.dot(track.embedding, target["embedding"]))
                    if cos_sim > highest_sim:
                        highest_sim = cos_sim
                        best_target = target
                        _, highest_conf = cosine_sim_to_confidence(cos_sim)

                if highest_sim >= sim_threshold and best_target is not None:
                    track.matched_target = best_target
                    track.match_score = highest_sim

                    if not track.alert_fired:
                        track.alert_fired = True
                        timestamp_sec = round(frame_idx / max(1.0, fps_source), 2)

                        # Save thumbnail
                        thumb_filename = f"alert_track{track.track_id}_{int(time.time()*1000)}.jpg"
                        thumb_path = os.path.join(resources_dir, thumb_filename)
                        thumb_img = crop_face_thumbnail(frame, track.bbox)
                        if thumb_img is not None and save_thumbnails:
                            cv2.imwrite(thumb_path, cv2.cvtColor(thumb_img, cv2.COLOR_RGB2BGR))
                            thumb_url = f"/api/resources/{thumb_filename}"
                        else:
                            thumb_url = None

                        alerts.append({
                            "track_id": track.track_id,
                            "case_id": best_target["id"],
                            "case_name": best_target["name"],
                            "confidence_pct": round(highest_conf, 1),
                            "similarity_score": round(highest_sim, 4),
                            "timestamp_sec": timestamp_sec,
                            "frame_number": frame_idx,
                            "bbox": track.bbox,
                            "thumbnail_url": thumb_url,
                        })

            if progress_callback and total_frames > 0:
                progress_callback((frame_idx / total_frames) * 100.0)

    finally:
        cap.release()
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

    elapsed = max(0.001, time.time() - t0)
    effective_fps = round(processed_count / elapsed, 1)

    return {
        "status": "success",
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "fps": effective_fps,
        "elapsed_seconds": round(elapsed, 2),
        "hardware_backend": hw_backend,
        "alerts_count": len(alerts),
        "alerts": alerts,
    }


def run_rebuild_index_job(progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
    """
    Rebuilds the FAISS vector index from all active cases.
    """
    if progress_callback:
        progress_callback(10.0)

    case_ids, embeddings = get_train_data()
    if progress_callback:
        progress_callback(50.0)

    index_wrapper = BiometricVectorIndex()
    res = index_wrapper.build(case_ids, embeddings)

    if progress_callback:
        progress_callback(90.0)

    saved = index_wrapper.save()
    if progress_callback:
        progress_callback(100.0)

    return {
        "build_result": res,
        "persisted": saved,
        "cases_indexed": len(case_ids),
        "backend": index_wrapper.backend,
    }
