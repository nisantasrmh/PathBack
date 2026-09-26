"""
Utility and Computer Vision Helper Functions for PathBack.

Includes ONNX model management, SFace deep embedding extraction with strict L2-normalization,
motion detection gating with OpenCV MOG2, and multi-object SORT/IoU face tracking with deduplication.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
import PIL.Image
import streamlit as st

from pages.helper.model_cache import (
    YUNET_PATH,
    SFACE_PATH,
    get_cached_face_detector,
    get_cached_face_recognizer,
    get_mog2_subtractor,
    probe_hardware_backend,
)

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def ensure_models_exist() -> None:
    """Ensure YuNet and SFace ONNX models are present on disk."""
    models_dir = os.path.dirname(YUNET_PATH)
    os.makedirs(models_dir, exist_ok=True)
    if not os.path.exists(YUNET_PATH):
        try:
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        except Exception as e:
            print(f"Error downloading YuNet: {e}")
    if not os.path.exists(SFACE_PATH):
        try:
            urllib.request.urlretrieve(SFACE_URL, SFACE_PATH)
        except Exception as e:
            print(f"Error downloading SFace: {e}")


def get_face_detector(
    input_size: Tuple[int, int] = (320, 320),
    conf_threshold: float = 0.6,
    nms_threshold: float = 0.3
) -> Optional[cv2.FaceDetectorYN]:
    """Initializes and returns cached OpenCV YuNet face detector."""
    ensure_models_exist()
    return get_cached_face_detector(
        input_size=input_size,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold
    )


def get_face_recognizer() -> Optional[cv2.FaceRecognizerSF]:
    """Initializes and returns cached OpenCV SFace deep face recognizer."""
    ensure_models_exist()
    return get_cached_face_recognizer()


def image_obj_to_numpy(image_obj: Any) -> np.ndarray:
    """Convert a Streamlit-uploaded image object or path to a numpy BGR/RGB array."""
    if isinstance(image_obj, np.ndarray):
        return image_obj
    elif isinstance(image_obj, str):
        image = PIL.Image.open(image_obj)
    elif hasattr(image_obj, "read"):
        image_obj.seek(0)
        image = PIL.Image.open(image_obj)
    else:
        image = image_obj
    
    if hasattr(image, "mode") and image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def l2_normalize_vector(v: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Strictly L2-normalizes a 128-D vector to the unit sphere so that
    Dot Product is mathematically identical to Cosine Similarity.
    """
    arr = np.asarray(v, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm > 1e-12:
        return (arr / norm).astype(np.float32)
    return arr.astype(np.float32)


def extract_deep_face_embeddings(
    image_input: Any,
    return_all: bool = False
) -> Union[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Detects faces in an image using YuNet and generates 128-D deep embeddings with SFace.
    All embeddings are strictly L2-normalized.
    
    Returns:
        - If return_all=False: dict with primary face embedding, bbox, and confidence (or None).
        - If return_all=True: list of dicts for every detected face in the image.
    """
    try:
        image_np = image_obj_to_numpy(image_input)
        h, w = image_np.shape[:2]
        if h == 0 or w == 0:
            return [] if return_all else None

        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        detector = get_face_detector(input_size=(w, h), conf_threshold=0.5)
        recognizer = get_face_recognizer()

        if detector is None or recognizer is None:
            return [] if return_all else None

        detector.setInputSize((w, h))
        _, faces = detector.detect(bgr_image)

        if faces is None or len(faces) == 0:
            return [] if return_all else None

        results = []
        for face in faces:
            bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
            conf = float(face[14])
            aligned_face = recognizer.alignCrop(bgr_image, face)
            feature = recognizer.feature(aligned_face)
            raw_embedding = feature[0]
            norm_embedding = l2_normalize_vector(raw_embedding).tolist()

            results.append({
                "bbox": bbox,
                "confidence": conf,
                "embedding": norm_embedding,
                "landmarks": [float(x) for x in face[4:14]]
            })

        if return_all:
            return results
        
        # Return the face with highest detection confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[0]

    except Exception as e:
        print(f"Error in extract_deep_face_embeddings: {e}")
        return [] if return_all else None


def compute_similarity(
    embedding1: Union[np.ndarray, List[float]],
    embedding2: Union[np.ndarray, List[float]]
) -> Tuple[float, float]:
    """
    Computes Cosine Similarity and converted Match Confidence Percentage between two 128-D embeddings.
    Cosine Similarity: -1.0 to 1.0.
    Confidence Percentage: 0% to 100%.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0, 0.0

    v1 = l2_normalize_vector(embedding1)
    v2 = l2_normalize_vector(embedding2)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0, 0.0

    cos_sim = float(np.dot(v1, v2))
    cos_sim = max(-1.0, min(1.0, cos_sim))

    # Scale SFace cosine similarity to 0-100% confidence
    # 0.363 is standard SFace threshold (~60% confidence)
    if cos_sim <= 0.0:
        confidence = 0.0
    elif cos_sim < 0.363:
        confidence = (cos_sim / 0.363) * 60.0
    else:
        confidence = 60.0 + ((cos_sim - 0.363) / (1.0 - 0.363)) * 40.0

    confidence = min(100.0, max(0.0, confidence))
    return cos_sim, confidence


def extract_face_mesh_landmarks(image: np.ndarray) -> Optional[List[float]]:
    """
    Extract face mesh landmarks from an image using MediaPipe.
    """
    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        ) as face_mesh:
            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                return [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
            else:
                return None
    except Exception:
        return None


# ==========================================================
# Motion Detection & Video Optimization Helpers
# ==========================================================
def detect_motion_mog2(
    subtractor: cv2.BackgroundSubtractorMOG2,
    frame_bgr: np.ndarray,
    min_motion_percent: float = 0.005,
    downsample_width: int = 320,
) -> Tuple[bool, float]:
    """
    Gating mechanism to determine if significant motion exists in the frame.
    Avoids running heavy deep neural network inference on static CCTV feeds.

    Args:
        subtractor: Background subtractor instance (MOG2).
        frame_bgr: Current BGR video frame.
        min_motion_percent: Minimum foreground area ratio (default 0.5% of frame).
        downsample_width: Width to resize for high-throughput motion calculation.

    Returns:
        Tuple[bool, float]: (has_motion, motion_percent)
    """
    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return False, 0.0

    # Downsample for ultra-fast motion estimation
    scale = downsample_width / float(w)
    small_h = int(h * scale)
    small_frame = cv2.resize(frame_bgr, (downsample_width, small_h), interpolation=cv2.INTER_LINEAR)
    
    # Apply Gaussian blur to reduce camera sensor noise
    blurred = cv2.GaussianBlur(small_frame, (5, 5), 0)
    fg_mask = subtractor.apply(blurred)

    # Threshold foreground mask (remove shadow artifacts)
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    motion_pixels = cv2.countNonZero(thresh)
    total_pixels = downsample_width * small_h
    motion_ratio = motion_pixels / float(total_pixels)

    has_motion = motion_ratio >= min_motion_percent
    return has_motion, motion_ratio * 100.0


def crop_face_thumbnail(
    frame_bgr: np.ndarray,
    bbox: List[int],
    margin_ratio: float = 0.25,
) -> np.ndarray:
    """
    Safely crops face bounding box from frame with contextual padding.

    Args:
        frame_bgr: Full BGR video frame.
        bbox: [x, y, w, h] face bounding box.
        margin_ratio: Padding expansion ratio around face.

    Returns:
        np.ndarray: Cropped face RGB image.
    """
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox

    # Expand bbox with margin
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# ==========================================================
# Real-Time Multi-Object Face Tracking & Deduplication (IoU Tracker)
# ==========================================================
def compute_bbox_iou(box1: List[int], box2: List[int]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x, y, w, h].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


class TrackedFace:
    """Represents an active face track across video frames."""

    def __init__(self, track_id: int, bbox: List[int], embedding: np.ndarray, frame_idx: int) -> None:
        self.track_id: int = track_id
        self.bbox: List[int] = bbox
        self.embedding: np.ndarray = embedding
        self.first_frame: int = frame_idx
        self.last_frame: int = frame_idx
        self.hits: int = 1
        self.time_since_update: int = 0
        self.alert_emitted: bool = False
        self.matched_target_id: Optional[str] = None
        self.matched_target_name: Optional[str] = None
        self.highest_confidence: float = 0.0


class SimpleFaceTracker:
    """
    High-throughput face tracker (SORT/ByteTrack IoU logic) for surveillance feeds.
    Maintains track identity across frames and prevents alert duplication.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 15) -> None:
        self.iou_threshold: float = iou_threshold
        self.max_lost_frames: int = max_lost_frames
        self.tracks: List[TrackedFace] = []
        self.next_track_id: int = 1

    def update(
        self,
        detected_faces: List[Dict[str, Any]],
        frame_idx: int
    ) -> List[TrackedFace]:
        """
        Updates active tracks with newly detected face bounding boxes.

        Args:
            detected_faces: List of dicts containing 'bbox' [x, y, w, h] and 'embedding'.
            frame_idx: Current frame index.

        Returns:
            List[TrackedFace]: Currently active and updated tracks.
        """
        # Increment time since update for existing tracks
        for t in self.tracks:
            t.time_since_update += 1

        unmatched_detections = list(range(len(detected_faces)))
        matched_track_indices = set()

        if self.tracks and detected_faces:
            # Build IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(detected_faces)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detected_faces):
                    iou_matrix[t_idx, d_idx] = compute_bbox_iou(track.bbox, det["bbox"])

            # Greedy bipartite matching
            while True:
                max_iou = float(np.max(iou_matrix)) if iou_matrix.size > 0 else 0.0
                if max_iou < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                
                # Update matched track
                matched_track = self.tracks[t_idx]
                matched_track.bbox = detected_faces[d_idx]["bbox"]
                matched_track.embedding = detected_faces[d_idx]["embedding"]
                matched_track.last_frame = frame_idx
                matched_track.hits += 1
                matched_track.time_since_update = 0

                matched_track_indices.add(t_idx)
                if d_idx in unmatched_detections:
                    unmatched_detections.remove(d_idx)

                # Zero out row and column
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Initialize new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detected_faces[d_idx]
            new_track = TrackedFace(
                track_id=self.next_track_id,
                bbox=det["bbox"],
                embedding=det["embedding"],
                frame_idx=frame_idx
            )
            self.next_track_id += 1
            self.tracks.append(new_track)

        # Prune dead tracks
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_lost_frames
        ]

        return self.tracks
