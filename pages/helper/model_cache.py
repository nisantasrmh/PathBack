"""
Model Caching and Hardware Acceleration Engine for PathBack.

Provides cached instances of OpenCV YuNet (Face Detector) and SFace (Face Recognizer)
with automatic hardware backend detection (CUDA vs OpenCV CPU) and background subtractor caching.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple
import cv2
import streamlit as st

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models"
)
YUNET_PATH = os.path.join(MODELS_DIR, "yunet.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "sface.onnx")


def probe_hardware_backend() -> Tuple[int, int, str]:
    """
    Probes system capabilities to configure optimal OpenCV DNN backend and target.
    Utilizes CUDA GPU acceleration if available, falling back smoothly to CPU.

    Returns:
        Tuple[int, int, str]: (backend_id, target_id, backend_name)
    """
    try:
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA, "CUDA_GPU"
    except Exception:
        pass
    
    return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "OPENCV_CPU"


@st.cache_resource(show_spinner=False)
def get_cached_face_detector(
    input_size: Tuple[int, int] = (320, 320),
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> Optional[cv2.FaceDetectorYN]:
    """
    Creates and caches the YuNet FaceDetectorYN instance across Streamlit reruns.

    Args:
        input_size: (width, height) input resolution tuple.
        conf_threshold: Confidence threshold for face candidate detection.
        nms_threshold: Non-maximum suppression threshold.
        top_k: Maximum candidates to retain before NMS.

    Returns:
        Optional[cv2.FaceDetectorYN]: Configured face detector instance.
    """
    if not os.path.exists(YUNET_PATH):
        return None

    backend_id, target_id, _ = probe_hardware_backend()
    
    try:
        detector = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config="",
            input_size=input_size,
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=backend_id,
            target_id=target_id,
        )
        return detector
    except Exception as e:
        print(f"[model_cache] Error initializing FaceDetectorYN with backend {backend_id}: {e}")
        # Fallback to default CPU initialization
        return cv2.FaceDetectorYN.create(
            YUNET_PATH, "", input_size, conf_threshold, nms_threshold, top_k
        )


@st.cache_resource(show_spinner=False)
def get_cached_face_recognizer() -> Optional[cv2.FaceRecognizerSF]:
    """
    Creates and caches the SFace FaceRecognizerSF instance across Streamlit reruns.

    Returns:
        Optional[cv2.FaceRecognizerSF]: Configured deep face recognizer instance.
    """
    if not os.path.exists(SFACE_PATH):
        return None

    backend_id, target_id, _ = probe_hardware_backend()
    
    try:
        recognizer = cv2.FaceRecognizerSF.create(
            model=SFACE_PATH,
            config="",
            backend_id=backend_id,
            target_id=target_id,
        )
        return recognizer
    except Exception as e:
        print(f"[model_cache] Error initializing FaceRecognizerSF with backend {backend_id}: {e}")
        # Fallback to default CPU
        return cv2.FaceRecognizerSF.create(SFACE_PATH, "")


def get_mog2_subtractor(
    history: int = 500,
    var_threshold: float = 16.0,
    detect_shadows: bool = False
) -> cv2.BackgroundSubtractorMOG2:
    """
    Initializes an OpenCV MOG2 background subtractor for motion-detection gating.

    Args:
        history: Length of the history for background modeling.
        var_threshold: Mahalanobis distance threshold for foreground classification.
        detect_shadows: Whether to detect and mark shadows.

    Returns:
        cv2.BackgroundSubtractorMOG2: Configured background subtractor.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )
