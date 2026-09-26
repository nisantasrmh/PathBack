import os
import json
import urllib.request
import PIL.Image
import numpy as np
import cv2
import streamlit as st
import mediapipe as mp

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
YUNET_PATH = os.path.join(MODELS_DIR, "yunet.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "sface.onnx")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def ensure_models_exist():
    """Ensure YuNet and SFace ONNX models are present on disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
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


def get_face_detector(input_size=(320, 320), conf_threshold=0.6, nms_threshold=0.3):
    """Initializes OpenCV YuNet face detector."""
    ensure_models_exist()
    if os.path.exists(YUNET_PATH):
        detector = cv2.FaceDetectorYN.create(
            YUNET_PATH, "", input_size, conf_threshold, nms_threshold
        )
        return detector
    return None


def get_face_recognizer():
    """Initializes OpenCV SFace deep face recognizer."""
    ensure_models_exist()
    if os.path.exists(SFACE_PATH):
        recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
        return recognizer
    return None


def image_obj_to_numpy(image_obj) -> np.ndarray:
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


def extract_deep_face_embeddings(image_input, return_all=False):
    """
    Detects faces in an image using YuNet and generates 128-d deep embeddings with SFace.
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
            embedding = feature[0].tolist()

            results.append({
                "bbox": bbox,
                "confidence": conf,
                "embedding": embedding,
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


def compute_similarity(embedding1, embedding2):
    """
    Computes Cosine Similarity and converted Match Confidence Percentage between two 128-d embeddings.
    Cosine Similarity: -1.0 to 1.0 (typical match threshold ~0.363+).
    Confidence Percentage: 0% to 100%.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0, 0.0

    v1 = np.array(embedding1, dtype=np.float32).flatten()
    v2 = np.array(embedding2, dtype=np.float32).flatten()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0, 0.0

    cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))

    # Scale SFace cosine similarity to intuitive 0-100% confidence
    # 0.36 is standard threshold. Values above 0.36 scale from 60% to 100%.
    if cos_sim <= 0.0:
        confidence = 0.0
    elif cos_sim < 0.363:
        confidence = (cos_sim / 0.363) * 60.0
    else:
        confidence = 60.0 + ((cos_sim - 0.363) / (1.0 - 0.363)) * 40.0

    confidence = min(100.0, max(0.0, confidence))
    return cos_sim, confidence


def extract_face_mesh_landmarks(image: np.ndarray):
    """
    Extract face mesh landmarks from an image using MediaPipe (Legacy / Fallback).
    Returns a flattened list of all (x, y, z) landmarks if a face is found, else None.
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
