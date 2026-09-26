import os
import time
import tempfile
import cv2
import numpy as np
import streamlit as st
from sqlmodel import Session, select

from pages.helper import db_queries
from pages.helper.data_models import RegisteredCases, PublicSubmissions
from pages.helper.utils import (
    get_face_detector,
    get_face_recognizer,
    compute_similarity,
    extract_deep_face_embeddings,
)
from pages.helper.match_algo import parse_or_extract_embedding

st.set_page_config(page_title="PathBack - CCTV & Video Scanner", page_icon="📹", layout="wide")

if "login_status" not in st.session_state:
    st.warning("Please log in from the Home page to access CCTV & Video Scanning.")

elif st.session_state["login_status"]:
    user = st.session_state.user

    st.title("📹 PathBack — CCTV & Video Surveillance Scanner")
    st.markdown("Scan CCTV footage, surveillance clips, or video files frame-by-frame to locate missing individuals.")

    # Load active missing person profiles and embeddings into memory
    with Session(db_queries.engine) as session:
        reg_cases = session.exec(select(RegisteredCases).where(RegisteredCases.status == "NF")).all()

    targets = []
    for case in reg_cases:
        emb = parse_or_extract_embedding(case.id, case.face_mesh)
        if emb:
            targets.append({
                "id": case.id,
                "name": case.name,
                "age": case.age,
                "last_seen": case.last_seen,
                "embedding": np.array(emb, dtype=np.float32),
                "image_path": f"./resources/{case.id}.jpg"
            })

    if not targets:
        st.warning("⚠️ No active registered missing person cases with valid facial profiles found. Please register at least one case first.")
    else:
        st.info(f"Loaded **{len(targets)} active missing person profile(s)** into real-time search index.")

        col_settings, col_video = st.columns([1, 2], gap="large")

        with col_settings:
            st.subheader("⚙️ Scanner Settings")
            confidence_threshold = st.slider("Match Confidence Threshold", 40, 95, 60, 5)
            frame_skip = st.slider("Frame Processing Interval", 1, 30, 5, help="Process every N-th frame for faster speed.")
            
            # Convert confidence % to cosine similarity threshold
            if confidence_threshold <= 60:
                sim_threshold = (confidence_threshold / 60.0) * 0.363
            else:
                sim_threshold = 0.363 + ((confidence_threshold - 60.0) / 40.0) * (1.0 - 0.363)

            video_file = st.file_uploader(
                "Upload CCTV / Video File",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key="cctv_file_uploader"
            )

            start_scan = st.button("▶️ Start CCTV AI Scan", type="primary", use_container_width=True)

        with col_video:
            st.subheader("Live Analysis Feed & Alerts")
            video_placeholder = st.empty()
            progress_bar = st.empty()
            status_text = st.empty()

        st.divider()
        st.subheader("🚨 Detected Sighting Alerts")
        alerts_container = st.container()

        if start_scan and video_file is not None:
            # Save uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            
            recognizer = get_face_recognizer()

            detected_matches = []
            frame_idx = 0
            processed_count = 0

            status_text.text("Scanning CCTV footage...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                processed_count += 1
                h, w = frame.shape[:2]
                
                # Resize if frame is excessively large for speed
                scale = 1.0
                if w > 1280:
                    scale = 1280.0 / w
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    h, w = frame.shape[:2]

                detector = get_face_detector(input_size=(w, h), conf_threshold=0.5)
                if detector is not None:
                    detector.setInputSize((w, h))
                    _, faces = detector.detect(frame)

                    if faces is not None and len(faces) > 0 and recognizer is not None:
                        for face in faces:
                            bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
                            aligned_face = recognizer.alignCrop(frame, face)
                            face_feat = recognizer.feature(aligned_face)[0]

                            # Compare with all registered target persons
                            for target in targets:
                                cos_sim, conf_pct = compute_similarity(face_feat, target["embedding"])

                                if cos_sim >= sim_threshold:
                                    # Detected match!
                                    timestamp_sec = frame_idx / fps
                                    time_str = time.strftime("%H:%M:%S", time.gmtime(timestamp_sec))
                                    
                                    # Highlight bounding box on frame
                                    x, y, bw, bh = bbox
                                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                                    cv2.putText(
                                        frame,
                                        f"MATCH: {target['name']} ({int(conf_pct)}%)",
                                        (x, max(25, y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 0, 255),
                                        2,
                                    )

                                    # Save detected match snapshot
                                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    match_record = {
                                        "target_name": target["name"],
                                        "target_id": target["id"],
                                        "confidence": conf_pct,
                                        "time_str": time_str,
                                        "frame_idx": frame_idx,
                                        "snapshot": rgb_frame,
                                        "target_img": target["image_path"]
                                    }
                                    detected_matches.append(match_record)

                # Show live video frame in Streamlit
                rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_display, caption=f"Processing Frame {frame_idx}/{total_frames}", use_container_width=True)

                if total_frames > 0:
                    progress_bar.progress(min(1.0, frame_idx / total_frames))

            cap.release()
            tfile.close()

            status_text.success(f" Scan complete! Processed {processed_count} frames. Found {len(detected_matches)} potential match alert(s).")

            # Render detected alerts
            if detected_matches:
                with alerts_container:
                    for i, match_info in enumerate(detected_matches):
                        st.error(f"🚨 ALERT #{i+1}: Matched **{match_info['target_name']}** at timestamp **{match_info['time_str']}** (Confidence: **{match_info['confidence']:.1f}%**)")
                        a_col1, a_col2 = st.columns([1, 2])
                        with a_col1:
                            if os.path.exists(match_info["target_img"]):
                                st.image(match_info["target_img"], caption=f"Missing Profile: {match_info['target_name']}", width=180)
                        with a_col2:
                            st.image(match_info["snapshot"], caption=f"CCTV Capture at {match_info['time_str']}", use_container_width=True)
                        st.markdown("---")
            else:
                with alerts_container:
                    st.info(f"No individuals from active missing person records were detected at {confidence_threshold}%+ confidence threshold.")
