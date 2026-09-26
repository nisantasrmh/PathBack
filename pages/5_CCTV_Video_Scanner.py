"""
PathBack - Real-Time CCTV & Surveillance Video AI Scanner.

High-throughput facial recognition pipeline featuring:
- Configurable frame-sampling (process every Nth frame)
- OpenCV MOG2 background subtraction for motion-detection gating
- Model caching (YuNet + SFace) with CUDA/CPU backend acceleration
- SORT/IoU multi-object face tracking and alert deduplication
- Real-time FPS counter, inference latency metrics, and timestamped thumbnail alert feed
"""

import os
import time
import tempfile
import cv2
import numpy as np
import streamlit as st
from sqlmodel import Session, select

from pages.helper import db_queries
from pages.helper.data_models import RegisteredCases, ConfidenceTier
from pages.helper.model_cache import (
    get_cached_face_detector,
    get_cached_face_recognizer,
    get_mog2_subtractor,
    probe_hardware_backend,
)
from pages.helper.utils import (
    compute_similarity,
    l2_normalize_vector,
    detect_motion_mog2,
    crop_face_thumbnail,
    SimpleFaceTracker,
)
from pages.helper.match_algo import (
    parse_or_extract_embedding,
    cosine_sim_to_confidence,
    calculate_confidence_tier,
)

st.set_page_config(
    page_title="PathBack - CCTV & Video Scanner",
    page_icon="📹",
    layout="wide"
)

if "login_status" not in st.session_state:
    st.warning("Please log in from the Home page to access CCTV & Video Scanning.")

elif st.session_state["login_status"]:
    user = st.session_state.user

    # Hardware backend probe
    _, _, hw_backend = probe_hardware_backend()

    st.title("📹 CCTV & Video Surveillance AI Scanner")
    st.markdown(
        "Real-time video surveillance pipeline with **MOG2 motion gating**, **IoU multi-face tracking**, "
        "and **SFace biometric indexing**."
    )

    # Load active missing person profiles into search index
    db_queries.create_db()
    with Session(db_queries.engine) as session:
        reg_cases = session.exec(
            select(RegisteredCases).where(RegisteredCases.status == "NF")
        ).all()

    targets = []
    for case in reg_cases:
        emb = parse_or_extract_embedding(case.id, case.face_mesh)
        if emb is not None:
            targets.append({
                "id": case.id,
                "name": case.name,
                "age": case.age or "N/A",
                "last_seen": case.last_seen or "N/A",
                "mobile": case.complainant_mobile or "N/A",
                "embedding": emb,  # L2-normalized 128-D vector
                "image_path": f"./resources/{case.id}.jpg"
            })

    if not targets:
        st.warning("⚠️ No active registered missing person cases with valid facial profiles found. Please register at least one case first.")
    else:
        # Top Metrics & Status Bar
        top_c1, top_c2, top_c3 = st.columns(3)
        top_c1.info(f"👤 **{len(targets)} Missing Person Profiles** loaded into search index.")
        top_c2.success(f"⚡ **Hardware Acceleration:** `{hw_backend}`")
        top_c3.caption(f"Logged in as: **{user}**")

        st.divider()

        # Layout: Control Panel & Live Analysis Feed
        col_settings, col_feed = st.columns([1, 2], gap="large")

        with col_settings:
            st.subheader("⚙️ Pipeline Configuration")
            
            confidence_threshold = st.slider(
                "Match Confidence Threshold (%)",
                min_value=40,
                max_value=95,
                value=60,
                step=5,
                help="Confidence required to trigger a sighting alert. Recommended: 60%+."
            )

            # Convert confidence % to cosine similarity threshold
            if confidence_threshold <= 60:
                sim_threshold = (confidence_threshold / 60.0) * 0.363
            else:
                sim_threshold = 0.363 + ((confidence_threshold - 60.0) / 40.0) * (1.0 - 0.363)

            frame_skip = st.slider(
                "Frame Sampling Interval (N)",
                min_value=1,
                max_value=30,
                value=5,
                help="Process every Nth frame. Higher values increase pipeline FPS drastically."
            )

            enable_motion_gating = st.checkbox(
                "Enable MOG2 Motion-Detection Gating",
                value=True,
                help="Skips deep neural network inference on static frames with no movement, boosting throughput 3-5x."
            )

            min_motion_pct = st.slider(
                "Motion Sensitivity (%)",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="Minimum frame pixel change percentage required to trigger face detection."
            ) if enable_motion_gating else 0.0

            video_file = st.file_uploader(
                "Upload Surveillance / CCTV Video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key="cctv_file_uploader"
            )

            start_scan = st.button("▶️ Launch Real-Time Surveillance Scan", type="primary", use_container_width=True)

        with col_feed:
            st.subheader("📺 Real-Time Analysis Feed")
            
            # Real-Time Telemetry Counters
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            fps_metric = m_c1.empty()
            latency_metric = m_c2.empty()
            frames_metric = m_c3.empty()
            motion_metric = m_c4.empty()

            fps_metric.metric("Pipeline FPS", "0.0")
            latency_metric.metric("Frame Latency", "0 ms")
            frames_metric.metric("Processed Frames", "0")
            motion_metric.metric("Motion State", "Idle")

            video_placeholder = st.empty()
            progress_bar = st.empty()
            status_text = st.empty()

        st.divider()
        st.subheader("🚨 Sighting Detections & Verified Alerts Feed")
        alerts_container = st.container()

        if start_scan and video_file is not None:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_source = cap.get(cv2.CAP_PROP_FPS) or 25.0

            # Initialize Cached Models & Pipeline Components
            recognizer = get_cached_face_recognizer()
            tracker = SimpleFaceTracker(iou_threshold=0.3, max_lost_frames=int(fps_source * 1.5))
            mog2_subtractor = get_mog2_subtractor() if enable_motion_gating else None

            detected_alerts = []
            frame_idx = 0
            processed_count = 0
            motion_triggered_count = 0
            start_time = time.time()

            status_text.text("⚡ Scanning video stream with AI pipeline...")

            while cap.isOpened():
                frame_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                processed_count += 1
                h, w = frame.shape[:2]

                # Downscale large 4K / 1080p frames for high-throughput detection
                display_frame = frame.copy()
                scale = 1.0
                if w > 960:
                    scale = 960.0 / w
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    h, w = frame.shape[:2]

                # 1. Motion Gating Check (OpenCV MOG2)
                has_motion = True
                motion_val = 0.0
                if enable_motion_gating and mog2_subtractor is not None:
                    has_motion, motion_val = detect_motion_mog2(
                        mog2_subtractor, frame, min_motion_percent=(min_motion_pct / 100.0)
                    )

                detected_faces_for_tracker = []

                # 2. Deep Face Detection & Feature Extraction (Only if motion detected)
                if has_motion:
                    motion_triggered_count += 1
                    detector = get_cached_face_detector(input_size=(w, h), conf_threshold=0.5)
                    if detector is not None:
                        detector.setInputSize((w, h))
                        _, faces = detector.detect(frame)

                        if faces is not None and len(faces) > 0 and recognizer is not None:
                            for face in faces:
                                bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
                                aligned_face = recognizer.alignCrop(frame, face)
                                face_feat = recognizer.feature(aligned_face)[0]
                                norm_feat = l2_normalize_vector(face_feat)

                                detected_faces_for_tracker.append({
                                    "bbox": bbox,
                                    "embedding": norm_feat,
                                    "raw_face": face,
                                })

                # 3. Multi-Object Face Tracking & Deduplication (IoU Tracker)
                active_tracks = tracker.update(detected_faces_for_tracker, frame_idx)

                # Keep a pristine clean copy of the frame for spotless face thumbnail captures
                clean_frame = frame.copy()

                # 4. Target Comparison & Visual Annotation (Only highlight verified matches)
                for track in active_tracks:
                    tx, ty, tbw, tbh = track.bbox
                    
                    # Compare track embedding against gallery of missing targets
                    best_match_target = None
                    highest_sim = -1.0
                    highest_conf = 0.0

                    for target in targets:
                        # Vector dot product (Cosine similarity on L2-normalized embeddings)
                        cos_sim = float(np.dot(track.embedding, target["embedding"]))
                        _, conf_pct = cosine_sim_to_confidence(cos_sim)

                        if cos_sim > highest_sim:
                            highest_sim = cos_sim
                            highest_conf = conf_pct
                            best_match_target = target

                    # Check match threshold
                    is_match = (highest_sim >= sim_threshold) and (best_match_target is not None)

                    # Only draw highlight box and alert label if a missing person is positively matched
                    if is_match and best_match_target:
                        cv2.rectangle(frame, (tx, ty), (tx + tbw, ty + tbh), (0, 0, 255), 2)
                        label = f"MATCH: {best_match_target['name']} ({int(highest_conf)}%)"
                        cv2.putText(
                            frame,
                            label,
                            (tx, max(20, ty - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2,
                        )

                        # Deduplication: Emit alert only once per track ID (or on higher confidence)
                        if not track.alert_emitted or highest_conf > (track.highest_confidence + 5.0):
                            track.alert_emitted = True
                            track.matched_target_id = best_match_target["id"]
                            track.matched_target_name = best_match_target["name"]
                            track.highest_confidence = highest_conf

                            # Timestamp calculation
                            timestamp_sec = frame_idx / fps_source
                            time_str = time.strftime("%H:%M:%S", time.gmtime(timestamp_sec))

                            # Crop spotless face thumbnail from the clean unannotated frame
                            thumbnail_rgb = crop_face_thumbnail(clean_frame, track.bbox, margin_ratio=0.35)

                            alert_record = {
                                "track_id": track.track_id,
                                "target_name": best_match_target["name"],
                                "target_id": best_match_target["id"],
                                "target_age": best_match_target["age"],
                                "target_last_seen": best_match_target["last_seen"],
                                "target_mobile": best_match_target["mobile"],
                                "confidence_pct": round(highest_conf, 1),
                                "similarity_score": round(highest_sim, 4),
                                "timestamp": time_str,
                                "frame_idx": frame_idx,
                                "thumbnail": thumbnail_rgb,
                                "target_img": best_match_target["image_path"],
                            }
                            detected_alerts.append(alert_record)

                # Compute Telemetry
                frame_latency_ms = (time.perf_counter() - frame_start) * 1000.0
                elapsed_total = time.time() - start_time
                current_fps = (processed_count / elapsed_total) if elapsed_total > 0 else 0.0

                # Update Telemetry Display
                fps_metric.metric("Pipeline FPS", f"{current_fps:.1f} fps")
                latency_metric.metric("Frame Latency", f"{frame_latency_ms:.0f} ms")
                frames_metric.metric("Processed", f"{processed_count} / {total_frames}")
                motion_metric.metric("Motion Area", f"{motion_val:.1f}%" if has_motion else "Gated (Static)")

                # Render Live Frame in Streamlit
                rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(
                    rgb_display,
                    caption=f"Frame {frame_idx}/{total_frames} | Pipeline: {current_fps:.1f} FPS",
                    use_container_width=True
                )

                if total_frames > 0:
                    progress_bar.progress(min(1.0, frame_idx / total_frames))

            cap.release()
            tfile.close()

            status_text.success(
                f"✅ Scan Complete! Processed {processed_count} frames across video feed. "
                f"Generated {len(detected_alerts)} deduplicated sighting alert(s)."
            )

            # Render Sighting Alerts Feed (Ranked strictly by highest match confidence percentage)
            if detected_alerts:
                # Sort alerts by match percentage descending and keep maximum 10 top matches
                detected_alerts.sort(key=lambda x: (x["confidence_pct"], x["similarity_score"]), reverse=True)
                detected_alerts = detected_alerts[:10]

                with alerts_container:
                    st.markdown(f"### Total Verified Sightings: `{len(detected_alerts)}` (Top 10 Ranked by Match %)")
                    
                    for idx, alert in enumerate(detected_alerts):
                        t_name = alert["target_name"]
                        t_conf = alert["confidence_pct"]
                        t_time = alert["timestamp"]
                        t_id = alert["target_id"]
                        track_id = alert["track_id"]

                        with st.container():
                            st.error(
                                f"**Rank #{idx + 1}** — Matched Missing Person: **{t_name}** | "
                                f"Confidence: **{t_conf}%** | Video Timestamp: **{t_time}** | Track ID: `#{track_id}`"
                            )

                            col_thumb, col_ref, col_info, col_btn = st.columns([1.5, 1.5, 3, 2], gap="small")

                            with col_thumb:
                                st.image(alert["thumbnail"], caption="CCTV Face Capture", use_container_width=True)

                            with col_ref:
                                if os.path.exists(alert["target_img"]):
                                    st.image(alert["target_img"], caption=f"Registered: {t_name}", use_container_width=True)
                                else:
                                    st.caption("No registered photo")

                            with col_info:
                                st.write(f"**Missing Person:** {t_name} (Age: {alert['target_age']})")
                                st.write(f"**Last Known Location:** {alert['target_last_seen']}")
                                st.write(f"**Complainant Mobile:** {alert['target_mobile']}")
                                st.write(f"**Similarity Score:** `{alert['similarity_score']}`")

                            with col_btn:
                                st.write("")
                                if st.button("✅ Confirm Sighting", key=f"cctv_conf_{t_id}_{track_id}_{idx}", type="primary", use_container_width=True):
                                    db_queries.update_registered_case_status(t_id, "F")
                                    st.success(f"Case for {t_name} confirmed found!")
                                    st.rerun()

                            st.markdown("---")
            else:
                with alerts_container:
                    st.info(f"🔍 No individuals from active missing person records were detected at {confidence_threshold}%+ confidence threshold.")
