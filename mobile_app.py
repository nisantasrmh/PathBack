import uuid
import json
import streamlit as st
import numpy as np
import cv2

from pages.helper import db_queries
from pages.helper.data_models import PublicSubmissions
from pages.helper.utils import (
    image_obj_to_numpy,
    extract_deep_face_embeddings,
    extract_face_mesh_landmarks,
)

st.set_page_config(
    page_title="PathBack - Sighting & Missing Person Portal",
    page_icon="🚨",
    initial_sidebar_state="collapsed",
    layout="centered"
)

st.title("🚨 PathBack — Report a Sighting")
st.markdown("If you have spotted a missing person or someone needing help, upload their photo or snap a picture.")

source_type = st.radio("Choose Input Method", ["Upload Photo", "Take Photo with Camera"], horizontal=True)

image_obj = None
if source_type == "Upload Photo":
    image_obj = st.file_uploader(
        "Upload Photograph", type=["jpg", "jpeg", "png"], key="user_sighting_file"
    )
else:
    image_obj = st.camera_input("Snap Picture", key="user_sighting_cam")

face_data = None
preview_container = st.container()

if image_obj:
    image_numpy = image_obj_to_numpy(image_obj)
    
    with st.spinner("Analyzing image with AI Face Recognition..."):
        face_detection = extract_deep_face_embeddings(image_numpy)
        face_mesh = extract_face_mesh_landmarks(image_numpy)

    if face_detection:
        preview_img = image_numpy.copy()
        x, y, w, h = face_detection["bbox"]
        cv2.rectangle(preview_img, (x, y), (x + w, y + h), (0, 255, 120), 3)
        cv2.putText(
            preview_img,
            f"Face Detected ({int(face_detection['confidence']*100)}%)",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 120),
            2,
        )
        preview_container.image(preview_img, caption="AI Detected Face", use_container_width=True)
        st.success(f" Face detected ({int(face_detection['confidence']*100)}% clarity). Deep facial features extracted for matching.")
        
        face_data = {
            "embedding": face_detection["embedding"],
            "bbox": face_detection["bbox"],
            "mesh": face_mesh if face_mesh else []
        }
    else:
        preview_container.image(image_numpy, caption="Uploaded Photo", use_container_width=True)
        st.info("ℹ️ Face features will be scanned against records upon submission.")
        face_data = {"embedding": [], "bbox": [], "mesh": []}

    st.markdown("---")
    st.subheader("Sighting Information")

    with st.form(key="sighting_submission_form"):
        name = st.text_input("Your Name / Informant", placeholder="e.g. Citizen / Officer Name")
        mobile_number = st.text_input("Your Contact Mobile *", placeholder="Phone number for verification")
        email = st.text_input("Your Email (Optional)", placeholder="email@example.com")
        address = st.text_input("Exact Location of Sighting *", placeholder="e.g. Bus Stand near Central Mall, Delhi")
        birth_marks = st.text_input("Visible Marks / Distinguishing Details", placeholder="e.g. Red jacket, blue backpack, scar")

        submit_bt = st.form_submit_button(" Submit Sighting Report", use_container_width=True)

        if submit_bt:
            if not address or not mobile_number:
                st.error("Please provide at least your Contact Mobile and the Sighting Location.")
            else:
                unique_id = str(uuid.uuid4())
                uploaded_file_path = f"./resources/{unique_id}.jpg"
                
                with open(uploaded_file_path, "wb") as f:
                    if hasattr(image_obj, "getbuffer"):
                        f.write(image_obj.getbuffer())
                    else:
                        f.write(image_obj.read())

                public_submission_details = PublicSubmissions(
                    id=unique_id,
                    submitted_by=name or "Anonymous",
                    location=address,
                    email=email or "",
                    face_mesh=json.dumps(face_data),
                    mobile=mobile_number,
                    birth_marks=birth_marks or "",
                    status="NF",
                )

                db_queries.new_public_case(public_submission_details)
                st.success(" Sighting report submitted successfully! Our AI is comparing this sighting with registered missing persons.")
                st.balloons()
