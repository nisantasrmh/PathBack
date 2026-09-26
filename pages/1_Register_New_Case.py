import uuid
import numpy as np
import streamlit as st
import json
import base64
import cv2

from pages.helper.data_models import RegisteredCases
from pages.helper import db_queries
from pages.helper.utils import (
    image_obj_to_numpy,
    extract_deep_face_embeddings,
    extract_face_mesh_landmarks,
)
from pages.helper.streamlit_helpers import require_login

st.set_page_config(page_title="PathBack - Register Case", page_icon="🔍", layout="wide")

if "login_status" not in st.session_state:
    st.warning("Please log in from the Home page to register cases.")

elif st.session_state["login_status"]:
    user = st.session_state.user

    st.title("📋 Register New Missing Person Case")
    st.markdown("Upload a clear photograph and enter details of the missing individual.")

    image_col, form_col = st.columns([1, 1], gap="large")
    image_obj = None
    save_flag = False
    face_data = None

    with image_col:
        st.subheader("1. Photograph & AI Facial Scan")
        image_obj = st.file_uploader(
            "Upload Clear Photograph", type=["jpg", "jpeg", "png"], key="new_case_photo"
        )

        if image_obj:
            image_numpy = image_obj_to_numpy(image_obj)
            
            with st.spinner("Analyzing facial features with AI..."):
                face_detection = extract_deep_face_embeddings(image_numpy)
                face_mesh = extract_face_mesh_landmarks(image_numpy)

            if face_detection:
                # Draw bounding box preview
                preview_img = image_numpy.copy()
                x, y, w, h = face_detection["bbox"]
                cv2.rectangle(preview_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    preview_img,
                    f"Face Detected ({int(face_detection['confidence']*100)}%)",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                st.image(preview_img, caption="AI Face Detection Preview", use_container_width=True)
                st.success(f" Face detected with {int(face_detection['confidence']*100)}% detection confidence. 128-D Deep Metric Embeddings extracted.")

                face_data = {
                    "embedding": face_detection["embedding"],
                    "bbox": face_detection["bbox"],
                    "mesh": face_mesh if face_mesh else []
                }
            else:
                st.image(image_numpy, caption="Uploaded Image", use_container_width=True)
                st.warning("⚠️ No clear face detected by AI. Please upload a clear, front-facing image for accurate matching.")

    with form_col:
        st.subheader("2. Person & Contact Details")
        with st.form(key="new_case_form"):
            name = st.text_input("Full Name of Missing Person *", placeholder="e.g. John Doe")
            fathers_name = st.text_input("Father's / Guardian's Name", placeholder="e.g. Robert Doe")
            
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", min_value=1, max_value=120, value=15, step=1)
            mobile_number = c2.text_input("Contact Mobile Number *", placeholder="10-digit mobile")

            address = st.text_input("Home Address / Town *", placeholder="e.g. Sector 18, City")
            last_seen = st.text_input("Last Seen Location & Date *", placeholder="e.g. Near Metro Station on Oct 12")
            
            c3, c4 = st.columns(2)
            adhaar_card = c3.text_input("Aadhaar / ID Card Number", placeholder="Optional ID number")
            birthmarks = c4.text_input("Distinctive Marks / Tattoos", placeholder="e.g. Mole on left cheek")
            
            description = st.text_area("Additional Description / Clothing Details", placeholder="Clothing worn when last seen, height, identifying features...")

            c5, c6 = st.columns(2)
            complainant_name = c5.text_input("Reporting Officer / Complainant", value=user)
            complainant_phone = c6.text_input("Complainant Official Phone")

            submit_bt = st.form_submit_button(" Save & Register Case", use_container_width=True)

            if submit_bt:
                if not image_obj:
                    st.error("Please upload a photograph of the missing person.")
                elif not name or not last_seen:
                    st.error("Please fill in required fields: Name and Last Seen location.")
                else:
                    unique_id = str(uuid.uuid4())
                    uploaded_file_path = f"./resources/{unique_id}.jpg"
                    
                    # Save image file
                    with open(uploaded_file_path, "wb") as f:
                        if hasattr(image_obj, "getbuffer"):
                            f.write(image_obj.getbuffer())
                        else:
                            f.write(image_obj.read())

                    # If no face was detected, still store placeholder structure
                    if not face_data:
                        face_data = {"embedding": [], "bbox": [], "mesh": []}

                    new_case_details = RegisteredCases(
                        id=unique_id,
                        submitted_by=user,
                        name=name,
                        father_name=fathers_name,
                        age=str(age),
                        complainant_mobile=mobile_number or complainant_phone,
                        complainant_name=complainant_name,
                        face_mesh=json.dumps(face_data),
                        adhaar_card=adhaar_card or "N/A",
                        birth_marks=birthmarks or "None",
                        address=address or "Unknown",
                        last_seen=last_seen,
                        status="NF",
                        matched_with="",
                    )

                    db_queries.register_new_case(new_case_details)
                    save_flag = True

        if save_flag:
            st.success(f" Missing person case for **{name}** has been registered successfully with AI deep facial embeddings!")
            st.balloons()
else:
    st.write("You don't have access to this page")
