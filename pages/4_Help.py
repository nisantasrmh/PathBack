"""
PathBack - Help, Documentation & Operational Guide.

Provides comprehensive instructions, best practices, architecture explanations,
and troubleshooting tips for investigators, administrators, and field officers.
"""

import streamlit as st

st.set_page_config(
    page_title="PathBack - Guide & Help Center",
    page_icon="❓",
    layout="wide"
)

st.title("❓ PathBack Help & Operational Guide")
st.markdown(
    "Comprehensive guide on AI facial recognition, FAISS vector search, real-time CCTV surveillance, "
    "geospatial filtering, and case resolution workflows."
)

tab_overview, tab_biometrics, tab_cctv, tab_geo, tab_faq = st.tabs([
    "🚀 Getting Started & Workflows",
    "🧠 AI Biometrics & FAISS Indexing",
    "📹 CCTV Scanner & Motion Gating",
    "📍 Geolocation & Multi-Modal Sightings",
    "💡 FAQs & Troubleshooting"
])

# ==========================================================
# Tab 1: Getting Started & Workflows
# ==========================================================
with tab_overview:
    st.subheader("📋 Step-by-Step Investigation Workflow")
    
    col_w1, col_w2, col_w3 = st.columns(3)
    
    with col_w1:
        st.markdown("""
        #### 1️⃣ Register Missing Person
        * Navigate to **📋 Register New Case**.
        * Upload a clear photograph.
        * The AI automatically detects faces using **YuNet** and generates **128-D SFace embeddings**.
        * Fill in personal details, last known location/coordinates, and distinctive marks.
        """)
        
    with col_w2:
        st.markdown("""
        #### 2️⃣ Scan Public Sightings & CCTV
        * Field sightings submitted via the **Mobile Portal** (`:8502`) are compared in real-time.
        * Use **🎯 AI Facial Match** to run cross-database vector comparisons.
        * Use **📹 CCTV & Video Scanner** to process surveillance footage frame-by-frame.
        """)

    with col_w3:
        st.markdown("""
        #### 3️⃣ Review & Resolve Cases
        * Inspect candidate matches categorized by confidence tiers.
        * Compare the registered photo with the sighting/CCTV capture.
        * Click **✅ Confirm & Mark Found** to update case status across the system.
        """)

    st.divider()

    st.subheader("📂 Case Status Lifecycle")
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.error("🔴 **Active / Unresolved (`NF`)**\n\nMissing individual actively being scanned against sightings and surveillance feeds.")
    s_col2.warning("🟡 **Under Review**\n\nCandidate match detected; pending officer verification or field investigation.")
    s_col3.success("🟢 **Solved / Found (`F`)**\n\nIndividual located and case officially resolved and archived.")

# ==========================================================
# Tab 2: AI Biometrics & FAISS Indexing
# ==========================================================
with tab_biometrics:
    st.subheader("🧠 Deep Metric Face Recognition & Vector Search")
    st.markdown("""
    PathBack utilizes an advanced computer vision and vector indexing pipeline engineered for accuracy and sub-millisecond throughput:
    """)

    b_col1, b_col2 = st.columns([1, 1], gap="large")

    with b_col1:
        st.markdown("""
        #### 🔬 Biometric Pipeline Architecture
        1. **YuNet Neural Face Detection**: Ultra-lightweight ONNX detector capable of finding faces across scale, orientation, and occlusions.
        2. **5-Point Landmark Alignment**: Normalizes pitch, roll, and yaw angles to align the eyes, nose, and mouth.
        3. **SFace 128-D Embedding Extraction**: Projects facial features into a 128-dimensional metric hyperspace.
        4. **Strict L2 Unit-Norm Normalization**:
           $$\\mathbf{v}_{\\text{norm}} = \\frac{\\mathbf{v}}{\\|\\mathbf{v}\\|_2}$$
           On the unit sphere, **Inner Product strictly equals Cosine Similarity** ($\langle \\mathbf{u}, \\mathbf{v} \\rangle = \\cos(\\theta)$), allowing accelerated SIMD search.
        5. **FAISS Vector Indexing (`IndexFlatIP` & `IndexHNSWFlat`)**: Executes sub-millisecond similarity lookups against thousands of profiles with automated NumPy vectorized fallback.
        """)

    with b_col2:
        st.markdown("#### 🎯 Match Confidence Scoring & Tiers")
        
        st.markdown("""
        | Confidence Tier | Confidence % | Cosine Score | Action Recommendation |
        | :--- | :--- | :--- | :--- |
        | 🟢 **High Confidence** | **$\ge 75\%$** | $\ge 0.50$ | **Strong Match:** Immediate investigator notification and family contact. |
        | 🟡 **Review Needed** | **$50\% - 74\%$** | $0.30 - 0.49$ | **Potential Match:** Requires visual cross-check of birthmarks, clothing, and angle. |
        | ⚪ **Dismissed** | **$< 50\%$** | $< 0.30$ | Filtered out below noise threshold. |
        """)

        st.caption("ℹ️ OpenCV SFace standard verification baseline is **0.363 cosine similarity (~60% confidence)**.")

# ==========================================================
# Tab 3: CCTV Scanner & Motion Gating
# ==========================================================
with tab_cctv:
    st.subheader("📹 Real-Time CCTV Video Surveillance Scanner")
    st.markdown("""
    The CCTV scanner is optimized to process high-definition 1080p and 4K surveillance video streams at real-time speeds.
    """)

    c_col1, c_col2 = st.columns([1, 1], gap="large")

    with c_col1:
        st.markdown("""
        #### ⚡ Performance Optimizations
        * **MOG2 Motion-Detection Gating**:
          - Uses OpenCV background subtraction to measure frame motion pixel density.
          - **Bypasses static scenes**: If no movement is detected, neural network inference is skipped, saving $60\%-80\%$ compute.
        * **IoU Multi-Object Face Tracking (SORT/ByteTrack)**:
          - Tracks facial bounding boxes across consecutive frames with persistent track IDs.
        * **Alert Deduplication**:
          - Emits **at most one alert per person** throughout a video clip, preventing repeated notification spam.
        * **Spotless Face Thumbnail Extraction**:
          - Thumbnails are cropped directly from clean unannotated frames for crisp alert records.
        """)

    with c_col2:
        st.markdown("""
        #### ⚙️ Recommended Scanner Settings
        * **Frame Sampling Interval ($N$)**:
          - **$N=5$ (Default)**: Best balance of speed and coverage for normal CCTV (process 5 frames/sec on a 25 FPS stream).
          - **$N=1-2$**: High-precision mode for fast-moving crowds.
          - **$N=10-15$**: Ultra-fast preview mode for scanning multi-hour long surveillance footage.
        * **Motion Sensitivity**: Set to `0.5%` for indoor corridors; `1.0% - 2.0%` for busy outdoor streets with background foliage movement.
        * **Match Sensitivity Threshold**: Set to `60%` for balanced identification.
        """)

# ==========================================================
# Tab 4: Geolocation & Multi-Modal Sightings
# ==========================================================
with tab_geo:
    st.subheader("📍 Geospatial Intelligence & Multi-Modal Sightings")
    
    g_col1, g_col2 = st.columns([1, 1], gap="large")

    with g_col1:
        st.markdown("""
        #### 🌐 Haversine Radius Filtering
        PathBack can filter candidate sightings and missing person profiles based on great-circle geographic proximity:
        $$d = 2R \\arcsin\\left(\\sqrt{\\sin^2\\left(\\frac{\\Delta\\phi}{2}\\right) + \\cos(\\phi_1)\\cos(\\phi_2)\\sin^2\\left(\\frac{\\Delta\\lambda}{2}\\right)}\\right)$$
        where $R = 6,371.01\\text{ km}$.

        * Filter sightings within $5\\text{ km}$, $25\\text{ km}$, or $100\\text{ km}$ of the person's last known location.
        * Correlate travel time estimates with sighting timestamps.
        """)

    with g_col2:
        st.markdown("""
        #### 📱 Mobile Sighting Portal (`mobile_app.py`)
        Citizens and patrol officers can access the dedicated mobile interface on port `:8502`:
        * **Instant Camera Snap**: Capture on-the-spot photos from mobile browsers.
        * **GPS Geotagging**: Automatically records latitude and longitude coordinates.
        * **Informant Metadata**: Records informant contact number, location description, and visible markings for rapid field follow-up.
        * **Administrative Audit**: System administrators can review and verify citizen submissions.
        """)

# ==========================================================
# Tab 5: FAQs & Troubleshooting
# ==========================================================
with tab_faq:
    st.subheader("💡 Frequently Asked Questions")

    with st.expander("❓ What should I do if a registered photo is low quality or blurry?"):
        st.markdown("""
        * If possible, upload a clear, front-facing image where both eyes and the nose are visible.
        * Avoid photos with heavy sunglasses, hats casting dark shadows, or extreme head tilts ($> 45^\\circ$).
        * If only a low-resolution photo is available, lower the **Match Confidence Threshold** in **AI Facial Match** to `50% - 55%` for exploratory search.
        """)

    with st.expander("❓ How do I refresh the FAISS vector index after adding many cases?"):
        st.markdown("""
        The system automatically updates its search index dynamically. If you want to rebuild the persistent offline index files on disk, execute:
        ```bash
        python pages/helper/train_model.py
        ```
        This updates `models/biometric_index.faiss` and `models/biometric_index_meta.pkl`.
        """)

    with st.expander("❓ How does the CCTV Scanner prevent multiple alerts for the same person?"):
        st.markdown("""
        The integrated **IoU Face Tracker** assigns a unique tracking ID to each face detected across consecutive video frames. Once an alert has been generated for a given track ID, duplicate alerts are automatically suppressed unless a significantly clearer frame (higher confidence score) is captured.
        """)

    with st.expander("❓ How can I access the citizen mobile portal from a smartphone on the local network?"):
        st.markdown("""
        1. Ensure your computer and smartphone are connected to the same Wi-Fi network.
        2. Run the mobile app:
           ```bash
           python -m streamlit run .\\mobile_app.py --server.port 8502
           ```
        3. On your smartphone browser, open the **Network URL** (e.g. `http://192.168.x.x:8502`).
        """)

    with st.expander("❓ Where is the database stored and how do I back it up?"):
        st.markdown("""
        All case records, sighting reports, and biometric metadata are stored in `sqlite_database.db` in the project root directory. To back up your data, simply make a copy of this file. The system features automatic schema migrations, so your data will persist safely across updates.
        """)

st.divider()
st.caption("PathBack AI Missing Person Recovery System | Developed for Humanitarian Recovery & Law Enforcement Operations.")
