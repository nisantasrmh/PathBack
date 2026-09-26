import streamlit as st

st.set_page_config(page_title="PathBack - Guide & Help", page_icon="❓", layout="wide")

st.title("❓ PathBack Guide & Best Practices")
st.markdown("Comprehensive operational guidelines for investigating missing persons with PathBack AI.")

tab1, tab2, tab3 = st.tabs(["📖 How the AI Works", "📷 Photo Quality Guidelines", "⚙️ Scanner & Sensitivity"])

with tab1:
    st.markdown("""
    ### Deep Face Metric Recognition Engine
    This system uses state-of-the-art **YuNet + SFace Deep Convolutional Neural Networks**:
    1. **YuNet Face Detector**: Ultra-fast face detection that locates faces even in challenging angles, low lighting, or crowded frames.
    2. **5-Point Landmark Alignment**: Aligns the face so that roll, pitch, and yaw distortions are normalized.
    3. **128-Dimensional Deep Feature Embeddings**: Converts the facial structure into an identity vector.
    4. **Cosine Similarity Matching**: Compares identity vectors across high-dimensional space.
    """)

with tab2:
    st.markdown("""
    ### Best Practices for Case Photos
    * **Lighting**: Ensure even lighting across the face without harsh shadows.
    * **Angle**: Frontal or 3/4 angle portraits work best.
    * **Occlusion**: Clear view of eyes, nose, and mouth (avoid sunglasses or heavy masks when possible).
    * **Resolution**: Minimum recommended face resolution is 100x100 pixels.
    """)

with tab3:
    st.markdown("""
    ### Sensitivity / Confidence Threshold Tuning
    * **Strict Mode (80% - 95%)**: Low false positive rate. Best for confirming identity when photos are clear.
    * **Balanced Mode (60% - 79%)**: Standard operational default. Good balance of recall and precision.
    * **Exploratory Mode (40% - 59%)**: High recall. Useful for grainy CCTV footage or distant sightings where manual investigator verification will follow.
    """)
