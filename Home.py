import yaml
import base64
import streamlit as st
from yaml import SafeLoader
import streamlit_authenticator as stauth

from pages.helper import db_queries
from pages.helper.utils import ensure_models_exist
from pages.helper.theme import inject_custom_theme

st.set_page_config(
    page_title="PathBack - AI Missing Person Recovery System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Modern Dark Glassmorphic Design System
inject_custom_theme()

# Ensure models are prepared on startup
ensure_models_exist()

if "login_status" not in st.session_state:
    st.session_state["login_status"] = False

try:
    with open("login_config.yml") as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Configuration file 'login_config.yml' not found")
    st.stop()

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# Perform login with exception handling for stale session tokens
try:
    authenticator.login(location="main")
except Exception:
    st.session_state["authentication_status"] = None
    st.session_state["login_status"] = False


if st.session_state.get("authentication_status"):
    authenticator.logout("🚪 Sign Out", "sidebar")

    st.session_state["login_status"] = True
    user_info = config["credentials"]["usernames"][st.session_state["username"]]
    st.session_state["user"] = user_info["name"]

    # Header section with Hero Banner
    st.markdown(f"""
    <div class="pb-hero-banner">
      <div class="pb-badge">
        <span class="pb-pulse-dot"></span>
        Investigation Portal Active
      </div>
      <h1 style="font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.4rem;">
        Welcome, {user_info['name']} 👋
      </h1>
      <p style="color: var(--pb-text-muted); font-size: 0.95rem;">
        <strong>Jurisdiction / Command:</strong> {user_info['area']}, {user_info['city']} &nbsp;•&nbsp; 
        <strong>Access Level:</strong> <span style="color:#60a5fa; font-weight:600;">{user_info['role']}</span> &nbsp;•&nbsp;
        <strong>Biometrics:</strong> <span style="color:#34d399; font-weight:600;">YuNet + SFace 128-D Active</span>
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics section
    found_cases = db_queries.get_registered_cases_count(user_info["name"], "F")
    non_found_cases = db_queries.get_registered_cases_count(user_info["name"], "NF")
    all_public = db_queries.list_public_cases()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚨 Active Missing Cases", len(non_found_cases) if non_found_cases else 0)
    m2.metric("✅ Solved / Reunited", len(found_cases) if found_cases else 0)
    m3.metric("📍 Public Sighting Tips", len(all_public) if all_public else 0)
    m4.metric("⚡ Vector Indexing", "FAISS + SFace")

    st.write("")
    st.write("")

    st.subheader("⚡ Core Operational Modules")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="pb-action-card">
          <div>
            <div class="pb-action-icon">📋</div>
            <h3>Register Case</h3>
            <p>Record a new missing person bulletin with automated YuNet face scanning and 128-D deep metric feature extraction.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/1_Register_New_Case.py", label="Open Registration ➡️", icon="📝")

    with col2:
        st.markdown("""
        <div class="pb-action-card">
          <div>
            <div class="pb-action-icon">📂</div>
            <h3>Case Records</h3>
            <p>Search, filter, and track all missing person profiles, citizen reports, and manage case resolution status.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/2_All_Cases.py", label="Manage Cases ➡️", icon="🔍")

    with col3:
        st.markdown("""
        <div class="pb-action-card">
          <div>
            <div class="pb-action-icon">🎯</div>
            <h3>AI Match Engine</h3>
            <p>Execute sub-millisecond vector similarity search comparing missing records against incoming citizen sightings.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/3_Match_Cases.py", label="Launch AI Matching ➡️", icon="⚡")

    with col4:
        st.markdown("""
        <div class="pb-action-card">
          <div>
            <div class="pb-action-icon">📹</div>
            <h3>CCTV Scanner</h3>
            <p>Process live video and surveillance feeds with MOG2 motion-detection gating and IoU multi-face tracking.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/5_CCTV_Video_Scanner.py", label="Launch Video Scanner ➡️", icon="🎥")

    st.write("")
    st.divider()
    st.caption("💡 Field officers and citizen volunteers can submit sighting tips directly from mobile via the mobile portal (`:8502`).")

elif st.session_state.get("authentication_status") == False:
    st.error("Username/password is incorrect.")
elif st.session_state.get("authentication_status") == None:
    st.info("🔐 Please enter your credentials to access the PathBack Recovery Portal.")
    st.caption("Administrator Login: Username `admin` | Password `admin123`")
    st.session_state["login_status"] = False

