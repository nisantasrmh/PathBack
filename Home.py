import yaml
import base64
import streamlit as st
from yaml import SafeLoader
import streamlit_authenticator as stauth

from pages.helper import db_queries
from pages.helper.utils import ensure_models_exist

st.set_page_config(
    page_title="PathBack - AI Missing Person Recovery System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    authenticator.logout("🚪 Logout", "sidebar")

    st.session_state["login_status"] = True
    user_info = config["credentials"]["usernames"][st.session_state["username"]]
    st.session_state["user"] = user_info["name"]

    # Header section
    st.title(f"Welcome, {user_info['name']} 👋")
    st.markdown(f"**Jurisdiction / Area:** {user_info['area']}, {user_info['city']} &nbsp;|&nbsp; **Role:** `{user_info['role']}`")
    
    st.divider()

    # Metrics section
    found_cases = db_queries.get_registered_cases_count(user_info["name"], "F")
    non_found_cases = db_queries.get_registered_cases_count(user_info["name"], "NF")
    all_public = db_queries.list_public_cases()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚨 Active Missing Cases", len(non_found_cases))
    m2.metric("✅ Solved / Found Cases", len(found_cases))
    m3.metric("📍 Total Public Sightings", len(all_public) if all_public else 0)
    m4.metric("🤖 AI Engine", "SFace Deep Metric")

    st.divider()

    st.subheader("⚡ Quick Navigation & Operations")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("### 📋 Register Case\nRecord a new missing person profile with AI facial feature scanning.")
        st.page_link("pages/1_Register_New_Case.py", label="Go to Register Case ➡️", icon="📝")

    with col2:
        st.info("### 📂 All Cases\nBrowse, filter, and track all active and resolved missing person investigations.")
        st.page_link("pages/2_All_Cases.py", label="View All Cases ➡️", icon="🔍")

    with col3:
        st.info("### 🎯 Match Engine\nRun automated deep facial comparisons between sightings and missing profiles.")
        st.page_link("pages/3_Match_Cases.py", label="Launch AI Matching ➡️", icon="⚡")

    with col4:
        st.info("### 📹 CCTV Scanner\nScan CCTV footage and surveillance video clips frame-by-frame for alerts.")
        st.page_link("pages/5_CCTV_Video_Scanner.py", label="Open CCTV Scanner ➡️", icon="🎥")

    st.divider()
    st.markdown("💡 *Field officers and citizens can submit sightings directly via the mobile portal (`mobile_app.py`).*")

elif st.session_state.get("authentication_status") == False:
    st.error("Username/password is incorrect.")
elif st.session_state.get("authentication_status") == None:
    st.info("🔐 Please enter your credentials to access the PathBack Recovery Portal.")
    st.caption("Administrator Login: Username `admin` | Password `admin123`")
    st.session_state["login_status"] = False

