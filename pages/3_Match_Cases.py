import os
import streamlit as st
from pages.helper import db_queries, match_algo
from pages.helper.streamlit_helpers import require_login

st.set_page_config(page_title="PathBack - AI Facial Match", page_icon="🎯", layout="wide")

if "login_status" not in st.session_state:
    st.warning("Please log in from the Home page to access AI Case Matching.")

elif st.session_state["login_status"]:
    user = st.session_state.user

    st.title("🎯 AI Deep Face Matching Engine")
    st.markdown("Compare registered missing person profiles against public sightings and field reports using deep neural facial embeddings.")

    # Controls Bar
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 1])
    
    with ctrl_c1:
        confidence_slider = st.slider(
            "Match Sensitivity / Confidence Threshold",
            min_value=40,
            max_value=95,
            value=60,
            step=5,
            help="Higher values return only near-identical matches (lower false positives). Lower values allow matches under tough lighting or angle variations."
        )

    # Convert confidence % back to cosine threshold
    if confidence_slider <= 60:
        sim_threshold = (confidence_slider / 60.0) * 0.363
    else:
        sim_threshold = 0.363 + ((confidence_slider - 60.0) / 40.0) * (1.0 - 0.363)

    with ctrl_c2:
        filter_option = st.selectbox("Search Scope", ["All Active Cases", "Only Cases Registered by Me"])
        filter_user = user if filter_option == "Only Cases Registered by Me" else None

    with ctrl_c3:
        st.write("")
        st.write("")
        run_match_btn = st.button("🚀 Run AI Match Scan", type="primary", use_container_width=True)

    st.divider()

    # Automatically scan or scan on button click
    if run_match_btn or "last_matches" in st.session_state:
        with st.spinner("Running deep facial feature comparison across database..."):
            match_data = match_algo.match(similarity_threshold=sim_threshold, filter_user=filter_user)
            st.session_state["last_matches"] = match_data

        if not match_data["status"]:
            st.error(f"Error during matching: {match_data.get('message', 'Unknown error')}")
        else:
            matches = match_data.get("matches", [])
            tot_reg = match_data.get("total_registered", 0)
            tot_pub = match_data.get("total_public", 0)

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Active Missing Profiles Scanned", tot_reg)
            m2.metric("Public Sightings Scanned", tot_pub)
            m3.metric("Potential AI Matches Found", len(matches))

            st.write("---")

            if not matches:
                st.info(f"🔍 No matches found at **{confidence_slider}%+ confidence threshold**. Try lowering the sensitivity threshold or verifying photo quality.")
            else:
                for idx, match_item in enumerate(matches):
                    conf = match_item["confidence_pct"]
                    sim = match_item["similarity_score"]
                    reg_id = match_item["registered_case_id"]
                    pub_id = match_item["public_case_id"]

                    # Determine color badge based on confidence
                    badge_color = "green" if conf >= 80 else ("orange" if conf >= 65 else "gray")

                    with st.container():
                        st.subheader(f"Match #{idx + 1} — Confidence: {conf}% ({'High' if conf >= 80 else 'Moderate'} Match)")

                        col_left, col_mid, col_right = st.columns([3, 1, 3], gap="medium")

                        # Left Column: Missing Person Registered Profile
                        with col_left:
                            st.markdown("### 👤 Registered Missing Person")
                            if os.path.exists(match_item["reg_image"]):
                                st.image(match_item["reg_image"], caption=f"Missing: {match_item['registered_case_name']}", width=220)
                            else:
                                st.warning("Registered image not found on disk")
                            
                            st.write(f"**Name:** {match_item['registered_case_name']}")
                            st.write(f"**Age:** {match_item['registered_case_age']}")
                            st.write(f"**Last Seen:** {match_item['registered_case_last_seen']}")
                            st.write(f"**Distinct Marks:** {match_item['registered_case_birth_marks']}")
                            st.write(f"**Contact Mobile:** {match_item['registered_case_mobile']}")

                        # Middle Column: Match Metrics & Action
                        with col_mid:
                            st.markdown("### ⚡ AI Match")
                            st.metric("Similarity", f"{conf}%", delta=f"Score: {sim}")
                            st.progress(conf / 100.0)
                            
                            st.write("")
                            if st.button("✅ Confirm & Mark Found", key=f"confirm_{reg_id}_{pub_id}", type="primary"):
                                db_queries.update_found_status(reg_id, pub_id)
                                st.success(f"Case for {match_item['registered_case_name']} marked as FOUND and resolved!")
                                st.session_state.pop("last_matches", None)
                                st.rerun()

                        # Right Column: Sighting Details
                        with col_right:
                            st.markdown("### 📍 Public Sighting / Field Report")
                            if os.path.exists(match_item["pub_image"]):
                                st.image(match_item["pub_image"], caption="Sighted Photo", width=220)
                            else:
                                st.warning("Sighted image not found on disk")

                            st.write(f"**Sighting Location:** {match_item['public_location']}")
                            st.write(f"**Reported By:** {match_item['public_submitted_by']}")
                            st.write(f"**Informant Phone:** {match_item['public_mobile']}")
                            st.write(f"**Date Reported:** {match_item['public_submitted_on']}")

                        st.markdown("---")
else:
    st.write("You don't have access to this page")
