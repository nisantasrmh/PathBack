import os
import streamlit as st
from pages.helper import db_queries, match_algo
from pages.helper.data_models import ConfidenceTier
from pages.helper.streamlit_helpers import require_login

st.set_page_config(page_title="PathBack - AI Facial Match", page_icon="🎯", layout="wide")

if "login_status" not in st.session_state:
    st.warning("Please log in from the Home page to access AI Case Matching.")

elif st.session_state["login_status"]:
    user = st.session_state.user

    st.title("🎯 AI Deep Face Matching & Ranking Engine")
    st.markdown("Compares registered missing person profiles against public sightings using FAISS vector indexing, ranked strictly by match confidence percentage.")

    # Controls Bar
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 1])
    
    with ctrl_c1:
        confidence_slider = st.slider(
            "Match Sensitivity / Confidence Threshold (%)",
            min_value=40,
            max_value=95,
            value=55,
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
            raw_matches = match_data.get("matches", [])
            tot_reg = match_data.get("total_registered", 0)
            tot_pub = match_data.get("total_public", 0)

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Active Missing Profiles", tot_reg)
            m2.metric("Public Sightings Scanned", tot_pub)
            m3.metric("Candidate Matches Found", len(raw_matches))
            top_score_str = f"{raw_matches[0]['confidence_pct']}%" if raw_matches else "N/A"
            m4.metric("Top Ranked Match %", top_score_str)

            st.write("---")

            if not raw_matches:
                st.info(f"🔍 No candidate matches found at **{confidence_slider}%+ confidence threshold**. Try lowering the sensitivity threshold or verifying photo clarity.")
            else:
                # Top Filter and Sorting Bar
                sort_col, filter_col = st.columns([2, 2])
                with sort_col:
                    sort_order = st.selectbox(
                        "Match Ranking Order",
                        [
                            "Highest Match % First (Descending)",
                            "Lowest Match % First (Ascending)",
                            "Most Recent Sighting First"
                        ]
                    )

                with filter_col:
                    tier_filter = st.selectbox(
                        "Filter by Confidence Tier",
                        ["All Tiers", "High Confidence Only (>=75%)", "Review Needed Only (50%-74%)"]
                    )

                # Filter by Tier
                filtered_matches = list(raw_matches)
                if tier_filter == "High Confidence Only (>=75%)":
                    filtered_matches = [m for m in filtered_matches if m.get("confidence_tier") == ConfidenceTier.HIGH_CONFIDENCE.value or m["confidence_pct"] >= 75.0]
                elif tier_filter == "Review Needed Only (50%-74%)":
                    filtered_matches = [m for m in filtered_matches if m.get("confidence_tier") == ConfidenceTier.REVIEW_NEEDED.value or (50.0 <= m["confidence_pct"] < 75.0)]

                # Apply sorting
                if sort_order == "Highest Match % First (Descending)":
                    filtered_matches.sort(key=lambda x: (x["confidence_pct"], x["similarity_score"]), reverse=True)
                elif sort_order == "Lowest Match % First (Ascending)":
                    filtered_matches.sort(key=lambda x: (x["confidence_pct"], x["similarity_score"]), reverse=False)
                elif sort_order == "Most Recent Sighting First":
                    filtered_matches.sort(key=lambda x: str(x.get("public_submitted_on", "")), reverse=True)

                st.markdown(f"### Showing `{len(filtered_matches)}` Ranked Candidate Matches:")

                for idx, match_item in enumerate(filtered_matches):
                    conf = match_item["confidence_pct"]
                    sim = match_item["similarity_score"]
                    reg_id = match_item["registered_case_id"]
                    pub_id = match_item["public_case_id"]
                    tier = match_item.get("confidence_tier", "Review Needed")

                    with st.container():
                        st.subheader(f"Rank #{idx + 1} — Match Confidence: {conf:.1f}% ({tier})")

                        col_left, col_mid, col_right = st.columns([3, 1.2, 3], gap="medium")

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
                            st.metric("Similarity", f"{conf:.1f}%", delta=f"Score: {sim:.4f}")
                            st.progress(conf / 100.0)
                            st.caption(f"**Tier:** `{tier}`")
                            
                            st.write("")
                            if st.button("✅ Confirm & Mark Found", key=f"confirm_{reg_id}_{pub_id}_{idx}", type="primary", use_container_width=True):
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
