import os
import streamlit as st
from pages.helper import db_queries
from pages.helper.streamlit_helpers import require_login

st.set_page_config(page_title="PathBack - Case Management", page_icon="📂", layout="wide")

if "login_status" not in st.session_state:
    st.warning("Please log in from the Home page to view and manage cases.")

elif st.session_state["login_status"]:
    user = st.session_state.user

    st.title("📂 Missing Person Case Management")
    st.markdown("View, filter, update case resolution status, or delete registered records.")

    # Top Filter & Search Bar
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])
    
    with filter_col1:
        status_filter = st.selectbox(
            "Filter by Category",
            options=["All Registered Cases", "Unresolved (Active)", "Solved (Found)", "Public Sightings Portal"],
        )

    with filter_col2:
        search_query = st.text_input("🔍 Search by Name or Location", placeholder="Type name or keywords...")

    with filter_col3:
        sort_order = st.selectbox("Sort", ["Latest First", "Oldest First"])

    st.divider()

    # Public Cases View
    if status_filter == "Public Sightings Portal":
        cases_data = db_queries.fetch_public_cases(train_data=False, status="All")
        
        if not cases_data:
            st.info("No public sighting submissions found.")
        else:
            st.subheader(f"Total Public Sighting Reports: {len(cases_data)}")
            for pub_case in cases_data:
                case_id = str(pub_case[0])
                status_code = pub_case[1]
                location = pub_case[2] or "Unknown Location"
                mobile = pub_case[3] or "N/A"
                birth_marks = pub_case[4] or "None"
                submitted_on = pub_case[5]
                submitted_by = pub_case[6] or "Anonymous"

                # Search filter
                if search_query and search_query.lower() not in f"{location} {submitted_by} {birth_marks}".lower():
                    continue

                is_solved = (status_code == "F")
                status_label = "✅ Solved / Found" if is_solved else "⏳ Unresolved / Active Sighting"

                with st.container():
                    c_img, c_info, c_actions = st.columns([1, 2, 1], gap="medium")

                    with c_img:
                        img_path = f"./resources/{case_id}.jpg"
                        if os.path.exists(img_path):
                            st.image(img_path, width=160, caption="Sighted Photo")
                        else:
                            st.caption("No photo available")

                    with c_info:
                        st.markdown(f"### Sighting at {location}")
                        st.markdown(f"**Status:** `{status_label}`")
                        st.write(f"**Reported By:** {submitted_by} &nbsp;|&nbsp; **Contact:** {mobile}")
                        st.write(f"**Marks / Details:** {birth_marks}")
                        st.caption(f"Submitted on: {submitted_on}")

                    with c_actions:
                        st.markdown("**Actions:**")
                        if not is_solved:
                            if st.button("✅ Mark as Solved", key=f"solv_pub_{case_id}", use_container_width=True):
                                db_queries.update_public_case_status(case_id, "F")
                                st.success("Marked as Solved!")
                                st.rerun()
                        else:
                            if st.button("⏳ Mark as Unresolved", key=f"unres_pub_{case_id}", use_container_width=True):
                                db_queries.update_public_case_status(case_id, "NF")
                                st.info("Reopened as Unresolved!")
                                st.rerun()

                        if st.button("🗑️ Delete Record", key=f"del_pub_{case_id}", type="secondary", use_container_width=True):
                            db_queries.delete_public_case(case_id)
                            st.warning("Sighting report deleted.")
                            st.rerun()

                    st.markdown("---")

    # Registered Cases View
    else:
        status_map = {
            "All Registered Cases": "All",
            "Unresolved (Active)": "Unresolved",
            "Solved (Found)": "Solved",
        }
        query_status = status_map.get(status_filter, "All")
        registered_cases = db_queries.fetch_registered_cases(submitted_by=user, status=query_status)

        if not registered_cases:
            st.info("No registered missing person cases found under this filter.")
        else:
            # Sort cases
            cases_list = list(registered_cases)
            if sort_order == "Latest First":
                cases_list.reverse()

            st.subheader(f"Total Registered Cases: {len(cases_list)}")

            for case_tuple in cases_list:
                case_id = str(case_tuple[0])
                name = case_tuple[1]
                age = case_tuple[2] or "Unknown"
                status_code = case_tuple[3]
                last_seen = case_tuple[4] or "N/A"
                matched_with_id = case_tuple[5]
                mobile = case_tuple[6] or "N/A"
                submitted_by = case_tuple[7]

                # Search filter
                if search_query and search_query.lower() not in f"{name} {last_seen} {mobile}".lower():
                    continue

                is_solved = (status_code == "F")
                status_badge = "🟢 Solved / Found" if is_solved else "🔴 Unresolved / Missing"

                with st.container():
                    c_img, c_details, c_ops = st.columns([1, 2, 1], gap="medium")

                    with c_img:
                        img_path = f"./resources/{case_id}.jpg"
                        if os.path.exists(img_path):
                            st.image(img_path, width=170, caption=f"Photo of {name}")
                        else:
                            st.caption("No photo on disk")

                    with c_details:
                        st.markdown(f"### {name} (Age: {age})")
                        st.markdown(f"**Case Status:** `{status_badge}`")
                        st.write(f"**Last Seen Location:** {last_seen}")
                        st.write(f"**Complainant / Contact:** {mobile}")
                        st.caption(f"Registered By Officer: {submitted_by} &nbsp;|&nbsp; Case ID: `{case_id}`")

                    with c_ops:
                        st.markdown("**Manage Case:**")
                        
                        # Option 1: Mark Solved
                        if not is_solved:
                            if st.button("✅ 1. Mark as Solved", key=f"mark_solved_{case_id}", type="primary", use_container_width=True):
                                db_queries.update_registered_case_status(case_id, "F")
                                st.success(f"Case for {name} marked as SOLVED!")
                                st.rerun()
                        
                        # Option 2: Mark Unresolved
                        if is_solved:
                            if st.button("⏳ 2. Mark as Unresolved", key=f"mark_unres_{case_id}", use_container_width=True):
                                db_queries.update_registered_case_status(case_id, "NF")
                                st.info(f"Case for {name} reopened as UNRESOLVED!")
                                st.rerun()

                        # Option 3: Delete Case
                        if st.button("🗑️ 3. Delete Case", key=f"delete_{case_id}", type="secondary", use_container_width=True):
                            db_queries.delete_registered_case(case_id)
                            st.warning(f"Case for {name} was permanently deleted.")
                            st.rerun()

                    st.markdown("---")
else:
    st.write("You don't have access to this page")
