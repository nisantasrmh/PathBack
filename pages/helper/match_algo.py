import os
import json
import traceback
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from sqlmodel import Session, select

from pages.helper import db_queries
from pages.helper.data_models import RegisteredCases, PublicSubmissions
from pages.helper.utils import (
    extract_deep_face_embeddings,
    compute_similarity,
    image_obj_to_numpy,
)

warnings.filterwarnings(action="ignore")


def parse_or_extract_embedding(case_id: str, face_mesh_json: str):
    """
    Parses embedding from stored JSON string or extracts deep SFace embedding from image if needed.
    """
    try:
        data = json.loads(face_mesh_json)
        if isinstance(data, dict) and "embedding" in data:
            return data["embedding"]
        elif isinstance(data, list) and len(data) == 128:
            return data
    except Exception:
        pass

    # Fallback: extract deep embedding from the saved image file
    img_path = f"./resources/{case_id}.jpg"
    if os.path.exists(img_path):
        res = extract_deep_face_embeddings(img_path)
        if res and "embedding" in res:
            return res["embedding"]

    return None


def match(similarity_threshold=0.363, filter_user=None):
    """
    Matches public sightings against registered missing person cases using Deep Face Metric Embeddings (SFace).
    
    Args:
        similarity_threshold: float (default 0.363, corresponds to ~60% match confidence)
        filter_user: optional username filter for registered cases
    
    Returns:
        dict: {"status": bool, "matches": list of match objects, "total_scanned": int}
    """
    try:
        with Session(db_queries.engine) as session:
            # Fetch registered cases (Not Found)
            reg_query = select(RegisteredCases).where(RegisteredCases.status == "NF")
            if filter_user:
                reg_query = reg_query.where(RegisteredCases.submitted_by == filter_user)
            registered_cases = session.exec(reg_query).all()

            # Fetch public sighting submissions (Not Found)
            pub_cases = session.exec(
                select(PublicSubmissions).where(PublicSubmissions.status == "NF")
            ).all()

        if not registered_cases or not pub_cases:
            return {
                "status": True,
                "matches": [],
                "message": "No active missing person cases or public sightings to compare.",
                "total_registered": len(registered_cases),
                "total_public": len(pub_cases)
            }

        # Cache embeddings for registered cases
        reg_embeddings = []
        for reg in registered_cases:
            emb = parse_or_extract_embedding(reg.id, reg.face_mesh)
            if emb:
                reg_embeddings.append({"case": reg, "embedding": emb})

        # Cache embeddings for public cases
        pub_embeddings = []
        for pub in pub_cases:
            emb = parse_or_extract_embedding(pub.id, pub.face_mesh)
            if emb:
                pub_embeddings.append({"case": pub, "embedding": emb})

        matched_results = []

        # Perform pairwise deep cosine similarity comparisons
        for pub_item in pub_embeddings:
            pub_case = pub_item["case"]
            pub_emb = pub_item["embedding"]

            best_match_for_pub = None
            highest_sim = -1.0

            for reg_item in reg_embeddings:
                reg_case = reg_item["case"]
                reg_emb = reg_item["embedding"]

                cos_sim, confidence_pct = compute_similarity(pub_emb, reg_emb)

                if cos_sim >= similarity_threshold:
                    match_record = {
                        "registered_case_id": reg_case.id,
                        "registered_case_name": reg_case.name,
                        "registered_case_age": reg_case.age,
                        "registered_case_last_seen": reg_case.last_seen,
                        "registered_case_mobile": reg_case.complainant_mobile,
                        "registered_case_birth_marks": reg_case.birth_marks,
                        "public_case_id": pub_case.id,
                        "public_location": pub_case.location,
                        "public_submitted_by": pub_case.submitted_by,
                        "public_mobile": pub_case.mobile,
                        "public_submitted_on": pub_case.submitted_on.strftime("%Y-%m-%d %H:%M") if hasattr(pub_case.submitted_on, 'strftime') else str(pub_case.submitted_on),
                        "similarity_score": round(float(cos_sim), 4),
                        "confidence_pct": round(float(confidence_pct), 1),
                        "reg_image": f"./resources/{reg_case.id}.jpg",
                        "pub_image": f"./resources/{pub_case.id}.jpg"
                    }
                    matched_results.append(match_record)

        # Sort matches by highest confidence first
        matched_results.sort(key=lambda x: x["confidence_pct"], reverse=True)

        return {
            "status": True,
            "matches": matched_results,
            "total_registered": len(registered_cases),
            "total_public": len(pub_cases)
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": False, "message": str(e), "matches": []}


if __name__ == "__main__":
    res = match()
    print("Match result:", res)
