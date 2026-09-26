"""
Biometric Vector Search and Matching Engine for PathBack.

Provides high-throughput sub-millisecond facial similarity search utilizing
FAISS vector indexing (IndexFlatIP / IndexHNSWFlat) with graceful fallback to
vectorized NumPy Cosine Similarity for environments without FAISS or low-volume databases.
"""

from __future__ import annotations

import json
import os
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sqlmodel import Session, select

from pages.helper import db_queries
from pages.helper.data_models import (
    BiometricMatchMetadata,
    ConfidenceTier,
    HIGH_CONFIDENCE_THRESHOLD,
    REVIEW_NEEDED_THRESHOLD,
    SFACE_BASE_COSINE_THRESHOLD,
    PublicSubmissions,
    RegisteredCases,
)
from pages.helper.train_model import (
    EMBEDDING_DIM,
    FAISS_AVAILABLE,
    MIN_CASES_FOR_FAISS,
    l2_normalize,
    BiometricVectorIndex,
)
from pages.helper.utils import (
    extract_deep_face_embeddings,
    compute_similarity,
)

warnings.filterwarnings(action="ignore")


def parse_or_extract_embedding(
    case_id: str,
    face_mesh_json: Optional[str]
) -> Optional[np.ndarray]:
    """
    Parses stored 128-D face embedding from JSON or extracts it on-the-fly from saved image.
    Strictly L2-normalizes the embedding to unit Euclidean length.

    Args:
        case_id: Case unique identifier (used for image fallback lookup).
        face_mesh_json: Serialized JSON string containing embedding.

    Returns:
        Optional[np.ndarray]: L2-normalized float32 128-D vector, or None if extraction fails.
    """
    if face_mesh_json:
        try:
            data = json.loads(face_mesh_json)
            if isinstance(data, dict) and "embedding" in data and len(data["embedding"]) == EMBEDDING_DIM:
                return l2_normalize(data["embedding"])
            elif isinstance(data, list) and len(data) == EMBEDDING_DIM:
                return l2_normalize(data)
        except Exception:
            pass

    # Fallback: extract deep embedding from the saved image file on disk
    img_path = f"./resources/{case_id}.jpg"
    if os.path.exists(img_path):
        res = extract_deep_face_embeddings(img_path)
        if res and "embedding" in res and len(res["embedding"]) == EMBEDDING_DIM:
            return l2_normalize(res["embedding"])

    return None


def calculate_confidence_tier(
    confidence_fraction: float,
    high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    review_threshold: float = REVIEW_NEEDED_THRESHOLD,
) -> str:
    """
    Determines the confidence tier based on normalized confidence probability [0.0 - 1.0].

    Args:
        confidence_fraction: Probability score between 0.0 and 1.0.
        high_threshold: Threshold for High Confidence tier (default 0.75).
        review_threshold: Threshold for Review Needed tier (default 0.50).

    Returns:
        str: ConfidenceTier value ("High Confidence", "Review Needed", "Dismiss").
    """
    if confidence_fraction >= high_threshold:
        return ConfidenceTier.HIGH_CONFIDENCE.value
    elif confidence_fraction >= review_threshold:
        return ConfidenceTier.REVIEW_NEEDED.value
    return ConfidenceTier.DISMISS.value


def cosine_sim_to_confidence(cos_sim: float) -> Tuple[float, float]:
    """
    Converts raw SFace cosine similarity [-1.0, 1.0] to calibrated probability [0.0, 1.0]
    and percentage [0.0, 100.0].

    OpenCV SFace standard verification threshold is ~0.363 (maps to 60% confidence).

    Args:
        cos_sim: Dot product of unit L2-normalized vectors.

    Returns:
        Tuple[float, float]: (confidence_fraction [0.0 - 1.0], confidence_pct [0.0 - 100.0])
    """
    cos_sim = max(-1.0, min(1.0, float(cos_sim)))

    if cos_sim <= 0.0:
        conf_pct = 0.0
    elif cos_sim < SFACE_BASE_COSINE_THRESHOLD:
        conf_pct = (cos_sim / SFACE_BASE_COSINE_THRESHOLD) * 60.0
    else:
        conf_pct = 60.0 + ((cos_sim - SFACE_BASE_COSINE_THRESHOLD) / (1.0 - SFACE_BASE_COSINE_THRESHOLD)) * 40.0

    conf_pct = max(0.0, min(100.0, conf_pct))
    conf_fraction = conf_pct / 100.0
    return conf_fraction, conf_pct


def run_vector_search(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    top_k: int = 5,
    prefer_faiss: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Executes maximum inner product / cosine similarity search against gallery embeddings.
    Uses FAISS CPU (IndexFlatIP) if available and gallery count >= 10, otherwise executes
    vectorized NumPy dot product matrix multiplication.

    Args:
        query_embeddings: (M, 128) float32 matrix of query embeddings (L2-normalized).
        gallery_embeddings: (N, 128) float32 matrix of gallery embeddings (L2-normalized).
        top_k: Number of nearest neighbors to retrieve.
        prefer_faiss: Whether to use FAISS when criteria are met.

    Returns:
        Tuple[np.ndarray, np.ndarray, str]:
            - similarities: (M, K) matrix of cosine similarity scores [-1.0, 1.0]
            - indices: (M, K) matrix of matching row indices in gallery
            - backend: Name of search engine utilized ("FAISS_IndexFlatIP" or "NumPy_Vectorized")
    """
    m_queries = query_embeddings.shape[0]
    n_gallery = gallery_embeddings.shape[0]
    effective_k = min(top_k, n_gallery)

    if n_gallery == 0 or m_queries == 0 or effective_k == 0:
        return np.empty((m_queries, 0), dtype=np.float32), np.empty((m_queries, 0), dtype=np.int64), "EMPTY"

    # Verify L2 normalization
    q_norm = l2_normalize(query_embeddings)
    g_norm = l2_normalize(gallery_embeddings)

    # Use FAISS if available and dataset size >= 10
    if prefer_faiss and FAISS_AVAILABLE and n_gallery >= MIN_CASES_FOR_FAISS:
        try:
            import faiss
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            index.add(g_norm)
            sims, idxs = index.search(q_norm, effective_k)
            return sims, idxs, "FAISS_IndexFlatIP"
        except Exception as e:
            print(f"[match_algo] FAISS search execution failed: {e}. Falling back to NumPy.")

    # Graceful NumPy Vectorized Fallback: (M, 128) @ (128, N) = (M, N)
    sim_matrix = np.matmul(q_norm, g_norm.T)  # Shape (M, N)

    if effective_k >= n_gallery:
        # Sort entire row in descending order
        sorted_indices = np.argsort(-sim_matrix, axis=1)
    else:
        # Use argpartition for fast top-k selection on larger sets
        partitioned = np.argpartition(-sim_matrix, effective_k - 1, axis=1)[:, :effective_k]
        # Sort only top-k
        row_indices = np.arange(m_queries)[:, None]
        part_scores = sim_matrix[row_indices, partitioned]
        top_sorted = np.argsort(-part_scores, axis=1)
        sorted_indices = np.take_along_axis(partitioned, top_sorted, axis=1)

    top_scores = np.take_along_axis(sim_matrix, sorted_indices, axis=1)
    return top_scores, sorted_indices, "NumPy_Vectorized"


def search_single_embedding(
    query_embedding: Union[np.ndarray, List[float]],
    top_k: int = 5,
    min_confidence: float = 0.50,
    filter_user: Optional[str] = None,
) -> List[BiometricMatchMetadata]:
    """
    Performs sub-millisecond 1-to-N biometric vector lookup for a single face embedding
    against all active missing person cases in the database.

    Args:
        query_embedding: 128-D face embedding vector.
        top_k: Maximum candidate matches to return.
        min_confidence: Minimum confidence threshold [0.0 - 1.0].
        filter_user: Optional username filter for registered cases.

    Returns:
        List[BiometricMatchMetadata]: Ranked list of candidate matches.
    """
    norm_query = l2_normalize(query_embedding).reshape(1, EMBEDDING_DIM)

    with Session(db_queries.engine) as session:
        reg_query = select(RegisteredCases).where(RegisteredCases.status == "NF")
        if filter_user:
            reg_query = reg_query.where(RegisteredCases.submitted_by == filter_user)
        registered_cases = session.exec(reg_query).all()

    if not registered_cases:
        return []

    reg_list: List[RegisteredCases] = []
    reg_embs: List[np.ndarray] = []

    for reg in registered_cases:
        emb = parse_or_extract_embedding(reg.id, reg.face_mesh)
        if emb is not None:
            reg_list.append(reg)
            reg_embs.append(emb)

    if not reg_embs:
        return []

    gallery_matrix = np.vstack(reg_embs)
    sims, idxs, engine_name = run_vector_search(norm_query, gallery_matrix, top_k=top_k)

    results: List[BiometricMatchMetadata] = []
    for k in range(sims.shape[1]):
        score = float(sims[0, k])
        reg_idx = int(idxs[0, k])
        matched_case = reg_list[reg_idx]

        conf_frac, conf_pct = cosine_sim_to_confidence(score)
        if conf_frac < min_confidence:
            continue

        tier = calculate_confidence_tier(conf_frac)
        meta = BiometricMatchMetadata(
            registered_case_id=matched_case.id,
            registered_case_name=matched_case.name,
            registered_case_age=str(matched_case.age or "N/A"),
            registered_case_last_seen=str(matched_case.last_seen or "N/A"),
            registered_case_mobile=str(matched_case.complainant_mobile or "N/A"),
            registered_case_birth_marks=str(matched_case.birth_marks or "None"),
            public_case_id="REALTIME_QUERY",
            public_location="Live Query",
            public_submitted_by="Live Scanner",
            public_mobile="N/A",
            public_submitted_on="Real-time",
            similarity_score=round(score, 4),
            confidence_pct=round(conf_pct, 1),
            confidence_tier=tier,
            distance_metric=f"Cosine Similarity ({engine_name})",
            search_engine=engine_name,
            reg_image=f"./resources/{matched_case.id}.jpg",
            pub_image="",
        )
        results.append(meta)

    return results


def match(
    similarity_threshold: float = SFACE_BASE_COSINE_THRESHOLD,
    filter_user: Optional[str] = None,
    high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    review_threshold: float = REVIEW_NEEDED_THRESHOLD,
    dismiss_threshold: float = 0.50,
    top_k_per_sighting: int = 5,
) -> Dict[str, Any]:
    """
    Production-grade biometric matching engine for PathBack.
    
    Compares all unresolved public sighting reports against active registered missing person cases
    using high-throughput vector indexing (FAISS IndexFlatIP) with NumPy vectorized fallback.
    
    Args:
        similarity_threshold: Minimum cosine similarity to consider a match (~0.363 for SFace).
        filter_user: Optional username filter to restrict registered cases.
        high_threshold: Probability threshold for High Confidence tier (default 0.75 / 75%).
        review_threshold: Probability threshold for Review Needed tier (default 0.50 / 50%).
        dismiss_threshold: Probability threshold below which candidates are dismissed (default 0.50).
        top_k_per_sighting: Number of candidate matches evaluated per sighting report.
    
    Returns:
        Dict[str, Any]:
            - "status": bool
            - "matches": List[Dict[str, Any]] (structured match metadata)
            - "total_registered": int
            - "total_public": int
            - "search_engine": str
            - "thresholds": Dict[str, float]
    """
    try:
        db_queries.create_db()
        with Session(db_queries.engine) as session:
            # Fetch registered cases (Not Found)
            reg_query = select(RegisteredCases).where(RegisteredCases.status == "NF")
            if filter_user and filter_user not in ("All", "System Administrator"):
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
                "message": "No active missing person cases or public sightings available for matching.",
                "total_registered": len(registered_cases),
                "total_public": len(pub_cases),
                "search_engine": "NONE",
            }

        # Cache & L2-normalize embeddings for registered cases (Gallery)
        valid_reg_cases: List[RegisteredCases] = []
        reg_embeddings: List[np.ndarray] = []
        for reg in registered_cases:
            emb = parse_or_extract_embedding(reg.id, reg.face_mesh)
            if emb is not None:
                valid_reg_cases.append(reg)
                reg_embeddings.append(emb)

        # Cache & L2-normalize embeddings for public cases (Queries)
        valid_pub_cases: List[PublicSubmissions] = []
        pub_embeddings: List[np.ndarray] = []
        for pub in pub_cases:
            emb = parse_or_extract_embedding(pub.id, pub.face_mesh)
            if emb is not None:
                valid_pub_cases.append(pub)
                pub_embeddings.append(emb)

        if not valid_reg_cases or not valid_pub_cases:
            return {
                "status": True,
                "matches": [],
                "message": "No valid facial embeddings could be parsed from active cases.",
                "total_registered": len(registered_cases),
                "total_public": len(pub_cases),
                "search_engine": "NONE",
            }

        query_matrix = np.vstack(pub_embeddings).astype(np.float32)
        gallery_matrix = np.vstack(reg_embeddings).astype(np.float32)

        # Run vector index search (FAISS IndexFlatIP or NumPy vectorized fallback)
        sim_scores, matched_indices, engine_used = run_vector_search(
            query_embeddings=query_matrix,
            gallery_embeddings=gallery_matrix,
            top_k=min(top_k_per_sighting, len(valid_reg_cases)),
            prefer_faiss=True,
        )

        matched_results: List[Dict[str, Any]] = []

        # Parse match matrices into structured match records
        num_sightings, num_candidates = sim_scores.shape
        for q_idx in range(num_sightings):
            pub_case = valid_pub_cases[q_idx]
            formatted_date = (
                pub_case.submitted_on.strftime("%Y-%m-%d %H:%M")
                if hasattr(pub_case.submitted_on, "strftime")
                else str(pub_case.submitted_on)
            )

            for c_idx in range(num_candidates):
                cos_sim = float(sim_scores[q_idx, c_idx])
                g_idx = int(matched_indices[q_idx, c_idx])
                reg_case = valid_reg_cases[g_idx]

                conf_fraction, conf_pct = cosine_sim_to_confidence(cos_sim)

                # Filter against configured thresholds
                if cos_sim < similarity_threshold or conf_fraction < dismiss_threshold:
                    continue

                tier = calculate_confidence_tier(
                    conf_fraction,
                    high_threshold=high_threshold,
                    review_threshold=review_threshold,
                )

                record = BiometricMatchMetadata(
                    registered_case_id=reg_case.id,
                    registered_case_name=reg_case.name,
                    registered_case_age=str(reg_case.age or "Unknown"),
                    registered_case_last_seen=str(reg_case.last_seen or "N/A"),
                    registered_case_mobile=str(reg_case.complainant_mobile or "N/A"),
                    registered_case_birth_marks=str(reg_case.birth_marks or "None"),
                    public_case_id=pub_case.id,
                    public_location=str(pub_case.location or "Unknown Location"),
                    public_submitted_by=str(pub_case.submitted_by or "Anonymous"),
                    public_mobile=str(pub_case.mobile or "N/A"),
                    public_submitted_on=formatted_date,
                    similarity_score=round(cos_sim, 4),
                    confidence_pct=round(conf_pct, 1),
                    confidence_tier=tier,
                    distance_metric=f"Dot Product / Cosine Similarity ({engine_used})",
                    search_engine=engine_used,
                    reg_image=f"./resources/{reg_case.id}.jpg",
                    pub_image=f"./resources/{pub_case.id}.jpg",
                )
                matched_results.append(record.to_dict())

        # Sort matches by highest confidence percentage first
        matched_results.sort(key=lambda x: x["confidence_pct"], reverse=True)

        return {
            "status": True,
            "matches": matched_results,
            "total_registered": len(registered_cases),
            "total_public": len(pub_cases),
            "search_engine": engine_used,
            "thresholds": {
                "similarity_threshold": similarity_threshold,
                "high_confidence_threshold": high_threshold,
                "review_needed_threshold": review_threshold,
                "dismiss_threshold": dismiss_threshold,
            },
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "status": False,
            "message": str(e),
            "matches": [],
            "search_engine": "ERROR",
        }


if __name__ == "__main__":
    res = match()
    print(f"Match Result (Engine: {res.get('search_engine')}):", len(res.get("matches", [])))
