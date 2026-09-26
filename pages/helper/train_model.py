"""
Biometric Vector Index Training and Persistence Engine.

This module builds and manages high-throughput, production-grade vector search indices
(FAISS IndexFlatIP / IndexHNSWFlat) for 128-D SFace biometric embeddings with graceful
NumPy matrix fallback for low-data volumes or environments lacking native FAISS binaries.
"""

from __future__ import annotations

import json
import os
import pickle
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sqlmodel import Session, select

from pages.helper import db_queries
from pages.helper.data_models import RegisteredCases

# Check for FAISS availability (e.g., faiss-cpu)
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE: bool = True
except (ImportError, ModuleNotFoundError):
    faiss = None
    FAISS_AVAILABLE = False

# Default file paths for persisted index & metadata
DEFAULT_INDEX_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models"
)
FAISS_INDEX_PATH = os.path.join(DEFAULT_INDEX_DIR, "biometric_index.faiss")
INDEX_METADATA_PATH = os.path.join(DEFAULT_INDEX_DIR, "biometric_index_meta.pkl")
LEGACY_MODEL_PATH = "classifier.pkl"

EMBEDDING_DIM = 128
MIN_CASES_FOR_FAISS = 10


def l2_normalize(
    vectors: Union[np.ndarray, List[float], List[List[float]]]
) -> np.ndarray:
    """
    Strictly L2-normalizes 128-D embedding vector(s) to the unit hypersphere.
    
    Ensures that Dot Product / Inner Product is mathematically identical to Cosine Similarity.
    
    Args:
        vectors: 1D array of shape (128,) or 2D array of shape (N, 128).
        
    Returns:
        np.ndarray: L2-normalized float32 numpy array with unit Euclidean norm.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dimension {EMBEDDING_DIM}, got {arr.shape[0]}")
        norm = np.linalg.norm(arr)
        return (arr / norm).astype(np.float32) if norm > 1e-12 else arr.astype(np.float32)
    elif arr.ndim == 2:
        if arr.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dimension {EMBEDDING_DIM}, got {arr.shape[1]}")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return (arr / norms).astype(np.float32)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


def parse_embedding_from_mesh_json(face_mesh_json: str) -> Optional[np.ndarray]:
    """
    Parses and extracts 128-D face embedding from stored JSON string and returns
    an L2-normalized float32 vector.
    
    Args:
        face_mesh_json: Serialized JSON containing embedding list or dict.
        
    Returns:
        Optional[np.ndarray]: Normalized (128,) float32 array if valid, else None.
    """
    if not face_mesh_json:
        return None
    try:
        data = json.loads(face_mesh_json)
        emb = None
        if isinstance(data, dict) and "embedding" in data:
            emb = data["embedding"]
        elif isinstance(data, list) and len(data) == EMBEDDING_DIM:
            emb = data
            
        if emb and len(emb) == EMBEDDING_DIM:
            return l2_normalize(emb)
    except Exception:
        pass
    return None


def get_train_data(
    submitted_by: Optional[str] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Fetches active (Not Found / Unresolved) registered case records from the SQLite database
    and returns corresponding case IDs and strictly L2-normalized 128-D embedding matrix.

    Args:
        submitted_by: Optional filter to restrict cases to a specific officer/user.

    Returns:
        Tuple[List[str], np.ndarray]:
            - List of case IDs (str)
            - 2D numpy array of shape (N, 128), dtype=float32
    """
    try:
        db_queries.create_db()
        with Session(db_queries.engine) as session:
            query = select(RegisteredCases.id, RegisteredCases.face_mesh).where(
                RegisteredCases.status == "NF"
            )
            if submitted_by and submitted_by not in ("All", "System Administrator"):
                query = query.where(RegisteredCases.submitted_by == submitted_by)
            records = session.exec(query).all()

        case_ids: List[str] = []
        embeddings: List[np.ndarray] = []

        for case_id, face_mesh in records:
            emb = parse_embedding_from_mesh_json(face_mesh)
            if emb is not None:
                case_ids.append(str(case_id))
                embeddings.append(emb)

        if not embeddings:
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        return case_ids, np.vstack(embeddings).astype(np.float32)

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to fetch training data: {e}") from e


class BiometricVectorIndex:
    """
    Production-grade biometric vector index wrapper supporting FAISS CPU indices
    (IndexFlatIP and IndexHNSWFlat) with automatic fallback to vectorized NumPy Cosine Search.
    """

    def __init__(
        self,
        index_type: str = "IndexFlatIP",
        hnsw_m: int = 32,
        index_path: str = FAISS_INDEX_PATH,
        meta_path: str = INDEX_METADATA_PATH,
    ) -> None:
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.index_path = index_path
        self.meta_path = meta_path
        self.case_ids: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self.backend: str = "NONE"
        self.faiss_index: Any = None

    def build(
        self,
        case_ids: List[str],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Builds the vector index from normalized embeddings.
        
        Args:
            case_ids: List of unique case identifiers corresponding to rows.
            embeddings: (N, 128) float32 matrix of L2-normalized vectors.
            
        Returns:
            Dict[str, Any]: Build summary and index characteristics.
        """
        num_cases = len(case_ids)
        self.case_ids = case_ids
        
        if num_cases == 0:
            self.embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            self.backend = "EMPTY"
            self.faiss_index = None
            return {
                "status": True,
                "backend": self.backend,
                "total_indexed": 0,
                "message": "Empty index built (0 cases).",
            }

        # Normalize matrix
        norm_embeddings = l2_normalize(embeddings)
        self.embeddings = norm_embeddings

        # Determine backend: Use FAISS if available and dataset >= 10, else NumPy fallback
        if FAISS_AVAILABLE and num_cases >= MIN_CASES_FOR_FAISS:
            try:
                if self.index_type == "IndexHNSWFlat":
                    # HNSW Graph with Inner Product metric for cosine similarity
                    idx = faiss.IndexHNSWFlat(EMBEDDING_DIM, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
                    idx.hnsw.efSearch = 64
                    idx.hnsw.efConstruction = 64
                else:
                    # IndexFlatIP (Exact Maximum Inner Product Search / Cosine Similarity)
                    idx = faiss.IndexFlatIP(EMBEDDING_DIM)

                idx.add(norm_embeddings)
                self.faiss_index = idx
                self.backend = f"FAISS_{self.index_type}"
            except Exception as ex:
                # Fallback to NumPy on unexpected FAISS failure
                print(f"[BiometricVectorIndex] FAISS build failed, falling back to NumPy: {ex}")
                self.backend = "NUMPY_FALLBACK"
                self.faiss_index = None
        else:
            # Standard NumPy fallback (< 10 cases or FAISS unavailable)
            self.backend = "NUMPY_FALLBACK"
            self.faiss_index = None

        return {
            "status": True,
            "backend": self.backend,
            "total_indexed": num_cases,
            "embedding_dim": EMBEDDING_DIM,
            "message": f"Index successfully created using {self.backend} backend.",
        }

    def save(self) -> bool:
        """
        Persists the index and metadata to disk.
        """
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS binary index if using FAISS
            if self.faiss_index is not None and FAISS_AVAILABLE:
                faiss.write_index(self.faiss_index, self.index_path)
            elif os.path.exists(self.index_path):
                # Remove stale FAISS index file if falling back to NumPy
                try:
                    os.remove(self.index_path)
                except Exception:
                    pass

            # Save metadata and fallback embeddings matrix
            meta_data = {
                "case_ids": self.case_ids,
                "embeddings": self.embeddings,
                "backend": self.backend,
                "index_type": self.index_type,
                "embedding_dim": EMBEDDING_DIM,
                "total_cases": len(self.case_ids),
            }
            with open(self.meta_path, "wb") as f:
                pickle.dump(meta_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            return True
        except Exception as e:
            traceback.print_exc()
            return False

    def load(self) -> bool:
        """
        Loads the persisted index and metadata from disk.
        """
        try:
            if not os.path.exists(self.meta_path):
                return False

            with open(self.meta_path, "rb") as f:
                meta = pickle.load(f)

            self.case_ids = meta.get("case_ids", [])
            self.embeddings = meta.get("embeddings", np.empty((0, EMBEDDING_DIM), dtype=np.float32))
            self.backend = meta.get("backend", "NUMPY_FALLBACK")
            self.index_type = meta.get("index_type", "IndexFlatIP")

            # Load FAISS index if applicable
            if FAISS_AVAILABLE and os.path.exists(self.index_path) and len(self.case_ids) >= MIN_CASES_FOR_FAISS:
                self.faiss_index = faiss.read_index(self.index_path)
                self.backend = f"FAISS_{self.index_type}"
            else:
                self.faiss_index = None
                self.backend = "NUMPY_FALLBACK"

            return True
        except Exception as e:
            print(f"[BiometricVectorIndex] Load error: {e}")
            return False


def train(
    submitted_by: Optional[str] = None,
    index_type: str = "IndexFlatIP"
) -> Dict[str, Any]:
    """
    Main training and indexing routine for the PathBack facial search system.
    Replaces legacy KNN BallTree with production FAISS index and NumPy fallback.

    Args:
        submitted_by: Optional user filter for registered cases.
        index_type: Vector index type ("IndexFlatIP" or "IndexHNSWFlat").

    Returns:
        Dict[str, Any]: Operation status, backend used, total cases indexed, and message.
    """
    # Clean up legacy pickle model if present
    if os.path.isfile(LEGACY_MODEL_PATH):
        try:
            os.remove(LEGACY_MODEL_PATH)
        except Exception:
            pass

    try:
        case_ids, embeddings = get_train_data(submitted_by)
        
        if len(case_ids) == 0:
            return {
                "status": False,
                "backend": "NONE",
                "total_indexed": 0,
                "message": "No active missing person cases available to index.",
            }

        indexer = BiometricVectorIndex(index_type=index_type)
        build_result = indexer.build(case_ids, embeddings)
        save_status = indexer.save()

        if not save_status:
            return {
                "status": False,
                "backend": build_result.get("backend", "UNKNOWN"),
                "total_indexed": len(case_ids),
                "message": "Index built successfully in-memory but failed to save to disk.",
            }

        return {
            "status": True,
            "backend": build_result.get("backend", "UNKNOWN"),
            "total_indexed": len(case_ids),
            "message": f"Biometric Index refreshed ({build_result.get('backend')}, {len(case_ids)} records).",
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "status": False,
            "backend": "ERROR",
            "total_indexed": 0,
            "message": f"Indexing error: {str(e)}",
        }


if __name__ == "__main__":
    result = train()
    print("Train / Indexing Result:", result)
