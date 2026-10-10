"""
PathBack Enterprise FastAPI Backend Server.

Provides RESTful endpoints for:
- Case Registration & Profile Management
- Sighting Tips & Geolocation Submissions
- FAISS Biometric Face Vector Search & Batch Ranking
- Asynchronous CCTV Surveillance Scanner with Alert Feeds
- Geospatial Radius Queries (Haversine)
- Health, Hardware Telemetry, and Background Worker Job Status
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import Session, select

from pages.helper import db_queries, match_algo
from pages.helper.data_models import PublicSubmissions, RegisteredCases
from pages.helper.model_cache import (
    SFACE_PATH,
    YUNET_PATH,
    probe_hardware_backend,
)
from pages.helper.train_model import (
    EMBEDDING_DIM,
    FAISS_AVAILABLE,
    l2_normalize,
)
from pages.helper.utils import (
    extract_deep_face_embeddings,
    extract_face_mesh_landmarks,
    image_obj_to_numpy,
)
from backend.schemas import (
    CaseResponse,
    CaseStatusUpdateRequest,
    HealthCheckResponse,
    MatchCandidateResponse,
    MatchRunResponse,
    SightingResponse,
    SightingVerificationRequest,
    SingleEmbeddingSearchRequest,
)
from backend.worker import (
    run_rebuild_index_job,
    run_video_surveillance_job,
    worker_queue,
)

logger = logging.getLogger("pathback.api")
logging.basicConfig(level=logging.INFO)

# Root project directory & resources
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

app = FastAPI(
    title="PathBack AI Biometric API",
    description="High-performance missing person recovery & facial biometric matching backend powered by FAISS, YuNet, and SFace.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontends (Desktop/Mobile/Streamlit/Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize SQLite database, check models, and run schema migrations."""
    db_queries.create_db()
    logger.info("PathBack Backend API initialized and database ready.")


# ==========================================================
# 1. Health & System Telemetry Endpoints
# ==========================================================
@app.get("/api/health", response_model=HealthCheckResponse, tags=["System"])
def get_health() -> HealthCheckResponse:
    """Returns hardware acceleration state, FAISS status, and active case counts."""
    _, _, hw_backend = probe_hardware_backend()
    models_ready = os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH)

    with Session(db_queries.engine) as session:
        active_cases = len(
            session.exec(select(RegisteredCases).where(RegisteredCases.status.in_(["NF", "Active"]))).all()
        )
        active_sightings = len(
            session.exec(select(PublicSubmissions).where(PublicSubmissions.status.in_(["NF", "Active"]))).all()
        )

    return HealthCheckResponse(
        status="healthy",
        version="2.0.0",
        faiss_available=FAISS_AVAILABLE,
        hardware_backend=hw_backend,
        models_ready=models_ready,
        total_active_cases=active_cases,
        total_active_sightings=active_sightings,
    )


@app.get("/api/resources/{filename}", tags=["Resources"])
def get_resource_file(filename: str):
    """Serves case images and alert thumbnails securely."""
    # Sanitize filename
    safe_name = os.path.basename(filename)
    file_path = os.path.join(RESOURCES_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Requested resource image not found")
    return FileResponse(file_path, media_type="image/jpeg")


# ==========================================================
# 2. Registered Cases Endpoints
# ==========================================================
@app.get("/api/cases", response_model=List[CaseResponse], tags=["Cases"])
def list_cases(
    status_filter: str = Query("All", description="Status filter: 'All', 'Active', 'Solved'"),
    submitted_by: Optional[str] = Query(None, description="Filter by submitting officer"),
    search: Optional[str] = Query(None, description="Keyword search in name, address, or last seen"),
) -> List[CaseResponse]:
    """Retrieves missing person case profiles with optional keyword and status filters."""
    cases_records = db_queries.fetch_registered_cases(submitted_by=submitted_by, status=status_filter)
    results: List[CaseResponse] = []

    for c in cases_records:
        cid, name, age, st, last_seen, matched_with, mobile, submitter = c
        if search:
            kw = search.lower()
            combined = f"{name} {last_seen} {mobile} {submitter}".lower()
            if kw not in combined:
                continue

        has_photo = os.path.exists(os.path.join(RESOURCES_DIR, f"{cid}.jpg"))
        image_url = f"/api/resources/{cid}.jpg" if has_photo else None

        results.append(
            CaseResponse(
                id=cid,
                name=name,
                age=age,
                last_seen=last_seen,
                complainant_mobile=mobile,
                status=st,
                submitted_by=submitter,
                matched_with=matched_with,
                has_photo=has_photo,
                image_url=image_url,
            )
        )
    return results


@app.get("/api/cases/{case_id}", response_model=CaseResponse, tags=["Cases"])
def get_case(case_id: str) -> CaseResponse:
    """Retrieves full missing person record by unique ID."""
    with Session(db_queries.engine) as session:
        case = session.exec(select(RegisteredCases).where(RegisteredCases.id == case_id)).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        has_photo = os.path.exists(os.path.join(RESOURCES_DIR, f"{case.id}.jpg"))
        return CaseResponse(
            id=case.id,
            name=case.name,
            age=case.age,
            gender=case.gender,
            father_name=case.father_name,
            last_seen=case.last_seen,
            last_known_latitude=case.last_known_latitude,
            last_known_longitude=case.last_known_longitude,
            complainant_mobile=case.complainant_mobile,
            complainant_name=case.complainant_name,
            status=case.status,
            submitted_by=case.submitted_by,
            submitted_on=case.submitted_on,
            matched_with=case.matched_with,
            has_photo=has_photo,
            image_url=f"/api/resources/{case.id}.jpg" if has_photo else None,
        )


@app.post("/api/cases", response_model=CaseResponse, status_code=status.HTTP_201_CREATED, tags=["Cases"])
async def register_case(
    name: str = Form(...),
    submitted_by: str = Form(...),
    photo: UploadFile = File(...),
    age: Optional[str] = Form(None),
    gender: Optional[str] = Form("Unknown"),
    father_name: Optional[str] = Form(""),
    complainant_name: Optional[str] = Form(""),
    complainant_mobile: Optional[str] = Form(""),
    adhaar_card: Optional[str] = Form(""),
    last_seen: Optional[str] = Form(""),
    last_known_latitude: Optional[float] = Form(None),
    last_known_longitude: Optional[float] = Form(None),
    address: Optional[str] = Form(""),
    birth_marks: Optional[str] = Form(""),
) -> CaseResponse:
    """
    Registers a new missing person profile, saves their photograph, and extracts
    128-D SFace biometric feature embeddings and facial landmarks.
    """
    photo_bytes = await photo.read()
    if not photo_bytes:
        raise HTTPException(status_code=400, detail="Empty photo uploaded")

    # Decode image to numpy array
    np_arr = np.frombuffer(photo_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Biometric feature extraction
    face_detection = extract_deep_face_embeddings(img_rgb)
    face_mesh = extract_face_mesh_landmarks(img_rgb)

    if face_detection:
        face_data = {
            "embedding": face_detection["embedding"],
            "bbox": face_detection["bbox"],
            "mesh": face_mesh if face_mesh else [],
        }
        primary_vector = json.dumps(face_detection["embedding"])
    else:
        face_data = {"embedding": [], "bbox": [], "mesh": []}
        primary_vector = None

    case_id = str(uuid.uuid4())
    save_path = os.path.join(RESOURCES_DIR, f"{case_id}.jpg")
    with open(save_path, "wb") as f:
        f.write(photo_bytes)

    new_case = RegisteredCases(
        id=case_id,
        submitted_by=submitted_by,
        name=name,
        father_name=father_name or "",
        age=age or "",
        gender=gender or "Unknown",
        complainant_name=complainant_name or "",
        complainant_mobile=complainant_mobile or "",
        adhaar_card=adhaar_card or "",
        last_seen=last_seen or "",
        last_known_latitude=last_known_latitude,
        last_known_longitude=last_known_longitude,
        address=address or "",
        birth_marks=birth_marks or "",
        face_mesh=json.dumps(face_data),
        primary_photo_vector=primary_vector,
        status="NF",
    )

    db_queries.register_new_case(new_case)

    return CaseResponse(
        id=new_case.id,
        name=new_case.name,
        age=new_case.age,
        gender=new_case.gender,
        father_name=new_case.father_name,
        last_seen=new_case.last_seen,
        last_known_latitude=new_case.last_known_latitude,
        last_known_longitude=new_case.last_known_longitude,
        complainant_mobile=new_case.complainant_mobile,
        complainant_name=new_case.complainant_name,
        status=new_case.status,
        submitted_by=new_case.submitted_by,
        submitted_on=new_case.submitted_on,
        has_photo=True,
        image_url=f"/api/resources/{case_id}.jpg",
    )


@app.patch("/api/cases/{case_id}/status", tags=["Cases"])
def update_case_status(case_id: str, payload: CaseStatusUpdateRequest) -> Dict[str, Any]:
    """Updates case resolution status (Active/NF, Solved/F, Under Review)."""
    success = db_queries.update_registered_case_status(case_id, payload.status)
    if not success:
        raise HTTPException(status_code=404, detail="Case not found or status update failed")
    return {"status": "success", "case_id": case_id, "new_status": payload.status}


@app.delete("/api/cases/{case_id}", tags=["Cases"])
def delete_case(case_id: str) -> Dict[str, Any]:
    """Deletes registered case record and linked image."""
    success = db_queries.delete_registered_case(case_id)
    if not success:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"status": "success", "message": f"Case {case_id} deleted successfully"}


# ==========================================================
# 3. Public Sightings Endpoints
# ==========================================================
@app.get("/api/sightings", response_model=List[SightingResponse], tags=["Sightings"])
def list_sightings(
    status_filter: str = Query("All", description="'All', 'Active'/'NF', 'Solved'/'F'")
) -> List[SightingResponse]:
    """Fetches public sighting reports."""
    records = db_queries.fetch_public_cases(train_data=False, status=status_filter)
    sightings: List[SightingResponse] = []

    for sid, st, loc, mob, marks, sub_on, sub_by in records:
        has_photo = os.path.exists(os.path.join(RESOURCES_DIR, f"{sid}.jpg"))
        sightings.append(
            SightingResponse(
                id=str(sid),
                status=st,
                location=loc,
                mobile=mob,
                submitted_on=sub_on,
                submitted_by=sub_by,
                image_url=f"/api/resources/{sid}.jpg" if has_photo else None,
            )
        )
    return sightings


@app.post("/api/sightings", response_model=SightingResponse, status_code=status.HTTP_201_CREATED, tags=["Sightings"])
async def submit_sighting(
    location: str = Form(...),
    photo: UploadFile = File(...),
    submitted_by: Optional[str] = Form("Anonymous"),
    mobile: Optional[str] = Form(""),
    email: Optional[str] = Form(""),
    birth_marks: Optional[str] = Form(""),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    device_metadata: Optional[str] = Form(""),
) -> SightingResponse:
    """Submits a field sighting report with photo and instant biometric extraction."""
    photo_bytes = await photo.read()
    if not photo_bytes:
        raise HTTPException(status_code=400, detail="Empty photo uploaded")

    np_arr = np.frombuffer(photo_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_detection = extract_deep_face_embeddings(img_rgb)
    face_mesh = extract_face_mesh_landmarks(img_rgb)

    if face_detection:
        face_data = {
            "embedding": face_detection["embedding"],
            "bbox": face_detection["bbox"],
            "mesh": face_mesh if face_mesh else [],
        }
        confidence_val = float(face_detection.get("confidence", 0.0))
    else:
        face_data = {"embedding": [], "bbox": [], "mesh": []}
        confidence_val = None

    sighting_id = str(uuid.uuid4())
    save_path = os.path.join(RESOURCES_DIR, f"{sighting_id}.jpg")
    with open(save_path, "wb") as f:
        f.write(photo_bytes)

    sighting = PublicSubmissions(
        id=sighting_id,
        submitted_by=submitted_by or "Anonymous",
        location=location,
        mobile=mobile or "",
        email=email or "",
        birth_marks=birth_marks or "",
        latitude=latitude,
        longitude=longitude,
        confidence_score=confidence_val,
        device_metadata=device_metadata or "",
        face_mesh=json.dumps(face_data),
        status="NF",
    )

    db_queries.new_public_case(sighting)

    return SightingResponse(
        id=sighting.id,
        location=sighting.location,
        submitted_by=sighting.submitted_by,
        mobile=sighting.mobile,
        status=sighting.status,
        submitted_on=sighting.submitted_on,
        latitude=sighting.latitude,
        longitude=sighting.longitude,
        verified_by_admin=False,
        image_url=f"/api/resources/{sighting_id}.jpg",
    )


@app.patch("/api/sightings/{sighting_id}/verify", tags=["Sightings"])
def verify_sighting(sighting_id: str, payload: SightingVerificationRequest) -> Dict[str, Any]:
    """Records administrative audit verification on a citizen sighting tip."""
    success = db_queries.update_sighting_verification(
        sighting_id=sighting_id,
        verified=payload.verified,
        verified_by=payload.verified_by,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Sighting not found")
    return {"status": "success", "sighting_id": sighting_id, "verified": payload.verified}


# ==========================================================
# 4. Biometric Vector Search & Matching Engine
# ==========================================================
@app.post("/api/match/run", response_model=MatchRunResponse, tags=["Biometrics"])
def run_batch_match(
    similarity_threshold: float = Query(0.363, ge=0.1, le=1.0),
    high_threshold: float = Query(0.75, ge=0.5, le=1.0),
    review_threshold: float = Query(0.50, ge=0.3, le=0.8),
    filter_user: Optional[str] = Query(None),
) -> MatchRunResponse:
    """
    Executes full database cross-matching comparing public sightings against active missing persons.
    Ranked strictly by highest match confidence percentage.
    """
    result = match_algo.match(
        similarity_threshold=similarity_threshold,
        filter_user=filter_user,
        high_threshold=high_threshold,
        review_threshold=review_threshold,
    )

    if not result.get("status"):
        raise HTTPException(status_code=500, detail=result.get("message", "Matching failed"))

    parsed_candidates: List[MatchCandidateResponse] = []
    for m in result.get("matches", []):
        parsed_candidates.append(
            MatchCandidateResponse(
                rank=m["rank"],
                registered_case_id=m["registered_case_id"],
                registered_case_name=m["registered_case_name"],
                registered_case_age=m["registered_case_age"],
                registered_case_last_seen=m["registered_case_last_seen"],
                registered_case_mobile=m["registered_case_mobile"],
                public_case_id=m["public_case_id"],
                public_location=m["public_location"],
                public_submitted_by=m["public_submitted_by"],
                public_mobile=m["public_mobile"],
                public_submitted_on=m["public_submitted_on"],
                similarity_score=m["similarity_score"],
                confidence_pct=m["confidence_pct"],
                confidence_tier=m["confidence_tier"],
                distance_metric=m["distance_metric"],
                search_engine=m["search_engine"],
                reg_image_url=f"/api/resources/{m['registered_case_id']}.jpg",
                pub_image_url=f"/api/resources/{m['public_case_id']}.jpg",
            )
        )

    return MatchRunResponse(
        status=True,
        search_engine=result.get("search_engine", "UNKNOWN"),
        total_registered=result.get("total_registered", 0),
        total_public=result.get("total_public", 0),
        matches_count=len(parsed_candidates),
        matches=parsed_candidates,
        thresholds=result.get("thresholds", {}),
        message=result.get("message"),
    )


@app.post("/api/match/search-face", tags=["Biometrics"])
async def search_by_photo(
    photo: UploadFile = File(...),
    top_k: int = Form(5),
    min_confidence: float = Form(0.50),
    filter_user: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Sub-millisecond 1-to-N biometric vector lookup from an uploaded face image.
    """
    photo_bytes = await photo.read()
    np_arr = np.frombuffer(photo_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid photo")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_detection = extract_deep_face_embeddings(img_rgb)
    if not face_detection or "embedding" not in face_detection:
        raise HTTPException(status_code=400, detail="No detectable face in query image")

    query_emb = face_detection["embedding"]
    matches = match_algo.search_single_embedding(
        query_embedding=query_emb,
        top_k=top_k,
        min_confidence=min_confidence,
        filter_user=filter_user,
    )

    formatted = []
    for idx, m in enumerate(matches, start=1):
        d = m.to_dict()
        d["rank"] = idx
        d["reg_image_url"] = f"/api/resources/{m.registered_case_id}.jpg"
        formatted.append(d)

    return {
        "status": "success",
        "matches_count": len(formatted),
        "matches": formatted,
    }


# ==========================================================
# 5. Geospatial Radius Filter Endpoints (Haversine)
# ==========================================================
@app.get("/api/geo/sightings-nearby", tags=["Geospatial"])
def get_nearby_sightings(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    radius_km: float = Query(50.0, ge=1.0, le=500.0),
    status: str = Query("All"),
) -> List[Dict[str, Any]]:
    """Retrieves all sightings within a specified kilometer radius sorted by proximity."""
    sightings = db_queries.fetch_sightings_within_radius(lat, lon, radius_km=radius_km, status=status)
    return [
        {
            "sighting_id": s.id,
            "location": s.location,
            "distance_km": dist,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "submitted_on": s.submitted_on.isoformat() if hasattr(s.submitted_on, "isoformat") else str(s.submitted_on),
            "status": s.status,
            "image_url": f"/api/resources/{s.id}.jpg",
        }
        for s, dist in sightings
    ]


@app.get("/api/geo/cases-nearby", tags=["Geospatial"])
def get_nearby_cases(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    radius_km: float = Query(50.0, ge=1.0, le=500.0),
    status: str = Query("All"),
) -> List[Dict[str, Any]]:
    """Retrieves all missing person cases last seen within a kilometer radius."""
    cases = db_queries.fetch_cases_within_radius(lat, lon, radius_km=radius_km, status=status)
    return [
        {
            "case_id": c.id,
            "name": c.name,
            "age": c.age,
            "distance_km": dist,
            "last_known_latitude": c.last_known_latitude,
            "last_known_longitude": c.last_known_longitude,
            "last_seen": c.last_seen,
            "status": c.status,
            "image_url": f"/api/resources/{c.id}.jpg",
        }
        for c, dist in cases
    ]


# ==========================================================
# 6. Background Surveillance Scanner & Worker Jobs
# ==========================================================
@app.post("/api/scanner/submit-video", tags=["Surveillance Scanner"])
async def submit_surveillance_video(
    video: UploadFile = File(...),
    confidence_threshold: float = Form(60.0),
    frame_skip: int = Form(5),
    enable_motion_gating: bool = Form(True),
    min_motion_pct: float = Form(0.5),
) -> Dict[str, Any]:
    """
    Submits a surveillance / CCTV video for background AI processing with MOG2 motion gating,
    IoU multi-face tracking, and alert deduplication. Returns a background Job ID.
    """
    # Save video to temporary file
    suffix = os.path.splitext(video.filename)[-1] or ".mp4"
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(video.file, temp_video)
    temp_video.close()

    job_id = worker_queue.submit_job(
        job_type="VIDEO_SURVEILLANCE_SCAN",
        fn=run_video_surveillance_job,
        video_path=temp_video.name,
        confidence_threshold=confidence_threshold,
        frame_skip=frame_skip,
        enable_motion_gating=enable_motion_gating,
        min_motion_pct=min_motion_pct,
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "message": "Video surveillance scanning job successfully dispatched to background worker.",
        "poll_url": f"/api/jobs/{job_id}",
    }


@app.post("/api/index/rebuild", tags=["System"])
def rebuild_faiss_index() -> Dict[str, Any]:
    """Dispatches background rebuild of the FAISS vector index."""
    job_id = worker_queue.submit_job(
        job_type="FAISS_INDEX_REBUILD",
        fn=run_rebuild_index_job,
    )
    return {
        "status": "queued",
        "job_id": job_id,
        "message": "FAISS vector index rebuild job dispatched.",
        "poll_url": f"/api/jobs/{job_id}",
    }


@app.get("/api/jobs/{job_id}", tags=["Background Worker"])
def get_job_status(job_id: str) -> Dict[str, Any]:
    """Queries background worker task status and results."""
    job = worker_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")

    return {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "status": job.status.value,
        "progress_pct": round(job.progress_pct, 1),
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "result": job.result,
        "error": job.error,
    }


@app.get("/api/jobs", tags=["Background Worker"])
def list_jobs(limit: int = 20) -> List[Dict[str, Any]]:
    """Lists recent background tasks with execution status."""
    return worker_queue.list_jobs(limit=limit)
