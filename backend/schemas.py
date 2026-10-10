"""
Pydantic Schemas for PathBack REST API & Background Workers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -------------------------------------------------------------
# Case Schemas
# -------------------------------------------------------------
class CaseCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="Full name of missing person")
    submitted_by: str = Field(..., min_length=1, max_length=64, description="Reporting officer / user")
    age: Optional[str] = Field(None, description="Reported age or age string")
    age_at_disappearance: Optional[int] = Field(None, ge=0, le=120)
    current_estimated_age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field("Unknown", max_length=32)
    father_name: Optional[str] = Field("", max_length=128)
    complainant_name: Optional[str] = Field("", max_length=128)
    complainant_mobile: Optional[str] = Field("", max_length=20)
    adhaar_card: Optional[str] = Field("", max_length=20)
    last_seen: Optional[str] = Field("", max_length=128)
    last_known_latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    last_known_longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    address: Optional[str] = Field("", max_length=512)
    birth_marks: Optional[str] = Field("", max_length=512)


class CaseStatusUpdateRequest(BaseModel):
    status: str = Field(..., description="Target status: 'NF', 'Active', 'F', 'Solved', 'Under Review', 'Dismissed'")


class CaseResponse(BaseModel):
    id: str
    name: str
    age: Optional[str] = None
    gender: Optional[str] = None
    father_name: Optional[str] = None
    last_seen: Optional[str] = None
    last_known_latitude: Optional[float] = None
    last_known_longitude: Optional[float] = None
    complainant_mobile: Optional[str] = None
    complainant_name: Optional[str] = None
    status: str
    submitted_by: str
    submitted_on: Optional[datetime] = None
    matched_with: Optional[str] = None
    has_photo: bool = False
    image_url: Optional[str] = None


# -------------------------------------------------------------
# Sighting Schemas
# -------------------------------------------------------------
class SightingCreateRequest(BaseModel):
    location: str = Field(..., min_length=1, max_length=256, description="Sighting location description")
    submitted_by: Optional[str] = Field("Anonymous", max_length=128)
    mobile: Optional[str] = Field("", max_length=20)
    email: Optional[str] = Field("", max_length=64)
    birth_marks: Optional[str] = Field("", max_length=512)
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    device_metadata: Optional[str] = Field("", max_length=1024)


class SightingVerificationRequest(BaseModel):
    verified: bool
    verified_by: str = Field(..., min_length=1, max_length=128)


class SightingResponse(BaseModel):
    id: str
    location: Optional[str] = None
    submitted_by: Optional[str] = None
    mobile: Optional[str] = None
    status: str
    submitted_on: Optional[datetime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    verified_by_admin: bool = False
    verified_by: Optional[str] = None
    image_url: Optional[str] = None


# -------------------------------------------------------------
# Matching & Search Schemas
# -------------------------------------------------------------
class MatchCandidateResponse(BaseModel):
    rank: int
    registered_case_id: str
    registered_case_name: str
    registered_case_age: str
    registered_case_last_seen: str
    registered_case_mobile: str
    public_case_id: str
    public_location: str
    public_submitted_by: str
    public_mobile: str
    public_submitted_on: str
    similarity_score: float
    confidence_pct: float
    confidence_tier: str
    distance_metric: str
    search_engine: str
    reg_image_url: str
    pub_image_url: str


class MatchRunResponse(BaseModel):
    status: bool
    search_engine: str
    total_registered: int
    total_public: int
    matches_count: int
    matches: List[MatchCandidateResponse]
    thresholds: Dict[str, float]
    message: Optional[str] = None


class SingleEmbeddingSearchRequest(BaseModel):
    top_k: int = Field(5, ge=1, le=50)
    min_confidence: float = Field(0.50, ge=0.0, le=1.0)
    filter_user: Optional[str] = None


# -------------------------------------------------------------
# Video Surveillance & Analytics Schemas
# -------------------------------------------------------------
class VideoScanAlert(BaseModel):
    track_id: int
    case_id: str
    case_name: str
    confidence_pct: float
    similarity_score: float
    timestamp_sec: float
    frame_number: int
    bbox: List[int]
    thumbnail_url: Optional[str] = None


class VideoScanResponse(BaseModel):
    job_id: str
    total_frames: int
    processed_frames: int
    fps: float
    elapsed_seconds: float
    hardware_backend: str
    alerts_count: int
    alerts: List[VideoScanAlert]


# -------------------------------------------------------------
# System & Health Schemas
# -------------------------------------------------------------
class HealthCheckResponse(BaseModel):
    status: str
    version: str
    faiss_available: bool
    hardware_backend: str
    models_ready: bool
    total_active_cases: int
    total_active_sightings: int
