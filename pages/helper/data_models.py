"""
Data Models and Biometric Schemas for PathBack.

Defines SQLModel ORM entities for registered cases and public sightings with support for
multi-modal features, geolocation attributes, admin audit verification, and biometric metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel, create_engine


# ==========================================================
# Biometric Confidence Thresholds & Status Tiers
# ==========================================================
class ConfidenceTier(str, Enum):
    HIGH_CONFIDENCE = "High Confidence"      # Prob >= 0.75 (75%+)
    REVIEW_NEEDED = "Review Needed"          # 0.50 <= Prob < 0.75 (50% - 74%)
    DISMISS = "Dismiss"                      # Prob < 0.50 (< 50%)


class CaseStatus(str, Enum):
    ACTIVE = "Active"                  # Open / Unresolved missing person record (legacy 'NF')
    SOLVED = "Solved"                  # Found / Resolved case (legacy 'F')
    UNDER_REVIEW = "Under Review"      # Candidate match pending administrative verification
    DISMISSED = "Dismissed"            # Inactive or closed without resolution


HIGH_CONFIDENCE_THRESHOLD: float = 0.75
REVIEW_NEEDED_THRESHOLD: float = 0.50
SFACE_BASE_COSINE_THRESHOLD: float = 0.363  # Standard OpenCV SFace verification threshold (~60% confidence)


@dataclass
class BiometricMatchMetadata:
    """Structured biometric metadata returned for each candidate match."""
    registered_case_id: str
    registered_case_name: str
    registered_case_age: str
    registered_case_last_seen: str
    registered_case_mobile: str
    registered_case_birth_marks: str
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
    reg_image: str
    pub_image: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registered_case_id": self.registered_case_id,
            "registered_case_name": self.registered_case_name,
            "registered_case_age": self.registered_case_age,
            "registered_case_last_seen": self.registered_case_last_seen,
            "registered_case_mobile": self.registered_case_mobile,
            "registered_case_birth_marks": self.registered_case_birth_marks,
            "public_case_id": self.public_case_id,
            "public_location": self.public_location,
            "public_submitted_by": self.public_submitted_by,
            "public_mobile": self.public_mobile,
            "public_submitted_on": self.public_submitted_on,
            "similarity_score": self.similarity_score,
            "confidence_pct": self.confidence_pct,
            "confidence_tier": self.confidence_tier,
            "distance_metric": self.distance_metric,
            "search_engine": self.search_engine,
            "reg_image": self.reg_image,
            "pub_image": self.pub_image,
        }


# ==========================================================
# Database Table Models
# ==========================================================
class PublicSubmissions(SQLModel, table=True):
    """
    Public Sighting entity representing field reports, citizen tips, and camera captures.
    """
    __table_args__ = {"extend_existing": True}

    id: str = Field(
        primary_key=True, default_factory=lambda: str(uuid4()), nullable=False
    )
    submitted_by: Optional[str] = Field(default="Anonymous", max_length=128, nullable=True)
    face_mesh: str = Field(nullable=False)  # JSON string of deep embeddings, bbox, and landmarks
    location: Optional[str] = Field(default="", max_length=128, nullable=True)
    address_text: Optional[str] = Field(default="", max_length=256, nullable=True)
    
    # Geolocation Coordinates
    latitude: Optional[float] = Field(default=None, nullable=True)
    longitude: Optional[float] = Field(default=None, nullable=True)
    
    # Sighting Intelligence & Audit
    confidence_score: Optional[float] = Field(default=None, nullable=True)
    device_metadata: Optional[str] = Field(default="", max_length=1024, nullable=True)
    verified_by_admin: bool = Field(default=False, nullable=False)
    verified_by: Optional[str] = Field(default="", max_length=128, nullable=True)
    
    mobile: Optional[str] = Field(default="", max_length=20, nullable=True)
    email: Optional[str] = Field(default="", max_length=64, nullable=True)
    status: str = Field(default="NF", max_length=32, nullable=False)
    birth_marks: Optional[str] = Field(default="", max_length=512, nullable=True)
    submitted_on: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )


class RegisteredCases(SQLModel, table=True):
    """
    Registered Case entity representing official missing person profiles.
    """
    __table_args__ = {"extend_existing": True}

    id: str = Field(
        primary_key=True, default_factory=lambda: str(uuid4()), nullable=False
    )
    submitted_by: str = Field(max_length=64, nullable=False)
    name: str = Field(max_length=128, nullable=False)
    father_name: Optional[str] = Field(default="", max_length=128, nullable=True)
    age: Optional[str] = Field(default="", max_length=8, nullable=True)
    
    # Age Progression & Demographic Attributes
    age_at_disappearance: Optional[int] = Field(default=None, nullable=True)
    current_estimated_age: Optional[int] = Field(default=None, nullable=True)
    gender: Optional[str] = Field(default="Unknown", max_length=32, nullable=True)
    
    # Biometric Vector Storage (L2-normalized 128-D embedding JSON/String)
    primary_photo_vector: Optional[str] = Field(default=None, nullable=True)
    
    complainant_name: Optional[str] = Field(default="", max_length=128, nullable=True)
    complainant_mobile: Optional[str] = Field(default="", max_length=20, nullable=True)
    adhaar_card: Optional[str] = Field(default="", max_length=20, nullable=True)
    last_seen: Optional[str] = Field(default="", max_length=128, nullable=True)
    
    # Geolocation Coordinates for Last Known Location
    last_known_latitude: Optional[float] = Field(default=None, nullable=True)
    last_known_longitude: Optional[float] = Field(default=None, nullable=True)
    
    address: Optional[str] = Field(default="", max_length=512, nullable=True)
    face_mesh: str = Field(nullable=False)  # JSON string of deep embeddings and landmarks
    submitted_on: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )
    status: str = Field(default="NF", max_length=32, nullable=False)  # 'NF' / 'Active', 'F' / 'Solved', 'Under Review'
    birth_marks: Optional[str] = Field(default="", max_length=512, nullable=True)
    matched_with: Optional[str] = Field(default="", nullable=True)


# Convenience Model Aliases
Case = RegisteredCases
Sighting = PublicSubmissions


if __name__ == "__main__":
    sqlite_url = "sqlite:///sqlite_database.db"
    engine = create_engine(sqlite_url)

    RegisteredCases.__table__.create(engine)
    PublicSubmissions.__table__.create(engine)
