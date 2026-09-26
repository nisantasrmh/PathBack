"""
Database Query and Management Engine for PathBack.

Provides ORM query interfaces, automated SQLite schema migrations, multi-modal case lookups,
administrative verification tracking, and Haversine geospatial radius filtering.
"""

from __future__ import annotations

import math
import os
import sqlite3
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import text
from sqlmodel import Session, create_engine, select

from pages.helper.data_models import (
    Case,
    CaseStatus,
    PublicSubmissions,
    RegisteredCases,
    Sighting,
)

sqlite_url = "sqlite:///sqlite_database.db"
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

EARTH_RADIUS_KM = 6371.0088


# ==========================================================
# Automated SQLite Migration Engine
# ==========================================================
def run_migrations() -> None:
    """
    Lightweight automated SQLite migration utility.
    Inspects existing database tables and dynamically adds any missing columns
    (e.g., geolocation, demographic fields, verification status) without breaking existing data.
    """
    column_definitions = {
        "registeredcases": {
            "age_at_disappearance": "INTEGER",
            "current_estimated_age": "INTEGER",
            "gender": "VARCHAR(32) DEFAULT 'Unknown'",
            "primary_photo_vector": "TEXT",
            "last_known_latitude": "FLOAT",
            "last_known_longitude": "FLOAT",
        },
        "publicsubmissions": {
            "address_text": "VARCHAR(256) DEFAULT ''",
            "latitude": "FLOAT",
            "longitude": "FLOAT",
            "confidence_score": "FLOAT",
            "device_metadata": "VARCHAR(1024) DEFAULT ''",
            "verified_by_admin": "BOOLEAN DEFAULT 0",
            "verified_by": "VARCHAR(128) DEFAULT ''",
        },
    }

    try:
        with engine.connect() as conn:
            for table_name, columns in column_definitions.items():
                # Query existing columns in table
                res = conn.execute(text(f"PRAGMA table_info({table_name});")).fetchall()
                existing_cols = {row[1].lower() for row in res} if res else set()

                if not existing_cols:
                    continue

                for col_name, col_type in columns.items():
                    if col_name.lower() not in existing_cols:
                        try:
                            conn.execute(
                                text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type};")
                            )
                            conn.commit()
                        except Exception as ex:
                            print(f"[Migration] Note: Could not add column {col_name} to {table_name}: {ex}")
    except Exception as e:
        print(f"[Migration] Migration check note: {e}")


def create_db() -> None:
    """Initializes tables and runs automatic schema migrations."""
    try:
        RegisteredCases.__table__.create(engine, checkfirst=True)
        PublicSubmissions.__table__.create(engine, checkfirst=True)
        run_migrations()
    except Exception:
        pass


# ==========================================================
# Geospatial Radius Queries (Haversine Formula)
# ==========================================================
def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculates great-circle distance between two GPS coordinates in kilometers.

    Args:
        lat1, lon1: Latitude and longitude of point 1 (decimal degrees).
        lat2, lon2: Latitude and longitude of point 2 (decimal degrees).

    Returns:
        float: Distance in kilometers.
    """
    try:
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
        return float(EARTH_RADIUS_KM * c)
    except Exception:
        return float("inf")


def fetch_sightings_within_radius(
    center_lat: float,
    center_lon: float,
    radius_km: float = 50.0,
    status: str = "All",
) -> List[Tuple[PublicSubmissions, float]]:
    """
    Retrieves all public sightings within a specified kilometer radius of coordinates,
    sorted by proximity (closest first).

    Args:
        center_lat: Center latitude.
        center_lon: Center longitude.
        radius_km: Search radius in kilometers.
        status: Filter by status ("All", "Not Found", "Found").

    Returns:
        List[Tuple[PublicSubmissions, float]]: List of (Sighting, distance_km) tuples.
    """
    create_db()
    with Session(engine) as session:
        query = select(PublicSubmissions)
        if status in ("NF", "Not Found", "Unresolved"):
            query = query.where(PublicSubmissions.status == "NF")
        elif status in ("F", "Found", "Solved"):
            query = query.where(PublicSubmissions.status == "F")

        all_sightings = session.exec(query).all()

    matches: List[Tuple[PublicSubmissions, float]] = []
    for s in all_sightings:
        if s.latitude is not None and s.longitude is not None:
            dist = haversine_distance(center_lat, center_lon, s.latitude, s.longitude)
            if dist <= radius_km:
                matches.append((s, round(dist, 2)))

    matches.sort(key=lambda x: x[1])
    return matches


def fetch_cases_within_radius(
    center_lat: float,
    center_lon: float,
    radius_km: float = 50.0,
    status: str = "All",
) -> List[Tuple[RegisteredCases, float]]:
    """
    Retrieves all registered missing person cases with last known coordinates
    within a specified kilometer radius.

    Args:
        center_lat: Center latitude.
        center_lon: Center longitude.
        radius_km: Search radius in kilometers.
        status: Filter by status ("All", "Not Found", "Found").

    Returns:
        List[Tuple[RegisteredCases, float]]: List of (Case, distance_km) tuples.
    """
    create_db()
    with Session(engine) as session:
        query = select(RegisteredCases)
        if status in ("NF", "Not Found", "Unresolved"):
            query = query.where(RegisteredCases.status == "NF")
        elif status in ("F", "Found", "Solved"):
            query = query.where(RegisteredCases.status == "F")

        all_cases = session.exec(query).all()

    matches: List[Tuple[RegisteredCases, float]] = []
    for c in all_cases:
        if c.last_known_latitude is not None and c.last_known_longitude is not None:
            dist = haversine_distance(
                center_lat, center_lon, c.last_known_latitude, c.last_known_longitude
            )
            if dist <= radius_km:
                matches.append((c, round(dist, 2)))

    matches.sort(key=lambda x: x[1])
    return matches


# ==========================================================
# Core Registration & CRUD Operations
# ==========================================================
def register_new_case(case_details: RegisteredCases) -> None:
    """Registers a new missing person profile."""
    create_db()
    with Session(engine) as session:
        session.add(case_details)
        session.commit()


def fetch_registered_cases(
    submitted_by: Optional[str] = None, status: str = "All"
) -> List[Any]:
    """Fetches registered cases with optional officer and status filters."""
    create_db()
    if status == "All":
        status_filter = ["F", "NF", "Active", "Solved", "Under Review"]
    elif status in ("Found", "Solved", "F"):
        status_filter = ["F", "Solved"]
    elif status in ("Not Found", "Unresolved", "NF", "Active"):
        status_filter = ["NF", "Active", "Under Review"]
    else:
        status_filter = ["F", "NF"]

    with Session(engine) as session:
        query = select(
            RegisteredCases.id,
            RegisteredCases.name,
            RegisteredCases.age,
            RegisteredCases.status,
            RegisteredCases.last_seen,
            RegisteredCases.matched_with,
            RegisteredCases.complainant_mobile,
            RegisteredCases.submitted_by,
        ).where(RegisteredCases.status.in_(status_filter))

        if submitted_by and submitted_by not in ("All", "System Administrator"):
            query = query.where(RegisteredCases.submitted_by == submitted_by)

        return session.exec(query).all()


def fetch_public_cases(
    train_data: bool = False, status: str = "All"
) -> List[Any]:
    """Fetches public sighting reports."""
    create_db()
    if train_data:
        with Session(engine) as session:
            status_val = "NF" if status in ("Not Found", "Unresolved", "NF") else status
            return session.exec(
                select(
                    PublicSubmissions.id,
                    PublicSubmissions.face_mesh,
                ).where(PublicSubmissions.status == status_val)
            ).all()

    with Session(engine) as session:
        query = select(
            PublicSubmissions.id,
            PublicSubmissions.status,
            PublicSubmissions.location,
            PublicSubmissions.mobile,
            PublicSubmissions.birth_marks,
            PublicSubmissions.submitted_on,
            PublicSubmissions.submitted_by,
        )
        if status in ("Found", "Solved", "F"):
            query = query.where(PublicSubmissions.status.in_(["F", "Solved"]))
        elif status in ("Not Found", "Unresolved", "NF", "Active"):
            query = query.where(PublicSubmissions.status.in_(["NF", "Active"]))

        return session.exec(query).all()


def get_not_confirmed_registered_cases(submitted_by: str) -> List[RegisteredCases]:
    create_db()
    with Session(engine) as session:
        return session.query(RegisteredCases).all()


def get_training_data(submitted_by: Optional[str] = None) -> List[Any]:
    """Fetches unresolved cases and their embeddings for indexing."""
    create_db()
    with Session(engine) as session:
        query = select(RegisteredCases.id, RegisteredCases.face_mesh).where(
            RegisteredCases.status.in_(["NF", "Active"])
        )
        if submitted_by and submitted_by not in ("All", "System Administrator"):
            query = query.where(RegisteredCases.submitted_by == submitted_by)
        return session.exec(query).all()


def new_public_case(public_case_details: PublicSubmissions) -> None:
    """Inserts a new sighting submission."""
    create_db()
    with Session(engine) as session:
        session.add(public_case_details)
        session.commit()


def get_public_case_detail(case_id: str) -> List[Any]:
    create_db()
    with Session(engine) as session:
        return session.exec(
            select(
                PublicSubmissions.location,
                PublicSubmissions.submitted_by,
                PublicSubmissions.mobile,
                PublicSubmissions.birth_marks,
            ).where(PublicSubmissions.id == str(case_id))
        ).all()


def get_registered_case_detail(case_id: str) -> List[Any]:
    create_db()
    with Session(engine) as session:
        return session.exec(
            select(
                RegisteredCases.name,
                RegisteredCases.complainant_mobile,
                RegisteredCases.age,
                RegisteredCases.last_seen,
                RegisteredCases.birth_marks,
            ).where(RegisteredCases.id == str(case_id))
        ).all()


def list_public_cases() -> List[PublicSubmissions]:
    create_db()
    with Session(engine) as session:
        return session.exec(select(PublicSubmissions)).all()


def update_found_status(register_case_id: str, public_case_id: str) -> None:
    """Marks both the registered case and public sighting as resolved/found."""
    create_db()
    with Session(engine) as session:
        registered_case = session.exec(
            select(RegisteredCases).where(RegisteredCases.id == str(register_case_id))
        ).first()
        if registered_case:
            registered_case.status = "F"
            registered_case.matched_with = str(public_case_id)
            session.add(registered_case)

        public_case = session.exec(
            select(PublicSubmissions).where(PublicSubmissions.id == str(public_case_id))
        ).first()
        if public_case:
            public_case.status = "F"
            session.add(public_case)

        session.commit()


def update_registered_case_status(case_id: str, new_status: str) -> bool:
    """Update registered case status to 'F' / 'Solved' or 'NF' / 'Active'."""
    create_db()
    with Session(engine) as session:
        case = session.exec(
            select(RegisteredCases).where(RegisteredCases.id == str(case_id))
        ).first()
        if case:
            case.status = new_status
            if new_status in ("NF", "Active"):
                case.matched_with = ""
            session.add(case)
            session.commit()
            return True
        return False


def update_sighting_verification(
    sighting_id: str, verified: bool, verified_by: Optional[str] = None
) -> bool:
    """
    Updates administrative verification status on a public sighting report.
    """
    create_db()
    with Session(engine) as session:
        sighting = session.exec(
            select(PublicSubmissions).where(PublicSubmissions.id == str(sighting_id))
        ).first()
        if sighting:
            sighting.verified_by_admin = verified
            if verified_by:
                sighting.verified_by = verified_by
            session.add(sighting)
            session.commit()
            return True
        return False


def delete_registered_case(case_id: str) -> bool:
    """Permanently delete a registered case and remove its saved image."""
    create_db()
    with Session(engine) as session:
        case = session.exec(
            select(RegisteredCases).where(RegisteredCases.id == str(case_id))
        ).first()
        if case:
            session.delete(case)
            session.commit()

            img_path = f"./resources/{case_id}.jpg"
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass
            return True
        return False


def update_public_case_status(case_id: str, new_status: str) -> bool:
    """Update public sighting case status."""
    create_db()
    with Session(engine) as session:
        case = session.exec(
            select(PublicSubmissions).where(PublicSubmissions.id == str(case_id))
        ).first()
        if case:
            case.status = new_status
            session.add(case)
            session.commit()
            return True
        return False


def delete_public_case(case_id: str) -> bool:
    """Permanently delete a public submission and remove its saved image."""
    create_db()
    with Session(engine) as session:
        case = session.exec(
            select(PublicSubmissions).where(PublicSubmissions.id == str(case_id))
        ).first()
        if case:
            session.delete(case)
            session.commit()

            img_path = f"./resources/{case_id}.jpg"
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass
            return True
        return False


def get_registered_cases_count(submitted_by: str, status: str) -> List[Any]:
    create_db()
    with Session(engine) as session:
        query = select(RegisteredCases).where(RegisteredCases.status == status)
        if submitted_by and submitted_by not in ("System Administrator", "All"):
            query = query.where(RegisteredCases.submitted_by == submitted_by)
        return session.exec(query).all()
