import os
import sqlite3
from sqlmodel import create_engine, Session, select

from pages.helper.data_models import RegisteredCases, PublicSubmissions

sqlite_url = "sqlite:///sqlite_database.db"
engine = create_engine(sqlite_url)


def create_db():
    try:
        RegisteredCases.__table__.create(engine)
        PublicSubmissions.__table__.create(engine)
    except Exception:
        pass


def register_new_case(case_details: RegisteredCases):
    create_db()
    with Session(engine) as session:
        session.add(case_details)
        session.commit()


def fetch_registered_cases(submitted_by: str = None, status: str = "All"):
    create_db()
    if status == "All":
        status_filter = ["F", "NF"]
    elif status == "Found" or status == "Solved":
        status_filter = ["F"]
    elif status == "Not Found" or status == "Unresolved":
        status_filter = ["NF"]
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

        # If submitted_by is provided and not "All" or "System Administrator", filter by user
        if submitted_by and submitted_by != "All" and submitted_by != "System Administrator":
            query = query.where(RegisteredCases.submitted_by == submitted_by)

        result = session.exec(query).all()
        return result


def fetch_public_cases(train_data: bool = False, status: str = "All"):
    create_db()
    if train_data:
        with Session(engine) as session:
            status_val = "NF" if status == "Not Found" else status
            result = session.exec(
                select(
                    PublicSubmissions.id,
                    PublicSubmissions.face_mesh,
                ).where(PublicSubmissions.status == status_val)
            ).all()
            return result

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
        if status == "Found" or status == "Solved":
            query = query.where(PublicSubmissions.status == "F")
        elif status == "Not Found" or status == "Unresolved":
            query = query.where(PublicSubmissions.status == "NF")

        result = session.exec(query).all()
        return result


def get_not_confirmed_registered_cases(submitted_by: str):
    create_db()
    with Session(engine) as session:
        result = session.query(RegisteredCases).all()
        return result


def get_training_data(submitted_by: str):
    create_db()
    with Session(engine) as session:
        result = session.exec(
            select(RegisteredCases.id, RegisteredCases.face_mesh)
            .where(RegisteredCases.submitted_by == submitted_by)
            .where(RegisteredCases.status == "NF")
        ).all()
        return result


def new_public_case(public_case_details: PublicSubmissions):
    create_db()
    with Session(engine) as session:
        session.add(public_case_details)
        session.commit()


def get_public_case_detail(case_id: str):
    create_db()
    with Session(engine) as session:
        result = session.exec(
            select(
                PublicSubmissions.location,
                PublicSubmissions.submitted_by,
                PublicSubmissions.mobile,
                PublicSubmissions.birth_marks,
            ).where(PublicSubmissions.id == str(case_id))
        ).all()
        return result


def get_registered_case_detail(case_id: str):
    create_db()
    with Session(engine) as session:
        result = session.exec(
            select(
                RegisteredCases.name,
                RegisteredCases.complainant_mobile,
                RegisteredCases.age,
                RegisteredCases.last_seen,
                RegisteredCases.birth_marks,
            ).where(RegisteredCases.id == str(case_id))
        ).all()
        return result


def list_public_cases():
    create_db()
    with Session(engine) as session:
        result = session.exec(select(PublicSubmissions)).all()
        return result


def update_found_status(register_case_id: str, public_case_id: str):
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


def update_registered_case_status(case_id: str, new_status: str):
    """Update registered case status to 'F' (Solved) or 'NF' (Unresolved)."""
    create_db()
    with Session(engine) as session:
        case = session.exec(
            select(RegisteredCases).where(RegisteredCases.id == str(case_id))
        ).first()
        if case:
            case.status = new_status
            if new_status == "NF":
                case.matched_with = ""
            session.add(case)
            session.commit()
            return True
        return False


def delete_registered_case(case_id: str):
    """Permanently delete a registered case and remove its saved image from disk."""
    create_db()
    with Session(engine) as session:
        case = session.exec(
            select(RegisteredCases).where(RegisteredCases.id == str(case_id))
        ).first()
        if case:
            session.delete(case)
            session.commit()

            # Remove image if exists
            img_path = f"./resources/{case_id}.jpg"
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass
            return True
        return False


def update_public_case_status(case_id: str, new_status: str):
    """Update public sighting case status to 'F' (Solved) or 'NF' (Unresolved)."""
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


def delete_public_case(case_id: str):
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


def get_registered_cases_count(submitted_by: str, status: str):
    create_db()
    with Session(engine) as session:
        query = select(RegisteredCases).where(RegisteredCases.status == status)
        if submitted_by and submitted_by != "System Administrator" and submitted_by != "All":
            query = query.where(RegisteredCases.submitted_by == submitted_by)
        result = session.exec(query).all()
        return result
