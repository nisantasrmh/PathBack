from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, create_engine, SQLModel


class PublicSubmissions(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: str = Field(
        primary_key=True, default_factory=lambda: str(uuid4()), nullable=False
    )
    submitted_by: Optional[str] = Field(default="", max_length=128, nullable=True)
    face_mesh: str = Field(nullable=False)  # JSON string of deep embeddings and landmarks
    location: Optional[str] = Field(default="", max_length=128, nullable=True)
    mobile: Optional[str] = Field(default="", max_length=20, nullable=True)
    email: Optional[str] = Field(default="", max_length=64, nullable=True)
    status: str = Field(default="NF", max_length=16, nullable=False)
    birth_marks: Optional[str] = Field(default="", max_length=512, nullable=True)
    submitted_on: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )


class RegisteredCases(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: str = Field(
        primary_key=True, default_factory=lambda: str(uuid4()), nullable=False
    )
    submitted_by: str = Field(max_length=64, nullable=False)
    name: str = Field(max_length=128, nullable=False)
    father_name: Optional[str] = Field(default="", max_length=128, nullable=True)
    age: Optional[str] = Field(default="", max_length=8, nullable=True)
    complainant_name: Optional[str] = Field(default="", max_length=128, nullable=True)
    complainant_mobile: Optional[str] = Field(default="", max_length=20, nullable=True)
    adhaar_card: Optional[str] = Field(default="", max_length=20, nullable=True)
    last_seen: Optional[str] = Field(default="", max_length=128, nullable=True)
    address: Optional[str] = Field(default="", max_length=512, nullable=True)
    face_mesh: str = Field(nullable=False)  # JSON string of deep embeddings and landmarks
    submitted_on: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )
    status: str = Field(default="NF", max_length=16, nullable=False)
    birth_marks: Optional[str] = Field(default="", max_length=512, nullable=True)
    matched_with: Optional[str] = Field(default="", nullable=True)


if __name__ == "__main__":
    sqlite_url = "sqlite:///sqlite_database.db"
    engine = create_engine(sqlite_url)

    RegisteredCases.__table__.create(engine)
    PublicSubmissions.__table__.create(engine)
