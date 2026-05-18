import os
import ssl
from urllib.parse import urlparse, urlunparse
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.environ["DATABASE_URL"]
parsed = urlparse(DATABASE_URL)
clean_url = urlunparse(parsed._replace(query=""))
ASYNC_URL = (
    clean_url
    .replace("postgresql://", "postgresql+asyncpg://")
    .replace("postgres://", "postgresql+asyncpg://")
)
ssl_context = ssl.create_default_context()
engine = create_async_engine(ASYNC_URL, echo=False, connect_args={"ssl": ssl_context})
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    photo_data = Column(Text)
    face_description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Sighting(Base):
    __tablename__ = "sightings"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    person_name = Column(String(100), default="Unknown")
    location = Column(String(100))
    task_description = Column(Text)
    activity_type = Column(String(50), nullable=True)
    confidence = Column(Float, default=0.0)
    thumbnail_data = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class Chore(Base):
    __tablename__ = "chores"
    id = Column(Integer, primary_key=True, index=True)
    person_name = Column(String(100), nullable=False)
    chore_name = Column(String(200), nullable=False)
    frequency = Column(String(20), default="daily")
    estimated_time = Column(String(50), nullable=True)
    points = Column(Integer, default=10)
    chore_type = Column(String(20), default="standard")   # core / rotating / standard
    rotating_with = Column(String(100), nullable=True)
    active = Column(Boolean, default=True)
    show_on_dashboard = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChoreAssessment(Base):
    __tablename__ = "chore_assessments"
    id = Column(Integer, primary_key=True, index=True)
    person_name = Column(String(100))
    chore_name = Column(String(200))
    status = Column(String(20))
    score = Column(Integer, default=0)
    points_earned = Column(Integer, default=0)
    assessment_text = Column(Text)
    commentary_text = Column(Text)
    thumbnail_data = Column(Text)
    assessed_date = Column(String(10))
    time_spent_mins = Column(Integer, default=0)
    dispute_status = Column(String(20), nullable=True)   # pending / approved / denied
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class ChoreDispute(Base):
    """User-raised dispute against an AI chore assessment."""
    __tablename__ = "chore_disputes"
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("chore_assessments.id"))
    disputed_by = Column(String(100))
    dispute_note = Column(Text)
    reviewed_by = Column(String(100), nullable=True)
    status = Column(String(20), default="pending")   # pending / approved / denied
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class PersonDailyStat(Base):
    """Per-person daily time-tracking totals."""
    __tablename__ = "person_daily_stats"
    id = Column(Integer, primary_key=True, index=True)
    person_name = Column(String(100), nullable=False)
    stat_date = Column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    morning_arrival = Column(String(8), nullable=True)           # HH:MM (UTC)
    first_activity = Column(Text, nullable=True)
    kitchen_mins = Column(Float, default=0.0)
    personal_mins = Column(Float, default=0.0)
    family_mins = Column(Float, default=0.0)


class ChoreViolation(Base):
    """Rework violations spotted by the camera agent."""
    __tablename__ = "chore_violations"
    id = Column(Integer, primary_key=True, index=True)
    person_name = Column(String(100))
    violation_code = Column(String(50))
    description = Column(Text)
    callout = Column(Text)
    location = Column(String(100))
    thumbnail_data = Column(Text)
    violation_date = Column(String(10))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class TrendReport(Base):
    """Bi-weekly AI trend observation reports."""
    __tablename__ = "trend_reports"
    id = Column(Integer, primary_key=True, index=True)
    period_start = Column(String(10))
    period_end = Column(String(10))
    report_text = Column(Text)
    raw_data_json = Column(Text)
    generated_at = Column(DateTime, default=datetime.utcnow, index=True)


# Safe idempotent migrations
_MIGRATIONS = [
    "ALTER TABLE sightings ADD COLUMN IF NOT EXISTS activity_type VARCHAR(50)",
    "ALTER TABLE chores ADD COLUMN IF NOT EXISTS estimated_time VARCHAR(50)",
    "ALTER TABLE chores ADD COLUMN IF NOT EXISTS points INTEGER DEFAULT 10",
    "ALTER TABLE chore_assessments ADD COLUMN IF NOT EXISTS points_earned INTEGER DEFAULT 0",
    "ALTER TABLE chores ADD COLUMN IF NOT EXISTS chore_type VARCHAR(20) DEFAULT 'standard'",
    "ALTER TABLE chores ADD COLUMN IF NOT EXISTS rotating_with VARCHAR(100)",
    "ALTER TABLE chore_assessments ADD COLUMN IF NOT EXISTS commentary_text TEXT",
    "ALTER TABLE chore_assessments ADD COLUMN IF NOT EXISTS time_spent_mins INTEGER DEFAULT 0",
    "ALTER TABLE chores ADD COLUMN IF NOT EXISTS show_on_dashboard BOOLEAN DEFAULT TRUE",
    "ALTER TABLE chore_assessments ADD COLUMN IF NOT EXISTS dispute_status VARCHAR(20)",
]


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for stmt in _MIGRATIONS:
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass


async def get_db():
    async with SessionLocal() as session:
        yield session
