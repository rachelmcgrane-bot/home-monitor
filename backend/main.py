import os
import base64
import io
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, update as sql_update
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from database import (init_db, get_db, SessionLocal,
                      Person, Sighting, Chore, ChoreAssessment, ChoreViolation,
                      PersonDailyStat, ChoreDispute, TrendReport,
                      ViolationReview, ChorePointProposal)
from ai import (describe_face, analyse_frame, assess_chore,
                suggest_chore_points, check_point_weights, recheck_violation,
                announce_morning_chores, generate_summary,
                generate_comparison_report, generate_trend_report)

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Home Monitor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── In-memory state ───────────────────────────────────────────────────────────
_morning_announced: dict = {}    # {person_name: {"date": str, "text": str}}
_chore_sessions: dict = {}       # {person_name: {"chore_name": str, "start": datetime}}
_person_day_tracking: dict = {}  # {person_name: {date, morning_arrival, ...}}

# ── Motion detection state ────────────────────────────────────────────────────
_last_frame_pixels: dict = {}   # {location: list[int]}  grayscale 80×60 pixels
_last_ai_call: dict = {}        # {location: datetime}   time of last AI analysis
_last_ai_result: dict = {}      # {location: dict}       cached last AI result

MOTION_THRESHOLD = 8            # mean abs pixel diff (0–255) to count as motion
MIN_AI_INTERVAL_SECS = 60       # min seconds between AI calls per camera (1 min)
WAKING_HOURS = (5, 23)          # UTC hour range — 5:00–23:59 UTC ≈ 6am–midnight Irish

# ── Frame deduplication ───────────────────────────────────────────────────────
_processed_hashes: dict = {}    # {sha256_hex: datetime} — skip already-analysed frames

# ── Dashboard push-refresh ────────────────────────────────────────────────────
_last_assessment_ts: Optional[str] = None   # ISO string updated on each new ChoreAssessment

def _is_duplicate_frame(data: bytes) -> bool:
    """Return True if this exact frame was already analysed in the past 2 hours."""
    h = hashlib.sha256(data).hexdigest()
    now = datetime.utcnow()
    expired = [k for k, v in _processed_hashes.items() if (now - v).total_seconds() > 7200]
    for k in expired:
        del _processed_hashes[k]
    if h in _processed_hashes:
        return True
    _processed_hashes[h] = now
    return False


def _detect_motion(location: str, frame_bytes: bytes) -> bool:
    """Compare incoming frame against last stored frame. Returns True if motion detected."""
    try:
        if not _PIL_AVAILABLE:
            return True
        img = Image.open(io.BytesIO(frame_bytes)).convert("L").resize((80, 60))
        pixels = list(img.getdata())
        prev = _last_frame_pixels.get(location)
        _last_frame_pixels[location] = pixels
        if prev is None:
            return True  # first frame always processes
        diff = sum(abs(a - b) for a, b in zip(pixels, prev)) / len(pixels)
        return diff > MOTION_THRESHOLD
    except Exception:
        return True  # on error, process anyway


def _should_call_ai(location: str) -> bool:
    """Returns True if enough time has passed since last AI call for this camera."""
    last = _last_ai_call.get(location)
    if last is None:
        return True
    return (datetime.utcnow() - last).total_seconds() >= MIN_AI_INTERVAL_SECS


def _is_waking_hours() -> bool:
    """Returns True if current UTC hour is within waking hours window."""
    hour = datetime.utcnow().hour
    return WAKING_HOURS[0] <= hour <= WAKING_HOURS[1]

# ── Default chore seed ────────────────────────────────────────────────────────
# Format: (person, name, frequency, points, chore_type, rotating_with)
DEFAULT_CHORES = [
    # ── Rachel — Core chores (same every day) ──
    ("Rachel", "Dishwasher unload",              "daily", 25, "core", None),
    ("Rachel", "Clear kitchen counters",          "daily", 25, "core", None),
    ("Rachel", "Wipe kitchen counters (morning)", "daily", 25, "core", None),
    ("Rachel", "Wipe kitchen counters (evening)", "daily", 25, "core", None),
    # ── Liam — Core chores (same every day) ──
    ("Liam", "Dishwasher load",       "daily", 25, "core", None),
    ("Liam", "Clear kitchen table",   "daily", 25, "core", None),
    ("Liam", "Wipe kitchen table",    "daily", 25, "core", None),
    ("Liam", "Kitchen surfaces wipe", "daily", 25, "core", None),
    # ── Rotating pool — bins ──
    ("Liam",   "General bin",      "daily",        10, "rotating", "Rachel"),
    ("Liam",   "Recycle bin",      "daily",        10, "rotating", "Rachel"),
    ("Liam",   "Food waste bin",   "every-2-days",  8, "rotating", "Rachel"),
    # ── Rotating pool — household ──
    ("Liam",   "Toy pickup",                   "daily", 15, "rotating", "Rachel"),
    ("Rachel", "Kitchen floor sweep",          "daily", 15, "rotating", "Liam"),
    ("Rachel", "Downstairs toilet quick wipe", "daily", 10, "rotating", "Liam"),
    # ── Laundry — both people, when suits, weekly target ──
    ("both", "Laundry – put on wash", "when-suits", 12, "standard", None),
    ("both", "Laundry – sort",        "when-suits", 20, "standard", None),
    ("both", "Laundry – bring up clothes", "when-suits", 15, "standard", None),
    # ── Unassigned — whoever does it gets the credit ──
    ("unassigned", "Bring in milk delivery",   "daily",  5, "standard", None),
    ("unassigned", "Put milk in fridge",        "daily",  5, "standard", None),
    ("unassigned", "Collect post / mail",       "daily",  3, "standard", None),
    ("unassigned", "Lock front door at night",  "daily",  3, "standard", None),
    ("unassigned", "Fill water filter jug",     "daily",  3, "standard", None),
    ("unassigned", "Put out school bags",       "daily",  5, "standard", None),
    # ── Liam — Weekly ──
    ("Liam", "Main bathroom clean",           "weekly", 20, "standard", None),
    ("Liam", "Hoover downstairs high-traffic","weekly", 20, "standard", None),
    ("Liam", "Tidy utility room",             "weekly", 12, "standard", None),
    ("Liam", "Meal planning + shopping list", "weekly", 25, "standard", None),
    ("Liam", "Grocery shopping + put away",   "weekly", 60, "standard", None),
    ("Liam", "Meal prep: batch cook",         "weekly", 45, "standard", None),
    ("Liam", "Bathroom deep clean",           "weekly", 30, "standard", None),
    ("Liam", "Clear washing machine filter",  "weekly", 12, "standard", None),
    ("Liam", "Full downstairs vacuum",        "weekly", 30, "standard", None),
    ("Liam", "Mopping",                       "weekly", 30, "standard", None),
    # ── Liam — Monthly ──
    ("Liam", "Wipe walls – stairs",   "monthly", 15, "standard", None),
    ("Liam", "Wipe skirting boards",  "monthly", 20, "standard", None),
    ("Liam", "Clean sides of stairs", "monthly", 10, "standard", None),
    ("Liam", "Wipe kitchen cupboards","monthly", 20, "standard", None),
    # ── Rachel — Weekly ──
    ("Rachel", "Clean dryer lint filter",        "weekly",  8, "standard", None),
    ("Rachel", "Wipe inside of dryer",           "weekly", 12, "standard", None),
    ("Rachel", "Run washing machine drum clean", "weekly",  8, "standard", None),
    ("Rachel", "Bed linen: own beds",            "weekly", 30, "standard", None),
    ("Rachel", "Tidy the sitting room",          "weekly", 20, "standard", None),
    ("Rachel", "Tidy the family room",           "weekly", 20, "standard", None),
    ("Rachel", "Vacuum kitchen",                 "weekly", 20, "standard", None),
    ("Rachel", "Cook home made dinner",          "weekly", 40, "standard", None),
    ("Rachel", "Bring out the bins",             "weekly", 12, "standard", None),
    # ── Rachel — Monthly ──
    ("Rachel", "Clean out the fridge",      "monthly", 15, "standard", None),
    ("Rachel", "Deep clean bins/frame",     "monthly", 20, "standard", None),
    ("Rachel", "Clean windows (interior)",  "monthly", 30, "standard", None),
    ("Rachel", "Clean bins (wash out)",     "monthly", 15, "standard", None),
    ("Rachel", "Clean leather sofas",       "monthly", 10, "standard", None),
    ("Rachel", "Clean around bin frame",    "monthly",  5, "standard", None),
]

_CORE_MAP = {
    "Rachel": [
        "Dishwasher unload", "Clear kitchen counters",
        "Wipe kitchen counters (morning)", "Wipe kitchen counters (evening)",
    ],
    "Liam": [
        "Dishwasher load", "Clear kitchen table",
        "Wipe kitchen table", "Kitchen surfaces wipe",
    ],
}

# Pool of rotating chores per person — 2 selected per Mon/Fri assignment period
_ROTATING_POOLS = {
    "Liam":   ["General bin", "Recycle bin", "Food waste bin", "Toy pickup"],
    "Rachel": ["Kitchen floor sweep", "Downstairs toilet quick wipe"],
}

_ROTATING_MAP = [
    ("Liam",   "General bin",                  "Rachel"),
    ("Liam",   "Recycle bin",                  "Rachel"),
    ("Liam",   "Food waste bin",               "Rachel"),
    ("Liam",   "Toy pickup",                   "Rachel"),
    ("Rachel", "Kitchen floor sweep",          "Liam"),
    ("Rachel", "Downstairs toilet quick wipe", "Liam"),
]


def _generate_pwa_icons():
    """Generate 192×192 and 512×512 PWA app icons using PIL."""
    try:
        from PIL import Image, ImageDraw
        for size in [192, 512]:
            path = static_dir / f"icon-{size}.png"
            if path.exists():
                continue
            img = Image.new("RGB", (size, size), "#0d0f1a")
            draw = ImageDraw.Draw(img)
            s = size
            # Rounded card background
            draw.rounded_rectangle([s*.06,s*.06,s*.94,s*.94], radius=s*.18, fill="#141726")
            # House body
            draw.rectangle([s*.28,s*.52,s*.72,s*.78], fill="#3b82f6")
            # Roof triangle
            draw.polygon([(s*.5,s*.2),(s*.18,s*.54),(s*.82,s*.54)], fill="#818cf8")
            # Door
            draw.rounded_rectangle([s*.42,s*.62,s*.58,s*.78], radius=s*.03, fill="#0d0f1a")
            # Window
            draw.rounded_rectangle([s*.32,s*.58,s*.44,s*.7], radius=s*.02, fill="#22d3ee")
            # Chimney
            draw.rectangle([s*.62,s*.26,s*.72,s*.4], fill="#c084fc")
            img.save(path, "PNG", optimize=True)
    except Exception:
        pass  # icons are optional — app still works without them


@app.on_event("startup")
async def startup():
    _generate_pwa_icons()
    await init_db()
    async with SessionLocal() as db:
        # ── Seed only if the DB is completely empty (first install) ─────────
        count_res = await db.execute(select(func.count(Chore.id)))
        if int(count_res.scalar() or 0) == 0:
            for person, name, freq, pts, ctype, rwith in DEFAULT_CHORES:
                db.add(Chore(person_name=person, chore_name=name, frequency=freq,
                             points=pts, chore_type=ctype, rotating_with=rwith))
            await db.commit()

        # ── One-time cleanup: deactivate chores that were added by mistake ──
        _erroneously_added = [
            ("Rachel", "Clean kitchen sink"),
            ("Rachel", "Wipe microwave"),
            ("Rachel", "Tidy kitchen before bed"),
            ("Liam",   "Wipe hob after cooking"),
            ("Liam",   "Empty kitchen bin"),
            ("Liam",   "Set dishwasher to run"),
        ]
        for person, name in _erroneously_added:
            await db.execute(
                sql_update(Chore)
                .where(Chore.person_name == person, Chore.chore_name == name)
                .values(active=False))

        # ── Deduplicate: if the same (person, name) has multiple active rows,
        #    keep the one with the lowest id and deactivate the rest ──────────
        all_active_res = await db.execute(
            select(Chore.id, Chore.person_name, Chore.chore_name)
            .where(Chore.active == True)
            .order_by(Chore.person_name, Chore.chore_name, Chore.id))
        seen: dict = {}
        for row in all_active_res.all():
            key = (row.person_name, row.chore_name)
            if key in seen:
                # Duplicate — deactivate this later-id copy
                await db.execute(
                    sql_update(Chore).where(Chore.id == row.id).values(active=False))
            else:
                seen[key] = row.id

        # ── Ensure correct chore_type on known chores ────────────────────────
        for person, names in _CORE_MAP.items():
            for name in names:
                await db.execute(
                    sql_update(Chore)
                    .where(Chore.person_name == person, Chore.chore_name == name)
                    .values(chore_type="core"))
        for person, name, partner in _ROTATING_MAP:
            await db.execute(
                sql_update(Chore)
                .where(Chore.person_name == person, Chore.chore_name == name)
                .values(chore_type="rotating", rotating_with=partner))
        await db.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _thumb(data: bytes, size=(320, 240)) -> str:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return base64.standard_b64encode(buf.getvalue()).decode()

def _uri(b64: str) -> str:
    return f"data:image/jpeg;base64,{b64}"

async def _read(upload: UploadFile) -> bytes:
    return await upload.read()

def _today() -> str:
    """Return current chore-day date. Day starts at 06:00 UTC (≈ 7am Irish time).
    Anything between midnight and 06:00 UTC is still counted as the previous day."""
    return (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d")

def _date_minus(date_str: str, days: int) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")

def _chore_matches(chore_name: str, task: str) -> bool:
    """Return True if the AI task description plausibly refers to this chore.
    Uses 5-char word stems with a 60% match threshold to reduce false positives."""
    stop = {"the","a","an","is","are","and","or","to","of","in","on","up","out","down"}
    words = [w.strip(".,!?").lower() for w in chore_name.split()
             if w.lower() not in stop and len(w) > 3]
    task_low = task.lower()
    if not words:
        return False
    # Use 5-char stem and require 60% of significant words to match
    matched = sum(1 for w in words if w[:5] in task_low)
    return matched / len(words) >= 0.6

def _day_qualifies_every2(chore_id: int, today: str) -> bool:
    """Return True if an every-2-days chore should appear today."""
    day_of_year = datetime.strptime(today, "%Y-%m-%d").timetuple().tm_yday
    return (day_of_year + chore_id) % 2 == 0


def _assignment_date(today_str: str) -> str:
    """Return the most recent Monday or Friday on or before today_str.
    Rotating chores are assigned fresh on Monday morning and Friday morning,
    each with a 3.5-day (84-hour) deadline."""
    dt = datetime.strptime(today_str, "%Y-%m-%d")
    wd = dt.weekday()          # 0=Mon … 4=Fri … 6=Sun
    if wd in (0, 4):           # Monday or Friday → assigned today
        return today_str
    elif wd == 1:              # Tuesday → last Monday
        return (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    elif wd == 2:              # Wednesday → last Monday
        return (dt - timedelta(days=2)).strftime("%Y-%m-%d")
    elif wd == 3:              # Thursday → last Monday
        return (dt - timedelta(days=3)).strftime("%Y-%m-%d")
    elif wd == 5:              # Saturday → last Friday
        return (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    else:                      # Sunday → last Friday
        return (dt - timedelta(days=2)).strftime("%Y-%m-%d")


def _select_rotating_pair(person: str, assign_date_str: str) -> list:
    """Pick 2 rotating chores for a person for the given assignment period.
    Cycles through the pool so different chores are active each period."""
    pool = _ROTATING_POOLS.get(person, [])
    if len(pool) <= 2:
        return pool[:]
    dt = datetime.strptime(assign_date_str, "%Y-%m-%d")
    iso_week = dt.isocalendar()[1]
    # Alternate between even/odd ISO weeks; both Mon and Fri within a week
    # share the same pair so there's continuity mid-week.
    if iso_week % 2 == 0:
        return [pool[0], pool[1]]
    else:
        return [pool[2 % len(pool)], pool[3 % len(pool)]]

async def _upsert_person_stats(person_name: str, tracking: dict, db: AsyncSession):
    date_str = tracking["date"]
    res = await db.execute(
        select(PersonDailyStat).where(
            PersonDailyStat.person_name == person_name,
            PersonDailyStat.stat_date == date_str))
    row = res.scalar_one_or_none()
    if row:
        row.kitchen_mins = tracking["kitchen_mins"]
        row.personal_mins = tracking["personal_mins"]
        row.family_mins = tracking["family_mins"]
        if tracking.get("morning_arrival") and not row.morning_arrival:
            row.morning_arrival = tracking["morning_arrival"]
        if tracking.get("first_activity") and not row.first_activity:
            row.first_activity = tracking["first_activity"]
    else:
        db.add(PersonDailyStat(
            person_name=person_name,
            stat_date=date_str,
            morning_arrival=tracking.get("morning_arrival"),
            first_activity=tracking.get("first_activity"),
            kitchen_mins=tracking["kitchen_mins"],
            personal_mins=tracking["personal_mins"],
            family_mins=tracking["family_mins"],
        ))
    await db.commit()


def _rotating_assignee(chore: Chore, yest_ass: list, today_day: int) -> str:
    primary, partner = chore.person_name, chore.rotating_with
    if not partner:
        return primary
    def best_score(name):
        m = [a for a in yest_ass if a.person_name == name and a.chore_name == chore.chore_name]
        return max((a.score for a in m), default=0)
    if best_score(primary) >= 5: return partner
    if best_score(partner) >= 5: return primary
    return primary if (today_day + chore.id) % 2 == 0 else partner


# ── Persons ───────────────────────────────────────────────────────────────────

@app.post("/api/persons")
async def enrol_person(name: str = Form(...), photo: UploadFile = File(...),
                       db: AsyncSession = Depends(get_db)):
    data = await _read(photo)
    face_desc = await describe_face(data)
    person = Person(name=name, photo_data=_thumb(data, (300, 300)), face_description=face_desc)
    db.add(person); await db.commit(); await db.refresh(person)
    return {"id": person.id, "name": person.name, "face_description": face_desc}

@app.get("/api/persons")
async def list_persons(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Person).order_by(Person.name))
    return [{"id": p.id, "name": p.name,
             "photo_url": _uri(p.photo_data) if p.photo_data else None,
             "created_at": p.created_at.isoformat()} for p in result.scalars().all()]

@app.delete("/api/persons/{pid}")
async def delete_person(pid: int, db: AsyncSession = Depends(get_db)):
    p = await db.get(Person, pid)
    if not p: raise HTTPException(404, "Not found")
    await db.delete(p); await db.commit()
    return {"ok": True}


# ── Chore management ──────────────────────────────────────────────────────────

@app.get("/api/chores/list")
async def list_chores(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Chore).where(Chore.active == True)
        .order_by(Chore.person_name, Chore.chore_type, Chore.frequency, Chore.chore_name))
    return [{"id": c.id, "person_name": c.person_name, "chore_name": c.chore_name,
             "frequency": c.frequency, "points": c.points,
             "chore_type": c.chore_type or "standard",
             "rotating_with": c.rotating_with or "",
             "show_on_dashboard": c.show_on_dashboard if c.show_on_dashboard is not None else True,
             } for c in result.scalars().all()]

@app.post("/api/chores/list")
async def add_chore(
    person_name: str = Form(...), chore_name: str = Form(...),
    frequency: str = Form("daily"), points: int = Form(10),
    chore_type: str = Form("standard"), rotating_with: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    c = Chore(person_name=person_name.strip(), chore_name=chore_name.strip(),
              frequency=frequency, points=points,
              chore_type=chore_type, rotating_with=rotating_with.strip() or None)
    db.add(c); await db.commit(); await db.refresh(c)
    return {"id": c.id, "person_name": c.person_name, "chore_name": c.chore_name,
            "frequency": c.frequency, "points": c.points,
            "chore_type": c.chore_type, "rotating_with": c.rotating_with or ""}

@app.put("/api/chores/list/{cid}")
async def update_chore(
    cid: int,
    chore_name: Optional[str] = Form(None), frequency: Optional[str] = Form(None),
    points: Optional[int] = Form(None), chore_type: Optional[str] = Form(None),
    rotating_with: Optional[str] = Form(None), show_on_dashboard: Optional[str] = Form(None),
    person_name: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    c = await db.get(Chore, cid)
    if not c: raise HTTPException(404, "Not found")
    if chore_name is not None: c.chore_name = chore_name.strip()
    if frequency is not None: c.frequency = frequency
    if points is not None: c.points = points
    if chore_type is not None: c.chore_type = chore_type
    if rotating_with is not None: c.rotating_with = rotating_with.strip() or None
    if person_name is not None: c.person_name = person_name.strip()
    if show_on_dashboard is not None: c.show_on_dashboard = show_on_dashboard in ("1", "true", "True")
    await db.commit()
    return {"ok": True}

@app.delete("/api/chores/list/{cid}")
async def delete_chore(cid: int, db: AsyncSession = Depends(get_db)):
    c = await db.get(Chore, cid)
    if not c: raise HTTPException(404, "Not found")
    c.active = False; await db.commit()
    return {"ok": True}

@app.post("/api/chores/suggest-points")
async def suggest_points(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Chore).where(Chore.active == True))
    chores = result.scalars().all()
    by_person: dict[str, list[dict]] = {}
    for c in chores:
        by_person.setdefault(c.person_name, []).append(
            {"name": c.chore_name, "estimated_time": "", "frequency": c.frequency,
             "id": c.id, "points": c.points})
    suggestions = await suggest_chore_points(by_person)
    updated = 0
    for person, chore_pts in suggestions.items():
        for chore_name, pts in chore_pts.items():
            for c in chores:
                if c.person_name == person and c.chore_name == chore_name:
                    c.points = int(pts); updated += 1
    await db.commit()
    # Also run weight check
    weight_feedback = await check_point_weights(by_person)
    return {"ok": True, "updated": updated, "suggestions": suggestions,
            "weight_feedback": weight_feedback}

@app.get("/api/chores/weight-check")
async def weight_check(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Chore).where(Chore.active == True))
    chores = result.scalars().all()
    by_person: dict[str, list[dict]] = {}
    for c in chores:
        by_person.setdefault(c.person_name, []).append(
            {"name": c.chore_name, "frequency": c.frequency,
             "points": c.points, "id": c.id})
    feedback = await check_point_weights(by_person)
    return {"feedback": feedback}


# ── Add a chore to today manually ─────────────────────────────────────────────

@app.post("/api/chores/day-add")
async def add_chore_to_day(
    person_name: str = Form(...),
    chore_name: str = Form(...),
    score: int = Form(10),
    notes: str = Form(""),
    meal_name: str = Form(""),
    date: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    target_date = date or _today()
    # Look up the chore to get points value
    res = await db.execute(
        select(Chore).where(
            Chore.chore_name == chore_name,
            Chore.active == True
        ))
    chore = res.scalar_one_or_none()
    base_pts = chore.points if chore else 10
    score = max(0, min(10, score))
    pts_earned = int(base_pts * score / 10)
    status = "done" if score >= 7 else "partial" if score >= 4 else "not_done"

    # Build assessment text — prepend meal name for cooking chores
    if meal_name.strip():
        assessment_text = f"🍽️ {meal_name.strip()}" + (f" — {notes.strip()}" if notes.strip() else "")
    else:
        assessment_text = notes.strip() or "Manually recorded"

    ca = ChoreAssessment(
        person_name=person_name,
        chore_name=chore_name,
        status=status,
        score=score,
        points_earned=pts_earned,
        assessment_text=assessment_text,
        commentary_text="",
        assessed_date=target_date,
        time_spent_mins=0,
    )
    db.add(ca)
    await db.commit()
    await db.refresh(ca)
    return {"ok": True, "id": ca.id, "points_earned": pts_earned,
            "status": status, "date": target_date}


# ── Recalculate daily totals ───────────────────────────────────────────────────

@app.post("/api/stats/recalculate")
async def recalculate_stats(db: AsyncSession = Depends(get_db)):
    """Re-sync in-memory tracking from DB and return fresh daily stats."""
    today = _today()
    for person_name in ["Liam", "Rachel"]:
        stat_res = await db.execute(
            select(PersonDailyStat).where(
                PersonDailyStat.person_name == person_name,
                PersonDailyStat.stat_date == today))
        stat_row = stat_res.scalar_one_or_none()
        if stat_row:
            _person_day_tracking[person_name] = {
                "date": today,
                "morning_arrival": stat_row.morning_arrival,
                "first_activity": stat_row.first_activity,
                "last_seen": None,
                "last_activity_type": "present",
                "last_in_kitchen": False,
                "kitchen_mins": float(stat_row.kitchen_mins or 0),
                "personal_mins": float(stat_row.personal_mins or 0),
                "family_mins": float(stat_row.family_mins or 0),
            }
    return {"ok": True, "message": "Totals refreshed from database"}


# ── AI connectivity test ──────────────────────────────────────────────────────

@app.get("/api/test-ai")
async def test_ai():
    """Probe multiple model names to find which ones work on this API key."""
    import anthropic
    from ai import client as ai_client
    candidates = [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-latest",
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
    ]
    results = {}
    for model in candidates:
        try:
            msg = await ai_client.messages.create(
                model=model, max_tokens=10,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            results[model] = "✅ works"
        except anthropic.APIStatusError as e:
            results[model] = f"❌ {e.status_code}"
        except Exception as e:
            results[model] = f"❌ {str(e)[:60]}"
    working = [m for m,r in results.items() if r.startswith("✅")]
    return {"results": results, "working": working}

# ── Debug / health ───────────────────────────────────────────────────────────

@app.get("/api/debug")
async def debug_info(db: AsyncSession = Depends(get_db)):
    """Diagnostic endpoint — shows current system state at a glance."""
    import anthropic as _ant
    sdk_ver = getattr(_ant, "__version__", "unknown")

    # DB counts
    persons_res = await db.execute(select(func.count(Person.id)))
    persons_count = int(persons_res.scalar() or 0)

    sightings_res = await db.execute(select(func.count(Sighting.id)))
    sightings_count = int(sightings_res.scalar() or 0)

    today = _today()
    today_sight_res = await db.execute(
        select(func.count(Sighting.id)).where(
            Sighting.timestamp >= datetime.utcnow() - timedelta(hours=24)))
    today_sightings = int(today_sight_res.scalar() or 0)

    # Last sighting
    last_res = await db.execute(
        select(Sighting).order_by(desc(Sighting.timestamp)).limit(1))
    last = last_res.scalar_one_or_none()

    return {
        "ok": True,
        "sdk_version": sdk_ver,
        "persons_enrolled": persons_count,
        "total_sightings": sightings_count,
        "sightings_last_24h": today_sightings,
        "last_sighting": {
            "person": last.person_name,
            "task": last.task_description,
            "timestamp": last.timestamp.isoformat(),
            "location": last.location,
        } if last else None,
        "waking_hours_now": _is_waking_hours(),
        "utc_hour": datetime.utcnow().hour,
        "min_ai_interval_secs": MIN_AI_INTERVAL_SECS,
        "motion_threshold": MOTION_THRESHOLD,
        "locations_tracked": list(_last_ai_call.keys()),
        "today_date": today,
    }


# ── Camera frame ──────────────────────────────────────────────────────────────

@app.post("/api/frame")
async def submit_frame(location: str = Form(...), frame: UploadFile = File(...),
                       db: AsyncSession = Depends(get_db)):
    data = await _read(frame)

    # ── Cost gates: dedup + motion + waking hours + minimum interval ─────────
    if _is_duplicate_frame(data):
        return {"id": None, "person_name": None, "task": "Duplicate frame",
                "activity_type": "other", "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat(),
                "etiquette_violation": None, "etiquette_nudge": None,
                "violations": [], "morning_announcement": None, "chore_assessment": None,
                "skipped": "duplicate"}

    motion = _detect_motion(location, data)
    waking = _is_waking_hours()
    interval_ok = _should_call_ai(location)

    if not waking:
        # Outside waking hours — skip AI entirely, return empty
        return {"id": None, "person_name": None, "task": "Outside active hours",
                "activity_type": "other", "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat(),
                "etiquette_violation": None, "etiquette_nudge": None,
                "violations": [], "morning_announcement": None, "chore_assessment": None,
                "skipped": "outside_hours"}

    if not motion and not interval_ok:
        # No motion and too soon to re-call — return cached result
        cached = _last_ai_result.get(location, {})
        return {**cached, "skipped": "no_motion",
                "timestamp": datetime.utcnow().isoformat()}

    result = await db.execute(select(Person))
    enrolled = [{"id": p.id, "name": p.name, "face_description": p.face_description or ""}
                for p in result.scalars().all()]

    try:
        analysis = await analyse_frame(data, enrolled)
    except Exception as exc:
        err_str = str(exc)
        raise HTTPException(status_code=500, detail=f"AI error: {err_str}")
    _last_ai_call[location] = datetime.utcnow()
    _last_ai_result[location] = {
        "id": None, "person_name": analysis.get("person_name", "Unknown"),
        "task": analysis.get("task", ""), "activity_type": analysis.get("activity_type", "present"),
        "confidence": float(analysis.get("confidence", 0.0)),
        "etiquette_violation": analysis.get("etiquette_violation"),
        "etiquette_nudge": analysis.get("etiquette_nudge"),
        "violations": analysis.get("violations", []),
        "morning_announcement": None, "chore_assessment": None,
    }

    person_name = analysis.get("person_name", "Unknown")

    # ── Skip unknown persons entirely — no storage, no tracking ─────────────
    if person_name in ("Unknown", None) or float(analysis.get("confidence", 0.0)) < 0.35:
        return {"id": None, "person_name": "Unknown", "task": "Person not recognised",
                "activity_type": "present", "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat(),
                "etiquette_violation": None, "etiquette_nudge": None,
                "violations": [], "morning_announcement": None, "chore_assessment": None,
                "skipped": "unknown_person"}

    thumb = _thumb(data)

    sighting = Sighting(
        person_id=analysis.get("person_id"),
        person_name=person_name,
        location=location,
        task_description=analysis.get("task", ""),
        activity_type=analysis.get("activity_type", "present"),
        confidence=float(analysis.get("confidence", 0.0)),
        thumbnail_data=thumb,
    )
    db.add(sighting); await db.commit(); await db.refresh(sighting)
    task_text = analysis.get("task", "")
    etiquette_violation = analysis.get("etiquette_violation")
    etiquette_nudge = analysis.get("etiquette_nudge")
    frame_violations = analysis.get("violations", [])

    chore_result = None
    morning_announcement = None
    now = datetime.utcnow()
    today = _today()
    today_day = int(today.split("-")[2])

    # ── Save frame-level violations ───────────────────────────────────────────
    # Cap at 3 per person per day
    if frame_violations and person_name:
        viol_count_res = await db.execute(
            select(func.count(ChoreViolation.id)).where(
                ChoreViolation.person_name == person_name,
                ChoreViolation.violation_date == today))
        today_viol_count = int(viol_count_res.scalar() or 0)
        slots = max(0, 3 - today_viol_count)
        saved = 0
        for v in frame_violations:
            if saved >= slots:
                break
            cv = ChoreViolation(
                person_name=person_name,
                violation_code=v.get("code", "UNKNOWN"),
                description=v.get("description", ""),
                callout=v.get("callout", ""),
                location=location,
                thumbnail_data=thumb,
                violation_date=today,
            )
            db.add(cv)
            saved += 1
        if saved:
            await db.commit()

    # ── Person daily time-tracking ────────────────────────────────────────────
    if True:  # person is always known here (unknown returns early above)
        act_type = analysis.get("activity_type", "present")
        in_kitchen = "kitchen" in location.lower()

        tracking = _person_day_tracking.get(person_name)
        if tracking is None or tracking["date"] != today:
            stat_res = await db.execute(
                select(PersonDailyStat).where(
                    PersonDailyStat.person_name == person_name,
                    PersonDailyStat.stat_date == today))
            stat_row = stat_res.scalar_one_or_none()
            tracking = {
                "date": today,
                "morning_arrival": stat_row.morning_arrival if stat_row else None,
                "first_activity": stat_row.first_activity if stat_row else None,
                "last_seen": None,
                "last_activity_type": act_type,
                "last_in_kitchen": in_kitchen,
                "kitchen_mins": float(stat_row.kitchen_mins) if stat_row else 0.0,
                "personal_mins": float(stat_row.personal_mins) if stat_row else 0.0,
                "family_mins": float(stat_row.family_mins) if stat_row else 0.0,
            }
            _person_day_tracking[person_name] = tracking

        if tracking["morning_arrival"] is None:
            tracking["morning_arrival"] = now.strftime("%H:%M")
            tracking["first_activity"] = analysis.get("task", "")

        if tracking["last_seen"] is not None:
            elapsed_mins = min((now - tracking["last_seen"]).total_seconds(), 120) / 60
            prev_act = tracking["last_activity_type"]
            if tracking["last_in_kitchen"]:
                tracking["kitchen_mins"] += elapsed_mins
            # personal_mins: "present" time (detected but not cleaning or family)
            if prev_act == "present":
                tracking["personal_mins"] += elapsed_mins
            # family_mins: active time spent with the children
            elif prev_act == "family":
                tracking["family_mins"] += elapsed_mins
            # cleaning time: tracked via ChoreAssessments, not bucketed here
            # cleaning time is tracked implicitly via ChoreAssessments — no separate bucket needed

        tracking["last_seen"] = now
        tracking["last_activity_type"] = act_type
        tracking["last_in_kitchen"] = in_kitchen
        await _upsert_person_stats(person_name, tracking, db)

    confidence = float(analysis.get("confidence", 0.0))
    if confidence >= 0.45:
        # ── Morning announcement ──────────────────────────────────────────────
        if _morning_announced.get(person_name, {}).get("date") != today:
            chores_res2 = await db.execute(select(Chore).where(Chore.active == True))
            all_c2 = chores_res2.scalars().all()
            todays_chores = []
            _ann_rotating = _select_rotating_pair(person_name, _assignment_date(today))
            for c in all_c2:
                if c.chore_type == "core" and c.person_name == person_name and c.frequency == "daily":
                    todays_chores.append({"chore_name": c.chore_name, "chore_type": "core", "points": c.points})
                elif (c.chore_type == "rotating" and c.person_name == person_name
                      and c.chore_name in _ann_rotating):
                    todays_chores.append({"chore_name": c.chore_name, "chore_type": "rotating", "points": c.points})
            if todays_chores:
                ann_text = await announce_morning_chores(person_name, todays_chores)
                _morning_announced[person_name] = {"date": today, "text": ann_text, "person": person_name}
                morning_announcement = ann_text

        # ── Chore assessment ──────────────────────────────────────────────────
        # Only attempt to match a chore when the AI has classified this frame
        # as active cleaning or cooking — this prevents passive activities
        # (sitting on the couch, watching TV, etc.) from triggering assessments.
        act_type_for_chore = analysis.get("activity_type", "other")
        chores_res = await db.execute(
            select(Chore).where(Chore.active == True))
        all_active = chores_res.scalars().all()
        # Include chores for this person, both, and unassigned
        person_chores = [c for c in all_active
                         if c.person_name in (person_name, "both", "unassigned")]
        matched = None
        if act_type_for_chore == "cleaning":
            matched = next((c for c in person_chores
                            if _chore_matches(c.chore_name, task_text)), None)
        if matched:
            session = _chore_sessions.get(person_name, {})
            if session.get("chore_name") == matched.chore_name:
                duration_mins = int((now - session["start"]).total_seconds() / 60)
            else:
                _chore_sessions[person_name] = {"chore_name": matched.chore_name, "start": now}
                duration_mins = 0

            assessment = await assess_chore(data, person_name, matched.chore_name, duration_mins)
            score = int(assessment.get("score", 0))
            if etiquette_violation and score > 1:
                score = max(score - 1, 1)
            pts_earned = int(matched.points * score / 10)

            ca = ChoreAssessment(
                person_name=person_name, chore_name=matched.chore_name,
                status=assessment.get("status", "unknown"), score=score,
                points_earned=pts_earned,
                assessment_text=assessment.get("assessment", ""),
                commentary_text=assessment.get("commentary", ""),
                thumbnail_data=thumb, assessed_date=today,
                time_spent_mins=duration_mins,
            )
            db.add(ca); await db.commit(); await db.refresh(ca)

            # Notify dashboard to refresh (polled by frontend every ~8s)
            global _last_assessment_ts
            _last_assessment_ts = datetime.utcnow().isoformat()

            # Cap assessment violations too — never exceed 3/person/day total
            if assessment.get("violations") and person_name != "Unknown":
                viol_count_res2 = await db.execute(
                    select(func.count(ChoreViolation.id)).where(
                        ChoreViolation.person_name == person_name,
                        ChoreViolation.violation_date == today))
                today_viol_count2 = int(viol_count_res2.scalar() or 0)
                slots2 = max(0, 3 - today_viol_count2)
                saved2 = 0
                for v in assessment.get("violations", []):
                    if saved2 >= slots2:
                        break
                    cv = ChoreViolation(
                        person_name=person_name,
                        violation_code=v.get("code", "UNKNOWN"),
                        description=v.get("description", ""),
                        callout=v.get("callout", ""),
                        location=location,
                        thumbnail_data=thumb,
                        violation_date=today,
                    )
                    db.add(cv)
                    saved2 += 1
                if saved2:
                    await db.commit()

            chore_result = {
                "chore": ca.chore_name, "status": ca.status,
                "score": ca.score, "points_earned": pts_earned,
                "assessment": ca.assessment_text,
                "commentary": assessment.get("commentary"),
                "reminder": assessment.get("reminder"),
                "time_spent_mins": duration_mins,
                "violations": assessment.get("violations", []),
                "chore_type": matched.chore_type,
            }

    return {"id": sighting.id, "person_name": sighting.person_name,
            "task": sighting.task_description, "activity_type": sighting.activity_type,
            "confidence": sighting.confidence, "timestamp": sighting.timestamp.isoformat(),
            "etiquette_violation": etiquette_violation,
            "etiquette_nudge": etiquette_nudge,
            "violations": frame_violations,
            "morning_announcement": morning_announcement,
            "chore_assessment": chore_result}


# ── Dashboard push-refresh polling ───────────────────────────────────────────

@app.get("/api/events/latest-ts")
async def latest_event_ts():
    """Lightweight endpoint the dashboard polls every ~8 s to detect new assessments.
    Returns the ISO timestamp of the most recent ChoreAssessment created via the camera.
    No DB query — purely in-memory."""
    return {"ts": _last_assessment_ts}


# ── Announcements ─────────────────────────────────────────────────────────────

@app.get("/api/announce/latest")
async def get_announcements():
    today = _today()
    return {p: d for p, d in _morning_announced.items() if d.get("date") == today}


# ── Violations / rework ───────────────────────────────────────────────────────

@app.get("/api/violations/today")
async def get_violations_today(date: Optional[str] = None,
                               db: AsyncSession = Depends(get_db)):
    target = date or _today()
    result = await db.execute(
        select(ChoreViolation)
        .where(ChoreViolation.violation_date == target)
        .order_by(desc(ChoreViolation.timestamp)))
    violations = result.scalars().all()

    by_person: dict[str, int] = {}
    for v in violations:
        by_person[v.person_name] = by_person.get(v.person_name, 0) + 1

    items = [{"id": v.id, "person_name": v.person_name,
              "violation_code": v.violation_code, "description": v.description,
              "callout": v.callout, "location": v.location,
              "thumbnail_url": _uri(v.thumbnail_data) if v.thumbnail_data else None,
              "timestamp": v.timestamp.isoformat()} for v in violations]

    return {"date": target, "by_person": by_person, "violations": items}


# ── Violation review workflow ─────────────────────────────────────────────────

@app.post("/api/violations/{vid}/review-request")
async def request_violation_review(vid: int, requested_by: str = Form(...),
                                    db: AsyncSession = Depends(get_db)):
    """Person disputes a violation — creates a peer-review request."""
    v = await db.get(ChoreViolation, vid)
    if not v: raise HTTPException(404, "Violation not found")
    review = ViolationReview(violation_id=vid, requested_by=requested_by,
                              status="peer_pending")
    db.add(review); await db.commit(); await db.refresh(review)
    return {"ok": True, "review_id": review.id}

@app.get("/api/violations/pending-reviews")
async def get_pending_reviews(db: AsyncSession = Depends(get_db)):
    """Get violations with pending peer reviews (for the other person to act on)."""
    result = await db.execute(
        select(ViolationReview).where(ViolationReview.status == "peer_pending")
        .order_by(desc(ViolationReview.created_at)))
    reviews = result.scalars().all()
    out = []
    for r in reviews:
        v = await db.get(ChoreViolation, r.violation_id)
        if v:
            out.append({
                "review_id": r.id, "violation_id": v.id,
                "requested_by": r.requested_by,
                "person_name": v.person_name,
                "violation_code": v.violation_code,
                "description": v.description,
                "callout": v.callout,
                "location": v.location,
                "thumbnail_url": _uri(v.thumbnail_data) if v.thumbnail_data else None,
                "timestamp": v.timestamp.isoformat(),
                "created_at": r.created_at.isoformat(),
            })
    return out

@app.post("/api/violations/{vid}/peer-verdict")
async def peer_verdict(vid: int, verdict: str = Form(...), reviewed_by: str = Form(...),
                       db: AsyncSession = Depends(get_db)):
    """Other person gives verdict: 'keep' or 'remove'."""
    result = await db.execute(
        select(ViolationReview).where(
            ViolationReview.violation_id == vid,
            ViolationReview.status == "peer_pending"))
    review = result.scalar_one_or_none()
    if not review: raise HTTPException(404, "No pending review for this violation")
    review.reviewed_by = reviewed_by
    if verdict == "remove":
        review.status = "peer_removed"
        v = await db.get(ChoreViolation, vid)
        if v: await db.delete(v)
    else:
        review.status = "peer_kept"
        v = await db.get(ChoreViolation, vid)
        if v: v.confirmed = True
    await db.commit()
    return {"ok": True, "verdict": verdict}

@app.post("/api/violations/{vid}/ai-recheck")
async def ai_recheck(vid: int, db: AsyncSession = Depends(get_db)):
    """Send violation image back to AI for a second opinion."""
    v = await db.get(ChoreViolation, vid)
    if not v: raise HTTPException(404, "Violation not found")
    if not v.thumbnail_data:
        raise HTTPException(400, "No image stored for this violation")
    img_bytes = base64.standard_b64decode(v.thumbnail_data)
    result = await recheck_violation(img_bytes, v.violation_code, v.description or "")
    confirmed = result.get("confirmed", False)
    reason = result.get("reason", "")
    # Update or create review record
    rev_res = await db.execute(
        select(ViolationReview).where(ViolationReview.violation_id == vid))
    review = rev_res.scalar_one_or_none()
    if review:
        review.status = "ai_confirmed" if confirmed else "ai_uncertain"
        review.ai_reason = reason
    else:
        db.add(ViolationReview(violation_id=vid, requested_by="system",
                                status="ai_confirmed" if confirmed else "ai_uncertain",
                                ai_reason=reason))
    if not confirmed:
        v.confirmed = False
        await db.delete(v)
    else:
        v.confirmed = True
    await db.commit()
    return {"ok": True, "confirmed": confirmed, "confidence": result.get("confidence", 0.0),
            "reason": reason, "action": "kept" if confirmed else "removed"}

@app.delete("/api/violations/{vid}")
async def delete_violation(vid: int, db: AsyncSession = Depends(get_db)):
    """Manually remove a violation."""
    v = await db.get(ChoreViolation, vid)
    if not v: raise HTTPException(404, "Violation not found")
    await db.delete(v); await db.commit()
    return {"ok": True}


# ── Point proposals ───────────────────────────────────────────────────────────

@app.post("/api/chores/proposals")
async def create_proposal(chore_id: int = Form(...), proposed_points: int = Form(...),
                           proposed_by: str = Form(...),
                           db: AsyncSession = Depends(get_db)):
    """Propose a point-value change — requires the other person to approve."""
    chore = await db.get(Chore, chore_id)
    if not chore: raise HTTPException(404, "Chore not found")
    prop = ChorePointProposal(
        chore_id=chore_id, chore_name=chore.chore_name,
        person_name=chore.person_name,
        current_points=chore.points, proposed_points=proposed_points,
        proposed_by=proposed_by, status="pending")
    db.add(prop); await db.commit(); await db.refresh(prop)
    return {"ok": True, "proposal_id": prop.id}

@app.get("/api/chores/proposals")
async def list_proposals(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChorePointProposal)
        .where(ChorePointProposal.status == "pending")
        .order_by(desc(ChorePointProposal.created_at)))
    return [{"id": p.id, "chore_id": p.chore_id, "chore_name": p.chore_name,
             "person_name": p.person_name, "current_points": p.current_points,
             "proposed_points": p.proposed_points, "proposed_by": p.proposed_by,
             "created_at": p.created_at.isoformat()} for p in result.scalars().all()]

@app.post("/api/chores/proposals/{pid}/verdict")
async def verdict_proposal(pid: int, verdict: str = Form(...), approved_by: str = Form(...),
                            db: AsyncSession = Depends(get_db)):
    prop = await db.get(ChorePointProposal, pid)
    if not prop: raise HTTPException(404, "Proposal not found")
    prop.status = verdict   # "approved" or "denied"
    prop.approved_by = approved_by
    if verdict == "approved" and prop.chore_id:
        chore = await db.get(Chore, prop.chore_id)
        if chore: chore.points = prop.proposed_points
    await db.commit()
    return {"ok": True, "verdict": verdict}


# ── Sightings ─────────────────────────────────────────────────────────────────

@app.get("/api/sightings")
async def get_sightings(limit: int = 50, location: Optional[str] = None,
                        person_id: Optional[int] = None,
                        person_name: Optional[str] = None,
                        activity_type: Optional[str] = None,
                        date: Optional[str] = None,
                        include_thumbnails: bool = False,
                        db: AsyncSession = Depends(get_db)):
    # include_thumbnails=False by default — thumbnails are base64 images stored in the DB
    # and each one is ~50-100KB. Fetching 200 at once was consuming Neon's transfer budget.
    # Pass include_thumbnails=true only from the activity feed where images are needed.
    if include_thumbnails:
        q = select(Sighting).order_by(desc(Sighting.timestamp)).limit(limit)
    else:
        # Exclude the thumbnail_data column entirely to keep the query lightweight
        q = select(
            Sighting.id, Sighting.person_name, Sighting.location,
            Sighting.task_description, Sighting.activity_type,
            Sighting.confidence, Sighting.timestamp
        ).order_by(desc(Sighting.timestamp)).limit(limit)
    if location: q = q.where(Sighting.location == location)
    if person_id: q = q.where(Sighting.person_id == person_id)
    if person_name: q = q.where(Sighting.person_name == person_name)
    if activity_type: q = q.where(Sighting.activity_type == activity_type)
    if date: q = q.where(func.date(Sighting.timestamp) == date)
    result = await db.execute(q)
    rows = result.all() if not include_thumbnails else [(s,) for s in result.scalars().all()]
    out = []
    for row in rows:
        if include_thumbnails:
            s = row[0]
            out.append({"id": s.id, "person_name": s.person_name, "location": s.location,
                         "task": s.task_description, "activity_type": s.activity_type or "other",
                         "confidence": s.confidence,
                         "thumbnail_url": _uri(s.thumbnail_data) if s.thumbnail_data else None,
                         "timestamp": s.timestamp.isoformat()})
        else:
            out.append({"id": row.id, "person_name": row.person_name, "location": row.location,
                         "task": row.task_description, "activity_type": row.activity_type or "other",
                         "confidence": row.confidence,
                         "thumbnail_url": None,
                         "timestamp": row.timestamp.isoformat()})
    return out

@app.get("/api/sightings/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    since = datetime.utcnow() - timedelta(hours=24)
    result = await db.execute(select(Sighting).where(Sighting.timestamp >= since))
    sightings = result.scalars().all()
    by_person: dict[str, int] = {}
    by_location: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for s in sightings:
        by_person[s.person_name] = by_person.get(s.person_name, 0) + 1
        by_location[s.location] = by_location.get(s.location, 0) + 1
        t = s.activity_type or "other"
        by_type[t] = by_type.get(t, 0) + 1
    return {"total_24h": len(sightings), "by_person": by_person,
            "by_location": by_location, "by_type": by_type}


# ── Chore assessments & plan ──────────────────────────────────────────────────

@app.get("/api/chores/assessments")
async def get_assessments(date: Optional[str] = None, person_name: Optional[str] = None,
                          db: AsyncSession = Depends(get_db)):
    target = date or _today()
    q = (select(ChoreAssessment).where(ChoreAssessment.assessed_date == target)
         .order_by(desc(ChoreAssessment.timestamp)))
    if person_name: q = q.where(ChoreAssessment.person_name == person_name)
    result = await db.execute(q)
    return [{"id": a.id, "person_name": a.person_name, "chore_name": a.chore_name,
             "status": a.status, "score": a.score, "points_earned": a.points_earned or 0,
             "assessment": a.assessment_text, "commentary": a.commentary_text or "",
             "time_spent_mins": a.time_spent_mins or 0,
             "dispute_status": a.dispute_status,
             "thumbnail_url": _uri(a.thumbnail_data) if a.thumbnail_data else None,
             "timestamp": a.timestamp.isoformat()} for a in result.scalars().all()]


@app.get("/api/chores/daily-plan")
async def get_daily_plan(date: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    import calendar as cal_module
    today = date or _today()
    yesterday = _date_minus(today, 1)
    day_before = _date_minus(today, 2)
    today_day = int(today.split("-")[2])

    # Saturday-only logic for rotating and monthly chores
    weekday = datetime.strptime(today, "%Y-%m-%d").weekday()  # 5=Saturday
    is_saturday = (weekday == 5)
    first_day = datetime.strptime(today[:7] + '-01', '%Y-%m-%d')
    days_to_sat = (5 - first_day.weekday()) % 7
    first_saturday = (first_day + timedelta(days=days_to_sat)).strftime('%Y-%m-%d')
    is_first_saturday = (today == first_saturday and is_saturday)

    chores_res = await db.execute(select(Chore).where(Chore.active == True))
    all_chores = chores_res.scalars().all()

    today_res = await db.execute(
        select(ChoreAssessment).where(ChoreAssessment.assessed_date == today))
    today_ass = today_res.scalars().all()

    yest_res = await db.execute(
        select(ChoreAssessment).where(ChoreAssessment.assessed_date == yesterday))
    yest_ass = yest_res.scalars().all()

    day_before_res = await db.execute(
        select(ChoreAssessment).where(ChoreAssessment.assessed_date == day_before))
    day_before_ass = day_before_res.scalars().all()

    def best(assessments, person, chore):
        matches = [a for a in assessments if a.person_name == person and a.chore_name == chore]
        return max(matches, key=lambda a: a.score, default=None)

    def carry_pts(original_pts: int, ya, day_before_ya) -> int:
        if ya and ya.score < 5:
            if day_before_ya and day_before_ya.score < 5:
                return 0
            return max(original_pts // 2, 1)
        return original_pts

    # Collect all persons including "both" chore partners
    persons_set = set()
    for c in all_chores:
        if c.person_name != "both":
            persons_set.add(c.person_name)
        if c.rotating_with:
            persons_set.add(c.rotating_with)
    persons = sorted(persons_set)

    plan = {}
    for person in persons:
        chore_list = []
        total_pts = 0
        earned_pts = 0
        weekly_pts = 0
        weekly_earned = 0

        # ── Core chores ───────────────────────────────────────────────────────
        for c in all_chores:
            if c.chore_type == "core" and c.person_name == person:
                ta = best(today_ass, person, c.chore_name)
                ya = best(yest_ass, person, c.chore_name)
                score = ta.score if ta else 0
                pts_earned = ta.points_earned if ta else 0
                done = score >= 5
                chore_list.append({
                    "id": c.id, "chore_name": c.chore_name,
                    "frequency": c.frequency, "chore_type": "core",
                    "rotating_with": None, "points": c.points,
                    "points_earned": pts_earned, "score": score,
                    "status": ta.status if ta else "pending", "done": done,
                    "carried_forward": not done and ya is not None and ya.score < 5,
                    "when_suits": False, "is_weekly_target": False,
                    "assessment": ta.assessment_text if ta else None,
                    "commentary": ta.commentary_text if ta else None,
                    "time_spent_mins": ta.time_spent_mins if ta else 0,
                    "thumbnail_url": _uri(ta.thumbnail_data) if ta and ta.thumbnail_data else None,
                    "assessment_id": ta.id if ta else None,
                    "dispute_status": ta.dispute_status if ta else None,
                })
                total_pts += c.points
                earned_pts += pts_earned

        # ── Rotating chores (assigned Mon & Fri, 3.5-day / 84-hr deadline) ─────
        _assign_str  = _assignment_date(today)
        _assign_dt   = datetime.strptime(_assign_str, "%Y-%m-%d")
        _deadline_dt = _assign_dt + timedelta(hours=84)
        _hours_left  = max(0.0, (_deadline_dt - datetime.utcnow()).total_seconds() / 3600)
        _is_overdue  = datetime.utcnow() > _deadline_dt
        _sel_rotating = _select_rotating_pair(person, _assign_str)

        for c in all_chores:
            if c.chore_type == "rotating" and c.person_name == person and c.chore_name in _sel_rotating:
                ta = best(today_ass, person, c.chore_name)
                score = ta.score if ta else 0
                pts_earned = ta.points_earned if ta else 0
                done = score >= 5
                chore_list.append({
                    "id": c.id, "chore_name": c.chore_name,
                    "frequency": c.frequency, "chore_type": "rotating",
                    "rotating_with": c.rotating_with, "points": c.points,
                    "points_earned": pts_earned, "score": score,
                    "status": ta.status if ta else "pending", "done": done,
                    "carried_forward": False,
                    "when_suits": False, "is_weekly_target": False,
                    "assessment": ta.assessment_text if ta else None,
                    "commentary": ta.commentary_text if ta else None,
                    "time_spent_mins": ta.time_spent_mins if ta else 0,
                    "thumbnail_url": _uri(ta.thumbnail_data) if ta and ta.thumbnail_data else None,
                    "assessment_id": ta.id if ta else None,
                    "dispute_status": ta.dispute_status if ta else None,
                    # Deadline info for frontend badge
                    "assign_date": _assign_str,
                    "deadline": _deadline_dt.strftime("%Y-%m-%d"),
                    "hours_left": round(_hours_left, 1),
                    "is_overdue": _is_overdue,
                })
                total_pts += c.points
                earned_pts += pts_earned

        # ── Monthly chores (first Saturday only, 1 per person) ───────────────
        if is_first_saturday:
            monthly_candidates = sorted(
                [c for c in all_chores
                 if c.chore_type == "standard" and c.frequency == "monthly"
                 and c.person_name == person],
                key=lambda c: c.id
            )
            if monthly_candidates:
                c = monthly_candidates[0]
                ta = best(today_ass, person, c.chore_name)
                score = ta.score if ta else 0
                pts_earned = ta.points_earned if ta else 0
                done = score >= 5
                chore_list.append({
                    "id": c.id, "chore_name": c.chore_name,
                    "frequency": c.frequency, "chore_type": "standard",
                    "rotating_with": None, "points": c.points,
                    "points_original": c.points,
                    "person_name": c.person_name,
                    "points_earned": pts_earned, "score": score,
                    "status": ta.status if ta else "pending", "done": done,
                    "carried_forward": False,
                    "when_suits": False, "is_weekly_target": False,
                    "assessment": ta.assessment_text if ta else None,
                    "commentary": ta.commentary_text if ta else None,
                    "time_spent_mins": ta.time_spent_mins if ta else 0,
                    "thumbnail_url": _uri(ta.thumbnail_data) if ta and ta.thumbnail_data else None,
                    "assessment_id": ta.id if ta else None,
                    "dispute_status": ta.dispute_status if ta else None,
                })
                total_pts += c.points
                earned_pts += pts_earned

        # ── Standard and "both" chores (skip monthly here — handled above) ──
        for c in all_chores:
            is_for_person = (c.person_name == person or c.person_name == "both")
            if c.chore_type == "standard" and is_for_person and c.frequency != "monthly":
                ta = best(today_ass, person, c.chore_name)
                ya = best(yest_ass, person, c.chore_name)
                day_before_ya = best(day_before_ass, person, c.chore_name)
                score = ta.score if ta else 0
                pts_earned = ta.points_earned if ta else 0
                done = score >= 5
                is_when_suits = c.frequency == "when-suits"
                is_weekly = c.frequency in ("weekly", "when-suits")
                carried = (c.frequency == "daily" and not done
                           and (not ya or ya.score < 5))
                adj_pts = carry_pts(c.points, ya, day_before_ya) if carried else c.points
                chore_list.append({
                    "id": c.id, "chore_name": c.chore_name,
                    "frequency": c.frequency, "chore_type": "standard",
                    "rotating_with": None, "points": adj_pts,
                    "points_original": c.points,
                    "person_name": c.person_name,
                    "points_earned": pts_earned, "score": score,
                    "status": ta.status if ta else "pending", "done": done,
                    "carried_forward": carried,
                    "when_suits": is_when_suits,
                    "is_weekly_target": is_weekly,
                    "assessment": ta.assessment_text if ta else None,
                    "commentary": ta.commentary_text if ta else None,
                    "time_spent_mins": ta.time_spent_mins if ta else 0,
                    "thumbnail_url": _uri(ta.thumbnail_data) if ta and ta.thumbnail_data else None,
                    "assessment_id": ta.id if ta else None,
                    "dispute_status": ta.dispute_status if ta else None,
                })
                if c.frequency == "daily":
                    total_pts += adj_pts
                    earned_pts += pts_earned
                elif is_weekly:
                    weekly_pts += c.points
                    weekly_earned += pts_earned

        # Sum ALL assessments for this person today (for KPI total, not just plan items)
        total_earned_today = sum(a.points_earned or 0 for a in today_ass if a.person_name == person)
        plan[person] = {
            "chores": chore_list,
            "total_points": total_pts,
            "earned_points": earned_pts,
            "total_earned_today": total_earned_today,
            "pct": int(earned_pts / total_pts * 100) if total_pts else 0,
            "weekly_points": weekly_pts,
            "weekly_earned": weekly_earned,
            "weekly_pct": int(weekly_earned / weekly_pts * 100) if weekly_pts else 0,
        }

    # ── Unassigned chores (anyone can pick these up today) ───────────────────
    unassigned_chores = [c for c in all_chores if c.person_name == "unassigned"]
    unassigned_list = []
    for c in unassigned_chores:
        # Who did this today (any person)?
        done_by = [a for a in today_ass if a.chore_name == c.chore_name]
        ta = max(done_by, key=lambda a: a.score, default=None)
        unassigned_list.append({
            "id": c.id, "chore_name": c.chore_name, "frequency": c.frequency,
            "points": c.points, "done": ta is not None,
            "done_by": ta.person_name if ta else None,
            "score": ta.score if ta else 0,
            "points_earned": ta.points_earned if ta else 0,
            "assessment_id": ta.id if ta else None,
        })

    return {"date": today, "plan": plan, "unassigned": unassigned_list}


@app.get("/api/chores/cooking-history")
async def cooking_history(person_name: Optional[str] = None, limit: int = 60,
                          db: AsyncSession = Depends(get_db)):
    """Return all cooking/meal-related chore assessments, newest first."""
    COOKING_KW = ("cook", "dinner", "meal", "lunch", "breakfast",
                  "bake", "roast", "laundry")  # keep laundry out — literal cooking only
    COOKING_KW = ("cook", "dinner", "meal", "lunch", "breakfast", "bake", "roast")
    q = select(ChoreAssessment).order_by(
        desc(ChoreAssessment.assessed_date), desc(ChoreAssessment.timestamp))
    if person_name:
        q = q.where(ChoreAssessment.person_name == person_name)
    result = await db.execute(q)
    all_ass = result.scalars().all()
    out = []
    for a in all_ass:
        if any(kw in a.chore_name.lower() for kw in COOKING_KW):
            # Extract meal name from assessment_text if present
            meal = ""
            if a.assessment_text and a.assessment_text.startswith("🍽️"):
                meal = a.assessment_text[2:].split("—")[0].strip()
            elif a.assessment_text and a.assessment_text != "Manually recorded":
                meal = a.assessment_text.split("—")[0].strip()
            out.append({
                "id": a.id, "person_name": a.person_name,
                "chore_name": a.chore_name, "meal_name": meal,
                "assessment_text": a.assessment_text or "",
                "score": a.score or 0, "points_earned": a.points_earned or 0,
                "assessed_date": a.assessed_date,
                "timestamp": a.timestamp.isoformat(),
                "thumbnail_url": _uri(a.thumbnail_data) if a.thumbnail_data else None,
            })
            if len(out) >= limit:
                break
    return out


@app.get("/api/chores/calendar")
async def get_calendar(month: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    if not month:
        month = datetime.utcnow().strftime("%Y-%m")
    chores_res = await db.execute(
        select(Chore).where(Chore.active == True, Chore.frequency == "daily"))
    daily_chores = chores_res.scalars().all()
    person_max: dict[str, int] = {}
    for c in daily_chores:
        if c.person_name != "both":
            person_max[c.person_name] = person_max.get(c.person_name, 0) + c.points

    result = await db.execute(
        select(ChoreAssessment).where(ChoreAssessment.assessed_date.like(f"{month}%"))
        .order_by(ChoreAssessment.assessed_date))
    assessments = result.scalars().all()

    raw: dict[str, dict[str, int]] = {}
    for a in assessments:
        raw.setdefault(a.assessed_date, {})
        raw[a.assessed_date][a.person_name] = (
            raw[a.assessed_date].get(a.person_name, 0) + (a.points_earned or 0))

    calendar: dict[str, dict] = {}
    for date, by_person in raw.items():
        calendar[date] = {}
        for person, earned in by_person.items():
            max_pts = person_max.get(person, 100)
            pct = int(earned / max_pts * 100) if max_pts else 0
            calendar[date][person] = {
                "earned": earned, "max": max_pts, "pct": pct,
                "status": "done" if pct >= 80 else "partial" if pct >= 40 else "not_done",
            }
    return {"month": month, "days": calendar, "person_max": person_max}


@app.get("/api/chores/progress")
async def get_progress(date: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    today = date or _today()
    year_month = today[:7]
    today_dt = datetime.strptime(today, "%Y-%m-%d")

    chores_res = await db.execute(select(Chore).where(Chore.active == True))
    all_chores = chores_res.scalars().all()

    today_res = await db.execute(
        select(ChoreAssessment).where(ChoreAssessment.assessed_date == today))
    today_ass = today_res.scalars().all()

    month_res = await db.execute(
        select(ChoreAssessment).where(ChoreAssessment.assessed_date.like(f"{year_month}%")))
    month_ass = month_res.scalars().all()

    persons = sorted({c.person_name for c in all_chores if c.person_name != "both"})
    output = []
    for person in persons:
        daily = [c for c in all_chores if c.person_name == person and c.frequency == "daily"]
        total_pts = sum(c.points for c in daily)
        earned_today = sum(a.points_earned or 0 for a in today_ass if a.person_name == person)
        pct_today = int(earned_today / total_pts * 100) if total_pts else 0
        days_done = len({a.assessed_date for a in month_ass
                         if a.person_name == person and (a.points_earned or 0) >= total_pts * 0.5})
        output.append({
            "person_name": person,
            "daily": {"total_pts": total_pts, "earned_pts": earned_today, "pct": pct_today},
            "monthly": {"days_elapsed": today_dt.day, "days_done": days_done,
                        "pct": int(days_done / today_dt.day * 100) if today_dt.day else 0},
        })
    return {"date": today, "persons": output}


@app.get("/api/chores/comparison")
async def get_comparison(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChoreAssessment).order_by(ChoreAssessment.chore_name))
    assessments = result.scalars().all()

    if not assessments:
        return {"report": "No chore data yet — get to work, folks!"}

    stats: dict[str, dict[str, dict]] = {}
    for a in assessments:
        chore = a.chore_name
        person = a.person_name
        stats.setdefault(chore, {})
        stats[chore].setdefault(person, {"count": 0, "total_score": 0, "total_pts": 0, "total_mins": 0})
        s = stats[chore][person]
        s["count"] += 1
        s["total_score"] += a.score or 0
        s["total_pts"] += a.points_earned or 0
        s["total_mins"] += a.time_spent_mins or 0

    chore_stats = {}
    for chore, by_person in stats.items():
        if len(by_person) > 1 or sum(v["count"] for v in by_person.values()) >= 2:
            chore_stats[chore] = {
                person: {
                    "count": v["count"],
                    "avg_score": round(v["total_score"] / v["count"], 1) if v["count"] else 0,
                    "total_pts": v["total_pts"],
                    "total_mins": v["total_mins"],
                }
                for person, v in by_person.items()
            }

    if not chore_stats:
        return {"report": "Not enough data yet for a comparison — keep going!"}

    report = await generate_comparison_report(chore_stats)
    return {"report": report, "stats": chore_stats}


@app.get("/api/chores/summary")
async def get_summary(period: str = "daily", date: Optional[str] = None,
                      db: AsyncSession = Depends(get_db)):
    today = date or _today()
    if period == "daily":
        start, end = today, today
    elif period == "weekly":
        dt = datetime.strptime(today, "%Y-%m-%d")
        start = (dt - timedelta(days=6)).strftime("%Y-%m-%d")
        end = today
    elif period == "bimonthly":
        dt = datetime.strptime(today, "%Y-%m-%d")
        start = (dt - timedelta(days=13)).strftime("%Y-%m-%d")
        end = today
    else:
        start = today[:7] + "-01"
        end = today

    result = await db.execute(
        select(ChoreAssessment)
        .where(ChoreAssessment.assessed_date >= start,
               ChoreAssessment.assessed_date <= end)
        .order_by(ChoreAssessment.person_name, ChoreAssessment.chore_name))
    assessments = result.scalars().all()

    by_person: dict[str, list[dict]] = {}
    for a in assessments:
        by_person.setdefault(a.person_name, []).append({
            "chore_name": a.chore_name, "score": a.score or 0,
            "status": a.status, "points_earned": a.points_earned or 0,
        })

    if not by_person:
        return {"period": period, "summary": "No chore data yet — get scrubbing, folks!"}

    text = await generate_summary(by_person, period)
    return {"period": period, "date_range": f"{start} to {end}", "summary": text}


# ── Disputes ─────────────────────────────────────────────────────────────────

@app.post("/api/disputes")
async def create_dispute(
    assessment_id: int = Form(...),
    disputed_by: str = Form(...),
    dispute_note: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    d = ChoreDispute(assessment_id=assessment_id, disputed_by=disputed_by,
                     dispute_note=dispute_note)
    db.add(d)
    ca = await db.get(ChoreAssessment, assessment_id)
    if ca:
        ca.dispute_status = "pending"
    await db.commit()
    return {"ok": True, "dispute_id": d.id}


@app.get("/api/disputes")
async def list_disputes(status: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    q = select(ChoreDispute).order_by(desc(ChoreDispute.created_at))
    if status:
        q = q.where(ChoreDispute.status == status)
    result = await db.execute(q)
    return [{"id": d.id, "assessment_id": d.assessment_id, "disputed_by": d.disputed_by,
             "dispute_note": d.dispute_note, "status": d.status,
             "created_at": d.created_at.isoformat()} for d in result.scalars().all()]


@app.put("/api/disputes/{did}")
async def review_dispute(
    did: int, status: str = Form(...), reviewed_by: str = Form("Admin"),
    db: AsyncSession = Depends(get_db),
):
    d = await db.get(ChoreDispute, did)
    if not d: raise HTTPException(404, "Not found")
    d.status = status
    d.reviewed_by = reviewed_by
    ca = await db.get(ChoreAssessment, d.assessment_id)
    if ca:
        ca.dispute_status = status
    await db.commit()
    return {"ok": True}


# ── Trend reports ─────────────────────────────────────────────────────────────

@app.get("/api/trends/latest")
async def get_latest_trend(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TrendReport).order_by(desc(TrendReport.generated_at)).limit(1))
    report = result.scalar_one_or_none()
    if not report:
        return {"report": None, "generated_at": None}
    return {"report": report.report_text, "period_start": report.period_start,
            "period_end": report.period_end, "generated_at": report.generated_at.isoformat()}


@app.post("/api/trends/generate")
async def generate_trends(db: AsyncSession = Depends(get_db)):
    end = _today()
    start = _date_minus(end, 13)

    ass_res = await db.execute(
        select(ChoreAssessment)
        .where(ChoreAssessment.assessed_date >= start,
               ChoreAssessment.assessed_date <= end))
    assessments = ass_res.scalars().all()

    stat_res = await db.execute(
        select(PersonDailyStat)
        .where(PersonDailyStat.stat_date >= start,
               PersonDailyStat.stat_date <= end))
    stats = stat_res.scalars().all()

    viol_res = await db.execute(
        select(ChoreViolation)
        .where(ChoreViolation.violation_date >= start,
               ChoreViolation.violation_date <= end))
    violations = viol_res.scalars().all()

    fact_data: dict = {}
    for a in assessments:
        p = a.person_name
        fact_data.setdefault(p, {"total_done": 0, "total_score": 0, "count": 0,
                                  "violation_count": 0, "kitchen_mins": 0,
                                  "family_mins": 0, "personal_mins": 0, "days_seen": set()})
        if a.status == "done":
            fact_data[p]["total_done"] += 1
        fact_data[p]["total_score"] += a.score or 0
        fact_data[p]["count"] += 1

    for v in violations:
        p = v.person_name
        fact_data.setdefault(p, {"total_done": 0, "total_score": 0, "count": 0,
                                  "violation_count": 0, "kitchen_mins": 0,
                                  "family_mins": 0, "personal_mins": 0, "days_seen": set()})
        fact_data[p]["violation_count"] += 1

    for s in stats:
        p = s.person_name
        fact_data.setdefault(p, {"total_done": 0, "total_score": 0, "count": 0,
                                  "violation_count": 0, "kitchen_mins": 0,
                                  "family_mins": 0, "personal_mins": 0, "days_seen": set()})
        fact_data[p]["kitchen_mins"] += s.kitchen_mins or 0
        fact_data[p]["family_mins"] += s.family_mins or 0
        fact_data[p]["personal_mins"] += s.personal_mins or 0
        fact_data[p]["days_seen"].add(s.stat_date)

    for p in fact_data:
        d = fact_data[p]
        d["avg_score"] = d["total_score"] / d["count"] if d["count"] else 0
        d["days_seen"] = len(d["days_seen"])

    if not fact_data:
        return {"ok": False, "reason": "No data for the past 14 days"}

    report_text = await generate_trend_report(fact_data)
    tr = TrendReport(period_start=start, period_end=end, report_text=report_text,
                     raw_data_json=str(fact_data))
    db.add(tr)
    await db.commit()
    return {"ok": True, "report": report_text, "period": f"{start} to {end}"}


# ── Person daily/monthly time stats ──────────────────────────────────────────

@app.get("/api/stats/daily")
async def get_daily_stats(db: AsyncSession = Depends(get_db)):
    today = _today()
    res = await db.execute(
        select(PersonDailyStat).where(PersonDailyStat.stat_date == today))
    db_rows = {r.person_name: r for r in res.scalars().all()}

    output = {}
    for person in ["Liam", "Rachel"]:
        t = _person_day_tracking.get(person, {})
        row = db_rows.get(person)
        if t.get("date") == today:
            output[person] = {
                "morning_arrival": t.get("morning_arrival"),
                "first_activity": t.get("first_activity"),
                "kitchen_mins": round(t["kitchen_mins"]),
                "personal_mins": round(t["personal_mins"]),
                "family_mins": round(t["family_mins"]),
            }
        elif row:
            output[person] = {
                "morning_arrival": row.morning_arrival,
                "first_activity": row.first_activity,
                "kitchen_mins": round(row.kitchen_mins or 0),
                "personal_mins": round(row.personal_mins or 0),
                "family_mins": round(row.family_mins or 0),
            }
        else:
            output[person] = {
                "morning_arrival": None, "first_activity": None,
                "kitchen_mins": 0, "personal_mins": 0, "family_mins": 0,
            }
    return output


@app.get("/api/stats/monthly")
async def get_monthly_stats(month: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    if not month:
        month = datetime.utcnow().strftime("%Y-%m")

    # ── PersonDailyStat rows (kitchen/personal/family mins + arrivals) ─────────
    res = await db.execute(
        select(PersonDailyStat).where(PersonDailyStat.stat_date.like(f"{month}%")))
    rows = res.scalars().all()

    output: dict = {}
    for r in rows:
        o = output.setdefault(r.person_name, {
            "kitchen_mins": 0, "personal_mins": 0, "family_mins": 0,
            "days_seen": 0, "arrivals": [],
        })
        o["kitchen_mins"] += round(r.kitchen_mins or 0)
        o["personal_mins"] += round(r.personal_mins or 0)
        o["family_mins"] += round(r.family_mins or 0)
        o["days_seen"] += 1
        if r.morning_arrival:
            o["arrivals"].append(r.morning_arrival)

    for person in ["Liam", "Rachel"]:
        output.setdefault(person, {
            "kitchen_mins": 0, "personal_mins": 0, "family_mins": 0,
            "days_seen": 0, "arrivals": [],
        })

    # ── Days present: count distinct calendar days each person was sighted ─────
    # Use Sighting table with confidence >= 0.5 — more reliable than PersonDailyStat
    # which only gets written when time-tracking runs. Cast timestamp to date in SQL.
    month_start = f"{month}-01"
    month_end_dt = datetime.strptime(month_start, "%Y-%m-%d")
    # last day of month = first day of next month
    if month_end_dt.month == 12:
        next_month = month_end_dt.replace(year=month_end_dt.year + 1, month=1, day=1)
    else:
        next_month = month_end_dt.replace(month=month_end_dt.month + 1, day=1)

    for person in ["Liam", "Rachel"]:
        sighting_days_res = await db.execute(
            select(func.count(func.distinct(func.date(Sighting.timestamp))))
            .where(
                Sighting.person_name == person,
                Sighting.confidence >= 0.45,
                Sighting.timestamp >= datetime.strptime(month_start, "%Y-%m-%d"),
                Sighting.timestamp < next_month,
            )
        )
        days_present = int(sighting_days_res.scalar() or 0)
        output[person]["days_present"] = days_present

    # ── Quality: avg assessment score % this month ──────────────────────────
    next_month_str = next_month.strftime("%Y-%m-%d")
    for person in ["Liam", "Rachel"]:
        quality_res = await db.execute(
            select(func.avg(ChoreAssessment.score), func.count(ChoreAssessment.id))
            .where(
                ChoreAssessment.person_name == person,
                ChoreAssessment.status.in_(["done", "partial"]),
                ChoreAssessment.assessed_date >= month_start,
                ChoreAssessment.assessed_date < next_month_str,
            )
        )
        q_row = quality_res.one_or_none()
        avg_score = float(q_row[0] or 0) if q_row and q_row[0] else 0.0
        q_count = int(q_row[1] or 0) if q_row else 0
        output[person]["avg_quality_pct"] = round(avg_score / 10 * 100) if q_count > 0 else None
        output[person]["quality_count"] = q_count

    return {"month": month, "persons": output}


# ── NFC tag endpoints ─────────────────────────────────────────────────────────

NFC_SECRET = "hm2026"

def _nfc_token(chore_id: int, person: str) -> str:
    raw = f"{chore_id}{person}{NFC_SECRET}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


@app.get("/api/nfc/token")
async def get_nfc_token(chore_id: int, person: str):
    token = _nfc_token(chore_id, person)
    return {"chore_id": chore_id, "person": person, "token": token,
            "url": f"/nfc/complete?chore_id={chore_id}&person={person}&token={token}"}


@app.get("/nfc/complete", response_class=HTMLResponse)
async def nfc_complete(chore_id: int, person: str, token: str,
                       db: AsyncSession = Depends(get_db)):
    expected = _nfc_token(chore_id, person)
    if token != expected:
        return HTMLResponse("<html><body><h2>❌ Invalid token</h2></body></html>", status_code=403)
    chore = await db.get(Chore, chore_id)
    if not chore:
        return HTMLResponse("<html><body><h2>❌ Chore not found</h2></body></html>", status_code=404)
    chore_name = chore.chore_name
    target_date = _today()
    ca = ChoreAssessment(
        person_name=person,
        chore_name=chore_name,
        status="done",
        score=10,
        points_earned=chore.points,
        assessment_text="Recorded via NFC tag",
        commentary_text="",
        assessed_date=target_date,
        time_spent_mins=0,
    )
    db.add(ca)
    await db.commit()
    display_person = "Dad" if person == "Liam" else "Mom" if person == "Rachel" else person
    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="3;url=/">
<title>NFC Complete</title>
<style>body{{font-family:system-ui,sans-serif;background:#0d0f1a;color:#e2e8f0;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center}}
.box{{background:#141726;border-radius:16px;padding:2rem 2.5rem;border:1px solid #1e2540;box-shadow:0 8px 32px rgba(0,0,0,.4)}}
h1{{font-size:2.5rem;margin:.3rem 0}}.name{{color:#60a5fa;font-weight:700}}.chore{{color:#4ade80;font-weight:700}}
p{{color:#94a3b8;font-size:.9rem}}</style></head>
<body><div class="box">
<h1>✅</h1>
<h2><span class="chore">{chore_name}</span> marked complete<br>for <span class="name">{display_person}</span>!</h2>
<p>+{chore.points} pts · Redirecting to dashboard in 3 seconds…</p>
</div></body></html>"""
    return HTMLResponse(html)


@app.get("/nfc", response_class=HTMLResponse)
async def nfc_setup_page(db: AsyncSession = Depends(get_db)):
    chores_res = await db.execute(
        select(Chore).where(Chore.active == True)
        .order_by(Chore.person_name, Chore.chore_name))
    chores = chores_res.scalars().all()

    rows = ""
    for c in chores:
        if c.person_name in ("Liam", "Rachel"):
            tok = _nfc_token(c.id, c.person_name)
            url = f"/nfc/complete?chore_id={c.id}&person={c.person_name}&token={tok}"
            display = "Dad" if c.person_name == "Liam" else "Mom"
            rows += f"<tr><td>{display}</td><td>{c.chore_name}</td><td>{c.frequency}</td><td>{c.points}pts</td><td><a href='{url}' style='color:#60a5fa;word-break:break-all'>{url}</a></td></tr>\n"

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFC Setup</title>
<style>body{{font-family:system-ui,sans-serif;background:#0d0f1a;color:#e2e8f0;padding:1.5rem;max-width:1100px;margin:0 auto}}
h1{{color:#60a5fa;margin-bottom:1rem}}table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{text-align:left;padding:.5rem .7rem;color:#94a3b8;border-bottom:2px solid #1e2540;font-size:.78rem;text-transform:uppercase}}
td{{padding:.45rem .7rem;border-bottom:1px solid #1e2540}}tr:hover td{{background:#141726}}
a{{color:#60a5fa}}.back{{color:#94a3b8;font-size:.85rem;margin-bottom:1rem;display:block}}</style></head>
<body>
<a class="back" href="/">← Back to Dashboard</a>
<h1>📡 NFC Tag Setup</h1>
<p style="color:#94a3b8;margin-bottom:1rem">Copy a URL below and write it to an NFC tag. Tapping the tag will mark that chore as done.</p>
<table><thead><tr><th>Person</th><th>Chore</th><th>Frequency</th><th>Points</th><th>NFC URL</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>"""
    return HTMLResponse(html)


# ── Pages ─────────────────────────────────────────────────────────────────────

def _html(name): return (BASE_DIR / "templates" / name).read_text(encoding="utf-8")

@app.get("/", response_class=HTMLResponse)
async def dashboard(): return _html("dashboard.html")

@app.get("/camera", response_class=HTMLResponse)
async def camera_page(): return _html("camera.html")
