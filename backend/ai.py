import os
import base64
import io
import json
import re
from anthropic import AsyncAnthropic
from PIL import Image

client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── Household context ─────────────────────────────────────────────────────────
HOUSEHOLD_CONTEXT = """
Household: Liam and Rachel are the two adults responsible for all chores.
Children: Ben (age 7), Louis (age 4), Lidia (age 1.5).
The children may cause messes but are not responsible for chores.
When spotting violations, note if a child is likely responsible vs adult negligence.
"""

# ── Rework violation codes ────────────────────────────────────────────────────
VIOLATION_PROMPT = """
Also scan for these REWORK VIOLATIONS and return them in the violations list:
- SINK_PLATES: Plates, bowls, cups, cutlery, or pots LEFT in the kitchen sink long-term
  (EXCEPTION: items placed there temporarily on way to dishwasher are acceptable)
- FOOD_IN_SINK: Food scraps or debris emptied into the sink without clearing the drain
- CARDBOARD_NOT_RECYCLED: Cardboard boxes or packaging left in the kitchen instead of recycling area
- DRYER_LINT_NOT_BINNED: Dryer lint emptied onto a surface or floor rather than into a bin
- SURFACE_NOT_PROPERLY_WIPED: A visibly dirty/sticky surface not wiped with cloth and cleaning product
- LAUNDRY_NOT_SORTED: Laundry left as one big unsorted pile instead of sorted by person (Ben/Louis/Lidia/adults)
- CLUTTER_HIDDEN: Items stuffed into a bag, box, pile or container to clear space rather than properly put away

MARGIN RULE — these must NEVER be flagged, even if clearly visible:
- A few items on the kitchen counter (cups, appliances, condiments, keys) — counters are ALWAYS in use
- A single dirty plate, glass or mug left in or near the sink temporarily
- Children's toys anywhere — children make messes, that is normal
- Remote controls, books, papers, bags, chargers, phones on surfaces
- Any item that has an obvious, legitimate reason to be where it is

Only flag something if the space looks SEVERELY and OBVIOUSLY neglected — e.g. a sink
completely full of dishes that have been there a long time, a pile of rubbish on the floor,
a clearly soiled surface that nobody has touched in days.

SURFACE_NOT_PROPERLY_WIPED: Only flag if the surface is visibly dirty with food, liquid or
grease residue. Do NOT flag a counter just because there are items on it or it looks used.

CONFIDENCE REQUIREMENT: You must be at least 95% certain before flagging any violation.
If there is ANY reasonable interpretation that makes the scene acceptable, do NOT flag it.
It is far better to miss 10 genuine violations than to falsely accuse someone once.

For each violation: code, description (what you see), callout (fun 70s presenter message using person's name).
Return empty list [] if no violations found.
"""

# ── Bias-free assessment rules ────────────────────────────────────────────────
ASSESSMENT_RULES = """
ASSESSMENT STANDARDS (apply strictly and without bias):
- A PERFECT clean (9-10): Everything cleared and put in its correct place.
  Surfaces wiped with cloth AND cleaning product. Floor swept AND mopped.
  No visible clutter or out-of-place items. No toys or things left around.
- A GOOD clean (7-8): As above but minor areas missed (e.g. one corner, one surface).
- PARTIAL (5-6): Visible effort but key elements missing (e.g. wiped but no product used,
  cleared but not put away properly, swept but not mopped).
- POOR (3-4): Minimal effort. Some clearing but surfaces dirty, items left out.
- NOT DONE (0-2): Essentially untouched or barely started.

BONUS: If the person is observed putting items back in their exact original place,
add +1 point and note it. This rewards proper tidying behaviour.

Make NO assumptions. Base all assessments strictly on what is visually observable.
Do not assume something is clean because it looks tidy from a distance.
"""


def _to_jpeg_b64(data: bytes) -> str:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


async def describe_face(image_bytes: bytes) -> str:
    b64 = _to_jpeg_b64(image_bytes)
    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=400,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": (
                "You are helping a family set up a home recognition system. "
                "Describe the visual appearance of the main person in this photo. "
                "Include: hair colour and style, approximate age, skin tone, eye colour, "
                "glasses, facial hair, face shape, and any other visible features. "
                "Be factual and visual only. Output plain text."
            )},
        ]}],
    )
    return msg.content[0].text.strip()


async def analyse_frame(image_bytes: bytes, enrolled_persons: list[dict]) -> dict:
    b64 = _to_jpeg_b64(image_bytes)

    if enrolled_persons:
        profiles = "\n".join(
            f'- ID {p["id"]}: {p["name"]} — {p["face_description"]}'
            for p in enrolled_persons
        )
        identity_prompt = (
            f"Known people in this home:\n{profiles}\n\n"
            "Match the person in the image to a profile if possible. "
        )
    else:
        identity_prompt = "No enrolled profiles yet. "

    prompt = (
        HOUSEHOLD_CONTEXT + "\n\n"
        + identity_prompt
        + VIOLATION_PROMPT + "\n\n"
        "Return a JSON object with these exact keys:\n"
        '  "person_id": matching person ID or null,\n'
        '  "person_name": matching name or "Unknown",\n'
        '  "task": one concise sentence describing exactly what the person is physically doing right now,\n'
        '  "activity_type": MUST be exactly one of the values below. Choose based solely on what you\n'
        '    can visually observe the person ACTIVELY doing this moment — not the state of the room.\n\n'
        '    "cleaning"  — ONLY when you can see active physical cleaning effort: wiping/scrubbing a\n'
        '                  surface with a cloth or sponge, vacuuming, mopping, sweeping, washing dishes\n'
        '                  at the sink, loading/unloading the dishwasher, folding/putting away laundry,\n'
        '                  tidying by physically picking things up and putting them away.\n'
        '                  DO NOT use "cleaning" if the person is merely standing in a tidy room,\n'
        '                  walking past a clean surface, or sitting anywhere.\n\n'
        '    "cooking"   — actively preparing food: chopping, stirring, using the hob/oven/microwave,\n'
        '                  washing vegetables, plating up food.\n\n'
        '    "eating"    — consuming food or drink (at table, counter, sofa, or anywhere).\n'
        '                  Counts as relaxing/personal time — NOT cooking, NOT cleaning.\n\n'
        '    "tv"        — watching a TV screen, laptop video, or phone video from a relaxed seated\n'
        '                  or reclined position.\n\n'
        '    "resting"   — lying down, napping, eyes closed, clearly motionless and inactive.\n\n'
        '    "personal"  — any passive/leisure activity that is NOT one of the above:\n'
        '                  sitting on a sofa or chair, browsing phone/tablet, reading, chatting\n'
        '                  with another adult, self-care (hair, makeup), just standing around.\n'
        '                  USE THIS for anyone who is sitting on the couch or sofa in any capacity.\n\n'
        '    "family"    — ONLY when physically and actively spending time WITH the children:\n'
        '                  helping a child with homework, feeding a child (putting food in front\n'
        '                  of them or spoon-feeding), getting a child dressed or ready for school,\n'
        '                  playing a board game / card game / physical game WITH the children,\n'
        '                  reading to a child, cleaning up a mess made by the children (e.g.\n'
        '                  wiping food off a child, picking up toys the children left out).\n'
        '                  The adult must be actively engaged WITH the child — not just nearby.\n'
        '                  A parent sitting on the sofa while children play alone = "personal".\n\n'
        '    "other"     — working at a desk/computer for non-leisure purposes, exercising,\n'
        '                  carrying bags in/out of the house, anything else not covered above.\n\n'
        '  CLASSIFICATION RULES (apply in order — first match wins):\n'
        '    1. Sitting or lying anywhere → "tv", "resting", "personal", or "eating". NEVER "cleaning".\n'
        '    2. Only classify as "cleaning" if you see a cleaning tool in hand or active physical effort.\n'
        '    3. A tidy room does not mean the person is cleaning. A messy room does not mean they are not.\n'
        '    4. When unsure between "cleaning" and anything else → choose the non-cleaning option.\n'
        '    5. When unsure between "personal" and "other" → choose "personal".\n\n'
        '  "confidence": float 0.0-1.0 for how confident you are in the identity match (not activity),\n'
        '  "etiquette_violation": null or brief description of any obvious etiquette issue seen,\n'
        '  "etiquette_nudge": null or a short funny message calling it out,\n'
        '  "violations": array of {code, description, callout} for rework violations, or []\n'
        "Output only valid JSON, no markdown fences."
    )

    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=600,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    raw = _strip_fences(msg.content[0].text)
    try:
        result = json.loads(raw)
        valid_types = {"cleaning", "cooking", "eating", "tv", "resting", "personal", "family", "other"}
        if result.get("activity_type") not in valid_types:
            result["activity_type"] = "other"
        if "violations" not in result:
            result["violations"] = []
        return result
    except json.JSONDecodeError:
        return {"person_id": None, "person_name": "Unknown", "task": raw[:200],
                "activity_type": "other", "confidence": 0.0,
                "etiquette_violation": None, "etiquette_nudge": None, "violations": []}


async def assess_chore(image_bytes: bytes, person_name: str, chore_name: str,
                       duration_mins: int = 0) -> dict:
    b64 = _to_jpeg_b64(image_bytes)
    dur_note = f"The person has been working on this for approximately {duration_mins} minute(s)." if duration_mins > 0 else ""
    prompt = (
        HOUSEHOLD_CONTEXT + "\n\n"
        + ASSESSMENT_RULES + "\n\n"
        f"You are a flamboyant 1970s American TV game show presenter assessing whether "
        f"{person_name} has completed the chore: '{chore_name}'. {dur_note}\n\n"
        "SCORING SCALE:\n"
        "- 9-10: 'A PERFECT performance, the crowd goes wild!'\n"
        "- 7-8:  'Solid work, not quite Hall of Fame but we're talking!'\n"
        "- 5-6:  'Mmm, we've seen better days on this stage, folks.'\n"
        "- 3-4:  'Oh dear oh dear, the judges are NOT impressed.'\n"
        "- 1-2:  'Almost nothing here — practically invisible effort!'\n"
        "- 0:    Not done — generate a funny personalised voice reminder.\n\n"
        + VIOLATION_PROMPT + "\n\n"
        "Return a JSON object with these exact keys:\n"
        '  "status": "done", "partial", or "not_done",\n'
        '  "score": integer 0-10,\n'
        '  "assessment": 2-3 sentences — what you see and your conclusion (factual, no assumptions),\n'
        '  "commentary": theatrical 70s TV host announcement of the score (1-2 sentences),\n'
        f'  "reminder": if score 0-2, a funny personalised reminder for {person_name}; otherwise null,\n'
        '  "time_estimate": estimated minutes to complete this chore properly (integer),\n'
        '  "violations": array of {code, description, callout} for any rework violations spotted.\n'
        "Output only valid JSON, no markdown."
    )
    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=700,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    raw = _strip_fences(msg.content[0].text)
    try:
        result = json.loads(raw)
        if "violations" not in result:
            result["violations"] = []
        return result
    except json.JSONDecodeError:
        return {"status": "unknown", "score": 0, "assessment": raw[:300],
                "commentary": None, "reminder": None, "time_estimate": 0, "violations": []}


async def suggest_chore_points(chores_by_person: dict[str, list[dict]]) -> dict:
    lines = []
    for person, chores in chores_by_person.items():
        daily = [c for c in chores if c["frequency"] == "daily"]
        other = [c for c in chores if c["frequency"] != "daily"]
        if daily:
            lines.append(f"{person}'s daily chores: " +
                         ", ".join(f'{c["name"]}' for c in daily))
        if other:
            lines.append(f"{person}'s other chores: " +
                         ", ".join(f'{c["name"]} [{c["frequency"]}]' for c in other))

    prompt = (
        "You are helping fairly distribute household chore points between two adults.\n"
        "Rules:\n"
        "- Each person's DAILY chores should total exactly 100 points.\n"
        "- Weekly chores: 14-20 points each.\n"
        "- Monthly chores: 5-10 points each.\n"
        "- Base points on effort — be fair and unbiased.\n\n"
        "Chore lists:\n" + "\n".join(lines) + "\n\n"
        "Return JSON: keys are person names, values map chore name to integer points. "
        "Output only valid JSON."
    )
    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _strip_fences(msg.content[0].text)
    try:
        return json.loads(raw)
    except Exception:
        return {}


async def announce_morning_chores(person_name: str, chores: list[dict]) -> str:
    core = [c["chore_name"] for c in chores if c.get("chore_type") == "core"]
    rotating = [c["chore_name"] for c in chores if c.get("chore_type") == "rotating"]
    other = [c["chore_name"] for c in chores
             if c.get("chore_type") not in ("core", "rotating")]
    lines = []
    if core:
        lines.append("Core duties (every day): " + ", ".join(core))
    if rotating:
        lines.append("Today's rotating chores: " + ", ".join(rotating))
    if other:
        lines.append("Other tasks: " + ", ".join(other))
    prompt = (
        HOUSEHOLD_CONTEXT + "\n\n"
        f"You are a flamboyant 1970s American TV game show presenter. "
        f"{person_name} has just come downstairs for the morning. "
        f"Announce their chores for the day with theatrical flair and humour. "
        f"Be warm, dramatic, encouraging. Use their name. Under 90 words.\n\n"
        f"{person_name}'s chores today:\n" + "\n".join(lines)
    )
    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


async def generate_comparison_report(chore_stats: dict) -> str:
    lines = []
    for chore, by_person in chore_stats.items():
        parts = []
        for person, stats in by_person.items():
            mins = f", ~{stats['total_mins']}min total" if stats['total_mins'] else ""
            parts.append(
                f"{person}: {stats['count']}x done, avg score {stats['avg_score']:.1f}/10{mins}"
            )
        lines.append(f"{chore}: " + " | ".join(parts))

    prompt = (
        HOUSEHOLD_CONTEXT + "\n\n"
        "You are a flamboyant 1970s TV presenter delivering a CHORE COMPARISON REPORT for Liam and Rachel.\n\n"
        "Data (per chore, who has done it more and at what quality):\n"
        + "\n".join(lines) + "\n\n"
        "Deliver a dramatic, personalised analysis. Call out:\n"
        "1. Who does each chore more consistently\n"
        "2. Who does them to a higher standard (avg score)\n"
        "3. Any chores one person dominates entirely\n"
        "4. Gentle roasting of the lower performer by name\n"
        "5. Encouraging sign-off\n"
        "Keep it fun, under 250 words. Plain text."
    )
    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


async def generate_summary(assessments_by_person: dict[str, list[dict]],
                           period: str = "daily") -> str:
    lines = []
    for person, items in assessments_by_person.items():
        total_score = sum(i.get("score", 0) for i in items)
        avg_score = total_score / len(items) if items else 0
        done = sum(1 for i in items if i.get("status") == "done")
        total_pts = sum(i.get("points_earned", 0) for i in items)
        lines.append(
            f"{person}: {done}/{len(items)} chores done, avg score {avg_score:.1f}/10, "
            f"{total_pts} points earned. "
            "Chores: " + ", ".join(f'{i["chore_name"]} ({i["score"]}/10)' for i in items)
        )
    period_label = {"daily": "TODAY'S", "weekly": "THIS WEEK'S",
                    "bimonthly": "BI-MONTHLY", "monthly": "THIS MONTH'S"}.get(period, period.upper() + "'S")
    prompt = (
        HOUSEHOLD_CONTEXT + "\n\n"
        f"You are a flamboyant 1970s TV presenter delivering {period_label} chore performance summary.\n\n"
        "Data:\n" + "\n".join(lines) + "\n\n"
        "Deliver a dramatic, fun, personalised summary. Celebrate high scorers, "
        "gently roast low scorers, end with an encouraging sign-off. Under 200 words. Plain text."
    )
    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


async def check_point_weights(chores_by_person: dict) -> list:
    """Analyse chore list and return a list of potentially mis-weighted chores."""
    lines = []
    for person, chores in chores_by_person.items():
        for c in chores:
            lines.append(
                f"{person}: {c['name']} [{c['frequency']}] = {c['points']} pts"
            )
    prompt = (
        "You are a fair household chore analyst. Review this chore point list and identify "
        "any chores that appear significantly over-weighted or under-weighted compared to "
        "the effort and time each chore typically takes.\n\n"
        "Rough benchmarks:\n"
        "- A 2-minute task: ~5-8 pts\n"
        "- A 5-minute task: ~10-15 pts\n"
        "- A 15-minute task: ~20-30 pts\n"
        "- A 30-minute task: ~35-50 pts\n"
        "- A 60-minute task: ~55-70 pts\n"
        "- Daily totals per person should be close to equal (within 15%)\n"
        "- Weekly/monthly chores: proportionally less per minute (they're optional)\n\n"
        "Chore list:\n" + "\n".join(lines) + "\n\n"
        "Return ONLY a JSON array. Each element: "
        "{\"chore_name\": str, \"person\": str, \"current_points\": int, "
        "\"suggested_points\": int, \"reason\": str}. "
        "Only flag chores that are >30% off from the expected range. "
        "If everything looks fair, return []."
    )
    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _strip_fences(msg.content[0].text)
    try:
        return json.loads(raw)
    except Exception:
        return []


async def recheck_violation(image_bytes: bytes, violation_code: str,
                             original_description: str) -> dict:
    """Re-examine image for a specific violation. Returns {confirmed, confidence, reason}."""
    b64 = _to_jpeg_b64(image_bytes)
    prompt = (
        HOUSEHOLD_CONTEXT + "\n\n"
        f"A violation was flagged: {violation_code}\n"
        f"Original description: {original_description}\n\n"
        "Please carefully re-examine this image with fresh eyes.\n"
        "IMPORTANT: Only confirm if you are MORE THAN 90% certain this is a genuine violation.\n"
        "If there is ANY reasonable doubt, return confirmed=false — the person gets the benefit of the doubt.\n\n"
        'Return JSON: {"confirmed": bool, "confidence": float 0.0-1.0, "reason": "one concise sentence"}\n'
        "Output only valid JSON."
    )
    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=150,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    raw = _strip_fences(msg.content[0].text)
    try:
        return json.loads(raw)
    except Exception:
        return {"confirmed": False, "confidence": 0.0, "reason": "Unable to re-assess — benefit of the doubt applies"}


async def generate_trend_report(fact_data: dict) -> str:
    """
    fact_data: {person_name: {chore_counts, avg_scores, violation_counts,
                              kitchen_mins, family_mins, personal_mins, days_seen}}
    Generates a 2-week trend observation report.
    """
    lines = []
    for person, d in fact_data.items():
        lines.append(
            f"{person} over past 2 weeks: "
            f"chores done {d.get('total_done',0)}x, avg score {d.get('avg_score',0):.1f}/10, "
            f"{d.get('violation_count',0)} rework violations, "
            f"~{d.get('kitchen_mins',0):.0f}min kitchen, "
            f"~{d.get('family_mins',0):.0f}min family time, "
            f"~{d.get('personal_mins',0):.0f}min personal time, "
            f"seen {d.get('days_seen',0)} days."
        )
    prompt = (
        HOUSEHOLD_CONTEXT + "\n\n"
        "You are a home productivity analyst delivering a BI-WEEKLY TREND REPORT.\n\n"
        "Raw data for the past 14 days:\n" + "\n".join(lines) + "\n\n"
        "Produce specific, factual observations and trends. Examples:\n"
        "- Who is more consistent at doing chores\n"
        "- Who spends more time with the kids\n"
        "- Who creates more rework violations\n"
        "- Any patterns in personal/leisure time\n"
        "- Areas of improvement for each person\n"
        "Be specific with numbers. Use names. Under 300 words. Plain text."
    )
    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()
