import os
from datetime import datetime, timedelta, time
from typing import Any, Optional, List, Tuple, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from requests.exceptions import RequestException, Timeout
from zoneinfo import ZoneInfo

# ====== CONFIG ======
APP_TZ = ZoneInfo(os.environ.get("TZ", "America/New_York"))
HCP_KEY = os.environ.get("HCP_API_KEY")
HCP_BASE = "https://api.housecallpro.com"

# Бизнес-правила LocalPRO
VISIT_MINUTES_DEFAULT = 120   # длительность визита: 2 часа
DRIVE_BUFFER_MINUTES  = 20    # буфер на дорогу (на этапе маршрутов)
DEFAULT_WORK_START = time(8, 0)   # 08:00
DEFAULT_WORK_END   = time(18, 0)  # 18:00

# ====== APP ======
app = FastAPI(title="LocalPRO Scheduler Bridge")

# ====== INPUT MODELS ======
class SuggestIn(BaseModel):
    address: str
    days_ahead: int = 3
    preferred_days: Optional[List[str]] = None  # ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    window_start: Optional[str] = None          # "09:00"
    window_end: Optional[str] = None            # "16:00"
    job_type: Optional[str] = None
    visit_minutes: Optional[int] = None

# ====== HCP HELPERS ======
def hcp_get(path: str, params=None):
    if not HCP_KEY:
        raise HTTPException(status_code=500, detail="HCP_API_KEY is not configured")
    try:
        r = requests.get(
            f"{HCP_BASE}{path}",
            headers={"Accept": "application/json", "Authorization": f"Token {HCP_KEY}"},
            params=params or {},
            timeout=15,  # делаем короче, чтобы не вылетать в 502 по таймауту
        )
        r.raise_for_status()
        return r.json()
    except Timeout as e:
        raise HTTPException(status_code=504, detail=f"HCP timeout on {path}") from e
    except RequestException as e:
        status = getattr(e.response, "status_code", 502)
        raise HTTPException(status_code=status, detail=f"HCP error on {path}") from e

# ====== MISC HELPERS ======
def address_to_str(addr):
    if not isinstance(addr, dict):
        return str(addr)
    parts = []
    for k in ["street", "street_line_2", "city", "state", "zip", "country"]:
        v = addr.get(k)
        if v:
            parts.append(str(v))
    return ", ".join(parts)

def parse_iso_utc(s: str) -> datetime:
    if not isinstance(s, str):
        raise ValueError("Not a string")
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("UTC"))

def to_local(dt_utc: datetime) -> datetime:
    return dt_utc.astimezone(APP_TZ)

def from_local(naive_date: datetime, hhmm: Optional[str]) -> Optional[datetime]:
    if not hhmm:
        return None
    h, m = [int(x) for x in hhmm.split(":")]
    # гарантируем tz-aware
    if naive_date.tzinfo is None:
        naive_date = naive_date.replace(tzinfo=APP_TZ)
    return naive_date.replace(hour=h, minute=m, second=0, microsecond=0)

def weekday_code(dt_local: datetime) -> str:
    return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dt_local.weekday()]

def _walk(obj: Any, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            yield from _walk(v, new_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = f"{path}[{i}]"
            yield from _walk(v, new_path)
    else:
        yield (path, obj)

def pick_time_and_tech(job: dict) -> Tuple[Optional[datetime], Optional[datetime], Optional[str], Optional[str]]:
    start_utc = end_utc = None
    employee_id = None

    schedule = job.get("schedule") or {}
    s = schedule.get("scheduled_start")
    e = schedule.get("scheduled_end")
    if isinstance(s, str):
        start_utc = parse_iso_utc(s)
    if isinstance(e, str):
        end_utc = parse_iso_utc(e)

    assigned_employees = job.get("assigned_employees")
    if isinstance(assigned_employees, list) and assigned_employees:
        first = assigned_employees[0]
        if isinstance(first, dict):
            employee_id = first.get("id")

    addr_str = address_to_str(job.get("address") or job.get("service_address") or {})

    if not start_utc or not end_utc:
        for p, v in _walk(job):
            low = p.lower()
            if isinstance(v, str) and "start" in low and not start_utc and any(ch.isdigit() for ch in v):
                try:
                    start_utc = parse_iso_utc(v)
                except Exception:
                    pass
            if isinstance(v, str) and "end" in low and not end_utc and any(ch.isdigit() for ch in v):
                try:
                    end_utc = parse_iso_utc(v)
                except Exception:
                    pass

    return start_utc, end_utc, employee_id, addr_str

# ====== DEBUG ROUTES ======
@app.get("/health")
def health():
    return {"status": "ok", "tz": str(APP_TZ)}

@app.get("/debug/hcp-company")
def debug_hcp_company():
    data = hcp_get("/company")
    return {"name": data.get("name", "unknown"), "ok": True}

@app.get("/debug/hcp-jobs")
def debug_hcp_jobs():
    data = hcp_get("/jobs")
    jobs = []
    raw = data if isinstance(data, list) else data.get("results") or data.get("jobs") or []
    for item in (raw[:5] if isinstance(raw, list) else []):
        jobs.append({
            "id": item.get("id"),
            "address": item.get("address") or item.get("full_address") or item.get("location")
        })
    return {"count_preview": len(jobs), "jobs": jobs}

@app.get("/debug/hcp-one-raw")
def debug_hcp_one_raw():
    data = hcp_get("/jobs")
    raw = data if isinstance(data, list) else data.get("results") or data.get("jobs") or []
    if not isinstance(raw, list) or not raw:
        return {"ok": True, "job": None}
    item = raw[0]
    keep_keys = [
        "id","address","service_address","schedule",
        "assigned_employees","assigned_to","technician","user",
        "status","title","type"
    ]
    trimmed = {k: item.get(k) for k in keep_keys if k in item}
    return {"ok": True, "job": trimmed}

@app.get("/debug/hcp-jobs-compact")
def debug_hcp_jobs_compact():
    data = hcp_get("/jobs")
    raw = data if isinstance(data, list) else data.get("results") or data.get("jobs") or []
    out = []
    for item in (raw[:5] if isinstance(raw, list) else []):
        start_utc, end_utc, emp_id, _addr = pick_time_and_tech(item)
        out.append({
            "id": item.get("id"),
            "address": address_to_str(item.get("address") or item.get("service_address") or {}),
            "start": start_utc.isoformat() if start_utc else None,
            "end":   end_utc.isoformat() if end_utc else None,
            "employee_id": emp_id
        })
    return {"count_preview": len(out), "jobs": out}

@app.get("/debug/hcp-users")
def debug_hcp_users():
    data = hcp_get("/users")
    raw = data if isinstance(data, list) else data.get("results") or data.get("users") or []
    out = []
    for u in (raw[:10] if isinstance(raw, list) else []):
        out.append({
            "id": u.get("id"),
            "name": u.get("name") or u.get("full_name") or u.get("first_name"),
            "email": u.get("email"),
            "role": u.get("role") or u.get("type")
        })
    return {"count_preview": len(out), "users": out}

# ====== CORE SCHEDULING (без расстояний) ======
def collect_busy_windows(days_ahead: int) -> Tuple[Dict[str, List[Tuple[datetime, datetime, str]]], Dict[str, str]]:
    jobs_data = hcp_get("/jobs")
    raw_jobs = jobs_data if isinstance(jobs_data, list) else jobs_data.get("results") or jobs_data.get("jobs") or []

    users_data = hcp_get("/users")
    raw_users = users_data if isinstance(users_data, list) else users_data.get("results") or users_data.get("users") or []
    name_by_emp: Dict[str, str] = {}
    for u in (raw_users if isinstance(raw_users, list) else []):
        uid = u.get("id")
        nm = u.get("name") or u.get("full_name") or u.get("first_name")
        if uid:
            name_by_emp[uid] = nm or "Technician"

    now_local = datetime.now(APP_TZ)
    horizon_end = now_local + timedelta(days=max(1, days_ahead))

    busy_by_emp: Dict[str, List[Tuple[datetime, datetime, str]]] = {}
    for j in (raw_jobs if isinstance(raw_jobs, list) else []):
        start_utc, end_utc, emp_id, addr_str = pick_time_and_tech(j)
        if not (start_utc and end_utc and emp_id):
            continue
        start_local = to_local(start_utc)
        end_local = to_local(end_utc)
        if end_local < now_local or start_local > horizon_end:
            continue
        busy_by_emp.setdefault(emp_id, []).append((start_local, end_local, addr_str))

    for emp_id in busy_by_emp:
        busy_by_emp[emp_id].sort(key=lambda t: t[0])

    return busy_by_emp, name_by_emp

def generate_free_slots(
    busy_by_emp: Dict[str, List[Tuple[datetime, datetime, str]]],
    name_by_emp: Dict[str, str],
    days_ahead: int,
    preferred_days: Optional[List[str]],
    window_start_str: Optional[str],
    window_end_str: Optional[str],
    visit_minutes: int
) -> List[dict]:
    now_local = datetime.now(APP_TZ)
    horizon_end = now_local + timedelta(days=max(1, days_ahead))
    results: List[dict] = []

    for emp_id, intervals in busy_by_emp.items():
        day_ptr = now_local.date()
        while datetime.combine(day_ptr, DEFAULT_WORK_START, tzinfo=APP_TZ) < horizon_end:
            day_start = datetime.combine(day_ptr, DEFAULT_WORK_START, tzinfo=APP_TZ)
            day_end   = datetime.combine(day_ptr, DEFAULT_WORK_END, tzinfo=APP_TZ)

            wstart = from_local(day_start, window_start_str) or day_start
            wend   = from_local(day_start, window_end_str) or day_end
            if wend <= wstart:
                wend = day_end

            if preferred_days and weekday_code(day_start) not in set(preferred_days):
                day_ptr = (day_ptr + timedelta(days=1))
                continue

            day_busy = []
            for s, e, _addr in intervals:
                if e <= day_start or s >= day_end:
                    continue
                day_busy.append((max(s, wstart), min(e, wend)))

            day_busy.append((wstart, wstart))
            day_busy.append((wend, wend))
            day_busy = sorted(day_busy, key=lambda t: t[0])

            for i in range(len(day_busy) - 1):
                gap_start = day_busy[i][1]
                gap_end   = day_busy[i+1][0]
                if (gap_end - gap_start).total_seconds() >= visit_minutes * 60:
                    results.append({
                        "employee_id": emp_id,
                        "tech_name": name_by_emp.get(emp_id) or "Technician",
                        "slot_start": gap_start.isoformat(),
                        "slot_end": (gap_start + timedelta(minutes=visit_minutes)).isoformat(),
                        "window_end": gap_end.isoformat(),
                        "notes": "free window"
                    })
            day_ptr = (day_ptr + timedelta(days=1))

    results.sort(key=lambda x: x["slot_start"])
    return results[:6]

# ====== MAIN SUGGEST ======
@app.post("/suggest")
def suggest(body: SuggestIn):
    try:
        visit_minutes = body.visit_minutes or VISIT_MINUTES_DEFAULT
        busy_by_emp, name_by_emp = collect_busy_windows(days_ahead=body.days_ahead)

        if not busy_by_emp:
            return {
                "address": body.address,
                "timezone": str(APP_TZ),
                "suggestions": [],
                "info": "No scheduled jobs found in horizon; try increasing days_ahead or ensure jobs are scheduled."
            }

        slots = generate_free_slots(
            busy_by_emp=busy_by_emp,
            name_by_emp=name_by_emp,
            days_ahead=body.days_ahead,
            preferred_days=body.preferred_days,
            window_start_str=body.window_start,
            window_end_str=body.window_end,
            visit_minutes=visit_minutes
        )
        return {
            "address": body.address,
            "timezone": str(APP_TZ),
            "visit_minutes": visit_minutes,
            "suggestions": slots
        }
    except Exception as e:
        # Вместо 502 вернём понятное сообщение
        return {
            "address": body.address,
            "timezone": str(APP_TZ),
            "error": str(e)
        }
