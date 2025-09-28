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
GMAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")  # Google Distance Matrix

# ====== FALLBACK NAMES & HOMES ======
NAME_MAP: Dict[str, str] = {
    "pro_5d48854aad6542a28f9da12d0c1b65f2": "Alex Yakush",
    "pro_e07e8bc2e5464dfdba36866c66a5f62d": "Vladimir Kovalev",
    "pro_17e6723ece4e47af95bcb6c1766bbb47": "Nick Litvinov",
}
HOME_MAP: Dict[str, str] = {
    # "pro_...": "City, ST"
}

# ====== BUSINESS RULES (v4.3) ======
VISIT_MINUTES_DEFAULT = 120
DEFAULT_WORK_START = time(9, 0)
DEFAULT_WORK_END   = time(18, 0)

# ETA thresholds
VERY_CLOSE_THRESH = 15   # <=15 -> low
NORMAL_THRESH     = 40   # 16..40 -> medium; >40 -> high

# 3-hour grid by whole hours
GRID_STARTS = [9, 10, 11, 12, 13, 14, 15]  # 9–12,10–1,11–2,12–3,1–4,2–5,3–6

# Saturday: only Alex works
ALEX_ID = "pro_5d48854aad6542a28f9da12d0c1b65f2"

# ====== APP ======
app = FastAPI(title="LocalPRO Scheduler Bridge (v4.3)")

# ====== MODELS ======
class SuggestIn(BaseModel):
    address: str
    days_ahead: int = 8
    preferred_days: Optional[List[str]] = None  # ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    job_type: Optional[str] = None
    visit_minutes: Optional[int] = None
    buffer_minutes: Optional[int] = None

class SuggestOut(BaseModel):
    address: str
    timezone: str
    visit_minutes: int
    suggestions: List[dict]

# ====== HCP helpers (with pagination) ======
def hcp_get(path: str, params=None):
    if not HCP_KEY:
        raise HTTPException(status_code=500, detail="HCP_API_KEY is not configured")
    try:
        r = requests.get(
            f"{HCP_BASE}{path}",
            headers={"Accept": "application/json", "Authorization": f"Token {HCP_KEY}"},
            params=params or {},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Timeout as e:
        raise HTTPException(status_code=504, detail=f"HCP timeout on {path}") from e
    except RequestException as e:
        status = getattr(e.response, "status_code", 502)
        raise HTTPException(status_code=status, detail=f"HCP error on {path}") from e

def hcp_paged_list(path: str, max_pages: int = 12, per_page: int = 100) -> List[dict]:
    """Try common HCP-style pagination: ?page=1..N&per_page=100. Stop when page empty or < per_page."""
    items: List[dict] = []
    for p in range(1, max_pages + 1):
        data = hcp_get(path, params={"page": p, "per_page": per_page})
        arr = data if isinstance(data, list) else data.get("results") or data.get("jobs") or data.get("events") or []
        if not isinstance(arr, list):
            break
        items.extend(arr)
        if len(arr) < per_page:
            break
    # Fallback: if nothing, try without params once
    if not items:
        data = hcp_get(path)
        arr = data if isinstance(data, list) else data.get("results") or data.get("jobs") or data.get("events") or []
        if isinstance(arr, list):
            items = arr
    return items

# ====== UTIL ======
def safe_city(addr_str: str) -> str:
    try:
        parts = [p.strip() for p in addr_str.split(",")]
        for i in range(len(parts)-1):
            if len(parts[i+1]) == 2 and parts[i+1].isalpha():
                return parts[i]
        if len(parts) >= 2:
            return parts[-3] if len(parts) >= 3 else parts[1]
        return addr_str
    except Exception:
        return addr_str

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

def pick_time_and_tech(job: dict) -> Tuple[Optional[datetime], Optional[datetime], Optional[str], str]:
    start_utc = end_utc = None
    employee_id = None

    schedule = job.get("schedule") or {}
    s = schedule.get("scheduled_start") or job.get("scheduled_start") or job.get("start") or job.get("starts_at") or job.get("start_time")
    e = schedule.get("scheduled_end")   or job.get("scheduled_end")   or job.get("end")   or job.get("ends_at")   or job.get("end_time")
    if isinstance(s, str):
        try: start_utc = parse_iso_utc(s)
        except Exception: start_utc = None
    if isinstance(e, str):
        try: end_utc = parse_iso_utc(e)
        except Exception: end_utc = None

    assigned_employees = job.get("assigned_employees")
    if isinstance(assigned_employees, list) and assigned_employees:
        first = assigned_employees[0]
        if isinstance(first, dict):
            employee_id = first.get("id")

    addr_str = address_to_str(job.get("address") or job.get("service_address") or {})

    if not start_utc or not end_utc:
        # last resort scan
        for p, v in _walk(job):
            low = p.lower()
            if isinstance(v, str) and not start_utc and ("start" in low or "begin" in low) and any(ch.isdigit() for ch in v):
                try: start_utc = parse_iso_utc(v)
                except Exception: pass
            if isinstance(v, str) and not end_utc and ("end" in low or "finish" in low) and any(ch.isdigit() for ch in v):
                try: end_utc = parse_iso_utc(v)
                except Exception: pass

    return start_utc, end_utc, employee_id, addr_str

# ====== RESOLVERS ======
def resolve_name(emp_id: str) -> str:
    env_key = f"NAME_PRO_{emp_id}"
    if env_key in os.environ and os.environ[env_key].strip():
        return os.environ[env_key].strip()
    return NAME_MAP.get(emp_id, "Technician")

def resolve_home(emp_id: str) -> Optional[str]:
    env_key = f"HOME_PRO_{emp_id}"
    if env_key in os.environ and os.environ[env_key].strip():
        return os.environ[env_key].strip()
    return HOME_MAP.get(emp_id)

# ====== ETA ======
def google_eta_minutes(src_addr: str, dst_addr: str) -> Optional[int]:
    if not GMAPS_KEY:
        return None
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            params={"origins": src_addr, "destinations": dst_addr, "key": GMAPS_KEY, "units": "imperial"},
            timeout=15
        )
        data = resp.json()
        if data.get("status") != "OK":
            return None
        rows = data.get("rows", [])
        if not rows or not rows[0].get("elements"):
            return None
        el = rows[0]["elements"][0]
        if el.get("status") != "OK":
            return None
        seconds = el["duration"]["value"]
        return int(round(seconds / 60))
    except Exception:
        return None

NEAR_CITIES = {"st. augustine", "st augustine", "jacksonville", "orange park"}
def estimate_eta_heuristic(src_addr: str, dst_addr: str) -> int:
    def city(a: str) -> str:
        try:
            parts = [p.strip() for p in a.split(",")]
            for i in range(len(parts)-1):
                if len(parts[i+1]) == 2 and parts[i+1].isalpha():
                    return parts[i].lower()
            if len(parts) >= 2:
                return parts[-3].lower() if len(parts) >= 3 else parts[1].lower()
            return a.lower()
        except Exception:
            return a.lower()
    a = city(src_addr or "")
    b = city(dst_addr or "")
    if not a or not b:
        return 30
    if a == b:
        return 12
    if a in NEAR_CITIES and b in NEAR_CITIES:
        return 22
    return 45

def eta_minutes(src_addr: str, dst_addr: str) -> int:
    if not src_addr or not dst_addr:
        return 30
    real = google_eta_minutes(src_addr, dst_addr)
    return real if real is not None else estimate_eta_heuristic(src_addr, dst_addr)

def risk_label(eta: int) -> str:
    if eta <= VERY_CLOSE_THRESH:
        return "low"
    if eta <= NORMAL_THRESH:
        return "medium"
    return "high"

# ====== DEBUG ======
@app.get("/health")
def health():
    return {"status": "ok", "tz": str(APP_TZ), "gmaps": bool(GMAPS_KEY)}

@app.get("/debug/hcp-jobs-compact")
def debug_hcp_jobs_compact():
    data = hcp_paged_list("/jobs")
    out = []
    for item in (data[:20] if isinstance(data, list) else []):
        s_utc, e_utc, emp_id, addr_str = pick_time_and_tech(item)
        out.append({
            "id": item.get("id"),
            "address": addr_str,
            "arrival_window_start_local": to_local(s_utc).isoformat() if s_utc else None,
            "arrival_window_end_local":   to_local(e_utc).isoformat() if e_utc else None,
            "employee_id": emp_id,
            "tech_name": resolve_name(emp_id) if emp_id else None
        })
    return {"count_preview": len(out), "jobs": out}

@app.get("/debug/homes")
def debug_homes():
    out = {}
    keys = set(list(HOME_MAP.keys()))
    try:
        data = hcp_paged_list("/jobs")
        for item in (data if isinstance(data, list) else []):
            _, _, emp_id, _ = pick_time_and_tech(item)
            if emp_id:
                keys.add(emp_id)
    except Exception:
        pass
    for k in keys:
        out[k] = resolve_home(k)
    return {"homes": out, "gmaps": bool(GMAPS_KEY)}

# ====== EVENTS SUPPORT ======
def try_collect_events_paged() -> List[dict]:
    try:
        return hcp_paged_list("/events")
    except Exception:
        return []

def parse_event_time(ev: dict) -> Tuple[Optional[datetime], Optional[datetime]]:
    # try multiple common keys
    for k_start, k_end in [
        ("start", "end"),
        ("scheduled_start", "scheduled_end"),
        ("starts_at", "ends_at"),
        ("start_time", "end_time"),
        ("schedule.scheduled_start", "schedule.scheduled_end"),
    ]:
        s = ev.get(k_start)
        e = ev.get(k_end)
        if s and e and isinstance(s, str) and isinstance(e, str):
            try:
                return parse_iso_utc(s), parse_iso_utc(e)
            except Exception:
                continue
        # nested schedule
        if "." in k_start:
            sch = ev.get("schedule") or {}
            s = sch.get("scheduled_start")
            e = sch.get("scheduled_end")
            if isinstance(s, str) and isinstance(e, str):
                try:
                    return parse_iso_utc(s), parse_iso_utc(e)
                except Exception:
                    pass
    return None, None

def event_employee_id(ev: dict) -> Optional[str]:
    ae = ev.get("assigned_employees")
    if isinstance(ae, list) and ae:
        if isinstance(ae[0], dict) and ae[0].get("id"):
            return ae[0]["id"]
    title = (ev.get("title") or ev.get("summary") or ev.get("name") or "").lower()
    for emp_id, name in NAME_MAP.items():
        if name.lower() in title:
            return emp_id
    return None

# ====== WINDOW TYPE ======
# (win_start_local, win_end_local, address_str, is_event)
Window = Tuple[datetime, datetime, str, bool]

def collect_arrival_windows(days_ahead: int) -> Dict[str, List[Window]]:
    """ arrival_by_emp[employee_id] = [(ws,we,addr,is_event), ...] """
    now_local = datetime.now(APP_TZ)
    horizon_end = now_local + timedelta(days=max(1, days_ahead))

    arrival_by_emp: Dict[str, List[Window]] = {}

    # Jobs (paged)
    jobs = hcp_paged_list("/jobs")
    for j in jobs if isinstance(jobs, list) else []:
        start_utc, end_utc, emp_id, addr_str = pick_time_and_tech(j)
        if not (start_utc and end_utc and emp_id):
            continue
        ws = to_local(start_utc)
        we = to_local(end_utc)
        if we < now_local or ws > horizon_end:
            continue
        arrival_by_emp.setdefault(emp_id, []).append((ws, we, addr_str, False))

    # Events (paged, strict blockers)
    events = try_collect_events_paged()
    for ev in events:
        s_utc, e_utc = parse_event_time(ev)
        emp_id = event_employee_id(ev)
        if not (s_utc and e_utc and emp_id):
            continue
        ws = to_local(s_utc)
        we = to_local(e_utc)
        if we < now_local or ws > horizon_end:
            continue
        addr_str = address_to_str(ev.get("address") or {}) or (ev.get("title") or ev.get("summary") or "Event")
        arrival_by_emp.setdefault(emp_id, []).append((ws, we, addr_str, True))

    for emp_id in arrival_by_emp:
        arrival_by_emp[emp_id].sort(key=lambda t: t[0])
    return arrival_by_emp

# ====== GRID & OVERLAP RULES ======
def grid_slot(dt_day: datetime, start_hour: int) -> Tuple[datetime, datetime]:
    s = datetime.combine(dt_day.date(), time(start_hour, 0), tzinfo=APP_TZ)
    e = s + timedelta(hours=3)
    return s, e

def within_client_window(s: datetime, e: datetime, day_start: datetime, day_end: datetime,
                         win_start_str: Optional[str], win_end_str: Optional[str]) -> bool:
    cl_s = from_local(day_start, win_start_str) or day_start
    cl_e = from_local(day_start, win_end_str) or day_end
    return (s >= cl_s) and (e <= cl_e)

def overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)

def overlap_minutes(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if start >= end:
        return 0
    return int((end - start).total_seconds() // 60)

def overlapping_windows(s: datetime, e: datetime, windows: List[Window]) -> List[Window]:
    return [(ws, we, addr, is_event) for ws, we, addr, is_event in windows if overlaps(s, e, ws, we)]

def same_interval_exists(s: datetime, e: datetime, windows: List[Window]) -> bool:
    for ws, we, _, _ in windows:
        if ws == s and we == e:
            return True
    return False

def allowed_step_overlap(prev: Optional[Window],
                         nxt: Optional[Window],
                         new_s: datetime, new_e: datetime,
                         eta_prev_min: Optional[int]) -> bool:
    """
    Only allowed overlaps (v4.3):
    - with EVENT: any overlap -> forbidden
    - with prev (JOB): shifted (prev+1h, ETA≤15) or base (prev+2h). Overlap ≤120 min.
    - with next (JOB): if overlapping, only when next.start == new.start+2h and overlap ≤60 min.
    """
    if prev:
        prev_s, prev_e, _, prev_is_event = prev
        if prev_is_event:
            if overlaps(new_s, new_e, prev_s, prev_e):
                return False
        else:
            prev_start_hour = prev_s.hour
            cond_shifted = (new_s.hour == prev_start_hour + 1) and (eta_prev_min is not None and eta_prev_min <= VERY_CLOSE_THRESH)
            cond_base    = (new_s.hour == prev_start_hour + 2)
            if overlaps(new_s, new_e, prev_s, prev_e):
                if not (cond_shifted or cond_base):
                    return False
                if overlap_minutes(new_s, new_e, prev_s, prev_e) > 120:
                    return False

    if nxt:
        nxt_s, nxt_e, _, nxt_is_event = nxt
        if nxt_is_event:
            if overlaps(new_s, new_e, nxt_s, nxt_e):
                return False
        else:
            if overlaps(new_s, new_e, nxt_s, nxt_e):
                if nxt_s.hour != new_s.hour + 2:
                    return False
                if overlap_minutes(new_s, new_e, nxt_s, nxt_e) > 60:
                    return False

    return True

# ====== BUILD CANDIDATES ======
def build_candidates_for_day(
    emp_id: str,
    dt_day: datetime,
    todays_windows: List[Window],
    new_addr: str,
    win_start_str: Optional[str],
    win_end_str: Optional[str],
    home_addr: Optional[str]
) -> List[dict]:
    day_start = datetime.combine(dt_day.date(), DEFAULT_WORK_START, tzinfo=APP_TZ)
    day_end   = datetime.combine(dt_day.date(), DEFAULT_WORK_END, tzinfo=APP_TZ)

    occupied_hours = set(ws.hour for ws, _, _, _ in todays_windows)
    results: List[dict] = []

    if not todays_windows:
        for h in GRID_STARTS:
            s, e = grid_slot(dt_day, h)
            if not within_client_window(s, e, day_start, day_end, win_start_str, win_end_str):
                continue
            eta_from_home = eta_minutes(home_addr, new_addr) if home_addr else NORMAL_THRESH
            results.append({
                "employee_id": emp_id, "tech_name": resolve_name(emp_id),
                "slot_start": s.isoformat(), "slot_end": e.isoformat(),
                "ETA_prev": eta_from_home, "ETA_next": None,
                "risk": risk_label(eta_from_home), "note": "first of day (home proximity)"
            })
        return results

    anchors = [(None, None, None, None)] + todays_windows + [(None, None, None, None)]
    for i in range(len(anchors)-1):
        prev = anchors[i] if anchors[i][0] is not None else None
        nxt  = anchors[i+1] if anchors[i+1][0] is not None else None

        base_after_prev = 9 if not prev else prev[0].hour + 2
        shifted_after_prev = base_after_prev - 1

        candidate_hours = []
        for h in GRID_STARTS:
            if prev and h <= prev[0].hour - 1:
                continue
            if nxt and h >= nxt[0].hour + 1:
                continue
            candidate_hours.append(h)

        for h in candidate_hours:
            s, e = grid_slot(dt_day, h)
            if not within_client_window(s, e, day_start, day_end, win_start_str, win_end_str):
                continue

            if h in occupied_hours:
                continue
            if same_interval_exists(s, e, todays_windows):
                continue

            # First-of-day candidate (prev=None) cannot overlap anything
            if prev is None and overlapping_windows(s, e, todays_windows):
                continue

            eta_prev = None
            if prev:
                eta_prev = eta_minutes(prev[2], new_addr)
                if h == shifted_after_prev:
                    if not (eta_prev is not None and eta_prev <= VERY_CLOSE_THRESH):
                        continue
                elif h != base_after_prev:
                    continue

            overlaps_list = overlapping_windows(s, e, todays_windows)
            if overlaps_list:
                allowed_refs = set()
                if prev: allowed_refs.add((prev[0], prev[1], prev[2], prev[3]))
                if nxt:  allowed_refs.add((nxt[0], nxt[1], nxt[2], nxt[3]))
                for ow in overlaps_list:
                    if ow not in allowed_refs:
                        overlaps_list = None
                        break
                if overlaps_list is None:
                    continue
                if not allowed_step_overlap(prev, nxt, s, e, eta_prev):
                    continue

            eta_next = eta_minutes(new_addr, nxt[2]) if nxt else (eta_minutes(new_addr, home_addr) if home_addr else None)
            worst_eta = max([x for x in [eta_prev, eta_next] if x is not None], default=NORMAL_THRESH)

            used_shift = (prev is not None and h == shifted_after_prev and eta_prev is not None and eta_prev <= VERY_CLOSE_THRESH)
            note = ("shifted (<=15min)" if used_shift else ("base step" if prev else "between blocks"))
            if nxt is None:
                note += "; last of day (home)"

            results.append({
                "employee_id": emp_id, "tech_name": resolve_name(emp_id),
                "slot_start": s.isoformat(), "slot_end": e.isoformat(),
                "ETA_prev": eta_prev, "ETA_next": eta_next,
                "risk": risk_label(worst_eta), "note": note
            })

    return results

# ====== MAIN JSON ======
@app.post("/suggest", response_model=SuggestOut)
def suggest(body: SuggestIn):
    try:
        visit_minutes = body.visit_minutes or VISIT_MINUTES_DEFAULT
        arrival_by_emp = collect_arrival_windows(days_ahead=body.days_ahead)
        if not arrival_by_emp:
            return {"address": body.address, "timezone": str(APP_TZ), "visit_minutes": visit_minutes, "suggestions": []}

        now_local = datetime.now(APP_TZ)
        suggestions: List[dict] = []

        for emp_id, windows in arrival_by_emp.items():
            buckets: Dict[str, List[Window]] = {}
            for ws, we, addr, is_event in windows:
                buckets.setdefault(ws.date().isoformat(), []).append((ws, we, addr, is_event))

            home = resolve_home(emp_id)

            for d in range(max(1, body.days_ahead)):
                day_dt = now_local + timedelta(days=d)
                day_key = day_dt.date().isoformat()
                todays = sorted(buckets.get(day_key, []), key=lambda t: t[0])

                dow = day_dt.weekday()  # Mon=0 .. Sun=6
                if dow == 6:  # Sunday
                    continue
                if dow == 5 and emp_id != ALEX_ID:  # Saturday only Alex
                    continue

                if body.preferred_days and weekday_code(day_dt) not in set(body.preferred_days):
                    continue

                cand = build_candidates_for_day(
                    emp_id=emp_id, dt_day=day_dt, todays_windows=todays,
                    new_addr=body.address, win_start_str=body.window_start,
                    win_end_str=body.window_end, home_addr=home
                )
                suggestions.extend(cand)

        risk_order = {"low": 0, "medium": 1, "high": 2}
        suggestions.sort(key=lambda x: (risk_order.get(x.get("risk", "medium"), 1), x["slot_start"]))

        seen = set()
        deduped = []
        for s in suggestions:
            key = (s["employee_id"], s["slot_start"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(s)

        return {
            "address": body.address,
            "timezone": str(APP_TZ),
            "visit_minutes": visit_minutes,
            "suggestions": deduped[:50]
        }

    except Exception as e:
        return {"address": body.address, "timezone": str(APP_TZ), "visit_minutes": VISIT_MINUTES_DEFAULT, "suggestions": [], "error": str(e)}

# ====== COMPACT ======
def _fmt_hour(dt: datetime) -> str:
    hh = dt.hour
    suf = "am" if hh < 12 else "pm"
    base = hh if 1 <= hh <= 12 else (hh - 12 if hh > 12 else 12)
    return f"{base}{suf}"

def _slot_human(start_iso: str, end_iso: str) -> str:
    s = datetime.fromisoformat(start_iso).astimezone(APP_TZ)
    e = datetime.fromisoformat(end_iso).astimezone(APP_TZ)
    return f"{_fmt_hour(s)}–{_fmt_hour(e)}"

def format_compact(suggestions: List[dict]) -> str:
    grouped: Dict[Tuple[str, str], List[str]] = {}
    order: List[Tuple[str, str]] = []
    for s in suggestions:
        tech = s.get("tech_name") or "Technician"
        dt = datetime.fromisoformat(s["slot_start"]).astimezone(APP_TZ).strftime("%a %m/%d")
        slot_label = _slot_human(s["slot_start"], s["slot_end"])
        risk = s.get("risk", "")
        item = f"{slot_label} ({risk})" if risk else slot_label
        key = (tech, dt)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(item)
    return "\n".join(f"{tech} ({dt}): " + ", ".join(grouped[(tech, dt)]) for tech, dt in order)

@app.post("/suggest-compact")
def suggest_compact(body: SuggestIn):
    raw = suggest(body)
    if isinstance(raw, dict) and "suggestions" in raw:
        text = format_compact(raw["suggestions"])
        if not text.strip():
            text = "Нет подходящих слотов в выбранный горизонт. Попробуйте увеличить days_ahead или ослабить ограничения."
        return {"address": raw["address"], "timezone": raw["timezone"], "summary": text}
    return {"error": "failed"}
