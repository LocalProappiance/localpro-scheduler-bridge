# app.py
# LocalPRO Scheduler Bridge — v4.5.0 (HCP Schedule OAuth + fallback to Jobs/Events Token)
import os
from datetime import datetime, timedelta, time, date
from typing import Any, Optional, List, Tuple, Dict, Set
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import requests
from requests.exceptions import RequestException, Timeout
from zoneinfo import ZoneInfo

# ====== CONFIG ======
APP_TZ = ZoneInfo(os.environ.get("TZ", "America/New_York"))

# --- HCP AUTH (two modes) ---
# 1) OAuth (MAX/XL plan) — for /v1/schedule (Bearer)
HCP_CLIENT_ID = os.environ.get("HCP_CLIENT_ID")
HCP_CLIENT_SECRET = os.environ.get("HCP_CLIENT_SECRET")
# Optional, if HCP uses audience; keep empty if not needed
HCP_OAUTH_AUDIENCE = os.environ.get("HCP_OAUTH_AUDIENCE", "")
OAUTH_TOKEN_URL = os.environ.get("HCP_OAUTH_TOKEN_URL", "https://api.housecallpro.com/oauth/token")

# 2) Token (public API) — for /jobs and /events (Token)
HCP_KEY = os.environ.get("HCP_API_KEY")

HCP_BASE = os.environ.get("HCP_BASE", "https://api.housecallpro.com")
GMAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

# ====== NAMES & HOMES ======
NAME_MAP: Dict[str, str] = {
    "pro_5d48854aad6542a28f9da12d0c1b65f2": "Alex Yakush",
    "pro_e07e8bc2e5464dfdba36866c66a5f62d": "Vladimir Kovalev",
    "pro_17e6723ece4e47af95bcb6c1766bbb47": "Nick Litvinov",
}
HOME_MAP: Dict[str, str] = {
    # "pro_...": "City, ST"
}

ALEX_ID = "pro_5d48854aad6542a28f9da12d0c1b65f2"  # Saturday only Alex

# ====== BUSINESS RULES ======
VISIT_MINUTES_DEFAULT = 120
DEFAULT_WORK_START = time(9, 0)
DEFAULT_WORK_END   = time(18, 0)

VERY_CLOSE_THRESH = 15    # ≤15 -> low
NORMAL_THRESH     = 40    # 16..40 -> medium; >40 -> high

GRID_STARTS = [9, 10, 11, 12, 13, 14, 15]  # 9–12..3–6

# Overlap caps
MAX_OVERLAP_PREV_MIN = 60
MAX_OVERLAP_NEXT_MIN = 60
FULL_WINDOW_MIN      = 180  # never
MAX_OVERLAP_TOTAL_MIN = 60  # sum cap for job overlaps

# ====== APP ======
app = FastAPI(title="LocalPRO Scheduler Bridge (v4.5.0)")

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

# ====== SIMPLE UTILS ======
def address_to_str(addr):
    if not isinstance(addr, dict):
        return str(addr)
    parts = []
    for k in ["street", "street_line_2", "city", "state", "zip", "country"]:
        v = addr.get(k)
        if v:
            parts.append(str(v))
    return ", ".join(parts)

def parse_iso_guess_tz(s: str) -> datetime:
    """
    ISO-строка:
    - если содержит смещение (Z/±HH:MM) — используем его;
    - если без смещения — считаем ЛОКАЛЬНОЙ (APP_TZ).
    Возвращаем в UTC.
    """
    if not isinstance(s, str):
        raise ValueError("Not a string")
    has_tz = ("Z" in s) or ("+" in s[10:]) or ("-" in s[10:])
    if has_tz:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(ZoneInfo("UTC"))
    # naive => local
    dt_local = datetime.fromisoformat(s)
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=APP_TZ)
    return dt_local.astimezone(ZoneInfo("UTC"))

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

# ====== NAME TOKENS & ALIASES ======
def resolve_name(emp_id: Optional[str]) -> str:
    if not emp_id:
        return "Technician"
    env_key = f"NAME_PRO_{emp_id}"
    if env_key in os.environ and os.environ[env_key].strip():
        return os.environ[env_key].strip()
    return NAME_MAP.get(emp_id, "Technician")

def resolve_home(emp_id: str) -> Optional[str]:
    env_key = f"HOME_PRO_{emp_id}"
    if env_key in os.environ and os.environ[env_key].strip():
        return os.environ[env_key].strip()
    return HOME_MAP.get(emp_id)

def name_tokens_for_emp(emp_id: str) -> Set[str]:
    tokens: Set[str] = set()
    full = resolve_name(emp_id)
    for t in full.replace("-", " ").split():
        t = t.strip()
        if len(t) >= 2:
            tokens.add(t.lower())
    alias_key = f"ALIAS_PRO_{emp_id}"
    if alias_key in os.environ and os.environ[alias_key].strip():
        for t in os.environ[alias_key].replace(",", " ").split():
            t = t.strip()
            if len(t) >= 2:
                tokens.add(t.lower())
    return tokens

def title_mentions_emp(title: str, emp_id: str) -> bool:
    low = (title or "").lower()
    for tok in name_tokens_for_emp(emp_id):
        if tok and tok in low:
            return True
    return False

# ====== EMP ID EXTRACTION (robust) ======
def _collect_emp_ids(obj: Any) -> Set[str]:
    ids: Set[str] = set()
    if obj is None:
        return ids

    def add_id(val):
        if not val:
            return
        s = str(val).strip()
        if s.startswith("pro_") and len(s) > 4:
            ids.add(s)

    if isinstance(obj, dict):
        for k in ["assigned_employees", "employees", "participants", "assignees"]:
            if k in obj and isinstance(obj[k], list):
                for it in obj[k]:
                    if isinstance(it, dict):
                        add_id(it.get("id") or it.get("employee_id") or it.get("pro_id"))
                    else:
                        add_id(it)
        for k in ["employee_id", "pro_id", "tech_id", "assignee_id"]:
            if k in obj:
                add_id(obj[k])
        if "employee" in obj and isinstance(obj["employee"], dict):
            add_id(obj["employee"].get("id"))
    elif isinstance(obj, list):
        for it in obj:
            ids |= _collect_emp_ids(it)
    return ids

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
            if len(parts) >= 3:
                return parts[-3].lower()
            if len(parts) >= 2:
                return parts[1].lower()
            return a.lower()
        except Exception:
            return a.lower()
    a = city(src_addr or ""); b = city(dst_addr or "")
    if not a or not b: return 30
    if a == b: return 12
    if a in NEAR_CITIES and b in NEAR_CITIES: return 22
    return 45

def eta_minutes(src_addr: str, dst_addr: str) -> int:
    if not src_addr or not dst_addr:
        return 30
    real = google_eta_minutes(src_addr, dst_addr)
    return real if real is not None else estimate_eta_heuristic(src_addr, dst_addr)

def risk_label(eta: int) -> str:
    if eta <= VERY_CLOSE_THRESH: return "low"
    if eta <= NORMAL_THRESH:     return "medium"
    return "high"

# ====== HCP AUTH HELPERS (OAUTH) ======
_oauth_cache: Dict[str, Any] = {"token": None, "exp": 0}

def _oauth_now() -> int:
    return int(datetime.utcnow().timestamp())

def get_bearer_token() -> Optional[str]:
    # return cached
    if _oauth_cache["token"] and _oauth_cache["exp"] > _oauth_now() + 30:
        return _oauth_cache["token"]
    if not (HCP_CLIENT_ID and HCP_CLIENT_SECRET):
        return None
    try:
        payload = {
            "grant_type": "client_credentials",
            "client_id": HCP_CLIENT_ID,
            "client_secret": HCP_CLIENT_SECRET,
        }
        if HCP_OAUTH_AUDIENCE:
            payload["audience"] = HCP_OAUTH_AUDIENCE
        r = requests.post(OAUTH_TOKEN_URL, data=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        token = data.get("access_token")
        expires_in = int(data.get("expires_in", 900))
        if token:
            _oauth_cache["token"] = token
            _oauth_cache["exp"] = _oauth_now() + max(60, expires_in - 30)
            return token
        return None
    except Exception:
        return None

def hcp_get_token(path: str, params=None):
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

def hcp_get_bearer(path: str, params=None):
    token = get_bearer_token()
    if not token:
        raise HTTPException(status_code=500, detail="HCP OAuth token is not available")
    try:
        r = requests.get(
            f"{HCP_BASE}{path}",
            headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
            params=params or {},
            timeout=20,
        )
        if r.status_code == 401:
            # refresh and retry once
            _oauth_cache["token"] = None
            token = get_bearer_token()
            r = requests.get(
                f"{HCP_BASE}{path}",
                headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
                params=params or {},
                timeout=20,
            )
        r.raise_for_status()
        return r.json()
    except Timeout as e:
        raise HTTPException(status_code=504, detail=f"HCP timeout on {path}") from e
    except RequestException as e:
        status = getattr(e.response, "status_code", 502)
        raise HTTPException(status_code=status, detail=f"HCP error on {path}") from e

# ====== HCP PAGINATION (Token) ======
def hcp_paged_list_token(path: str, max_pages: int = 12, per_page: int = 100) -> List[dict]:
    items: List[dict] = []
    for p in range(1, max_pages + 1):
        data = hcp_get_token(path, params={"page": p, "per_page": per_page})
        arr = data if isinstance(data, list) else data.get("results") or data.get("jobs") or data.get("events") or []
        if not isinstance(arr, list):
            break
        items.extend(arr)
        if len(arr) < per_page:
            break
    if not items:
        data = hcp_get_token(path)
        arr = data if isinstance(data, list) else data.get("results") or data.get("jobs") or data.get("events") or []
        if isinstance(arr, list):
            items = arr
    return items

# ====== PICK TIME/TECH FROM JOB ======
def pick_time_and_tech(job: dict) -> Tuple[Optional[datetime], Optional[datetime], Optional[str], str]:
    start_utc = end_utc = None
    employee_id = None

    schedule = job.get("schedule") or {}
    s = schedule.get("scheduled_start") or job.get("scheduled_start") or job.get("start") or job.get("starts_at") or job.get("start_time")
    e = schedule.get("scheduled_end")   or job.get("scheduled_end")   or job.get("end")   or job.get("ends_at")   or job.get("end_time")
    if isinstance(s, str):
        try: start_utc = parse_iso_guess_tz(s)
        except Exception: start_utc = None
    if isinstance(e, str):
        try: end_utc = parse_iso_guess_tz(e)
        except Exception: end_utc = None

    emp_ids = _collect_emp_ids(job)
    if emp_ids:
        employee_id = next(iter(emp_ids))

    addr_str = address_to_str(job.get("address") or job.get("service_address") or {})

    if not start_utc or not end_utc:
        for p, v in _walk(job):
            low = p.lower()
            if isinstance(v, str) and not start_utc and ("start" in low or "begin" in low) and any(ch.isdigit() for ch in v):
                try: start_utc = parse_iso_guess_tz(v)
                except Exception: pass
            if isinstance(v, str) and not end_utc and ("end" in low or "finish" in low) and any(ch.isdigit() for ch in v):
                try: end_utc = parse_iso_guess_tz(v)
                except Exception: pass

    return start_utc, end_utc, employee_id, addr_str

# ====== EVENTS DETECTION ======
EVENT_TYPE_KEYWORDS = {"event","block","blocked","calendar","personal","pto","vacation","meeting","busy","not available","unavailable","dispatch by name"}

def looks_like_event_job(job: dict) -> bool:
    t1 = str(job.get("type") or "").lower()
    t2 = str(job.get("job_type") or "").lower()
    title = (job.get("title") or job.get("summary") or job.get("name") or "")
    no_service_addr = not (job.get("address") or job.get("service_address"))
    if any(k in t1 for k in EVENT_TYPE_KEYWORDS): return True
    if any(k in t2 for k in EVENT_TYPE_KEYWORDS): return True
    if any(k in title.lower() for k in EVENT_TYPE_KEYWORDS): return True
    if no_service_addr: return True
    for emp_id in NAME_MAP.keys():
        if title_mentions_emp(title, emp_id):
            return True
    return False

def event_employee_id_from_title(title: str) -> Optional[str]:
    for emp_id in NAME_MAP.keys():
        if title_mentions_emp(title, emp_id):
            return emp_id
    return None

def parse_event_time(ev: dict) -> Tuple[Optional[datetime], Optional[datetime]]:
    for k_start, k_end in [
        ("start", "end"),
        ("scheduled_start", "scheduled_end"),
        ("starts_at", "ends_at"),
        ("start_time", "end_time"),
    ]:
        s = ev.get(k_start); e = ev.get(k_end)
        if isinstance(s, str) and isinstance(e, str):
            try: return parse_iso_guess_tz(s), parse_iso_guess_tz(e)
            except Exception: pass
    sch = ev.get("schedule") or {}
    s = sch.get("scheduled_start"); e = sch.get("scheduled_end")
    if isinstance(s, str) and isinstance(e, str):
        try: return parse_iso_guess_tz(s), parse_iso_guess_tz(e)
        except Exception: pass
    return None, None

def event_employee_id(ev: dict) -> Optional[str]:
    ids = _collect_emp_ids(ev)
    if ids:
        return next(iter(ids))
    title = ev.get("title") or ev.get("summary") or ev.get("name") or ""
    return event_employee_id_from_title(title)

# ====== SCHEDULE API (OAUTH) ======
def try_fetch_schedule(day_start: date, day_end: date) -> Optional[List[dict]]:
    """
    GET /v1/schedule?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
    Возвращает items с job/event/dispatch-by-name.
    """
    try:
        params = {"start_date": day_start.isoformat(), "end_date": day_end.isoformat()}
        data = hcp_get_bearer("/v1/schedule", params=params)
        arr = data if isinstance(data, list) else data.get("results") or data.get("items") or []
        if isinstance(arr, list):
            return arr
        return []
    except Exception:
        return None  # означает — используем фолбэк

def schedule_item_to_window(it: dict) -> Optional[Tuple[datetime, datetime, Optional[str], str, bool]]:
    """
    Возвращает (start_local, end_local, employee_id, addr_or_title, is_event)
    """
    # 1) время
    s_utc = e_utc = None
    for k_start, k_end in [("start", "end"), ("scheduled_start", "scheduled_end"),
                           ("starts_at", "ends_at"), ("start_time", "end_time")]:
        s = it.get(k_start); e = it.get(k_end)
        if isinstance(s, str) and isinstance(e, str):
            try:
                s_utc = parse_iso_guess_tz(s)
                e_utc = parse_iso_guess_tz(e)
                break
            except Exception:
                pass
    if not (s_utc and e_utc):
        return None

    ws = to_local(s_utc); we = to_local(e_utc)

    # 2) кто сотрудник
    emp_ids = _collect_emp_ids(it)
    emp_id = next(iter(emp_ids)) if emp_ids else None

    # 3) адрес/заголовок
    title = it.get("title") or it.get("summary") or it.get("name") or ""
    addr = address_to_str(it.get("address") or it.get("service_address") or {}) or title or "Event"

    # 4) тип
    typ = str(it.get("type") or it.get("kind") or "").lower()
    # признаки "dispatch by name" / busy
    is_ev = False
    if any(k in typ for k in EVENT_TYPE_KEYWORDS):
        is_ev = True
    else:
        tt = (title or "").lower()
        if any(k in tt for k in EVENT_TYPE_KEYWORDS):
            is_ev = True
        if not (it.get("address") or it.get("service_address")):
            is_ev = True

    # 5) маппинг dispatch-by-name: если emp_id нет, но в title есть имя техника — привязываем
    if not emp_id:
        for k in NAME_MAP.keys():
            if title_mentions_emp(title, k):
                emp_id = k
                break

    return (ws, we, emp_id, addr if not is_ev else (title or "Event"), is_ev)

# ====== DEBUG ======
@app.get("/health")
def health():
    return {
        "status": "ok",
        "tz": str(APP_TZ),
        "gmaps": bool(GMAPS_KEY),
        "oauth": bool(HCP_CLIENT_ID and HCP_CLIENT_SECRET),
        "token": bool(HCP_KEY),
        "version": "v4.5.0"
    }

@app.get("/debug/hcp-schedule")
def debug_hcp_schedule(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD (inclusive)")
):
    try:
        d1 = datetime.fromisoformat(start).date()
        d2 = datetime.fromisoformat(end).date()
    except Exception:
        raise HTTPException(status_code=400, detail="Bad date format, expected YYYY-MM-DD")
    raw = try_fetch_schedule(d1, d2)
    return {"range": [start, end], "count": len(raw or []), "items_preview": (raw or [])[:20]}

@app.get("/debug/hcp-jobs-compact")
def debug_hcp_jobs_compact():
    data = hcp_paged_list_token("/jobs") if HCP_KEY else []
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

@app.get("/debug/day")
def debug_day(date_str: str = Query(..., description="YYYY-MM-DD")):
    arrival_by_emp = collect_arrival_windows(days_ahead=30)
    try:
        target = datetime.fromisoformat(date_str).date()
    except Exception:
        raise HTTPException(status_code=400, detail="Bad date format, expected YYYY-MM-DD")
    report: Dict[str, List[dict]] = {}
    for emp_id, wins in arrival_by_emp.items():
        lines = []
        for ws, we, addr, is_event in wins:
            if ws.date() == target or we.date() == target:
                lines.append({
                    "start_local": ws.isoformat(),
                    "end_local": we.isoformat(),
                    "is_event": is_event,
                    "address_or_title": addr
                })
        if lines:
            report[resolve_name(emp_id)] = lines
    return {"date": date_str, "seen": report}

@app.get("/debug/events-orphaned")
def debug_events_orphaned():
    orphaned = []
    # Смотрим только fallback /events (Token), если доступен
    events = []
    try:
        events = hcp_paged_list_token("/events") if HCP_KEY else []
    except Exception:
        events = []
    for ev in events:
        s_utc, e_utc = parse_event_time(ev)
        if not (s_utc and e_utc):
            continue
        emp = event_employee_id(ev)
        if not emp:
            orphaned.append({
                "title": ev.get("title") or ev.get("summary") or ev.get("name"),
                "start_utc": s_utc.isoformat(), "end_utc": e_utc.isoformat(),
                "raw": {"assigned_employees": ev.get("assigned_employees")}
            })
    return {"orphaned_count": len(orphaned), "items": orphaned[:50]}

# ====== WINDOW TYPE ======
Window = Tuple[datetime, datetime, str, bool]  # (start_local, end_local, address/title, is_event)

# ====== COLLECT ARRIVAL WINDOWS ======
def collect_arrival_windows(days_ahead: int) -> Dict[str, List[Window]]:
    now_local = datetime.now(APP_TZ)
    horizon_end = now_local + timedelta(days=max(1, days_ahead))
    arrival_by_emp: Dict[str, List[Window]] = {}

    # 1) PRIMARY: SCHEDULE (OAuth)
    d1 = now_local.date()
    d2 = horizon_end.date()
    schedule_items = try_fetch_schedule(d1, d2)  # None => fallback
    if schedule_items is not None:
        # соберём по техникам
        for it in schedule_items:
            mapped = schedule_item_to_window(it)
            if not mapped:
                continue
            ws, we, emp_id, addr_or_title, is_event = mapped
            if we < now_local or ws > horizon_end:
                continue
            # если emp_id так и не нашли — считаем глобальным блоком (редко, но бывает)
            if not emp_id:
                # глобально на всех известных техников
                for all_emp in NAME_MAP.keys():
                    arrival_by_emp.setdefault(all_emp, []).append((ws, we, addr_or_title, True))
            else:
                arrival_by_emp.setdefault(emp_id, []).append((ws, we, addr_or_title, is_event))
    else:
        # 2) FALLBACK: JOBS + EVENTS (Token)
        jobs = hcp_paged_list_token("/jobs") if HCP_KEY else []
        for j in jobs if isinstance(jobs, list) else []:
            start_utc, end_utc, emp_id, addr_str = pick_time_and_tech(j)
            if not (start_utc and end_utc):
                continue
            ws = to_local(start_utc); we = to_local(end_utc)
            if we < now_local or ws > horizon_end:
                continue
            # Try to infer employee from title if missing
            if not emp_id:
                title = j.get("title") or j.get("summary") or j.get("name") or ""
                emp_id = event_employee_id_from_title(title)
            is_event_job = looks_like_event_job(j)
            if emp_id:
                arrival_by_emp.setdefault(emp_id, []).append(
                    (ws, we, addr_str if not is_event_job else (j.get("title") or j.get("summary") or "Event"), is_event_job)
                )
        events = []
        try:
            events = hcp_paged_list_token("/events") if HCP_KEY else []
        except Exception:
            events = []
        for ev in events:
            s_utc, e_utc = parse_event_time(ev)
            emp_id = event_employee_id(ev)
            if not (s_utc and e_utc):
                continue
            ws = to_local(s_utc); we = to_local(e_utc)
            if we < now_local or ws > horizon_end:
                continue
            addr_str = address_to_str(ev.get("address") or {}) or (ev.get("title") or ev.get("summary") or "Event")
            if emp_id:
                arrival_by_emp.setdefault(emp_id, []).append((ws, we, addr_str, True))
            else:
                for all_emp in NAME_MAP.keys():
                    arrival_by_emp.setdefault(all_emp, []).append((ws, we, addr_str, True))

    # normalize & sort
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
    start = max(a_start, b_start); end = min(a_end, b_end)
    if start >= end: return 0
    return int((end - start).total_seconds() // 60)

def overlapping_windows(s: datetime, e: datetime, windows: List[Window]) -> List[Window]:
    return [(ws, we, addr, is_event) for ws, we, addr, is_event in windows if overlaps(s, e, ws, we)]

def same_interval_exists(s: datetime, e: datetime, windows: List[Window]) -> bool:
    for ws, we, _, _ in windows:
        if ws == s and we == e:
            return True
    return False

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

    occupied_start_hours = set(ws.hour for ws, _, _, _ in todays_windows)  # safety: no same-hour start
    results: List[dict] = []

    # Пустой день
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
        shifted_after_prev = base_after_prev - 1  # only if ETA ≤ 15

        candidate_hours = []
        for h in GRID_STARTS:
            if prev and h <= prev[0].hour - 1:
                continue
            if nxt and h >= nxt[0].hour + 1:
                continue
            candidate_hours.append(h)

        for h in candidate_hours:
            if h in occupied_start_hours:
                continue

            s, e = grid_slot(dt_day, h)
            if not within_client_window(s, e, day_start, day_end, win_start_str, win_end_str):
                continue

            if same_interval_exists(s, e, todays_windows):
                continue

            # Первый визит дня не может перекрывать НИ одно окно
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
                # Полный запрет пересечений с евентами
                if any(ow[3] for ow in overlaps_list):
                    continue

                # Разрешаем перекрытие только с одним окном (prev ИЛИ next), суммарно ≤ 60 минут
                allowed_refs = set()
                if prev: allowed_refs.add((prev[0], prev[1], prev[2], prev[3]))
                if nxt:  allowed_refs.add((nxt[0], nxt[1], nxt[2], nxt[3]))

                if any(ow not in allowed_refs for ow in overlaps_list):
                    continue
                if len(overlaps_list) > 1:
                    continue

                ow = overlaps_list[0]
                ws, we, _, _ = ow
                ov = overlap_minutes(s, e, ws, we)
                if ov >= FULL_WINDOW_MIN:
                    continue
                if prev and ws == prev[0] and we == prev[1]:
                    if ov > MAX_OVERLAP_PREV_MIN: 
                        continue
                elif nxt and ws == nxt[0] and we == nxt[1]:
                    if ov > MAX_OVERLAP_NEXT_MIN: 
                        continue
                else:
                    continue
                if ov > MAX_OVERLAP_TOTAL_MIN:
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

                dow = day_dt.weekday()  # Mon=0..Sun=6
                if dow == 6:  # Sunday
                    continue
                if dow == 5 and emp_id != ALEX_ID:
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

        return {"address": body.address, "timezone": str(APP_TZ), "visit_minutes": visit_minutes, "suggestions": deduped[:50]}
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
