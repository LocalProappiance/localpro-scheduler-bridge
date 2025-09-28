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

# Google Distance Matrix (реальный ETA)
GMAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")  # добавь в DO App Settings -> Environment Variables

# Фолбэк-имена (если ENV NAME_PRO_* не задан)
NAME_MAP: Dict[str, str] = {
    "pro_5d48854aad6542a28f9da12d0c1b65f2": "Alex Yakush",
    "pro_e07e8bc2e5464dfdba36866c66a5f62d": "Vladimir Kovalev",
    "pro_17e6723ece4e47af95bcb6c1766bbb47": "Nick Litvinov",
}

# Фолбэк-дома (если ENV HOME_PRO_* не задан)
HOME_MAP: Dict[str, str] = {
    # "pro_...": "City, ST"
}

# Бизнес-правила (политика v3)
VISIT_MINUTES_DEFAULT = 120    # информативно; окна прибытия = 3ч
DEFAULT_WORK_START = time(9, 0)
DEFAULT_WORK_END   = time(18, 0)
VERY_CLOSE_THRESH  = 15        # ETA ≤ 15 → смещённое окно разрешено (10–1 и т.п.)
NORMAL_THRESH      = 30        # 15–30 → обычная ступенька 11–2; >30 — тоже можно 11–2 (просто "high" риск)

# Допустимые 3-часовые окна по целому часу
GRID_STARTS = [9, 10, 11, 12, 13, 14, 15]  # 9–12,10–1,11–2,12–3,1–4,2–5,3–6

# ====== APP ======
app = FastAPI(title="LocalPRO Scheduler Bridge")

# ====== MODELS ======
class SuggestIn(BaseModel):
    address: str
    days_ahead: int = 3
    preferred_days: Optional[List[str]] = None  # ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    window_start: Optional[str] = None          # например "12:00" (только после 12)
    window_end: Optional[str] = None            # например "18:00"
    job_type: Optional[str] = None
    visit_minutes: Optional[int] = None         # информативно
    buffer_minutes: Optional[int] = None        # не используется в логике v3 (оставлено для совместимости)

class SuggestOut(BaseModel):
    address: str
    timezone: str
    visit_minutes: int
    suggestions: List[dict]

# ====== HCP HELPERS ======
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

# ====== RESOLVERS: name & home из ENV с фолбэком ======
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
CITY_EQ_MIN   = 12
CITY_NEAR_MIN = 22
CITY_FAR_MIN  = 35
NEAR_CITIES = {"st. augustine", "st augustine", "jacksonville", "orange park"}

def estimate_eta_heuristic(src_addr: str, dst_addr: str) -> int:
    a = safe_city(src_addr).lower()
    b = safe_city(dst_addr).lower()
    if not a or not b:
        return CITY_NEAR_MIN
    if a == b:
        return CITY_EQ_MIN
    if a in NEAR_CITIES and b in NEAR_CITIES:
        return CITY_NEAR_MIN
    return CITY_FAR_MIN

def google_eta_minutes(src_addr: str, dst_addr: str) -> Optional[int]:
    if not GMAPS_KEY:
        return None
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            params={
                "origins": src_addr,
                "destinations": dst_addr,
                "key": GMAPS_KEY,
                "units": "imperial"
            },
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

def eta_minutes(src_addr: str, dst_addr: str) -> int:
    if src_addr is None or dst_addr is None:
        return CITY_NEAR_MIN
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
    data = hcp_get("/jobs")
    raw = data if isinstance(data, list) else data.get("results") or data.get("jobs") or []
    out = []
    for item in (raw[:10] if isinstance(raw, list) else []):
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
        data = hcp_get("/jobs")
        raw = data if isinstance(data, list) else data.get("results") or data.get("jobs") or []
        for item in (raw if isinstance(raw, list) else []):
            _, _, emp_id, _ = pick_time_and_tech(item)
            if emp_id:
                keys.add(emp_id)
    except Exception:
        pass
    for k in keys:
        out[k] = resolve_home(k)
    return {"homes": out, "gmaps": bool(GMAPS_KEY)}

# ====== COLLECT ARRIVAL WINDOWS ======
def collect_arrival_windows(days_ahead: int) -> Dict[str, List[Tuple[datetime, datetime, str]]]:
    """
    arrival_by_emp[employee_id] = список (win_start_local, win_end_local, job_address)
    """
    jobs_data = hcp_get("/jobs")
    raw_jobs = jobs_data if isinstance(jobs_data, list) else jobs_data.get("results") or jobs_data.get("jobs") or []
    now_local = datetime.now(APP_TZ)
    horizon_end = now_local + timedelta(days=max(1, days_ahead))

    arrival_by_emp: Dict[str, List[Tuple[datetime, datetime, str]]] = {}
    for j in (raw_jobs if isinstance(raw_jobs, list) else []):
        start_utc, end_utc, emp_id, addr_str = pick_time_and_tech(j)
        if not (start_utc and end_utc and emp_id):
            continue
        ws = to_local(start_utc)
        we = to_local(end_utc)
        if we < now_local or ws > horizon_end:
            continue
        arrival_by_emp.setdefault(emp_id, []).append((ws, we, addr_str))

    for emp_id in arrival_by_emp:
        arrival_by_emp[emp_id].sort(key=lambda t: t[0])
    return arrival_by_emp

# ====== GRID UTILS ======
def grid_slot(dt_day: datetime, start_hour: int) -> Tuple[datetime, datetime]:
    s = datetime.combine(dt_day.date(), time(start_hour, 0), tzinfo=APP_TZ)
    e = s + timedelta(hours=3)
    return s, e

def within_client_window(s: datetime, e: datetime, day_start: datetime, day_end: datetime,
                         win_start_str: Optional[str], win_end_str: Optional[str]) -> bool:
    cl_s = from_local(day_start, win_start_str) or day_start
    cl_e = from_local(day_start, win_end_str) or day_end
    return (s >= cl_s) and (e <= cl_e)

# ====== BUILD CANDIDATES (POLICY v3 + real ETA) ======
def build_candidates_for_day(
    emp_id: str,
    dt_day: datetime,
    todays_windows: List[Tuple[datetime, datetime, str]],  # [(ws,we,addr)]
    new_addr: str,
    win_start_str: Optional[str],
    win_end_str: Optional[str],
    home_addr: Optional[str]
) -> List[dict]:
    day_start = datetime.combine(dt_day.date(), DEFAULT_WORK_START, tzinfo=APP_TZ)
    day_end   = datetime.combine(dt_day.date(), DEFAULT_WORK_END, tzinfo=APP_TZ)

    occupied_hours = set(ws.hour for ws, _, _ in todays_windows)
    results: List[dict] = []

    # Если нет визитов в этот день — опираемся на дом техника
    if not todays_windows:
        for h in GRID_STARTS:
            if h in occupied_hours:
                continue
            s, e = grid_slot(dt_day, h)
            if not within_client_window(s, e, day_start, day_end, win_start_str, win_end_str):
                continue
            eta_from_home = eta_minutes(home_addr, new_addr) if home_addr else NORMAL_THRESH
            results.append({
                "employee_id": emp_id,
                "tech_name": resolve_name(emp_id),
                "slot_start": s.isoformat(),
                "slot_end": e.isoformat(),
                "ETA_prev": eta_from_home,
                "ETA_next": None,
                "risk": risk_label(eta_from_home),
                "note": "first of day (home proximity)"
            })
        return results

    anchors = [(None, None, None)] + todays_windows + [(None, None, None)]
    for i in range(len(anchors)-1):
        A = anchors[i]
        C = anchors[i+1]
        prev = A if A[0] is not None else None
        nxt  = C if C[0] is not None else None

        # базовая «соседняя ступенька» после prev (9->11, 11->13, ...)
        base_after_prev = 9
        if prev:
            base_after_prev = prev[0].hour + 2
        shifted_after_prev = base_after_prev - 1  # смещение на час раньше (10–1), если очень близко

        # допустимые сеточные старты внутри интервала
        candidate_hours = []
        for h in GRID_STARTS:
            if prev and h <= prev[0].hour - 1:
                continue
            if nxt and h >= nxt[0].hour + 1:
                continue
            candidate_hours.append(h)

        for h in candidate_hours:
            if h in occupied_hours:
                continue
            s, e = grid_slot(dt_day, h)
            if not within_client_window(s, e, day_start, day_end, win_start_str, win_end_str):
                continue

            allow = False
            used_shift = False
            eta_prev = None

            if prev:
                eta_prev = eta_minutes(prev[2], new_addr)
                if eta_prev <= VERY_CLOSE_THRESH and h == shifted_after_prev:
                    allow = True
                    used_shift = True
                elif h == base_after_prev:
                    allow = True
                else:
                    allow = False
            else:
                allow = True  # первый кандидат между None и C

            if not allow:
                continue

            # ETA к следующему визиту, либо к дому, если это «последний»
            eta_next = eta_minutes(new_addr, nxt[2]) if nxt else (eta_minutes(new_addr, home_addr) if home_addr else None)
            worst_eta = max([x for x in [eta_prev, eta_next] if x is not None], default=NORMAL_THRESH)

            results.append({
                "employee_id": emp_id,
                "tech_name": resolve_name(emp_id),
                "slot_start": s.isoformat(),
                "slot_end": e.isoformat(),
                "ETA_prev": eta_prev,
                "ETA_next": eta_next,
                "risk": risk_label(worst_eta),
                "note": ("shifted (<=15min)" if used_shift else "base step")
                        + ("; last of day (home)" if nxt is None else "")
            })

    return results

# ====== MAIN JSON ======
@app.post("/suggest", response_model=SuggestOut)
def suggest(body: SuggestIn):
    try:
        visit_minutes = body.visit_minutes or VISIT_MINUTES_DEFAULT

        arrival_by_emp = collect_arrival_windows(days_ahead=body.days_ahead)
        if not arrival_by_emp:
            return {
                "address": body.address,
                "timezone": str(APP_TZ),
                "visit_minutes": visit_minutes,
                "suggestions": []
            }

        now_local = datetime.now(APP_TZ)
        suggestions: List[dict] = []

        for emp_id, windows in arrival_by_emp.items():
            # группируем по дате
            buckets: Dict[str, List[Tuple[datetime, datetime, str]]] = {}
            for ws, we, addr in windows:
                buckets.setdefault(ws.date().isoformat(), []).append((ws, we, addr))

            home = resolve_home(emp_id)

            for d in range(max(1, body.days_ahead)):
                day_dt = now_local + timedelta(days=d)
                day_key = day_dt.date().isoformat()
                todays = sorted(buckets.get(day_key, []), key=lambda t: t[0])

                if body.preferred_days and weekday_code(day_dt) not in set(body.preferred_days):
                    continue

                cand = build_candidates_for_day(
                    emp_id=emp_id,
                    dt_day=day_dt,
                    todays_windows=todays,
                    new_addr=body.address,
                    win_start_str=body.window_start,
                    win_end_str=body.window_end,
                    home_addr=home
                )
                suggestions.extend(cand)

        # Ранжируем: риск (low < medium < high), затем ближайшее время
        risk_order = {"low": 0, "medium": 1, "high": 2}
        suggestions.sort(key=lambda x: (risk_order.get(x.get("risk", "medium"), 1), x["slot_start"]))

        # дедуп по (tech, slot_start)
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
            "suggestions": deduped[:50]  # оставим побольше, компактный вывод сам сгруппирует
        }

    except Exception as e:
        return {
            "address": body.address,
            "timezone": str(APP_TZ),
            "visit_minutes": VISIT_MINUTES_DEFAULT,
            "suggestions": [],
            "error": str(e)
        }

# ====== COMPACT TEXT OUTPUT ======
def _fmt_hour(dt: datetime) -> str:
    # "9am" / "12pm" / "1pm"
    hh = dt.hour
    suf = "am" if hh < 12 else "pm"
    base = hh if 1 <= hh <= 12 else (hh - 12 if hh > 12 else 12)
    return f"{base}{suf}"

def _slot_human(start_iso: str, end_iso: str) -> str:
    s = datetime.fromisoformat(start_iso).astimezone(APP_TZ)
    e = datetime.fromisoformat(end_iso).astimezone(APP_TZ)
    return f"{_fmt_hour(s)}–{_fmt_hour(e)}"

def format_compact(suggestions: List[dict]) -> str:
    # Группировка: (tech, date_str) -> [ "12–3 (low)", ... ]
    grouped: Dict[Tuple[str, str], List[str]] = {}
    order_keys: List[Tuple[str, str]] = []
    for s in suggestions:
        tech = s.get("tech_name") or "Technician"
        dt = datetime.fromisoformat(s["slot_start"]).astimezone(APP_TZ).strftime("%a %m/%d")
        slot_label = _slot_human(s["slot_start"], s["slot_end"])
        risk = s.get("risk", "")
        item = f"{slot_label} ({risk})" if risk else slot_label
        key = (tech, dt)
        if key not in grouped:
            grouped[key] = []
            order_keys.append(key)
        grouped[key].append(item)

    lines = []
    for tech, dt in order_keys:
        lines.append(f"{tech} ({dt}): " + ", ".join(grouped[(tech, dt)]))
    return "\n".join(lines)

@app.post("/suggest-compact")
def suggest_compact(body: SuggestIn):
    raw = suggest(body)  # используем логику /suggest
    if isinstance(raw, dict) and "suggestions" in raw:
        text = format_compact(raw["suggestions"])
        return {
            "address": raw["address"],
            "timezone": raw["timezone"],
            "summary": text
        }
    return {"error": "failed"}
