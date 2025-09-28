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

# === Настройка имён техников (замени значения на реальные имена) ===
NAME_MAP: Dict[str, str] = {
    "pro_5d48854aad6542a28f9da12d0c1b65f2": "Technician A",
    "pro_e07e8bc2e5464dfdba36866c66a5f62d": "Technician B",
    "pro_17e6723ece4e47af95bcb6c1766bbb47": "Technician C",
}

# === Домашние адреса/города техников (для первого/последнего слота) ===
HOME_MAP: Dict[str, str] = {
    # "pro_5d4...": "St. Augustine, FL",
    # "pro_e07...": "Jacksonville, FL",
    # "pro_17e...": "Orange Park, FL",
}

# Бизнес-правила
VISIT_MINUTES_DEFAULT = 120    # длительность визита: 2 часа
BUFFER_MINUTES_DEFAULT = 0     # отображаемый буфер до/после (логика v3 без доп. буферов в вычислениях)
DEFAULT_WORK_START = time(9, 0)   # 09:00 (по твоему требованию)
DEFAULT_WORK_END   = time(18, 0)  # 18:00
VERY_CLOSE_THRESH  = 15           # ETA ≤ 15 мин
NORMAL_THRESH      = 30           # 15 < ETA ≤ 30 мин (иначе >30 = далеко)

# Допустимые окна (всегда 3 часа длиной)
GRID_STARTS = [9, 10, 11, 12, 13, 14, 15]  # 9–12, 10–1, 11–2, 12–3, 1–4, 2–5, 3–6

# ====== APP ======
app = FastAPI(title="LocalPRO Scheduler Bridge")

# ====== INPUT ======
class SuggestIn(BaseModel):
    address: str
    days_ahead: int = 3
    preferred_days: Optional[List[str]] = None  # ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    window_start: Optional[str] = None          # "12:00"  (например: "только после 12")
    window_end: Optional[str] = None            # "18:00"
    job_type: Optional[str] = None
    visit_minutes: Optional[int] = None         # длительность визита (3ч окно прибытия фикс)
    buffer_minutes: Optional[int] = None        # только визуальный атрибут в ответе
    # ВНИМАНИЕ: ETA в этой версии — эвристика (по городам). Реальный ETA добавим отдельно.

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
    # очень простая вырезка города из "Street, City, ST, Zip, Country"
    try:
        parts = [p.strip() for p in addr_str.split(",")]
        # ищем сегмент перед штатом (2-букв. код), грубо:
        for i in range(len(parts)-1):
            if len(parts[i+1]) == 2 and parts[i+1].isalpha():
                return parts[i]
        # запасной вариант — второй сегмент
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
    """
    Возвращает:
      start_utc, end_utc — ОКНА ПРИБЫТИЯ (scheduled_start/end) в UTC,
      employee_id,
      address_str
    """
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

    # запасной поиск по вложениям
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

# ——— ETA (эвристика до подключения реальных карт) ———
CITY_EQ_MIN   = 12  # если города совпадают
CITY_NEAR_MIN = 22  # если разные, но оба в нашем регионе (очень грубо)
CITY_FAR_MIN  = 35  # далеко

def estimate_eta_minutes(src_addr: str, dst_addr: str) -> int:
    city_a = safe_city(src_addr).lower()
    city_b = safe_city(dst_addr).lower()
    if not city_a or not city_b:
        return CITY_NEAR_MIN
    if city_a == city_b:
        return CITY_EQ_MIN
    # примитивно: внутри North-East FL — "near", иначе "far"
    near_cities = {"st. augustine", "jacksonville", "orange park", "st augustine"}
    if city_a in near_cities and city_b in near_cities:
        return CITY_NEAR_MIN
    return CITY_FAR_MIN

def risk_label(eta: int) -> str:
    if eta <= VERY_CLOSE_THRESH:
        return "low"
    if eta <= NORMAL_THRESH:
        return "medium"
    return "high"

# ====== DEBUG ======
@app.get("/health")
def health():
    return {"status": "ok", "tz": str(APP_TZ)}

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
            "employee_id": emp_id
        })
    return {"count_preview": len(out), "jobs": out}

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

def slot_label(s: datetime, e: datetime) -> str:
    def fmt(dt: datetime) -> str:
        hh = dt.hour
        suf = "am" if hh < 12 else "pm"
        base = hh if 1 <= hh <= 12 else (hh-12 if hh>12 else 12)
        return f"{base}:00{suf}"
    return f"{fmt(s)}–{fmt(e)}"

def within_client_window(s: datetime, e: datetime, day_start: datetime, day_end: datetime,
                         win_start_str: Optional[str], win_end_str: Optional[str]) -> bool:
    cl_s = from_local(day_start, win_start_str) or day_start
    cl_e = from_local(day_start, win_end_str) or day_end
    return (s >= cl_s) and (e <= cl_e)

# ====== BUILD CANDIDATES BY POLICY V3 ======
def build_candidates_for_day(
    emp_id: str,
    dt_day: datetime,
    todays_windows: List[Tuple[datetime, datetime, str]],  # [(ws,we,addr)]
    new_addr: str,
    win_start_str: Optional[str],
    win_end_str: Optional[str],
    first_slot_home: Optional[str],
    last_slot_home: Optional[str]
) -> List[dict]:
    """
    Возвращает кандидатов по сетке согласно политике v3 + клиентским ограничениям.
    """
    day_start = datetime.combine(dt_day.date(), DEFAULT_WORK_START, tzinfo=APP_TZ)
    day_end   = datetime.combine(dt_day.date(), DEFAULT_WORK_END, tzinfo=APP_TZ)

    # Занятые сеточные старт-часы у техника в этот день:
    occupied_hours = set()
    for ws, we, _addr in todays_windows:
        occupied_hours.add(ws.hour)

    results: List[dict] = []

    # Помощники для ETA от/к дому
    def eta_from_home(start_hour: int) -> int:
        if not first_slot_home:
            return NORMAL_THRESH  # без данных считаем "нормально"
        s, e = grid_slot(dt_day, start_hour)
        return estimate_eta_minutes(first_slot_home, new_addr)

    def eta_to_home(start_hour: int) -> int:
        if not last_slot_home:
            return NORMAL_THRESH
        s, e = grid_slot(dt_day, start_hour)
        return estimate_eta_minutes(new_addr, last_slot_home)

    # Если нет визитов в этот день: первый слот — базовая сетка с учётом клиента и дома
    if not todays_windows:
        for h in GRID_STARTS:
            if h in occupied_hours:
                continue
            s, e = grid_slot(dt_day, h)
            if not within_client_window(s, e, day_start, day_end, win_start_str, win_end_str):
                continue
            # ранжируем по ETA от дома и к дому (последний слот пока неизвестен, учитываем только от дома)
            eta_h = eta_from_home(h)
            results.append({
                "employee_id": emp_id,
                "tech_name": NAME_MAP.get(emp_id, "Technician"),
                "slot_start": s.isoformat(),
                "slot_end": e.isoformat(),
                "ETA_prev": eta_h,
                "ETA_next": None,
                "risk": risk_label(eta_h),
                "note": "first of day (home proximity)"
            })
        return results

    # Иначе — вставки между окнами + после последнего
    # Рассматриваем каждую пару (A=prev, B=next) + «хвост» после последнего
    anchors = [(None, None, None)] + todays_windows + [(None, None, None)]
    for i in range(len(anchors)-1):
        A = anchors[i]
        C = anchors[i+1]
        prev = A if A[0] is not None else None
        nxt  = C if C[0] is not None else None

        # базовый час после A
        base_after_prev = 9
        if prev:
            base_after_prev = prev[0].hour + 2  # соседняя ступенька (9->11, 11->13, …)
        # возможен "смещённый" час при очень близкой дороге
        shifted_after_prev = base_after_prev - 1  # 9->10, 11->12, …

        # перебор разрешённых слотов для этого интервала
        # правила: если ETA(prev->new) ≤15 → можно shifted (10–1), иначе — base (11–2)
        # если prev отсутствует (первый возможный в середине дня) — ориентируемся на базовую сетку
        candidate_hours = []
        for h in GRID_STARTS:
            # Слот должен хронологически идти после prev и до next (по сетке)
            if prev and h <= prev[0].hour - 1:  # безопасная отсечка
                continue
            if nxt and h >= nxt[0].hour + 1:    # не уходить далеко за next (контроль плотности)
                continue
            candidate_hours.append(h)

        for h in candidate_hours:
            if h in occupied_hours:
                continue
            s, e = grid_slot(dt_day, h)
            if not within_client_window(s, e, day_start, day_end, win_start_str, win_end_str):
                continue

            # Проверяем правило ETA и смещения
            allow = False
            used_shift = False
            eta_prev = None
            if prev:
                eta_prev = estimate_eta_minutes(prev[2], new_addr)
                if eta_prev <= VERY_CLOSE_THRESH and h == shifted_after_prev:
                    allow = True
                    used_shift = True
                elif h == base_after_prev:
                    allow = True
                else:
                    allow = False
            else:
                # Нет предыдущего (теоретически это "первый в блоке между None и С"),
                # используем базовую сетку без смещения
                if h in GRID_STARTS:
                    allow = True

            if not allow:
                continue

            # Рассчитываем ETA к next/к дому, если слот последний
            eta_next = None
            if nxt:
                eta_next = estimate_eta_minutes(new_addr, nxt[2])
            else:
                eta_next = eta_to_home(h)

            # Риск — по худшему из prev/next
            worst_eta = max(eta for eta in [eta_prev, eta_next] if eta is not None)
            results.append({
                "employee_id": emp_id,
                "tech_name": NAME_MAP.get(emp_id, "Technician"),
                "slot_start": s.isoformat(),
                "slot_end": e.isoformat(),
                "ETA_prev": eta_prev,
                "ETA_next": eta_next,
                "risk": risk_label(worst_eta),
                "note": ("shifted (<=15min)" if used_shift else "base step")
                        + ("; last of day (home)" if nxt is None else "")
            })

    return results

# ====== MAIN ======
class SuggestOut(BaseModel):
    address: str
    timezone: str
    visit_minutes: int
    suggestions: List[dict]

@app.post("/suggest", response_model=SuggestOut)
def suggest(body: SuggestIn):
    try:
        visit_minutes = body.visit_minutes or VISIT_MINUTES_DEFAULT
        buffer_minutes = body.buffer_minutes or BUFFER_MINUTES_DEFAULT

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

        # По каждому технику, по каждому дню горизонта
        for emp_id, windows in arrival_by_emp.items():
            # группируем по дате
            buckets: Dict[str, List[Tuple[datetime, datetime, str]]] = {}
            for ws, we, addr in windows:
                key = ws.date().isoformat()
                buckets.setdefault(key, []).append((ws, we, addr))
            # проход по дням
            for d in range(max(1, body.days_ahead)):
                day_dt = now_local + timedelta(days=d)
                day_key = day_dt.date().isoformat()
                todays = sorted(buckets.get(day_key, []), key=lambda t: t[0])

                # фильтр по дням недели (клиент «только ср и пт», например)
                if body.preferred_days and weekday_code(day_dt) not in set(body.preferred_days):
                    continue

                first_home = HOME_MAP.get(emp_id)
                last_home  = HOME_MAP.get(emp_id)

                cand = build_candidates_for_day(
                    emp_id=emp_id,
                    dt_day=day_dt,
                    todays_windows=todays,
                    new_addr=body.address,
                    win_start_str=body.window_start,
                    win_end_str=body.window_end,
                    first_slot_home=first_home,
                    last_slot_home=last_home
                )
                suggestions.extend(cand)

        # Ранжирование: риск (low < medium < high), потом ближайшее время
        risk_order = {"low": 0, "medium": 1, "high": 2}
        suggestions.sort(key=lambda x: (risk_order.get(x.get("risk","medium"), 1), x["slot_start"]))

        # Уберём возможные дубли по (tech, slot_start)
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
            "suggestions": deduped[:6]
        }

    except Exception as e:
        # не падаем 502, а отдаём читаемую ошибку
        return {"address": body.address, "timezone": str(APP_TZ), "visit_minutes": VISIT_MINUTES_DEFAULT, "suggestions": [], "error": str(e)}
