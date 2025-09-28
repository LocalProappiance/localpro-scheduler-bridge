
import os
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

APP_TZ = os.environ.get("TZ", "America/New_York")
HCP_KEY = os.environ.get("HCP_API_KEY")  # keep this in DO env settings
HCP_BASE = "https://api.housecallpro.com"

app = FastAPI(title="LocalPRO Scheduler Bridge")

class SuggestIn(BaseModel):
    address: str
    days_ahead: int = 3
    preferred_days: Optional[List[str]] = None  # e.g. ["Mon","Tue","Wed"]
    window_start: Optional[str] = None          # "09:00"
    window_end: Optional[str] = None            # "16:00"
    job_type: Optional[str] = None

def hcp_get(path: str, params=None):
    if not HCP_KEY:
        raise HTTPException(status_code=500, detail="HCP_API_KEY is not configured")
    r = requests.get(
        f"{HCP_BASE}{path}",
        headers={
            "Accept": "application/json",
            "Authorization": f"Token {HCP_KEY}"
        },
        params=params or {},
        timeout=20
    )
    if r.status_code != 200:
        # don't leak secrets; pass through minimal info
        raise HTTPException(status_code=502, detail=f"HCP error {r.status_code}")
    return r.json()

@app.get("/health")
def health():
    return {"status": "ok", "tz": APP_TZ}

def placeholder_suggestions(address: str, days_ahead: int):
    # Minimal logic just to confirm the deployment works.
    now = datetime.now()
    base_day = now + timedelta(days=1)
    slots = []
    for i in range(min(days_ahead, 3)):
        day = base_day + timedelta(days=i)
        slot_start = day.replace(hour=10, minute=0, second=0, microsecond=0)
        slot_end = slot_start + timedelta(hours=2)
        slots.append({
            "tech_name": "Tech A (example)",
            "slot_start": slot_start.isoformat(),
            "slot_end": slot_end.isoformat(),
            "delta_drive_min": 18,
            "notes": "example placeholder"
        })
    return slots

@app.post("/suggest")
def suggest(body: SuggestIn):
    # In the next iteration we'll use real HCP data:
    # jobs = hcp_get("/jobs")
    # users = hcp_get("/users")
    # For now, return placeholder suggestions so the app is easy to deploy.
    return {
        "address": body.address,
        "timezone": APP_TZ,
        "suggestions": placeholder_suggestions(body.address, body.days_ahead)
    }
    # === DEBUG ROUTES TO CONFIRM HCP ACCESS FROM THE SERVER ===

@app.get("/debug/hcp-company")
def debug_hcp_company():
    """Quick check: read company info from Housecall Pro."""
    data = hcp_get("/company")
    # Вернём только безопасный минимум
    return {"name": data.get("name", "unknown"), "ok": True}

@app.get("/debug/hcp-jobs")
def debug_hcp_jobs():
    """List first 5 jobs (id + address) to confirm we can read jobs."""
    data = hcp_get("/jobs")
    jobs = []
    # На всякий случай: разные аккаунты могут возвращать список в разных ключах
    raw = data if isinstance(data, list) else data.get("results") or data.get("jobs") or []
    for item in (raw[:5] if isinstance(raw, list) else []):
        jobs.append({
            "id": item.get("id"),
            "address": item.get("address") or item.get("full_address") or item.get("location")
        })
    return {"count_preview": len(jobs), "jobs": jobs}
    # === HELPERS ===
def address_to_str(addr):
    if not isinstance(addr, dict):
        return str(addr)
    parts = []
    for k in ["street", "street_line_2", "city", "state", "zip", "country"]:
        v = addr.get(k)
        if v:
            parts.append(str(v))
    return ", ".join(parts)

def pick_time(job):
    """
    Под разные аккаунты HCP время может лежать в разных ключах.
    Попробуем по очереди, вернём что найдём.
    """
    # варианты: scheduled_start/End, start/End, window_start/End
    candidates = [
        ("scheduled_start", "scheduled_end"),
        ("start", "end"),
        ("window_start", "window_end"),
        ("start_time", "end_time")
    ]
    for s_key, e_key in candidates:
        s = job.get(s_key)
        e = job.get(e_key)
        if s or e:
            return s, e
    return None, None

def pick_tech(job):
    """
    Аналогично для техника: могут быть assigned_to, technician, user или массив assignees.
    """
    for key in ["assigned_to", "technician", "user", "assignees"]:
        if key in job:
            return job.get(key)
    return None

# === NEW DEBUG ROUTES ===

@app.get("/debug/hcp-jobs-compact")
def debug_hcp_jobs_compact():
    """
    Вернём 5 заявок: id, полный адрес строкой, время (если есть), и кого видим назначенным.
    """
    data = hcp_get("/jobs")
    raw = data if isinstance(data, list) else data.get("results") or data.get("jobs") or []
    out = []
    for item in (raw[:5] if isinstance(raw, list) else []):
        addr_str = address_to_str(item.get("address") or item.get("service_address") or {})
        start, end = pick_time(item)
        tech = pick_tech(item)
        out.append({
            "id": item.get("id"),
            "address": addr_str,
            "start": start,
            "end": end,
            "assigned": tech
        })
    return {"count_preview": len(out), "jobs": out}

@app.get("/debug/hcp-one-raw")
def debug_hcp_one_raw():
    """
    Вернём урезанный «сырой» объект первой заявки (без секретов), чтобы увидеть точные ключи.
    """
    data = hcp_get("/jobs")
    raw = data if isinstance(data, list) else data.get("results") or data.get("jobs") or []
    if not isinstance(raw, list) or not raw:
        return {"ok": True, "job": None}
    item = raw[0]
    # Урезаем до полезных ключей верхнего уровня (без вложенных больших списков)
    keep_keys = ["id","address","service_address","assigned_to","technician","user",
                 "assignees","scheduled_start","scheduled_end","start","end",
                 "window_start","window_end","start_time","end_time","status","title","type"]
    trimmed = {k: item.get(k) for k in keep_keys if k in item}
    return {"ok": True, "job": trimmed}


