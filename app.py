import os
from datetime import datetime, timedelta
from typing import Any, Optional, List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

APP_TZ = os.environ.get("TZ", "America/New_York")
HCP_KEY = os.environ.get("HCP_API_KEY")  # ключ хранится в переменных окружения на DO
HCP_BASE = "https://api.housecallpro.com"

app = FastAPI(title="LocalPRO Scheduler Bridge")

class SuggestIn(BaseModel):
    address: str
    days_ahead: int = 3
    preferred_days: Optional[List[str]] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None
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
        raise HTTPException(status_code=502, detail=f"HCP error {r.status_code}")
    return r.json()

@app.get("/health")
def health():
    return {"status": "ok", "tz": APP_TZ}

# ===== Helpers =====
def address_to_str(addr):
    if not isinstance(addr, dict):
        return str(addr)
    parts = []
    for k in ["street", "street_line_2", "city", "state", "zip", "country"]:
        v = addr.get(k)
        if v:
            parts.append(str(v))
    return ", ".join(parts)

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

def pick_time_recursive(job: dict) -> Tuple[Tuple[str, Any] | None, Tuple[str, Any] | None]:
    start_candidate = None
    end_candidate = None
    start_keys = ["scheduled_start","window_start","start_time","start","scheduled.start","schedule.start"]
    end_keys   = ["scheduled_end","window_end","end_time","end","scheduled.end","schedule.end"]

    for p, v in _walk(job):
        key = p.split(".")[-1].lower()
        if any(k.split(".")[-1] == key for k in start_keys) and v:
            start_candidate = (p, v) if not start_candidate else start_candidate
        if any(k.split(".")[-1] == key for k in end_keys) and v:
            end_candidate = (p, v) if not end_candidate else end_candidate

    if not start_candidate or not end_candidate:
        for p, v in _walk(job):
            if isinstance(v, str):
                low = p.lower()
                if ("start" in low) and not start_candidate and any(ch.isdigit() for ch in v):
                    start_candidate = (p, v)
                if ("end" in low) and not end_candidate and any(ch.isdigit() for ch in v):
                    end_candidate = (p, v)
    return start_candidate, end_candidate

def pick_tech_recursive(job: dict):
    for key in ["assignees","assigned_to","technician","user","assigned","assigned_user","assigned_employee"]:
        if key in job:
            node = job[key]
            if isinstance(node, list) and node:
                first = node[0]
                if isinstance(first, dict):
                    return (key+"[0]", {
                        "id": first.get("id") or first.get("user_id"),
                        "name": first.get("name") or first.get("full_name") or first.get("first_name")
                    })
                else:
                    return (key+"[0]", {"id": first, "name": None})
            if isinstance(node, dict):
                return (key, {
                    "id": node.get("id") or node.get("user_id"),
                    "name": node.get("name") or node.get("full_name") or node.get("first_name")
                })
            return (key, {"id": node, "name": None})
    for p, v in _walk(job):
        low = p.lower()
        if any(t in low for t in ["assignee","assigned","technician","tech","user"]) and isinstance(v, (dict, list, str, int)):
            if isinstance(v, dict):
                return (p, {"id": v.get("id") or v.get("user_id"), "name": v.get("name") or v.get("full_name")})
            if isinstance(v, list) and v:
                first = v[0]
                if isinstance(first, dict):
                    return (p+"[0]", {"id": first.get("id") or first.get("user_id"), "name": first.get("name")})
                return (p+"[0]", {"id": first, "name": None})
            return (p, {"id": v, "name": None})
    return (None, None)

# ===== Debug routes =====
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

@app.get("/debug/hcp-jobs-compact")
def debug_hcp_jobs_compact():
    data = hcp_get("/jobs")
    raw = data if isinstance(data, list) else data.get("results") or data.get("jobs") or []
    out = []
    for item in (raw[:5] if isinstance(raw, list) else []):
        addr_str = address_to_str(item.get("address") or item.get("service_address") or {})
        start_pair, end_pair = pick_time_recursive(item)
        tech_path, tech_obj = pick_tech_recursive(item)
        out.append({
            "id": item.get("id"),
            "address": addr_str,
            "start_path": start_pair[0] if start_pair else None,
            "start": start_pair[1] if start_pair else None,
            "end_path": end_pair[0] if end_pair else None,
            "end": end_pair[1] if end_pair else None,
            "tech_path": tech_path,
            "assigned": tech_obj
        })
    return {"count_preview": len(out), "jobs": out}

@app.get("/debug/hcp-one-raw")
def debug_hcp_one_raw():
    data = hcp_get("/jobs")
    raw = data if isinstance(data, list) else data.get("results") or data.get("jobs") or []
    if not isinstance(raw, list) or not raw:
        return {"ok": True, "job": None}
    item = raw[0]
    keep_keys = ["id","address","service_address","assigned_to","technician","user",
                 "assignees","scheduled_start","scheduled_end","start","end",
                 "window_start","window_end","start_time","end_time","status","title","type"]
    trimmed = {k: item.get(k) for k in keep_keys if k in item}
    return {"ok": True, "job": trimmed}

@app.get("/debug/hcp-users")
def debug_hcp_users():
    data = hcp_get("/users")
    raw = data if isinstance(data, list) else data.get("results") or data.get("users") or []
    out = []
    for u in (raw[:5] if isinstance(raw, list) else []):
        out.append({
            "id": u.get("id"),
            "name": u.get("name") or u.get("full_name") or u.get("first_name"),
            "email": u.get("email"),
            "role": u.get("role") or u.get("type")
        })
    return {"count_preview": len(out), "users": out}

# ===== Suggest (пока заглушка) =====
def placeholder_suggestions(address: str, days_ahead: int):
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
    return {
        "address": body.address,
        "timezone": APP_TZ,
        "suggestions": placeholder_suggestions(body.address, body.days_ahead)
    }
