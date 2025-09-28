
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
