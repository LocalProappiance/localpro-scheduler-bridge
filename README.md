
# LocalPRO Scheduler Bridge (FastAPI)

This is a tiny API you can deploy on **DigitalOcean App Platform**. It keeps your **Housecall Pro** API key on *your* side and exposes a safe endpoint I (ChatGPT) can call to suggest appointment slots.

## What it does (today)
- `/health` — quick check.
- `/suggest` — accepts a new client address and returns example slots (placeholder logic).

> The placeholder logic will be enough to finish the DO setup. After it's live, we’ll switch to real distance matrix & your real schedule.

---

## Files
- `app.py` — FastAPI app.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — makes DigitalOcean deploy deterministic.
- `.env.example` — shows env vars (copy to DO as App Settings → Environment Variables).

---

## Step‑by‑step: GitHub → DigitalOcean

### 1) Create a GitHub repo (from web UI)
1. Go to https://github.com/new
2. **Repository name:** `localpro-scheduler-bridge`
3. Visibility: Public or Private — either is fine.
4. Click **Create repository**.
5. Click **Upload an existing file** and upload the four files from this ZIP:
   - `app.py`
   - `requirements.txt`
   - `Dockerfile`
   - `.env.example`
6. Click **Commit changes**.

### 2) Connect DigitalOcean App Platform
1. In DigitalOcean, **Create** → **Apps** → **Create App**.
2. Choose **GitHub** and select your `localpro-scheduler-bridge` repo and the default branch.
3. DO will detect a Dockerfile and suggest **Web Service**.
4. For **Run Command**, keep default (Dockerfile CMD handles it).

### 3) Configure environment variables (very important)
- Add a variable:
  - **Key:** `HCP_API_KEY`
  - **Value:** your Housecall Pro **read‑only** API key
  - **Type:** Encrypted/Secret
- (Optional) `TZ = America/New_York`

### 4) Deploy
- Click **Deploy**.
- When live, open **App URL** and append `/health` to test.
  - Expect: `{"status":"ok"}`

### 5) Test /suggest
Use the public URL (replace `YOUR_APP_URL`):
```bash
curl -X POST "https://YOUR_APP_URL/suggest"   -H "Content-Type: application/json"   -d '{"address":"24 Cathedral Pl, St. Augustine, FL", "days_ahead": 3}'
```

You should get a JSON with 1–3 example slots.

---

## What’s next
After deployment works, we’ll:
1. Swap the placeholder logic for real Housecall Pro schedule reads.
2. Add Distance Matrix (Google/OSRM) for accurate drive times.
3. Add your rules (2h visit, buffers, zones) and ranking.
