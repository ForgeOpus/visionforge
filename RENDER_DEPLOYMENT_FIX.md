# Render Deployment Guide - Single Service

## Important: Use `/api` NOT Full URL

For single-service deployment (Django serves both API and React), **always use `/api`** as a relative path:
- ✅ Correct: `VITE_API_URL=/api`
- ❌ Wrong: `VITE_API_URL=https://your-app.onrender.com/api`

The relative path ensures the frontend talks to the same server (no CORS issues).

---

## Manual Render Setup (No render.yaml)

### Step 1: Create Web Service
1. Go to Render Dashboard → **New +** → **Web Service**
2. Connect your GitHub repository
3. Select branch: `deploy2` (or your main branch)
4. Configure:
   - **Name**: `visionforge` (or your choice)
   - **Region**: Oregon (or closest to you)
   - **Branch**: `deploy2`
   - **Root Directory**: `project`
   - **Runtime**: Python 3
   - **Build Command**:
     ```bash
     pip install -r requirements.txt && chmod +x build_frontend.sh && ./build_frontend.sh && python manage.py collectstatic --no-input
     ```
   - **Start Command**:
     ```bash
     gunicorn backend.wsgi:application
     ```

### Step 2: Environment Variables

Add these in Render Dashboard → Environment tab:

```bash
# Django Core (REQUIRED)
SECRET_KEY=<generate-secure-random-key>
DEBUG=False
RENDER_EXTERNAL_HOSTNAME=<your-app-name>.onrender.com

# Frontend API URL (CRITICAL)
VITE_API_URL=/api

# AI Provider (choose one)
AI_PROVIDER=gemini
GEMINI_API_KEY=<your-gemini-api-key>

# OR if using Claude:
# AI_PROVIDER=claude
# ANTHROPIC_API_KEY=<your-anthropic-key>

# Python Version (optional but recommended)
PYTHON_VERSION=3.12.5
```

**To generate SECRET_KEY:**
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### Step 3: Deploy

1. Click **Create Web Service**
2. Wait for deployment (takes 5-10 minutes on free tier)
3. Your app will be at: `https://<your-app-name>.onrender.com`

---

## Why `/api` Instead of Full URL?

**Single-Service Architecture:**
```
https://your-app.onrender.com/          → React frontend
https://your-app.onrender.com/api/...   → Django API
```

Both served from **the same domain** = no CORS issues!

When frontend uses `/api`, the browser automatically makes requests to:
- `https://your-app.onrender.com/api/projects/`
- `https://your-app.onrender.com/api/node-definitions`
- etc.

**Two-Service Architecture** (not used here):
```
https://frontend.onrender.com/     → React
https://backend.onrender.com/api/  → Django API
```
This requires CORS configuration and is more complex.

---

## Troubleshooting

### Issue: "Failed to fetch" or CORS errors
**Cause**: `VITE_API_URL` not set or set incorrectly
**Fix**: Set `VITE_API_URL=/api` in Render environment variables and redeploy

### Issue: "Unexpected token '<'" or getting HTML instead of JSON
**Cause**: URL routing issue - catch-all intercepting API routes
**Fix**: Already fixed in `backend/urls.py` - ensure `path('api/', ...)` comes before catch-all

### Issue: 403 Forbidden on POST requests
**Cause**: CSRF token not being sent
**Fix**: Already implemented in `apiUtils.ts` and `projectApi.ts`

### Issue: Old JavaScript being served
**Cause**: Browser cache
**Fix**: Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

---

## Current Environment Variables

**Your Current Setup:**
```bash
SECRET_KEY=<generate-new-for-production>
DEBUG=False
RENDER_EXTERNAL_HOSTNAME=visionforge-bz3s.onrender.com
GEMINI_API_KEY=AIzaSyBphxkgfPinCAeXDf0MhKJekgB3pVeLOUE
VITE_API_URL=/api
```

---

## Local Development

For local development, frontend uses `frontend/.env`:
```bash
VITE_API_URL=http://localhost:8000/api
```

This allows the React dev server (port 5173) to talk to Django (port 8000).

**Production** uses `/api` (relative path) since both are served from the same domain.
