# VisionForge - Single-Service Render Deployment Quick Start

## ✅ Code Changes Complete

All necessary code changes have been made for single-service deployment!

## 📋 What Changed

### 1. **Backend (`project/backend/settings.py`)**
- ✅ Added WhiteNoise middleware for serving static files
- ✅ Configured `STATIC_ROOT` and `STATICFILES_DIRS`
- ✅ Updated `TEMPLATES` to include `frontend_build/`
- ✅ Set dynamic `ALLOWED_HOSTS` for Render
- ✅ Simplified CORS (only for local dev)
- ✅ Added production-ready SECRET_KEY handling

### 2. **URLs (`project/backend/urls.py`)**
- ✅ Added catch-all route to serve React's `index.html`
- ✅ Ensures React Router works on all routes

### 3. **Frontend API (`project/frontend/src/lib/api.ts`)**
- ✅ Changed to use relative `/api` path (no CORS needed)
- ✅ Falls back to `http://localhost:8000/api` for local dev

### 4. **Dependencies (`project/requirements.txt`)**
- ✅ Added `gunicorn` (production server)
- ✅ Added `whitenoise` (static file serving)

### 5. **Build Scripts**
- ✅ Created `build_frontend.sh` (macOS/Linux)
- ✅ Created `build_frontend.bat` (Windows)

---

## 🚀 Deploy to Render - Step by Step

### Step 1: Generate SECRET_KEY
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```
**Save this output!**

---

### Step 2: Create Web Service on Render

1. Go to https://dashboard.render.com/
2. Click **"New +"** → **"Web Service"**
3. Connect your Git repository
4. Fill in these settings:

| Field | Value |
|-------|-------|
| **Name** | `visionforge` |
| **Root Directory** | `project` |
| **Runtime** | `Python 3` |
| **Build Command** | `bash build_frontend.sh && pip install -r requirements.txt && python manage.py collectstatic --noinput` |
| **Start Command** | `gunicorn backend.wsgi:application --bind 0.0.0.0:$PORT` |
| **Instance Type** | `Free` |

---

### Step 3: Add Environment Variables

Click **"Advanced"** → Add these:

| Key | Value |
|-----|-------|
| `SECRET_KEY` | *Your generated key from Step 1* |
| `DEBUG` | `False` |
| `PYTHON_VERSION` | `3.11.0` |

Optional (if using AI):
| Key | Value |
|-----|-------|
| `ANTHROPIC_API_KEY` | *Your key* |
| `GOOGLE_API_KEY` | *Your key* |

---

### Step 4: Deploy!

1. Click **"Create Web Service"**
2. Wait 10-15 minutes for first build
3. Visit your URL: `https://visionforge.onrender.com`

---

## 🧪 Test Locally First (Optional)

### On Windows:
```bash
cd project
build_frontend.bat
python manage.py collectstatic --noinput
python manage.py runserver
```

### On macOS/Linux:
```bash
cd project
bash build_frontend.sh
python manage.py collectstatic --noinput
python manage.py runserver
```

Visit: http://localhost:8000

---

## 🔍 Verify Deployment

Open your deployed site and check:
- ✅ React app loads
- ✅ No CORS errors in console (F12)
- ✅ API calls use `/api` (relative path)
- ✅ Routing works (try navigating)

---

## 📝 Development Workflow

For **local development**, continue using separate servers:

**Terminal 1 - Backend:**
```bash
cd project
python manage.py runserver
```

**Terminal 2 - Frontend:**
```bash
cd project/frontend
npm run dev
```

Create `project/frontend/.env`:
```bash
VITE_API_URL=http://localhost:8000/api
```

---

## 🎯 Key Benefits

✅ **Single service** - simpler deployment
✅ **No CORS issues** - same origin
✅ **Free tier friendly** - only uses one service
✅ **Production ready** - WhiteNoise + Gunicorn

---

## 📚 Full Documentation

See `RENDER_SINGLE_SERVICE_DEPLOYMENT.md` for:
- Complete architecture explanation
- Troubleshooting guide
- Database upgrade instructions
- Advanced configuration

---

## ⚡ That's It!

Your code is ready to deploy! Just follow the 4 steps above.

**Questions?** Check the full guide: `RENDER_SINGLE_SERVICE_DEPLOYMENT.md`
