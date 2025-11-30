# Single-Service Render Deployment Guide

This guide explains how to deploy VisionForge as a **single Django service** that serves both the API and React frontend on Render's free tier.

## Architecture Overview

- **Single Service**: Django Web Service (serves API + React static files)
- **No CORS issues**: Frontend and backend on same origin
- **WhiteNoise**: Serves React static files efficiently
- **Simpler**: Only one service to manage

## How It Works

1. React app is built into static files (HTML, CSS, JS)
2. Static files are copied to Django's `frontend_build` directory
3. WhiteNoise serves the static assets
4. Django catches all routes and serves `index.html`
5. React Router handles client-side routing
6. API calls use relative paths (`/api`) - same origin

---

## Step 1: Generate Django SECRET_KEY

Before deploying, generate a new secret key for production:

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Save this output - you'll need it for environment variables.

---

## Step 2: Test Locally (Optional but Recommended)

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

Visit `http://localhost:8000` - you should see your React app served by Django!

---

## Step 3: Deploy to Render

### 3.1 Create Web Service

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your Git repository
4. Configure the service:

**Service Configuration:**

| Field | Value |
|-------|-------|
| **Name** | `visionforge` (or your preferred name) |
| **Region** | Choose closest to your users |
| **Branch** | `main` (or your default branch) |
| **Root Directory** | `project` |
| **Runtime** | `Python 3` |
| **Instance Type** | `Free` |

**Build Command:**
```bash
bash build_frontend.sh && pip install -r requirements.txt && python manage.py collectstatic --noinput
```

**Start Command:**
```bash
gunicorn backend.wsgi:application --bind 0.0.0.0:$PORT
```

### 3.2 Add Environment Variables

Click **"Advanced"** → **"Add Environment Variable"**

**Required Variables:**

| Key | Value | Description |
|-----|-------|-------------|
| `SECRET_KEY` | Your generated secret key | Django secret (KEEP SECRET!) |
| `DEBUG` | `False` | Disable debug mode in production |
| `PYTHON_VERSION` | `3.11.0` | Your Python version |

**Optional Variables (if using AI features):**

| Key | Value | Description |
|-----|-------|-------------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | For Claude AI |
| `GOOGLE_API_KEY` | Your Google API key | For Gemini AI |

**Note**: `RENDER_EXTERNAL_HOSTNAME` is automatically set by Render.

### 3.3 Deploy

1. Click **"Create Web Service"**
2. Wait for build to complete (first deploy takes 10-15 minutes)
3. Note your URL: `https://visionforge.onrender.com`

---

## Step 4: Verify Deployment

1. Visit your Render URL: `https://your-service-name.onrender.com`
2. Open browser DevTools (F12) → Network tab
3. Verify:
   - ✅ React app loads
   - ✅ API calls go to `/api` (relative path)
   - ✅ No CORS errors
   - ✅ Static assets load from `/static/`

---

## Understanding the Build Process

### What happens during Render build:

1. **`bash build_frontend.sh`**
   - Installs npm packages
   - Runs `npm run build` (creates `frontend/dist/`)
   - Copies `dist/` contents to `frontend_build/`

2. **`pip install -r requirements.txt`**
   - Installs Python dependencies including:
     - `gunicorn` (production server)
     - `whitenoise` (static file serving)
     - Django and other packages

3. **`python manage.py collectstatic --noinput`**
   - Collects all static files into `staticfiles/`
   - Includes React assets from `frontend_build/assets/`
   - WhiteNoise compresses and optimizes files

4. **Service starts with Gunicorn**
   - Django serves on port `$PORT` (set by Render)
   - WhiteNoise middleware serves static files
   - All routes go to Django
   - Unmatched routes serve React's `index.html`

---

## How Routing Works

### API Routes (handled by Django):
- `/api/*` → Django REST API endpoints
- `/admin/` → Django admin panel

### Frontend Routes (handled by React):
- `/` → React app (index.html)
- `/dashboard` → React Router
- `/settings` → React Router
- Everything else → React Router

Django's catch-all route in `urls.py`:
```python
re_path(r'^.*$', TemplateView.as_view(template_name='index.html'))
```

This serves `index.html` for any non-API route, allowing React Router to handle navigation.

---

## Development vs Production

### Development (two separate servers):

**Backend:**
```bash
cd project
python manage.py runserver
# Runs on http://localhost:8000
```

**Frontend:**
```bash
cd project/frontend
npm run dev
# Runs on http://localhost:5173
```

- Frontend uses `VITE_API_URL=http://localhost:8000/api` in `.env`
- CORS is enabled for `localhost:5173`

### Production (single server):

**Single Django server:**
```bash
cd project
bash build_frontend.sh
python manage.py collectstatic --noinput
gunicorn backend.wsgi:application
# Runs on Render's assigned port
```

- Frontend uses relative path `/api` (no VITE_API_URL)
- No CORS needed (same origin)
- Static files served by WhiteNoise

---

## Troubleshooting

### Build Fails

**Problem**: `bash: build_frontend.sh: command not found`

**Solution**: Make script executable:
```bash
chmod +x build_frontend.sh
```

Or use inline build command in Render:
```bash
cd frontend && npm install && npm run build && cd .. && mkdir -p frontend_build && cp -r frontend/dist/* frontend_build/ && pip install -r requirements.txt && python manage.py collectstatic --noinput
```

---

### Static Files Not Loading

**Problem**: 404 errors for CSS/JS files

**Solutions**:
1. Check `collectstatic` ran successfully in build logs
2. Verify `STATIC_ROOT` and `STATICFILES_DIRS` in settings.py
3. Ensure WhiteNoise is in MIDDLEWARE
4. Check that `frontend_build/assets/` exists after build

---

### React Routes Return 404

**Problem**: Direct navigation to `/dashboard` returns 404

**Solution**: Ensure catch-all route is **last** in `urls.py`:
```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('block_manager.urls')),
    re_path(r'^.*$', TemplateView.as_view(template_name='index.html')),  # Must be last!
]
```

---

### API Calls Fail

**Problem**: API calls to `/api/` return 404 or 500

**Solutions**:
1. Check Django logs in Render dashboard
2. Verify `SECRET_KEY` is set in environment variables
3. Ensure `DEBUG=False` in production
4. Check that API routes in `block_manager/urls.py` are correct

---

### Cold Starts

**Problem**: First request after 15 minutes is very slow (30+ seconds)

**Explanation**: Render's free tier spins down services after 15 minutes of inactivity.

**Solutions**:
- Accept the tradeoff for free hosting
- Upgrade to paid tier ($7/month) for always-on
- Use UptimeRobot to ping every 14 minutes (keeps service awake)

---

## File Structure After Build

```
project/
├── backend/              # Django backend
├── block_manager/        # Django app
├── frontend/             # React source code
│   └── dist/            # Vite build output (generated)
├── frontend_build/       # Copy of dist/ for Django (generated)
│   ├── index.html
│   └── assets/
│       ├── index-abc123.js
│       └── index-def456.css
├── staticfiles/          # Collected static files (generated)
│   └── assets/          # Frontend assets
├── manage.py
├── requirements.txt
└── build_frontend.sh
```

---

## Environment Variables Summary

### Production (Render):

```bash
# Required
SECRET_KEY=your-generated-secret-key-here
DEBUG=False
PYTHON_VERSION=3.11.0

# Optional
ANTHROPIC_API_KEY=your-key
GOOGLE_API_KEY=your-key

# Auto-set by Render
RENDER_EXTERNAL_HOSTNAME=your-service.onrender.com
PORT=10000
```

### Development (Local):

Create `project/frontend/.env`:
```bash
# For development with separate backend server
VITE_API_URL=http://localhost:8000/api
```

For production build (or omit entirely to use relative `/api`):
```bash
# Leave blank or omit - uses relative path
# VITE_API_URL=
```

---

## Updating Your Deployment

### Automatic Deploys

Render auto-deploys when you push to your Git repository:

1. Make changes locally
2. Commit and push:
   ```bash
   git add .
   git commit -m "Your changes"
   git push origin main
   ```
3. Render automatically rebuilds and deploys

### Manual Deploy

In Render dashboard:
1. Go to your service
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

---

## Database Considerations

### Current Setup: SQLite (Not Recommended for Production)

SQLite databases on Render are **ephemeral** - they reset when the service restarts.

### Upgrade to PostgreSQL (Recommended)

1. **Create PostgreSQL Database in Render:**
   - Click **"New +"** → **"PostgreSQL"**
   - Choose Free tier (90 days, then $7/month)
   - Note the `DATABASE_URL`

2. **Update `requirements.txt`:**
   ```txt
   psycopg2-binary>=2.9.9
   dj-database-url>=2.1.0
   ```

3. **Update `settings.py`:**
   ```python
   import dj_database_url
   
   DATABASES = {
       'default': dj_database_url.config(
           default='sqlite:///' + str(BASE_DIR / 'db.sqlite3'),
           conn_max_age=600
       )
   }
   ```

4. **Add `DATABASE_URL` to Render environment variables:**
   - Copy from PostgreSQL database info page
   - Add to web service environment variables

5. **Run migrations on first deploy:**
   - In Render shell: `python manage.py migrate`

---

## Cost Summary

**Free Tier:**
- 750 hours/month web service
- 512MB RAM
- Services spin down after 15 minutes inactivity
- 90 days free PostgreSQL, then $7/month

**Single Service Usage:**
- ~750 hours/month (within free tier)
- **Total Cost**: $0/month (with SQLite)
- **With PostgreSQL**: $7/month after 90 days

---

## Advantages of Single-Service

✅ **No CORS issues** - same origin
✅ **Simpler deployment** - one service
✅ **Uses less free tier hours** - no separate frontend
✅ **Easier to manage** - one URL, one service
✅ **Better for free tier** - static site serves from CDN anyway

## Disadvantages

❌ **Longer builds** - must build frontend every deploy
❌ **Coupled deployment** - frontend/backend deploy together
❌ **Less scalable** - can't scale frontend separately

---

## Next Steps

1. ✅ Set up PostgreSQL for persistent data
2. Set up custom domain (optional)
3. Configure monitoring and alerts
4. Set up CI/CD for automated testing
5. Add health check endpoint
6. Set up logging and error tracking

---

## Quick Reference Commands

### Local Development (Separate Servers):
```bash
# Terminal 1 - Backend
cd project
python manage.py runserver

# Terminal 2 - Frontend  
cd project/frontend
npm run dev
```

### Local Testing (Single Server):
```bash
# Windows
cd project
build_frontend.bat
python manage.py collectstatic --noinput
python manage.py runserver

# macOS/Linux
cd project
bash build_frontend.sh
python manage.py collectstatic --noinput
python manage.py runserver
```

### Render Deployment:
**Build Command:**
```bash
bash build_frontend.sh && pip install -r requirements.txt && python manage.py collectstatic --noinput
```

**Start Command:**
```bash
gunicorn backend.wsgi:application --bind 0.0.0.0:$PORT
```

---

## Support Resources

- [Render Documentation](https://render.com/docs)
- [Django Deployment Checklist](https://docs.djangoproject.com/en/stable/howto/deployment/checklist/)
- [WhiteNoise Documentation](http://whitenoise.evans.io/)
- [Gunicorn Documentation](https://docs.gunicorn.org/)

---

## Summary

You've successfully configured VisionForge for single-service deployment! 

The app will:
- Build React into static files during deployment
- Serve both API and frontend from one Django service
- Have no CORS issues (same origin)
- Work on Render's free tier

**Ready to deploy!** 🚀
