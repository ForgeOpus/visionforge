# Installation Guide

Set up VisionForge on your system with this comprehensive installation guide.

## 🎯 Overview

VisionForge consists of two main components:
- **Backend**: Django-based API server
- **Frontend**: React-based web interface

## 📋 Prerequisites

Before installing VisionForge, ensure you have:

### Required Software
- **Python 3.8+** - Backend runtime
- **Node.js 16+** - Frontend development
- **npm** or **yarn** - Package manager

### Optional but Recommended
- **Git** - Version control
- **VS Code** - Code editor with extensions
- **Google Gemini API Key** - For AI assistant features

## 🚀 Quick Installation

### Option 1: Using Git (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/devgunnu/visionforge.git
   cd visionforge
   ```

2. **Install backend dependencies**
   ```bash
   cd project
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

### Option 2: Download ZIP

1. Download and extract the ZIP file
2. Follow steps 2-3 from Option 1

## 🔧 Detailed Setup

### Backend Setup

1. **Navigate to project directory**
   ```bash
   cd visionforge/project
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file and add your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Initialize database**
   ```bash
   python manage.py migrate
   ```

6. **Create superuser** (optional)
   ```bash
   python manage.py createsuperuser
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd visionforge/project/frontend
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env.local
   ```

## 🏃‍♂️ Running the Application

### Start Backend Server

1. **Navigate to project directory**
   ```bash
   cd visionforge/project
   ```

2. **Start Django server**
   ```bash
   python manage.py runserver
   ```

3. **Verify backend is running**
   - Open `http://localhost:8000` in browser
   - You should see Django welcome page or API documentation

### Start Frontend Development Server

1. **Navigate to frontend directory**
   ```bash
   cd visionforge/project/frontend
   ```

2. **Start development server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

3. **Access VisionForge**
   - Open `http://localhost:5173` in browser
   - You should see the VisionForge interface

## 🔍 Verification

### Check Backend Health

```bash
# Test API endpoints
curl http://localhost:8000/api/projects/
```

### Check Frontend Health

- Open browser developer tools
- Check for any console errors
- Verify network requests to backend

## 🛠️ Common Issues & Solutions

### Port Already in Use

**Problem**: `Port 8000 is already in use`

**Solution**:
```bash
# Kill process on port 8000
# On Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# On macOS/Linux
lsof -ti:8000 | xargs kill -9

# Or use different port
python manage.py runserver 8080
```

### Node.js Version Issues

**Problem**: `Node.js version not supported`

**Solution**:
```bash
# Check current version
node --version

# Upgrade Node.js using nvm
nvm install 18
nvm use 18
```

### Python Dependencies

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or upgrade pip first
pip install --upgrade pip
```

### CORS Issues

**Problem**: `CORS policy: No 'Access-Control-Allow-Origin'`

**Solution**:
- Ensure both servers are running
- Check backend CORS settings in `settings.py`
- Verify frontend API URL configuration

### Database Issues

**Problem**: `django.db.utils.OperationalError`

**Solution**:
```bash
# Delete and recreate database
rm db.sqlite3
python manage.py migrate
```

## 🐳 Docker Installation (Alternative)

### Using Docker Compose

1. **Create Dockerfile**
   ```dockerfile
   # Backend Dockerfile
   FROM python:3.9
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
   ```

2. **Create docker-compose.yml**
   ```yaml
   version: '3.8'
   services:
     backend:
       build: ./project
       ports:
         - "8000:8000"
       environment:
         - GEMINI_API_KEY=${GEMINI_API_KEY}
     
     frontend:
       build: ./project/frontend
       ports:
         - "5173:5173"
       depends_on:
         - backend
   ```

3. **Run with Docker**
   ```bash
   docker-compose up
   ```

## 📱 Development Environment Setup

### VS Code Extensions

Install these extensions for optimal development:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.flake8",
    "ms-python.black-formatter",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-typescript-next"
  ]
}
```

### Git Configuration

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## 🚀 Production Deployment

### Backend Production

1. **Set environment variables**
   ```bash
   export DEBUG=False
   export SECRET_KEY=your_production_secret_key
   export ALLOWED_HOSTS=yourdomain.com
   ```

2. **Collect static files**
   ```bash
   python manage.py collectstatic
   ```

3. **Use production server**
   ```bash
   pip install gunicorn
   gunicorn --bind 0.0.0.0:8000 backend.wsgi:application
   ```

### Frontend Production

1. **Build for production**
   ```bash
   npm run build
   ```

2. **Serve static files**
   ```bash
   npm install -g serve
   serve -s dist
   ```

## ✅ Installation Checklist

Before proceeding:

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] Database migrated
- [ ] Environment variables configured
- [ ] Backend server running on port 8000
- [ ] Frontend server running on port 5173
- [ ] No CORS errors in browser console
- [ ] API endpoints accessible

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**:
   - Backend: Django console output
   - Frontend: Browser developer tools

2. **Verify requirements**:
   - Python and Node.js versions
   - All dependencies installed

3. **Consult documentation**:
   - [Troubleshooting Guide](../../troubleshooting/common-issues.md)
   - [GitHub Issues](https://github.com/devgunnu/visionforge/issues)

4. **Community support**:
   - Join our Discord community
   - Ask questions on GitHub Discussions

---

**Ready to start?** → [Quick Start Guide](quickstart.md)
