# VisionForge

Build AI models visually. Export production-ready code.

## Overview

VisionForge is a visual AI model builder that allows you to design neural network architectures using a drag-and-drop interface and export production-ready PyTorch or TensorFlow code.

## Tech Stack

- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **Backend**: Django + Django REST Framework
- **Database**: Oracle Cloud Database
- **Authentication**: Firebase Authentication
- **AI Integration**: Google Gemini / Anthropic Claude

## Environment Setup

This project uses **TWO separate .env files** for security and architecture reasons:

### 1. Backend Environment (`project/.env`)
Contains **server-side secrets** that are NEVER exposed to the browser:
- Django secret key
- Firebase Admin SDK credentials
- Oracle database credentials
- AI API keys (Gemini, Claude)

### 2. Frontend Environment (`frontend/.env`)
Contains **client-side configuration** that IS exposed to the browser:
- API URLs (must have `VITE_` prefix)
- Firebase Client SDK configuration

### Setup Instructions

1. **Copy example files and configure:**
   ```bash
   # Backend environment
   cp project/.env.example project/.env
   # Edit project/.env with your actual secrets

   # Frontend environment
   cp frontend/.env.example frontend/.env
   # Edit frontend/.env with your API URL and Firebase client config
   ```

2. **Install backend dependencies:**
   ```bash
   cd project
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   ```

4. **Run database migrations:**
   ```bash
   cd project
   python manage.py migrate
   ```

5. **Start the development servers:**

   **Backend** (in one terminal):
   ```bash
   cd project
   python manage.py runserver
   ```

   **Frontend** (in another terminal):
   ```bash
   cd frontend
   npm run dev
   ```

6. **Access the application:**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000/api

## Why Two .env Files?

**Security**: Separating environment variables prevents accidentally exposing server-side secrets to the browser. Vite bundles all `VITE_` prefixed variables into the client bundle, making them publicly accessible. By keeping backend secrets in a separate file, we ensure they never get exposed.

**Architecture**: The frontend and backend are separate applications that can be deployed independently. Each has its own configuration needs.

## Environment Variables Reference

### Backend (`project/.env`)
```bash
# Django
DJANGO_SECRET_KEY=              # Django secret key
DJANGO_DEBUG=True              # Debug mode (False in production)
DJANGO_ALLOWED_HOSTS=          # Comma-separated list of allowed hosts

# Environment Mode
ENVIRONMENT=DEV                # DEV/LOCAL (use server API keys) or PROD (users bring own keys)

# AI Provider
AI_PROVIDER=gemini            # 'gemini' or 'claude'
GEMINI_API_KEY=               # Google Gemini API key (DEV mode only)
ANTHROPIC_API_KEY=            # Anthropic Claude API key (DEV mode only)

# Firebase Admin SDK
FIREBASE_PROJECT_ID=          # Firebase project ID
FIREBASE_PRIVATE_KEY=         # Firebase service account private key
FIREBASE_CLIENT_EMAIL=        # Firebase service account email

# Oracle Database
ORACLE_USER=                  # Oracle database username
ORACLE_PASSWORD=              # Oracle database password
ORACLE_DSN=                   # Oracle connection string
ORACLE_WALLET_LOCATION=       # Path to Oracle wallet
ORACLE_WALLET_PASSWORD=       # Oracle wallet password
```

### Frontend (`frontend/.env`)
```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000/api

# Firebase Client SDK
VITE_FIREBASE_API_KEY=        # Firebase API key (safe to expose)
VITE_FIREBASE_AUTH_DOMAIN=    # Firebase auth domain
VITE_FIREBASE_PROJECT_ID=     # Firebase project ID
VITE_FIREBASE_STORAGE_BUCKET= # Firebase storage bucket
VITE_FIREBASE_MESSAGING_SENDER_ID= # Firebase messaging sender ID
VITE_FIREBASE_APP_ID=         # Firebase app ID
VITE_FIREBASE_MEASUREMENT_ID= # Firebase measurement ID (optional)
```

## Features

- Visual neural network builder with drag-and-drop interface
- Support for PyTorch and TensorFlow code generation
- Firebase authentication (Google, GitHub)
- Guest mode for quick exploration
- Project management with dashboard
- Real-time validation and error checking
- Export models as production-ready code with training scripts

## Project Structure

```
project/
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── contexts/      # React contexts (Auth, API Keys)
│   │   ├── lib/          # Utility functions and API clients
│   │   ├── landing/      # Landing page components
│   │   └── styles/       # CSS and theme files
│   ├── .env              # Frontend environment variables
│   └── package.json
│
├── backend/              # Django backend
│   ├── settings.py       # Django settings
│   └── urls.py          # URL routing
│
├── authentication/       # User authentication app
│   ├── models.py        # User models
│   ├── views.py         # Auth endpoints
│   └── middleware.py    # Auth middleware
│
├── block_manager/       # Core model builder app
│   ├── views/          # API endpoints
│   ├── services/       # Code generation services
│   └── serializers.py  # API serializers
│
├── .env                # Backend environment variables
├── requirements.txt    # Python dependencies
└── manage.py          # Django management script
```

## Development

### Running Tests
```bash
# Backend tests
cd project
python manage.py test

# Frontend tests
cd frontend
npm test
```

### Building for Production
```bash
# Frontend build
cd frontend
npm run build

# Backend (no build needed, deploy directly)
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Commit with clear, concise messages
5. Push and create a pull request

## License

All rights reserved.
