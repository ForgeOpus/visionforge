#!/usr/bin/env bash
# Build frontend and copy to Django for single-service deployment

set -e  # Exit on error

echo "================================"
echo "Building Frontend for Django..."
echo "================================"

# Navigate to frontend directory
cd frontend

echo "📦 Installing frontend dependencies..."
npm install

echo "🔨 Building React app..."
npm run build

echo "🗑️  Cleaning old build..."
cd ..
rm -rf frontend_build
rm -rf staticfiles

echo "📂 Copying build to Django..."
mkdir -p frontend_build
cp -r frontend/dist/* frontend_build/

echo "✅ Frontend build complete!"
echo ""
echo "Next steps:"
echo "1. Collect static files: python manage.py collectstatic --noinput"
echo "2. Run Django: python manage.py runserver"
echo "3. Visit: http://localhost:8000"
