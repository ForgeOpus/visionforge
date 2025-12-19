#!/usr/bin/env bash
# Build script for Render deployment

set -e  # Exit on any error

echo "=== Building VisionForge Frontend ==="
echo "Current directory: $(pwd)"

# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
echo "Installing frontend dependencies..."
npm ci --prefer-offline --no-audit

# Build the React app with Vite
echo "Building React application..."
npm run build

# Verify build output
if [ -d "dist" ]; then
    echo "Build successful! Frontend built to frontend/dist/"

    # Create root directory for files to be served at root URL
    echo "Organizing static files..."
    mkdir -p dist/root

    # Move root-level assets (logo, video, etc.) to root directory
    # These files will be served at / via WHITENOISE_ROOT
    for file in dist/*.{png,jpg,jpeg,gif,svg,mp4,webm,ico,txt,xml}; do
        if [ -f "$file" ] && [ "$(basename "$file")" != "index.html" ]; then
            mv "$file" dist/root/ 2>/dev/null || true
        fi
    done

    echo "Static files organized successfully"
    ls -la dist/
    ls -la dist/root/ 2>/dev/null || echo "No root-level assets found"
else
    echo "ERROR: Build failed - dist directory not found"
    exit 1
fi

echo "=== Frontend build complete ==="
