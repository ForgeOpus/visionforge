#!/usr/bin/env bash
# Build script for Render deployment

set -e  # Exit on any error

echo "=== Building VisionForge Frontend ==="
echo "Current directory: $(pwd)"

# Navigate to frontend directory
cd project/frontend

# Install Node.js dependencies
echo "Installing frontend dependencies..."
npm ci --prefer-offline --no-audit

# Build the React app with Vite
echo "Building React application..."
npm run build

# Verify build output
if [ -d "dist" ]; then
    echo "Build successful! Frontend built to project/frontend/dist/"
    ls -la dist/
else
    echo "ERROR: Build failed - dist directory not found"
    exit 1
fi

echo "=== Frontend build complete ==="
