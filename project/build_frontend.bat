@echo off
REM Build frontend and copy to Django for single-service deployment (Windows)

echo ================================
echo Building Frontend for Django...
echo ================================

cd frontend

echo Installing frontend dependencies...
call npm install
if errorlevel 1 goto error

echo Building React app...
call npm run build
if errorlevel 1 goto error

cd ..

echo Cleaning old build...
if exist frontend_build rmdir /s /q frontend_build
if exist staticfiles rmdir /s /q staticfiles

echo Copying build to Django...
mkdir frontend_build
xcopy /E /I /Y frontend\dist frontend_build

echo.
echo ✅ Frontend build complete!
echo.
echo Next steps:
echo 1. Collect static files: python manage.py collectstatic --noinput
echo 2. Run Django: python manage.py runserver
echo 3. Visit: http://localhost:8000
goto end

:error
echo.
echo ❌ Build failed!
exit /b 1

:end
