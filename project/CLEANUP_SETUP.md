# File Cleanup System Setup Guide

This guide explains how to set up automatic cleanup of uploaded files on Render's free tier.

## Overview

The file cleanup system prevents disk space issues by automatically deleting old uploaded files. Since Render's free tier doesn't support cron jobs, we provide multiple options for scheduling cleanup.

## Components

1. **Django Management Command**: `python manage.py cleanup_uploaded_files`
2. **Maintenance API Endpoints**: Remote triggers for cleanup
3. **Automated Scheduling**: GitHub Actions or external cron service

## Configuration

### 1. Environment Variables

Add to your `.env` file or Render environment variables:

```bash
# File cleanup settings
CLEANUP_SECRET_TOKEN=your-random-secret-token-here  # Generate: python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

### 2. Manual Cleanup (SSH into Render)

```bash
# SSH into your Render instance
render ssh <your-service-name>

# Run cleanup
cd project
python manage.py cleanup_uploaded_files

# Dry run (see what would be deleted)
python manage.py cleanup_uploaded_files --dry-run

# Custom retention period
python manage.py cleanup_uploaded_files --retention-hours 1
```

## Automated Cleanup Options

### Option 1: GitHub Actions (Recommended for Free Tier)

1. **Add GitHub Secrets** in your repository:
   - `CLEANUP_ENDPOINT_URL`: `https://your-app.onrender.com/api/v1/maintenance/cleanup-files`
   - `CLEANUP_SECRET`: Your `CLEANUP_SECRET_TOKEN` value

2. **Enable GitHub Actions**:
   - The workflow file is already created at `.github/workflows/cleanup_files.yml`
   - It runs every 2 hours automatically
   - You can also trigger it manually from the Actions tab

3. **Manual Trigger**:
   - Go to your GitHub repo → Actions tab
   - Select "Cleanup Uploaded Files"
   - Click "Run workflow"

### Option 2: External Cron Service (cron-job.org)

1. **Sign up** at https://cron-job.org (free)

2. **Create a cron job**:
   - URL: `https://your-app.onrender.com/api/v1/maintenance/cleanup-files`
   - Method: `POST`
   - Schedule: Every 2 hours (`0 */2 * * *`)
   - Request Body:
     ```json
     {
       "secret": "your-cleanup-secret-token"
     }
     ```
   - Headers:
     ```
     Content-Type: application/json
     ```

### Option 3: Render Paid Plan (Native Cron Jobs)

If you upgrade to a paid Render plan, you can use native cron jobs:

1. **Update `render.yaml`**:
   ```yaml
   - type: cron
     name: cleanup-files
     schedule: "0 */2 * * *"  # Every 2 hours
     buildCommand: |
       cd project
       pip install -r requirements.txt
     command: |
       cd project
       python manage.py cleanup_uploaded_files
   ```

## Monitoring

### Check Upload Statistics

**Via API** (protected endpoint):
```bash
curl "https://your-app.onrender.com/api/v1/maintenance/upload-stats?secret=your-cleanup-secret"
```

**Response**:
```json
{
  "success": true,
  "stats": {
    "total_size_mb": 15.32,
    "file_count": 47,
    "oldest_file_age_hours": 1.5,
    "retention_hours": 2,
    "upload_directory": "/app/temp_uploads"
  }
}
```

### Trigger Manual Cleanup

```bash
curl -X POST https://your-app.onrender.com/api/v1/maintenance/cleanup-files \
  -H "Content-Type: application/json" \
  -d '{"secret": "your-cleanup-secret"}'
```

## How It Works

1. **File Upload**: When users upload files to chat, they're saved to `temp_uploads/` with a timestamp
2. **Immediate Cleanup**: Files are deleted immediately after AI processing
3. **Scheduled Cleanup**: Runs every 2 hours to catch any orphaned files
4. **Retention**: Files older than 2 hours are automatically deleted

## Retention Period

Default: **2 hours**

To change, update `UPLOAD_RETENTION_HOURS` in `settings.py` or add to environment variables.

## Security

- Cleanup endpoints are protected by `CLEANUP_SECRET_TOKEN`
- Unauthorized attempts are logged
- Token should be kept secret and never committed to git

## Troubleshooting

### Files Not Being Deleted

1. Check GitHub Actions logs for errors
2. Verify `CLEANUP_SECRET_TOKEN` matches in all locations
3. Check Render logs: `render logs <service-name>`
4. Run manual cleanup to test

### Disk Space Issues

```bash
# Check current usage
python manage.py cleanup_uploaded_files --dry-run

# Force cleanup with shorter retention
python manage.py cleanup_uploaded_files --retention-hours 0
```

### GitHub Actions Not Running

1. Ensure Actions are enabled in your repository settings
2. Check workflow file permissions
3. Verify secrets are correctly configured

## Best Practices

1. **Monitor regularly**: Check upload stats weekly
2. **Adjust retention**: Lower if disk space is limited
3. **Test cleanup**: Run dry-run before production
4. **Keep secrets secure**: Rotate `CLEANUP_SECRET_TOKEN` periodically
5. **Check logs**: Review cleanup output for errors
