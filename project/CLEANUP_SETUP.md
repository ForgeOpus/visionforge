# File Cleanup System Setup Guide

This guide explains how to set up automatic cleanup of uploaded files on Render's free tier.

## Overview

The file cleanup system prevents disk space issues by automatically deleting old uploaded files. Files are cleaned up immediately after processing, with a scheduled cleanup every 2 hours to catch any orphaned files.

## Configuration

### Environment Variables

Add to your Render environment variables:

```bash
CLEANUP_SECRET_TOKEN=your-random-secret-token-here
```

**Generate a secure token**:
```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

## How It Works

1. **Immediate Cleanup**: Files are deleted right after AI processing
2. **Scheduled Cleanup**: Runs every 2 hours via automated triggers
3. **Retention**: Files older than 2 hours are automatically deleted

## Setup Options

### Option 1: GitHub Actions (Recommended)

GitHub Actions is free and runs automatically every 2 hours.

**Setup Steps**:

1. **Add GitHub Secrets** to your repository (Settings → Secrets and variables → Actions):
   - `CLEANUP_ENDPOINT_URL`: `https://your-app.onrender.com/api/v1/maintenance/cleanup-files`
   - `CLEANUP_SECRET`: Your `CLEANUP_SECRET_TOKEN` value

2. **Workflow is Ready**: The workflow file already exists at `.github/workflows/cleanup_files.yml`

3. **Enable and Test**:
   - Push your code to GitHub
   - Go to Actions tab → "Cleanup Uploaded Files"
   - Click "Run workflow" to test manually
   - It will run automatically every 2 hours

### Option 2: External Cron Service (cron-job.org)

Free alternative if you don't want to use GitHub Actions.

**Setup Steps**:

1. **Sign up** at https://cron-job.org (free)

2. **Create a cron job**:
   - **URL**: `https://your-app.onrender.com/api/v1/maintenance/cleanup-files`
   - **Method**: `POST`
   - **Schedule**: Every 2 hours (`0 */2 * * *`)
   - **Request Headers**:
     ```
     Content-Type: application/json
     ```
   - **Request Body**:
     ```json
     {
       "secret": "your-cleanup-secret-token"
     }
     ```

3. **Test**: Click "Test execution" to verify it works

### Option 3: UptimeRobot Webhook

Use UptimeRobot's free HTTP(S) monitoring to trigger cleanup.

**Setup Steps**:

1. **Sign up** at https://uptimerobot.com (free)

2. **Create Monitor**:
   - Type: HTTP(S)
   - URL: `https://your-app.onrender.com/api/v1/maintenance/cleanup-files`
   - Monitoring Interval: 120 minutes (2 hours)

3. **Configure Request**:
   - Method: POST
   - Headers: `Content-Type: application/json`
   - Body: `{"secret": "your-cleanup-secret-token"}`

## Monitoring

### Check Upload Statistics

```bash
curl "https://your-app.onrender.com/api/v1/maintenance/upload-stats?secret=your-secret-token"
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

**Response**:
```json
{
  "success": true,
  "message": "File cleanup completed",
  "stats": {
    "deleted_count": 12,
    "deleted_size_mb": 3.45,
    "error_count": 0,
    "retention_hours": 2
  }
}
```

## Deployment Configuration

The `render.yaml` file is configured for easy deployment. For Render's free tier, the automated cleanup via GitHub Actions or external cron services is required.

## Security

- All endpoints are protected by `CLEANUP_SECRET_TOKEN`
- Unauthorized attempts are logged with IP addresses
- Never commit the secret token to git
- Rotate the token periodically for security

## Troubleshooting

### Cleanup Not Running

1. **Check logs**: View your service logs on Render dashboard
2. **Verify secret**: Ensure token matches in all locations
3. **Test manually**: Use curl command above to test
4. **Check GitHub Actions**: View workflow runs in Actions tab

### Disk Space Issues

If disk space is critically low, trigger manual cleanup immediately:

```bash
# Trigger cleanup now
curl -X POST https://your-app.onrender.com/api/v1/maintenance/cleanup-files \
  -H "Content-Type: application/json" \
  -d '{"secret": "your-cleanup-secret"}'
```

### GitHub Actions Not Running

1. Ensure Actions are enabled: Repo Settings → Actions → Allow all actions
2. Check workflow file syntax
3. Verify secrets are set correctly
4. Look for error messages in Actions tab

## Best Practices

1. **Monitor weekly**: Check upload stats to ensure cleanup is working
2. **Secure your token**: Use a strong, random token and keep it secret
3. **Test after deployment**: Trigger manual cleanup to verify it works
4. **Set up alerts**: Use UptimeRobot to alert if cleanup fails

## Configuration Options

You can adjust the retention period by setting in `settings.py`:

```python
UPLOAD_RETENTION_HOURS = 2  # Delete files older than 2 hours
```

For shorter retention (if disk space is limited):
```python
UPLOAD_RETENTION_HOURS = 1  # Delete files older than 1 hour
```

## Support

If you encounter issues:
1. Check Render logs for errors
2. Verify environment variables are set
3. Test endpoints manually with curl
4. Ensure secret token is correctly configured
