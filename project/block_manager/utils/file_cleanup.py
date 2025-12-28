"""
Utilities for managing temporary file uploads.
"""
import os
import time
import logging
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)


def save_uploaded_file_temporarily(uploaded_file):
    """
    Save an uploaded file to the temporary directory with timestamp.

    Args:
        uploaded_file: Django UploadedFile object

    Returns:
        Path object of saved file
    """
    upload_dir = Path(getattr(settings, 'TEMP_UPLOAD_DIR', '/tmp/visionforge_uploads'))
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename to prevent path traversal attacks
    # Extract only the basename to remove any path components
    original_name = os.path.basename(uploaded_file.name)
    # Remove path separators that might have been missed (defense in depth)
    safe_name = original_name.replace('/', '_').replace('\\', '_')
    # Remove null bytes which can cause issues
    safe_name = safe_name.replace('\x00', '')
    
    timestamp = int(time.time())
    safe_filename = f"{timestamp}_{safe_name}"
    file_path = upload_dir / safe_filename
    
    # Verify the resolved path is within the upload directory (additional security check)
    resolved_path = file_path.resolve()
    if not str(resolved_path).startswith(str(upload_dir.resolve())):
        raise ValueError("Invalid file path detected - potential path traversal attack")

    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    logger.info(f"Saved temporary file: {safe_filename} ({uploaded_file.size} bytes)")
    return file_path


def cleanup_file_after_processing(file_path):
    """
    Immediately delete a file after processing.

    Args:
        file_path: Path object or string path to file
    """
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up processed file: {file_path.name}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")


def get_upload_directory_size():
    """
    Get the total size of all files in the upload directory.

    Returns:
        Total size in bytes
    """
    upload_dir = Path(getattr(settings, 'TEMP_UPLOAD_DIR', '/tmp/visionforge_uploads'))

    if not upload_dir.exists():
        return 0

    total_size = sum(f.stat().st_size for f in upload_dir.rglob('*') if f.is_file())
    return total_size
