"""
API Security, Magic-Byte Verification & Upload Sanitization
===========================================================
Protects medical imaging endpoints against path traversal attacks,
corrupted byte sequences, executable masquerading, and oversized payloads.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Tuple, Optional
from fastapi import HTTPException, status

MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 15))
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Common image and medical format magic bytes signatures
MAGIC_SIGNATURES = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff",
    "bmp": b"BM",
    "tiff_le": b"II*\x00",
    "tiff_be": b"MM\x00*",
}


def sanitize_filename(filename: Optional[str]) -> str:
    """
    Sanitize uploaded filename to prevent directory traversal and injection attacks.
    Returns a secure unique filename with UUID prefix.
    """
    if not filename:
        return f"{uuid.uuid4().hex}.png"

    # Strip directory components (path traversal prevention)
    clean_name = os.path.basename(filename)
    # Remove dangerous characters
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean_name)
    stem = Path(clean_name).stem[:50]
    suffix = Path(clean_name).suffix.lower()

    if suffix not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"]:
        suffix = ".png"

    return f"{uuid.uuid4().hex[:12]}_{stem}{suffix}"


def verify_magic_bytes(header: bytes) -> bool:
    """
    Check if file header matches known authentic image magic bytes.
    """
    if len(header) < 4:
        return False

    # Standard formats
    if header.startswith(MAGIC_SIGNATURES["png"]):
        return True
    if header.startswith(MAGIC_SIGNATURES["jpeg"]):
        return True
    if header.startswith(MAGIC_SIGNATURES["bmp"]):
        return True
    if header.startswith(MAGIC_SIGNATURES["tiff_le"]) or header.startswith(MAGIC_SIGNATURES["tiff_be"]):
        return True

    # DICOM (starts with 128-byte preamble followed by 'DICM')
    if len(header) >= 132 and header[128:132] == b"DICM":
        return True

    return False


def validate_image_payload(
    file_bytes: bytes,
    filename: Optional[str] = None,
    max_size_bytes: int = MAX_UPLOAD_BYTES,
) -> str:
    """
    Validate size, verify magic-bytes header, and return sanitized filename.
    Raises HTTPException 400 or 413 on security violations.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file payload submitted."
        )

    if len(file_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum permitted size of {max_size_bytes / (1024*1024):.1f} MB."
        )

    # Magic byte inspection
    header = file_bytes[:140]
    if not verify_magic_bytes(header):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content does not match authentic medical image signature (magic-byte check failed)."
        )

    return sanitize_filename(filename)
