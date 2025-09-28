import os
from datetime import datetime
from utils.logging_utils import get_custom_logger
from fastapi import UploadFile
import shutil

logger = get_custom_logger("./helpers.logs")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def get_dob(id: str):
    try:
        base_year = 1700 + (int(id[0]) * 100)
        year = base_year + (int(id[1:3]))
        month = int(id[3:5])
        day = int(id[5:7])
        dob = datetime(year=year, month=month, day=day)
        return dob.date().isoformat()
    except Exception as e:
        return None


def allowed_file(filename: str) -> bool:
    """Check if file has an allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_files(file_paths: list):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file {file_path}: {str(e)}")


def save_file(file: UploadFile, path: str):
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
