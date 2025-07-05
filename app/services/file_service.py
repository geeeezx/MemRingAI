"""File service for handling audio file uploads and validation."""

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import UploadFile, HTTPException

from app.config import get_settings

logger = logging.getLogger(__name__)


class FileService:
    """Service for handling file operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self._ensure_temp_dir()
    
    def _ensure_temp_dir(self) -> None:
        """Ensure the temporary directory exists."""
        temp_path = Path(self.settings.temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Temporary directory ensured: {temp_path}")
    
    def validate_file_extension(self, filename: str) -> bool:
        """
        Validate if the file extension is allowed.
        
        Args:
            filename: Name of the uploaded file
            
        Returns:
            True if extension is allowed, False otherwise
        """
        if not filename:
            return False
        
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.settings.allowed_audio_extensions
    
    def validate_file_size(self, file_size: int) -> bool:
        """
        Validate if the file size is within limits.
        
        Args:
            file_size: Size of the file in bytes
            
        Returns:
            True if size is acceptable, False otherwise
        """
        return file_size <= self.settings.max_file_size
    
    async def save_uploaded_file(self, file: UploadFile) -> str:
        """
        Save uploaded file to temporary directory.
        
        Args:
            file: Uploaded file from FastAPI
            
        Returns:
            Path to the saved file
            
        Raises:
            HTTPException: If file validation fails
        """
        # Validate file extension
        if not self.validate_file_extension(file.filename):
            allowed_exts = ", ".join(self.settings.allowed_audio_extensions)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed extensions: {allowed_exts}"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix.lower()
        temp_filename = f"{file_id}{file_ext}"
        temp_file_path = Path(self.settings.temp_dir) / temp_filename
        
        try:
            # Save file
            async with aiofiles.open(temp_file_path, "wb") as temp_file:
                content = await file.read()
                
                # Validate file size
                if not self.validate_file_size(len(content)):
                    max_size_mb = self.settings.max_file_size / (1024 * 1024)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {max_size_mb}MB"
                    )
                
                await temp_file.write(content)
            
            logger.info(f"File saved successfully: {temp_file_path}")
            return str(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to save uploaded file"
            )
    
    async def cleanup_temp_file(self, file_path: str) -> None:
        """
        Clean up temporary file.
        
        Args:
            file_path: Path to the file to delete
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {file_path}: {str(e)}")
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        try:
            stat = os.stat(file_path)
            return {
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "exists": True
            }
        except OSError:
            return {"exists": False}
    
    async def save_uploaded_file_from_bytes(self, content: bytes, filename: str) -> str:
        """
        Save file content from bytes to temporary directory.
        
        Args:
            content: File content as bytes
            filename: Name of the file
            
        Returns:
            Path to the saved file
            
        Raises:
            HTTPException: If file validation fails
        """
        # Validate file extension
        if not self.validate_file_extension(filename):
            allowed_exts = ", ".join(self.settings.allowed_audio_extensions)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed extensions: {allowed_exts}"
            )
        
        # Validate file size
        if not self.validate_file_size(len(content)):
            max_size_mb = self.settings.max_file_size / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_size_mb}MB"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = Path(filename).suffix.lower()
        temp_filename = f"{file_id}{file_ext}"
        temp_file_path = Path(self.settings.temp_dir) / temp_filename
        
        try:
            # Save file
            async with aiofiles.open(temp_file_path, "wb") as temp_file:
                await temp_file.write(content)
            
            logger.info(f"File saved successfully: {temp_file_path}")
            return str(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to save file"
            ) 