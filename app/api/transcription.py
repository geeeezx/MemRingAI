"""Transcription API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models import (
    TranscriptionRequest,
    TranscriptionResponse,
    ErrorResponse,
    HealthResponse
)
from app.services.openai_service import OpenAIService
from app.services.file_service import FileService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_openai_service() -> OpenAIService:
    """Dependency to get OpenAI service instance."""
    return OpenAIService()


def get_file_service() -> FileService:
    """Dependency to get file service instance."""
    return FileService()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    openai_service: OpenAIService = Depends(get_openai_service)
) -> HealthResponse:
    """
    Health check endpoint to verify service status and OpenAI configuration.
    
    Returns:
        HealthResponse with service status and OpenAI configuration status
    """
    try:
        openai_configured = await openai_service.validate_api_key()
        return HealthResponse(openai_configured=openai_configured)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(openai_configured=False)


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: Optional[str] = Form("whisper-1", description="Whisper model to use"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es')"),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide transcription"),
    response_format: Optional[str] = Form("verbose_json", description="Response format"),
    temperature: Optional[float] = Form(0.0, description="Temperature for sampling"),
    openai_service: OpenAIService = Depends(get_openai_service),
    file_service: FileService = Depends(get_file_service)
) -> TranscriptionResponse:
    """
    Transcribe an uploaded audio file using OpenAI Whisper API.
    
    Args:
        file: Audio file to transcribe
        model: Whisper model to use
        language: Language code for transcription
        prompt: Optional prompt to guide transcription
        response_format: Response format (verbose_json, json, text, srt, vtt)
        temperature: Temperature for sampling (0.0 to 1.0)
        openai_service: OpenAI service instance
        file_service: File service instance
        
    Returns:
        TranscriptionResponse with transcription results
        
    Raises:
        HTTPException: If transcription fails or file is invalid
    """
    temp_file_path = None
    
    try:
        # Validate and save uploaded file
        temp_file_path = await file_service.save_uploaded_file(file)
        
        # Create transcription request
        request = TranscriptionRequest(
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature
        )
        
        # Perform transcription
        logger.info(f"Starting transcription for file: {file.filename}")
        result = await openai_service.transcribe_audio(temp_file_path, request)
        
        logger.info(f"Transcription completed successfully for file: {file.filename}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as they're already properly formatted
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path:
            await file_service.cleanup_temp_file(temp_file_path)


@router.post("/transcribe/url", response_model=TranscriptionResponse)
async def transcribe_from_url(
    url: str = Form(..., description="URL of the audio file to transcribe"),
    model: Optional[str] = Form("whisper-1", description="Whisper model to use"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es')"),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide transcription"),
    response_format: Optional[str] = Form("verbose_json", description="Response format"),
    temperature: Optional[float] = Form(0.0, description="Temperature for sampling"),
    openai_service: OpenAIService = Depends(get_openai_service),
    file_service: FileService = Depends(get_file_service)
) -> TranscriptionResponse:
    """
    Transcribe an audio file from URL using OpenAI Whisper API.
    
    Args:
        url: URL of the audio file to transcribe
        model: Whisper model to use
        language: Language code for transcription
        prompt: Optional prompt to guide transcription
        response_format: Response format (verbose_json, json, text, srt, vtt)
        temperature: Temperature for sampling (0.0 to 1.0)
        openai_service: OpenAI service instance
        file_service: File service instance
        
    Returns:
        TranscriptionResponse with transcription results
        
    Raises:
        HTTPException: If transcription fails or URL is invalid
    """
    temp_file_path = None
    
    try:
        import httpx
        
        from pathlib import Path
        
        # Download file from URL
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Save to temporary file
            temp_file_path = await file_service.save_uploaded_file_from_bytes(
                response.content, 
                f"audio_from_url{Path(url).suffix or '.mp3'}"
            )
        
        # Create transcription request
        request = TranscriptionRequest(
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature
        )
        
        # Perform transcription
        logger.info(f"Starting transcription for URL: {url}")
        result = await openai_service.transcribe_audio(temp_file_path, request)
        
        logger.info(f"Transcription completed successfully for URL: {url}")
        return result
        
    except httpx.HTTPError as e:
        logger.error(f"Failed to download file from URL: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download file from URL: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path:
            await file_service.cleanup_temp_file(temp_file_path)


@router.get("/supported-formats")
async def get_supported_formats() -> dict:
    """
    Get supported audio formats and configuration.
    
    Returns:
        Dictionary with supported formats and limits
    """
    settings = get_settings()
    return {
        "supported_formats": settings.allowed_audio_extensions,
        "max_file_size_mb": settings.max_file_size / (1024 * 1024),
        "default_model": settings.default_model,
        "default_response_format": settings.default_response_format
    } 