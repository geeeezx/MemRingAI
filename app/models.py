"""Pydantic models for request and response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """Request model for transcription parameters."""
    
    model: Optional[str] = Field("whisper-1", description="Whisper model to use")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'es')")
    prompt: Optional[str] = Field(None, description="Optional prompt to guide transcription")
    response_format: Optional[str] = Field("verbose_json", description="Response format")
    temperature: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Temperature for sampling")
    timestamp_granularities: Optional[List[str]] = Field(
        ["word", "segment"], 
        description="Timestamp granularities"
    )


class TranscriptionSegment(BaseModel):
    """Model for transcription segments with timestamps."""
    
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class TranscriptionResponse(BaseModel):
    """Response model for transcription results."""
    
    task: str
    language: str
    duration: float
    segments: List[TranscriptionSegment]
    text: str
    status: str = "success"
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    
    status: str = "error"
    message: str
    error_code: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = "healthy"
    service: str = "MemRingAI Transcription Service"
    version: str = "0.1.0"
    openai_configured: bool 