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


class IdeaReport(BaseModel):
    """Structure for the generated report from report agent."""
    idea_summary: str
    market_analysis: str
    technical_feasibility: str
    implementation_steps: List[str]
    potential_challenges: List[str]
    success_factors: List[str]
    estimated_timeline: str
    next_actions: List[str]


class TranscriptionReportRequest(BaseModel):
    """Request model for combined transcription + report generation."""
    
    # Transcription parameters
    model: Optional[str] = Field("whisper-1", description="Whisper model to use")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'es')")
    prompt: Optional[str] = Field(None, description="Optional prompt to guide transcription")
    response_format: Optional[str] = Field("verbose_json", description="Response format")
    temperature: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Temperature for sampling")
    timestamp_granularities: Optional[List[str]] = Field(
        ["word", "segment"], 
        description="Timestamp granularities"
    )
    provider: Optional[str] = Field("auto", description="ASR provider (openai, volcengine, auto)")
    
    # Report generation parameters
    generate_report: bool = Field(True, description="Whether to generate a business report from transcription")
    report_focus: Optional[str] = Field(None, description="Optional focus area for the report (e.g., 'technical', 'market', 'implementation')")


class TranscriptionReportResponse(BaseModel):
    """Response model for combined transcription + report results."""
    
    # Transcription results
    transcription: TranscriptionResponse
    
    # Report results
    report: Optional[IdeaReport] = None
    report_generation_success: bool = False
    report_error: Optional[str] = None
    
    # Combined metadata
    total_tokens_used: Optional[int] = 0
    processing_time_seconds: Optional[float] = None
    status: str = "success"
    message: Optional[str] = None 