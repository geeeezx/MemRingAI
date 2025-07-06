"""Pydantic models for intent recognition requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class IntentEngine(str, Enum):
    """Available intent recognition engines."""
    RULE_BASED = "rule_based"
    ML_MODEL = "ml_model"
    LLM_BASED = "llm_based"


class IntentCategory(str, Enum):
    """Predefined intent categories."""
    # TODO: Define specific intent categories based on business needs
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    BOOKING = "booking"
    CANCELLATION = "cancellation"
    INFORMATION = "information"
    SUPPORT = "support"
    OTHER = "other"


class IntentResult(BaseModel):
    """Single intent recognition result."""
    
    intent: IntentCategory
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    engine: IntentEngine = Field(..., description="Engine used for recognition")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    
    # TODO: Add entity extraction results
    # entities: Optional[List[EntityResult]] = None


class IntentRequest(BaseModel):
    """Request model for intent recognition."""
    
    text: str = Field(..., description="Text to analyze for intent")
    engine: Optional[IntentEngine] = Field(IntentEngine.RULE_BASED, description="Preferred engine")
    include_details: bool = Field(False, description="Include detailed analysis")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
    # TODO: Add context parameters
    # context: Optional[Dict[str, Any]] = None
    # user_id: Optional[str] = None
    # session_id: Optional[str] = None


class IntentResponse(BaseModel):
    """Response model for intent recognition."""
    
    text: str = Field(..., description="Original text analyzed")
    results: List[IntentResult] = Field(..., description="Intent recognition results")
    primary_intent: Optional[IntentResult] = Field(None, description="Primary intent with highest confidence")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    status: str = Field("success", description="Response status")
    message: Optional[str] = Field(None, description="Additional message")
    
    # TODO: Add summary statistics
    # total_confidence: Optional[float] = None
    # alternative_intents: Optional[List[IntentResult]] = None


class BatchIntentRequest(BaseModel):
    """Request model for batch intent recognition."""
    
    texts: List[str] = Field(..., description="List of texts to analyze")
    engine: Optional[IntentEngine] = Field(IntentEngine.RULE_BASED, description="Preferred engine")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
    # TODO: Add batch processing options
    # parallel_processing: bool = Field(True, description="Enable parallel processing")
    # max_concurrent: int = Field(5, description="Maximum concurrent processing")


class BatchIntentResponse(BaseModel):
    """Response model for batch intent recognition."""
    
    results: List[IntentResponse] = Field(..., description="List of intent recognition results")
    total_processed: int = Field(..., description="Total number of texts processed")
    total_time: Optional[float] = Field(None, description="Total processing time")
    status: str = Field("success", description="Response status")
    message: Optional[str] = Field(None, description="Additional message")
    
    # TODO: Add batch statistics
    # success_rate: Optional[float] = None
    # average_confidence: Optional[float] = None


class CombinedTranscriptionIntentResponse(BaseModel):
    """Combined response for transcription + intent recognition."""
    
    # Transcription part
    transcription: Dict[str, Any] = Field(..., description="Transcription result")
    
    # Intent recognition part
    intent_analysis: IntentResponse = Field(..., description="Intent recognition result")
    
    # Combined metadata
    processing_time: Optional[float] = Field(None, description="Total processing time")
    status: str = Field("success", description="Combined response status")
    
    # TODO: Add correlation analysis
    # correlation_score: Optional[float] = None
    # confidence_alignment: Optional[float] = None


class IntentEngineStatus(BaseModel):
    """Status model for intent recognition engines."""
    
    engine: IntentEngine
    available: bool = Field(..., description="Engine availability")
    last_used: Optional[str] = Field(None, description="Last usage timestamp")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    
    # TODO: Add health metrics
    # accuracy: Optional[float] = None
    # average_response_time: Optional[float] = None
    # error_rate: Optional[float] = None


class IntentConfigResponse(BaseModel):
    """Response model for intent configuration."""
    
    available_engines: List[IntentEngine] = Field(..., description="Available engines")
    available_categories: List[IntentCategory] = Field(..., description="Available intent categories")
    default_engine: IntentEngine = Field(..., description="Default engine")
    confidence_threshold: float = Field(..., description="Default confidence threshold")
    
    # TODO: Add configuration details
    # engine_configs: Optional[Dict[str, Any]] = None
    # category_descriptions: Optional[Dict[str, str]] = None 