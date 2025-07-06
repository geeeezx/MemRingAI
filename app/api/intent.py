"""Intent recognition API endpoints."""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import JSONResponse

from app.models.intent_models import (
    IntentRequest,
    IntentResponse,
    BatchIntentRequest,
    BatchIntentResponse,
    IntentConfigResponse,
    IntentCategory,
    IntentEngine
)
from app.services.intent_service import IntentService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_intent_service() -> IntentService:
    """Dependency to get intent service instance."""
    return IntentService()


@router.post("/analyze", response_model=IntentResponse)
async def analyze_intent(
    request: IntentRequest = Body(..., description="Intent analysis request"),
    intent_service: IntentService = Depends(get_intent_service)
) -> IntentResponse:
    """
    Analyze text for intent recognition.
    
    Args:
        request: Intent recognition request containing text and parameters
        intent_service: Intent service instance
        
    Returns:
        IntentResponse with recognition results
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        logger.info(f"Intent analysis requested for text: {request.text[:50]}...")
        
        # Validate request
        if not request.text or not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text is required and cannot be empty"
            )
        
        # Perform intent analysis
        result = await intent_service.analyze_intent(request)
        
        if result.status == "error":
            raise HTTPException(
                status_code=500,
                detail=result.message or "Intent analysis failed"
            )
        
        logger.info(f"Intent analysis completed successfully. Primary intent: {result.primary_intent.intent if result.primary_intent else 'None'}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intent analysis endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Intent analysis failed: {str(e)}"
        )


@router.post("/analyze/batch", response_model=BatchIntentResponse)
async def analyze_batch_intent(
    request: BatchIntentRequest = Body(..., description="Batch intent analysis request"),
    intent_service: IntentService = Depends(get_intent_service)
) -> BatchIntentResponse:
    """
    Analyze multiple texts for intent recognition.
    
    Args:
        request: Batch intent recognition request containing texts and parameters
        intent_service: Intent service instance
        
    Returns:
        BatchIntentResponse with all recognition results
        
    Raises:
        HTTPException: If batch analysis fails
    """
    try:
        logger.info(f"Batch intent analysis requested for {len(request.texts)} texts")
        
        # Validate request
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="Texts list is required and cannot be empty"
            )
        
        # Limit batch size to prevent abuse
        MAX_BATCH_SIZE = 100  # TODO: Make this configurable
        if len(request.texts) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size cannot exceed {MAX_BATCH_SIZE} texts"
            )
        
        # Perform batch intent analysis
        result = await intent_service.analyze_batch(request)
        
        if result.status == "error":
            raise HTTPException(
                status_code=500,
                detail=result.message or "Batch intent analysis failed"
            )
        
        logger.info(f"Batch intent analysis completed successfully. Processed {result.total_processed} texts")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch intent analysis endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch intent analysis failed: {str(e)}"
        )


@router.get("/config", response_model=IntentConfigResponse)
async def get_intent_config(
    intent_service: IntentService = Depends(get_intent_service)
) -> IntentConfigResponse:
    """
    Get intent recognition configuration.
    
    Args:
        intent_service: Intent service instance
        
    Returns:
        IntentConfigResponse with configuration details
    """
    try:
        logger.info("Intent configuration requested")
        
        config = IntentConfigResponse(
            available_engines=intent_service.get_available_engines(),
            available_categories=intent_service.get_available_categories(),
            default_engine=intent_service.get_default_engine(),
            confidence_threshold=intent_service.get_confidence_threshold()
        )
        
        logger.info("Intent configuration retrieved successfully")
        return config
        
    except Exception as e:
        logger.error(f"Intent configuration endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get intent configuration: {str(e)}"
        )


@router.get("/categories", response_model=List[str])
async def get_intent_categories() -> List[str]:
    """
    Get list of available intent categories.
    
    Returns:
        List of available intent categories
    """
    try:
        logger.info("Intent categories requested")
        
        categories = [category.value for category in IntentCategory]
        
        logger.info(f"Returned {len(categories)} intent categories")
        return categories
        
    except Exception as e:
        logger.error(f"Intent categories endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get intent categories: {str(e)}"
        )


@router.get("/engines", response_model=List[str])
async def get_intent_engines(
    intent_service: IntentService = Depends(get_intent_service)
) -> List[str]:
    """
    Get list of available intent recognition engines.
    
    Args:
        intent_service: Intent service instance
        
    Returns:
        List of available intent recognition engines
    """
    try:
        logger.info("Intent engines requested")
        
        engines = [engine.value for engine in intent_service.get_available_engines()]
        
        logger.info(f"Returned {len(engines)} intent engines")
        return engines
        
    except Exception as e:
        logger.error(f"Intent engines endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get intent engines: {str(e)}"
        )


@router.get("/health")
async def intent_health_check() -> dict:
    """
    Health check endpoint for intent recognition service.
    
    Returns:
        Health status of intent recognition service
    """
    try:
        logger.info("Intent service health check requested")
        
        # Basic health check
        # TODO: Add more comprehensive health checks
        # - Check if models are loaded
        # - Check if rule files exist
        # - Check memory usage
        # - Check response times
        
        return {
            "status": "healthy",
            "service": "Intent Recognition Service",
            "version": "0.1.0",
            "engines_available": [engine.value for engine in IntentEngine],
            "categories_available": [category.value for category in IntentCategory]
        }
        
    except Exception as e:
        logger.error(f"Intent health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Intent service health check failed: {str(e)}"
        )


# TODO: Add more endpoints
# @router.post("/rules/update")
# async def update_intent_rules():
#     """Update intent recognition rules."""
#     pass

# @router.get("/stats")
# async def get_intent_stats():
#     """Get intent recognition statistics."""
#     pass

# @router.post("/train")
# async def train_intent_model():
#     """Train/retrain intent recognition model."""
#     pass 