"""Main FastAPI application for MemRingAI transcription service."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.api.transcription import router as transcription_router
from app.config import get_settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MemRingAI Transcription Service...")
    
    # Validate configuration
    settings = get_settings()
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not configured!")
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    logger.info(f"Service configured for host: {settings.host}, port: {settings.port}")
    logger.info(f"Temporary directory: {settings.temp_dir}")
    logger.info(f"Max file size: {settings.max_file_size / (1024 * 1024)}MB")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MemRingAI Transcription Service...")


# Create FastAPI application
app = FastAPI(
    title="MemRingAI Transcription Service",
    description="FastAPI service for transcription using OpenAI Whisper API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }
    )


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "MemRingAI Transcription Service",
        "version": "0.1.0",
        "description": "FastAPI service for transcription using OpenAI Whisper API",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe",
            "transcribe_url": "/transcribe/url",
            "supported_formats": "/supported-formats",
            "docs": "/docs"
        }
    }


# Include routers
app.include_router(transcription_router, prefix="/api/v1", tags=["transcription"])


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    ) 