#!/usr/bin/env python3
"""Script to run the MemRingAI transcription service."""

import uvicorn
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    
    print("🚀 Starting MemRingAI Transcription Service...")
    print(f"📍 Host: {settings.host}")
    print(f"🔌 Port: {settings.port}")
    print(f"🐛 Debug: {settings.debug}")
    print(f"📁 Temp Directory: {settings.temp_dir}")
    print(f"📏 Max File Size: {settings.max_file_size / (1024 * 1024)}MB")
    print()
    print("📚 API Documentation will be available at:")
    print(f"   Swagger UI: http://{settings.host}:{settings.port}/docs")
    print(f"   ReDoc: http://{settings.host}:{settings.port}/redoc")
    print()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    ) 
