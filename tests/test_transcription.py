"""Tests for the transcription service."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.models import TranscriptionResponse, TranscriptionSegment

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MemRingAI Transcription Service"
    assert data["version"] == "0.1.0"


def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "MemRingAI Transcription Service"


def test_supported_formats_endpoint():
    """Test the supported formats endpoint."""
    response = client.get("/api/v1/supported-formats")
    assert response.status_code == 200
    data = response.json()
    assert "supported_formats" in data
    assert "max_file_size_mb" in data
    assert "default_model" in data


@pytest.mark.asyncio
async def test_transcription_request_model():
    """Test the TranscriptionRequest model."""
    from app.models import TranscriptionRequest
    
    request = TranscriptionRequest(
        model="whisper-1",
        language="en",
        prompt="Test prompt",
        response_format="verbose_json",
        temperature=0.0
    )
    
    assert request.model == "whisper-1"
    assert request.language == "en"
    assert request.prompt == "Test prompt"
    assert request.response_format == "verbose_json"
    assert request.temperature == 0.0


@pytest.mark.asyncio
async def test_transcription_response_model():
    """Test the TranscriptionResponse model."""
    from app.models import TranscriptionResponse, TranscriptionSegment
    
    segment = TranscriptionSegment(
        id=0,
        seek=0,
        start=0.0,
        end=2.0,
        text="Hello world",
        tokens=[1, 2, 3],
        temperature=0.0,
        avg_logprob=-0.5,
        compression_ratio=1.0,
        no_speech_prob=0.1
    )
    
    response = TranscriptionResponse(
        task="transcribe",
        language="en",
        duration=2.0,
        segments=[segment],
        text="Hello world"
    )
    
    assert response.task == "transcribe"
    assert response.language == "en"
    assert response.duration == 2.0
    assert len(response.segments) == 1
    assert response.text == "Hello world"


def test_invalid_file_type():
    """Test rejection of invalid file types."""
    with open("test.txt", "w") as f:
        f.write("This is not an audio file")
    
    try:
        with open("test.txt", "rb") as f:
            response = client.post(
                "/api/v1/transcribe",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid file type" in data["detail"]
    finally:
        import os
        if os.path.exists("test.txt"):
            os.remove("test.txt")


if __name__ == "__main__":
    pytest.main([__file__]) 