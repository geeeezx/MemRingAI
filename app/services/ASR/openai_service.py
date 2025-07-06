"""OpenAI service for handling Whisper API calls asynchronously."""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import aiofiles
from openai import AsyncOpenAI
from openai.types.audio import Transcription

from app.config import get_settings
from app.models import TranscriptionRequest, TranscriptionResponse, TranscriptionSegment

logger = logging.getLogger(__name__)


class OpenAIService:
    """Service for handling OpenAI API interactions."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            organization=self.settings.openai_organization
        )
    
    async def transcribe_audio(
        self,
        audio_file_path: str,
        request: TranscriptionRequest
    ) -> TranscriptionResponse:
        """
        Transcribe audio file using OpenAI Whisper API asynchronously.
        
        Args:
            audio_file_path: Path to the audio file
            request: Transcription request parameters
            
        Returns:
            TranscriptionResponse with transcription results
        """
        try:
            logger.info(f"Starting transcription for file: {audio_file_path}")
            
            # Prepare parameters for OpenAI API
            params: Dict[str, Any] = {
                "model": request.model,
                "response_format": request.response_format,
                "temperature": request.temperature,
            }
            
            if request.language:
                params["language"] = request.language
            
            if request.prompt:
                params["prompt"] = request.prompt
            
            if request.timestamp_granularities:
                params["timestamp_granularities"] = request.timestamp_granularities
            
            # Open and transcribe the audio file
            async with aiofiles.open(audio_file_path, "rb") as audio_file:
                audio_data = await audio_file.read()
                
                transcription: Transcription = await self.client.audio.transcriptions.create(
                    file=("audio", audio_data, "audio/mpeg"),
                    **params
                )
            
            # Process the response
            if request.response_format == "verbose_json":
                return self._process_verbose_response(transcription)
            else:
                return self._process_simple_response(transcription)
                
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
    
    def _process_verbose_response(self, transcription: Transcription) -> TranscriptionResponse:
        """Process verbose JSON response from Whisper API."""
        if not hasattr(transcription, 'segments') or not transcription.segments:
            raise ValueError("Expected verbose JSON response with segments")
        
        segments = [
            TranscriptionSegment(
                id=segment.id,
                seek=segment.seek,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                tokens=segment.tokens,
                temperature=segment.temperature,
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob
            )
            for segment in transcription.segments
        ]
        
        return TranscriptionResponse(
            task=transcription.task,
            language=transcription.language,
            duration=transcription.duration,
            segments=segments,
            text=transcription.text
        )
    
    def _process_simple_response(self, transcription: Transcription) -> TranscriptionResponse:
        """Process simple text response from Whisper API."""
        return TranscriptionResponse(
            task="transcribe",
            language="unknown",
            duration=0.0,
            segments=[],
            text=transcription.text
        )
    
    async def validate_api_key(self) -> bool:
        """Validate that the OpenAI API key is working."""
        try:
            # Make a simple API call to test the key
            await self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI API key validation failed: {str(e)}")
            return False 