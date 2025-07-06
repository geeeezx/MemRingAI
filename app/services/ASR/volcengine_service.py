"""Volcengine (DouBao) ASR service for handling speech recognition API calls asynchronously."""

import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import aiofiles
import httpx
from pydantic import BaseModel

from app.config import get_settings
from app.models import TranscriptionRequest, TranscriptionResponse, TranscriptionSegment

logger = logging.getLogger(__name__)


class VolcengineConfig(BaseModel):
    """Volcengine API configuration."""
    app_id: str
    access_token: str
    resource_id: str
    base_url: str = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"


class VolcengineService:
    """Service for handling Volcengine (DouBao) ASR API interactions."""
    
    def __init__(self):
        self.settings = get_settings()
        self.config = VolcengineConfig(
            app_id=self.settings.volcengine_app_id,
            access_token=self.settings.volcengine_access_token,
            resource_id=self.settings.volcengine_resource_id
        )
        self.client = httpx.AsyncClient(timeout=300.0)  # 5分钟超时
    
    async def transcribe_audio(
        self,
        audio_file_path: str,
        request: TranscriptionRequest
    ) -> TranscriptionResponse:
        """
        Transcribe audio file using Volcengine ASR API asynchronously.
        
        Args:
            audio_file_path: Path to the audio file
            request: Transcription request parameters
            
        Returns:
            TranscriptionResponse with transcription results
        """
        try:
            logger.info(f"Starting Volcengine transcription for file: {audio_file_path}")
            
            # Read and encode audio file
            async with aiofiles.open(audio_file_path, "rb") as audio_file:
                audio_data = await audio_file.read()
                base64_data = base64.b64encode(audio_data).decode('utf-8')
            
            # Prepare request payload
            payload = {
                "user": {
                    "uid": self.config.app_id
                },
                "audio": {
                    "data": base64_data
                },
                "request": {
                    "model_name": "bigmodel"
                }
            }
            
            # Add optional parameters if provided
            if request.language:
                payload["request"]["language"] = request.language
            
            if request.prompt:
                payload["request"]["prompt"] = request.prompt
            
            # Prepare headers
            headers = {
                "X-Api-App-Key": self.config.app_id,
                "X-Api-Access-Key": self.config.access_token,
                "X-Api-Resource-Id": self.config.resource_id,
                "X-Api-Request-Id": str(uuid.uuid4()),
                "X-Api-Sequence": "-1",
                "Content-Type": "application/json"
            }
            
            # Make API request
            response = await self.client.post(
                self.config.base_url,
                json=payload,
                headers=headers
            )
            
            # Check response status
            status_code = response.headers.get('X-Api-Status-Code')
            if status_code != '20000000':
                error_msg = response.headers.get('X-Api-Message', 'Unknown error')
                raise Exception(f"Volcengine API error: {status_code} - {error_msg}")
            
            # Parse response
            result = response.json()
            return self._process_response(result, request)
                
        except Exception as e:
            logger.error(f"Error during Volcengine transcription: {str(e)}")
            raise
    
    def _process_response(self, result: Dict[str, Any], request: TranscriptionRequest) -> TranscriptionResponse:
        """Process Volcengine API response."""
        try:
            audio_info = result.get("audio_info", {})
            result_data = result.get("result", {})
            
            # Extract basic information
            duration = audio_info.get("duration", 0) / 1000.0  # Convert from ms to seconds
            text = result_data.get("text", "")
            language = request.language or "unknown"
            
            # Process utterances/segments
            segments = []
            utterances = result_data.get("utterances", [])
            
            for i, utterance in enumerate(utterances):
                # Create segment from utterance
                segment = TranscriptionSegment(
                    id=i,
                    seek=0,
                    start=utterance.get("start_time", 0) / 1000.0,  # Convert from ms to seconds
                    end=utterance.get("end_time", 0) / 1000.0,  # Convert from ms to seconds
                    text=utterance.get("text", ""),
                    tokens=[],  # Volcengine doesn't provide tokens
                    temperature=0.0,
                    avg_logprob=0.0,
                    compression_ratio=1.0,
                    no_speech_prob=0.0
                )
                segments.append(segment)
            
            return TranscriptionResponse(
                task="transcribe",
                language=language,
                duration=duration,
                segments=segments,
                text=text
            )
            
        except Exception as e:
            logger.error(f"Error processing Volcengine response: {str(e)}")
            raise
    
    async def transcribe_from_url(
        self,
        audio_url: str,
        request: TranscriptionRequest
    ) -> TranscriptionResponse:
        """
        Transcribe audio from URL using Volcengine ASR API.
        
        Args:
            audio_url: URL of the audio file
            request: Transcription request parameters
            
        Returns:
            TranscriptionResponse with transcription results
        """
        try:
            logger.info(f"Starting Volcengine transcription from URL: {audio_url}")
            
            # Prepare request payload with URL
            payload = {
                "user": {
                    "uid": self.config.app_id
                },
                "audio": {
                    "url": audio_url
                },
                "request": {
                    "model_name": "bigmodel"
                }
            }
            
            # Add optional parameters if provided
            if request.language:
                payload["request"]["language"] = request.language
            
            if request.prompt:
                payload["request"]["prompt"] = request.prompt
            
            # Prepare headers
            headers = {
                "X-Api-App-Key": self.config.app_id,
                "X-Api-Access-Key": self.config.access_token,
                "X-Api-Resource-Id": self.config.resource_id,
                "X-Api-Request-Id": str(uuid.uuid4()),
                "X-Api-Sequence": "-1",
                "Content-Type": "application/json"
            }
            
            # Make API request
            response = await self.client.post(
                self.config.base_url,
                json=payload,
                headers=headers
            )
            
            # Check response status
            status_code = response.headers.get('X-Api-Status-Code')
            if status_code != '20000000':
                error_msg = response.headers.get('X-Api-Message', 'Unknown error')
                raise Exception(f"Volcengine API error: {status_code} - {error_msg}")
            
            # Parse response
            result = response.json()
            return self._process_response(result, request)
                
        except Exception as e:
            logger.error(f"Error during Volcengine URL transcription: {str(e)}")
            raise
    
    async def validate_api_key(self) -> bool:
        """Validate that the Volcengine API credentials are working."""
        try:
            # Make a simple test request with minimal data
            test_payload = {
                "user": {"uid": self.config.app_id},
                "audio": {"data": ""},  # Empty data for validation
                "request": {"model_name": "bigmodel"}
            }
            
            headers = {
                "X-Api-App-Key": self.config.app_id,
                "X-Api-Access-Key": self.config.access_token,
                "X-Api-Resource-Id": self.config.resource_id,
                "X-Api-Request-Id": str(uuid.uuid4()),
                "X-Api-Sequence": "-1",
                "Content-Type": "application/json"
            }
            
            response = await self.client.post(
                self.config.base_url,
                json=test_payload,
                headers=headers
            )
            
            # Even if it fails due to empty audio, we can validate credentials
            status_code = response.headers.get('X-Api-Status-Code')
            return status_code in ['20000000', '45000002']  # Success or empty audio error
            
        except Exception as e:
            logger.error(f"Volcengine API validation failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose() 