"""ASR Service Factory - Choose between different ASR providers."""

from enum import Enum
from typing import Optional

from app.config import get_settings
from app.models import TranscriptionRequest, TranscriptionResponse
from app.services.ASR.openai_service import OpenAIService
from app.services.ASR.volcengine_service import VolcengineService


class ASRProvider(str, Enum):
    """Available ASR providers."""
    OPENAI = "openai"
    VOLCENGINE = "volcengine"
    AUTO = "auto"  # Automatically choose based on configuration


class ASRServiceFactory:
    """Factory for creating ASR services."""
    
    @staticmethod
    def create_service(provider: ASRProvider = ASRProvider.AUTO) -> "BaseASRService":
        """
        Create an ASR service based on the specified provider.
        
        Args:
            provider: The ASR provider to use
            
        Returns:
            An ASR service instance
            
        Raises:
            ValueError: If no valid ASR service can be created
        """
        settings = get_settings()
        
        if provider == ASRProvider.AUTO:
            # Auto-detect based on available configuration
            if settings.volcengine_app_id and settings.volcengine_access_token:
                return VolcengineService()
            elif settings.openai_api_key:
                return OpenAIService()
            else:
                raise ValueError("No ASR service configuration found")
        
        elif provider == ASRProvider.OPENAI:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            return OpenAIService()
        
        elif provider == ASRProvider.VOLCENGINE:
            if not settings.volcengine_app_id or not settings.volcengine_access_token:
                raise ValueError("Volcengine credentials not configured")
            return VolcengineService()
        
        else:
            raise ValueError(f"Unknown ASR provider: {provider}")


class BaseASRService:
    """Base class for ASR services."""
    
    async def transcribe_audio(
        self,
        audio_file_path: str,
        request: TranscriptionRequest
    ) -> TranscriptionResponse:
        """Transcribe audio file."""
        raise NotImplementedError
    
    async def transcribe_from_url(
        self,
        audio_url: str,
        request: TranscriptionRequest
    ) -> TranscriptionResponse:
        """Transcribe audio from URL."""
        raise NotImplementedError
    
    async def validate_api_key(self) -> bool:
        """Validate API credentials."""
        raise NotImplementedError


# Convenience functions
def get_asr_service(provider: ASRProvider = ASRProvider.AUTO) -> BaseASRService:
    """Get an ASR service instance."""
    return ASRServiceFactory.create_service(provider)


def get_available_providers() -> list[ASRProvider]:
    """Get list of available ASR providers based on configuration."""
    settings = get_settings()
    providers = []
    
    if settings.openai_api_key:
        providers.append(ASRProvider.OPENAI)
    
    if settings.volcengine_app_id and settings.volcengine_access_token:
        providers.append(ASRProvider.VOLCENGINE)
    
    return providers 