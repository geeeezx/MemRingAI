"""Configuration settings for the transcription service."""

import os
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_organization: Optional[str] = Field(None, env="OPENAI_ORGANIZATION")
    
    # Volcengine (DouBao) Configuration
    volcengine_app_id: Optional[str] = Field(None, env="VOLCENGINE_APP_ID")
    volcengine_access_token: Optional[str] = Field(None, env="VOLCENGINE_ACCESS_TOKEN")
    volcengine_resource_id: str = Field("volc.bigasr.auc_turbo", env="VOLCENGINE_RESOURCE_ID")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(False, env="DEBUG")
    
    # File Upload Configuration
    max_file_size: int = Field(25 * 1024 * 1024, env="MAX_FILE_SIZE")  # 25MB default
    allowed_audio_extensions: list[str] = Field(
        [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".opus"],
        env="ALLOWED_AUDIO_EXTENSIONS"
    )
    
    # Whisper Configuration
    default_model: str = Field("gpt-4o-transcribe", env="DEFAULT_WHISPER_MODEL")
    default_language: Optional[str] = Field(None, env="DEFAULT_LANGUAGE")
    default_response_format: str = Field("json", env="DEFAULT_RESPONSE_FORMAT")
    
    # Temporary file storage
    temp_dir: str = Field("./temp", env="TEMP_DIR")
    
    # Audio Acceleration Configuration
    enable_audio_acceleration: bool = Field(True, env="ENABLE_AUDIO_ACCELERATION")
    audio_acceleration_factor: float = Field(1.5, env="AUDIO_ACCELERATION_FACTOR")
    audio_min_segment_duration_for_acceleration: float = Field(1.0, env="AUDIO_MIN_SEGMENT_DURATION_FOR_ACCELERATION")
    audio_max_acceleration_factor: float = Field(3.0, env="AUDIO_MAX_ACCELERATION_FACTOR")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings 