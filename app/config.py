"""Configuration settings for the transcription service."""

import os
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_organization: Optional[str] = Field(None, env="OPENAI_ORGANIZATION")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(False, env="DEBUG")
    
    # File Upload Configuration
    max_file_size: int = Field(25 * 1024 * 1024, env="MAX_FILE_SIZE")  # 25MB default
    allowed_audio_extensions: list[str] = Field(
        [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"],
        env="ALLOWED_AUDIO_EXTENSIONS"
    )
    
    # Whisper Configuration
    default_model: str = Field("whisper-1", env="DEFAULT_WHISPER_MODEL")
    default_language: Optional[str] = Field(None, env="DEFAULT_LANGUAGE")
    default_response_format: str = Field("verbose_json", env="DEFAULT_RESPONSE_FORMAT")
    
    # Temporary file storage
    temp_dir: str = Field("./temp", env="TEMP_DIR")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings 