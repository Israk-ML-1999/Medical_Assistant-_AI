"""
config.py
Central configuration for the Doctor Template Generation project.
Loads environment variables and sets up Vertex AI / Document AI credentials.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Doctor Template Generation API"
    DEBUG: bool = True

    # GCP common
    GCP_PROJECT_ID: str = Field(..., env="GCP_PROJECT_ID")
    GCP_LOCATION: str = Field(default="eu", env="GCP_LOCATION")

    # Credentials paths
    VERTEX_AI_CREDENTIALS_PATH: str = Field(
        default="secret/agent_platform_secret.json", env="VERTEX_AI_CREDENTIALS_PATH"
    )
    DOCUMENT_AI_CREDENTIALS_PATH: str = Field(
        default="secret/document_ai_secret.json", env="DOCUMENT_AI_CREDENTIALS_PATH"
    )

    # Document AI
    DOCAI_PROCESSOR_ID: str = Field(default="", env="DOCAI_PROCESSOR_ID")
    DOCAI_LOCATION: str = Field(default="us-central1", env="DOCAI_LOCATION")

    # Gemini models
    GEMINI_MODEL_NAME: str = Field(default="gemini-3.5-flash", env="GEMINI_MODEL_NAME")
    GEMINI_FLASH_CACHE_MODEL: str = Field(default="gemini-3.5-flash", env="GEMINI_FLASH_CACHE_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def get_vertex_ai_credentials_path() -> str:
    """Absolute path to the Vertex AI / Agent Platform service account JSON."""
    return os.path.abspath(settings.VERTEX_AI_CREDENTIALS_PATH)


def get_document_ai_credentials_path() -> str:
    """Absolute path to the Document AI service account JSON."""
    return os.path.abspath(settings.DOCUMENT_AI_CREDENTIALS_PATH)
