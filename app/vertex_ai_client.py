"""
app/vertex_ai_client.py

Shared helper to initialize Vertex AI (Gemini Enterprise Agent Platform) using the
Agent Platform service-account credentials via google-genai SDK.
"""

from functools import lru_cache

from google import genai
from google.oauth2 import service_account

from config import settings, get_vertex_ai_credentials_path


@lru_cache()
def get_client() -> genai.Client:
    """Initialize google-genai Client once using Agent Platform service account credentials."""
    creds_path = get_vertex_ai_credentials_path()
    credentials = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    
    client = genai.Client(
        vertexai=True,
        project="medai-test-project-260608",
        location="eu",
        credentials=credentials,
    )
    return client


def get_model_name(model_name: str = None) -> str:
    """Returns the model name to use, falling back to the configured default."""
    return model_name or settings.GEMINI_MODEL_NAME


def get_credentials():
    """Returns the loaded service account credentials if needed."""
    creds_path = get_vertex_ai_credentials_path()
    return service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
