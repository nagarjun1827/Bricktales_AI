"""
Application settings and configuration.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    GEMINI_API_KEY: str
    
    # Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "root"
    DB_NAME: str = "boq_database"

    model_config = ConfigDict(
        env_file=".env",
        extra="ignore"  # This will ignore extra fields from .env
    )


settings = Settings()