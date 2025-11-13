import logging
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

class Settings(BaseSettings):
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
        extra="ignore"
    )

settings = Settings()