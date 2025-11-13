import os
from dotenv import load_dotenv

def load_env():
    env_type = os.getenv("ENV", "dev") 
    env_file_path = f".env.{env_type}" 
    load_dotenv(dotenv_path=env_file_path)
