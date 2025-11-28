from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):

    lm_base_url: str 
    lm_api_key: str 
    model_name: str 

    emb_model: str 
    docs_dir: Path 
    index_dir: Path 
    md_knowledge_base: Path 

    spring_base_url: str 

    user_agent: str 


    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

@lru_cache
def get_settings() -> Settings:
    return Settings()