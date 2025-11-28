import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from .config import get_settings

settings = get_settings()

os.environ.setdefault("USER_AGENT", settings.user_agent)

llm = ChatOpenAI(
    model= settings.model_namem,
    base_url= settings.lm_base_url,
    api_key= settings.lm_api_key,
    temperature= 0.2,
    max_tokens= 1024,
)

embeddings= HuggingFaceEmbeddings(model_name= settings.emb_model)