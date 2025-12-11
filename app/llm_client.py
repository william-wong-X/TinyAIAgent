import os
from langchain_openai import ChatOpenAI

from .config import AppConfig

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-local-placeholder")

def create_llm(config: AppConfig):
    return ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], 
        base_url=config.llm.base_url, 
        model=config.llm.model, 
        temperature=config.llm.temperature, 
        streaming=config.llm.streaming,
        extra_body={
            "enable_thinking": config.llm.enable_thinking
        },
    )