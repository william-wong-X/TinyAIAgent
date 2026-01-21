import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.config import AppConfig

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

def create_embedding(config: AppConfig):
    return OpenAIEmbeddings(
        api_key=os.environ["OPENAI_API_KEY"], 
        base_url=config.embedding.base_url,
        model=config.embedding.model,
        check_embedding_ctx_length=False
    )