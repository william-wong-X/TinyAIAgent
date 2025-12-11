from typing import Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from .config import AppConfig
from .llm_client import create_llm
from ui.cli import chat_cli

store: Dict[str, ChatMessageHistory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个中文助手，请准确、简洁地回答。"), 
    MessagesPlaceholder(variable_name="history"), 
    ("human", "{input}")
])

def get_chat_chain(llm):
    chain = prompt | llm

    return RunnableWithMessageHistory(
        chain, 
        lambda session_id: get_history(session_id or "default"), 
        input_messages_key="input", 
        history_messages_key="history"
    )

def agent_run(config:AppConfig):
    llm = create_llm(config)

    chat_chain = get_chat_chain(llm)

    chat_cli(config, chat_chain)