import os
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

ENABLE_THINKING = True
USE_STREAMING = True

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-local-placeholder")

store: Dict[str, ChatMessageHistory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"], 
    base_url="http://localhost:8000/v1", 
    model="qwen3-8b", 
    temperature=0.7, 
    streaming=USE_STREAMING,
    extra_body={
        "enable_thinking": ENABLE_THINKING
    },
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个中文助手，请准确、简洁地回答。"), 
    MessagesPlaceholder(variable_name="history"), 
    ("human", "{input}")
])

chain = prompt | llm

chat_with_history = RunnableWithMessageHistory(
    chain, 
    lambda session_id: get_history(session_id or "default"), 
    input_messages_key="input", 
    history_messages_key="history"
)

def chat_cli(session_id: str = "user"):
    print(f"可以开始对话，输入 /exit 或 /quit 退出。")
    while True:
        user = input("user: ").strip()
        if not user:
            continue
        if user.lower() in {"/exit", "/quit"}:
            print("再见！")
            break

        if not USE_STREAMING:
            print("Qwen3: ", end="")
            ai_msg = chat_with_history.invoke(
                {"input": user},
                config={"configurable": {"session_id": session_id}},
            )

            answer_text = ai_msg.content or ""

            print(answer_text)
        else:
            print("Qwen3: ", end="", flush=True)

            got_any_chunk = False
            for chunk in chat_with_history.stream(
                {"input": user},
                config={"configurable": {"session_id": session_id}},
            ):
                got_any_chunk = True
                text_seg = chunk.content or ""
                if text_seg:
                    print(text_seg, end="", flush=True)

            if got_any_chunk:
                print()
            else:
                print("[无输出]")

        # 一轮对话结束自动换行
        print()
    
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")
    chat_cli()