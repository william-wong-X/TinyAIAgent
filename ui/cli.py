import uuid
import readline

from config.config import AppConfig
from app.agent import Agent

def chat_cli(config: AppConfig, agent: Agent):
    thread_id = str(uuid.uuid4())
    print(f"可以开始对话，输入 /exit 或 /quit 退出。")
    while True:
        user_input = input("\033[32muser\033[0m: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit"}:
            print("再见！")
            break

        if not config.llm.streaming:
            print(f"{config.llm.model}: ", end="")
            ai_message = agent.invoke(user_input, thread_id=thread_id)
            answer_text = ai_message or ""

            print(answer_text)
        else:
            print(f"{config.llm.model}: ", end="", flush=True)
            got_any_chunk = False

            for chunk in agent.stream(user_input, thread_id=thread_id):
                got_any_chunk = True
                text_seg = chunk or ""
                if text_seg:
                    print(text_seg, end="", flush=True)
            
            if got_any_chunk:
                print()
            else:
                print("[无输出]")
        print()