from langchain_core.runnables import RunnableWithMessageHistory

from app.config import AppConfig

def chat_cli(config: AppConfig, chat_chain: RunnableWithMessageHistory, session_id: str="user"):
    print(f"可以开始对话，输入 /exit 或 /quit 退出。")
    while True:
        user = input("user: ").strip()
        if not user:
            continue
        if user.lower() in {"/exit", "/quit"}:
            print("再见！")
            break

        if not config.llm.streaming:
            print("Qwen3: ", end="")
            ai_message = chat_chain.invoke(
                {"input": user}, 
                config={"configurable": {"session_id": session_id}}
            )

            answer_text = ai_message.content or ""

            print(answer_text)
        else:
            print("Qwen3: ", end="", flush=True)
            got_any_chunk = False

            for chunk in chat_chain.stream(
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