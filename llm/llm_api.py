import time
import argparse
import torch
import threading
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import uvicorn

from utils.utils import get_dir
from config.config import load_config

# ====================== Config ======================
parser = argparse.ArgumentParser()    
parser.add_argument('-c', '--config', type=str, default="./config/config.yaml", help="Config path")
args = parser.parse_args()
config = load_config(args.config)

# ====================== LLM ======================
class ChatLLM:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        print(f"正在从 {self.model_path} 加载模型...")
        try:
            # load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                local_files_only=True, 
                trust_remote_code=True
            )
            # load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16, 
                device_map=self.device, 
                local_files_only=True, 
                trust_remote_code=True
            )
            self.model.eval()
            print(f"模型加载完成，使用的设备: {self.device}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    @torch.no_grad()
    def generate(
        self, 
        messages: List[dict], 
        max_new_tokens: int = 32765,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool = True,
    ) -> str:
        if not isinstance(messages, list):
            messages = [messages]

        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=enable_thinking
        )
        model_input = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_input, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature if temperature > 0 else None, 
            top_p=top_p, 
            do_sample=temperature > 0
        )
        output_ids = generated_ids[0][len(model_input.input_ids[0]):].tolist()

        full_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        return full_text.strip("\n")
    
    def stream_generate(
            self,
            messages: List[dict],
            max_new_tokens: int = 32765,
            temperature: float = 0.7,
            top_p: float = 0.9,
            enable_thinking: bool = True,
    ):
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=enable_thinking
        )
        model_input = self.tokenizer([text], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            decode_kwargs={"skip_special_tokens": False},
        )

        generate_kwargs = dict(
            **model_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            do_sample=temperature > 0,
            streamer=streamer,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        for new_text in streamer:
            if not new_text:
                continue
            yield new_text
    
# ====================== API Data Model (OpenAI) ======================
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = config.llm.model
    messages: List[Message]
    max_tokens: Optional[int] = 32765
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    enable_thinking: Optional[bool] = True
    stream: Optional[bool] = True

class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = config.llm.model
    choices: List[Choice]
    usage: Usage = Usage()

# ====================== FastAPI App ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = get_dir(config.llm.model_path)
    app.state.llm = ChatLLM(model_path)
    yield

app = FastAPI(
    title="LLM API",
    description="OpenAI-style LLM API",
    lifespan=lifespan,
)

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": config.llm.model, "object": "model"}]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: Request, body: ChatCompletionRequest):
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        raise HTTPException(status_code=503, detail="Model Unload")
    
    messages=[{"role": m.role, "content": m.content} for m in body.messages]

    # ============ Stream Model ============
    if body.stream:
        def event_generator():
            chat_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())
            first_chunk = True

            for text in llm.stream_generate(
                messages=messages,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                enable_thinking=body.enable_thinking,
            ):

                if not text:
                    continue

                delta = {"content": text}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False

                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ============ Non-stream Model ============
    content = llm.generate(
        messages=messages,
        max_new_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        enable_thinking=body.enable_thinking,
    )

    return ChatCompletionResponse(
        model=body.model,
        choices=[
            Choice(
                message=Message(role="assistant", content=content)
            )
        ]
    )

@app.get("/health")
async def health_check(request: Request):
    return {"status": "ok", "model_loaded": getattr(request.app.state, "llm", None) is not None}

# ====================== Strat ======================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)