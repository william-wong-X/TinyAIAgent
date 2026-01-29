import time
import argparse
import torch
import threading
import json
import re
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
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
    
    def _apply_template(self, messages, tools, enable_thinking):
        try:
            return self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, 
                add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            pass

        try:
            return self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, 
                add_generation_prompt=True
            )
        except TypeError:
            pass

        print("Warning: Tokenizer does not support 'tools' or 'enable_thinking'. Fallback to basic template.")
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, 
            add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self, 
        messages: List[dict], 
        tools: Optional[List[dict]], 
        max_new_tokens: int = 32765,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool = True,
    ) -> str:
        if not isinstance(messages, list):
            messages = [messages]

        text = self._apply_template(messages, tools, enable_thinking)

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

        full_text = full_text.replace(self.tokenizer.eos_token, "")
        for stop_token in ["<|im_end|>", "<|endoftext|>"]:
            if full_text.endswith(stop_token):
                full_text = full_text[:-len(stop_token)]

        return full_text.strip("\n")
    
    def stream_generate(
            self,
            messages: List[dict],
            tools: Optional[List[dict]] = None, 
            max_new_tokens: int = 32765,
            temperature: float = 0.7,
            top_p: float = 0.9,
            enable_thinking: bool = True,
    ):
        text = self._apply_template(messages, tools, enable_thinking)

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

            for stop_token in ["<|im_end|>", "<|endoftext|>"]:
                if stop_token in new_text:
                    new_text = new_text.replace(stop_token, "")
            
            if not new_text:
                continue

            yield new_text
    
# ====================== API Data Model (OpenAI) ======================
class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class Message(BaseModel):
    role: str
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]] = None

class RequestMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model: str = config.llm.model
    messages: List[RequestMessage]
    max_tokens: Optional[int] = 32765
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    enable_thinking: Optional[bool] = config.llm.enable_thinking
    stream: Optional[bool] = config.llm.streaming
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, List[Dict[str, Any]]]] = None

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

# ====================== Tool Parsing Logic ======================
def extract_tool_calls(content: str) -> Optional[List[ToolCall]]:
    if not content:
        return None
    
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        return None, None
    
    tool_calls_list = []
    for json_str in matches:
        try:
            tool_data = json.loads(json_str.strip())
            tool_call = ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}", 
                type="function", 
                function=FunctionCall(
                    name=tool_data.get("name"), 
                    arguments=json.dumps(tool_data.get("arguments", {}))
                )
            )
            tool_calls_list.append(tool_call)
        except Exception as e:
            print(f"JSON parsing error: {e} \nContent: {json_str}")

    cleaned_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
    if not cleaned_content:
        cleaned_content = None

    return cleaned_content, tool_calls_list

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
    
    formatted_messages=[]
    for m in body.messages:
        msg = {"role": m.role, "content": m.content}
        if m.tool_calls:
            msg["tool_calls"] = m.tool_calls
        formatted_messages.append(msg)
    
    is_streaming = body.stream
    # ============ Stream Model ============
    if is_streaming:
        def event_generator():
            chat_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())
            
            first_chunk = {
                "id": chat_id, 
                "object": "chat.completion.chunk", 
                "created": created, 
                "model": body.model,
                "choices": [
                    {
                        "index": 0, 
                        "delta": {"role": "assistant"}, 
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

            buffer = ""
            tool_mode = False

            generator = llm.stream_generate(
                messages=formatted_messages, 
                tools=body.tools, 
                max_new_tokens=body.max_tokens, 
                temperature=body.temperature, 
                top_p=body.top_p, 
                enable_thinking=body.enable_thinking
            )

            for new_text in generator:
                buffer += new_text

                if "<tool_call>" in buffer and not tool_mode:
                    pre_tool_content, tool_part = buffer.split("<tool_call>", 1)

                    if pre_tool_content:
                        content_chunk = {
                            "id": chat_id, 
                            "object": "chat.completion.chunk", 
                            "created": created, 
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0, 
                                    "delta": {"content": pre_tool_content}, 
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"
                    buffer = tool_part 
                    tool_mode = True
                    continue

                if tool_mode:
                    if "</tool_call>" in buffer:
                        json_str, remaining_post_tool = buffer.split("</tool_call>", 1)

                        try:
                            tool_data = json.loads(json_str.strip())
                            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"

                            tool_chunk_start = {
                                "id": chat_id, "object": "chat.completion.chunk", "created": created, "model": body.model,
                                "choices": [
                                    {
                                        "index": 0, 
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": 0,
                                                    "id": tool_call_id,
                                                    "type": "function",
                                                    "function": {"name": tool_data.get("name"), "arguments": ""}
                                            }
                                        ]
                                        }, 
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(tool_chunk_start, ensure_ascii=False)}\n\n"

                            args_str = json.dumps(tool_data.get("arguments", {}))
                            tool_chunk_args = {
                                "id": chat_id, 
                                "object": "chat.completion.chunk", 
                                "created": created, 
                                "model": body.model,
                                "choices": [
                                    {
                                        "index": 0, 
                                        "delta": {
                                            "tool_calls": [{"index": 0, "function": {"arguments": args_str}}]
                                        }, 
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(tool_chunk_args, ensure_ascii=False)}\n\n"

                        except Exception as e:
                            print(f"Stream Tool Parse Error: {e}")
                            error_chunk = {
                                "id": chat_id, 
                                "object": "chat.completion.chunk", 
                                "created": created, 
                                "model": body.model,
                                "choices": [
                                    {
                                        "index": 0, 
                                        "delta": {"content": f"<tool_call>{json_str}</tool_call>"}, 
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                        buffer = remaining_post_tool
                        tool_mode = False
                    else:
                        pass
                else:
                    if "<" in buffer:
                        last_bracket = buffer.rfind("<")
                        if last_bracket != -1 and len(buffer) - last_bracket < 12:
                            to_send = buffer[:last_bracket]
                            buffer = buffer[last_bracket:]
                            if to_send:
                                chunk = {
                                   "id": chat_id, 
                                   "object": "chat.completion.chunk", 
                                   "created": created, 
                                   "model": body.model,
                                   "choices": [
                                       {
                                           "index": 0, 
                                           "delta": {"content": to_send}, 
                                           "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        else:
                            chunk = {
                               "id": chat_id, 
                               "object": "chat.completion.chunk", 
                               "created": created, 
                               "model": body.model,
                               "choices": [
                                   {
                                       "index": 0, 
                                       "delta": {"content": buffer}, 
                                       "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            buffer = ""
                    else:
                        chunk = {
                            "id": chat_id, 
                            "object": "chat.completion.chunk", 
                            "created": created, 
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0, 
                                    "delta": {"content": buffer}, 
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        buffer = ""

            if buffer and not tool_mode:
                chunk = {
                    "id": chat_id, "object": "chat.completion.chunk", "created": created, "model": body.model,
                    "choices": [{"index": 0, "delta": {"content": buffer}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            finish_reason = "tool_calls" if (tool_mode is False and body.tools and "tool_calls" in str(first_chunk)) else "stop"
            
            final_chunk = {
                "id": chat_id, 
                "object": "chat.completion.chunk", 
                "created": created, 
                "model": body.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ============ Non-stream Model ============
    content = llm.generate(
        messages=formatted_messages, 
        tools=body.tools, 
        max_new_tokens=body.max_tokens, 
        temperature=body.temperature, 
        top_p=body.top_p, 
        enable_thinking=body.enable_thinking
    )

    cleaned_content, tool_calls = extract_tool_calls(content)

    if tool_calls:
        response_message = Message(
            role="assistant", 
            content=cleaned_content, 
            tool_calls=tool_calls
        )
        finish_reason = "tool_calls"
    else:
        response_message = Message(
            role="assistant", 
            content=content
        )
        finish_reason = "stop"

    return ChatCompletionResponse(
        model=body.model, 
        choices=[
            Choice(
                message=response_message,
                finish_reason=finish_reason
            )
        ]
    )

@app.get("/health")
async def health_check(request: Request):
    return {"status": "ok", "model_loaded": getattr(request.app.state, "llm", None) is not None}

# ====================== Strat ======================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)