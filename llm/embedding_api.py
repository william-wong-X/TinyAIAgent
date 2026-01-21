import argparse
import torch
from torch import Tensor
import torch.nn.functional as F
from contextlib import asynccontextmanager
from typing import List, Union
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import uvicorn

from utils.utils import get_dir
from config.config import load_config

# ====================== Config ======================
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="./config/config.yaml", help="Config path")
args = parser.parse_args()
config = load_config(args.config)

# ====================== Embedding Model ======================
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class EmbeddingModel:
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
                padding_side="left", 
                local_files_only=True, 
                trust_remote_code=True
            )
            # load model
            self.model = AutoModel.from_pretrained(
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
    def embed(self, texts: List[str], max_seq_len: int = 512):
        if not isinstance(texts, list):
            texts = [texts]

        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=max_seq_len, 
            return_tensors="pt"
        )
        attention_mask = encoded["attention_mask"]  # [B, L]
        encoded.to(self.model.device)

        outputs = self.model(**encoded)

        if hasattr(outputs, "last_hidden_state"):
            token_embeddings = outputs.last_hidden_state    # [B, L, H]
        else:
            token_embeddings = outputs[0]
        
        embeddings = last_token_pool(token_embeddings, attention_mask.to(token_embeddings.device))  # [B, H]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        token_count = int(attention_mask.sum().item())
        return embeddings.cpu().tolist(), token_count
    
# ====================== API Data Model (OpenAI) ======================
class EmbeddingRequest(BaseModel):
    model: str = Field(default_factory=lambda: getattr(config.embedding, "model", "qwen3-wmbedding-0.6b"))
    input: Union[str, List[str]]

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    prompt_token: int
    total_token: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage

# ====================== FastAPI App ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = get_dir(config.embedding.model_path)
    app.state.embedding = EmbeddingModel(model_path)
    yield

app = FastAPI(
    title="Embedding API", 
    description="OpenAI-style Embedding API", 
    lifespan=lifespan
)

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list", 
        "data": [{"id": config.embedding.model, "object": "model"}]
    }

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: Request, body: EmbeddingRequest):
    embedding_model: EmbeddingModel = getattr(request.app.state, "embedding", None)
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding Model Unload")
    
    if isinstance(body.input, str):
        texts = [body.input]
    else:
        texts = body.input
    
    embeddings, token_count = embedding_model.embed(texts)

    data = [
        EmbeddingData(
            embedding=vec, 
            index=i
        )
        for i, vec in enumerate(embeddings)
    ]

    usage = EmbeddingUsage(
        prompt_token=token_count, 
        total_token=token_count
    )

    return EmbeddingResponse(
        object="list", 
        data=data, 
        model=body.model, 
        usage=usage
    )

@app.get("/health")
async def health_check(request: Request):
    return {"status": "ok", "embedding_loaded": getattr(request.app.state, "embedding", None) is not None}

# ====================== Strat ======================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)