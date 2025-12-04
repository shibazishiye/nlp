# main.py
import asyncio
import time
import numpy as np
import onnxruntime as ort
from typing import List
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoTokenizer
import uvicorn
import math

# ===================== CONFIG =====================
ONNX_PATH = "/app/models/roberta_wwm_mlm_dynamic_new.onnx"
TOKENIZER_NAME = "hfl/chinese-roberta-wwm-ext"
USE_CUDA = False

INTRA_THREADS = 8
INTER_THREADS = 4

MICROBATCH_MAX_SIZE = 128      # 单次合并处理的最大请求数
MICROBATCH_TIMEOUT = 0.020     # 20ms flush timeout → 不影响用户体感

PPL_THRESHOLD = 20

# ==================================================

# ----- ONNX Session -----
so = ort.SessionOptions()
so.intra_op_num_threads = INTRA_THREADS
so.inter_op_num_threads = INTER_THREADS
so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.log_severity_level = 3

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if USE_CUDA else ["CPUExecutionProvider"]
)

sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=providers)

# ----- Tokenizer -----
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# =========== Request / Response Models ============
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class PPLResponse(BaseModel):
    ppl: float
    threshold: int = PPL_THRESHOLD
    remark:str = "大于threshold阈值的句子通常没有实际意义。"

class BatchPPLResponse(BaseModel):
    ppls: List[float]    
    threshold: int = PPL_THRESHOLD
    remark:str = "大于threshold阈值的句子通常没有实际意义。"

class HealthResponse(BaseModel):
    status: str
    version: str
    
# =========== MicroBatch Queue ============
class MicroBatchQueue:
    def __init__(self):
        self.queue = []
        self.event = asyncio.Event()
        self.lock = asyncio.Lock()

    async def add(self, text):
        fut = asyncio.get_event_loop().create_future()
        async with self.lock:
            self.queue.append((text, fut))
            self.event.set()
        return fut

    async def get_batch(self):
        """
        当 queue 满足条件时，取 batch
        """
        while True:
            await self.event.wait()

            async with self.lock:
                if len(self.queue) == 0:
                    self.event.clear()
                    continue

                # batch full OR timeout
                batch_size = min(len(self.queue), MICROBATCH_MAX_SIZE)
                batch = self.queue[:batch_size]
                self.queue = self.queue[batch_size:]
                if len(self.queue) == 0:
                    self.event.clear()

            return batch

microbatcher = MicroBatchQueue()

# ------------------- MLM Core Logic ---------------------

def logsumexp(arr, axis=-1):
    mx = np.max(arr, axis=axis, keepdims=True)
    return mx + np.log(np.sum(np.exp(arr - mx), axis=axis, keepdims=True))

def compute_ppl_batch(text_list: List[str]):
    """
    一次性计算多条句子的 perplexity
    """
    enc = tokenizer(text_list, return_tensors="np", padding=True, truncation=True)
    input_ids = enc["input_ids"].astype("int64")
    attention_mask = enc["attention_mask"].astype("int64")

    M, L = input_ids.shape

    mask_positions = []
    orig_token_values = []
    sentence_index = []

    for sid in range(M):
        valid_pos = np.where(attention_mask[sid] == 1)[0]
        valid_pos = valid_pos[(valid_pos != 0) & (valid_pos != (L - 1))]
        for pos in valid_pos:
            mask_positions.append(pos)
            orig_token_values.append(int(input_ids[sid, pos]))
            sentence_index.append(sid)

    if not mask_positions:
        return [float("inf")] * M

    N = len(mask_positions)
    batch_input = np.empty((N, L), dtype="int64")
    batch_att = np.empty((N, L), dtype="int64")
    batch_tok = np.zeros((N, L), dtype="int64")

    src_ids = np.array(sentence_index, dtype=np.int64)
    batch_input[:] = input_ids[src_ids, :]
    batch_att[:] = attention_mask[src_ids, :]

    mask_token_id = tokenizer.mask_token_id
    rows = np.arange(N)
    cols = np.array(mask_positions, dtype=np.int64)
    batch_input[rows, cols] = mask_token_id

    logits = sess.run(
        ["logits"],
        {
            "input_ids": batch_input,
            "attention_mask": batch_att,
            "token_type_ids": batch_tok,
        },
    )[0]

    logits_at_pos = logits[np.arange(N), cols, :]  # (N, V)
    lse = np.squeeze(logsumexp(logits_at_pos, axis=1), axis=1)
    target_logits = logits_at_pos[np.arange(N), np.array(orig_token_values)]
    nll = -(target_logits - lse)

    sent_losses = [[] for _ in range(M)]
    for i in range(N):
        sid = sentence_index[i]
        sent_losses[sid].append(nll[i])

    ppls = []
    for sid in range(M):
        if not sent_losses[sid]:
            ppls.append(float("inf"))
        else:
            mean_nll = float(np.mean(sent_losses[sid]))
            ppls.append(float(math.exp(mean_nll)))

    return ppls

# ================= MicroBatch Worker ====================
async def microbatch_worker():
    """
    每次等待 MicroBatchQueue，直到 batch 满足条件（满size或timeout）
    flush 出来后计算 ppl 并把结果分发回 future。
    """
    while True:
        t_start = time.time()
        batch = await microbatcher.get_batch()

        # timeout: 等 20ms
        while True:
            await asyncio.sleep(0.001)
            if len(microbatcher.queue) == 0:
                break
            if time.time() - t_start > MICROBATCH_TIMEOUT:
                break
            if len(microbatcher.queue) >= MICROBATCH_MAX_SIZE:
                break

        # Build final batch
        texts = [x[0] for x in batch]
        futs  = [x[1] for x in batch]

        # Inference
        ppls = compute_ppl_batch(texts)

        # return results
        for fut, ppl in zip(futs, ppls):
            if not fut.done():
                fut.set_result(ppl)

# ---------------- FastAPI App ----------------
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(microbatch_worker())
    print("Microbatch worker started.")

@app.post("/ppl", response_model=PPLResponse)
async def get_ppl(req: TextRequest):
    fut = await microbatcher.add(req.text)
    ppl = await fut
    return PPLResponse(ppl=ppl)

@app.post("/ppl_batch", response_model=BatchPPLResponse)
async def get_ppl_batch(req: BatchRequest):
    # 这里不用 microbatch，直接批量计算即可
    ppls = compute_ppl_batch(req.texts)
    return BatchPPLResponse(ppls=ppls)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status and model availability
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )

@app.get("/index")
async def read_root():
    try:
        with open("/app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content, media_type="text/html")
    except FileNotFoundError:
        return Response("File not found", status_code=404)
        
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        API information
    """
    return {
        "name": "困惑度检查API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ppl": "检查一句话的PPL",
            "GET /ppl_batch": "检查一批话的PPL",
            "GET /health": "健康检查"
        }
    }
    
# ---------------- Run server ----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
