"""多 Adapter 推論服務"""
from fastapi import FastAPI
from vllm import LLM
from vllm.lora.request import LoRARequest

app = FastAPI()
llm = LLM("meta-llama/Llama-2-7b-hf", enable_lora=True)

# 預載入的 adapters
adapters = {
    "finance": LoRARequest("finance", 1, "/path/to/finance"),
    "medical": LoRARequest("medical", 2, "/path/to/medical"),
}

@app.post("/generate")
async def generate(prompt: str, adapter: str = "finance"):
    lora_req = adapters.get(adapter)
    output = llm.generate(prompt, lora_request=lora_req)
    return {"response": output[0].outputs[0].text}
