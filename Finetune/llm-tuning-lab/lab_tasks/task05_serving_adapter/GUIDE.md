# Task 05: Adapter Serving

## vLLM + LoRA

動態載入多個 LoRA adapters

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM("meta-llama/Llama-2-7b-hf", enable_lora=True)

# 載入多個 adapters
adapters = {
    "finance": LoRARequest("finance", 1, "/path/to/finance_adapter"),
    "medical": LoRARequest("medical", 2, "/path/to/medical_adapter"),
}

# 推論時指定 adapter
output = llm.generate("...", lora_request=adapters["finance"])
```

詳見實作檔案
