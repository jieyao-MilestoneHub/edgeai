# 📑 推論服務與 Serving

> 高效多 Adapter 推論架構

## vLLM + LoRA

### 架構特點
- **零延遲切換**：Adapter 熱掛載
- **Batch 優化**：動態批次處理
- **PagedAttention**：高效記憶體管理

### 使用範例
```python
# 載入 base model + 多個 adapters
vllm_server.load_adapters([
    "adapter_finance",
    "adapter_medical",
    "adapter_legal"
])

# 請求時指定 adapter
response = generate(
    prompt="...",
    adapter_name="adapter_finance"
)
```

詳見 [Task 05 - Adapter Serving](../lab_tasks/task05_serving_adapter/)
