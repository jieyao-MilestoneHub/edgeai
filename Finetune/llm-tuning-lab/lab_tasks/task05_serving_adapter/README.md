# Task 05: Adapter Serving 與多模型熱掛載

> 實作高效多 Adapter 推論服務

## 🎯 學習目標

- ✅ vLLM 整合
- ✅ 動態 Adapter 載入
- ✅ 多 Adapter 批次推論
- ✅ 推論性能優化

## 核心功能

```python
# 單一 base model + 多個 adapters
server.load_adapters([
    "finance_adapter",
    "medical_adapter",
    "legal_adapter"
])

# 請求時指定 adapter
response = server.generate(
    prompt="...",
    adapter="finance_adapter"
)
```

詳見 [GUIDE.md](GUIDE.md)
