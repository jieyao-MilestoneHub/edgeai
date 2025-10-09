# 📕 QLoRA 與量化

> 4-bit 量化技術實現記憶體高效微調

## 核心概念

QLoRA = **Q**uantized + Lo**RA**

### 關鍵創新
1. **4-bit NormalFloat (NF4)**：針對正態分佈優化的量化格式
2. **雙重量化**：對量化常數再量化
3. **分頁優化器**：處理記憶體峰值

## 記憶體節省

```
LLaMA-65B Full Fine-tuning: >780GB
LLaMA-65B LoRA (FP16):     ~120GB  
LLaMA-65B QLoRA (4-bit):    ~48GB  ✅ 單卡可訓練！
```

## 實作要點

```python
from transformers import BitsAndBytesConfig

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

詳見 [Task 02 - QLoRA 實戰](../lab_tasks/task02_qlora/)
