# Task 02 詳細教學指引：QLoRA 實戰

> 使用 4-bit 量化技術，用消費級 GPU 訓練 7B/13B 模型

## 🎯 學習目標

完成本任務後，你將能夠：
- ✅ 理解量化的基本原理（INT8, INT4, NF4）
- ✅ 使用 bitsandbytes 進行 4-bit 量化
- ✅ 實作 QLoRA 訓練流程
- ✅ 對比 LoRA vs QLoRA 的記憶體與精度差異

---

## 第一部分：量化原理

### 1.1 為什麼需要量化？

**問題：大模型訓練的記憶體瓶頸**

```python
# LLaMA-7B 全參數微調
模型參數：7B
FP16 精度：2 bytes/param
模型權重：7B × 2 = 14GB

訓練時額外需求：
- Optimizer states (Adam): 14GB × 2 = 28GB
- Gradients: 14GB
- Activations: ~10GB
總計：~66GB  ❌ 單張 A100 (40GB) 無法訓練
```

**LoRA 的改善：**
```python
# LoRA (FP16)
凍結權重：14GB (只載入，不訓練)
LoRA 參數：~40MB
Optimizer: 80MB
總計：~14.2GB  ✅ 可以訓練，但仍需大記憶體載入
```

**QLoRA 的突破：**
```python
# QLoRA (4-bit 量化)
量化權重：7B × 0.5 bytes = 3.5GB  🎉
LoRA 參數：~40MB
Optimizer: 80MB
總計：~3.7GB  ✅✅ 消費級 GPU 也能訓練大模型！
```

### 1.2 量化基礎

#### 什麼是量化？

```
量化 = 用更少的 bits 表示數字

FP32 (32-bit)：精度最高，記憶體最大
    ↓ 2× 壓縮
FP16 (16-bit)：精度略降，記憶體減半
    ↓ 2× 壓縮
INT8 (8-bit)：整數表示，記憶體 1/4
    ↓ 2× 壓縮
INT4 (4-bit)：更激進，記憶體 1/8  ← QLoRA 使用
```

#### 量化過程

```python
# 原始 FP16 權重
weight_fp16 = torch.tensor([0.5234, -1.2341, 0.8923, ...])

# 量化到 INT4（範圍 0-15）
min_val = weight_fp16.min()  # -1.2341
max_val = weight_fp16.max()  #  0.8923

# 線性映射到 [0, 15]
scale = (max_val - min_val) / 15
weight_int4 = ((weight_fp16 - min_val) / scale).round().to(torch.uint8)
# [10, 0, 13, ...]

# 反量化（計算時）
weight_dequant = weight_int4 * scale + min_val
# [0.5234, -1.2341, 0.8923, ...] （有微小誤差）
```

### 1.3 NF4 (NormalFloat 4-bit)

QLoRA 的關鍵創新：**針對神經網絡權重分佈優化的量化格式**

**觀察：** 神經網絡權重通常服從正態分佈（大部分值接近 0）

```
普通 INT4：均勻分佈量化級別
[-15, -13, -11, ..., -1, 1, ..., 11, 13, 15]
         ↓
問題：浪費了接近 0 的精度

NF4：根據正態分佈優化
更多級別集中在 0 附近：
[-1.0, -0.6961, -0.5250, -0.3949, -0.2844, -0.1848, -0.0911, 0,
 0.0911, 0.1848, 0.2844, 0.3949, 0.5250, 0.6961, 1.0]
         ↓
優勢：在同樣 4-bit 下，精度更高！
```

### 1.4 雙重量化 (Double Quantization)

**進一步壓縮：連量化常數也量化！**

```python
# 第一次量化：權重
weight_int4, scale_fp32, zero_point_fp32 = quantize_weights(W)

# 問題：scale 和 zero_point 仍是 FP32
# 對於大模型，這些常數也佔空間

# 第二次量化：量化常數也量化到 INT8
scale_int8 = quantize_scale(scale_fp32)
zero_point_int8 = quantize_scale(zero_point_fp32)

# 總記憶體節省：
# 原本：4-bit weights + FP32 scales
# 現在：4-bit weights + INT8 scales
# 額外節省 ~0.4 GB (對 65B 模型)
```

---

## 第二部分：QLoRA 實作

### 2.1 使用 bitsandbytes

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ========== 4-bit 量化配置 ==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 啟用 4-bit 量化
    bnb_4bit_quant_type="nf4",              # 使用 NF4 格式
    bnb_4bit_compute_dtype=torch.bfloat16,  # 計算時的精度
    bnb_4bit_use_double_quant=True,         # 啟用雙重量化
)

# ========== 載入量化模型 ==========
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",  # 自動分配 GPU/CPU
    trust_remote_code=True,
)

# ========== 準備模型（啟用梯度檢查點等） ==========
model = prepare_model_for_kbit_training(model)

# ========== 添加 LoRA ==========
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

### 2.2 完整訓練腳本

詳見 `train_qlora.py`

---

## 第三部分：記憶體與精度對比

### 3.1 記憶體使用對比

```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def measure_memory(model_name, use_qlora=False):
    """測量模型記憶體使用"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if use_qlora:
        # QLoRA 載入
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
        )
    else:
        # 正常載入
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

    # 添加 LoRA
    model = get_peft_model(model, LoraConfig(...))

    # 測量
    memory_used = torch.cuda.max_memory_allocated() / 1e9
    return memory_used

# 對比
fp16_memory = measure_memory("meta-llama/Llama-2-7b-hf", use_qlora=False)
qlora_memory = measure_memory("meta-llama/Llama-2-7b-hf", use_qlora=True)

print(f"FP16 LoRA:  {fp16_memory:.2f} GB")
print(f"QLoRA:      {qlora_memory:.2f} GB")
print(f"節省:       {(1 - qlora_memory/fp16_memory)*100:.1f}%")

# 預期輸出：
# FP16 LoRA:  14.5 GB
# QLoRA:      3.8 GB
# 節省:       73.8%
```

### 3.2 精度對比實驗

```python
# 在相同數據上訓練兩個模型
# 1. LoRA (FP16)
# 2. QLoRA (4-bit)

# 比較 metrics
results = {
    "FP16 LoRA": {
        "final_loss": 1.234,
        "perplexity": 3.45,
        "training_time": "45 min",
    },
    "QLoRA": {
        "final_loss": 1.256,  # 略高，但可接受
        "perplexity": 3.52,
        "training_time": "52 min",  # 略慢（量化/反量化開銷）
    },
}

# 精度差異：通常 <2%
```

---

## 第四部分：進階技巧

### 4.1 分頁優化器 (Paged Optimizer)

QLoRA 的另一個創新：處理記憶體峰值

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    optim="paged_adamw_32bit",  # 使用分頁優化器
    # 當 GPU 記憶體不足時，自動轉移到 CPU
)
```

### 4.2 Gradient Checkpointing

```python
model.gradient_checkpointing_enable()

# 原理：
# 不儲存所有中間 activations
# 需要時重新計算
# 記憶體 ↓ 50%，速度 ↓ 20%
```

### 4.3 最佳實踐

**推薦配置（7B 模型）：**
```python
# 量化配置
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16  # 如果 GPU 支援
bnb_4bit_use_double_quant=True

# LoRA 配置
r=16  # 或 64（視任務複雜度）
lora_alpha=32  # = 2 × r
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 訓練配置
per_device_train_batch_size=4
gradient_accumulation_steps=4  # 有效 batch size = 16
gradient_checkpointing=True
optim="paged_adamw_32bit"
```

---

## 第五部分：常見問題

### Q1: 量化會損失多少精度？

A: 實驗顯示，NF4 + QLoRA 的精度損失通常 <2%，完全可接受。

### Q2: 訓練速度會變慢嗎？

A: 是的，約慢 20-30%（量化/反量化開銷），但記憶體節省是值得的。

### Q3: 推論時也需要量化嗎？

A: 可選。訓練後可以：
- 方案 A：合併 LoRA 權重，反量化為 FP16（推論更快）
- 方案 B：保持 4-bit + LoRA（節省顯存，可部署更多模型）

### Q4: 所有模型都能用 QLoRA 嗎？

A: 大部分 Transformer 架構都支援。檢查 `bitsandbytes` 相容性。

---

## 🎓 學習檢查清單

- [ ] 理解量化的基本原理
- [ ] 知道 NF4 與普通 INT4 的差異
- [ ] 能配置 BitsAndBytesConfig
- [ ] 成功訓練 QLoRA 模型
- [ ] 對比測量記憶體使用
- [ ] 分析精度損失

---

**恭喜完成 Task 02！你現在可以在消費級 GPU 上訓練大模型了！🎉**

[← 返回 Task 01](../task01_lora_basic/) | [下一篇：Task 03 SDK 與 API →](../task03_sdk_api/)
