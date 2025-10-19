# 模型訓練模組

> QLoRA 指令微調訓練腳本

## 📋 概述

本模組負責使用 QLoRA 技術微調 Qwen2.5-3B-Instruct 模型。

## 🚀 使用方式

### 訓練模型

```bash
# 使用基礎配置
python train_qwen_instruct.py

# 使用特定實驗配置
python train_qwen_instruct.py \
  --config ../config.yaml \
  --exp_config ../configs/exp4_r16_full.yaml

# 自定義輸出目錄
python train_qwen_instruct.py \
  --config ../config.yaml \
  --output_dir my_custom_output
```

### 測試模型

```bash
# 演示模式（預設測試案例）
python inference_example.py \
  --model_path ../output_task03_exp4_r16_full \
  --mode demo

# 互動模式
python inference_example.py \
  --model_path ../output_task03_exp4_r16_full \
  --mode interactive

# 單次測試
python inference_example.py \
  --model_path ../output_task03_exp4_r16_full \
  --mode text \
  --instruction "Explain VLSI design flow"
```

## ⚙️ 核心技術

### SFTTrainer（Supervised Fine-tuning Trainer）

使用 `trl.SFTTrainer` 進行指令微調，優勢：

- 自動處理 instruction-following 格式
- 內建 LoRA 支援
- 優化的記憶體管理
- 簡化的訓練循環

### Prompt 格式

訓練使用 Qwen 的 ChatML 格式：

```
<|im_start|>system
You are a helpful AI assistant specialized in engineering and technical domains.<|im_end|>
<|im_start|>user
{instruction}
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

## 📊 訓練輸出

訓練完成後會生成：

```
output_task03_exp*/
├── adapter_config.json       # LoRA 配置
├── adapter_model.safetensors # LoRA 權重（8-50MB）
├── training_curves.png       # 訓練曲線
├── training_log.txt          # 訓練日誌
└── tokenizer files           # Tokenizer 檔案
```

## 🔧 配置參數

關鍵配置項（`config.yaml`）：

```yaml
# LoRA 配置
lora:
  rank: 16              # 低秩維度
  alpha: 32.0           # 縮放因子
  target_modules:       # 目標層
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"      # 可選：MLP 層
    - "up_proj"
    - "down_proj"

# 訓練配置
training:
  num_epochs: 3
  batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
```

## 💾 記憶體優化

RTX 4060 8GB 優化策略：

1. **4-bit 量化**: 模型權重降至 1.5GB
2. **Gradient Checkpointing**: 節省 30-50% activation 記憶體
3. **8-bit Paged Optimizer**: 優化器狀態壓縮
4. **Small Batch + Gradient Accumulation**: batch_size=1, accumulation=16

## ⚠️ 常見問題

### Q: 訓練中斷後如何續訓？

A: 使用 checkpoint 續訓：

```bash
python train_qwen_instruct.py \
  --config ../config.yaml \
  --exp_config ../configs/exp4_r16_full.yaml \
  --resume_from_checkpoint output_task03_exp4_r16_full/checkpoint-500
```

### Q: 如何調整學習率？

A: 修改 `config.yaml` 或使用命令列覆寫：

```yaml
training:
  learning_rate: 5.0e-4  # 增加學習率
```

### Q: 訓練效果不佳怎麼辦？

A: 檢查清單：

1. Loss 是否下降？
2. 是否過擬合？（train loss << eval loss）
3. 學習率是否合適？
4. 數據品質是否良好？

## 📚 相關資源

- [SFTTrainer 文檔](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT 文檔](https://huggingface.co/docs/peft/)
- [Qwen2.5 文檔](https://qwenlm.github.io/)
