# Task 01: InfographicVQA微調實戰

> 使用QLoRA在RTX 3070 (8GB)上微調視覺語言模型

## 任務目標

在本任務中,你將:
1. 探索InfographicVQA資料集特性
2. 選擇適合8GB VRAM的模型
3. 使用QLoRA進行微調
4. 評估模型性能並分析結果
5. 對比不同模型的表現

**預計時間**: 4-6小時(包含訓練時間)

## 硬體需求

- **最低配置**: RTX 3070 8GB VRAM + 32GB RAM
- **推薦配置**: RTX 4070 12GB VRAM + 32GB RAM
- **替代方案**: Google Colab (免費T4 16GB)

## 檔案結構

```
task01_infographicvqa/
├── README.md                    # 本文件
├── requirements.txt             # 依賴套件
├── data_exploration.ipynb       # 資料集探索notebook
├── finetune_pix2struct.py       # Pix2Struct微調腳本
├── finetune_qwen_vl.py          # Qwen-VL微調腳本
├── evaluate.py                  # 評估腳本(計算ANLS)
├── inference.py                 # 推論範例
└── utils/
    ├── dataset_loader.py        # 資料集載入工具
    ├── memory_profiler.py       # VRAM監控
    └── metrics.py               # ANLS計算
```

## 快速開始

### 步驟1: 安裝依賴

```bash
cd lab_tasks/task01_infographicvqa
pip install -r requirements.txt
```

### 步驟2: 探索資料集

```bash
jupyter notebook data_exploration.ipynb
```

或使用命令行:
```bash
python -c "
from datasets import load_dataset
dataset = load_dataset('MMInstruction/InfographicVQA')
print(f'訓練集: {len(dataset[\"train\"])} 樣本')
sample = dataset['train'][0]
print(f'問題: {sample[\"question\"]}')
print(f'答案: {sample[\"answers\"]}')
"
```

### 步驟3: 選擇模型並開始微調

**選項A: Pix2Struct (推薦首次嘗試)**

```bash
python finetune_pix2struct.py \
  --model_name google/pix2struct-infographics-vqa-base \
  --output_dir ./outputs/pix2struct \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

**選項B: Qwen-VL-3B (最佳性能)**

```bash
python finetune_qwen_vl.py \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir ./outputs/qwen_vl \
  --num_epochs 3 \
  --batch_size 2 \
  --learning_rate 2e-4 \
  --use_qlora
```

### 步驟4: 評估模型

```bash
python evaluate.py \
  --model_path ./outputs/qwen_vl \
  --dataset_split validation \
  --output_file results.json
```

### 步驟5: 推論測試

```bash
python inference.py \
  --model_path ./outputs/qwen_vl \
  --image_path test_image.jpg \
  --question "What was the revenue in 2021?"
```

## 詳細說明

### 模型選擇指南

| 模型 | 參數量 | VRAM需求 | 訓練時間 | 預期ANLS |
|------|--------|---------|---------|---------|
| **Pix2Struct-base** | 282M | 3-4GB | 2-3小時 | 0.45-0.50 |
| **Qwen2.5-VL-3B** | 3B | 6-7GB | 4-5小時 | 0.52-0.58 |
| PaliGemma2-3B | 3B | 7-8GB | 5-6小時 | 0.50-0.55 |

**決策建議**:
- 首次學習或快速實驗 → Pix2Struct
- 追求最佳性能 → Qwen2.5-VL-3B
- 圖片細節很重要 → PaliGemma2-3B

### QLoRA配置說明

**完整配置範例**:

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig

# QLoRA 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # BF16計算
    bnb_4bit_use_double_quant=True,      # 雙重量化
)

# LoRA配置
lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Alpha = 2 × r
    target_modules=[         # 目標模組
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**參數調整建議**:

| 參數 | 預設值 | 調整方向 | 效果 |
|------|--------|---------|------|
| `r` | 16 | ↑32 | ↑性能, ↑VRAM, ↑過擬合風險 |
| `r` | 16 | ↓8 | ↓VRAM, ↓性能 |
| `lora_alpha` | 32 | = 2×r | 標準比例 |
| `target_modules` | 4個 | +FFN層 | ↑性能, ↑↑VRAM |
| `lora_dropout` | 0.05 | ↑0.1 | ↓過擬合(小資料集) |

### 記憶體優化技巧

**如果遇到OOM(Out of Memory)**:

1. **減小batch size**
```python
per_device_train_batch_size=1  # 從2降到1
gradient_accumulation_steps=8  # 從4增到8
```

2. **啟用gradient checkpointing**
```python
gradient_checkpointing=True  # 節省50%激活值記憶體
```

3. **限制序列長度**
```python
max_length=512  # 從2048降到512(根據資料調整)
```

4. **使用8-bit優化器**
```python
optim="paged_adamw_8bit"  # 支持CPU offload
```

5. **減少LoRA rank**
```python
r=8  # 從16降到8
```

### 訓練監控

**使用TensorBoard**:
```bash
tensorboard --logdir ./outputs/qwen_vl/logs
```

**監控指標**:
- `train_loss`: 應穩定下降
- `eval_loss`: 不應持續上升(過擬合警訊)
- `gpu_memory_usage`: 確保<8GB

**記憶體profiling**:
```python
from utils.memory_profiler import print_gpu_utilization

print_gpu_utilization()  # 訓練前
trainer.train()
print_gpu_utilization()  # 訓練後
```

### 評估與分析

**ANLS計算**:
```python
from utils.metrics import calculate_anls

# 單個樣本
score = calculate_anls(
    prediction="Barack Obama",
    ground_truths=["Barack H. Obama", "Obama"]
)
print(f"ANLS: {score:.4f}")  # 0.8667

# 整個資料集
from evaluate import evaluate_model
results = evaluate_model(
    model_path="./outputs/qwen_vl",
    dataset_split="validation"
)
print(f"平均ANLS: {results['avg_anls']:.4f}")
```

**錯誤分析**:
```bash
python evaluate.py \
  --model_path ./outputs/qwen_vl \
  --dataset_split validation \
  --save_predictions predictions.json \
  --analyze_errors
```

輸出會包含:
- 完全匹配率
- 部分匹配率
- 常見錯誤類型(OCR錯誤、計算錯誤等)
- 失敗案例範例

## 常見問題

### Q1: 訓練時出現CUDA Out of Memory

**解決方案**:
1. 檢查是否正確啟用4-bit量化
2. 減小batch size至1
3. 啟用gradient_checkpointing
4. 關閉其他佔用GPU的程序

### Q2: ANLS分數很低(<0.3)

**可能原因**:
1. 學習率過大或過小 → 嘗試1e-4到5e-4
2. 訓練時間不足 → 增加epochs
3. 模型沒有正確載入LoRA → 檢查checkpoint路徑
4. 答案格式不一致 → 檢查預處理和後處理

### Q3: 訓練很慢

**優化建議**:
1. 使用更小的模型(Pix2Struct-base)
2. 減少驗證頻率 (`eval_steps=1000`)
3. 使用混合精度 (`bf16=True`)
4. 檢查是否使用正確的CUDA版本

### Q4: 如何在Colab上運行?

**Colab notebook範例**:
```python
# 1. 檢查GPU
!nvidia-smi

# 2. 安裝依賴
!pip install -r requirements.txt

# 3. 使用較小batch size
!python finetune_qwen_vl.py \
  --batch_size 1 \
  --gradient_accumulation_steps 8
```

## 進階實驗

### 實驗1: 對比不同LoRA rank

```bash
for rank in 8 16 32; do
  python finetune_qwen_vl.py \
    --lora_rank $rank \
    --output_dir ./outputs/qwen_vl_r${rank}
done

python compare_ranks.py  # 生成對比報告
```

### 實驗2: 測試不同學習率

```bash
for lr in 1e-4 2e-4 5e-4; do
  python finetune_qwen_vl.py \
    --learning_rate $lr \
    --output_dir ./outputs/qwen_vl_lr${lr}
done
```

### 實驗3: 資料增強

在`finetune_qwen_vl.py`中啟用:
```python
--use_augmentation \
--aug_brightness 0.2 \
--aug_contrast 0.2
```

## 成功標準

完成本任務後,你應該能夠:

- [ ] 在8GB GPU上成功訓練Qwen2.5-VL-3B
- [ ] 驗證集ANLS達到0.50以上
- [ ] 理解QLoRA的配置參數含義
- [ ] 能診斷並解決OOM問題
- [ ] 實作自定義評估指標
- [ ] 分析並解釋失敗案例

## 參考資源

- [InfographicVQA論文](https://arxiv.org/abs/2104.12756)
- [QLoRA論文](https://arxiv.org/abs/2305.14314)
- [Qwen-VL文檔](https://github.com/QwenLM/Qwen-VL)
- [PEFT文檔](https://huggingface.co/docs/peft)

## 下一步

完成本任務後,繼續:
- [Task 02: 量化感知訓練(QAT)](../task02_qat/)
- [Task 03: 混合精度優化](../task03_deployment/)

或回到 [理論文檔](../../docs/01_infographicvqa_qlora.md) 深入理解原理
