# Task 03: Engineering Agent - QLoRA Fine-tuning

> 在有限硬體資源（RTX 4060 8GB）下，使用 QLoRA 微調 Qwen2.5-3B-Instruct，培養工程類指令理解與回應能力

## 🎯 任務目標

本任務旨在探索如何在消費級 GPU 上，利用 QLoRA 技術微調大型語言模型，使其具備：

1. **指令遵循能力**：理解並準確執行用戶指令
2. **工程領域知識**：掌握 IC 設計、系統架構、演算法等技術概念
3. **技術語境感知**：在工程場景下給出專業、準確的回應

### 為什麼選擇這個配置？

- **模型選擇**: Qwen2.5-3B-Instruct
  * 30 億參數，在性能與資源間取得平衡
  * 原生支援指令遵循，微調效果更好
  * 4-bit 量化後僅需 ~3.5GB 顯存

- **技術選擇**: QLoRA (Quantized LoRA)
  * 4-bit NF4 量化，記憶體需求降低 75%
  * 只訓練 0.07% 的參數，訓練速度快 5 倍
  * LoRA adapter 僅 8MB，便於分享與管理

- **數據策略**: 通用指令 + 技術知識
  * OpenHermes-2.5 (~20k): 穩定的指令遵循基礎
  * ScienceQA + PubMedQA (~5k): 工程相關技術知識

## 📁 專案結構

```
task03_engineering_agent/
├── README.md                          # 📖 本檔案：專案總覽
├── requirements.txt                   # 📦 依賴套件清單
├── config.yaml                        # ⚙️  基礎配置模板
│
├── configs/                           # 🧪 實驗配置目錄
│   ├── exp1_r8_attention.yaml        # Rank=8, Attention only
│   ├── exp2_r8_full.yaml             # Rank=8, Attention+MLP
│   ├── exp3_r16_attention.yaml       # Rank=16, Attention only
│   ├── exp4_r16_full.yaml            # Rank=16, Attention+MLP (基準)
│   ├── exp5_r32_attention.yaml       # Rank=32, Attention only
│   └── exp6_r32_full.yaml            # Rank=32, Attention+MLP
│
├── data_preparation/                # 📊 資料準備模組
│   ├── README.md                     # 資料準備說明
│   ├── download_datasets.py          # 下載數據集
│   ├── filter_engineering.py         # 過濾工程相關數據
│   └── merge_datasets.py             # 合併並統一格式
│
├── model_training/                  # 🚀 模型訓練模組
│   ├── README.md                     # 訓練說明
│   ├── train_qwen_instruct.py        # 主訓練腳本
│   └── inference_example.py          # 推論測試
│
├── experiments/                     # 🧪 實驗運行模組
│   ├── run_all_experiments.py        # 批次運行所有實驗
│   ├── compare_results.py            # 比較實驗結果
│   └── results/                      # 實驗結果保存
│
├── evaluation/                      # 📋 評估模組
│   ├── test_cases.yaml               # 工程類測試案例
│   └── evaluate_model.py             # 自動評估腳本
│
└── report_generation/               # 📝 報告生成模組
    ├── generate_report.py            # 自動生成報告
    └── final_report.md               # 最終報告輸出
```

## 🚀 快速開始

### 環境需求

**硬體**:
- GPU: RTX 4060 8GB 或以上
- RAM: 16GB 以上
- 硬碟: 20GB 可用空間

**軟體**:
- Python 3.10+
- CUDA 11.8+
- Git

### 步驟 1: 安裝依賴

```bash
# 安裝 PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install -r requirements.txt
```

### 步驟 2: 資料準備

```bash
cd data_preparation

# 下載數據集
python download_datasets.py

# 過濾工程相關數據
python filter_engineering.py

# 合併數據集
python merge_datasets.py

cd ..
```

**預期結果**: 在 `data_preparation/final_dataset/` 生成訓練用數據集

### 步驟 3: 訓練模型

#### 選項 A: 訓練單個配置

```bash
cd model_training

# 使用基準配置訓練
python train_qwen_instruct.py \
  --config ../config.yaml \
  --exp_config ../configs/exp4_r16_full.yaml
```

#### 選項 B: 運行所有實驗（推薦）

```bash
cd experiments

# 批次運行 6 個實驗配置
python run_all_experiments.py
```

**預期時間**: 每個實驗約 1-2 小時（RTX 4060），總計 6-12 小時

### 步驟 4: 評估模型

```bash
cd evaluation

# 評估訓練好的模型
python evaluate_model.py `
  --model_path ../model_training/output_task03_exp4_r16_full `
  --test_cases test_cases.yaml `
  --output exp4_evaluation_report.json
```

### 步驟 5: 生成報告

```bash
cd report_generation

# 生成綜合報告
python generate_report.py
```

**輸出**: `final_report.md` 包含完整的實驗分析與建議

## 🔬 實驗設計

### 調參比較維度

本任務設計了 **6 個實驗**，比較兩個維度：

1. **LoRA Rank**: 8, 16, 32
   - Rank 越高，模型表達能力越強，但參數量也越大

2. **Target Modules**: Attention vs Attention+MLP
   - Attention only: 只訓練注意力層，參數量較少
   - Attention+MLP: 訓練注意力層和 MLP 層，效果更好但記憶體需求更高

### 實驗配置總覽

| 實驗 | Rank | Target Modules | 預估參數量 | 預期記憶體 |
|------|------|----------------|------------|------------|
| Exp1 | 8    | Attention      | ~9M        | ~3.5GB     |
| Exp2 | 8    | Full           | ~14M       | ~4.0GB     |
| Exp3 | 16   | Attention      | ~18M       | ~3.8GB     |
| **Exp4** | **16** | **Full** | **~28M** | **~4.5GB** |
| Exp5 | 32   | Attention      | ~35M       | ~4.2GB     |
| Exp6 | 32   | Full           | ~56M       | ~5.0GB     |

*Exp4 為基準配置，平衡效果與效率*

## 📊 評估指標

### 1. 訓練指標
- **Loss**: 訓練損失與驗證損失
- **Perplexity**: 語言模型困惑度
- **訓練時間**: 每個 epoch 的訓練時間

### 2. 案例評估
- **關鍵字匹配率**: 回應是否包含預期的技術關鍵字
- **類別覆蓋**: 各技術類別（IC 設計、系統架構、演算法等）的評估分數

### 3. 參數效率
- **參數量**: 可訓練參數數量
- **記憶體使用**: 訓練時 GPU 記憶體峰值
- **訓練速度**: 每秒處理的樣本數

## 💡 核心技術說明

### QLoRA 原理

QLoRA = **Q**uantized + **LoRA**

**量化 (Quantization)**:
- 使用 4-bit NF4 (NormalFloat) 量化
- 權重從 FP16 (2 bytes) → NF4 (0.5 bytes)
- 記憶體需求降低 75%

**LoRA (Low-Rank Adaptation)**:
- 不修改原始權重，只添加低秩矩陣
- W' = W + BA，其中 B (out×rank), A (rank×in)
- rank << min(out, in)，參數量大幅減少

### 為什麼能在 8GB GPU 上訓練 3B 模型？

**記憶體分解** (Qwen2.5-3B, batch_size=1, rank=16):

| 項目 | 完整微調 | QLoRA |
|------|----------|-------|
| 模型權重 | 6GB (FP16) | 1.5GB (4-bit) |
| 梯度 | 6GB | 50MB (只有 LoRA) |
| Optimizer 狀態 | 12GB | 100MB (8-bit paged) |
| Activations | 3GB | 1GB (gradient checkpointing) |
| **總計** | **~27GB** | **~3.5GB** |

節省 **87%** 記憶體！

## 🔧 常見問題

### Q: 訓練太慢怎麼辦？

**A**: 調整 `config.yaml`:

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 32  # 增加梯度累積
data:
  max_length: 256  # 縮短序列長度
```

### Q: CUDA Out of Memory

**A**: 降低記憶體需求:

1. 使用較小的 rank (8 instead of 16)
2. 只訓練 Attention 層
3. 啟用 `gradient_checkpointing`
4. 降低 `max_length`

### Q: 如何選擇最適合的配置？

**A**: 決策樹:

```
記憶體充足 (24GB+)
  └─> Exp6 (rank=32, Full)

記憶體一般 (8-12GB)
  ├─ 追求效果 -> Exp4 (rank=16, Full) [推薦]
  └─ 追求速度 -> Exp3 (rank=16, Attention)

記憶體緊張 (<8GB)
  └─> Exp1 (rank=8, Attention)
```

### Q: 可以用 CPU 訓練嗎？

**A**: 理論上可以，但 **強烈不建議**:
- CPU 訓練速度慢 100+ 倍
- QLoRA 需要 bitsandbytes，依賴 CUDA
- 建議使用 Google Colab 免費 GPU

### Q: 如何在自己的數據上訓練？

**A**: 修改 `data_preparation/merge_datasets.py`:

```python
def format_custom_dataset(dataset):
    def format_example(example):
        return {
            "instruction": example['your_question_field'],
            "input": example.get('your_context_field', ''),
            "output": example['your_answer_field']
        }
    return dataset.map(format_example)
```

## 📚 延伸閱讀

### 論文
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### 實作參考
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [Hugging Face TRL](https://github.com/huggingface/trl)
- [Qwen2.5 官方文檔](https://qwenlm.github.io/blog/qwen2.5/)

### 教學資源
- [Hugging Face Course](https://huggingface.co/course)
- [QLoRA 官方 Colab](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k)

## 🤝 共同學習

這是一份共同學習的筆記，歡迎：
- 🐛 發現問題？提 Issue
- 💡 有更好的方法？提 PR
- 🤔 有疑問？在討論區發問

## 📄 授權

MIT License

---

## 🎉 完成檢查清單

訓練完成後，確認以下項目：

- [ ] 資料準備：生成 `final_dataset/`
- [ ] 模型訓練：至少完成一個實驗配置
- [ ] 模型評估：生成評估報告
- [ ] 推論測試：能成功生成工程類問題的回應
- [ ] 綜合報告：生成 `final_report.md`

**準備好了嗎？開始你的 QLoRA 微調之旅！** 🚀

```bash
# 一鍵運行完整流程（需要 6-12 小時）
cd data_preparation && python merge_datasets.py && \
cd ../experiments && python run_all_experiments.py && \
cd ../report_generation && python generate_report.py
```

有問題隨時回來查這份筆記 📖
