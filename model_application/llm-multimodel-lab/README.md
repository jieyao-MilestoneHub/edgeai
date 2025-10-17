# LLM Multimodal Lab

> 從理論到實踐：多模態大語言模型的量化與微調實驗室

## 專案簡介

本專案是一個完整的多模態大語言模型(Vision-Language Models, VLMs)學習路徑,專注於在資源受限的邊緣AI環境下,使用參數高效微調(PEFT)和量化技術來優化模型。

**核心目標**:
- 深入理解量化技術的數學原理(從LoRA到QLoRA)
- 掌握在消費級GPU上微調VLMs的實用技術
- 學習針對特定任務(如文檔理解、視覺問答)的模型優化策略

## 為什麼選擇這個專案?

### 真實場景驅動

你是一位Edge AI工程師,需要在RTX 3070 (8GB VRAM)上微調一個視覺語言模型來處理infographic問答任務。

**挑戰**:
- ✗ 標準7B模型需要14GB+ VRAM
- ✗ 完整微調需要40GB+ VRAM
- ✗ 推論速度無法滿足實時需求

**解決方案**: QLoRA
- ✓ 4-bit量化壓縮到3.5GB
- ✓ LoRA只訓練0.1-1%參數
- ✓ 8GB VRAM完全足夠

### 從數學原理到工程實踐

本專案**不只是**調用HuggingFace API,而是:
- 理解低秩分解的數學原理(`W' = W₀ + BA`)
- 手寫NF4量化算法
- 實作量化感知訓練(QAT)
- 設計混合精度策略

## 專案結構

```
llm-multimodel-lab/
├── docs/                          # 理論文檔與深度解析
│   ├── 00_overview.md             # 總覽:從LoRA到量化的完整技術棧
│   ├── 01_infographicvqa_qlora.md # 第一章:InfographicVQA與QLoRA微調
│   ├── 02_qat_advanced.md         # 第二章:量化感知訓練(QAT)
│   └── 03_mixed_precision.md      # 第三章:混合精度優化
│
├── lab_tasks/                     # 實作任務
│   ├── task01_infographicvqa/     # 任務1:InfographicVQA微調實戰
│   │   ├── README.md
│   │   ├── data_exploration.ipynb
│   │   ├── finetune_pix2struct.py
│   │   ├── finetune_qwen_vl.py
│   │   ├── evaluate.py
│   │   └── requirements.txt
│   │
│   ├── task02_qat/                # 任務2:量化感知訓練
│   └── task03_deployment/         # 任務3:模型部署優化
│
├── scripts/                       # 實用工具腳本
│   ├── memory_profiler.py         # VRAM使用分析
│   ├── model_converter.py         # 模型格式轉換
│   └── benchmark.py               # 性能基準測試
│
└── assets/                        # 資源文件
    ├── images/                    # 圖表與架構圖
    └── datasets/                  # 資料集說明
```

## 學習路徑

### 第0章:理論基礎(必讀)
📖 閱讀 [`docs/00_overview.md`](docs/00_overview.md)

**核心概念**:
- 為什麼需要量化?記憶體/速度/精度的trade-off
- LoRA原理:`W' = W₀ + BA`
- QLoRA技術:NF4量化、雙重量化、分頁優化器
- PTQ vs QAT的差異

**時間**: 1-2小時

---

### 第1章:InfographicVQA微調實戰
📖 閱讀 [`docs/01_infographicvqa_qlora.md`](docs/01_infographicvqa_qlora.md)
💻 實作 [`lab_tasks/task01_infographicvqa/`](lab_tasks/task01_infographicvqa/)

**學習目標**:
1. 理解InfographicVQA資料集的特性與挑戰
2. 選擇適合8GB VRAM的模型(Pix2Struct/Qwen-VL-3B)
3. 掌握QLoRA的配置與調優技巧
4. 實作完整的微調pipeline

**硬體需求**: RTX 3070 8GB (或Google Colab免費T4)

**預期成果**:
- 成功微調出InfographicVQA問答模型
- 理解記憶體優化技巧(gradient checkpointing、mixed precision)
- 掌握ANLS評估指標

---

### 第2章:量化感知訓練(進階)
📖 閱讀 [`docs/02_qat_advanced.md`](docs/02_qat_advanced.md)
💻 實作 [`lab_tasks/task02_qat/`](lab_tasks/task02_qat/)

**學習目標**:
1. 實作Straight-Through Estimator(STE)
2. 理解Fake Quantization機制
3. 對比PTQ vs QAT的精度差異
4. 學習QAT超參數調整

**關鍵洞察**:
> QAT在訓練時模擬量化,讓模型適應低精度,可將INT4精度損失從6.7%降至2.3%

---

### 第3章:混合精度優化(進階)
📖 閱讀 [`docs/03_mixed_precision.md`](docs/03_mixed_precision.md)
💻 實作 [`lab_tasks/task03_deployment/`](lab_tasks/task03_deployment/)

**學習目標**:
1. 敏感度分析:識別對量化敏感的層
2. 設計混合精度策略(不同層不同bit數)
3. 計算SQNR(Signal-to-Quantization-Noise Ratio)
4. 部署優化:模型融合、推論加速

---

## 快速開始

### 環境需求

**硬體**:
- GPU: RTX 3070 8GB (或更高)
- RAM: 32GB+ (推薦)
- 儲存: 50GB+

**軟體**:
- Python 3.9+
- CUDA 11.8+
- PyTorch 2.0+

### 安裝

```bash
# 克隆專案
cd model_application/llm-multimodel-lab

# 安裝task01依賴
cd lab_tasks/task01_infographicvqa
pip install -r requirements.txt

# 測試環境
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import bitsandbytes; print('bitsandbytes OK')"
```

### 第一個實驗(15分鐘)

```bash
# 進入task01
cd lab_tasks/task01_infographicvqa

# 探索資料集(Jupyter Notebook)
jupyter notebook data_exploration.ipynb

# 或直接開始微調Pix2Struct
python finetune_pix2struct.py \
  --model_name google/pix2struct-infographics-vqa-base \
  --output_dir ./outputs/pix2struct \
  --num_epochs 3 \
  --batch_size 4

# 評估模型
python evaluate.py --model_path ./outputs/pix2struct
```

## 核心技術棧

### 量化技術

| 方法 | 精度 | 記憶體節省 | 訓練時間 | 適用場景 |
|------|------|----------|---------|---------|
| **FP16** | Baseline | 0% | 1× | 充足VRAM |
| **INT8 PTQ** | -0.5~2% | 50% | 0 (無需訓練) | 快速部署 |
| **INT4 QLoRA** | -3~7% | 75% | 1.2× | 受限VRAM |
| **INT4 QAT** | -1~3% | 75% | 1.5× | 精度敏感 |
| **Mixed Precision** | -1~2% | 60% | 1.3× | 最佳平衡 |

### 模型選擇指南

**針對InfographicVQA任務(8GB VRAM)**:

1. **google/pix2struct-infographics-vqa-base** (282M)
   - ✓ 專為Infographic設計
   - ✓ QLORA後僅需4-5GB
   - ✓ 已在目標資料集上預訓練

2. **Qwen/Qwen2.5-VL-3B-Instruct** (3B)
   - ✓ 最佳文檔理解能力
   - ✓ 支持32k長上下文
   - ✓ QLORA後約6-8GB

3. **google/paligemma2-3b-pt-896** (3B)
   - ✓ 高解析度(896×896)
   - ✓ 適合細節識別
   - ✓ QLORA後約6-7GB

## 常見問題

### 「8GB VRAM真的夠嗎?」

**夠**,但需要優化技術:
- ✓ 使用4-bit QLoRA量化
- ✓ 啟用gradient checkpointing
- ✓ 使用小batch size(2-4)
- ✓ 使用mixed precision(BF16)

實測:Qwen2.5-VL-3B with QLoRA在8GB上batch=2可穩定訓練

---

### 「沒有GPU能學嗎?」

**理論部分可以**,實作建議使用:
- Google Colab (免費T4, 15GB VRAM)
- Kaggle Notebooks (免費P100)
- RunPod/Vast.ai (按小時租用)

---

### 「和LLM量化有什麼不同?」

VLM量化更複雜:
- **額外的視覺編碼器**:需要處理image encoder量化
- **多模態融合層**:cross-attention層對量化敏感
- **更長的序列**:image tokens增加記憶體壓力

本專案專注於這些VLM特有的挑戰

---

## 參考資源

### 核心論文

1. **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. **QLoRA**: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
3. **InfographicVQA**: [InfographicVQA Dataset](https://arxiv.org/abs/2104.12756)
4. **Pix2Struct**: [Screenshot Parsing as Pretraining](https://arxiv.org/abs/2210.03347)

### 實用工具

- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [Unsloth](https://github.com/unslothai/unsloth) - 更快的QLoRA訓練

---

## 貢獻指南

本專案歡迎貢獻!特別是:
- 新的任務範例(ChartQA、DocVQA等)
- 記憶體優化技巧
- 模型評測結果
- 文檔改進

---

## 授權

MIT License - 詳見 [LICENSE](LICENSE)

---

## 致謝

本專案靈感來自於edge AI實際部署需求,感謝以下開源專案:
- HuggingFace Transformers & PEFT
- TimDettmers/bitsandbytes
- Qwen、Google Research團隊

---

**準備好了嗎?**

從這裡開始 → [`docs/00_overview.md`](docs/00_overview.md)
