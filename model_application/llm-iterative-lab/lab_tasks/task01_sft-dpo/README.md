# Qwen2.5-3B AlignLoop

基於 Qwen2.5-3B 的多路徑訓練框架，支援 SFT、DPO、迭代訓練與閉環訓練。

---

## 專案設計

本框架提供 **4 條訓練路線**，對應不同的訓練需求：

### 路線 A：SFT 單次訓練

**適用場景**：快速指令微調，建立基礎能力

```bash
cd scripts
python run_sft.py --config ../config.yaml
```

**訓練流程**：
```
SFT 數據 → 模型訓練 → 保存模型
```

---

### 路線 B：DPO 單次訓練

**適用場景**：已有 SFT 模型，需要偏好對齊

```bash
python run_dpo.py --config ../config.yaml
```

**訓練流程**：
```
偏好對數據 → DPO 訓練 → 對齊偏好 → 保存模型
```

---

### 路線 C：Iterative 固定迭代訓練

**適用場景**：多輪數據收集與訓練

```bash
python run_iteration.py \
  --config ../config.yaml \
  --override ../configs/iterative/iteration_1.yaml \
  --iterations 3
```

**訓練流程**：
```
┌─────────────────────────────────────┐
│ 迭代 N                               │
│  數據收集 → SFT → DPO → 評估        │
└─────────────────────────────────────┘
           重複 N 次
```

**配置參數**：
```yaml
iterative:
  num_iterations: 3        # 固定迭代次數
  sft_epochs_per_iter: 2   # 每次 SFT epochs
  dpo_epochs_per_iter: 1   # 每次 DPO epochs
```

---

### 路線 D：Hybrid 迭代訓練

**適用場景**：持續數據累積，條件觸發 DPO，自動優化

```bash
python run_hybrid.py \
  --config ../config.yaml \
  --override ../configs/hybrid/default.yaml \
  --preference-data data/preference/example_preference_pairs.jsonl
```

**核心設計**：
- **閉環機制**：數據收集 → SFT → DPO(條件觸發) → 評估 → 繼續
- **條件觸發**：偏好對累積達閾值（如 5 筆）才執行 DPO
- **輕量 DPO**：每次僅訓練 50-100 steps，快速驗證效果
- **自動停止**：達標或達到最大迭代次數即停止

**訓練流程（Closed Loop）**：
```
┌─────────────────────────────────────┐
│ 迭代 1                               │
│  ├─ 數據收集（加載 5 筆偏好對）       │
│  ├─ SFT 訓練（100 steps）            │
│  ├─ 檢查池：5/5 → 觸發 Micro-DPO ✅   │
│  └─ 評估模型                         │
├─────────────────────────────────────┤
│ 迭代 2                               │
│  ├─ 數據收集（再加載 5 筆）           │
│  ├─ SFT 訓練（100 steps）            │
│  ├─ 檢查池：5/5 → 觸發 Micro-DPO ✅   │
│  └─ 評估 → 達標停止 ⏹️                │
└─────────────────────────────────────┘
```

**配置參數**：
```yaml
hybrid:
  num_iterations: 10             # 最大迭代次數
  sft_steps_per_iteration: 100   # 每次迭代的 SFT 步數
  preference_batch_size: 5       # DPO 觸發閾值
  micro_dpo_steps: 50            # 微型 DPO 步數
  micro_dpo_lr: 0.00005          # DPO 學習率
  quality_threshold: 0.7         # 偏好對品質過濾閾值
  evaluation_interval: 1         # 評估間隔
  evaluation_threshold: 0.95     # 達標停止閾值
```

**與 Iterative 的核心差異**：

| 特性 | Iterative (路線 C) | Hybrid (路線 D) |
|------|-------------------|-----------------|
| **DPO 執行** | 每次迭代**必定執行** | **條件觸發**（池>=閾值才執行） |
| **訓練單位** | Epochs（完整遍歷數據集） | Steps（固定步數，更靈活） |
| **DPO 規模** | 完整訓練（1 epoch = 全部數據） | 輕量微調（固定 50 steps） |
| **數據使用** | 批次加載（一次全部） | 增量加載（每次少量） |
| **適用場景** | 數據已備齊，固定流程 | 數據持續增加，靈活調整 |
| **停止條件** | 固定迭代次數 | 達標或最大次數（自動停止） |

**舉例說明**：
- **Iterative**: 你有 20 筆偏好對 → 每次迭代都用這 20 筆訓練 DPO → 重複 3 次
- **Hybrid**: 第 1 天有 5 筆 → 觸發 DPO → 第 2 天又來 5 筆 → 再觸發 DPO → 持續累積

---

## 如何選擇訓練路線？

```
開始
  │
  ▼
是否需要偏好對齊？
  │
  ├─ 否 → 路線 A (SFT)
  │
  └─ 是
      │
      ▼
  已有訓練好的 SFT 模型？
      │
      ├─ 是 → 路線 B (DPO)
      │
      └─ 否
          │
          ▼
      數據是否會持續增加？
          │
          ├─ 否 → 路線 C (Iterative)
          │
          └─ 是 → 路線 D (Hybrid) ⭐
```

---

## 快速開始

### 1. 環境安裝

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 數據準備

**SFT 數據**：
```bash
cd data_preparation
python download_datasets.py
python filter_engineering.py
python merge_datasets.py
```

**DPO 數據格式**：
```json
{"prompt": "問題", "chosen": "好回答", "rejected": "差回答", "confidence": 0.95}
```

### 3. 選擇路線並執行

參考上方「訓練路線說明」選擇對應的路線執行。

---

## 主要配置

編輯 `config.yaml` 設定通用參數：

```yaml
task_type: "hybrid"  # sft / dpo / iterative / hybrid

model:
  name: "Qwen/Qwen2.5-3B"

lora:
  rank: 16
  alpha: 32.0

training:
  num_epochs: 3
  batch_size: 1
  learning_rate: 0.0002
```

各路線特定配置請參考 `configs/` 目錄下的對應文件。

---

## 硬體需求

- **GPU**: 8GB+ VRAM (建議 RTX 3090 / A100)
- **RAM**: 16GB+
- **CUDA**: 11.8+

**記憶體不足處理**：
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 32
lora:
  rank: 8  # 降低 rank
```

---

## 專案結構

```
taskplus_Qwen-AlignLoop/
├── config.yaml              # 主配置
├── scripts/                 # 執行腳本
│   ├── run_sft.py          # 路線 A
│   ├── run_dpo.py          # 路線 B
│   ├── run_iteration.py    # 路線 C
│   └── run_hybrid.py       # 路線 D
├── configs/                 # 各路線配置
│   ├── sft/
│   ├── dpo/
│   ├── iterative/
│   └── hybrid/
├── src/                     # 核心代碼
│   ├── trainers/           # 訓練器
│   ├── data/               # 數據處理
│   ├── config/             # 配置管理
│   └── utils/              # 工具函數
├── data_preparation/        # 數據準備工具
└── evaluation/              # 模型評估系統 ⭐
    ├── test_suite.jsonl              # 測試問題集 (25題)
    ├── metrics.py                    # 自動評估指標
    ├── evaluate_models.py            # 主評估腳本
    ├── visualize_results.py          # 可視化工具
    ├── manual_scoring_template.csv   # 人工評分模板
    └── results/                      # 評估結果輸出
        ├── responses/                # 模型回答
        ├── metrics/                  # 評估指標
        ├── charts/                   # 可視化圖表
        ├── evaluation_summary.md     # 總結報告
        └── case_study.md             # 案例對比分析
```

---

## 模型評估系統 ⭐

完成訓練後，使用評估系統比較 **Pre-trained → SFT → DPO** 三階段模型的效果提升。

### 評估流程

#### 1️⃣ 執行模型評估

評估三個階段的模型在測試集上的表現：

```bash
cd evaluation

# 評估所有三個階段
python evaluate_models.py \
  --test-set test_suite.jsonl \
  --base-model Qwen/Qwen2.5-3B \
  --sft-model ../outputs/sft_model/checkpoint-xxx \
  --dpo-model ../outputs/dpo_model/checkpoint-xxx \
  --output results/ \
  --stages pretrained sft dpo
```

**參數說明**：
- `--base-model`: 基礎預訓練模型路徑
- `--sft-model`: SFT 訓練後的 LoRA adapter 路徑
- `--dpo-model`: DPO 訓練後的 LoRA adapter 路徑
- `--stages`: 要評估的階段（可選擇部分階段）

**輸出結果**：
- `results/responses/` - 每個階段的模型回答
- `results/metrics/` - 各項評估指標（JSON 格式）

---

#### 2️⃣ 生成可視化報告

根據評估結果自動生成圖表和分析報告：

```bash
python visualize_results.py \
  --results-dir results/ \
  --num-cases 5
```

**生成內容**：
1. **柱狀圖** (`charts/metrics_comparison_bar.png`)
   - 比較各階段在不同指標上的得分

2. **雷達圖** (`charts/metrics_comparison_radar.png`)
   - 多維度能力分布可視化

3. **提升曲線** (`charts/improvement_curve.png`)
   - 展示模型能力的迭代提升趨勢

4. **案例對比分析** (`case_study.md`)
   - 挑選 5 個改進最明顯的問答對比

5. **總結報告** (`evaluation_summary.md`)
   - 整體評估結果與改進幅度統計

---

#### 3️⃣ 人工評分（可選）

如果需要更精確的人工偏好評估：

1. 填寫評分模板：
   ```bash
   # 在 Excel 或文字編輯器中打開
   evaluation/manual_scoring_template.csv
   ```

2. 評分標準（1-5 分）：
   - **1** - 很差（完全不符合期望）
   - **2** - 差（有嚴重問題）
   - **3** - 普通（基本可用）
   - **4** - 好（符合期望）
   - **5** - 很好（超出期望）

3. 填寫完成後，可以將人工評分結果與自動指標結合分析

---

### 評估指標說明

#### 自動指標

| 指標 | 說明 | 計算方式 |
|------|------|---------|
| **關鍵詞覆蓋率** | 回答是否包含期望的關鍵詞 | 匹配關鍵詞數 / 總關鍵詞數 |
| **格式正確性** | 回答格式是否符合要求 | 根據期望格式（代碼、列表等）評分 |
| **長度適當性** | 回答長度是否合理 | 根據問題類別判斷是否過短/過長 |
| **結構分數** | 回答是否有清晰的結構 | 檢查段落、標點、結構性元素 |
| **拒答率** | 模型是否過度保守拒絕回答 | 檢測拒答關鍵詞 |
| **代碼品質** | 代碼回答的品質（僅針對 coding 類問題） | 檢查函數定義、註釋、結構等 |

#### 總分計算

總分為以上指標的加權平均（0-1 分），權重如下：
- 關鍵詞覆蓋率: 30%
- 格式正確性: 25%
- 結構分數: 20%
- 長度適當性: 15%
- 代碼品質: 10%（僅 coding 類）

---

### 測試集內容

`test_suite.jsonl` 包含 **25 個精選測試問題**，涵蓋多種類型：

| 類別 | 數量 | 說明 |
|------|------|------|
| **factual** | 2 | 事實性知識問答 |
| **coding** | 2 | 代碼編寫任務 |
| **reasoning** | 2 | 推理與分析 |
| **creative** | 2 | 創意生成 |
| **technical** | 2 | 技術深度解析 |
| **comparison** | 2 | 對比分析 |
| **debug** | 2 | 故障排除 |
| **application** | 2 | 實際應用場景 |
| **math** | 2 | 數學原理 |
| **safety** | 2 | AI 安全與倫理 |
| **practice** | 2 | 實務建議 |
| **edge_case** | 2 | 邊界情況測試 |
| **instruction_following** | 1 | 指令遵循能力 |

---

### 快速開始示例

完整的評估工作流程：

```bash
# 1. 確保已完成訓練
cd lab_tasks/task01_sft-dpo

# 2. 執行評估（假設已有訓練好的模型）
cd evaluation
python evaluate_models.py \
  --base-model Qwen/Qwen2.5-3B \
  --sft-model ../outputs/sft_final \
  --dpo-model ../outputs/dpo_final

# 3. 生成可視化報告
python visualize_results.py

# 4. 查看結果
# - 圖表: results/charts/*.png
# - 總結: results/evaluation_summary.md
# - 案例: results/case_study.md
```

---

### 預期改進效果

典型的三階段模型評估結果示例：

| 階段 | 關鍵詞覆蓋率 | 格式正確性 | 總分 | 提升幅度 |
|------|------------|-----------|------|---------|
| **Pre-trained** | 0.650 | 0.600 | 0.625 | - |
| **SFT** | 0.820 | 0.780 | 0.800 | +28.0% |
| **DPO** | 0.870 | 0.850 | 0.860 | +7.5% |

**觀察重點**：
- SFT 階段通常帶來**最大幅度提升**（適應任務格式）
- DPO 階段進行**精細調優**（對齊人類偏好）
- 結合可視化圖表可清楚看出進化曲線

---
