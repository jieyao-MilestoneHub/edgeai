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
└── data_preparation/        # 數據準備工具
```
