# 資料準備模組

> 自動化下載、過濾、合併工程類指令數據集

## 📋 概述

本模組負責準備訓練數據，包含三個步驟：

1. **下載數據集** (`download_datasets.py`)
2. **過濾工程相關數據** (`filter_engineering.py`)
3. **合併並統一格式** (`merge_datasets.py`)

## 🎯 數據策略

### 通用指令集（~20k 條）
- **來源**: OpenHermes-2.5
- **內容**: 整合 Alpaca、ShareGPT、FLAN 等多源指令
- **目的**: 建立穩定的指令遵循能力

### 技術領域知識集（~5k 條）
- **來源**: ScienceQA + PubMedQA
- **過濾**: 使用關鍵字匹配 engineering 相關內容
- **目的**: 增強技術語境理解能力

## 🚀 使用方式

### 步驟 1: 下載數據集

```bash
cd data_preparation
python download_datasets.py
```

**功能**：
- 自動從 Hugging Face Hub 下載三個數據集
- 使用 `datasets` 庫自動快取
- 顯示數據統計和範例

**輸出**：
- 數據集快取在 `~/.cache/huggingface/datasets/`

### 步驟 2: 過濾工程相關數據

```bash
python filter_engineering.py
```

**功能**：
- 從 ScienceQA 和 PubMedQA 過濾 engineering 相關問題
- 使用關鍵字匹配：engineering, circuit, semiconductor, design, algorithm, optimization, system, architecture
- 控制取樣數量

**輸出**：
- `filtered_data/scienceqa_engineering/` - ScienceQA 過濾結果
- `filtered_data/pubmedqa_engineering/` - PubMedQA 過濾結果

### 步驟 3: 合併數據集

```bash
python merge_datasets.py
```

**功能**：
- 載入所有數據集
- 統一格式為 instruction-following 格式
- 合併並分割訓練/驗證集 (90/10)

**輸出**：
- `final_dataset/` - 最終數據集
  - `train/` - 訓練集
  - `test/` - 驗證集
  - `dataset_stats.txt` - 統計信息

## 📊 數據格式

### 統一格式

所有數據統一為以下格式：

```json
{
    "instruction": "指令或問題",
    "input": "額外輸入（可選，可為空字符串）",
    "output": "期望輸出"
}
```

### 來源格式轉換

**OpenHermes-2.5**:
```json
// 原始格式
{
    "conversations": [
        {"from": "human", "value": "問題"},
        {"from": "gpt", "value": "回答"}
    ]
}

// 轉換後
{
    "instruction": "問題",
    "input": "",
    "output": "回答"
}
```

**ScienceQA**:
```json
// 原始格式
{
    "question": "問題",
    "choices": ["A", "B", "C", "D"],
    "answer": 0,
    "lecture": "相關知識",
    "solution": "解答"
}

// 轉換後
{
    "instruction": "問題\n\nChoices:\nA. ...\nB. ...",
    "input": "相關知識",
    "output": "Answer: A. ...\n\nExplanation: 解答"
}
```

**PubMedQA**:
```json
// 原始格式
{
    "question": "問題",
    "context": {"contexts": ["段落1", "段落2"]},
    "long_answer": "詳細答案",
    "final_decision": "yes"
}

// 轉換後
{
    "instruction": "Based on the provided context, answer: 問題",
    "input": "段落1\n\n段落2",
    "output": "Decision: YES\n\nExplanation: 詳細答案"
}
```

## ⚙️ 配置參數

在 `../config.yaml` 中調整：

```yaml
data:
  # 數據集名稱
  general_dataset: "teknium/OpenHermes-2.5"
  tech_dataset_1: "derek-thomas/ScienceQA"
  tech_dataset_2: "pubmed_qa"

  # 取樣數量
  general_samples: 20000
  tech_samples: 5000

  # 過濾關鍵字
  filter_keywords:
    - "engineering"
    - "circuit"
    - "semiconductor"
    # ...

  # 驗證集比例
  test_size: 0.1
```

## 📁 輸出結構

```
data_preparation/
├── filtered_data/              # 過濾後的數據
│   ├── scienceqa_engineering/
│   └── pubmedqa_engineering/
│
└── final_dataset/              # 最終數據集
    ├── train/
    ├── test/
    └── dataset_stats.txt
```

## 🔍 數據品質檢查

### 檢查點

1. **下載完成性**: 所有數據集都成功下載
2. **過濾率**: Engineering 相關數據的保留率
3. **格式統一性**: 所有數據都包含 instruction, input, output 欄位
4. **內容完整性**: 無空白的 instruction 或 output

### 查看統計

```bash
cat final_dataset/dataset_stats.txt
```

### 查看範例

```python
from datasets import load_from_disk

dataset = load_from_disk("final_dataset")
print(dataset['train'][0])
```

## ⚠️ 常見問題

### Q: 過濾後數據太少？

**A**:
- 調整 `filter_keywords` 增加更多關鍵字
- 或降低 `tech_samples` 數量
- 或直接使用所有 ScienceQA 數據

### Q: 想使用自己的數據？

**A**:
修改 `merge_datasets.py`，添加自定義格式化函數：

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

## 📚 相關資源

- [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA)
- [PubMedQA](https://huggingface.co/datasets/pubmed_qa)
- [Hugging Face Datasets 文檔](https://huggingface.co/docs/datasets/)

## 💡 下一步

完成資料準備後，前往訓練模組：

```bash
cd ../model_training
python train_qwen_instruct.py
```
