# Task 01: LoRA 基礎實作 - BERT 情感分類微調

> 一份幫助你順利完成第一次 LoRA 微調的實用筆記

## 🎯 我們要做什麼？

使用 **LoRA (Low-Rank Adaptation)** 微調 BERT 模型，在 **SST-2 情感分類**任務上達到 87-90% 準確率。

**為什麼用 LoRA？**
- 只訓練 **0.27%** 的參數（294K / 109M）
- 訓練速度快 3 倍
- LoRA adapter 只有 **2MB**（完整模型 420MB）
- 可以一個基礎模型配多個任務 adapter

**任務說明：**
- 數據集：GLUE SST-2（電影評論情感分類）
- 訓練樣本：67,349 條評論
- 驗證樣本：872 條評論
- 分類：正面 (1) / 負面 (0)

---

## 📁 核心檔案說明

### **lora_linear.py** - LoRA 核心實作

這是整個 task 最重要的檔案，實作了 LoRA 的數學原理。

#### 🔑 核心元件

**1. `LoRALayer` (第 27-84 行)**
```python
# 實作 y = (α/r) * B(Ax)
lx = F.linear(x, self.lora_A)      # 降維: (*, in) → (*, rank)
lx = F.linear(lx, self.lora_B)     # 升維: (*, rank) → (*, out)
return lx * self.scaling           # 縮放: α/r
```

**為什麼這樣設計？**
- 用兩個小矩陣 A (rank×in) 和 B (out×rank) 近似大矩陣 ΔW (out×in)
- B 初始化為 0 → 初始時 LoRA 輸出為 0，不干擾預訓練權重
- A 用 Kaiming 初始化 → 訓練時逐步學習增量

**關鍵參數：**
- `rank`: 低秩維度，越大能力越強但參數越多（建議 4-16）
- `alpha`: 縮放因子，控制 LoRA 影響力（通常 = 2 × rank）
- `dropout`: 防止過擬合（分類任務建議 0.05）

**2. `LinearWithLoRA` (第 89-146 行)**

把 LoRA 掛在原始 Linear 層上：
```python
output = Linear(x) + LoRA(x)
```

**重要方法：**
- `_freeze_linear()`: 凍結原始權重，只訓練 LoRA
- `merge_weights()`: 部署前合併權重 → W' = W + BA
- `unmerge_weights()`: 恢復分離狀態（續訓時用）

**3. `apply_lora_to_model` (第 180-238 行)**

自動找到模型中的目標層並替換成 LoRA 版本。

```python
apply_lora_to_model(
    model,
    target_modules=["query", "key", "value", "dense"],  # 選哪些層
    rank=8,
    alpha=16.0
)
```

**target_modules 怎麼選？**
- BERT: `query`, `key`, `value`, `dense` (注意力機制)
- GPT-2: `c_attn`, `c_proj`
- LLaMA: `q_proj`, `k_proj`, `v_proj`, `o_proj`

**4. 工具函數 (第 244-313 行)**

```python
mark_only_lora_as_trainable(model)     # 只訓練 LoRA 參數
count_lora_parameters(model)           # 統計參數量
get_lora_state_dict(model)             # 只取 LoRA 權重
load_lora_state_dict(model, state)     # 載入 LoRA 權重
merge_lora_weights(model)              # 合併所有 LoRA 到基礎模型
```

---

### **train_lora_basic.py** - 訓練腳本

完整的訓練流程，從資料載入到模型儲存。

#### 📋 訓練流程

```
1. 載入配置 (config.yaml)
   ↓
2. 載入 BERT + 套用 LoRA
   ↓
3. 凍結 BERT 權重，只訓練 LoRA
   ↓
4. 載入 SST-2 數據集
   ↓
5. 訓練循環 (train → eval → save best)
   ↓
6. 儲存結果 (權重、曲線圖、日誌)
```

#### 🔍 關鍵函數

**`prepare_dataset` (第 54-88 行)**
- 載入 SST-2: `{sentence, label}`
- Tokenize: 轉成 BERT 輸入格式 `{input_ids, attention_mask}`
- 設定 max_length=128（SST-2 句子較短）

**`train_one_epoch` (第 99-162 行)**
- 前向傳播 → 計算 loss
- 反向傳播 → 計算梯度
- 梯度裁剪 → 防止梯度爆炸
- 更新參數 → AdamW optimizer
- 即時顯示 loss 和 accuracy

**`evaluate` (第 165-208 行)**
- 驗證集上評估模型
- 計算準確率（Accuracy）
- 不計算梯度（`torch.no_grad()`）

**`plot_training_curves` (第 211-242 行)**
- 繪製 Loss 和 Accuracy 曲線
- 左圖：訓練/驗證 Loss
- 右圖：訓練/驗證 Accuracy

---

### **inference_example.py** - 推論測試

訓練完成後測試模型效果。

#### 🎮 三種模式

**1. Interactive 互動模式（推薦）**
```bash
python inference_example.py --mode interactive
```
輸入句子即時分析情感，適合體驗模型效果。

**2. Demo 範例模式**
```bash
python inference_example.py --mode demo
```
查看預設範例的分析結果，快速驗證模型。

**3. Text 單次預測**
```bash
python inference_example.py --mode text --text "This is amazing!"
```
分析單一句子，適合整合到其他程式。

#### 💡 載入流程

```python
# 1. 載入 checkpoint
checkpoint = torch.load("output/best_lora_model.pt")

# 2. 載入基礎 BERT 模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 3. 套用 LoRA 結構（與訓練時相同配置）
apply_lora_to_model(model, target_modules=["query", "key", "value", "dense"], rank=8)

# 4. 載入訓練好的 LoRA 權重
load_lora_state_dict(model, checkpoint['lora_state_dict'])
```

---

### **config.yaml** - 配置檔

所有超參數都在這裡調整。

```yaml
# 模型
model_name: "bert-base-uncased"
num_labels: 2

# LoRA 超參數
lora:
  rank: 8              # 低秩維度（影響參數量）
  alpha: 16.0          # 縮放因子（影響學習強度）
  dropout: 0.05        # Dropout 比例
  target_modules:      # 要套 LoRA 的層
    - "query"
    - "key"
    - "value"
    - "dense"

# 訓練超參數
training:
  num_epochs: 3
  batch_size: 16       # GPU 記憶體不足可降到 8
  learning_rate: 3.0e-4
  max_length: 128
  warmup_ratio: 0.1

# 數據集
data:
  dataset: "glue"
  subset: "sst2"
```

---

## 💻 環境需求

**硬體**
- CPU: 4 核心以上（建議）
- RAM: 8GB 以上
- GPU: 2GB VRAM 以上（建議 NVIDIA GPU with CUDA）
- 硬碟: 5GB 可用空間

**軟體**
- Python: 3.11+
- CUDA: 11.8+ (GPU 訓練需要)
- Git: 用於版本控制
- Hugging Face 帳號: 用於上傳模型（可選）

**網路**
- 需要網路連線下載模型和數據集
- 首次執行會下載 BERT (~420MB) 和 SST-2 數據集 (~7MB)

---

## 🚀 快速開始

### 步驟 1: 安裝依賴

```bash
pip install -r requirements.txt
```

環境需求：Python 3.11+、建議使用 GPU

### 步驟 2: 測試安裝

```bash
python test_installation.py
```

看到 `🎉 All tests passed!` 就可以開始了。

### 步驟 3: 開始訓練

```bash
python train_lora_basic.py
```

訓練時間：
- GPU (RTX 3060): 約 10-15 分鐘
- CPU: 約 2-3 小時

### 步驟 4: 測試模型

```bash
python inference_example.py
```

輸入句子測試情感分析效果。

### 步驟 5: 上傳到 Hugging Face (可選)

訓練完成後可以將模型上傳到 Hugging Face Hub 分享。

**前置作業**：
```bash
# 登入 Hugging Face
huggingface-cli login
```

**上傳模型**：
```bash
python train_lora_basic.py \
  --push_to_hub \
  --hub_model_id "your-username/bert-lora-sst2"
```

**參數說明**：
- `--push_to_hub`: 啟用上傳功能
- `--hub_model_id`: 你的 HF 模型 ID（格式：username/model-name）
- `--hub_private`: 設為私有模型（可選）

上傳後可在 `https://huggingface.co/your-username/bert-lora-sst2` 查看。

---

## 📊 理解訓練過程

### 訓練時你會看到這些輸出

```
📊 Parameter Statistics:
  Total parameters:           109,483,778    ← BERT 所有參數
  Trainable parameters:           294,912    ← 只訓練這些（LoRA）
  Frozen parameters:          109,188,866    ← 凍結不動
  Trainable percentage:            0.2694%   ← 不到 0.3%！
```

**這代表什麼？**
- BERT 有 1 億個參數，但我們只訓練 29 萬個
- 節省記憶體、加快訓練、降低過擬合風險

```
📍 Epoch 1/3
Training: 100%|████████| loss: 0.3245, acc: 0.8567
  Train Loss: 0.3421 | Train Acc: 0.8523
  Eval Loss:  0.2987 | Eval Acc:  0.8761
  💾 Best model saved! (eval_acc: 0.8761)
```

**這些數字代表什麼？**
- **Loss 下降**：模型在學習，預測越來越準
- **Accuracy 上升**：分類正確率提高
- **Train vs Eval**：
  - Train 高、Eval 低 → 可能過擬合
  - 兩者接近 → 訓練良好
- **Best model saved**：驗證準確率創新高時自動儲存

### 訓練完成後的輸出檔案

```
output/
├── best_lora_model.pt       # 最佳模型（準確率最高的 epoch）
├── final_lora_model.pt      # 最終模型（第 3 epoch）
├── lora_adapter.pt          # 純 LoRA 權重（僅 2MB）
├── training_curves.png      # Loss/Accuracy 曲線圖
└── training_log.txt         # 詳細訓練日誌
```

**查看 `training_curves.png`** 可以看到：
- 左圖：Loss 應該逐漸下降
- 右圖：Accuracy 應該逐漸上升
- 如果 eval 曲線波動大 → 可能需要降低 learning rate

---

## 🔍 關鍵程式碼解釋

### LoRA 的數學實作

**位置**: `lora_linear.py` 第 73-79 行

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.dropout(x)
    lx = F.linear(x, self.lora_A)      # 第一步：降維到 rank
    lx = F.linear(lx, self.lora_B)     # 第二步：升維回 out_features
    return lx * self.scaling           # 第三步：縮放 (α/r)
```

**為什麼要這樣？**
- 原本更新整個大矩陣 W (d×k) 需要 d×k 個參數
- LoRA 用兩個小矩陣 A (r×k) 和 B (d×r) 只需要 r×(d+k) 個參數
- 當 r << d, k 時，參數量大幅減少

**例如**：
- d = k = 768 (BERT hidden size)
- 完整微調：768 × 768 = 589,824 參數
- LoRA (r=8)：8 × (768+768) = 12,288 參數
- 減少 **48 倍**！

### 如何套用到模型

**位置**: `train_lora_basic.py` 第 283-292 行

```python
apply_lora_to_model(
    model,
    target_modules=config['lora']['target_modules'],
    rank=config['lora']['rank'],
    alpha=config['lora']['alpha'],
    dropout=config['lora']['dropout'],
)

mark_only_lora_as_trainable(model)  # 只訓練 LoRA 參數
```

**做了什麼？**
1. 遍歷模型所有層，找到名稱包含 `query`、`key`、`value`、`dense` 的 Linear 層
2. 用 `LinearWithLoRA` 替換這些層（保留原始權重）
3. 凍結原始 BERT 權重，只開啟 LoRA 參數的 `requires_grad`

**為什麼選這些層？**
- `query`, `key`, `value`: Self-Attention 的核心
- `dense`: Attention 輸出的投影層
- 這些層對任務適應最重要，效果最好

### 訓練循環的核心

**位置**: `train_lora_basic.py` 第 115-157 行

```python
for batch in dataloader:
    # 1. 前向傳播
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    # 2. 反向傳播
    optimizer.zero_grad()
    loss.backward()

    # 3. 梯度裁剪（防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 4. 更新參數
    optimizer.step()
    scheduler.step()
```

**注意**：雖然呼叫 `model.parameters()`，但實際只有 LoRA 參數的梯度非零（因為其他參數被凍結了）。

---

## ❓ 常見問題與調整

### Q: 訓練太慢怎麼辦？

**GPU 上訓練慢**：
```yaml
# config.yaml
training:
  batch_size: 8  # 從 16 降到 8
```

**CPU 上訓練慢**：
```bash
# 先跑 1 個 epoch 測試
python train_lora_basic.py --num_epochs 1
```

### Q: CUDA Out of Memory

**方案 1：降低 batch size**
```yaml
training:
  batch_size: 8  # 或更小 (4)
```

**方案 2：縮短序列長度**
```yaml
training:
  max_length: 64  # 從 128 降到 64
```

### Q: 準確率不理想怎麼辦？

**策略 1：增加 LoRA 容量**
```yaml
lora:
  rank: 16       # 從 8 增加到 16
  alpha: 32.0    # 對應調整（= 2 × rank）
```

**策略 2：調整學習率**
```bash
python train_lora_basic.py --alpha 32  # 命令列覆寫
```
嘗試 2e-4, 5e-4 看哪個效果好。

**策略 3：訓練更久**
```bash
python train_lora_basic.py --num_epochs 5
```

### Q: 如何換成其他數據集？

**修改 config.yaml**：
```yaml
data:
  dataset: "glue"
  subset: "mrpc"      # 改成其他 GLUE 任務
  train_split: "train"
  eval_split: "validation"
```

**調整資料處理**（如果格式不同）：
修改 `train_lora_basic.py` 的 `prepare_dataset` 函數。

### Q: 如何在自己的數據上訓練？

需要修改：
1. `prepare_dataset`: 載入你的數據
2. `tokenize_function`: 根據你的格式調整
3. `num_labels`: 根據你的分類數量

**範例**：
```python
# 載入自己的 CSV 檔案
import pandas as pd
df = pd.read_csv("my_data.csv")
dataset = Dataset.from_pandas(df)
```

---

## 🧪 實驗建議

### 基本實驗（確保理解）

1. **完成一次完整訓練**
   - 觀察參數統計輸出
   - 理解為何只訓練 0.27% 參數

2. **使用互動模式測試**
   - 輸入正面/負面句子
   - 觀察信心分數變化

3. **分析訓練曲線**
   - 打開 `output/training_curves.png`
   - 確認 Loss 下降、Accuracy 上升

### 進階實驗（深入理解）

1. **比較不同 rank 的效果**
   ```bash
   python train_lora_basic.py --rank 4
   python train_lora_basic.py --rank 8
   python train_lora_basic.py --rank 16
   ```
   觀察參數量、訓練時間、準確率的變化。

2. **修改 target_modules**
   ```yaml
   # config.yaml
   lora:
     target_modules:
       - "query"
       - "value"  # 只用 Q 和 V，不用 K
   ```
   看看準確率會下降多少。

3. **實作權重合併**
   ```python
   from lora_linear import merge_lora_weights
   merge_lora_weights(model)
   model.save_pretrained("./merged_bert")
   ```
   合併後的模型可以當標準 BERT 使用（但體積變大）。

---

## 📖 接下來可以學什麼

### 理解更深

**LoRA 原理**
- 📄 論文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- 🧮 為什麼低秩矩陣能有效近似？（矩陣分解理論）
- 💡 為什麼 Attention 層特別適合 LoRA？

**BERT 架構**
- 🏗️ Transformer 的 Self-Attention 機制
- 🔄 BERT 的預訓練任務（MLM, NSP）
- 📊 為什麼微調效果好？

**訓練技巧**
- 📈 Learning Rate Scheduling（為何需要 warmup？）
- 🎯 Gradient Clipping（如何防止梯度爆炸？）
- ⚖️ 如何判斷過擬合？

### 技能擴展

**下一個 Task**
- 🚀 **Task 02: QLoRA** - 4-bit 量化 + LoRA，記憶體再減一半
- 🚀 AdaLoRA - 自適應分配 rank
- 🚀 IA³ - 更激進的參數效率方法

**實際應用**
- 🌐 部署到 API 服務
- 🔄 多任務學習（一個 BERT + 多個 LoRA adapter）
- 💾 LoRA adapter 管理與切換

**工程優化**
- ⚡ Mixed Precision Training (FP16/BF16)
- 🔧 Gradient Checkpointing（節省記憶體）
- 📦 ONNX 導出與推論加速

### 程式能力

**PyTorch 進階**
- 🔨 自定義 `nn.Module`
- 🎨 Hook 機制（監控中間層輸出）
- 💾 Checkpoint 管理與續訓

**Transformers 庫**
- 📚 AutoModel 系列的使用
- 💿 模型儲存/載入的最佳實踐
- 🔧 Tokenizer 的細節

**資料處理**
- 📊 Hugging Face `datasets` 庫
- 🔄 資料增強（Data Augmentation）
- ⚖️ 處理類別不平衡

---

## 📚 延伸閱讀

### 論文
- [LoRA 原始論文](https://arxiv.org/abs/2106.09685)
- [BERT 原始論文](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### 實作參考
- [Hugging Face PEFT 庫](https://github.com/huggingface/peft) - 官方 LoRA 實作
- [Microsoft LoRA 實作](https://github.com/microsoft/LoRA)

### 教學資源
- [Hugging Face Course](https://huggingface.co/course) - Transformers 完整教學
- [PyTorch 官方教學](https://pytorch.org/tutorials/)

---

## 🤝 共同學習

這是一份共同學習的筆記，歡迎：
- 🐛 發現問題？提 Issue
- 💡 有更好的解釋？提 PR
- 🤔 有疑問？在討論區發問

---

## 📄 檔案結構總覽

```
task01_lora_basic/
├── lora_linear.py          # ✨ 核心：LoRA 數學實作
├── train_lora_basic.py     # 🚀 主要：完整訓練流程
├── inference_example.py    # 🎯 應用：推論測試
├── config.yaml             # ⚙️  配置：所有超參數
├── requirements.txt        # 📦 環境：套件清單
├── test_installation.py    # ✅ 檢查：安裝驗證
├── GUIDE.md                # 📖 教學：詳細指引
└── README.md               # 📝 本檔案：學習筆記
```

---

<div align="center">

**準備好了嗎？開始你的第一次 LoRA 微調！** 🚀

```bash
python train_lora_basic.py
```

有問題隨時回來查這份筆記 📖

</div>
