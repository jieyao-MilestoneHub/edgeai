# Task 02: QLoRA 實作 - Qwen2.5-3B 語言模型微調

> 一份幫助你在消費級 GPU 上完成大模型微調的實用筆記

## 🎯 我們要做什麼？

使用 **QLoRA (Quantized Low-Rank Adaptation)** 微調 Qwen2.5-3B 模型，在 **Wikitext-2** 語言模型任務上訓練。

**為什麼用 QLoRA？**
- 在 **GTX 4060 (8GB)** 上訓練 3B 參數模型 ✅
- 記憶體需求降低 **75%**（14GB → 3.5GB）
- 只訓練 **0.07%** 的參數（2M / 3B）
- LoRA adapter 只有 **8MB**（完整模型 6GB）
- 訓練速度比全參數快 **5 倍**

**任務說明：**
- 模型：Qwen/Qwen2.5-3B-Instruct（30億參數）
- 數據集：Wikitext-2（語言模型基準）
- 任務：因果語言模型（Causal Language Modeling）
- 量化：4-bit NF4 + Double Quantization

---

## 📁 核心檔案說明

### **quantization_utils.py** - QLoRA 核心工具

這是整個 task 最重要的檔案，實作了 QLoRA 的量化與 LoRA 配置。

#### 🔑 核心元件

**1. `create_bnb_config()` - 量化配置**
```python
# 建立 4-bit NF4 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # 啟用 4-bit
    bnb_4bit_quant_type="nf4",          # NF4 格式
    bnb_4bit_compute_dtype=torch.bfloat16,  # BF16 計算
    bnb_4bit_use_double_quant=True,     # 雙重量化
)
```

**為什麼這樣設計？**
- **NF4 vs FP4**: NF4 針對正態分布優化，精度提升 1.8×
- **BF16 vs FP16**: BF16 範圍大（10³⁸ vs 10⁴），訓練更穩定
- **Double Quant**: 對量化常數再量化，額外節省 8% 記憶體

**2. `create_lora_config()` - LoRA 配置**
```python
# 建立 LoRA 配置（Attention + MLP 層）
lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32.0,         # 縮放因子
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",     # Attention
        "gate_proj", "up_proj", "down_proj"          # MLP
    ]
)
```

**Target Modules 選擇：**
- **Attention 層**（必選）: Q/K/V/O 投影
- **MLP 層**（建議）: Gate/Up/Down 投影，效果提升 10-20%

**3. `load_model_and_tokenizer()` - 載入模型**

自動處理：
- 4-bit 量化載入
- Tokenizer 配置
- Padding token 設定

**4. `prepare_model_for_training()` - 準備訓練**

整合：
- k-bit 訓練準備
- Gradient Checkpointing（節省記憶體 30-50%）
- LoRA 套用

---

### **train_qlora.py** - 訓練腳本

完整的訓練流程，從資料載入到模型儲存。

#### 📋 訓練流程

```
1. 載入配置 (config.yaml)
   ↓
2. 載入 Qwen2.5-3B + 套用 4-bit 量化
   ↓
3. 套用 LoRA (Attention + MLP 層)
   ↓
4. 載入 Wikitext-2 數據集
   ↓
5. 訓練循環 (train → eval → save)
   ↓
6. 儲存結果 (adapter、曲線圖、日誌)
```

#### 🔍 關鍵函數

**`prepare_dataset` (第 80-143 行)**
- 載入 Wikitext-2
- Tokenize: 轉成 input_ids
- 設定 labels（語言模型：labels = input_ids）

**`plot_training_curves` (第 177-241 行)**
- 繪製 Loss 和 Learning Rate 曲線
- 左圖：訓練/驗證 Loss
- 右圖：Learning Rate 變化

**`main` (第 248-527 行)**
- 完整訓練流程
- 記憶體監控
- 結果儲存

---

### **inference_example.py** - 推論測試

訓練完成後測試模型的文本生成能力。

#### 🎮 三種模式

**1. Interactive 互動模式（推薦）**
```bash
python inference_example.py --mode interactive
```
即時輸入 prompt 並生成文本，適合體驗模型效果。

**2. Demo 範例模式**
```bash
python inference_example.py --mode demo
```
查看預設範例的生成結果，快速驗證模型。

**3. Text 單次生成**
```bash
python inference_example.py --mode text --prompt "Explain quantum computing"
```
對單一 prompt 進行生成，適合腳本調用。

#### 💡 載入流程

```python
# 1. 載入量化的基礎模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config  # 4-bit NF4
)

# 2. 載入 LoRA adapter
model = PeftModel.from_pretrained(base_model, "./output_qlora_qwen_3b")

# 3. 生成文本
outputs = model.generate(**inputs, max_new_tokens=128)
```

---

### **config.yaml** - 配置檔

所有超參數都在這裡調整。

```yaml
# 模型
model:
  name: "Qwen/Qwen2.5-3B-Instruct"

# 量化配置
quantization:
  load_in_4bit: true
  quant_type: "nf4"
  compute_dtype: "bfloat16"
  use_double_quant: true

# LoRA 超參數
lora:
  rank: 16                # 低秩維度
  alpha: 32.0             # 縮放因子
  dropout: 0.05           # Dropout 比例
  target_modules:         # 目標層
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

# 訓練超參數
training:
  num_epochs: 3
  batch_size: 1           # GTX 4060 (8GB) 建議 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"
```

---

## 💻 環境需求

**硬體**
- GPU: **8GB VRAM 以上**（GTX 4060 / RTX 3060 / RTX 3070）
- RAM: 16GB 以上（建議 32GB）
- 硬碟: 10GB 可用空間
- CPU: 4 核心以上

**軟體**
- Python: 3.10+
- CUDA: 11.8+ (GPU 訓練需要)
- PyTorch: 2.0+
- bitsandbytes: 0.41.0+

**網路**
- 需要網路連線下載模型和數據集
- 首次執行會下載 Qwen2.5-3B (~6GB) 和 Wikitext-2 (~4MB)

---

## 🚀 快速開始

### 步驟 1: 安裝依賴

```bash
# 先安裝 PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 再安裝其他依賴
pip install -r requirements.txt
```

### 步驟 2: 驗證安裝

```bash
# 檢查 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 檢查 bitsandbytes
python -c "import bitsandbytes; print('BitsAndBytes OK')"
```

### 步驟 3: 開始訓練

```bash
python train_qlora.py
```

**訓練時間：**
- GTX 4060 (8GB): 約 1-2 小時（3 epochs）
- RTX 3090 (24GB): 約 30-45 分鐘
- A100 (40GB): 約 15-20 分鐘

### 步驟 4: 測試模型

```bash
# 互動模式（推薦）
python inference_example.py --mode interactive

# 範例模式
python inference_example.py --mode demo

# 單次生成
python inference_example.py --mode text --prompt "Explain AI in simple terms"
```

---

## 📊 理解訓練過程

### 訓練時你會看到這些輸出

```
📊 Trainable Parameters:
trainable params: 2,097,152 || all params: 3,002,097,152 || trainable%: 0.0698
```

**這代表什麼？**
- Qwen2.5-3B 有 30 億個參數，但我們只訓練 210 萬個（LoRA）
- 訓練參數不到 **0.07%**，記憶體節省 **75%**

```
🖥️  Device Information:
   Device: cuda
   GPU: NVIDIA GeForce GTX 4060
   Total VRAM: 8.00 GB

💾 GPU Memory: Allocated=1.45 GB | Reserved=2.00 GB | Free=6.00 GB
```

**記憶體分配：**
- 量化模型：1.5 GB（14GB → 1.5GB，節省 89%）
- LoRA 參數：50 MB
- Optimizer states：100 MB（8-bit paged）
- Activations：1 GB（with gradient checkpointing）
- **總計：~3 GB**（8GB 顯卡安全）

### 訓練完成後的輸出檔案

```
output_qlora_qwen_3b/
├── adapter_config.json       # LoRA 配置
├── adapter_model.safetensors # LoRA 權重（僅 8MB）
├── training_curves.png       # Loss/LR 曲線圖
└── training_log.txt          # 詳細訓練日誌
```

---

## 🔍 關鍵程式碼解釋

### QLoRA 的量化配置

**位置**: `quantization_utils.py` 第 33-75 行

```python
def create_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,                  # 啟用 4-bit
        bnb_4bit_quant_type="nf4",          # NF4 格式
        bnb_4bit_compute_dtype=torch.bfloat16,  # BF16 計算
        bnb_4bit_use_double_quant=True,     # 雙重量化
    )
```

**為什麼這樣？**
- **NF4 (4-bit NormalFloat)**: 針對神經網絡權重的正態分布優化
  * 權重服從 N(0, σ²)
  * 使用分位數量化，每個區間包含相同數量的數據點
  * 精度比線性 INT4 高 1.8×

- **Double Quantization**: 兩次量化
  * 第一次：權重 (FP16) → 4-bit NF4
  * 第二次：量化常數 (FP32) → INT8
  * 額外節省 8% 記憶體

- **BF16 Compute**: 計算時使用 BFloat16
  * 範圍大（與 FP32 相同）
  * 訓練穩定（不易 NaN）
  * 速度快（硬體支援）

### LoRA 的目標模組選擇

**位置**: `quantization_utils.py` 第 130-137 行

```python
target_modules = [
    # Attention 層（核心，必須包含）
    "q_proj", "k_proj", "v_proj", "o_proj",
    # MLP 層（增強效果，建議包含）
    "gate_proj", "up_proj", "down_proj"
]
```

**為什麼選這些層？**
- **Attention 層** (Q/K/V/O): Self-Attention 的核心，對任務適應最重要
- **MLP 層** (Gate/Up/Down): Feed-Forward Network，增強表達能力
- 實驗顯示：加上 MLP 層效果提升 10-20%

**參數量計算**（Qwen2.5-3B, rank=16）：
```
單層 Attention (4096×4096):
  Q proj: 16 × (4096+4096) = 131K
  K proj: 16 × (4096+4096) = 131K
  V proj: 16 × (4096+4096) = 131K
  O proj: 16 × (4096+4096) = 131K
  小計: 524K

單層 MLP (4096→11008→4096):
  Gate proj: 16 × (4096+11008) = 241K
  Up proj:   16 × (4096+11008) = 241K
  Down proj: 16 × (11008+4096) = 241K
  小計: 723K

全模型（28 層）:
  Attention: 524K × 28 = 14.7M
  MLP: 723K × 28 = 20.2M
  總計: ~35M 參數（完整模型的 1.2%）
```

### 訓練循環的記憶體優化

**位置**: `train_qlora.py` 第 308-312 行

```python
model = prepare_model_for_training(
    model,
    lora_config,
    use_gradient_checkpointing=True  # 啟用梯度檢查點
)
```

**Gradient Checkpointing 原理**：
- 不儲存所有中間激活值
- 反向傳播時重新計算
- 記憶體節省 30-50%
- 訓練速度略降 10-20%
- 權衡：記憶體 vs 速度

**記憶體對比**（Qwen2.5-3B, batch_size=1, seq_len=512）：
```
無 Gradient Checkpointing:
  Activations: ~3 GB
  總記憶體: ~5.5 GB

有 Gradient Checkpointing:
  Activations: ~1 GB
  總記憶體: ~3.5 GB （節省 36%）
```

---

## ❓ 常見問題與調整

### Q: 訓練太慢怎麼辦？

**方案 1：增加 gradient accumulation**
```yaml
# config.yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 32  # 從 16 增加到 32
```

**方案 2：縮短序列長度**
```yaml
data:
  max_length: 256  # 從 512 降到 256
```

### Q: CUDA Out of Memory

**方案 1：降低 batch size（已經是 1 了）**

**方案 2：關閉 MLP 層的 LoRA**
```yaml
lora:
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    # 移除 gate_proj, up_proj, down_proj
```

**方案 3：減小 LoRA rank**
```yaml
lora:
  rank: 8  # 從 16 降到 8
  alpha: 16.0  # 對應調整
```

### Q: 生成效果不理想怎麼辦？

**策略 1：增加 LoRA rank**
```yaml
lora:
  rank: 32      # 從 16 增加到 32
  alpha: 64.0   # 對應調整（= 2 × rank）
```

**策略 2：訓練更久**
```bash
python train_qlora.py --num_epochs 5
```

**策略 3：調整生成參數**
```bash
python inference_example.py \
  --temperature 0.8 \    # 增加隨機性
  --top_p 0.95 \         # 擴大採樣範圍
  --max_new_tokens 256   # 生成更長文本
```

### Q: 如何換成其他模型？

**修改 config.yaml**：
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"  # 改成其他模型
```

**注意事項**：
- 確認模型支援 4-bit 量化
- 調整 target_modules（不同架構層名稱不同）
- 調整記憶體配置（根據模型大小）

### Q: 如何在自己的數據上訓練？

修改 `train_qlora.py` 的 `prepare_dataset` 函數：

```python
def prepare_dataset(config, tokenizer):
    # 載入自己的數據
    dataset = load_dataset("your-dataset-name")
    # 或從本地載入
    # dataset = load_dataset("csv", data_files="your_data.csv")

    def tokenize_function(examples):
        # 根據你的數據格式調整
        return tokenizer(
            examples["text"],  # 改成你的文本欄位名稱
            truncation=True,
            max_length=config['data']['max_length'],
        )

    return dataset.map(tokenize_function, batched=True)
```

---

## 🧪 實驗建議

### 基本實驗（確保理解）

1. **完成一次完整訓練**
   - 觀察記憶體使用情況
   - 理解為何只用 3GB 顯存就能訓練 3B 模型

2. **使用互動模式測試**
   - 輸入不同類型的 prompt
   - 觀察生成質量和多樣性

3. **分析訓練曲線**
   - 打開 `output_qlora_qwen_3b/training_curves.png`
   - 確認 Loss 下降、Learning Rate 正常變化

### 進階實驗（深入理解）

1. **比較不同 rank 的效果**
   ```bash
   python train_qlora.py --rank 8 --alpha 16 --output_dir output_r8
   python train_qlora.py --rank 16 --alpha 32 --output_dir output_r16
   python train_qlora.py --rank 32 --alpha 64 --output_dir output_r32
   ```
   觀察參數量、訓練時間、生成效果的變化。

2. **對比 Attention vs Attention+MLP**

   修改 config.yaml，只訓練 Attention 層：
   ```yaml
   lora:
     target_modules:
       - "q_proj"
       - "k_proj"
       - "v_proj"
       - "o_proj"
   ```

   對比效果差異。

3. **測試不同量化配置**

   修改 `quantization_utils.py`：
   ```python
   # 測試 FP4 vs NF4
   bnb_4bit_quant_type="fp4"  # 改成 fp4

   # 測試關閉雙重量化
   bnb_4bit_use_double_quant=False
   ```

   觀察記憶體和精度變化。

---

## 📖 接下來可以學什麼

### 理解更深

**QLoRA 原理**
- 📄 論文：[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- 🧮 為什麼 4-bit 量化幾乎無精度損失？
- 💡 NF4 如何針對正態分布優化？

**量化技術**
- 📊 Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)
- 🔢 INT8, INT4, NF4, FP4 的差異
- ⚖️ 精度與記憶體的權衡

**訓練技巧**
- 📈 Gradient Checkpointing 的原理
- 🎯 Paged Optimizer 如何降低記憶體峰值
- ⚡ Mixed Precision Training (BF16/FP16)

### 技能擴展

**下一個 Task**
- 🚀 **Task 03**: 更大模型（7B+）的微調
- 🚀 **Task 04**: 指令微調與對話模型
- 🚀 **Task 05**: 多任務學習與 adapter 管理

**實際應用**
- 🌐 部署到生產環境
- 🔄 持續學習與在線微調
- 💾 Adapter 版本管理與切換

**工程優化**
- ⚡ ONNX 導出與推論加速
- 🔧 Flash Attention 整合
- 📦 模型壓縮與蒸餾

### 程式能力

**PyTorch 進階**
- 🔨 自定義 CUDA kernels
- 🎨 分散式訓練（DDP/FSDP）
- 💾 Checkpoint 管理與斷點續訓

**Transformers 庫**
- 📚 Generation 策略（Beam Search, Sampling）
- 💿 模型量化與優化
- 🔧 Custom model 整合

**LLM 工程**
- 📊 Evaluation metrics（BLEU, ROUGE, Perplexity）
- 🔄 資料管道優化
- ⚖️ 模型監控與 A/B 測試

---

## 📚 延伸閱讀

### 論文

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)

### 實作參考

- [Hugging Face PEFT 庫](https://github.com/huggingface/peft) - 官方 LoRA/QLoRA 實作
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 量化核心庫
- [Qwen2.5 技術報告](https://qwenlm.github.io/blog/qwen2.5/)

### 教學資源

- [Hugging Face Course](https://huggingface.co/course) - Transformers 完整教學
- [PyTorch 官方教學](https://pytorch.org/tutorials/)
- [QLoRA 官方 Colab](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k)

---

## 🤝 共同學習

這是一份共同學習的筆記，歡迎：
- 🐛 發現問題？提 Issue
- 💡 有更好的解釋？提 PR
- 🤔 有疑問？在討論區發問

---

## 📄 檔案結構總覽

```
task02_qlora/
├── quantization_utils.py   # ✨ 核心：量化與 LoRA 工具
├── train_qlora.py          # 🚀 主要：完整訓練流程
├── inference_example.py    # 🎯 應用：推論測試（三種模式）
├── config.yaml             # ⚙️  配置：所有超參數
├── requirements.txt        # 📦 環境：套件清單
└── README.md               # 📝 本檔案：學習筆記
```

---

<div align="center">

**準備好了嗎？開始你的第一次 QLoRA 微調！** 🚀

```bash
python train_qlora.py
```

有問題隨時回來查這份筆記 📖

</div>
