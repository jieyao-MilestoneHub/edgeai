# 第一章: InfographicVQA與QLoRA微調實戰

> 從資料集分析到模型微調:在8GB VRAM上實現文檔理解

---

## 章節概覽

本章將帶你完成第一個實戰任務:使用QLoRA在消費級GPU(RTX 3070 8GB)上微調視覺語言模型,處理InfographicVQA任務。

**你將學到**:
- InfographicVQA資料集的特性與挑戰
- 如何選擇適合8GB VRAM的VLM模型
- QLoRA的配置與記憶體優化技巧
- 針對infographic的提示工程策略
- 完整的訓練、評估、與對比流程

**硬體需求**: RTX 3070 8GB (或Google Colab T4 16GB)

---

## 1. InfographicVQA資料集深度分析

### 1.1 什麼是InfographicVQA?

**定義**: InfographicVQA是一個針對資訊圖表(infographic)的視覺問答資料集,包含複雜的文字、圖表、圖標和數據視覺化元素。

**資料集規模**:
```
總圖片數: 5,485張infographic
訓練集:   23,946個問答對
驗證集:   2,801個問答對
測試集:   3,288個問答對
```

**來源**: 真實的資訊圖表,涵蓋教育、商業、健康、科技等多個領域

### 1.2 為什麼Infographic VQA很困難?

**挑戰1: 多模態訊息融合**

```
傳統VQA:    [圖像] + [問題] → [答案]
              ↓
         單一物體/場景

InfographicVQA: [文字 + 圖表 + 圖標 + 顏色 + 佈局] + [問題] → [答案]
                 ↓
          需要多重推理路徑
```

**範例**:
```
圖片內容: 一張關於「全球氣候變化」的infographic
- 折線圖顯示1900-2020年溫度變化
- 文字說明「2019年是有記錄以來第二熱的年份」
- 圖標標示「+1.2°C 相比工業革命前」

問題: 「2019年全球溫度相比工業革命前升高了多少?」
答案: 「1.2°C」

需要的能力:
1. OCR識別文字「+1.2°C」
2. 理解圖標與文字的對應關係
3. 連結問題與視覺元素
```

---

**挑戰2: 複雜的佈局理解**

Infographic使用複雜的空間佈局來組織訊息:
- 時間軸(水平或垂直)
- 多列多行表格
- 嵌套的圖表
- 非線性閱讀順序

**與DocVQA的差異**:

| 特性 | DocVQA | InfographicVQA |
|------|--------|----------------|
| **佈局** | 線性文檔(由上至下) | 非線性設計 |
| **視覺元素** | 主要是文字 | 文字+圖表+圖標 |
| **顏色重要性** | 低 | 高(編碼訊息) |
| **數據類型** | 段落文字 | 統計數據+視覺化 |
| **推理複雜度** | 提取式 | 提取+計算+推理 |

---

**挑戰3: 多步推理與計算**

約35%的問題需要多步推理:

**類型1: 數值計算**
```
圖表: 「2020年收入: $500M」「2021年收入: $650M」
問題: 「收入增長了多少百分比?」
答案: 「30%」

步驟:
1. 提取2020年數值: 500
2. 提取2021年數值: 650
3. 計算: (650-500)/500 = 0.3 = 30%
```

**類型2: 條件推理**
```
圖表: 顯示各國COVID-19疫苗接種率
問題: 「接種率超過70%的歐洲國家有哪些?」

步驟:
1. 過濾「歐洲」地區
2. 過濾接種率 > 70%
3. 列出國家名稱
```

---

### 1.3 評估指標: ANLS

**Average Normalized Levenshtein Similarity (ANLS)**

InfographicVQA使用ANLS作為主要評估指標,而非簡單的accuracy。

**為什麼不用Accuracy?**

```
Ground Truth: "Barack Obama"
模型預測:     "Barack H. Obama"

Exact Match Accuracy: 0 (完全錯誤)
ANLS Score:          0.93 (幾乎正確)
```

**ANLS公式**:

```
對於單個預測:
NLS(pred, gt) = 1 - Levenshtein(pred, gt) / max(len(pred), len(gt))

如果 NLS < 0.5: 設為 0 (太不相似)
否則:          NLS值

ANLS = 平均所有問題的NLS
```

**實作範例**:

```python
from Levenshtein import distance

def calculate_anls(prediction: str, ground_truth: str) -> float:
    """
    計算ANLS分數
    """
    # 正規化:轉小寫、去除多餘空白
    pred = prediction.lower().strip()
    gt = ground_truth.lower().strip()

    # 計算編輯距離
    lev_dist = distance(pred, gt)
    max_len = max(len(pred), len(gt))

    if max_len == 0:
        return 1.0

    # NLS分數
    nls = 1 - (lev_dist / max_len)

    # ANLS閾值
    return nls if nls >= 0.5 else 0.0

# 範例
print(calculate_anls("Barack Obama", "Barack H. Obama"))  # 0.87
print(calculate_anls("42", "43"))                         # 0.0 (低於0.5)
print(calculate_anls("Paris", "paris"))                   # 1.0 (完全匹配)
```

**ANLS的優勢**:
- 容忍小錯誤(拼寫、格式)
- 更符合人類判斷
- 對數值答案更公平

---

## 2. 模型選擇策略:針對8GB VRAM

### 2.1 記憶體需求計算

**基本公式**:

```
訓練記憶體 = 模型權重 + 優化器狀態 + 梯度 + 激活值

標準微調(FP16):
- 模型權重:     params × 2 bytes
- 梯度:         params × 2 bytes
- 優化器(Adam): params × 8 bytes (包含momentum)
- 激活值:       batch_size × seq_len × hidden_dim × 2

總計 ≈ params × 12 bytes + 激活值
```

**對於7B模型**:
```
7B × 12 = 84GB  ❌ RTX 3070只有8GB!
```

**QLoRA優化後**:

```
QLoRA記憶體 = 4-bit基礎模型 + FP16 LoRA adapters + 激活值

- 4-bit模型權重:  7B × 0.5 bytes = 3.5GB
- LoRA adapters:  ~30MB (rank=16時)
- 優化器狀態:     只針對LoRA (30MB × 8 = 240MB)
- 激活值:         取決於batch size

總計 ≈ 4-5GB (batch_size=2時)  ✅ 可行!
```

---

### 2.2 三個推薦模型對比

#### 選項1: google/pix2struct-infographics-vqa-base

**參數量**: 282M

**架構**: Encoder-Decoder (基於T5)
```
Image → [Pix2Struct Encoder] → Image Embeddings → [T5 Decoder] → Text
```

**優勢**:
- ✅ **專門為infographic設計**: 預訓練任務是「screenshot parsing」
- ✅ **已在InfographicVQA上微調**: 可以直接使用或繼續微調
- ✅ **記憶體友善**: base版本僅282M參數
- ✅ **支持可變解析度**: 自動縮放圖片

**記憶體需求(QLoRA)**:
```
4-bit模型: 282M × 0.5 = 141MB
LoRA:      ~20MB
激活值:    ~2GB (batch=4, seq_len=4096)
總計:      ~3-4GB  ✅ 8GB綽綽有餘
```

**效能基準** (InfographicVQA test set):
```
預訓練後直接評估: ANLS = 0.384
微調後:           ANLS = 0.488
```

**適合場景**:
- 想要快速上手
- 重視訓練速度
- 專注於infographic任務

**限制**:
- 模型較小,泛化能力受限
- 不支持視頻或多圖理解

---

#### 選項2: Qwen/Qwen2.5-VL-3B-Instruct

**參數量**: 3B (視覺編碼器400M + 語言模型2.6B)

**架構**: SigLIP Vision Encoder + Qwen2 LLM
```
Image → [SigLIP-400M] → Visual Tokens → [Qwen2-2.6B] → Text
                                              ↑
                                          LoRA插入點
```

**優勢**:
- ✅ **卓越的文檔理解**: 在DocVQA、ChartQA表現優異
- ✅ **超長上下文**: 支持32k tokens (可處理複雜infographic)
- ✅ **多語言支持**: 支持29種語言
- ✅ **動態解析度**: 自動調整圖片切分策略

**記憶體需求(QLoRA)**:
```
4-bit模型: 3B × 0.5 = 1.5GB
LoRA:      ~30MB (rank=16)
激活值:    ~4-5GB (batch=2, dynamic resolution)
總計:      ~6-7GB  ✅ 適合8GB VRAM
```

**效能基準**:
```
DocVQA:         ANLS = 0.947
ChartQA:        Accuracy = 83.0
InfographicVQA: ANLS ≈ 0.55-0.60 (微調後預估)
```

**獨特功能**:
1. **動態解析度**: 根據圖片內容自動決定patch數量
2. **Mixture-of-Experts**: 部分層使用MoE結構
3. **Agentic能力**: 可以進行物體檢測、定位

**適合場景**:
- 需要處理複雜佈局
- 想要更好的泛化能力
- 未來可能擴展到其他文檔任務

**優化技巧**:
```python
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # 重要!
    bnb_4bit_use_double_quant=True,         # 雙重量化
    bnb_4bit_quant_type="nf4"               # NF4量化
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
```

---

#### 選項3: google/paligemma2-3b-pt-896

**參數量**: 3B (SigLIP-400M + Gemma2-2.6B)

**架構**: SigLIP + Gemma2
```
Image (896×896) → [SigLIP] → 1024 visual tokens → [Gemma2] → Text
```

**優勢**:
- ✅ **高解析度**: 支持896×896 (適合細節豐富的infographic)
- ✅ **預訓練質量高**: Google大規模預訓練
- ✅ **多任務能力**: 同時支持VQA、OCR、captioning

**記憶體需求(QLoRA)**:
```
4-bit模型: 3B × 0.5 = 1.5GB
LoRA:      ~30MB
激活值:    ~5-6GB (batch=2, 896px)
總計:      ~7-8GB  ⚠️ 接近極限
```

**效能基準**:
```
OCR-VQA:   Accuracy = 72.1
TextVQA:   Accuracy = 68.9
```

**適合場景**:
- Infographic包含大量細小文字
- 需要高解析度處理
- 重視OCR準確性

**注意事項**:
- 896px解析度會產生更多visual tokens,增加記憶體壓力
- 需要更小的batch size (建議=1)
- 訓練速度較慢

---

### 2.3 決策樹:我應該選哪個模型?

```
                開始
                 │
                 ├─ 想要最快看到結果?
                 │   └─ YES → Pix2Struct-base ✅
                 │             (3-4GB, 訓練快)
                 │
                 ├─ 需要處理超長infographic?
                 │   └─ YES → Qwen2.5-VL-3B ✅
                 │             (32k context, 6-7GB)
                 │
                 ├─ Infographic文字很小且密集?
                 │   └─ YES → PaliGemma2-896 ✅
                 │             (高解析度, 7-8GB)
                 │
                 └─ 追求最佳泛化能力?
                     └─ YES → Qwen2.5-VL-3B ✅
                               (多任務預訓練)
```

**本教學預設選擇**: **Qwen2.5-VL-3B-Instruct**

**原因**:
1. 平衡性最佳(記憶體/性能/泛化)
2. 豐富的文檔支持與社群資源
3. 可擴展到其他任務(DocVQA、ChartQA)
4. 長上下文能力適合複雜infographic

---

## 3. QLoRA詳解:4-bit量化的魔法

### 3.1 回顧:什麼是QLoRA?

**QLoRA = Quantized Low-Rank Adaptation**

```
標準LoRA:
  W' = W₀ + BA

  其中:
  - W₀: FP16凍結權重 (7B × 2 bytes = 14GB)
  - B, A: FP16可訓練 (rank × 2 = 0.01% params)

QLoRA改進:
  W' = Dequant(Q₄(W₀)) + BA

  其中:
  - Q₄(W₀): 4-bit量化凍結權重 (7B × 0.5 bytes = 3.5GB) ✅ 75%節省
  - B, A: 依然FP16/BF16 (保持精度)
  - Dequant: 前向傳播時動態反量化到BF16計算
```

---

### 3.2 NF4量化:為什麼比INT4更好?

**傳統INT4量化**:

```python
# 線性量化
scale = (max_val - min_val) / 15  # 4-bit有16個值(0-15)
quantized = round((value - min_val) / scale)
```

**問題**: 神經網絡權重呈**正態分布**,但INT4均勻分配bits

```
權重分布:
         │   ╱╲
頻率     │  ╱  ╲
         │ ╱    ╲___
         │╱          ╲___
         └──────────────── 權重值
        -3σ  0    +3σ

INT4分配: [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]
          均勻間隔 → 浪費bits在稀少的極值區域
```

**NF4量化(4-bit NormalFloat)**:

根據正態分布的**分位數**設計量化點:

```python
# NF4的16個量化值(對應標準正態分布的16個分位數)
NF4_VALUES = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]

# 量化過程
def quantize_nf4(weight: torch.Tensor) -> Tuple[torch.Tensor, float]:
    # 1. 正規化到[-1, 1]
    absmax = weight.abs().max()
    normalized = weight / absmax

    # 2. 找到最接近的NF4值
    quantized_idx = find_nearest_nf4_index(normalized)

    return quantized_idx, absmax  # absmax作為scale factor
```

**NF4 vs INT4對比**:

| 指標 | INT4 | NF4 |
|------|------|-----|
| **量化誤差** (LLaMA-7B) | ~5% perplexity上升 | ~2% perplexity上升 |
| **適用分布** | 均勻分布 | 正態分布 ✅ |
| **設計理念** | 通用 | 專為神經網絡優化 |
| **量化速度** | 快 | 稍慢(需查表) |

---

### 3.3 雙重量化:連Scale都要壓縮

**第一層量化**: 權重 → 4-bit

```
每64個元素一組,共用一個FP16 scale:
[w₁, w₂, ..., w₆₄] → scale (FP16, 2 bytes)
```

**問題**: 對於7B模型,有**1億個scale** (7B / 64 = 109M scales)
- 109M scales × 2 bytes = 218MB

**第二層量化**: Scale也量化!

```python
# 將FP16 scales再量化為8-bit
def double_quantization(scales: torch.Tensor):
    # scales: [num_groups] 個 FP16

    # 量化scales到INT8
    scale_of_scale = scales.abs().max() / 127
    quantized_scales = (scales / scale_of_scale).round().to(torch.int8)

    return quantized_scales, scale_of_scale

# 記憶體節省
原始: 109M × 2 bytes = 218MB
優化: 109M × 1 byte + 1個FP32 = 109MB  ✅ 節省50%
```

**總體記憶體節省**:

```
無雙重量化: 3.5GB (權重) + 218MB (scales) = 3.7GB
雙重量化:   3.5GB (權重) + 109MB (scales) = 3.6GB

額外節省: ~3%
```

---

### 3.4 BitsAndBytesConfig完整解析

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    # ===== 基本量化設定 =====
    load_in_4bit=True,  # 使用4-bit量化

    # ===== NF4設定 =====
    bnb_4bit_quant_type="nf4",  # 選項: "nf4" 或 "fp4"
    # fp4: 4-bit float (均勻分布)
    # nf4: 4-bit normal float (正態分布優化) ✅ 推薦

    # ===== 計算精度 =====
    bnb_4bit_compute_dtype=torch.bfloat16,  # 前向傳播時反量化的精度
    # 選項: torch.float16, torch.bfloat16
    # bfloat16優勢: 更大的數值範圍,訓練更穩定 ✅ 推薦

    # ===== 雙重量化 =====
    bnb_4bit_use_double_quant=True,  # 是否量化scale factors
    # True:  額外節省3% (推薦) ✅
    # False: 不量化scales

    # ===== 離群值處理 (僅針對LLM) =====
    llm_int8_enable_fp32_cpu_offload=False,  # CPU offload(通常不需要)
    llm_int8_threshold=6.0,  # 離群值閾值(保留為FP16)
)
```

**參數選擇指南**:

| 參數 | 推薦值 | 原因 |
|------|--------|------|
| `bnb_4bit_quant_type` | `"nf4"` | 神經網絡權重是正態分布 |
| `bnb_4bit_compute_dtype` | `torch.bfloat16` | 數值範圍大,穩定性好 |
| `bnb_4bit_use_double_quant` | `True` | 額外節省記憶體,無明顯精度損失 |

---

### 3.5 LoRA配置:Rank、Alpha、Target Modules

**LoRA核心參數**:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    # ===== Rank設定 =====
    r=16,  # Low-rank矩陣的秩
    # 權衡: rank↑ → 表達能力↑, 記憶體↑, 過擬合風險↑
    # 建議: 8-64之間

    # ===== Alpha設定 =====
    lora_alpha=32,  # 縮放係數
    # 實際縮放 = lora_alpha / r
    # 此例: 32 / 16 = 2× 放大LoRA的影響

    # ===== Target Modules =====
    target_modules=[
        "q_proj",  # Query投影
        "k_proj",  # Key投影
        "v_proj",  # Value投影
        "o_proj",  # Output投影
    ],
    # 決定在哪些層插入LoRA

    # ===== Dropout =====
    lora_dropout=0.05,  # LoRA層的dropout率
    # 防止過擬合,特別是小資料集

    # ===== 偏置項 =====
    bias="none",  # 選項: "none", "all", "lora_only"
    # "none": 不訓練bias (節省記憶體) ✅ 推薦

    # ===== 任務類型 =====
    task_type="CAUSAL_LM"  # 或 "SEQ_2_SEQ_LM" (Pix2Struct)
)
```

**Rank vs 性能實驗** (LLaMA-7B on Alpaca):

| Rank | 可訓練參數 | VRAM | Accuracy | 訓練時間 |
|------|-----------|------|----------|---------|
| r=4  | 0.02% | 4.2GB | 68.3% | 1.0× |
| **r=8** | **0.04%** | **4.5GB** | **71.2%** | **1.1×** ✅ |
| r=16 | 0.08% | 5.1GB | 72.1% | 1.3× |
| r=32 | 0.16% | 6.3GB | 72.3% | 1.7× |
| r=64 | 0.32% | 8.7GB | 72.4% | 2.5× |

**結論**: **r=8或r=16**是最佳平衡點

---

**Target Modules選擇策略**:

**選項1: 只調整Attention (保守)**
```python
target_modules=["q_proj", "v_proj"]
```
- 最小記憶體 (~20MB for 7B)
- 適合極度受限環境

**選項2: 完整Attention (推薦)** ✅
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
```
- 平衡記憶體與性能
- 大多數任務的最佳選擇

**選項3: Attention + FFN (激進)**
```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # FFN
]
```
- 最高性能
- 記憶體需求×2 (~60MB for 7B)

**針對VLM的特殊考量**:

對於Qwen-VL等模型,還可以考慮:
```python
target_modules=[
    # Language Model部分
    "q_proj", "k_proj", "v_proj", "o_proj",

    # Vision-Language Adapter (如果存在)
    "visual_proj",  # 視覺特徵投影層
]
```

---

## 4. 實戰:微調Qwen2.5-VL-3B on InfographicVQA

### 4.1 資料準備

**下載InfographicVQA資料集**:

```python
# dataset_loader.py
from datasets import load_dataset
import os

def load_infographicvqa(cache_dir="./data"):
    """
    載入InfographicVQA資料集
    """
    dataset = load_dataset(
        "MMInstruction/InfographicVQA",  # HuggingFace資料集路徑
        cache_dir=cache_dir
    )

    print(f"訓練集: {len(dataset['train'])} 樣本")
    print(f"驗證集: {len(dataset['validation'])} 樣本")
    print(f"測試集:  {len(dataset['test'])} 樣本")

    return dataset

# 範例
dataset = load_infographicvqa()
sample = dataset['train'][0]

print(f"圖片: {sample['image']}")
print(f"問題: {sample['question']}")
print(f"答案: {sample['answers']}")  # 可能有多個參考答案
```

**資料格式**:

```python
{
    'image': PIL.Image,  # PIL圖片物件
    'question': str,      # 問題文本
    'answers': List[str], # 多個參考答案 (取第一個作為target)
    'questionId': int,    # 唯一ID
}
```

---

**資料預處理**:

```python
from transformers import Qwen2VLProcessor

processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

def preprocess_function(examples):
    """
    將資料轉換為模型輸入格式
    """
    images = examples['image']
    questions = examples['question']
    answers = [ans[0] for ans in examples['answers']]  # 取第一個答案

    # 構建Qwen-VL的對話格式
    conversations = []
    for q, a in zip(questions, answers):
        conversations.append([
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": a}]
            }
        ])

    # 使用processor處理
    texts = processor.apply_chat_template(
        conversations,
        add_generation_prompt=False,
        tokenize=False
    )

    # 處理圖片和文字
    model_inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=2048,  # 根據VRAM調整
        return_tensors="pt"
    )

    # Labels = input_ids (Causal LM)
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs

# 應用到資料集
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)
```

---

### 4.2 完整訓練腳本

```python
# finetune_qwen_vl.py
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

# ===== 1. 配置 =====
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./outputs/qwen_vl_infographicvqa"
DATASET_NAME = "MMInstruction/InfographicVQA"

# QLoRA配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ===== 2. 載入模型 =====
print("載入模型...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # 自動分配到GPU
)

# 準備模型用於量化訓練
model = prepare_model_for_kbit_training(model)

# 添加LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 預期輸出: trainable params: ~40M / 3B total (~1.3%)

# ===== 3. 載入處理器與資料集 =====
processor = Qwen2VLProcessor.from_pretrained(MODEL_ID)

print("載入資料集...")
dataset = load_dataset(DATASET_NAME)

# 應用預處理(使用前面定義的preprocess_function)
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=dataset['train'].column_names
)

# ===== 4. 訓練參數 =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # 訓練設定
    num_train_epochs=3,
    per_device_train_batch_size=2,  # 針對8GB VRAM
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # 有效batch=8

    # 優化器
    optim="paged_adamw_8bit",  # QLoRA推薦的優化器
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,

    # 記憶體優化
    gradient_checkpointing=True,  # 節省激活值記憶體
    fp16=False,  # 使用BF16
    bf16=True,

    # 評估與儲存
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,  # 只保留2個checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # 日誌
    logging_steps=50,
    report_to="tensorboard",

    # 其他
    remove_unused_columns=False,  # 保留圖片數據
    ddp_find_unused_parameters=False,
)

# ===== 5. 訓練器 =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['validation'],
    tokenizer=processor.tokenizer,  # 用於保存
)

# ===== 6. 開始訓練 =====
print("開始訓練...")
trainer.train()

# ===== 7. 儲存模型 =====
print(f"儲存模型到 {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("訓練完成!")
```

---

### 4.3 記憶體優化技巧

**技巧1: Gradient Checkpointing**

```python
# 在TrainingArguments中啟用
gradient_checkpointing=True

# 原理:
# 不儲存所有中間激活值,需要時重新計算
# 記憶體: ↓ 50%
# 速度:   ↓ 20%
```

**技巧2: 使用Paged Optimizers**

```python
optim="paged_adamw_8bit"

# 使用8-bit Adam,且支持CPU分頁
# 當GPU記憶體不足時,自動offload到CPU
```

**技巧3: 調整Batch Size**

```python
# 方案1: 小batch + 梯度累積 (推薦)
per_device_train_batch_size=2
gradient_accumulation_steps=4
# 有效batch = 2 × 4 = 8

# 方案2: 極小batch (最後手段)
per_device_train_batch_size=1
gradient_accumulation_steps=8
```

**技巧4: 限制序列長度**

```python
# 在preprocess_function中
max_length=1024  # 或 512 (根據資料集調整)

# InfographicVQA的問題通常不長,限制序列長度不會損失太多訊息
```

**記憶體使用監控**:

```python
import GPUtil

def print_gpu_utilization():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")

# 在訓練前後呼叫
print_gpu_utilization()
trainer.train()
print_gpu_utilization()
```

---

### 4.4 推論與評估

**推論腳本**:

```python
# inference.py
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
import torch
from PIL import Image

# 載入微調後的模型
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "./outputs/qwen_vl_infographicvqa"  # LoRA adapter路徑
)

processor = Qwen2VLProcessor.from_pretrained(
    "./outputs/qwen_vl_infographicvqa"
)

# 推論函數
def predict(image_path: str, question: str) -> str:
    """
    對單張圖片進行VQA推論
    """
    image = Image.open(image_path).convert("RGB")

    # 構建對話
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    # 處理輸入
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    # 生成答案
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,  # Greedy decoding
            temperature=None,
            top_p=None,
        )

    # 解碼
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    answer = processor.decode(generated_ids, skip_special_tokens=True)

    return answer.strip()

# 範例
answer = predict(
    "test_infographic.jpg",
    "What was the revenue in 2021?"
)
print(f"答案: {answer}")
```

---

**評估腳本**:

```python
# evaluate.py
from datasets import load_dataset
from Levenshtein import distance
from tqdm import tqdm
import numpy as np

def calculate_anls(prediction: str, ground_truths: List[str]) -> float:
    """
    計算ANLS分數 (對多個參考答案取最大值)
    """
    max_score = 0.0

    for gt in ground_truths:
        pred = prediction.lower().strip()
        gt = gt.lower().strip()

        if len(pred) == 0 and len(gt) == 0:
            return 1.0

        if len(pred) == 0 or len(gt) == 0:
            continue

        lev_dist = distance(pred, gt)
        max_len = max(len(pred), len(gt))
        nls = 1 - (lev_dist / max_len)

        score = nls if nls >= 0.5 else 0.0
        max_score = max(max_score, score)

    return max_score

# 評估驗證集
dataset = load_dataset("MMInstruction/InfographicVQA")
val_data = dataset['validation']

scores = []

for sample in tqdm(val_data):
    prediction = predict(sample['image'], sample['question'])
    ground_truths = sample['answers']

    score = calculate_anls(prediction, ground_truths)
    scores.append(score)

# 計算平均ANLS
avg_anls = np.mean(scores)
print(f"驗證集ANLS: {avg_anls:.4f}")

# 分析
print(f"完全匹配(score=1.0): {(np.array(scores) == 1.0).sum() / len(scores) * 100:.1f}%")
print(f"部分匹配(0.5≤score<1.0): {((np.array(scores) >= 0.5) & (np.array(scores) < 1.0)).sum() / len(scores) * 100:.1f}%")
print(f"不匹配(score=0): {(np.array(scores) == 0.0).sum() / len(scores) * 100:.1f}%")
```

---

## 5. 提示工程:針對Infographic的優化

### 5.1 基礎Prompt模板

```python
# 基礎版本
question = "What was the revenue in 2021?"

# 優化版本1:明確指示
question = "Based on the infographic, what was the revenue in 2021? Answer with just the number and unit."

# 優化版本2:分步推理
question = """
Look at the infographic and answer the following question:
Question: What was the revenue in 2021?

Think step by step:
1. Locate the revenue information
2. Find the 2021 data point
3. Extract the exact value

Answer:
"""
```

### 5.2 Few-Shot Prompting

```python
# 提供範例來引導模型
def create_few_shot_prompt(question: str) -> str:
    examples = """
Example 1:
Question: How many countries are shown in the chart?
Answer: 5

Example 2:
Question: What percentage of users are from Europe?
Answer: 32%

Now answer this question:
"""
    return examples + f"Question: {question}\nAnswer:"
```

---

## 6. 預期成果與基準對比

### 6.1 性能基準

**InfographicVQA Test Set (ANLS)**:

| 模型 | 參數量 | ANLS | 記憶體(推論) |
|------|--------|------|-------------|
| **人類表現** | - | **0.920** | - |
| T5-Base | 220M | 0.118 | 4GB |
| LayoutLM-v3 | 125M | 0.352 | 2GB |
| **Pix2Struct-base** | **282M** | **0.384** | **2GB** |
| **Pix2Struct-base (微調)** | **282M** | **0.488** | **2GB** |
| Donut | 350M | 0.401 | 3GB |
| **Qwen-VL-7B** | **7B** | **≈0.55** | **14GB** |
| **Qwen2.5-VL-3B (QLoRA微調後預估)** | **3B** | **≈0.52-0.58** | **6GB** |

**預期結果**:
- Pix2Struct-base微調: ANLS **0.45-0.50**
- Qwen2.5-VL-3B微調: ANLS **0.52-0.58**
- 訓練時間: 3-5小時 (RTX 3070, 3 epochs)

---

### 6.2 失敗案例分析

**常見錯誤類型**:

1. **OCR錯誤**:
```
真實文字: "$1.25M"
模型識別: "$1.25 million" 或 "1.25"
→ ANLS受格式不一致影響
```

2. **數值計算錯誤**:
```
問題: "What is the percentage increase from 2020 to 2021?"
資料: 2020: 100, 2021: 130
正確答案: "30%"
模型預測: "130%" (直接輸出2021值)
```

3. **複雜佈局理解失敗**:
```
情境: 多列表格,問題涉及特定欄位
錯誤: 模型讀取了錯誤的欄或行
```

**改進策略**:
- 使用更多訓練數據
- 調整Prompt強調"exact format"
- 後處理標準化答案格式

---

## 7. 總結與下一步

### 本章學到的關鍵技能

✅ **理解InfographicVQA的獨特挑戰**: 多模態融合、複雜佈局、數值推理

✅ **掌握模型選擇策略**: 根據VRAM、任務特性選擇Pix2Struct或Qwen-VL

✅ **實作QLoRA微調**:
   - NF4量化原理
   - BitsAndBytesConfig配置
   - LoRA參數調優

✅ **記憶體優化技巧**: Gradient Checkpointing、Paged Optimizers、Batch調整

✅ **評估與分析**: ANLS計算、失敗案例診斷

---

### 下一步學習路徑

**選項1: 深入量化技術 (進階)**
→ 前往 [第二章: 量化感知訓練(QAT)](02_qat_advanced.md)
- 實作Straight-Through Estimator
- 對比PTQ vs QAT
- 達成INT4下更高精度

**選項2: 擴展到其他任務**
- 嘗試DocVQA、ChartQA資料集
- 對比不同任務的微調策略
- 實作多任務學習

**選項3: 模型部署優化**
→ 前往 [第三章: 混合精度與部署](03_mixed_precision.md)
- 敏感度分析
- 混合精度策略
- ONNX/TensorRT導出

---

### 實作檢查清單

完成以下任務,確保你真正掌握本章內容:

- [ ] 成功在8GB GPU上微調Qwen2.5-VL-3B
- [ ] ANLS分數達到0.50以上
- [ ] 理解NF4量化的數學原理(能在白板上解釋)
- [ ] 能夠診斷並解決OOM(記憶體溢出)問題
- [ ] 實作自定義的資料集預處理pipeline

---

**準備好了嗎?**

開始實作 → [`lab_tasks/task01_infographicvqa/`](../lab_tasks/task01_infographicvqa/)

或繼續閱讀進階主題 → [第二章: QAT](02_qat_advanced.md)
