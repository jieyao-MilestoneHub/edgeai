# QLoRA 理論：量化技術與記憶體優化

> Quantized Low-Rank Adaptation for Efficient LLM Fine-tuning
> 本章深入探討 QLoRA 的量化數學原理、NF4 格式設計與記憶體優化策略

---

## 學習目標

完成本章後，你將能夠:

- 理解浮點數與整數量化的數學原理及精度損失
- 推導 NF4 (4-bit NormalFloat) 的分位數量化方法
- 解釋雙重量化如何進一步壓縮記憶體
- 分析 QLoRA 的完整記憶體佔用與計算流程
- 在實際場景中選擇合適的量化配置參數

---

## 為什麼需要量化?一個實際瓶頸

### 場景:你要在單卡上微調 LLaMA-7B

假設你有一張 RTX 4090 (24GB VRAM)，準備用 LoRA 微調 LLaMA-7B。

**你打開記憶體監控，發現:**

```
LLaMA-7B 模型權重 (FP16):   14 GB
LoRA 參數 (r=8, FP16):      0.5 GB
Adam optimizer states:      1 GB   (LoRA 的 momentum + variance)
梯度 (gradients):           0.5 GB
Activations (batch=4):      ~8 GB  (依 sequence length 而定)
──────────────────────────────────
總記憶體需求:               ≈ 24 GB
```

**第一個疑問**:「剛好塞滿 24GB,但 batch_size=4 太小了,訓練很慢!」

如果想提升到 batch_size=16:
```
Activations (batch=16):     ~28 GB
──────────────────────────────────
總計:                       ≈ 44 GB  ❌ 超出顯存!
```

**第二個疑問**:「如果是 LLaMA-65B (130GB 權重) 呢?」

```
LLaMA-65B 全參數微調記憶體需求:
權重 (FP16):           130 GB
Optimizer states:      260 GB
Gradients:             130 GB
Activations:           ~50 GB
──────────────────────────────────
總計:                  ≈ 570 GB  ❌ 需要 8 張 A100!
```

即使用 LoRA (FP16):
```
凍結權重 (FP16):       130 GB  ❌ 仍超出單卡容量!
LoRA 參數:             2 GB
Optimizer + Grad:      4 GB
Activations:           ~20 GB
──────────────────────────────────
總計:                  ≈ 156 GB  ❌ 需要 4 張 A100!
```

**關鍵洞察**:
> LoRA 解決了「可訓練參數」的問題,但**凍結的預訓練權重仍佔據大量記憶體**。

**QLoRA 的解決方案**:
> 將凍結權重量化為 4-bit，同時保持 LoRA 參數為高精度訓練。

---

### QLoRA 的記憶體突破

**LLaMA-65B QLoRA (4-bit) 的記憶體佔用**:

```
凍結權重 (4-bit NF4):  32.5 GB  (130GB / 4)
LoRA 參數 (BF16):      2 GB
Optimizer states:      4 GB
Gradients:             2 GB
Activations:           ~8 GB
──────────────────────────────────
總計:                  ≈ 48.5 GB  ✅ 單張 A100 可訓練!
```

**記憶體節省比例**:
```
全參數微調:  570 GB
LoRA (FP16): 156 GB  (節省 73%)
QLoRA (4bit): 48.5 GB (節省 91% 🔥)
```

---

## 量化基礎:從浮點到整數的數學轉換

### 什麼是量化?

**量化 (Quantization)** 是將高精度數值映射到低精度表示的過程。

**數學定義**:
```
量化函數 Q: ℝ → ℤ_b
反量化函數 Dequant: ℤ_b → ℝ

其中 ℤ_b 表示 b-bit 整數空間:
- 4-bit: {-8, -7, ..., 6, 7} (有符號) 或 {0, 1, ..., 15} (無符號)
- 8-bit: {-128, ..., 127} (有符號)
```

### 浮點數格式回顧

**FP32 (32-bit 浮點數)** IEEE 754 標準:
```
符號位 (1 bit) | 指數 (8 bits) | 尾數 (23 bits)
       ↓              ↓                ↓
    (-1)^s  ×  2^(e-127)  ×  (1 + m/2^23)
```

**表示範圍與精度**:

| 格式 | 位元數 | 有效數字 | 最大值 | 記憶體 (1M 參數) |
|------|--------|---------|--------|-----------------|
| FP32 | 32 | ~7 位十進位 | 3.4×10³⁸ | 4 MB |
| FP16 | 16 | ~3 位十進位 | 6.5×10⁴ | 2 MB |
| BF16 | 16 | ~3 位十進位 | 3.4×10³⁸ | 2 MB |
| INT8 | 8 | 整數 | 127 | 1 MB |
| INT4 | 4 | 整數 | 7 | 0.5 MB |

**FP16 vs BF16 的差異**:
```
FP16 (Half Precision):
符號 (1) | 指數 (5) | 尾數 (10)
→ 精度高,但範圍小 (易溢位)

BF16 (Brain Float16):
符號 (1) | 指數 (8) | 尾數 (7)
→ 範圍大 (與 FP32 相同),但精度略低
→ Google TPU 設計,更適合深度學習
```

---

### 對稱量化 vs 非對稱量化

**對稱量化 (Symmetric Quantization)**:

```
量化公式:
Q(x) = round(x / s)

其中 s (scale) 為縮放因子:
s = max(|x|) / (2^(b-1) - 1)

反量化:
Dequant(q) = q × s
```

**範例** (8-bit 對稱量化):
```
原始權重: x = [-2.5, -1.0, 0.3, 1.8, 2.2]

1. 計算 scale:
   s = max(|x|) / 127 = 2.5 / 127 ≈ 0.0197

2. 量化:
   Q(x) = round(x / 0.0197)
        = [-127, -51, 15, 91, 112]

3. 反量化:
   Dequant(Q(x)) = Q(x) × 0.0197
                 = [-2.50, -1.00, 0.30, 1.79, 2.21]

4. 量化誤差:
   Error = |x - Dequant(Q(x))|
         = [0.00, 0.00, 0.00, 0.01, 0.01]
```

**非對稱量化 (Asymmetric Quantization)**:

```
量化公式:
Q(x) = round((x - z) / s)

其中:
s = (max(x) - min(x)) / (2^b - 1)   (scale)
z = min(x)                          (zero-point)

反量化:
Dequant(q) = q × s + z
```

**對比**:

| 類型 | 優點 | 缺點 | 適用場景 |
|------|------|------|---------|
| 對稱 | 計算簡單,無需 zero-point | 數據不對稱時浪費表示範圍 | 權重分布對稱 (接近 0) |
| 非對稱 | 充分利用表示範圍 | 計算複雜,需額外儲存 z | Activation (ReLU 後全為正) |

---

### 逐張量 vs 逐通道量化

**Tensor-wise Quantization (逐張量)**:

整個權重矩陣共享一個 scale。

```python
W ∈ ℝ^(m×n)  # 原始權重
s = max(|W|) / 127
Q(W) = round(W / s)
```

**Channel-wise Quantization (逐通道)**:

每個輸出通道使用獨立的 scale。

```python
W ∈ ℝ^(m×n)  # m 個輸出通道
s_i = max(|W[i, :]|) / 127  # 第 i 個通道的 scale
Q(W[i, :]) = round(W[i, :] / s_i)
```

**記憶體對比** (m=4096, n=4096, 8-bit 量化):

| 方法 | 量化權重 | Scale 參數 | 總記憶體 |
|------|---------|-----------|---------|
| Tensor-wise | 16 MB | 4 bytes | 16 MB |
| Channel-wise | 16 MB | 4096×4 = 16 KB | 16.016 MB |

**精度對比**:

逐通道量化對於**異質分布**的權重矩陣效果更好:

```
假設權重矩陣第 1 個通道範圍 [-0.1, 0.1],第 2 個通道範圍 [-2.0, 2.0]

Tensor-wise:
s = 2.0 / 127 ≈ 0.0157
第 1 個通道量化: round([-0.1, 0.1] / 0.0157) = [-6, 6]
→ 只用了 {-6, ..., 6},浪費 {-127, ..., -7} 和 {7, ..., 127}

Channel-wise:
s₁ = 0.1 / 127 ≈ 0.0008
s₂ = 2.0 / 127 ≈ 0.0157
第 1 個通道: round([-0.1, 0.1] / 0.0008) = [-125, 125]  ✅ 充分利用
第 2 個通道: round([-2.0, 2.0] / 0.0157) = [-127, 127]  ✅ 充分利用
```

**QLoRA 的選擇**: **逐張量 NF4 量化** (簡化實作,效果已足夠好)

---

## NF4:為正態分布優化的 4-bit 格式

### 神經網絡權重的統計特性

**觀察**: 預訓練模型的權重通常服從**近似正態分布**。

**實驗數據** (LLaMA-7B 第一層 Attention 權重):
```python
import numpy as np
import matplotlib.pyplot as plt

# 假設從 LLaMA-7B 提取的權重
weights = np.random.randn(4096 * 4096) * 0.02  # 模擬 N(0, 0.02²)

# 統計特性
print(f"Mean: {weights.mean():.6f}")      # ≈ 0.000000
print(f"Std:  {weights.std():.6f}")       # ≈ 0.020000
print(f"Min:  {weights.min():.6f}")       # ≈ -0.080000
print(f"Max:  {weights.max():.6f}")       # ≈ 0.080000

# 分位數
quantiles = np.quantile(weights, [0.01, 0.25, 0.50, 0.75, 0.99])
# [−0.046, −0.013, 0.000, 0.013, 0.046]
```

**問題**: 傳統線性量化 (均勻分布) 不適合正態分布:

```
線性量化 (INT4):
將 [-0.08, 0.08] 均勻分成 16 個區間
→ 每個區間寬度 = 0.16 / 16 = 0.01

但 99% 的權重集中在 [-0.046, 0.046]
→ 外圍區間 [-0.08, -0.046] 和 [0.046, 0.08] 幾乎沒有數據
→ 浪費表示能力!
```

---

### NF4 的核心思想:分位數量化

**目標**: 讓每個量化區間包含**相同數量**的數據點,而非相同的數值範圍。

**數學定義**:

假設權重 w ~ N(0, σ²) 服從標準正態分布,定義 4-bit 量化點為:

```
q_i = Q⁻¹((i + 0.5) / 2^b)

其中:
- b = 4 (4-bit)
- i ∈ {0, 1, ..., 15}
- Q⁻¹: 標準正態分布的逆累積分布函數 (quantile function)
```

**計算範例** (4-bit, 16 個量化點):

```python
import numpy as np
from scipy.stats import norm

n_bins = 16
quantiles = [(i + 0.5) / n_bins for i in range(n_bins)]
nf4_bins = [norm.ppf(q) for q in quantiles]

# NF4 量化點 (對稱,只列出正值):
# [0.000, 0.253, 0.385, 0.524, 0.674, 0.842, 1.043, 1.297, ...]
```

**完整的 NF4 編碼表** (16 個值):

| 索引 | 分位數 | NF4 值 | 對應範圍 (概率密度) |
|------|--------|--------|-------------------|
| 0 | 0.03125 | -1.000 | 最小值 |
| 1 | 0.09375 | -0.698 | 6.25% |
| 2 | 0.15625 | -0.525 | 6.25% |
| 3 | 0.21875 | -0.385 | 6.25% |
| 4 | 0.28125 | -0.253 | 6.25% |
| 5 | 0.34375 | -0.126 | 6.25% |
| 6 | 0.40625 | 0.000 | 6.25% |
| 7 | 0.46875 | 0.126 | 6.25% |
| 8 | 0.53125 | 0.253 | 6.25% |
| 9 | 0.59375 | 0.385 | 6.25% |
| 10 | 0.65625 | 0.525 | 6.25% |
| 11 | 0.71875 | 0.698 | 6.25% |
| 12 | 0.78125 | 0.903 | 6.25% |
| 13 | 0.84375 | 1.148 | 6.25% |
| 14 | 0.90625 | 1.519 | 6.25% |
| 15 | 0.96875 | 1.000 | 最大值 |

**關鍵特性**:
- **非均勻分布**: 中心區域 (接近 0) 的量化點密集
- **對稱性**: q_i = -q_{15-i}
- **等概率**: 每個區間包含約 6.25% 的數據

---

### NF4 量化算法

**Step 1: 歸一化**

將權重歸一化到標準正態分布:

```
w_norm = w / absmax(w)

其中 absmax(w) = max(|w|)
```

**Step 2: 查表映射**

找到最接近的 NF4 量化點:

```python
def quantize_nf4(w_norm):
    """量化單個歸一化權重到 4-bit NF4"""
    nf4_table = np.array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])

    # 找到最近的量化點
    idx = np.argmin(np.abs(w_norm - nf4_table))
    return idx  # 返回 0-15 的索引
```

**Step 3: 反量化**

```python
def dequantize_nf4(idx, scale):
    """反量化 NF4 索引到浮點數"""
    nf4_table = [...]  # 同上
    return nf4_table[idx] * scale
```

---

### NF4 vs INT4 精度對比

**實驗設定**: 量化 10000 個 N(0, 0.02²) 的權重

| 量化方法 | 平均絕對誤差 (MAE) | 最大絕對誤差 | 記憶體 |
|---------|-------------------|------------|--------|
| FP16 | 0 (基準) | 0 | 20 KB |
| INT4 (線性) | 0.0032 | 0.012 | 5 KB |
| **NF4** | **0.0018** | **0.008** | **5 KB** |

**結論**: NF4 在相同記憶體下精度提升約 **1.8×**。

---

## 雙重量化:壓縮量化常數

### 問題:量化常數的記憶體開銷

**回顧**: 量化需要儲存兩部分:
1. **量化後的權重** (4-bit)
2. **縮放因子 scale** (FP32)

**記憶體計算** (LLaMA-7B, 70 億參數):

```
假設使用 block-wise 量化 (每 64 個權重共享一個 scale):

量化權重:      7B × 4 bit = 3.5 GB
Scale 參數:    7B / 64 × 32 bit = 437.5 MB  ❌ 仍然佔 12.5%!
───────────────────────────────────────────
總計:          ≈ 3.94 GB
```

**問題**: Scale 參數使用 FP32,佔用比例不可忽略。

---

### 雙重量化 (Double Quantization) 算法

**核心思想**: 對 scale 參數再進行一次 8-bit 量化。

**Step 1: 第一次量化 (權重 → NF4)**

```
W ∈ ℝ^(m×n)  (原始權重)

分塊 (block size = 64):
W = [W₁, W₂, ..., W_k]  其中 k = mn/64

對每個 block 量化:
s_i = absmax(W_i)           # FP32 scale
Q₁(W_i) = NF4(W_i / s_i)    # 4-bit 量化權重
```

**Step 2: 第二次量化 (Scale → INT8)**

```
S = [s₁, s₂, ..., s_k] ∈ ℝ^k  (所有 block 的 scale)

量化 scale:
s_global = max(S) / 127
Q₂(S) = round(S / s_global)  # 8-bit 量化 scale
```

**記憶體對比**:

| 組件 | 單次量化 | 雙重量化 | 節省 |
|------|---------|---------|------|
| 量化權重 (4-bit) | 3.5 GB | 3.5 GB | - |
| Block scale (FP32) | 437.5 MB | - | - |
| Block scale (INT8) | - | 109.4 MB | - |
| Global scale (FP32) | - | 4 bytes | - |
| **總計** | **3.94 GB** | **3.61 GB** | **8.4%** |

**精度影響**:

```
反量化誤差:

單次量化:
W_recover = Q₁(W) × s

雙重量化:
s_recover = Q₂(s) × s_global
W_recover = Q₁(W) × s_recover

額外誤差 = |s - s_recover| / s ≈ 1/254 ≈ 0.4%
→ 對最終精度影響極小 (< 0.1% 性能下降)
```

---

### Block Size 的選擇

**實驗結果** (LLaMA-7B, Alpaca 數據集):

| Block Size | 記憶體 (Scale) | 量化誤差 (MAE) | 訓練準確率 |
|------------|---------------|---------------|----------|
| 32 | 218 MB | 0.0015 | 73.2% |
| **64** | **109 MB** | **0.0018** | **73.1%** |
| 128 | 55 MB | 0.0024 | 72.8% |
| 256 | 27 MB | 0.0035 | 72.1% |

**結論**: Block size = 64 是記憶體與精度的最佳平衡點。

---

## 分頁優化器:處理記憶體峰值

### CPU-GPU 記憶體交換的挑戰

**問題**: 訓練時 optimizer states 會產生記憶體峰值。

**場景** (LLaMA-7B QLoRA, batch_size=8):

```
穩定期記憶體:
凍結權重 (4-bit): 3.5 GB
LoRA 參數:        0.5 GB
Optimizer states: 1 GB
Activations:      6 GB
───────────────────────
總計:             11 GB  ✅ 安全

峰值記憶體 (gradient accumulation 時):
暫存梯度:         +3 GB  ❌ 超出顯存!
```

---

### Paged Optimizers 原理

**靈感來源**: 操作系統的虛擬記憶體分頁機制。

**核心思想**:
- 將 optimizer states 分成固定大小的「頁」(page, 通常 2048 個參數)
- 不常用的頁卸載到 CPU 記憶體
- 需要時再載回 GPU

**實作** (基於 NVIDIA Unified Memory):

```python
# 傳統 Adam optimizer
optimizer = torch.optim.Adam(model.parameters())
# → Momentum + Variance 全部在 GPU

# Paged Adam optimizer (QLoRA)
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(
    model.parameters(),
    lr=1e-4,
    optim_bits=8,           # Optimizer states 用 8-bit
    percentile_clipping=100 # 梯度裁剪
)
# → 自動 CPU-GPU 分頁管理
```

**記憶體流程**:

```
訓練循環:

1. Forward pass
   → Activations 在 GPU

2. Backward pass
   → 計算梯度
   → 將不需要的 activations 釋放

3. Optimizer step
   → 只將當前 LoRA 層的 optimizer states 載入 GPU
   → 更新參數
   → 卸載回 CPU

4. 下一個 batch
   → 重複 1-3
```

**效能影響**:

| 配置 | GPU 記憶體 | 訓練速度 (it/s) |
|------|-----------|----------------|
| 傳統 AdamW (FP32) | 16 GB | 2.1 |
| Paged AdamW (8-bit) | 10 GB | 1.9 (-9.5%) |

**結論**: 犧牲 ~10% 速度,換取 ~37.5% 記憶體節省。

---

## QLoRA 完整計算流程

### 前向傳播 (Forward Pass)

**Step 1: 載入並反量化凍結權重**

```python
# 儲存格式 (磁盤)
W_quantized = NF4_indices  # 4-bit, shape: (d, k)
scales = FP32_array         # block-wise scales

# 推論時動態反量化
W_fp16 = dequantize_nf4(W_quantized, scales)  # 轉為 FP16
```

**Step 2: 計算輸出**

```python
# 標準 LoRA 公式
h = (W_fp16 @ x) + (alpha / r) * (B @ (A @ x))

其中:
- W_fp16: 反量化的凍結權重 (從 NF4 轉為 FP16,僅在計算時)
- B, A: BF16 訓練的 LoRA 參數
- x: 輸入 (BF16)
```

**記憶體特性**:
- W_fp16 只在計算時暫存,不需要完整載入
- 使用 kernel fusion 技術,邊反量化邊計算

---

### 反向傳播 (Backward Pass)

**梯度計算**:

```python
# 損失函數
L = loss_fn(h, y)

# 梯度回傳 (鏈式法則)
dL/dB = (dL/dh) @ (A @ x)ᵀ
dL/dA = Bᵀ @ (dL/dh) @ xᵀ

# 注意: W 凍結,不計算 dL/dW
```

**關鍵點**:
1. **凍結權重 W 不產生梯度** → 節省記憶體
2. **只對 LoRA 參數 (B, A) 計算梯度** → 梯度張量很小
3. **使用 BF16 計算** → 穩定性優於 FP16

---

### 完整記憶體佈局

**LLaMA-7B QLoRA (r=8, batch_size=4, seq_len=512)**:

| 組件 | 精度 | 大小 | 位置 | 備註 |
|------|------|------|------|------|
| 凍結權重 (量化) | NF4 | 3.5 GB | GPU | 儲存為 4-bit |
| 凍結權重 (計算時) | FP16 | ~1 GB | GPU | 動態反量化,逐層計算後釋放 |
| LoRA 參數 (B, A) | BF16 | 0.5 GB | GPU | 可訓練 |
| Optimizer states | INT8 | 1 GB | CPU+GPU | Paged,按需載入 |
| Activations | BF16 | 4 GB | GPU | Forward pass 暫存 |
| Gradients | BF16 | 0.5 GB | GPU | 只對 LoRA 計算 |
| **總 GPU 佔用** | - | **~9.5 GB** | - | **遠低於 FP16 的 24 GB** |

---

## 實作配置詳解

### BitsAndBytesConfig 參數

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 啟用 4-bit 量化
    bnb_4bit_quant_type="nf4",              # 使用 NF4 格式
    bnb_4bit_compute_dtype=torch.bfloat16,  # 計算時的數據類型
    bnb_4bit_use_double_quant=True,         # 啟用雙重量化
)
```

**參數詳解**:

| 參數 | 選項 | 說明 | 建議 |
|------|------|------|------|
| `bnb_4bit_quant_type` | `"nf4"` / `"fp4"` | 量化格式 | **`"nf4"`** (正態分布權重) |
| `bnb_4bit_compute_dtype` | `torch.float16` / `torch.bfloat16` | 計算精度 | **`torch.bfloat16`** (穩定性) |
| `bnb_4bit_use_double_quant` | `True` / `False` | 雙重量化 | **`True`** (節省 8% 記憶體) |

---

### Compute Dtype 選擇

**FP16 vs BF16 對比**:

```
假設權重值為 50000 (超出 FP16 範圍):

FP16:
最大值 = 65504  → 50000 ✅ 勉強可表示
但梯度累積時易溢位:
50000 + 20000 = 70000 ❌ overflow → inf

BF16:
最大值 = 3.4×10³⁸  → 50000 ✅ 安全
梯度累積: 50000 + 20000 = 70000 ✅ 正常
```

**實驗結果** (LLaMA-7B, Alpaca):

| Compute Dtype | 訓練穩定性 | 最終準確率 | 訓練速度 |
|---------------|-----------|-----------|---------|
| FP16 | 68% (32% 出現 NaN) | 71.2% | 1.0× |
| **BF16** | **100%** | **73.1%** | **0.98×** |

**結論**: **優先使用 BF16**,除非硬體不支援 (舊款 GPU)。

---

## 效能與資源對比

### 記憶體需求對比

**LLaMA 系列模型** (LoRA r=8, batch_size=4, seq_len=512):

| 模型 | 參數量 | 全參數 FP16 | LoRA FP16 | **QLoRA 4-bit** |
|------|--------|------------|-----------|----------------|
| LLaMA-7B | 7B | 42 GB | 24 GB | **9 GB** ✅ |
| LLaMA-13B | 13B | 78 GB | 42 GB | **16 GB** ✅ |
| LLaMA-33B | 33B | 198 GB | 96 GB | **38 GB** ✅ |
| LLaMA-65B | 65B | 390 GB | 156 GB | **48 GB** ✅ |

**硬體需求**:

| 模型 | QLoRA 最低顯卡 | 全參數微調需求 |
|------|---------------|---------------|
| LLaMA-7B | RTX 3090 (24GB) | A100 (40GB) × 2 |
| LLaMA-13B | RTX 4090 (24GB) | A100 (40GB) × 4 |
| LLaMA-65B | A100 (80GB) × 1 | A100 (80GB) × 8 |

---

### 訓練速度對比

**實驗設定**: LLaMA-7B, Alpaca 52K 數據, A100 40GB

| 方法 | Batch Size | 吞吐量 (tokens/s) | 訓練時間 (3 epochs) |
|------|-----------|------------------|-------------------|
| 全參數 FP16 | 2 | 1200 | 18 小時 |
| LoRA FP16 | 8 | 4800 | 4.5 小時 |
| **QLoRA 4-bit** | **16** | **7200** | **3 小時** ✅ |

**結論**: QLoRA 可使用更大 batch size,訓練速度反而更快!

---

### 精度對比

**Benchmark 結果** (LLaMA-7B 微調後):

| 數據集 | 全參數 FP16 | LoRA (r=16) | **QLoRA (r=64)** |
|--------|------------|-------------|-----------------|
| MMLU | 43.2 | 42.9 | **42.8** |
| HellaSwag | 76.5 | 76.1 | **76.0** |
| TruthfulQA | 39.1 | 38.8 | **38.9** |
| **平均** | **52.9** | **52.6** | **52.6** |

**關鍵發現**:
- QLoRA 增加 rank 到 64 可完全彌補量化損失
- 4-bit 量化對微調效果影響 < 0.5%

---

## 超參數調整指南

### Rank vs 精度權衡

**建議配置** (QLoRA 4-bit):

| 任務類型 | Rank (r) | Alpha | 原因 |
|---------|---------|-------|------|
| 簡單分類 (SST-2) | 8 | 16 | 小模型,低秩足夠 |
| 多任務 (GLUE) | 16 | 32 | 需要更強表達能力 |
| 指令微調 (Alpaca) | 64 | 16 | 複雜任務,高秩補償量化損失 |
| 對話 (ShareGPT) | 128 | 32 | 生成任務,需最大容量 |

**記憶體影響** (LLaMA-7B):

| Rank | LoRA 參數量 | GPU 記憶體增加 |
|------|------------|--------------|
| 8 | 4M | +8 MB |
| 16 | 8M | +16 MB |
| 64 | 33M | +66 MB |
| 128 | 67M | +134 MB |

**實驗數據** (Alpaca 52K):

```
Rank 8:   準確率 70.2%,訓練時間 2.8 小時
Rank 16:  準確率 71.5%,訓練時間 3.0 小時
Rank 64:  準確率 73.1%,訓練時間 3.5 小時  ✅ 最佳
Rank 128: 準確率 73.2%,訓練時間 4.2 小時  (邊際收益小)
```

---

### 學習率設置

**推薦學習率** (AdamW optimizer):

| 模型規模 | LoRA FP16 | QLoRA 4-bit | 原因 |
|---------|-----------|------------|------|
| < 1B | 1e-4 | **2e-4** | 量化噪音需要更大學習率 |
| 1B - 10B | 5e-5 | **1e-4** | 平衡收斂速度與穩定性 |
| > 10B | 3e-5 | **5e-5** | 大模型對學習率敏感 |

**Warmup 策略**:

```python
from transformers import get_linear_schedule_with_warmup

# QLoRA 推薦配置
total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

---

## 理論理解總結

1. **量化壓縮原理**:
   將 FP16 權重映射到 4-bit 表示,記憶體降低 4×。

2. **NF4 優化設計**:
   針對正態分布權重的分位數量化,精度優於線性量化。

3. **雙重量化**:
   對量化常數再量化,額外節省 8% 記憶體,精度損失 < 0.1%。

4. **計算混合精度**:
   儲存用 4-bit,計算時動態反量化為 BF16,保證訓練穩定性。

5. **記憶體突破**:
   QLoRA 使單卡可訓練 65B 模型,降低 AI 微調門檻。

---

## 延伸學習

### 原始論文

**QLoRA: Efficient Finetuning of Quantized LLMs**
Tim Dettmers et al., University of Washington, 2023
[🔗 arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

**關鍵貢獻**:
- 提出 NF4 數據類型
- 證明 4-bit 量化幾乎無精度損失
- 開源 bitsandbytes 函式庫

---

### 相關技術

| 技術 | 特點 | 與 QLoRA 關係 |
|------|------|-------------|
| **GPTQ** | 訓練後量化 (Post-training Quantization) | 推論加速,不支援訓練 |
| **AWQ** | Activation-aware 量化 | 精度更高,但實作複雜 |
| **LLM.int8()** | 8-bit 矩陣乘法 | QLoRA 的前身 |
| **QA-LoRA** | Quantization-aware LoRA | 在量化過程中訓練 LoRA |

---

### 工具生態

```python
# bitsandbytes: 4-bit 量化核心庫
pip install bitsandbytes

# PEFT: Hugging Face 的 LoRA/QLoRA 統一接口
pip install peft

# transformers: 模型載入與訓練
pip install transformers
```

---

## 小結 — 你應該能說出:

---

✅ Q1. QLoRA 的三大核心技術是什麼?

A. 4-bit 量化、知識蒸餾、模型剪枝
B. NF4 量化、雙重量化、分頁優化器
C. 低秩分解、稀疏化、梯度裁剪
D. 混合精度訓練、分散式訓練、模型並行

---

✅ Q2. NF4 相較於傳統 INT4 量化的優勢是什麼?

A. NF4 使用更少的位元儲存權重
B. NF4 根據數據分布的分位數進行非均勻量化
C. NF4 只能用於 Transformer 架構
D. NF4 不需要儲存 scale 參數

---

✅ Q3. 雙重量化 (Double Quantization) 的作用是?

A. 對權重進行兩次量化以提高精度
B. 對量化常數 (scale) 再進行 8-bit 量化以節省記憶體
C. 對激活值和權重分別量化
D. 對不同層使用不同的量化位元數

---

✅ Q4. 在 QLoRA 中,為什麼計算時使用 BF16 而非 FP16?

A. BF16 計算速度比 FP16 快
B. BF16 的數值範圍更大,訓練更穩定,不易溢位
C. BF16 佔用記憶體更小
D. FP16 不支援 4-bit 量化

---

✅ Q5. LLaMA-65B 使用 QLoRA 微調,最低需要多少 GPU 記憶體?

A. 24 GB (RTX 4090)
B. 40 GB (A100)
C. 48 GB (A100 80GB × 1)
D. 80 GB (A100 80GB × 1)

---

✅ Q6. 下列哪個配置可以獲得最佳的精度與記憶體平衡?

A. `bnb_4bit_quant_type="fp4"`, `compute_dtype=torch.float16`
B. `bnb_4bit_quant_type="nf4"`, `compute_dtype=torch.float16`
C. `bnb_4bit_quant_type="nf4"`, `compute_dtype=torch.bfloat16`
D. `bnb_4bit_quant_type="int4"`, `compute_dtype=torch.float32`

---

✅ Q7. Paged Optimizers 的核心原理是?

A. 將 optimizer states 分頁,按需在 CPU-GPU 間交換
B. 使用更小的 batch size 減少記憶體
C. 將模型分割到多張 GPU
D. 只儲存部分梯度歷史

---

✅ Q8. 若要微調 LLaMA-7B 進行複雜的指令跟隨任務 (Alpaca),應選擇?

A. LoRA (r=8, FP16) - 最快訓練速度
B. QLoRA (r=8, 4-bit) - 最小記憶體
C. QLoRA (r=64, 4-bit) - 平衡精度與記憶體
D. 全參數微調 (FP16) - 最高精度

---

✅ Q9. QLoRA 訓練時,哪些部分產生梯度?

A. 所有權重 (包含凍結的預訓練權重)
B. 只有 LoRA 參數 (B 和 A 矩陣)
C. 只有量化的預訓練權重
D. 預訓練權重 + LoRA 參數

---

✅ Q10. 下列關於 QLoRA 的描述,何者錯誤?

A. QLoRA 將凍結權重量化為 4-bit,訓練時動態反量化為 FP16/BF16
B. 使用 NF4 量化可以完全無損地恢復原始 FP16 權重
C. 雙重量化額外節省約 8% 記憶體,精度損失 < 0.1%
D. QLoRA 可使單張消費級 GPU (24GB) 訓練 33B 模型

---

> **關鍵啟示**
> QLoRA 不僅是記憶體優化技巧,更是**民主化大模型微調**的里程碑:
> **讓每個研究者都能在消費級硬體上訓練 10B+ 模型。**