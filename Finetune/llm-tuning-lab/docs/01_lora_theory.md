# 📙 LoRA 理論

> Low-Rank Adaptation of Large Language Models

## 🎯 學習目標

閱讀本文後，你將能夠：

- ✅ 理解為什麼需要 LoRA
- ✅ 掌握低秩分解 (Low-Rank Decomposition) 原理
- ✅ 計算 LoRA 的參數量與記憶體節省
- ✅ 手寫 LoRA 模組實作
- ✅ 調整 rank 與 alpha 超參數

---

## 🤔 為什麼需要 LoRA？

### 全參數微調的挑戰

假設我們要微調 LLaMA-7B 模型：

```
模型參數：7B (70 億)
精度：FP16 (2 bytes/param)
記憶體需求：7B × 2 = 14GB (僅權重)

訓練時額外需求：
- Optimizer states (Adam): 2× weights = 28GB
- Gradients: 1× weights = 14GB
- Activations: ~20GB

總計：~76GB
```

**問題：**
- ❌ 單張 A100 (40GB) 無法訓練
- ❌ 每個下游任務都需要完整模型副本
- ❌ 部署時需要載入整個模型

---

## 💡 LoRA 核心思想

### 關鍵洞察

> **假設：預訓練模型的權重更新存在於低秩子空間**

數學表述：

```
原始全參數更新：
W' = W₀ + ΔW

其中 ΔW ∈ ℝ^(d×k) 是低秩的：rank(ΔW) << min(d, k)

LoRA 近似：
W' = W₀ + BA

其中：
- B ∈ ℝ^(d×r)
- A ∈ ℝ^(r×k)
- r << min(d, k)  (r 是 rank)
```

### 視覺化理解

```
Full Fine-tuning:
┌─────────────┐
│   W₀ (d×k)  │  ─────>  ┌─────────────┐
│  (frozen)   │           │  W₀ + ΔW    │
└─────────────┘           │  (d×k)      │
                          └─────────────┘
                          可訓練參數：d × k

LoRA:
┌─────────────┐
│   W₀ (d×k)  │ (frozen)
└─────────────┘
       +
    ┌───┐     ┌───┐
    │ B │  ×  │ A │
    │d×r│     │r×k│
    └───┘     └───┘
    可訓練參數：d×r + r×k
```

---

## 🔬 數學原理

### 1. 低秩分解 (Low-Rank Decomposition)

**定理：** 任何矩陣 M ∈ ℝ^(m×n) 都可以分解為：

```
M = UΣVᵀ  (SVD)

其中：
- U ∈ ℝ^(m×r): 左奇異向量
- Σ ∈ ℝ^(r×r): 奇異值對角矩陣
- V ∈ ℝ^(n×r): 右奇異向量
- r = rank(M)
```

**低秩近似：** 保留前 k 個最大奇異值：

```
M ≈ M_k = U_k Σ_k V_k^T

其中 k << r
```

### 2. LoRA 前向傳播

```python
# 原始線性層
h = W₀x

# LoRA 修改後
h = W₀x + BAx
  = W₀x + (BA)x

其中：
- x ∈ ℝ^k: 輸入
- W₀ ∈ ℝ^(d×k): 凍結的原始權重
- B ∈ ℝ^(d×r): 可訓練
- A ∈ ℝ^(r×k): 可訓練
```

### 3. 縮放因子 Alpha

```python
h = W₀x + (α/r) × BAx

其中：
- α: 縮放超參數 (通常設為 rank 的 1-2 倍)
- r: rank
```

**為什麼需要 α/r？**
- 控制 LoRA 權重的影響程度
- 不同 rank 之間的學習率標準化
- 類似 Layer Normalization 的概念

---

## 📊 參數量與記憶體分析

### 參數量計算

假設對 LLaMA-7B 的一個 Attention 層使用 LoRA：

```
原始權重：
- Q: 4096 × 4096 = 16M
- K: 4096 × 4096 = 16M
- V: 4096 × 4096 = 16M
- O: 4096 × 4096 = 16M
總計：64M 參數

LoRA (r=8):
每個矩陣：
- B: 4096 × 8 = 32,768
- A: 8 × 4096 = 32,768
- 小計：65,536

四個矩陣：65,536 × 4 = 262,144 (0.26M)

參數減少：64M / 0.26M ≈ 246×
```

### 記憶體計算

```
Full Fine-tuning (7B model):
- Weights: 14GB
- Optimizer: 28GB
- Gradients: 14GB
- Total: ~56GB

LoRA (r=8):
- Frozen weights: 14GB (no grad)
- LoRA weights: ~30MB
- LoRA optimizer: ~60MB
- LoRA gradients: ~30MB
- Total: ~14.12GB

記憶體減少：~4×
```

---

## 💻 實作細節

### 1. LoRA Layer 實作

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # LoRA 可訓練權重
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 縮放
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, in_features)
        output: (batch, seq_len, out_features)
        """
        # LoRA 路徑：x → A → dropout → B
        lora_out = self.lora_B @ (self.lora_A @ x.T)
        lora_out = self.dropout(lora_out.T)

        return lora_out * self.scaling
```

### 2. 應用到 Linear Layer

```python
class LinearWithLoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
        )

        # 凍結原始權重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始輸出 + LoRA 輸出
        return self.linear(x) + self.lora(x)
```

### 3. 應用到 Transformer

```python
def apply_lora_to_model(model, rank=8, alpha=16):
    """將 LoRA 應用到模型的所有 Linear 層"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 通常只對 Q, K, V, O 使用 LoRA
            if any(key in name for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = model.get_submodule(parent_name)
                setattr(
                    parent,
                    child_name,
                    LinearWithLoRA(module, rank=rank, alpha=alpha)
                )
    return model
```

---

## 🎛️ 超參數調整

### Rank (r)

**影響：**
- ⬆️ 更高的 rank → 更強的表達能力
- ⬆️ 更高的 rank → 更多參數
- ⬇️ 過高的 rank → 可能過擬合

**經驗法則：**
```
Small models (< 1B):  r = 4-8
Medium models (1-7B): r = 8-16
Large models (> 7B):  r = 16-64
```

### Alpha (α)

**影響：**
- ⬆️ 更高的 α → LoRA 權重影響更大
- ⬇️ 過高的 α → 可能破壞預訓練知識

**經驗法則：**
```
α = r     # 標準設置
α = 2r    # 增強 LoRA 影響
α = r/2   # 保守設置
```

### Dropout

**建議：**
```
Small dataset: 0.1-0.2
Large dataset: 0.0-0.05
```

---

## 📈 效能對比

### 實驗結果 (論文數據)

| 模型 | 方法 | 參數量 | GLUE Score |
|------|------|--------|------------|
| GPT-3 175B | Full FT | 175B | 89.5 |
| GPT-3 175B | Adapter | 40M | 88.2 |
| GPT-3 175B | LoRA (r=4) | 4.7M | 89.3 |
| GPT-3 175B | LoRA (r=64) | 37.7M | **89.7** |

**結論：**
- ✅ LoRA 用 0.02% 的參數達到全參數微調的效果
- ✅ 甚至在某些任務上超越全參數微調

---

## 🔍 進階話題

### 1. LoRA 的理論保證

**假設：** 預訓練模型已經學習了一個高維空間的通用表示

**微調時：** 只需要在這個表示的低維子空間中進行調整

**數學證明：** (簡化版)
```
設 W₀ 是預訓練權重
微調目標：min_W L(W)

泰勒展開：
L(W₀ + ΔW) ≈ L(W₀) + ∇L(W₀)ᵀΔW + ...

如果 ∇L(W₀) 存在於低秩子空間，
則 ΔW 也可以低秩表示
```

### 2. LoRA 與其他方法的關係

```
Adapter ⊂ LoRA ⊂ Full Fine-tuning

其中：
- Adapter: 串行架構，增加推論延遲
- LoRA: 並行架構，零推論延遲
- Full FT: 所有參數可訓練
```

### 3. 合併 LoRA 權重

```python
def merge_lora_weights(model):
    """訓練後合併 LoRA 權重到原始權重"""
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            # W' = W₀ + BA
            merged_weight = (
                module.linear.weight.data +
                (module.lora.lora_B @ module.lora.lora_A) * module.lora.scaling
            )
            module.linear.weight.data = merged_weight

            # 移除 LoRA 層
            module.lora = nn.Identity()
```

---

## 🧪 實驗建議

### 最佳實踐

1. **選擇合適的層**
   - ✅ Attention 的 Q, K, V, O
   - ✅ FFN 的 up_proj, down_proj
   - ❌ Embedding, LayerNorm

2. **Rank 選擇策略**
   ```python
   # 從小開始，逐步增加
   ranks_to_try = [4, 8, 16, 32]

   for r in ranks_to_try:
       model = apply_lora(base_model, rank=r)
       score = evaluate(model)
       print(f"Rank {r}: {score}")
   ```

3. **學習率調整**
   ```python
   # LoRA 層需要更高的學習率
   optimizer = AdamW([
       {'params': lora_params, 'lr': 1e-4},  # LoRA 層
       {'params': other_params, 'lr': 1e-5}, # 其他層（如果有）
   ])
   ```

---

## 📚 延伸閱讀

### 必讀論文

1. **LoRA 原始論文**
   - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
   - Microsoft, 2021

2. **理論分析**
   - [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)

3. **改進版本**
   - [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
   - [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

---

## ❓ 常見問題

### Q1: LoRA 會降低模型性能嗎？

**A**: 不一定。在大多數情況下：
- 小數據集：LoRA 可能更好（避免過擬合）
- 大數據集：LoRA ≈ Full Fine-tuning
- 極大數據集：Full Fine-tuning 可能略勝

### Q2: 為什麼 LoRA 有效？

**A**: 核心原因：
1. **內在維度假說**：任務適應只需要低維子空間
2. **過擬合防護**：參數限制提供正則化
3. **預訓練知識保留**：凍結原始權重

### Q3: rank 如何影響效能？

**A**:
- 太小 (r < 4)：表達能力不足
- 適中 (r = 8-16)：平衡性能與效率
- 太大 (r > 64)：邊際收益遞減，接近全參數

---

## 🚀 下一步

完成 LoRA 理論學習後：

1. ✅ **實作練習**：[Task 01 - LoRA 基礎實作](../lab_tasks/task01_lora_basic/)
2. 📖 **進階學習**：[QLoRA 與量化](02_qlora_quantization.md)
3. 🔬 **實驗調參**：嘗試不同 rank 與 alpha 組合

---

<div align="center">

**理解原理，掌握實作！💡**

[← 返回總覽](00_overview.md) | [下一篇：QLoRA 量化 →](02_qlora_quantization.md)

</div>
