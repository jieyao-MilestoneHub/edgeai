# Precision-Aware Training: 混合精度與敏感度分析

> Mixed-Precision Quantization and Layer Sensitivity Analysis
> 本章探討如何識別模型中對量化敏感的層,並設計最佳的混合精度配置策略

---

## 學習目標

完成本章後,你將能夠:

- 理解為什麼不同層對量化的敏感度不同
- 實作層級敏感度分析 (SQNR, Hessian-based)
- 設計混合精度量化策略
- 掌握 Precision-Aware Training 的工程實踐
- 在精度與資源之間找到最佳平衡點

---

## 為什麼需要混合精度?一個實際困境

### 場景:量化 LLaMA-7B 時的兩難

假設你用 PTQ 將 LLaMA-7B 量化到 INT4,結果發現:

**全模型 INT4 量化**:
```
模型大小: 3.5 GB  ✅ 符合預期
推論速度: 120 tokens/s  ✅ 5× 加速
準確率: 68.3%  ❌ 從 73.2% 下降 4.9%!
```

**分析發現**:

```python
# 逐層量化後測試精度
layer_accuracy = {
    "embedding": 73.1%,      # 幾乎無損
    "attention.q_proj": 69.5%,  # 下降 3.7% ❌
    "attention.k_proj": 69.2%,  # 下降 4.0% ❌
    "attention.v_proj": 70.1%,  # 下降 3.1% ❌
    "attention.o_proj": 69.8%,  # 下降 3.4% ❌
    "ffn.w1": 72.5%,         # 下降 0.7% ✅
    "ffn.w2": 72.8%,         # 下降 0.4% ✅
    "layernorm": 73.2%,      # 無影響
}
```

**關鍵發現**:
> Attention 層對量化極度敏感,而 FFN 層相對穩定!

**混合精度方案**:

```
策略: Attention 用 INT8,FFN 用 INT4

模型大小: 4.5 GB   (3.5 GB → 4.5 GB,+28%)
推論速度: 95 tokens/s  (略慢於全 INT4)
準確率: 71.8%  ✅ 僅下降 1.4%!
```

**結論**:
> 混合精度能在記憶體、速度、精度三者之間找到**帕累托最優解**。

---

## 核心概念:量化敏感度

### 什麼是量化敏感度?

**定義**:
量化敏感度衡量某一層的權重或激活值被量化後,對模型整體性能的影響程度。

**數學表示**:

```
S(layer_i) = |Accuracy(full_precision) - Accuracy(quantized_layer_i)|

其中:
- S(layer_i): 第 i 層的敏感度
- Accuracy(full_precision): 全精度模型的準確率
- Accuracy(quantized_layer_i): 僅量化第 i 層後的準確率
```

**敏感度分類**:

| 敏感度 | 精度下降 | 量化策略 | 範例層 |
|--------|---------|---------|--------|
| **高敏感** | > 2% | 保持 FP16/INT8 | Attention Q,K,V,O |
| **中敏感** | 0.5-2% | INT8 | FFN 第一層 |
| **低敏感** | < 0.5% | INT4 | FFN 第二層 |
| **不敏感** | ≈ 0% | INT4 或更低 | Embedding |

---

## 延伸閱讀

### 必讀論文

1. **HAWQ (Hessian AWare Quantization)**
   [HAWQ: Hessian Aware Quantization of Neural Networks](https://arxiv.org/abs/1905.03696)

2. **Mixed-Precision Quantization**
   [HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/abs/1811.08886)

---

> **關鍵啟示**
> Precision-Aware Training 的本質是:
> **在有限資源下,用數據驅動的方式找到精度與效率的最佳平衡點。**
