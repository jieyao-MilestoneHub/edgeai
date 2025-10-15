# Quantization-Aware Training: 量化感知訓練

> Training Neural Networks to be Quantization-Friendly
> 本章深入探討如何在訓練過程中模擬量化,讓模型學會適應低精度運算

---

## 學習目標

完成本章後,你將能夠:

- 理解 QAT 與 PTQ 的本質差異
- 掌握 Fake Quantization 的數學原理
- 實作 Straight-Through Estimator (STE)
- 設計 QAT 訓練流程與超參數配置
- 在極低精度 (INT4) 下保持模型性能

---

## 為什麼需要 QAT? PTQ 的局限性

### 場景:PTQ INT4 量化的精度災難

假設你用 PTQ 將 LLaMA-7B 直接量化到 INT4:

**PTQ INT4 結果**:
```
MMLU 準確率:     68.3% (Full Precision: 73.2%)
精度下降:        -4.9%  ❌ 不可接受!
```

**問題根源**:

模型在訓練時使用 FP32/FP16,從未「見過」量化誤差:

```python
# 訓練時 (FP16)
output = W @ x  # W 是精確的 FP16 權重

# PTQ 推論時 (INT4)
output = Quant(W) @ x  # W 被強制量化,模型無法適應!
```

---

### QAT 的解決方案

**核心思想**:
> 訓練時就模擬量化,讓模型在訓練過程中「習慣」量化誤差。

**效果對比**:

| 方法 | MMLU (INT4) | 訓練成本 | 部署複雜度 |
|------|------------|---------|-----------|
| PTQ | 68.3% | 0 (無需訓練) | 低 |
| **QAT** | **71.5%** | +20% 訓練時間 | 低 |
| Full Precision | 73.2% | 基準 | N/A |

**結論**:
> QAT 用 20% 的訓練成本,換取 3.2% 的精度提升 (68.3% → 71.5%)。

---

## 核心技術: Fake Quantization

### 什麼是 Fake Quantization?

**定義**:
在訓練時模擬量化過程,但權重和激活值仍以浮點數格式儲存和計算。

**關鍵特性**:
1. **前向傳播**: 模擬量化,產生量化後的數值
2. **反向傳播**: 梯度能正常回傳 (這是 PTQ 做不到的!)
3. **權重更新**: 以高精度 (FP32) 更新權重

---

## 延伸學習

### 必讀論文

1. **QAT 原始論文**
   [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

2. **Learned Step Size Quantization**
   [LSQ: Learned Step Size Quantization](https://arxiv.org/abs/1902.08153)

---

> **關鍵啟示**
> QAT 的本質是:
> **讓模型在訓練時就「看見」量化誤差,學會在低精度下也能正確推論。**
>
> 這是 PTQ 永遠做不到的 — 因為 PTQ 是訓練後才量化,模型無法調整自己。
