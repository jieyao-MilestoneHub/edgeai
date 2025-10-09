# Task 02: QLoRA 實戰與量化比較

> 使用 4-bit 量化技術實現記憶體高效微調

## 🎯 學習目標

- ✅ 理解量化原理（INT8, INT4, NF4）
- ✅ 實作 QLoRA 訓練
- ✅ 比較 LoRA vs QLoRA 的記憶體與精度
- ✅ 使用 bitsandbytes 進行 4-bit 訓練

## 📋 前置知識

- 完成 Task 01
- 理解量化基本概念

## 🚀 核心內容

### 1. 4-bit NormalFloat (NF4) 量化
### 2. 雙重量化 (Double Quantization)
### 3. QLoRA 訓練實作
### 4. FP16 vs QLoRA 對比實驗

## 📊 預期成果

- 使用 <50% 記憶體訓練 7B 模型
- 量化誤差 <1%
- 理解量化 tradeoffs

詳見 [GUIDE.md](GUIDE.md)
