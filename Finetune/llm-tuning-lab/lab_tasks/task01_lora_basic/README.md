# Task 01: LoRA 基礎實作

> 從零開始手寫 LoRA 模組，理解參數高效微調核心原理

## 🎯 學習目標

完成本任務後，你將能夠：

- ✅ **理解 LoRA 數學原理**：低秩分解與矩陣分解
- ✅ **手寫 LoRA 模組**：實作完整的 LoRALayer
- ✅ **應用到 Transformer**：將 LoRA 整合到語言模型
- ✅ **訓練與評估**：完成一次完整的微調流程
- ✅ **分析參數與記憶體**：量化 LoRA 的效率優勢

---

## 📋 前置知識

### 必備知識
- Python 程式設計（中級）
- PyTorch 基礎（`nn.Module`, `nn.Linear`）
- Transformer 基本概念（Attention 機制）

### 建議預習
- 📖 [LoRA 理論](../../docs/01_lora_theory.md)
- 📄 [LoRA 論文](https://arxiv.org/abs/2106.09685)

---

## 🛠️ 環境設定

### 1. 依賴套件

```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install datasets
pip install matplotlib
pip install tensorboard
```

### 2. 硬體需求

- **最低配置**：GTX 1080 Ti (11GB)
- **建議配置**：RTX 3090 (24GB)
- **CPU Only**：可執行但速度較慢

---

## 📂 任務檔案結構

```
task01_lora_basic/
├── README.md              # 本文件
├── GUIDE.md               # 詳細教學指引
├── lora_linear.py         # 手寫 LoRA 模組
├── train_lora_basic.py    # 訓練腳本
├── config.yaml            # 訓練配置
├── utils.py               # 輔助函數
├── checklist.md           # 驗收清單
├── discussion.md          # 延伸問題
└── expected_output/       # 預期輸出範例
    ├── loss_curve.png
    ├── adapter_model.bin
    └── training_log.txt
```

---

## 🚀 快速開始

### Step 1: 閱讀指引

```bash
cat GUIDE.md
```

### Step 2: 檢查 LoRA 實作

```bash
# 檢視手寫 LoRA 模組
cat lora_linear.py
```

### Step 3: 執行訓練

```bash
# 使用預設配置訓練
python train_lora_basic.py --config config.yaml

# 自訂參數
python train_lora_basic.py \
    --model gpt2 \
    --rank 8 \
    --alpha 16 \
    --epochs 3 \
    --lr 2e-4
```

### Step 4: 查看結果

```bash
# 查看訓練曲線
tensorboard --logdir ./logs

# 檢查輸出檔案
ls -lh output/
```

---

## 📊 預期成果

完成訓練後，你應該獲得：

### 1. 輸出檔案

```
output/
├── adapter_model.bin           # LoRA 權重 (~2MB)
├── adapter_config.json         # LoRA 配置
├── training_loss_curve.png     # Loss 曲線圖
├── training_metrics.json       # 訓練指標
└── model_comparison.txt        # 參數量對比
```

### 2. 訓練指標

- **訓練 Loss**：從 ~3.5 降到 <2.0
- **驗證 Loss**：<2.5
- **訓練時間**：~15-30 分鐘 (RTX 3090)

### 3. 參數效率

```
原始模型參數：124M (GPT-2)
LoRA 可訓練參數：~294K (rank=8)
參數減少：~422× 🎉
記憶體節省：~3.5× 🚀
```

---

## 🎓 實作重點

### 核心概念

#### 1. LoRA 矩陣分解

```python
# 原始權重更新
W' = W₀ + ΔW  # ΔW 是 d×k 矩陣

# LoRA 低秩近似
W' = W₀ + BA  # B: d×r, A: r×k, r << min(d,k)
```

#### 2. 前向傳播

```python
class LoRALayer:
    def forward(self, x):
        # 基礎輸出（凍結）
        base_out = self.linear(x)

        # LoRA 路徑
        lora_out = self.lora_B(self.lora_A(x))

        # 合併輸出
        return base_out + lora_out * self.scaling
```

#### 3. 參數初始化

```python
# A: Kaiming uniform (有梯度)
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

# B: 零初始化（確保初始時 LoRA 輸出為 0）
nn.init.zeros_(self.lora_B)
```

---

## 🔍 關鍵實驗

### 實驗 1：不同 Rank 的影響

```bash
# 測試多個 rank 值
for rank in 2 4 8 16 32; do
    python train_lora_basic.py --rank $rank --output rank_${rank}
done

# 比較結果
python utils.py compare_ranks --results_dir ./results
```

**預期觀察：**
- rank=2：可訓練但效果較差
- rank=8：平衡點，推薦值
- rank=32：效果接近全參數微調

### 實驗 2：Alpha 縮放係數

```bash
# 固定 rank=8，測試不同 alpha
for alpha in 4 8 16 32; do
    python train_lora_basic.py --rank 8 --alpha $alpha
done
```

**預期觀察：**
- alpha 過小：LoRA 影響不足
- alpha 過大：可能破壞預訓練知識
- alpha = rank * 2：經驗最佳值

### 實驗 3：應用層選擇

```bash
# 只對 Q, V 使用 LoRA
python train_lora_basic.py --target_modules q_proj v_proj

# 對所有 Linear 層使用 LoRA
python train_lora_basic.py --target_modules all
```

---

## ✅ 驗收標準

完成後，請檢查 [checklist.md](checklist.md) 確認：

- [ ] `lora_linear.py` 實作正確
- [ ] 訓練成功完成，Loss 收斂
- [ ] 產生 `adapter_model.bin` 檔案
- [ ] 繪製 Loss 曲線圖
- [ ] 計算參數量並驗證節省效果
- [ ] 回答 `discussion.md` 中的延伸問題

---

## 🐛 常見問題

### Q1: 訓練時出現 CUDA Out of Memory

**A**: 嘗試以下方法：
```bash
# 減少 batch size
python train_lora_basic.py --batch_size 4

# 使用梯度累積
python train_lora_basic.py --gradient_accumulation_steps 4

# 使用更小的模型
python train_lora_basic.py --model distilgpt2
```

### Q2: Loss 沒有下降

**A**: 檢查以下項目：
- LoRA 參數是否正確設置 `requires_grad=True`
- 原始權重是否正確凍結
- 學習率是否合適（建議 1e-4 ~ 5e-4）

### Q3: 訓練速度很慢

**A**:
- 檢查是否在使用 GPU：`torch.cuda.is_available()`
- 減少序列長度：`--max_length 256`
- 使用混合精度訓練：`--fp16`

---

## 📚 延伸資源

### 進階閱讀
- [AdaLoRA 論文](https://arxiv.org/abs/2303.10512) - 動態調整 rank
- [IA³ 論文](https://arxiv.org/abs/2205.05638) - 另一種高效微調方法
- [Hugging Face PEFT 文檔](https://huggingface.co/docs/peft)

### 相關任務
- ⏭️ [Task 02: QLoRA 實戰](../task02_qlora/) - 加入量化技術
- ⏭️ [Task 03: SDK 與 API](../task03_sdk_api/) - 建立訓練服務

---

## 🤝 需要幫助？

如果遇到問題：

1. 查看 [GUIDE.md](GUIDE.md) 詳細教學
2. 閱讀 [discussion.md](discussion.md) 常見問題
3. 參考 [expected_output/](expected_output/) 範例輸出
4. 在 GitHub Issues 提問

---

<div align="center">

**準備好開始了嗎？🚀**

[開始學習 GUIDE.md](GUIDE.md) | [查看參考答案](lora_linear.py)

**祝學習順利！💪**

</div>
