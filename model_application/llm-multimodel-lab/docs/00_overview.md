# LLM 量化實驗室總覽

> Quantization Techniques for Large Language Models
> 從 Post-Training Quantization 到 Quantization-Aware Training 的完整探索

---

## 你可能遇到的問題

想像一下這個場景：

你是一位 Edge AI 工程師,需要將一個 LLaMA-7B 模型部署到邊緣設備上。你打開電腦,發現：

- **問題 1**: 模型權重 14GB,但目標設備只有 4GB 記憶體
- **問題 2**: 推論速度太慢,無法滿足實時性要求
- **問題 3**: 量化後精度大幅下降,不知道如何平衡
- **問題 4**: PTQ、QAT、Mixed-Precision 傻傻分不清楚
- **問題 5**: 不理解量化背後的數學原理,只會調用工具

這些問題,就是這個實驗室要解決的。

**這個專案的核心理念**：
> 不只是使用量化工具,而是從數學原理到工程實踐,深入理解模型壓縮的每一個細節。

---

## 為什麼需要量化？一個實際場景

### 場景：部署 LLaMA-7B 到邊緣設備

**硬體限制**：
```
目標設備: NVIDIA Jetson AGX Orin
GPU 記憶體: 32GB 共享記憶體
算力: 275 TOPS (INT8)
功耗限制: < 60W
```

**挑戰**：

**FP32 模型 (原始精度)**:
```
模型權重: 28 GB  ❌ 超出記憶體限制!
推論速度: 12 tokens/s
功耗: ~85W
```

**FP16 模型 (混合精度)**:
```
模型權重: 14 GB  ⚠️ 勉強可用,但無法處理大 batch
推論速度: 24 tokens/s
功耗: ~75W
```

**INT8 量化模型 (PTQ)**:
```
模型權重: 7 GB   ✅ 記憶體充裕
推論速度: 85 tokens/s  ✅ 3.5× 加速
功耗: ~45W  ✅ 符合限制
準確率下降: 2-5%  ⚠️ 可接受範圍
```

**INT4 量化模型 (QLoRA + QAT)**:
```
模型權重: 3.5 GB  ✅ 更小
推論速度: 120 tokens/s  ✅ 5× 加速
功耗: ~35W  ✅ 理想
準確率下降: 1-3%  ✅ 經過 QAT 優化
```

**關鍵洞察**：
> 量化不只是壓縮模型,而是在**記憶體、速度、精度、功耗**之間找到最佳平衡點。

---

## 從 LoRA 到量化：完整的模型優化路徑

### 技術堆疊的邏輯順序

```
第一步：LoRA (參數高效微調)
目標: 降低微調時的可訓練參數量
記憶體節省: 99% 可訓練參數
問題: 凍結權重仍佔大量記憶體

       ↓

第二步：QLoRA (量化 + LoRA)
目標: 壓縮凍結權重的儲存空間
記憶體節省: 4× (FP16 → 4-bit)
問題: 僅適用於訓練階段

       ↓

第三步：Precision-Aware Training
目標: 訓練時識別對精度敏感的層
技術: Mixed-Precision + 敏感度分析
成果: 為不同層分配不同量化精度

       ↓

第四步：Quantization-Aware Training (QAT)
目標: 訓練時模擬量化,適應精度損失
技術: Fake Quantization + 量化感知訓練
成果: 極低精度 (INT4) 下仍保持性能

       ↓

第五步：部署優化
目標: 實際部署到邊緣設備
技術: TensorRT、ONNX Runtime、模型融合
成果: 生產級推論系統
```

**本實驗室的範圍**：專注於前四步的**模型量化技術**,不涉及工程化部署。

---

## 五個核心模組：從理論到實踐

### Task 01: LoRA 基礎

**目標**: 理解低秩分解如何節省參數

**核心概念**:
```
W' = W₀ + BA

其中:
- W₀: 凍結的預訓練權重 (110M 參數)
- B, A: 可訓練的低秩矩陣 (0.6M 參數)
- 參數節省: 99.5%
```

**學習重點**:
- 手寫 `LoRALayer` 實作
- 推導前向傳播和梯度計算
- 理解 rank 和 alpha 的影響

**為什麼先學 LoRA**：
量化技術通常與 LoRA 結合使用 (QLoRA),理解 LoRA 是學習量化的基礎。

---

### Task 02: QLoRA (Post-Training Quantization 基礎)

**目標**: 將凍結權重量化為 4-bit,保持 LoRA 參數高精度

**核心技術**:

1. **NF4 (4-bit NormalFloat)**
   - 為正態分布權重設計的分位數量化
   - 精度優於傳統 INT4

2. **雙重量化**
   - 對量化常數 (scale) 再量化
   - 額外節省 8% 記憶體

3. **混合精度計算**
   - 儲存: 4-bit NF4
   - 計算: BF16 (動態反量化)

**記憶體對比**:
```
FP16:  14 GB
INT8:  7 GB   (50% 節省)
INT4:  3.5 GB (75% 節省)
```

**學習重點**:
- 實作 NF4 量化算法
- 理解量化誤差來源
- 掌握 BitsAndBytesConfig 配置

---

### Task 03: Precision-Aware Training

**目標**: 識別對量化敏感的層,分配不同精度

**核心問題**:
> 不是所有層都需要相同精度,如何找出敏感層?

**技術方案**:

1. **敏感度分析 (Sensitivity Analysis)**
   ```python
   for layer in model.layers:
       # 量化該層
       quantized_layer = quantize(layer, bits=8)

       # 測試精度下降
       accuracy_drop = evaluate(quantized_layer)

       # 記錄敏感度
       sensitivity[layer] = accuracy_drop
   ```

2. **混合精度配置**
   ```
   Attention Q,K,V:  INT8  (敏感度高)
   Attention Output: INT8  (敏感度高)
   FFN Layer 1:      INT4  (敏感度低)
   FFN Layer 2:      INT4  (敏感度低)
   LayerNorm:        FP16  (參數少,保持高精度)
   ```

**實驗結果** (LLaMA-7B):
```
全 INT8:       準確率 72.1%,記憶體 7 GB
全 INT4:       準確率 68.3%,記憶體 3.5 GB
混合精度:      準確率 71.8%,記憶體 4.5 GB  ✅ 最佳平衡
```

**學習重點**:
- 實作 SQNR (Signal-to-Quantization-Noise Ratio) 計算
- 理解 Hessian-based 敏感度分析
- 設計混合精度策略

---

### Task 04: Quantization-Aware Training (QAT)

**目標**: 訓練時模擬量化,讓模型適應低精度

**核心思想**:
> 訓練時插入「假量化」節點,讓梯度流過量化過程,模型學會容忍量化誤差。

**Fake Quantization 機制**:

```python
def fake_quantize(x, bits=8):
    """
    訓練時模擬量化,但保持梯度流動
    """
    # 前向傳播: 真的量化
    scale = (x.max() - x.min()) / (2**bits - 1)
    x_quant = torch.round(x / scale) * scale

    # 反向傳播: 直通估計器 (Straight-Through Estimator)
    x_quant = x + (x_quant - x).detach()

    return x_quant
```

**訓練流程**:

```
1. Forward Pass:
   x → [Fake Quant] → W @ x → [Fake Quant] → output

2. Backward Pass:
   梯度繞過量化節點 (STE),直接回傳

3. Weight Update:
   權重以 FP32 更新,下次 forward 再量化
```

**QAT vs PTQ 對比**:

| 方法 | 準確率 (INT4) | 訓練時間 | 記憶體需求 |
|------|--------------|---------|-----------|
| PTQ (直接量化) | 68.3% | 0 (無需訓練) | 低 |
| **QAT (量化感知訓練)** | **71.5%** | +20% 訓練時間 | 中 |
| Full Precision | 73.2% | 基準 | 高 |

**學習重點**:
- 實作 Straight-Through Estimator (STE)
- 理解量化感知的梯度計算
- 掌握 QAT 訓練超參數調整

---

### Task 05: 量化技術總結與實戰

**目標**: 整合所有技術,完成端到端的量化流程

**綜合實驗**:

```
情境: 將 LLaMA-7B 微調後的模型量化到 INT4,部署到邊緣設備

步驟 1: 用 LoRA 微調模型
→ 產出: adapter 權重 (30MB)

步驟 2: 合併 adapter 到 base model
→ 產出: 微調後的 FP16 模型 (14GB)

步驟 3: 敏感度分析
→ 產出: 混合精度配置策略

步驟 4: QAT 訓練
→ 產出: 量化感知的 INT4 模型 (3.5GB)

步驟 5: 評測與對比
→ 產出: 精度、速度、記憶體報告
```

**最終成果**:

| 指標 | FP16 | PTQ INT4 | QAT INT4 |
|------|------|----------|----------|
| 模型大小 | 14 GB | 3.5 GB | 3.5 GB |
| 推論速度 | 1× | 4× | 4× |
| 準確率 | 73.2% | 68.3% | 71.5% |
| **精度損失** | **0%** | **6.7%** | **2.3%** ✅ |

**學習重點**:
- 整合 LoRA + Quantization 完整流程
- 理解不同量化技術的適用場景
- 掌握量化模型的評測方法

---

## 技術選型：為什麼用這些工具？

### 量化框架選擇

| 框架 | 優勢 | 劣勢 | 本實驗室選擇 |
|------|------|------|------------|
| **bitsandbytes** | NF4 格式,與 HF 無縫整合 | 僅支援 CUDA | ✅ Task 02 (QLoRA) |
| **torch.quantization** | PyTorch 原生,靈活度高 | 需手動配置 | ✅ Task 04 (QAT) |
| **TensorRT** | 極致推論性能 | 學習曲線陡,部署導向 | ❌ 超出實驗室範圍 |
| **ONNX Runtime** | 跨平台,部署友善 | 量化選項有限 | ❌ 超出實驗室範圍 |

**選擇理由**:
- **bitsandbytes**: 學習 NF4 和 QLoRA 的最佳工具
- **torch.quantization**: 理解量化原理的底層接口
- **不選 TensorRT/ONNX**: 專注於模型層面,不涉及工程部署

---

### PTQ vs QAT：何時用哪個？

**決策樹**:

```
                    需要量化模型？
                         │
                         ├─ 是
                         │   │
                         │   └─ 有額外訓練資源？
                         │       │
                         │       ├─ 有 → QAT
                         │       │    (更高精度,需重訓練)
                         │       │
                         │       └─ 沒有 → PTQ
                         │            (快速量化,略損精度)
                         │
                         └─ 否 → 保持 FP16/BF16
```

**實際案例**:

| 場景 | 推薦方案 | 理由 |
|------|---------|------|
| 部署預訓練模型 (如 LLaMA) | PTQ (INT8) | 無需重訓練,精度損失小 |
| 微調後部署 (如客服機器人) | QAT (INT4) | 已有訓練流程,可接受額外訓練成本 |
| 實時推論 (如語音助手) | QAT (INT8) + TensorRT | 需極致性能 |
| 資源極度受限 (IoT 設備) | QAT (INT4) + 稀疏化 | 記憶體和算力雙重限制 |

---

## 學習路線：從入門到精通

### Week 1: 理解 LoRA 與參數高效微調

**學習任務**:
1. 閱讀 `01_lora_theory.md`
2. 完成 Task 01: 手寫 LoRA 實作
3. 實驗不同 rank 對精度的影響

**自我檢查**:
- [ ] 能在白板上推導 `W' = W₀ + BA`
- [ ] 理解為什麼 rank=8 通常是最佳選擇
- [ ] 能解釋 LoRA 如何節省記憶體

---

### Week 2: 掌握 QLoRA 與 NF4 量化

**學習任務**:
1. 閱讀 `02_qlora_quantization.md`
2. 完成 Task 02: 實作 NF4 量化
3. 對比 INT4 vs NF4 的精度差異

**自我檢查**:
- [ ] 能解釋 NF4 的分位數量化原理
- [ ] 理解雙重量化如何節省記憶體
- [ ] 掌握 BitsAndBytesConfig 配置

**常見卡關點**:
- **量化後精度下降太多**: 檢查是否使用 BF16 作為 compute_dtype
- **記憶體仍不足**: 確認 `use_double_quant=True`
- **訓練不穩定**: 降低學習率,使用 warmup

---

### Week 3: 探索混合精度與敏感度分析

**學習任務**:
1. 閱讀 `03_precision_aware.md`
2. 完成 Task 03: 實作層級敏感度分析
3. 設計混合精度配置策略

**自我檢查**:
- [ ] 能計算 SQNR (Signal-to-Quantization-Noise Ratio)
- [ ] 理解為什麼 Attention 層比 FFN 層敏感
- [ ] 能設計自定義的混合精度配置

**實驗提示**:
- 先對單層量化,觀察精度變化
- 用 Hessian 近似判斷敏感度
- 嘗試不同的精度組合 (8/4/4, 8/8/4, ...)

---

### Week 4: 掌握 Quantization-Aware Training

**學習任務**:
1. 閱讀 `04_qat.md`
2. 完成 Task 04: 實作 QAT 訓練流程
3. 對比 PTQ vs QAT 的精度差異

**自我檢查**:
- [ ] 能實作 Straight-Through Estimator
- [ ] 理解 Fake Quantization 的前向/反向傳播
- [ ] 掌握 QAT 的學習率調整策略

**常見卡關點**:
- **QAT 後精度沒提升**: 檢查學習率是否過大,導致過擬合
- **訓練不收斂**: 確認 STE 實作正確,梯度能正常回傳
- **比 PTQ 還差**: 可能訓練時間不足,增加 epochs

---

### Week 5: 綜合實戰與系統評測

**學習任務**:
1. 閱讀 `05_summary.md`
2. 完成 Task 05: 端到端量化流程
3. 撰寫量化技術對比報告

**實戰挑戰**:
- 能否在 30 分鐘內量化一個 LLaMA-7B 模型？
- 能否設計一套自動化的混合精度搜索算法？
- 能否在精度損失 < 2% 的前提下,將模型壓縮到 INT4？

---

## 完成後你能做什麼？

### 技術面試時

**面試官**: 「你了解模型量化嗎？」
**你**: 「我不只了解,我手寫過 NF4 量化算法,能解釋為什麼分位數量化優於線性量化。」

**面試官**: 「PTQ 和 QAT 有什麼區別？」
**你**: 「我實作過兩種方法,PTQ 是訓練後直接量化,速度快但精度略低；QAT 是訓練時模擬量化,精度高但需重訓練。我可以根據場景選擇合適方案。」

**面試官**: 「如何處理量化後精度下降？」
**你**: 「我會先做敏感度分析,識別關鍵層保持高精度；然後用 QAT 訓練,讓模型適應量化誤差；最後用混合精度策略平衡性能和資源。」

---

### 實際工作中

**場景 1**: 老闆要求將 13B 模型部署到邊緣設備
- 你能快速評估量化方案 (INT8 vs INT4)
- 你能預測精度損失範圍 (基於敏感度分析)
- 你能設計 QAT 訓練流程,最小化精度損失

**場景 2**: 量化後模型性能不佳
- 你能診斷問題 (是量化配置錯誤?還是敏感層未保護?)
- 你能設計實驗驗證假設
- 你能提出改進方案 (調整精度配置、增加 QAT 訓練等)

**場景 3**: 需要設計自動化量化流程
- 你理解底層原理,能整合不同工具
- 你能設計敏感度分析自動化流程
- 你能實作混合精度搜索算法

---

## 延伸閱讀

### 必讀論文

1. **LoRA 原始論文**
   [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

2. **QLoRA 論文**
   [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

3. **Quantization Primer**
   [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)

4. **QAT 經典論文**
   [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

---

### 實用資源

- [PyTorch Quantization Tutorial](https://pytorch.org/docs/stable/quantization.html)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [Hugging Face PEFT Library](https://github.com/huggingface/peft)
- [NVIDIA TensorRT Quantization Guide](https://docs.nvidia.com/deeplearning/tensorrt/)

---

## 常見疑問

### 「我只會 PyTorch 基礎,能學嗎？」

**能**,但需要補充線性代數基礎:
- 理解矩陣乘法、奇異值分解
- 了解梯度計算和鏈式法則
- 熟悉 PyTorch 的 `nn.Module` 和 `autograd`

**建議**:
- 先完成 PyTorch 官方教學的前 3 章
- 複習線性代數的矩陣運算
- 邊學邊查數學公式,不必全部背下來

---

### 「沒有 GPU 能學嗎？」

**理論部分可以**,實作部分會很困難:
- Task 01-02 需要 GPU (至少 12GB VRAM)
- Task 03-05 建議使用 GPU,但可用 CPU 替代 (很慢)

**替代方案**:
- Google Colab (免費 T4,15GB VRAM)
- Kaggle Notebooks (免費 P100)
- 租用雲端 GPU (AWS, GCP, Lambda Labs)

---

### 「學完能直接部署到生產環境嗎？」

**不能,但你會具備核心能力**:

**本實驗室教的**:
- 量化原理與數學基礎
- PTQ、QAT 實作方法
- 混合精度策略設計

**生產環境還需要**:
- 推論引擎整合 (TensorRT, ONNX Runtime)
- 模型融合與優化 (Kernel Fusion, Graph Optimization)
- 服務化部署 (API 設計、負載均衡、監控)

**建議學習路徑**:
完成本實驗室 → 學習 TensorRT/ONNX → 整合到實際系統

---

### 「量化會損失多少精度？」

**典型範圍** (基於 LLaMA-7B):

| 量化方法 | 精度損失 | 適用場景 |
|---------|---------|---------|
| INT8 (PTQ) | 0.5-2% | 一般部署 |
| INT8 (QAT) | < 0.5% | 高精度要求 |
| INT4 (PTQ) | 3-7% | 資源極度受限 |
| INT4 (QAT) | 1-3% | 平衡性能與資源 |
| Mixed-Precision (8/4) | 1-2% | 最佳平衡 |

**影響因素**:
- 模型架構 (Attention 敏感度高)
- 量化方法 (QAT 優於 PTQ)
- 資料分布 (校準資料集的選擇)

---

## 開始你的學習之旅

### 建議的第一步

**如果你想先了解全貌**:
- 花 20 分鐘瀏覽這份 overview
- 看看 [01_lora_theory.md](01_lora_theory.md) 的前半部分
- 理解「為什麼需要量化」

**如果你想直接動手**:
- 跳到 [Task 01](../lab_tasks/task01_lora/)
- 按照指引逐步實作
- 遇到不懂的再回來看理論

**如果你想評估是否適合**:
- 先看「常見疑問」部分
- 確認硬體和時間是否足夠
- 可以先跑 Task 01 試試手感

---

### 學習心態建議

**這個專案不是為了**:
- 快速完成拿證書 (沒有證書)
- 複製貼上就能跑的代碼 (需要思考)
- 讓你馬上成為專家 (需要持續學習)

**這個專案是為了**:
- 理解量化的底層數學原理
- 培養模型優化的系統思維
- 建立解決實際問題的能力

---

### 準備好了嗎？

選擇你的路徑：

1. **我要從理論開始** → [LoRA 理論](01_lora_theory.md)
2. **我要直接動手** → [Task 01: LoRA 基礎](../lab_tasks/task01_lora/)
3. **我想深入量化** → [QLoRA 量化理論](02_qlora_quantization.md)

---

> **最後提醒**
>
> 學習模型量化不是一蹴而就的過程,理解數學原理需要時間。
>
> 每完成一個 task,你就比 95% 只會調用 API 的人更懂原理。
>
> 量化不只是壓縮模型,而是理解深度學習的本質。
>
> 祝你學習愉快！
