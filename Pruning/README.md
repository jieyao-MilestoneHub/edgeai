# Neural Network Pruning Bootcamp

**21-Day Enterprise-Grade Pruning Training Program**

Learn neural network pruning from unstructured to structured techniques, targeting **≥40% FLOPs reduction** with **≤1.5% accuracy drop** on ResNet-18.

---

## Goal

Achieve enterprise-grade pruning skills by mastering:
- **Unstructured pruning** (Week 1)
- **Structured pruning** (Week 2)
- **Advanced techniques** (Week 3)

**Target**: Prune ResNet-18 (CIFAR-10/ImageNet-mini) with ≥40% FLOPs reduction and ≤1.5% Top-1 accuracy loss, with clear justification for all design choices.

---

## Quick Start

### Installation

This project uses `uv` for package management (isolated from parent repo):

```bash
# Navigate to Pruning directory
cd Pruning

# Install dependencies (CPU-only, recommended for most users)
uv sync

# OR for CUDA 12.1
uv sync --extra cuda121

# OR for CUDA 12.4
uv sync --extra cuda124

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### Basic Usage

```bash
# 1. Train baseline model
python pruning-bootcamp/scripts/train.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --save-dir results/baseline

# 2. Evaluate baseline
python pruning-bootcamp/scripts/eval.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth

# 3. Apply structured pruning (Week 2)
python pruning-bootcamp/scripts/prune_structured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method bn_scale \
  --sparsity 0.5 \
  --save-dir results/structured

# 4. Profile and compare
python pruning-bootcamp/scripts/profile.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --baseline results/baseline/best_model.pth \
  --pruned results/structured/pruned_model.pth \
  --measure-latency
```

---

## 📘 核心腳本詳解

每個腳本都有明確的學習目標與預期輸出，確保您知道「為什麼執行」與「會得到什麼」。

### 🎯 train.py - 基準模型訓練

**學習目標:**
- 理解完整訓練流程：data loading → forward → backward → optimization
- 掌握 Metrics 追蹤：FLOPs, Params, Accuracy
- 學習 TensorBoard 可視化訓練曲線
- 建立 Baseline 作為後續剪枝對比基準

**關鍵概念:**
- Data augmentation (RandomCrop, RandomHorizontalFlip)
- Learning rate scheduling (MultiStepLR)
- Early stopping (保存最佳模型)
- Checkpoint management

**執行命令:**
```bash
python pruning-bootcamp/scripts/train.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --save-dir results/baseline
```

**預期輸出:**
```
=== Baseline Model Metrics ===
FLOPs: 0.56 G
Params: 11.17 M

Epoch 1: Train Loss=1.234, Train Acc=45.23%, Val Loss=1.012, Val Acc=62.45%
...
Epoch 100: Train Loss=0.123, Train Acc=95.67%, Val Loss=0.234, Val Acc=93.12%

✓ Saved best model with accuracy: 93.12%

=== Training Complete ===
Best Validation Accuracy: 93.12%
```

**輸出檔案:**
- `results/baseline/best_model.pth` - 最佳模型權重
- `results/baseline/logs/` - TensorBoard 訓練曲線
- `results/baseline/metrics.yaml` - 完整指標記錄

**常見問題:**
- **OOM (Out of Memory):** 減小 batch_size (128 → 64)
- **準確率過低:** 檢查 data augmentation 是否過強
- **訓練不穩定:** 降低 learning rate

---

### 📊 eval.py - 模型評估與指標分析

**學習目標:**
- 建立標準化評估流程
- 理解 FLOPs 計算原理 (理論運算量)
- 理解 Latency 測量方法 (實際推理時間)
- **核心認知: FLOPs ≠ Latency**

**關鍵概念:**
- Inference mode (torch.no_grad(), model.eval())
- GPU warmup (避免首次推理偏差)
- CUDA events 精確計時
- Metrics normalization (相對於 baseline)

**執行命令:**
```bash
python pruning-bootcamp/scripts/eval.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --measure-latency
```

**預期輸出:**
```
Using device: cuda

✓ Loaded checkpoint from results/baseline/best_model.pth

=== Evaluating Model ===
Evaluating: 100%|████████| 79/79 [00:12<00:00,  6.45it/s]
Accuracy: 93.12%
FLOPs: 0.56 G
Params: 11.17 M

=== Measuring Latency ===
Latency: 2.45 ms

=== Normalized Metrics ===
accuracy: 93.1200
flops_g: 0.56
params_m: 11.17
latency_ms: 2.45
throughput_fps: 408.16

✓ Metrics saved to results/baseline/eval_metrics.yaml
```

**輸出檔案:**
- `results/baseline/eval_metrics.yaml` - 完整評估指標

**常見問題:**
- **Latency 測量不穩定:** 增加 num_runs (200 → 500)
- **FLOPs 計算失敗:** 安裝 `thop` library: `pip install thop`

---

### ✂️ prune_unstructured.py - 非結構化剪枝

**學習目標:**
- 掌握 4 種非結構化剪枝方法
  1. **L1 Magnitude** (layerwise): 每層獨立剪枝
  2. **Global Magnitude**: 全局權重排序剪枝
  3. **Taylor Sensitivity**: |weight × gradient| 重要度
  4. **Movement Pruning**: 追蹤權重向零移動趨勢
- 理解 Layerwise vs Global 分配策略
- 掌握 One-shot vs Iterative 剪枝差異

**關鍵概念:**
- **Sparsity ≠ FLOPs 減少** (非結構化不減少計算量!)
- Fine-tuning 至關重要 (LR = 1/10 original)
- Iterative pruning 降低準確率下降
- 剪枝後需 `remove_pruning_reparameterization()`

**執行命令:**
```bash
# 方法 1: L1 layerwise
python pruning-bootcamp/scripts/prune_unstructured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method l1 --mode layerwise --sparsity 0.5

# 方法 2: Global magnitude
python pruning-bootcamp/scripts/prune_unstructured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method global --mode global --sparsity 0.5

# 方法 3: Iterative pruning
python pruning-bootcamp/scripts/prune_unstructured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method global --iterative --num-iterations 5 --sparsity 0.5
```

**預期輸出:**
```
=== Baseline Model ===
Accuracy: 93.12%
Params: 11.17 M

=== Pruning: GLOBAL (global) ===
Target Sparsity: 50.0%
Mode: One-shot

=== Fine-tuning ===
Fine-tune Epoch 1: Val Acc = 89.23%
...
Fine-tune Epoch 30: Val Acc = 91.67%

=== Pruned Model Metrics ===
Actual Sparsity: 50.12%
Final Accuracy: 91.67%
Accuracy Drop: 1.45%

✓ Results saved to results/unstructured/global_global_sp0.5
```

**輸出檔案:**
- `pruned_model.pth` - 剪枝後模型
- `metrics.yaml` - 剪枝指標對比

**常見問題:**
- **準確率大幅下降 (>3%):** 降低 sparsity 或增加 fine-tune epochs
- **FLOPs 沒減少:** 這是正常的！非結構化不減少 FLOPs，需用 sparse kernel

---

### 🔧 prune_structured.py - 結構化剪枝

**學習目標:**
- **達成企業目標: ≥40% FLOPs 減少, ≤1.5% 準確率下降**
- 理解 Channel/Filter pruning 原理
- 掌握 BatchNorm Scaling (Network Slimming) 技術
- 學習依賴圖處理 (torch_pruning)

**關鍵概念:**
- **結構化剪枝真正減少 FLOPs/Latency**
- Dependency graph 自動處理 skip connections
- BatchNorm γ (scaling factor) 反映 channel 重要度
- 剪枝後模型需 shape verification

**執行命令:**
```bash
# 方法 1: L1-based channel pruning
python pruning-bootcamp/scripts/prune_structured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method l1 --sparsity 0.4

# 方法 2: BatchNorm Scaling (推薦)
python pruning-bootcamp/scripts/prune_structured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method bn_scale --sparsity 0.5
```

**預期輸出:**
```
=== Baseline Model ===
Accuracy: 93.12%
FLOPs: 0.56 G
Params: 11.17 M

=== Structured Pruning: BN_SCALE ===
Target Sparsity: 50.0%

=== After Pruning ===
FLOPs: 0.28 G (50.0% reduction)
Params: 5.58 M (50.0% reduction)

=== Fine-tuning ===
Fine-tune Epoch 1: Val Acc = 88.45%
...
Fine-tune Epoch 20: Val Acc = 92.34%

=== Final Results ===
Baseline Accuracy: 93.12%
Pruned Accuracy: 92.34%
Accuracy Drop: 0.78%
FLOPs Reduction: 50.0%
Params Reduction: 50.0%

✅ Goal MET (≥40% FLOPs, ≤1.5% acc drop)

✓ Results saved to results/structured/bn_scale_sp0.5
```

**輸出檔案:**
- `pruned_model.pth` - 壓縮後模型
- `metrics.yaml` - 包含 `goal_met` 判定

**常見問題:**
- **Shape mismatch error:** torch_pruning 自動處理，若仍報錯檢查 skip connections
- **FLOPs 減少不足:** 提高 sparsity (0.4 → 0.6)
- **準確率下降過大:** 使用 bn_scale 而非 l1

---

### 📦 export_onnx.py - 模型部署導出

**學習目標:**
- 將 PyTorch 模型轉換為部署格式
- 驗證 ONNX/TorchScript 數值一致性
- 理解 opset version 與兼容性

**關鍵概念:**
- ONNX: 跨框架模型交換格式
- TorchScript: PyTorch 原生序列化 (更快)
- Numerical verification (max diff < 1e-5)

**執行命令:**
```bash
python pruning-bootcamp/scripts/export_onnx.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/structured/pruned_model.pth \
  --format both \
  --verify
```

**預期輸出:**
```
Using device: cuda

✓ Loaded checkpoint from results/structured/pruned_model.pth

=== Exporting Model ===
✓ Exported to ONNX: results/structured/model.onnx
Max difference between PyTorch and ONNX: 0.000001
✓ ONNX export verified successfully

✓ Exported to TorchScript: results/structured/model.pt
Max difference between PyTorch and TorchScript: 0.000000
✓ TorchScript export verified successfully

✓ Export complete. Files saved to results/structured
```

**輸出檔案:**
- `model.onnx` - ONNX 格式模型
- `model.pt` - TorchScript 格式模型

**常見問題:**
- **ONNX 輸出差異過大:** 檢查 dynamic_axes 設置
- **導出失敗:** 降低 opset_version (13 → 11)

---

### ⚡ profile.py - 性能分析 (FLOPs vs Latency)

**學習目標:**
- **核心實驗: 證明 FLOPs ≠ Latency**
- 理解 memory bandwidth 瓶頸
- 分析 kernel launch overhead
- 建立正確性能評估方法

**關鍵概念:**
- FLOPs 只是理論計算量
- Latency 受 memory 影響更大
- 40% FLOPs 減少 → 僅 20-30% latency 減少
- GPU warmup 避免首次推理偏差

**執行命令:**
```bash
python pruning-bootcamp/scripts/profile.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --baseline results/baseline/best_model.pth \
  --pruned results/structured/pruned_model.pth \
  --measure-latency
```

**預期輸出:**
```
=== Profiling Baseline Model ===
FLOPs: 0.56 G
Params: 11.17 M
Latency: 2.45 ms
Throughput: 408.16 FPS

=== Profiling Pruned Model ===
FLOPs: 0.28 G
Params: 5.58 M
Latency: 1.89 ms
Throughput: 529.10 FPS

============================================================
BASELINE vs PRUNED MODEL COMPARISON
============================================================

Metric              Baseline         Pruned       Change
------------------------------------------------------------
FLOPs (G)               0.56           0.28       50.0%
Params (M)             11.17           5.58       50.0%
Latency (ms)            2.45           1.89       22.9%
Throughput (FPS)      408.16         529.10       29.6%
Speedup                                  1.30x

============================================================
FLOPs vs LATENCY ANALYSIS
============================================================
FLOPs Reduction:    50.0%
Latency Reduction:  22.9%
Discrepancy:        27.1% (FLOPs ≠ Latency!)

⚠  Significant discrepancy between FLOPs and Latency reduction!
   This demonstrates that FLOPs is not always a reliable proxy for latency.
   Factors: memory bandwidth, cache efficiency, kernel launch overhead, etc.
```

**常見問題:**
- **Latency 測量不穩定:** 關閉其他 GPU 程序、增加 num_runs
- **Speedup 低於預期:** 正常現象！這正是本腳本要證明的

---

## 21-Day Curriculum

### **Week 1: Unstructured Pruning Fundamentals** (Day 1-7)

#### **Day 1-2: Environment Setup & Baseline**
-  Setup environment with UV package manager
-  Train baseline ResNet-18 on CIFAR-10
-  Establish FLOPs/Latency/Accuracy measurement pipeline
-  Normalize metrics for consistent comparison

**Scripts**: `train.py`, `eval.py`
**Tools**: `metrics.py`, `latency.py`

```bash
python pruning-bootcamp/scripts/train.py --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml
python pruning-bootcamp/scripts/eval.py --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth --measure-latency
```

#### **Day 3-4: L1 & Global Magnitude Pruning**
-  Implement L1 unstructured pruning (layerwise)
-  Implement global magnitude pruning
-  Fine-tune pruned models
- Compare layerwise vs global strategies

**Scripts**: `prune_unstructured.py`
**Tools**: `pruning_utils.py`

```bash
# L1 layerwise pruning
python pruning-bootcamp/scripts/prune_unstructured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method l1 --mode layerwise --sparsity 0.5

# Global magnitude pruning
python pruning-bootcamp/scripts/prune_unstructured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method global --mode global --sparsity 0.5
```

#### **Day 5-7: Sparsity Allocation & Training Strategies**
- Analyze layerwise vs global sparsity distribution
- = Implement iterative pruning (gradual sparsity increase)
- Compare one-shot vs iterative approaches
- Tune fine-tuning strategies (LR, epochs, schedules)

```bash
# Iterative pruning
python pruning-bootcamp/scripts/prune_unstructured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method global --iterative --num-iterations 5 --sparsity 0.5
```

**Key Insights**:
- Global pruning often outperforms layerwise (cross-layer importance)
- Iterative pruning reduces accuracy drop but requires more training
- Fine-tuning LR is critical (typically 1/10 of original)

---

#### ✅ Week 1 Quick Validation

驗證您是否完成 Week 1 學習目標：

```bash
# 快速檢查清單
cd Pruning

# 1. 檢查 baseline 模型存在
ls results/baseline/best_model.pth
ls results/baseline/metrics.yaml

# 2. 驗證至少完成一種非結構化剪枝
ls results/unstructured/*/pruned_model.pth

# 3. 對比準確率 (應在 ≤2% drop)
python pruning-bootcamp/scripts/eval.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/unstructured/global_global_sp0.5/pruned_model.pth \
  --baseline-metrics results/baseline/metrics.yaml
```

**預期結果:**
- ✅ Baseline 準確率: ~93%
- ✅ Pruned 準確率: ~91-92%
- ⚠️ FLOPs 沒減少 (非結構化特性)

**如果失敗:**
- 準確率 < 90%: 降低 sparsity 或增加 fine-tune epochs
- 檔案不存在: 重新執行對應腳本

---

### **Week 2: Structured Pruning & Dependencies** (Day 8-14)

#### **Day 8-10: Channel/Filter Pruning**
-  Implement L1-norm channel pruning
-  Implement BatchNorm Scaling (Network Slimming)
- =
 Understand structured vs unstructured tradeoffs
- Measure actual FLOPs reduction (not just sparsity)

**Scripts**: `prune_structured.py`
**Key Concept**: Structured pruning actually reduces FLOPs/latency (vs unstructured needs sparse kernels)

```bash
# L1-based channel pruning
python pruning-bootcamp/scripts/prune_structured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method l1 --sparsity 0.4

# BatchNorm scaling (Network Slimming)
python pruning-bootcamp/scripts/prune_structured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method bn_scale --sparsity 0.5
```

#### **Day 11-12: Dependency Graph Handling**
- = Use `torch.fx` / `Torch-Pruning` for dependency tracking
- Handle skip connections and batch norm dependencies
-  Ensure tensor shape alignment after pruning
- = Debug common pruning errors (shape mismatches)

**Key Tools**: `torch_pruning` library handles dependency graphs automatically

**Common Issues**:
- Skip connections require matching channel counts
- BatchNorm layers depend on preceding Conv layers
- Global pooling requires consistent feature map sizes

#### **Day 13-14: FLOPs ≠ Latency Verification**
-  Export to ONNX/TorchScript for deployment
- Measure actual inference latency (CPU/GPU)
- =
 Analyze FLOPs vs Latency discrepancy
- Understand memory bandwidth bottlenecks

**Scripts**: `export_onnx.py`, `profile.py`

```bash
# Export to ONNX
python pruning-bootcamp/scripts/export_onnx.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/structured/pruned_model.pth \
  --format both --verify

# Profile latency
python pruning-bootcamp/scripts/profile.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --baseline results/baseline/best_model.pth \
  --pruned results/structured/pruned_model.pth \
  --measure-latency
```

**Key Insight**: 40% FLOPs reduction may only yield 20-30% latency reduction due to memory/overhead!

---

#### ✅ Week 2 Quick Validation

驗證企業目標是否達成：

```bash
# 核心檢查：Goal Met?
cd Pruning

# 1. 檢查結構化剪枝結果
cat results/structured/bn_scale_sp0.5/metrics.yaml | grep -E "goal_met|flops_reduction|accuracy_drop"

# 2. 驗證 FLOPs 真正減少 (≥40%)
python pruning-bootcamp/scripts/profile.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --baseline results/baseline/best_model.pth \
  --pruned results/structured/bn_scale_sp0.5/pruned_model.pth \
  --measure-latency \
  --save-results results/week2_validation.yaml

# 3. 檢查 ONNX 導出成功
ls results/structured/bn_scale_sp0.5/*.onnx
```

**預期結果:**
- ✅ `goal_met: true`
- ✅ `flops_reduction_pct: ≥40.0`
- ✅ `accuracy_drop: ≤1.5`
- ✅ FLOPs vs Latency discrepancy 分析報告

**如果失敗:**
- FLOPs 減少不足 (< 40%): 提高 sparsity (0.5 → 0.6)
- 準確率下降過大 (> 1.5%): 使用 `bn_scale` 方法、增加 fine-tune epochs
- Goal not met: 調整 sparsity 平衡點

---

### **Week 3: Advanced Techniques & Automation** (Day 15-21)

#### **Day 15-16: Importance Estimation Methods**
- > Implement Movement Pruning (track weight movement toward zero)
- Implement Taylor Sensitivity (|weight  gradient|)
- Compare different importance metrics
- Analyze correlation between importance and actual impact

**Tools**: Already implemented in `pruning_utils.py`

```bash
# Taylor sensitivity pruning
python pruning-bootcamp/scripts/prune_unstructured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method taylor --sparsity 0.5

# Movement pruning
python pruning-bootcamp/scripts/prune_unstructured.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/baseline/best_model.pth \
  --method movement --sparsity 0.5
```

#### **Day 17-18: N:M Sparsity (Optional)**
- =' Understand hardware-friendly sparsity patterns
- Implement 2:4 sparsity (NVIDIA Ampere support)
-  Benchmark actual speedup on compatible hardware
- Compare with unstructured sparsity

**Key Concept**: N:M sparsity (e.g., 2:4 = 2 zeros per 4 consecutive elements) is hardware-accelerated on modern GPUs

**Tools**: `n_m_sparsity()` in `pruning_utils.py`

#### **Day 19: Automated Hyperparameter Sweeper**
- > Automate sparsity/LR/epoch search
- Grid search or Bayesian optimization
- Track Pareto frontier (accuracy vs FLOPs)
- Log all experiments systematically

**Example Sweeper** (implement as needed):
```python
# Create scripts/sweep.py for grid search
for sparsity in [0.3, 0.4, 0.5, 0.6]:
    for lr in [0.001, 0.01, 0.05]:
        for epochs in [20, 30, 40]:
            run_pruning_experiment(sparsity, lr, epochs)
            log_results_to_csv()
```

#### **Day 20-21: Final Report & Lessons Learned**
- Document full pipeline and results
- Create ablation study tables
- L Analyze failure cases (e.g., over-pruning, poor recovery)
- Summarize key learnings

**Report Template**:
1. **Baseline Results**: Accuracy, FLOPs, Latency
2. **Pruning Results**: Method comparison table
3. **Ablation Studies**: Sparsity levels, fine-tuning strategies
4. **Failure Cases**: Over-aggressive pruning, sensitivity analysis
5. **Recommendations**: Best practices for production deployment

---

#### ✅ Week 3 Final Validation

完成 21 天訓練營最終檢驗：

```bash
# 綜合評估所有剪枝方法
cd Pruning

# 1. 生成完整方法對比表
echo "Method,Sparsity,FLOPs_Reduction,Accuracy_Drop,Goal_Met" > results/final_comparison.csv
for method in l1 global taylor movement; do
  python pruning-bootcamp/scripts/eval.py \
    --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
    --checkpoint results/unstructured/${method}_*/pruned_model.pth \
    --baseline-metrics results/baseline/metrics.yaml
done

# 2. 驗證最佳結構化剪枝模型
python pruning-bootcamp/scripts/eval.py \
  --config pruning-bootcamp/cfgs/cifar10_resnet18.yaml \
  --checkpoint results/structured/bn_scale_sp0.5/pruned_model.pth \
  --measure-latency

# 3. 完整性檢查
ls results/baseline/best_model.pth
ls results/unstructured/*/pruned_model.pth
ls results/structured/*/pruned_model.pth
ls results/structured/*/*.onnx
```

**預期成果:**
- ✅ Week 1: 完成 4 種非結構化剪枝實驗
- ✅ Week 2: 達成企業目標 (≥40% FLOPs, ≤1.5% acc drop)
- ✅ Week 3: 理解進階方法與 FLOPs ≠ Latency
- ✅ 完整實驗記錄與 ablation study

**畢業標準:**
1. Baseline 訓練完成 (accuracy ≥ 90%)
2. 至少 2 種非結構化方法對比
3. 結構化剪枝達成目標 (`goal_met: true`)
4. ONNX 導出與 latency profiling 完成
5. 撰寫實驗報告總結學習心得

**下一步:**
- 嘗試 ImageNet-mini 資料集
- 擴展到 MobileNet/EfficientNet
- 結合量化 (Quantization-aware pruning)
- 部署到邊緣設備實測

🎓 **恭喜完成 21 天剪枝訓練營！**

---

## 📁 Project Structure

```
Pruning/
-- pruning-bootcamp/
|   -- datasets/
|   |   -- __init__.py
|   |   -- loader.py          # CIFAR-10, ImageNet-mini loaders
|   -- scripts/
|   |   -- train.py           # Baseline training
|   |   -- eval.py            # Evaluation with metrics
|   |   -- prune_unstructured.py  # Week 1: L1, Global, Taylor, Movement
|   |   -- prune_structured.py    # Week 2: Channel pruning, BN scaling
|   |   -- export_onnx.py     # ONNX/TorchScript export
|   |   -- profile.py         # Latency profiling
|   -- cfgs/
|   |   -- cifar10_resnet18.yaml
|   |   -- imagenetmini_resnet18.yaml
|   -- results/               # Experiment outputs (auto-generated)
|   -- tools/
|       -- __init__.py
|       -- metrics.py         # FLOPs, params, accuracy metrics
|       -- latency.py         # Latency measurement
|       -- pruning_utils.py   # Pruning algorithms
-- pyproject.toml             # UV package config
-- .gitignore
-- README.md                  # This file
```

---

## Tools & Utilities

### **metrics.py**
- `compute_flops()`: FLOPs calculation
- `compute_params()`: Parameter counting
- `compute_sparsity()`: Weight sparsity
- `normalize_metrics()`: Relative to baseline
- `compare_metrics()`: Baseline vs pruned comparison

### **latency.py**
- `measure_latency()`: Accurate GPU/CPU timing
- `measure_latency_with_stats()`: Mean, std, percentiles
- `profile_layer_latency()`: Per-layer profiling
- `compare_latencies()`: FLOPs vs latency analysis

### **pruning_utils.py**
- `layerwise_magnitude_pruning()`: L1 per-layer
- `global_magnitude_pruning()`: Global L1
- `taylor_importance_pruning()`: Taylor sensitivity
- `movement_pruning()`: Movement-based
- `n_m_sparsity()`: Hardware-friendly 2:4 sparsity
- `remove_pruning_reparameterization()`: Make pruning permanent
- `print_layerwise_sparsity()`: Sparsity visualization

---

## Expected Results

| Method | Sparsity | FLOPs Reduction | Accuracy Drop | Goal Met? |
|--------|----------|----------------|---------------|-----------|
| Baseline | 0% | 0% | 0% | - |
| L1 Layerwise | 50% | ~0% (unstructured) | ~2% | ❌ |
| Global Magnitude | 50% | ~0% (unstructured) | ~1.5% | ❌ |
| Channel Pruning (L1) | 50% | 35-45% | 1-2% |  |
| BatchNorm Scaling | 50% | 40-50% | 0.5-1.5% |  |
| Taylor Sensitivity | 50% | ~0% (unstructured) | ~1.2% | ❌ |
| Movement Pruning | 50% | ~0% (unstructured) | ~1% | ❌ |

**Note**: Unstructured methods achieve high sparsity but don't reduce FLOPs without sparse kernels.

---

## Key Learnings

### **Unstructured vs Structured**
- **Unstructured**: High sparsity, minimal accuracy drop, but requires specialized sparse kernels
- **Structured**: Lower sparsity tolerance, but **actual** FLOPs/latency reduction

### **FLOPs ≠ Latency**
- Memory bandwidth bottlenecks
- Kernel launch overhead
- Cache efficiency matters
- Always measure real latency!

### **Fine-tuning is Critical**
- LR typically 1/10 - 1/5 of original training LR
- Longer fine-tuning helps but has diminishing returns
- Iterative pruning reduces shock to the network

### **Dependency Graphs**
- Use `torch_pruning` to handle complex dependencies
- Manual pruning prone to shape mismatch errors
- Test thoroughly after pruning (forward pass with dummy data)

---

## Troubleshooting

### **Shape Mismatch Errors**
```python
# Always test after pruning
dummy_input = torch.randn(1, 3, 32, 32).to(device)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should match expected
```

### **Poor Recovery After Pruning**
- Reduce sparsity target
- Increase fine-tuning epochs
- Use iterative pruning instead of one-shot
- Check if you're pruning critical layers (e.g., first/last)

### **FLOPs Not Reducing**
- You're using unstructured pruning (only adds sparsity, doesn't reduce ops)
- Use structured pruning for actual FLOPs reduction

---

## References

### **Papers**
- **Network Slimming** (Liu et al., 2017): BatchNorm-based channel pruning
- **Movement Pruning** (Sanh et al., 2020): Prune weights moving toward zero
- **Rethinking Network Pruning** (Liu et al., 2019): Structured pruning insights
- **The Lottery Ticket Hypothesis** (Frankle & Carbin, 2019): Importance of initialization

### **Libraries**
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning): Dependency-aware structured pruning
- [ONNX](https://onnx.ai/): Model export for deployment
- [TorchVision](https://pytorch.org/vision/): Pre-built models

---

## > Contributing

This is a self-contained learning repo. Extend it by:
1. Adding new pruning methods (e.g., lottery ticket, AutoML-based)
2. Supporting more architectures (MobileNet, EfficientNet)
3. Adding quantization-aware pruning
4. Implementing neural architecture search (NAS) integration

---

## License

MIT License - Free to use for educational and commercial purposes.

---

## Final Checklist

- [ ] Day 1-2: Train baseline, establish metrics
- [ ] Day 3-4: L1 & global magnitude pruning
- [ ] Day 5-7: Sparsity allocation, iterative pruning
- [ ] Day 8-10: Channel pruning, BatchNorm scaling
- [ ] Day 11-12: Dependency handling, shape alignment
- [ ] Day 13-14: ONNX export, FLOPs vs latency
- [ ] Day 15-16: Taylor & movement pruning
- [ ] Day 17-18: N:M sparsity (optional)
- [ ] Day 19: Automated sweeper
- [ ] Day 20-21: Final report, ablations, lessons learned

**Good luck on your pruning journey!**
