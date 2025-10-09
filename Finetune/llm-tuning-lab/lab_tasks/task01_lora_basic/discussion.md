# Task 01 延伸討論

這些問題旨在加深你對 LoRA 的理解。建議在完成實作後思考這些問題。

---

## 💡 理論深入

### Q1: 為什麼預訓練權重更新是低秩的？

**思考方向：**
- 預訓練模型已經學到了什麼？
- 微調時我們真正在改變什麼？
- "內在維度" (intrinsic dimensionality) 是什麼意思？

**延伸閱讀：**
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)

---

### Q2: LoRA 的 rank 應該如何選擇？

**實驗建議：**
```bash
# 嘗試不同 rank
for rank in 2 4 8 16 32 64; do
    python train_lora_basic.py --rank $rank --output_dir output_r${rank}
done
```

**分析維度：**
- 參數量 vs. 效能
- 訓練速度 vs. 精度
- 過擬合風險

**你的發現：**
- rank=2: _______________
- rank=8: _______________
- rank=16: _______________
- rank=32: _______________

---

### Q3: 為什麼 lora_B 初始化為零？

**對比實驗：**
```python
# 實驗 1: B 初始化為零（標準做法）
nn.init.zeros_(self.lora_B)

# 實驗 2: B 也使用 Kaiming 初始化
nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
```

**思考：**
- 兩種初始化對訓練有何影響？
- 為什麼零初始化更好？
- 這與「不破壞預訓練知識」有何關係？

---

## 🔬 實作深化

### Q4: LoRA 可以應用到哪些層？

**當前實作：**
```python
# 只對 attention 層使用 LoRA
target_modules = ['c_attn', 'c_proj']
```

**實驗：**
```python
# 實驗 1: 只對 Q, V
target_modules = ['q_proj', 'v_proj']

# 實驗 2: 對所有 Linear 層
target_modules = None

# 實驗 3: 包含 FFN
target_modules = ['c_attn', 'c_proj', 'c_fc']
```

**你的結論：**
- 最小可行配置：_______________
- 最佳性價比配置：_______________
- 效能天花板：_______________

---

### Q5: Alpha 的縮放真的必要嗎？

**實驗：**
```python
# 當前: output = lora_out * (alpha / rank)
# 測試: output = lora_out （無縮放）
```

**觀察指標：**
- 訓練穩定性
- 收斂速度
- 最終效能

**你的發現：**
______________________________

---

### Q6: 合併權重後性能有差異嗎？

**驗證實驗：**
```python
# 測試 1: 未合併
model.eval()
loss_unmerged = evaluate(model, dataloader)

# 測試 2: 合併後
for module in model.modules():
    if isinstance(module, LinearWithLoRA):
        module.merge_weights()
loss_merged = evaluate(model, dataloader)

# 比較差異
print(f"Difference: {abs(loss_merged - loss_unmerged)}")
```

**預期結果：**
差異應該 < 1e-6（數值誤差範圍內）

---

## 🚀 效能優化

### Q7: 如何減少記憶體使用？

**策略：**

1. **梯度累積**
   ```python
   for i, batch in enumerate(dataloader):
       loss = model(**batch).loss / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

2. **混合精度訓練**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       loss = model(**batch).loss
   scaler.scale(loss).backward()
   ```

3. **Gradient Checkpointing**
   ```python
   model.gradient_checkpointing_enable()
   ```

**實驗結果：**
| 方法 | 記憶體使用 | 訓練時間 |
|------|-----------|---------|
| 基準 | _______ | _______ |
| + 梯度累積 | _______ | _______ |
| + 混合精度 | _______ | _______ |
| + Checkpoint | _______ | _______ |

---

### Q8: LoRA 能用於推論加速嗎？

**思考：**
- 合併權重前後的推論速度
- 多 adapter 場景的優勢
- 與其他加速技術（pruning, quantization）的對比

**你的看法：**
______________________________

---

## 🏗️ 系統設計

### Q9: 如何設計多任務 LoRA 系統？

**場景：**
需要為 10 個不同任務分別微調模型。

**設計考量：**
1. 如何儲存 10 個 adapter？
2. 如何動態載入不同 adapter？
3. 如何實現 batching（同一 batch 不同 adapter）？

**你的設計：**
______________________________

---

### Q10: LoRA 與 Adapter、Prefix-tuning 的對比？

**對比維度：**
| 方法 | 參數量 | 推論延遲 | 訓練速度 | 效能 |
|------|--------|---------|---------|------|
| LoRA | ______ | ______ | ______ | ____ |
| Adapter | ______ | ______ | ______ | ____ |
| Prefix-tuning | ______ | ______ | ______ | ____ |

**你的選擇標準：**
______________________________

---

## 📈 進階應用

### Q11: 能否實作動態 rank（AdaLoRA）？

**想法：**
不同層使用不同 rank，甚至訓練中動態調整。

**偽代碼：**
```python
# 根據重要性分配 rank
importance_scores = compute_importance(model)
for layer, score in zip(model.layers, importance_scores):
    layer.lora.rank = allocate_rank(score)
```

**挑戰：**
______________________________

---

### Q12: LoRA 能用於 Vision Transformer 嗎？

**思考：**
- ViT 的架構與 LLM 的異同
- 哪些層應該使用 LoRA
- 效能會如何

**實驗計畫：**
______________________________

---

## 🎯 總結思考

### 完成這個 Task 後，你最大的收穫是什麼？

______________________________

### 還有哪些不理解的地方？

______________________________

### 下一步你想探索什麼？

______________________________

---

## 📚 推薦閱讀

### 必讀論文
1. LoRA (原始論文)
2. QLoRA (量化 + LoRA)
3. AdaLoRA (自適應 rank)

### 實作資源
1. Hugging Face PEFT 庫
2. Microsoft LoRA 官方實作
3. vLLM 的 LoRA 支援

---

**思考這些問題將幫助你深入理解 LoRA，而不僅僅是使用它！🚀**
