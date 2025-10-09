# Task 01 驗收清單

完成 Task 01 後，請逐項確認以下內容：

## ✅ 理論理解

- [ ] 能解釋 LoRA 的核心思想（低秩分解）
- [ ] 理解為什麼 LoRA 能節省參數
- [ ] 能計算給定 rank 下的參數量
- [ ] 理解 alpha 縮放因子的作用
- [ ] 知道為什麼 lora_B 初始化為零

## ✅ 程式實作

- [ ] 成功實作 `LoRALayer` 類別
- [ ] 成功實作 `LinearWithLoRA` 類別
- [ ] 實作 `apply_lora_to_model` 函數
- [ ] 所有測試通過（運行 `python lora_linear.py`）
- [ ] 程式碼有適當的註解與 docstring

## ✅ 訓練執行

- [ ] 成功運行 `train_lora_basic.py`
- [ ] 訓練 loss 正常下降
- [ ] 驗證 loss 收斂
- [ ] 沒有出現 OOM 錯誤

## ✅ 輸出檔案

- [ ] 產生 `output/best_adapter_model.bin`
- [ ] 產生 `output/final_adapter_model.bin`
- [ ] 產生 `output/training_loss_curve.png`
- [ ] 產生 `output/training_metrics.json`

## ✅ 結果驗證

- [ ] Loss 曲線圖清晰可讀
- [ ] 訓練 loss < 2.5
- [ ] 驗證 loss < 3.0
- [ ] LoRA 參數 < 總參數的 1%

## ✅ 實驗探索

- [ ] 嘗試不同的 rank (4, 8, 16)
- [ ] 觀察 rank 對效能的影響
- [ ] 嘗試不同的 alpha (8, 16, 32)
- [ ] 記錄實驗結果

## ✅ 延伸理解

- [ ] 完成 `discussion.md` 中的所有問題
- [ ] 能向他人解釋 LoRA 的原理
- [ ] 理解 LoRA 與全參數微調的差異

---

## 📊 參考指標

### 訓練結果（GPT-2 + WikiText-2）

| Metric | Expected Value |
|--------|----------------|
| 初始 Loss | ~4.0 - 5.0 |
| 最終訓練 Loss | 1.5 - 2.5 |
| 最終驗證 Loss | 2.0 - 3.0 |
| 訓練時間 (3 epochs) | 15-30 分鐘（RTX 3090）|

### 參數統計（rank=8）

| Metric | Expected Value |
|--------|----------------|
| 總參數 | ~124M |
| 可訓練參數 | ~200K - 300K |
| 參數減少 | >400× |
| 可訓練百分比 | <0.3% |

---

**完成所有項目後，恭喜你！你已經掌握了 LoRA 的核心實作！🎉**

繼續前往 [Task 02](../task02_qlora/) 學習量化技術。
