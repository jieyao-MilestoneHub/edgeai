# 常見問題

## Q1: 沒有 GPU 可以訓練嗎？

**A**: 可以！DistilGPT-2 在 CPU 上訓練約 15 分鐘。
- DistilGPT-2 (82M): ✅ CPU 友好
- GPT-2 (124M): ⚠️ CPU 較慢（~30分鐘）
- TinyLlama (1.1B): ❌ 需要 GPU

## Q2: 訓練時出現 CUDA Out of Memory

**解決方案：**
```python
# 減少 batch_size
--batch_size 2

# 使用更小的模型
--model_name distilgpt2

# 減少序列長度
--max_length 128
```

## Q3: 需要多少數據？

**最少：** 10-20 個高品質樣本
**推薦：** 100+ 樣本
**最佳：** 1000+ 樣本

數據品質比數量更重要！

## Q4: 如何改進模型效果？

1. **增加數據**：更多樣本 = 更好效果
2. **增加 epochs**：`--epochs 5`
3. **調整學習率**：`--learning_rate 1e-4`
4. **使用更大模型**：GPT-2 > DistilGPT-2

## Q5: 上傳失敗怎麼辦？

檢查：
- [ ] 是否已登入？ `huggingface-cli whoami`
- [ ] Token 是否有效？
- [ ] Repo 名稱格式：`username/model-name`
- [ ] 網路連接是否正常？

## Q6: 如何設定為私有模型？

```bash
python upload_to_hf.py \
    --model_path ./my_model \
    --repo_name username/my-model \
    --private  # 加這個參數
```

## Q7: 模型可以商用嗎？

取決於：
1. **Base 模型的授權**（檢查模型卡）
2. **你的訓練數據**（確保有使用權）
3. **HF 服務條款**

## Q8: 如何刪除已上傳的模型？

在 HF 網站：
1. 進入模型頁面
2. Settings → Delete this model

## Q9: 可以微調多語言模型嗎？

可以！推薦模型：
- **中文**：IDEA-CCNL/Wenzhong-GPT2-110M
- **多語言**：facebook/xglm-564M

## Q10: 訓練後模型變大了？

LoRA 微調只會增加很小的檔案：
- LoRA adapters: ~2-10 MB
- 完整模型：如果合併會變成原大小

**建議**：不合併，只上傳 LoRA adapters
