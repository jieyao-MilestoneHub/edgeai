# Task 09: 微調小模型並部署到 Hugging Face 🚀

> 30 分鐘完成你的第一個 AI 模型：從微調到線上部署！

## 🎯 你會做什麼？

1. ✅ 微調一個小型語言模型（DistilGPT-2, 82M 參數）
2. ✅ 使用你自己的數據訓練
3. ✅ 上傳到 Hugging Face Hub
4. ✅ 分享模型連結給朋友使用

**最棒的是：普通電腦（甚至 CPU）就能完成！**

---

## 💻 硬體需求

### 最低配置（可用 CPU）
- **模型**：DistilGPT-2 (82M)
- **記憶體**：8GB RAM
- **時間**：~15 分鐘

### 推薦配置
- **模型**：GPT-2 (124M) 或 TinyLlama (1.1B)
- **GPU**：任何 NVIDIA GPU（GTX 1060 也行）
- **時間**：~5-10 分鐘

---

## 📦 安裝

```bash
# 進入目錄
cd task09_finetune_deploy_hf

# 安裝依賴
pip install -r requirements.txt

# 登入 Hugging Face（取得你的 token: https://huggingface.co/settings/tokens）
huggingface-cli login
```

---

## 🚀 5 步驟完成

### Step 1️⃣: 準備你的數據

創建一個 `my_data.csv`（或使用範例數據）：

```csv
input,output
"如何學習 Python？","從基礎語法開始，多練習實際專案。"
"什麼是機器學習？","機器學習是讓電腦從數據中學習的技術。"
"推薦的編輯器？","VSCode 或 PyCharm 都很好用。"
```

或使用我們的範例：
```bash
# 客服機器人範例
cp examples/customer_service.csv my_data.csv
```

---

### Step 2️⃣: 微調模型

```bash
python simple_finetune.py \
    --model_name distilgpt2 \
    --data_file my_data.csv \
    --output_dir ./my_model \
    --epochs 3
```

**執行中會看到：**
```
🚀 開始微調 DistilGPT-2...
📊 載入數據: 10 個訓練樣本
🏋️ 訓練中...
Epoch 1/3: 100%|█████| loss: 2.341
Epoch 2/3: 100%|█████| loss: 1.856
Epoch 3/3: 100%|█████| loss: 1.423
✅ 訓練完成！模型保存到: ./my_model
```

---

### Step 3️⃣: 測試模型（本地）

```bash
python test_from_hf.py \
    --model_path ./my_model \
    --prompt "如何學習 Python？"
```

**輸出：**
```
🤖 AI: 從基礎語法開始，多練習實際專案。建議先學習變數、函數...
```

---

### Step 4️⃣: 上傳到 Hugging Face

```bash
python upload_to_hf.py \
    --model_path ./my_model \
    --repo_name your-username/my-awesome-model \
    --private False
```

**成功後：**
```
✅ 模型已上傳！
🌐 模型連結: https://huggingface.co/your-username/my-awesome-model
📝 現在任何人都可以使用你的模型了！
```

---

### Step 5️⃣: 測試線上模型

```bash
python test_from_hf.py \
    --model_name your-username/my-awesome-model \
    --prompt "測試問題"
```

或在 Python 中：
```python
from transformers import pipeline

generator = pipeline("text-generation",
                    model="your-username/my-awesome-model")

output = generator("如何學習 Python？", max_length=50)
print(output[0]['generated_text'])
```

---

## 🎨 進階：建立 Gradio 介面（可選）

```bash
python gradio_app.py --model_name your-username/my-awesome-model
```

訪問 `http://localhost:7860` 查看你的互動式 AI 聊天機器人！

**甚至可以部署到 Hugging Face Spaces（免費託管）：**
```bash
# 會自動創建 Space 並部署
python deploy_to_spaces.py --model your-username/my-awesome-model
```

---

## 📁 專案檔案

```
task09_finetune_deploy_hf/
├── README.md               # 本文件
├── GUIDE.md                # 詳細教學
├── simple_finetune.py      # 訓練腳本
├── prepare_your_data.py    # 數據準備
├── upload_to_hf.py         # 上傳助手
├── test_from_hf.py         # 測試腳本
├── gradio_app.py           # Gradio 介面
├── requirements.txt        # 依賴
├── examples/               # 範例數據
│   ├── customer_service.csv
│   ├── code_helper.json
│   └── qa_pairs.txt
├── checklist.md            # 驗收清單
└── FAQ.md                  # 常見問題
```

---

## 🎓 3 個實用範例

### 範例 1：客服機器人
```bash
python simple_finetune.py \
    --model_name distilgpt2 \
    --data_file examples/customer_service.csv \
    --output_dir ./customer_bot
```
**用途**：自動回答常見問題

### 範例 2：程式碼助手
```bash
python simple_finetune.py \
    --model_name gpt2 \
    --data_file examples/code_helper.json \
    --output_dir ./code_assistant
```
**用途**：生成程式碼片段

### 範例 3：Q&A 機器人
```bash
python simple_finetune.py \
    --model_name distilgpt2 \
    --data_file examples/qa_pairs.txt \
    --output_dir ./qa_bot
```
**用途**：問答系統

---

## ❓ 常見問題

### Q: 我沒有 GPU 可以嗎？
A: 可以！DistilGPT-2 在 CPU 上訓練只需 15 分鐘。

### Q: 需要多少數據？
A: 最少 10-20 個範例就能開始。當然越多越好。

### Q: 訓練要多久？
A: DistilGPT-2 (CPU): ~15 分鐘
   GPT-2 (GPU): ~5 分鐘
   TinyLlama (GPU): ~10 分鐘

### Q: 上傳的模型是公開的嗎？
A: 你可以選擇公開或私有（`--private True`）

### Q: 如何改進模型效果？
A: 1. 增加數據量
   2. 增加訓練 epochs
   3. 調整學習率

更多問題請看 [FAQ.md](FAQ.md)

---

## ✅ 完成檢查

完成後，確認：

- [ ] 成功微調模型
- [ ] 本地測試通過
- [ ] 上傳到 Hugging Face
- [ ] 可以從線上載入模型
- [ ] （可選）建立 Gradio demo

---

## 🎉 恭喜！

你現在已經：
- ✅ 微調了自己的 AI 模型
- ✅ 部署到了 Hugging Face
- ✅ 擁有了可分享的模型連結

**分享你的成果：**
- Twitter: "我剛微調了一個 AI 模型！[模型連結]"
- LinkedIn: 展示你的 AI 專案
- GitHub: 開源你的訓練數據和程式碼

---

## 🔗 相關資源

- [Hugging Face Models](https://huggingface.co/models)
- [PEFT 文檔](https://huggingface.co/docs/peft)
- [Gradio 教學](https://gradio.app/docs)

---

<div align="center">

**準備好了嗎？開始微調你的第一個模型！** 🚀

[詳細教學 →](GUIDE.md) | [常見問題 →](FAQ.md)

</div>
