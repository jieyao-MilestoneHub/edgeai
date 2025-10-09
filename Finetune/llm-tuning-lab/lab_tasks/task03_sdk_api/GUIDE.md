# Task 03: Tuning SDK 與 REST API 設計

## 核心概念

模擬企業級 AI 服務（如 OpenAI Fine-tuning API）

## API 設計

```python
# FastAPI 服務
POST /v1/tunings.create  # 建立訓練任務
GET /v1/tunings.get/{id}  # 查詢狀態
DELETE /v1/tunings.cancel/{id}  # 取消任務
GET /v1/tunings.list  # 列出所有任務
```

## SDK 使用

```python
from tuning_client import TuningClient

client = TuningClient(api_key="xxx", base_url="http://localhost:8000")

# 建立訓練任務
job = client.tunings.create(
    model="gpt2",
    training_file="data.jsonl",
    hyperparameters={"epochs": 3, "rank": 8}
)

# 查詢狀態
status = client.tunings.get(job.id)
print(status.status)  # "pending", "running", "completed"
```

詳見實作檔案
