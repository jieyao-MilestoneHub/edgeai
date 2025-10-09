# 📗 SDK 設計

> 企業級 AI 服務 SDK 設計模式

## API 設計原則

### RESTful API 規範
```
POST   /v1/tunings.create    # 建立訓練任務
GET    /v1/tunings.get       # 查詢任務狀態
DELETE /v1/tunings.cancel    # 取消任務
GET    /v1/tunings.list      # 列出所有任務
```

### SDK 設計模式
```python
from tuning_client import TuningClient

client = TuningClient(api_key="xxx")

# 非同步 API
job = client.tunings.create(
    model="llama-2-7b",
    training_file="data.jsonl",
    hyperparameters={"epochs": 3}
)

# 查詢狀態
status = client.tunings.get(job.id)
```

詳見 [Task 03 - SDK 與 API](../lab_tasks/task03_sdk_api/)
