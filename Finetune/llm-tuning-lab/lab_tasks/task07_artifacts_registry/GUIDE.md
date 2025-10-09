# Task 07: 模型註冊表

## MLflow 整合

```python
import mlflow

# 記錄模型
with mlflow.start_run():
    mlflow.log_param("rank", 8)
    mlflow.log_metric("loss", 1.23)
    mlflow.pytorch.log_model(model, "model")

# 載入模型
model = mlflow.pytorch.load_model("runs:/<run_id>/model")
```

詳見實作檔案
