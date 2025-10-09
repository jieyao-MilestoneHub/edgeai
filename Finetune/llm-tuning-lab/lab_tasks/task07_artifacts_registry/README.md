# Task 07: Artifact 與模型註冊表

> 使用 MLflow 管理模型生命週期

## 🎯 學習目標

- ✅ MLflow 整合
- ✅ 模型版本化
- ✅ Artifact 追蹤
- ✅ 模型回滾機制

## 核心功能

```python
# 記錄模型
mlflow.log_model(model, "lora_v1")

# 載入模型
model = mlflow.pytorch.load_model("models:/lora/1")

# 版本管理
mlflow.register_model(model_uri, "production")
```

詳見 [GUIDE.md](GUIDE.md)
