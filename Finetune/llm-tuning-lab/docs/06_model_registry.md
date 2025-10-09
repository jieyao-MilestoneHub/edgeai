# 📒 模型註冊與版本管理

> 使用 MLflow 管理模型生命週期

## 核心功能

### 1. 模型版本化
```python
mlflow.log_model(model, "lora_adapter_v1")
```

### 2. Artifact 追蹤
- 訓練配置
- 超參數
- Metrics 歷史

### 3. 模型回滾
```python
model = mlflow.pytorch.load_model(f"models:/{model_name}/1")
```

詳見 [Task 07 - 模型註冊](../lab_tasks/task07_artifacts_registry/)
