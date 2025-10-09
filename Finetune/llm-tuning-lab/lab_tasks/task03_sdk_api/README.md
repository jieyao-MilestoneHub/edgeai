# Task 03: Tuning SDK 與 REST API

> 建立企業級訓練服務 API 與客戶端 SDK

## 🎯 學習目標

- ✅ 設計 RESTful API
- ✅ 實作 Python SDK
- ✅ 管理訓練任務生命週期
- ✅ 實現非同步訓練服務

## 核心模組

1. **Tuning Service** (FastAPI)
   - `POST /v1/tunings.create`
   - `GET /v1/tunings.get`
   - `DELETE /v1/tunings.cancel`

2. **Python SDK**
   ```python
   client = TuningClient(api_key="xxx")
   job = client.tunings.create(...)
   ```

詳見 [GUIDE.md](GUIDE.md)
