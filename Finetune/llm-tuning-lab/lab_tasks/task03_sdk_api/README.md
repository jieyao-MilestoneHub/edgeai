# Task 03: Tuning SDK 與 REST API

> 將 LoRA/QLoRA 訓練包裝成 REST API，並提供 Python SDK

## 實驗目標

建立一個 FastAPI 訓練服務，讓用戶可以透過 HTTP API 或 Python SDK 來提交和管理訓練任務。

## 環境準備

```bash
# 進入目錄
cd lab_tasks/task03_sdk_api

# 安裝依賴
pip install -r requirements.txt
```

## SDK vs API 概念說明

**重要**：這個實驗包含兩個獨立的組件，請理解它們的區別和關係。

### RESTful API（服務端）

**是什麼**：FastAPI 後端服務
**位置**：`tuning_service/app.py`
**作用**：提供 HTTP 端點，處理訓練請求
**運行**：`uvicorn tuning_service.app:app --port 8000`
**訪問**：`http://localhost:8000`

這是**服務器端**，需要部署在有 GPU 的機器上。

### Python SDK（客戶端）

**是什麼**：Python 客戶端庫
**位置**：`tuning_sdk/client.py`
**作用**：封裝 HTTP 請求，提供易用的 Python API
**使用**：`from tuning_sdk import TuningClient`
**安裝**：`pip install -e .`（本地開發模式）

這是**客戶端**，可以在任何機器上使用，通過 HTTP 調用遠端 API。

### 關係圖

```
用戶代碼 (你的腳本)
    ↓
Python SDK (tuning_sdk/)  ← 封裝 HTTP 請求
    ↓ HTTP
RESTful API (tuning_service/)  ← 處理請求，執行訓練
    ↓
LoRA/QLoRA 訓練腳本
```

### 業界標準對照

這是業界標準做法，與以下服務相同：

| 服務 | API 端 | SDK 端 |
|------|--------|--------|
| **OpenAI** | `https://api.openai.com` | `pip install openai` |
| **AWS SageMaker** | AWS API Gateway | `pip install boto3` |
| **本實驗** | `http://localhost:8000` | `pip install -e .` |

### 為什麼分離？

1. **遠端調用**：SDK 可以調用任何地方的 API（本地、雲端）
2. **多語言支持**：可以開發 JavaScript、Go 等其他語言的 SDK
3. **獨立發布**：SDK 可以發布到 PyPI，用戶直接 `pip install`
4. **職責分離**：API 處理業務邏輯，SDK 處理網絡通信

### 實際部署場景

**開發階段**（本實驗）：
```bash
# 終端 1：啟動 API 服務
uvicorn tuning_service.app:app --reload

# 終端 2：使用 SDK 調用
python test_sdk.py
```

**生產環境**：
```
公司 GPU 服務器（部署 API）
    ↑ HTTP
員工筆記本（安裝 SDK，運行腳本）
```

## 實驗步驟

### 步驟 1: 啟動 API 服務

```bash
# 方法 1: 直接運行
python tuning_service/app.py

# 方法 2: 使用 uvicorn
uvicorn tuning_service.app:app --reload --port 8000
```

**驗證服務啟動**：
```bash
curl http://localhost:8000/health
```

應該返回：
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 5.2,
  "active_jobs": 0,
  "total_jobs": 0
}
```

**查看 API 文檔**：打開瀏覽器訪問 http://localhost:8000/docs

### 步驟 2: 測試 API（使用 curl）

**創建訓練任務**：
```bash
curl -X POST http://localhost:8000/v1/tunings.create \
  -H "Authorization: Bearer test-key-001" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "training_file": "data/train.jsonl",
    "hyperparameters": {
      "epochs": 3,
      "learning_rate": 0.0002,
      "batch_size": 4
    }
  }'
```

**查詢任務狀態**：
```bash
# 將 <job_id> 替換為上一步返回的 job id
curl -X GET http://localhost:8000/v1/tunings.get/<job_id> \
  -H "Authorization: Bearer test-key-001"
```

**列出所有任務**：
```bash
curl -X GET http://localhost:8000/v1/tunings.list \
  -H "Authorization: Bearer test-key-001"
```

**取消任務**：
```bash
curl -X POST http://localhost:8000/v1/tunings.cancel/<job_id> \
  -H "Authorization: Bearer test-key-001"
```

### 步驟 3: 使用 Python SDK

創建測試腳本 `test_sdk.py`：

```python
from tuning_sdk import TuningClient
import time

# 初始化客戶端
client = TuningClient(
    api_key="test-key-001",
    base_url="http://localhost:8000"
)

# 檢查服務健康狀態
health = client.health_check()
print(f"服務狀態: {health['status']}")

# 創建訓練任務
job = client.tunings.create(
    model="meta-llama/Llama-2-7b-hf",
    training_file="data/train.jsonl",
    hyperparameters={
        "epochs": 3,
        "learning_rate": 0.0002
    }
)
print(f"任務已創建: {job.id}")

# 等待訓練完成（帶進度顯示）
def show_progress(j):
    if j.metrics:
        print(f"進度: {j.metrics.progress * 100:.1f}% | "
              f"Epoch {j.metrics.current_epoch}/{j.metrics.total_epochs} | "
              f"Loss: {j.metrics.train_loss:.4f}")

final_job = client.tunings.wait(job.id, on_progress=show_progress)

# 顯示結果
if final_job.is_successful:
    print(f"\n訓練成功！")
    print(f"微調模型: {final_job.fine_tuned_model}")
    print(f"訓練時長: {final_job.duration}秒")
else:
    print(f"\n訓練失敗: {final_job.error}")

client.close()
```

運行：
```bash
python test_sdk.py
```

### 步驟 4: 運行範例

**基礎使用範例**：
```bash
python examples/basic_usage.py
```

**進階使用範例**（互動式選單）：
```bash
python examples/advanced_usage.py
```

進階範例包含：
1. 錯誤處理
2. Context Manager 用法
3. 取消訓練任務
4. 流式輸出日誌
5. 批量操作
6. 自定義進度條

### 步驟 5: 運行測試

```bash
# 測試 API
pytest tests/test_api.py -v

# 測試 SDK
pytest tests/test_sdk.py -v

# 查看測試覆蓋率
pytest tests/ -v --cov=tuning_service --cov=tuning_sdk
```

## 核心檔案說明

### `tuning_service/app.py` (920 行)
FastAPI 服務主檔案，包含：
- API 端點定義（創建、查詢、列出、取消、刪除）
- API Key 認證
- 異步訓練模擬器
- 全局異常處理
- 健康檢查端點

### `tuning_sdk/client.py` (1020 行)
Python SDK 客戶端，包含：
- `TuningClient` 主類
- HTTP 請求處理（自動重試、指數退避）
- 自定義異常類型
- 資料模型（Hyperparameters, TuningJob, TrainingMetrics）
- `wait()` 和 `stream_logs()` 方法

可透過 `pip install -e .` 安裝為本地包。

### `examples/`
- `basic_usage.py`: 7步驟完整示範
- `advanced_usage.py`: 6個進階使用場景
- `README.md`: 範例使用說明

### `tests/`
- `test_api.py`: 40+ FastAPI 測試用例
- `test_sdk.py`: SDK 功能測試

## 常見問題

### Q1: 服務啟動失敗，端口被佔用

```bash
# 使用不同端口
uvicorn tuning_service.app:app --port 8001

# 或查找並終止佔用進程
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9
```

### Q2: SDK 連接失敗

1. 確認服務已啟動：`curl http://localhost:8000/health`
2. 檢查 API Key：測試環境使用 `test-key-001` 或 `test-key-002`
3. 確認 URL：`base_url="http://localhost:8000"`

### Q3: 訓練任務一直是 pending 狀態

檢查服務日誌 `tuning_service.log`，可能原因：
- 達到並發任務上限（默認 10）
- 後台任務執行異常

### Q4: 如何加快測試速度？

修改 `tuning_service/app.py` 中的配置：
```python
class Config:
    TRAINING_SIMULATION_TIME = 10  # 從 30 改為 10 秒
```

或創建任務時減少 epochs：
```python
job = client.tunings.create(
    model="test-model",
    training_file="data/train.jsonl",
    hyperparameters={"epochs": 1}  # 減少訓練輪數
)
```

## 實驗檢查清單

完成以下任務即表示實驗成功：

- [ ] API 服務成功啟動
- [ ] 可以訪問 Swagger 文檔 (http://localhost:8000/docs)
- [ ] 使用 curl 成功創建訓練任務
- [ ] 使用 curl 成功查詢任務狀態
- [ ] SDK 測試腳本運行成功
- [ ] 基礎範例運行成功
- [ ] 至少運行一個進階範例
- [ ] 測試用例全部通過

## 檔案結構

```
task03_sdk_api/
├── tuning_service/             # RESTful API（服務端）
│   └── app.py                  # FastAPI 服務 (920 行)
├── tuning_sdk/                 # Python SDK（客戶端）
│   ├── __init__.py             # 包初始化
│   └── client.py               # SDK 主類 (1020 行)
├── examples/
│   ├── basic_usage.py          # 基礎範例
│   ├── advanced_usage.py       # 進階範例
│   └── README.md               # 範例說明
├── tests/
│   ├── test_api.py             # API 測試
│   └── test_sdk.py             # SDK 測試
├── setup.py                    # SDK 安裝配置
├── requirements.txt            # Python 依賴
└── README.md                   # 本檔案
```

## 下一步

- 查看 `GUIDE.md` 了解如何從零實作這個服務（6 階段，3-4 小時）
- 查看 `docs/03_sdk_design.md` 深入理解設計原理
- 嘗試整合 Task 01/02 的真實訓練代碼

---

完成實驗後，你將學會：
- ✅ 如何用 FastAPI 建立 REST API
- ✅ 如何設計異步任務管理系統
- ✅ 如何開發 Python SDK
- ✅ 如何實作 API Key 認證
- ✅ 如何處理 HTTP 錯誤與重試

有問題？查看 `examples/README.md` 或 `GUIDE.md` 中的詳細說明。
