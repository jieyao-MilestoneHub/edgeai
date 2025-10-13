# SDK 設計：企業級 AI 訓練服務

> 從零開始設計一個像 OpenAI Fine-tuning API 一樣專業的訓練服務

---

## 學習目標

完成本章後，你將能夠：

- 理解 RESTful API 的設計原則與最佳實踐
- 設計易用的 Python SDK，處理認證、錯誤、重試等細節
- 實作訓練任務的完整生命週期管理（狀態機）
- 處理非同步任務的排程與監控
- 應用企業級考量：安全性、可擴展性、可觀測性

---

## 為什麼需要 SDK？一個實際場景

### 場景：你是 ML 平台工程師

假設你完成了 Task 01 和 Task 02，現在可以用 LoRA/QLoRA 訓練模型了。

**老闆說**：「很好！但產品經理、數據科學家、其他工程師也想用，他們不會寫 Python 訓練腳本，能不能像 OpenAI 一樣提供 API？」

**你面臨的問題**：

1. **使用門檻高**：別人要懂 PyTorch、LoRA 配置、GPU 管理
2. **資源衝突**：3 個人同時在同一台 GPU 上訓練，OOM 了
3. **無法追蹤**：訓練跑了 2 小時，不知道進度，也不知道是否成功
4. **難以整合**：產品需要在網頁上提供微調功能，但你只有 Python 腳本
5. **管理混亂**：訓練了 20 個模型，忘記哪個效果好、配置是什麼

**解決方案**：建立一個訓練服務（Training Service）

```
用戶 → Python SDK → REST API → 訓練排程器 → LoRA/QLoRA 訓練
```

**這樣的好處**：

- ✅ 簡單易用：一行代碼提交訓練，無需懂 PyTorch
- ✅ 資源管理：自動排隊，避免 GPU 衝突
- ✅ 狀態追蹤：隨時查詢訓練進度、loss 曲線
- ✅ 語言無關：任何語言都能透過 HTTP 呼叫
- ✅ 版本管理：自動記錄每次訓練的配置與結果

---

## 設計參考：業界標準 API

在設計我們的 SDK 之前，先看看業界怎麼做。

### OpenAI Fine-tuning API

```python
import openai

# 建立訓練任務
job = openai.FineTuning.create(
    model="gpt-3.5-turbo",
    training_file="file-abc123",
    hyperparameters={"n_epochs": 3}
)

# 查詢狀態
status = openai.FineTuning.retrieve(job.id)
print(status.status)  # "running", "succeeded", "failed"

# 列出所有任務
jobs = openai.FineTuning.list()
```

**設計亮點**：

1. **簡潔的 API**：`create`, `retrieve`, `list`, `cancel` 四個核心方法
2. **非同步模式**：`create()` 立即返回，用戶輪詢查詢狀態
3. **標準化狀態**：`pending`, `running`, `succeeded`, `failed`, `cancelled`
4. **資源導向**：每個訓練任務是一個資源，有唯一 ID
5. **錯誤處理**：清楚的錯誤訊息和 HTTP 狀態碼

### Google Vertex AI

```python
from google.cloud import aiplatform

# 建立訓練任務
job = aiplatform.CustomTrainingJob(
    display_name="lora-training",
    script_path="train.py",
)

job.run(
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
)

# 查詢狀態
print(job.state)  # "RUNNING", "SUCCEEDED", "FAILED"
```

**設計亮點**：

1. **資源規格化**：明確指定機器類型、加速器
2. **自動擴展**：支援多副本訓練
3. **整合監控**：自動記錄到 Cloud Logging

### Hugging Face AutoTrain

```python
from autotrain import AutoTrain

# 建立訓練任務
project = AutoTrain.create_project(
    name="sentiment-analysis",
    task="text_classification",
)

project.train(
    model="bert-base-uncased",
    dataset="sst2",
    hyperparameters={"epochs": 3, "learning_rate": 2e-5}
)
```

**設計亮點**：

1. **高階抽象**：自動選擇最佳配置
2. **任務導向**：根據任務類型自動設定
3. **雲端整合**：自動部署到 Hugging Face Spaces

---

## RESTful API 設計原則

### 什麼是 RESTful API？

**REST (Representational State Transfer)** 是一種 API 設計風格，核心概念：

1. **資源導向 (Resource-Oriented)**：一切皆資源（訓練任務、模型、數據集）
2. **HTTP 動詞映射操作**：GET（查詢）、POST（建立）、PUT（更新）、DELETE（刪除）
3. **無狀態 (Stateless)**：每個請求獨立，不依賴 session
4. **統一接口 (Uniform Interface)**：標準化的 URL 結構和回應格式

### 資源設計

**訓練任務（Tuning Job）** 是我們的核心資源。

**資源結構**：

```json
{
  "id": "job-abc123",
  "object": "tuning.job",
  "model": "llama-2-7b",
  "status": "running",
  "created_at": 1699564800,
  "training_file": "file-xyz789",
  "hyperparameters": {
    "rank": 8,
    "alpha": 16.0,
    "epochs": 3
  },
  "result_files": [],
  "metrics": {
    "train_loss": 0.245,
    "eval_loss": 0.312,
    "progress": 0.67
  }
}
```

### URL 設計規範

**路徑結構**：

```
/{version}/{resource}.{action}
```

**我們的 API 設計**：

| HTTP 方法 | URL | 說明 | 請求體 |
|---------|-----|------|--------|
| POST | `/v1/tunings.create` | 建立訓練任務 | TuningRequest |
| GET | `/v1/tunings.get/{id}` | 查詢任務狀態 | - |
| GET | `/v1/tunings.list` | 列出所有任務 | - |
| DELETE | `/v1/tunings.cancel/{id}` | 取消任務 | - |
| GET | `/v1/tunings.metrics/{id}` | 查詢訓練指標 | - |

**為什麼用 `tunings.create` 而不是 `tunings`？**

- **OpenAI 風格**：更語義化，易於理解
- **避免歧義**：`/tunings` 可能被誤解為列表接口
- **一致性**：所有動作都有明確的動詞

### HTTP 狀態碼設計

**成功回應**：

- `200 OK`：查詢成功
- `201 Created`：建立成功
- `202 Accepted`：請求已接受（非同步任務）
- `204 No Content`：刪除成功

**客戶端錯誤**：

- `400 Bad Request`：請求格式錯誤
- `401 Unauthorized`：認證失敗
- `403 Forbidden`：無權限
- `404 Not Found`：資源不存在
- `429 Too Many Requests`：超出速率限制

**伺服器錯誤**：

- `500 Internal Server Error`：伺服器內部錯誤
- `503 Service Unavailable`：服務暫時不可用

### 錯誤回應格式

**標準錯誤格式**：

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Invalid training_file: file not found",
    "code": "file_not_found",
    "param": "training_file"
  }
}
```

**錯誤類型分類**：

```python
class ErrorType:
    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    PERMISSION = "permission_error"
    NOT_FOUND = "not_found_error"
    RATE_LIMIT = "rate_limit_error"
    SERVER_ERROR = "server_error"
```

---

## 訓練任務生命週期管理

### 狀態機設計

訓練任務有明確的生命週期，我們用**狀態機**來管理。

**狀態定義**：

```python
class JobStatus:
    PENDING = "pending"        # 已建立，等待執行
    RUNNING = "running"        # 正在訓練
    SUCCEEDED = "succeeded"    # 訓練成功
    FAILED = "failed"          # 訓練失敗
    CANCELLED = "cancelled"    # 用戶取消
```

**狀態轉換圖**：

```
        create()
           ↓
       [PENDING] ─────────────────┐
           ↓                      │
      start_training()       cancel()
           ↓                      │
       [RUNNING] ────────────────┤
           ↓          ↓           ↓
    training_success()  training_failed()  cancel()
           ↓              ↓           ↓
      [SUCCEEDED]    [FAILED]   [CANCELLED]
```

**狀態轉換規則**：

| 當前狀態 | 允許的操作 | 目標狀態 |
|---------|-----------|---------|
| PENDING | start_training | RUNNING |
| PENDING | cancel | CANCELLED |
| RUNNING | training_success | SUCCEEDED |
| RUNNING | training_failed | FAILED |
| RUNNING | cancel | CANCELLED |
| SUCCEEDED | - | - |
| FAILED | - | - |
| CANCELLED | - | - |

**非法轉換處理**：

```python
def cancel_job(job_id):
    job = get_job(job_id)

    if job.status in [JobStatus.SUCCEEDED, JobStatus.FAILED]:
        raise InvalidStateTransition(
            f"Cannot cancel job in {job.status} state"
        )

    if job.status == JobStatus.RUNNING:
        # 停止訓練進程
        stop_training_process(job)

    job.status = JobStatus.CANCELLED
    job.save()
```

### 非同步任務處理

**為什麼要非同步？**

訓練可能跑幾小時，HTTP 請求不能等這麼久：

- **HTTP 超時**：大部分負載均衡器 30-60 秒超時
- **連線佔用**：長連線浪費資源
- **用戶體驗差**：畫面卡住，無法做其他事

**非同步模式**：

```
用戶 → create() → 立即返回 job_id
       ↓
    後台任務 → 訓練執行
       ↓
用戶 → get(job_id) → 查詢狀態
```

**實作方式 1：後台線程**

```python
import threading

def create_tuning(request):
    job = Job.create(request)

    # 在後台線程執行訓練
    thread = threading.Thread(
        target=run_training,
        args=(job.id,)
    )
    thread.start()

    return {"job_id": job.id, "status": "pending"}
```

**問題**：進程重啟後線程消失

**實作方式 2：任務佇列（推薦）**

```python
from celery import Celery

app = Celery('tuning_service', broker='redis://localhost:6379')

@app.task
def run_training(job_id):
    job = Job.get(job_id)
    job.status = JobStatus.RUNNING
    job.save()

    try:
        # 執行訓練
        result = train_lora(job.config)
        job.status = JobStatus.SUCCEEDED
        job.result = result
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
    finally:
        job.save()

def create_tuning(request):
    job = Job.create(request)

    # 提交到任務佇列
    run_training.delay(job.id)

    return {"job_id": job.id, "status": "pending"}
```

**優勢**：

- ✅ 持久化：任務儲存在 Redis/RabbitMQ
- ✅ 重試機制：失敗自動重試
- ✅ 分散式：可以部署多個 worker
- ✅ 監控：Flower 提供 Web UI

---

## SDK 設計模式

### 設計目標

一個好的 SDK 應該：

1. **易用性 (Easy to Use)**：簡潔的 API，符合直覺
2. **健壯性 (Robust)**：處理網路錯誤、超時、重試
3. **可擴展性 (Extensible)**：容易增加新功能
4. **類型安全 (Type Safe)**：提供類型提示，IDE 支援自動補全
5. **良好的錯誤訊息**：出錯時清楚告知原因

### SDK 結構設計

**層次化設計**：

```
TuningClient
  ├── tunings (TuningResource)
  │     ├── create()
  │     ├── get()
  │     ├── list()
  │     ├── cancel()
  │     └── wait_for_completion()
  │
  ├── files (FileResource)
  │     ├── upload()
  │     ├── delete()
  │     └── list()
  │
  └── models (ModelResource)
        ├── list()
        └── get()
```

**完整 SDK 架構**：

```python
from typing import Optional, List, Dict
import requests
import time

class TuningClient:
    """主客戶端"""
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # 初始化資源
        self.tunings = TuningResource(self)
        self.files = FileResource(self)
        self.models = ModelResource(self)

    def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict:
        """統一的 HTTP 請求處理"""
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    raise APITimeoutError(f"Request timeout after {self.timeout}s")
                time.sleep(2 ** attempt)  # 指數退避

            except requests.exceptions.HTTPError as e:
                self._handle_http_error(e.response)

            except requests.exceptions.ConnectionError:
                if attempt == self.max_retries - 1:
                    raise APIConnectionError("Failed to connect to API")
                time.sleep(2 ** attempt)

    def _handle_http_error(self, response):
        """處理 HTTP 錯誤"""
        try:
            error_data = response.json()
            error = error_data.get("error", {})
        except:
            error = {"message": response.text}

        if response.status_code == 401:
            raise AuthenticationError(error.get("message"))
        elif response.status_code == 404:
            raise NotFoundError(error.get("message"))
        elif response.status_code == 429:
            raise RateLimitError(error.get("message"))
        else:
            raise APIError(error.get("message"))

class TuningResource:
    """訓練任務資源"""
    def __init__(self, client: TuningClient):
        self.client = client

    def create(
        self,
        model: str,
        training_file: str,
        hyperparameters: Optional[Dict] = None,
        **kwargs
    ) -> TuningJob:
        """建立訓練任務"""
        data = {
            "model": model,
            "training_file": training_file,
            "hyperparameters": hyperparameters or {}
        }
        data.update(kwargs)

        response = self.client._request("POST", "/v1/tunings.create", json=data)
        return TuningJob(self.client, response)

    def get(self, job_id: str) -> TuningJob:
        """查詢任務"""
        response = self.client._request("GET", f"/v1/tunings.get/{job_id}")
        return TuningJob(self.client, response)

    def list(
        self,
        limit: int = 20,
        after: Optional[str] = None
    ) -> List[TuningJob]:
        """列出所有任務"""
        params = {"limit": limit}
        if after:
            params["after"] = after

        response = self.client._request("GET", "/v1/tunings.list", params=params)
        return [TuningJob(self.client, job) for job in response["data"]]

    def cancel(self, job_id: str) -> TuningJob:
        """取消任務"""
        response = self.client._request("DELETE", f"/v1/tunings.cancel/{job_id}")
        return TuningJob(self.client, response)

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 10,
        timeout: Optional[int] = None
    ) -> TuningJob:
        """阻塞等待任務完成"""
        start_time = time.time()

        while True:
            job = self.get(job_id)

            if job.status in ["succeeded", "failed", "cancelled"]:
                return job

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job did not complete within {timeout}s")

            time.sleep(poll_interval)

class TuningJob:
    """訓練任務物件"""
    def __init__(self, client: TuningClient, data: Dict):
        self.client = client
        self._data = data

    @property
    def id(self) -> str:
        return self._data["id"]

    @property
    def status(self) -> str:
        return self._data["status"]

    @property
    def model(self) -> str:
        return self._data["model"]

    @property
    def metrics(self) -> Optional[Dict]:
        return self._data.get("metrics")

    def refresh(self) -> "TuningJob":
        """重新載入任務資訊"""
        job = self.client.tunings.get(self.id)
        self._data = job._data
        return self

    def cancel(self) -> "TuningJob":
        """取消任務"""
        return self.client.tunings.cancel(self.id)

    def wait_for_completion(
        self,
        poll_interval: int = 10,
        timeout: Optional[int] = None
    ) -> "TuningJob":
        """等待完成"""
        return self.client.tunings.wait_for_completion(
            self.id,
            poll_interval,
            timeout
        )

    def __repr__(self):
        return f"<TuningJob id={self.id} status={self.status}>"
```

### 錯誤處理與重試機制

**自定義異常**：

```python
class APIError(Exception):
    """基礎 API 錯誤"""
    pass

class AuthenticationError(APIError):
    """認證失敗"""
    pass

class RateLimitError(APIError):
    """超出速率限制"""
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after

class TimeoutError(APIError):
    """請求超時"""
    pass

class NotFoundError(APIError):
    """資源不存在"""
    pass

class InvalidRequestError(APIError):
    """請求參數錯誤"""
    def __init__(self, message, param=None):
        super().__init__(message)
        self.param = param
```

**重試策略**：

```python
from typing import Callable
import random

def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """指數退避重試裝飾器"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (TimeoutError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    raise

                # 計算延遲時間
                delay = initial_delay * (exponential_base ** attempt)

                # 加入隨機抖動，避免雷鳴羊群效應
                if jitter:
                    delay *= (0.5 + random.random())

                print(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s")
                time.sleep(delay)

    return wrapper
```

**使用範例**：

```python
@retry_with_exponential_backoff(max_retries=5)
def create_tuning_with_retry(client, **kwargs):
    return client.tunings.create(**kwargs)

# 自動重試最多 5 次
job = create_tuning_with_retry(
    client,
    model="llama-2-7b",
    training_file="data.jsonl"
)
```

---

## 認證與授權

### API Key 認證

**生成 API Key**：

```python
import secrets
import hashlib

def generate_api_key(prefix="sk"):
    """生成 API Key"""
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}-{random_part}"

def hash_api_key(api_key: str) -> str:
    """哈希 API Key 儲存"""
    return hashlib.sha256(api_key.encode()).hexdigest()
```

**驗證中間件**：

```python
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """驗證 API Key"""
    api_key = credentials.credentials

    # 查詢資料庫
    user = db.get_user_by_api_key_hash(hash_api_key(api_key))

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="Account suspended"
        )

    return user

# 使用
@app.post("/v1/tunings.create")
async def create_tuning(
    request: TuningRequest,
    user: User = Depends(verify_api_key)
):
    # user 是經過認證的用戶
    job = create_training_job(request, user_id=user.id)
    return job
```

### Rate Limiting（速率限制）

**防止 API 濫用**：

```python
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/tunings.create")
@limiter.limit("10/minute")  # 每分鐘最多 10 次
async def create_tuning(request: Request):
    ...
```

**進階：基於用戶的限制**：

```python
from redis import Redis
import time

redis_client = Redis()

def check_rate_limit(user_id: str, limit: int, window: int):
    """
    檢查速率限制

    Args:
        user_id: 用戶 ID
        limit: 允許的請求次數
        window: 時間窗口（秒）
    """
    key = f"rate_limit:{user_id}"
    current_time = time.time()

    # 清理過期的請求記錄
    redis_client.zremrangebyscore(key, 0, current_time - window)

    # 計算當前窗口內的請求次數
    request_count = redis_client.zcard(key)

    if request_count >= limit:
        # 計算重試時間
        oldest_request = redis_client.zrange(key, 0, 0, withscores=True)[0]
        retry_after = int(window - (current_time - oldest_request[1]))

        raise RateLimitError(
            f"Rate limit exceeded. Retry after {retry_after}s",
            retry_after=retry_after
        )

    # 記錄當前請求
    redis_client.zadd(key, {str(current_time): current_time})
    redis_client.expire(key, window)

# 使用
@app.post("/v1/tunings.create")
async def create_tuning(user: User = Depends(verify_api_key)):
    check_rate_limit(user.id, limit=10, window=60)
    ...
```

---

## 企業級考量

### 日誌與監控

**結構化日誌**：

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def log(self, level, event, **kwargs):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "event": event,
            **kwargs
        }
        self.logger.log(level, json.dumps(log_entry))

    def info(self, event, **kwargs):
        self.log(logging.INFO, event, **kwargs)

    def error(self, event, **kwargs):
        self.log(logging.ERROR, event, **kwargs)

# 使用
logger = StructuredLogger("tuning_service")

logger.info(
    "job_created",
    job_id="job-123",
    user_id="user-456",
    model="llama-2-7b"
)

logger.error(
    "training_failed",
    job_id="job-123",
    error="CUDA out of memory",
    gpu_memory_used="23.5 GB"
)
```

**指標監控（Prometheus）**：

```python
from prometheus_client import Counter, Histogram, Gauge

# 定義指標
job_created_total = Counter(
    "tuning_job_created_total",
    "Total number of tuning jobs created",
    ["model", "user_id"]
)

job_status_gauge = Gauge(
    "tuning_job_status",
    "Current status of tuning jobs",
    ["job_id", "status"]
)

job_duration_seconds = Histogram(
    "tuning_job_duration_seconds",
    "Time spent training",
    ["model", "status"]
)

# 記錄指標
@app.post("/v1/tunings.create")
async def create_tuning(request: TuningRequest, user: User):
    job = create_job(request)

    # 增加計數器
    job_created_total.labels(
        model=request.model,
        user_id=user.id
    ).inc()

    # 設定 gauge
    job_status_gauge.labels(
        job_id=job.id,
        status="pending"
    ).set(1)

    return job
```

### 容錯與高可用性

**健康檢查端點**：

```python
@app.get("/health")
async def health_check():
    """健康檢查"""
    checks = {
        "api": "ok",
        "database": check_database(),
        "redis": check_redis(),
        "gpu": check_gpu()
    }

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=status_code,
        content=checks
    )

def check_database():
    try:
        db.execute("SELECT 1")
        return "ok"
    except Exception as e:
        return f"error: {str(e)}"
```

**優雅關閉**：

```python
import signal
import sys

def signal_handler(sig, frame):
    """處理終止信號"""
    logger.info("Received shutdown signal")

    # 停止接受新請求
    server.stop_accepting_requests()

    # 等待現有請求完成
    server.wait_for_active_requests(timeout=30)

    # 關閉資料庫連線
    db.close()

    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

---

## 與 LoRA/QLoRA 整合

### 封裝訓練邏輯

將 Task01 的 LoRA 訓練封裝成服務函數：

```python
from task01_lora.train_lora_basic import train_model as train_lora
from task02_qlora.train_qlora import train_model as train_qlora

def run_training_job(job_id: str):
    """執行訓練任務"""
    job = get_job(job_id)
    job.status = JobStatus.RUNNING
    job.save()

    try:
        # 準備配置
        config = {
            "model_name": job.model,
            "training_file": job.training_file,
            "lora": job.hyperparameters.get("lora", {}),
            "training": job.hyperparameters.get("training", {}),
        }

        # 選擇訓練方法
        if job.hyperparameters.get("use_quantization"):
            result = train_qlora(config, job_id=job.id)
        else:
            result = train_lora(config, job_id=job.id)

        # 更新任務狀態
        job.status = JobStatus.SUCCEEDED
        job.result_files = result["model_path"]
        job.metrics = result["metrics"]

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        logger.error("training_failed", job_id=job.id, error=str(e))

    finally:
        job.finished_at = datetime.now()
        job.save()
```

### 實時進度回報

**回調機制**：

```python
class TrainingCallback:
    """訓練進度回調"""
    def __init__(self, job_id: str):
        self.job_id = job_id

    def on_epoch_end(self, epoch, metrics):
        """每個 epoch 結束時"""
        job = get_job(self.job_id)
        job.metrics = {
            "epoch": epoch,
            "train_loss": metrics["train_loss"],
            "eval_loss": metrics["eval_loss"],
            "progress": epoch / metrics["total_epochs"]
        }
        job.save()

    def on_batch_end(self, batch, metrics):
        """每個 batch 結束時（可選）"""
        # 更新更細粒度的進度
        pass

# 在訓練循環中使用
callback = TrainingCallback(job_id)

for epoch in range(num_epochs):
    train_metrics = train_one_epoch(model, dataloader)
    eval_metrics = evaluate(model, eval_dataloader)

    callback.on_epoch_end(epoch, {
        "train_loss": train_metrics["loss"],
        "eval_loss": eval_metrics["loss"],
        "total_epochs": num_epochs
    })
```

---

## 完整 API 規格

### POST /v1/tunings.create

**建立訓練任務**

**請求體**：

```json
{
  "model": "llama-2-7b",
  "training_file": "file-abc123",
  "validation_file": "file-def456",
  "hyperparameters": {
    "rank": 8,
    "alpha": 16.0,
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 3e-4,
    "use_quantization": true
  },
  "suffix": "my-custom-model"
}
```

**回應**（201 Created）：

```json
{
  "id": "job-abc123",
  "object": "tuning.job",
  "model": "llama-2-7b",
  "status": "pending",
  "created_at": 1699564800,
  "training_file": "file-abc123",
  "validation_file": "file-def456",
  "hyperparameters": {
    "rank": 8,
    "alpha": 16.0,
    "epochs": 3
  }
}
```

### GET /v1/tunings.get/{id}

**查詢任務狀態**

**回應**（200 OK）：

```json
{
  "id": "job-abc123",
  "object": "tuning.job",
  "model": "llama-2-7b",
  "status": "running",
  "created_at": 1699564800,
  "started_at": 1699564860,
  "metrics": {
    "train_loss": 0.245,
    "eval_loss": 0.312,
    "current_epoch": 2,
    "total_epochs": 3,
    "progress": 0.67
  }
}
```

### GET /v1/tunings.list

**列出所有任務**

**查詢參數**：

- `limit`: 返回數量（默認 20）
- `after`: 分頁游標
- `status`: 篩選狀態

**回應**（200 OK）：

```json
{
  "object": "list",
  "data": [
    {
      "id": "job-abc123",
      "model": "llama-2-7b",
      "status": "succeeded",
      "created_at": 1699564800
    },
    {
      "id": "job-def456",
      "model": "llama-2-7b",
      "status": "running",
      "created_at": 1699568400
    }
  ],
  "has_more": false
}
```

### DELETE /v1/tunings.cancel/{id}

**取消任務**

**回應**（200 OK）：

```json
{
  "id": "job-abc123",
  "object": "tuning.job",
  "status": "cancelled"
}
```

---

## 實作檢查清單

完成 Task 03 後，你應該能夠：

### API 設計
- [ ] 設計 RESTful 風格的訓練服務 API
- [ ] 定義清楚的資源結構和端點
- [ ] 設計合理的 HTTP 狀態碼和錯誤格式
- [ ] 實作 API 版本控制（/v1）

### 狀態管理
- [ ] 定義訓練任務的完整生命週期
- [ ] 實作狀態機，處理狀態轉換
- [ ] 處理非法狀態轉換
- [ ] 持久化任務狀態到資料庫

### 非同步處理
- [ ] 使用任務佇列（Celery/RQ）處理訓練任務
- [ ] 實作後台 worker 執行訓練
- [ ] 提供查詢進度的接口
- [ ] 支援取消正在執行的任務

### SDK 開發
- [ ] 設計易用的 Python SDK
- [ ] 實作錯誤處理與重試機制
- [ ] 提供類型提示和文檔字串
- [ ] 支援同步和非同步操作

### 認證授權
- [ ] 實作 API Key 認證
- [ ] 設計 Rate Limiting 機制
- [ ] 處理權限驗證
- [ ] 保護敏感操作

### 企業級功能
- [ ] 結構化日誌
- [ ] 指標監控（Prometheus）
- [ ] 健康檢查端點
- [ ] 優雅關閉處理

### 整合
- [ ] 將 Task01 LoRA 訓練封裝為服務
- [ ] 將 Task02 QLoRA 訓練封裝為服務
- [ ] 實作訓練進度回調
- [ ] 儲存訓練結果和模型

---

## 延伸學習

### 必讀文章

1. **RESTful API 設計**
   - [REST API Tutorial](https://restfulapi.net/)
   - [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines)

2. **非同步任務處理**
   - [Celery 官方文檔](https://docs.celeryq.dev/)
   - [Distributed Task Queue 模式](https://www.enterpriseintegrationpatterns.com/)

3. **SDK 設計**
   - [AWS SDK Design Principles](https://aws.amazon.com/blogs/developer/)
   - [Stripe API 設計](https://stripe.com/docs/api)

### 實作參考

- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [FastAPI 官方教學](https://fastapi.tiangolo.com/)
- [Celery Real-world Examples](https://github.com/celery/celery/tree/main/examples)

### 相關技術

| 技術 | 用途 | 推薦度 |
|------|------|-------|
| **FastAPI** | Web 框架 | ⭐⭐⭐⭐⭐ |
| **Celery** | 任務佇列 | ⭐⭐⭐⭐⭐ |
| **Redis** | 佇列後端 | ⭐⭐⭐⭐⭐ |
| **SQLAlchemy** | ORM | ⭐⭐⭐⭐ |
| **Pydantic** | 資料驗證 | ⭐⭐⭐⭐⭐ |
| **Prometheus** | 監控 | ⭐⭐⭐⭐ |
| **Docker** | 容器化 | ⭐⭐⭐⭐⭐ |

---

## 小結 — 你應該能說出

### Q1. 為什麼訓練服務要用非同步模式？

A. 因為 HTTP 請求不能等待，需要立即返回
B. 因為訓練可能跑幾小時，超過 HTTP 超時限制
C. 因為可以讓用戶同時提交多個訓練任務
D. 以上皆是

---

### Q2. RESTful API 設計中，哪個 HTTP 方法用於建立資源？

A. GET
B. POST
C. PUT
D. DELETE

---

### Q3. 訓練任務的狀態機中，從 RUNNING 狀態可以轉換到哪些狀態？

A. 只能到 SUCCEEDED
B. SUCCEEDED 或 FAILED
C. SUCCEEDED、FAILED 或 CANCELLED
D. 可以轉換到任何狀態

---

### Q4. SDK 的重試機制為什麼要用指數退避（Exponential Backoff）？

A. 因為這樣重試最快
B. 因為可以避免大量請求同時重試，造成伺服器過載
C. 因為這樣實作最簡單
D. 因為業界標準規定必須這樣做

---

### Q5. API Key 應該如何儲存在資料庫？

A. 明文儲存，方便查詢
B. 用 SHA-256 哈希後儲存
C. 用 AES 加密後儲存
D. 不需要儲存

---

### Q6. Rate Limiting 的主要目的是什麼？

A. 限制訓練時間
B. 限制模型大小
C. 防止 API 濫用和保護伺服器資源
D. 提升 API 速度

---

### Q7. 健康檢查端點（/health）應該檢查什麼？

A. 只檢查 API 是否啟動
B. 檢查所有依賴服務（資料庫、Redis、GPU）
C. 檢查用戶數量
D. 檢查訓練任務數量

---

### Q8. 哪個設計模式最適合組織 SDK 的功能？

A. Singleton 模式
B. Factory 模式
C. Resource 模式（資源導向）
D. Observer 模式

---

> **關鍵啟示**
> SDK 設計不只是封裝 HTTP 請求，而是**創造良好的開發體驗**：
> **讓複雜的分散式系統變得簡單易用。**
