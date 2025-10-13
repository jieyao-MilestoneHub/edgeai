"""
Python SDK for Tuning Service API
==================================

這是一個完整的 Python SDK，用於與微調訓練服務 API 進行交互。

主要功能：
---------
1. 統一的 HTTP 請求處理
2. 自動重試機制（指數退避）
3. 完整的錯誤處理
4. 類型提示與資料驗證
5. 資源導向的 API 設計
6. 同步與異步操作支持
7. 請求/響應日誌記錄

設計模式：
----------
- Resource Pattern: 將 API 端點組織為資源類（TuningResource）
- Client Pattern: 提供統一的客戶端入口（TuningClient）
- Retry Pattern: 自動處理臨時性錯誤
- Exception Hierarchy: 結構化的異常體系

使用範例：
----------
```python
from client import TuningClient

# 初始化客戶端
client = TuningClient(api_key="test-key-001")

# 創建訓練任務
job = client.tunings.create(
    model="meta-llama/Llama-2-7b-hf",
    training_file="data/train.jsonl",
    hyperparameters={"epochs": 3}
)

# 查詢任務狀態
job_status = client.tunings.get(job.id)
print(f"Status: {job_status.status}")

# 等待任務完成
final_job = client.tunings.wait(job.id)
print(f"Final model: {final_job.fine_tuned_model}")
```

作者：EdgeAI Lab
版本：1.0.0
更新日期：2025-10-13
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================================
# 第一部分：日誌配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 第二部分：異常定義
# ============================================================================

class TuningAPIError(Exception):
    """SDK 基礎異常類

    所有 SDK 異常的父類，提供統一的錯誤處理接口。
    """
    def __init__(self, message: str, status_code: Optional[int] = None,
                 error_type: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code

    def __str__(self):
        parts = [self.message]
        if self.status_code:
            parts.append(f"(status={self.status_code})")
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        return " ".join(parts)


class AuthenticationError(TuningAPIError):
    """認證錯誤（401）

    當 API Key 無效或缺失時拋出。
    """
    pass


class PermissionError(TuningAPIError):
    """權限錯誤（403）

    當用戶沒有訪問資源的權限時拋出。
    """
    pass


class NotFoundError(TuningAPIError):
    """資源不存在錯誤（404）

    當請求的訓練任務不存在時拋出。
    """
    pass


class ValidationError(TuningAPIError):
    """請求驗證錯誤（422）

    當請求參數不符合要求時拋出。
    """
    def __init__(self, message: str, validation_errors: Optional[List[Dict]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []


class RateLimitError(TuningAPIError):
    """速率限制錯誤（429）

    當超過 API 請求頻率限制時拋出。
    """
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ServerError(TuningAPIError):
    """服務器錯誤（5xx）

    當服務器內部錯誤時拋出。
    """
    pass


class TimeoutError(TuningAPIError):
    """請求超時錯誤

    當請求超過設定的超時時間時拋出。
    """
    pass


class ConnectionError(TuningAPIError):
    """連接錯誤

    當無法連接到服務器時拋出。
    """
    pass


# ============================================================================
# 第三部分：資料模型
# ============================================================================

class JobStatus(str, Enum):
    """訓練任務狀態枚舉"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """檢查是否為終止狀態"""
        return self in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]


@dataclass
class Hyperparameters:
    """訓練超參數"""
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_seq_length: int = 512
    warmup_steps: int = 100

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "max_seq_length": self.max_seq_length,
            "warmup_steps": self.warmup_steps
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Hyperparameters":
        """從字典創建實例"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingMetrics:
    """訓練指標"""
    current_epoch: int
    total_epochs: int
    current_step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    progress: float = 0.0
    throughput: Optional[float] = None
    estimated_finish_time: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["TrainingMetrics"]:
        """從字典創建實例"""
        if not data:
            return None
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TuningJob:
    """訓練任務對象

    表示一個完整的訓練任務，包含所有相關資訊。
    """
    id: str
    model: str
    status: JobStatus
    created_at: int
    training_file: str
    hyperparameters: Hyperparameters
    object: str = "tuning.job"
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    validation_file: Optional[str] = None
    fine_tuned_model: Optional[str] = None
    metrics: Optional[TrainingMetrics] = None
    error: Optional[str] = None
    result_files: List[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        """檢查任務是否已終止"""
        return JobStatus(self.status).is_terminal()

    @property
    def is_successful(self) -> bool:
        """檢查任務是否成功完成"""
        return self.status == JobStatus.SUCCEEDED

    @property
    def is_failed(self) -> bool:
        """檢查任務是否失敗"""
        return self.status == JobStatus.FAILED

    @property
    def is_cancelled(self) -> bool:
        """檢查任務是否被取消"""
        return self.status == JobStatus.CANCELLED

    @property
    def duration(self) -> Optional[int]:
        """計算任務執行時間（秒）"""
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    @classmethod
    def from_dict(cls, data: Dict) -> "TuningJob":
        """從 API 響應創建 TuningJob 實例"""
        return cls(
            id=data["id"],
            object=data.get("object", "tuning.job"),
            model=data["model"],
            status=JobStatus(data["status"]),
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            training_file=data["training_file"],
            validation_file=data.get("validation_file"),
            hyperparameters=Hyperparameters.from_dict(data["hyperparameters"]),
            fine_tuned_model=data.get("fine_tuned_model"),
            metrics=TrainingMetrics.from_dict(data.get("metrics")),
            error=data.get("error"),
            result_files=data.get("result_files", [])
        )

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            "id": self.id,
            "object": self.object,
            "model": self.model,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "training_file": self.training_file,
            "validation_file": self.validation_file,
            "hyperparameters": self.hyperparameters.to_dict(),
            "fine_tuned_model": self.fine_tuned_model,
            "metrics": self.metrics.__dict__ if self.metrics else None,
            "error": self.error,
            "result_files": self.result_files
        }


@dataclass
class JobList:
    """訓練任務列表"""
    data: List[TuningJob]
    total: int
    has_more: bool = False
    object: str = "list"

    @classmethod
    def from_dict(cls, data: Dict) -> "JobList":
        """從 API 響應創建實例"""
        return cls(
            object=data.get("object", "list"),
            data=[TuningJob.from_dict(job) for job in data["data"]],
            has_more=data.get("has_more", False),
            total=data["total"]
        )


# ============================================================================
# 第四部分：HTTP 客戶端
# ============================================================================

class HTTPClient:
    """HTTP 客戶端

    處理所有 HTTP 請求，包括：
    - 自動重試
    - 超時處理
    - 錯誤轉換
    - 請求日誌
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
    ):
        """初始化 HTTP 客戶端

        參數：
            api_key: API 密鑰
            base_url: API 基礎 URL
            timeout: 請求超時時間（秒）
            max_retries: 最大重試次數
            retry_backoff_factor: 重試退避因子
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # 創建帶有重試機制的 session
        self.session = requests.Session()

        # 配置重試策略（僅對特定錯誤重試）
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=[408, 429, 500, 502, 503, 504],  # 需要重試的 HTTP 狀態碼
            allowed_methods=["GET", "POST", "DELETE"]  # 允許重試的方法
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _get_headers(self) -> Dict[str, str]:
        """生成請求頭"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "tuning-sdk-python/1.0.0"
        }

    def _handle_error_response(self, response: requests.Response):
        """處理錯誤響應

        根據 HTTP 狀態碼拋出相應的異常。
        """
        try:
            error_data = response.json().get("error", {})
            message = error_data.get("message", response.text)
            error_type = error_data.get("type")
            error_code = error_data.get("code")
        except Exception:
            message = response.text or f"HTTP {response.status_code}"
            error_type = None
            error_code = None

        # 根據狀態碼拋出不同的異常
        if response.status_code == 401:
            raise AuthenticationError(message, status_code=401, error_type=error_type, error_code=error_code)
        elif response.status_code == 403:
            raise PermissionError(message, status_code=403, error_type=error_type, error_code=error_code)
        elif response.status_code == 404:
            raise NotFoundError(message, status_code=404, error_type=error_type, error_code=error_code)
        elif response.status_code == 422:
            validation_errors = error_data.get("details", [])
            raise ValidationError(message, validation_errors=validation_errors, status_code=422,
                                error_type=error_type, error_code=error_code)
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(message, retry_after=retry_after, status_code=429,
                               error_type=error_type, error_code=error_code)
        elif response.status_code >= 500:
            raise ServerError(message, status_code=response.status_code,
                            error_type=error_type, error_code=error_code)
        else:
            raise TuningAPIError(message, status_code=response.status_code,
                               error_type=error_type, error_code=error_code)

    def request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """發送 HTTP 請求

        參數：
            method: HTTP 方法（GET, POST, DELETE 等）
            path: API 路徑
            **kwargs: requests 庫的其他參數

        返回：
            JSON 響應數據

        異常：
            TuningAPIError: 各種 API 錯誤
        """
        url = f"{self.base_url}{path}"
        headers = self._get_headers()

        logger.debug(f"{method} {url}")

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                timeout=self.timeout,
                **kwargs
            )

            # 檢查響應狀態
            if response.status_code >= 400:
                self._handle_error_response(response)

            # 204 No Content 不返回數據
            if response.status_code == 204:
                return {}

            return response.json()

        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {url}")
            raise TimeoutError(f"Request timeout after {self.timeout}s") from e

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {url}")
            raise ConnectionError(f"Failed to connect to {self.base_url}") from e

        except (AuthenticationError, PermissionError, NotFoundError,
                ValidationError, RateLimitError, ServerError, TuningAPIError):
            # 重新拋出已知的 API 錯誤
            raise

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise TuningAPIError(f"Unexpected error: {e}") from e

    def get(self, path: str, **kwargs) -> Dict:
        """GET 請求"""
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> Dict:
        """POST 請求"""
        return self.request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs) -> Dict:
        """DELETE 請求"""
        return self.request("DELETE", path, **kwargs)

    def close(self):
        """關閉 session"""
        self.session.close()


# ============================================================================
# 第五部分：資源類
# ============================================================================

class TuningResource:
    """訓練任務資源類

    提供訓練任務的所有操作方法。
    """

    def __init__(self, client: HTTPClient):
        """初始化資源

        參數：
            client: HTTP 客戶端實例
        """
        self._client = client

    def create(
        self,
        model: str,
        training_file: str,
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict] = None,
        suffix: Optional[str] = None
    ) -> TuningJob:
        """創建訓練任務

        參數：
            model: 基礎模型名稱
            training_file: 訓練資料檔案路徑
            validation_file: 驗證資料檔案路徑（可選）
            hyperparameters: 訓練超參數字典（可選）
            suffix: 模型名稱後綴（可選）

        返回：
            TuningJob 對象

        範例：
            ```python
            job = client.tunings.create(
                model="meta-llama/Llama-2-7b-hf",
                training_file="data/train.jsonl",
                hyperparameters={"epochs": 3, "learning_rate": 2e-4}
            )
            print(f"Created job: {job.id}")
            ```
        """
        payload = {
            "model": model,
            "training_file": training_file,
        }

        if validation_file:
            payload["validation_file"] = validation_file

        if hyperparameters:
            payload["hyperparameters"] = hyperparameters

        if suffix:
            payload["suffix"] = suffix

        response = self._client.post("/v1/tunings.create", json=payload)
        job = TuningJob.from_dict(response)

        logger.info(f"Created training job: {job.id}")
        return job

    def get(self, job_id: str) -> TuningJob:
        """查詢訓練任務

        參數：
            job_id: 訓練任務 ID

        返回：
            TuningJob 對象

        範例：
            ```python
            job = client.tunings.get("job-abc123")
            print(f"Status: {job.status}, Progress: {job.metrics.progress if job.metrics else 0}")
            ```
        """
        response = self._client.get(f"/v1/tunings.get/{job_id}")
        job = TuningJob.from_dict(response)

        logger.debug(f"Retrieved job: {job.id}, status: {job.status}")
        return job

    def list(
        self,
        limit: int = 10,
        status_filter: Optional[JobStatus] = None
    ) -> JobList:
        """列出訓練任務

        參數：
            limit: 返回的最大任務數量
            status_filter: 按狀態過濾（可選）

        返回：
            JobList 對象

        範例：
            ```python
            # 列出所有任務
            jobs = client.tunings.list(limit=20)

            # 只列出正在運行的任務
            running_jobs = client.tunings.list(status_filter=JobStatus.RUNNING)
            ```
        """
        params = {"limit": limit}
        if status_filter:
            params["status_filter"] = status_filter.value

        response = self._client.get("/v1/tunings.list", params=params)
        job_list = JobList.from_dict(response)

        logger.info(f"Listed {len(job_list.data)}/{job_list.total} jobs")
        return job_list

    def cancel(self, job_id: str) -> TuningJob:
        """取消訓練任務

        參數：
            job_id: 訓練任務 ID

        返回：
            更新後的 TuningJob 對象

        範例：
            ```python
            job = client.tunings.cancel("job-abc123")
            print(f"Job cancelled: {job.status}")
            ```
        """
        response = self._client.post(f"/v1/tunings.cancel/{job_id}")
        job = TuningJob.from_dict(response)

        logger.info(f"Cancelled job: {job.id}")
        return job

    def delete(self, job_id: str) -> None:
        """刪除訓練任務

        參數：
            job_id: 訓練任務 ID

        範例：
            ```python
            client.tunings.delete("job-abc123")
            print("Job deleted")
            ```
        """
        self._client.delete(f"/v1/tunings.delete/{job_id}")
        logger.info(f"Deleted job: {job_id}")

    def wait(
        self,
        job_id: str,
        poll_interval: int = 2,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[TuningJob], None]] = None
    ) -> TuningJob:
        """等待訓練任務完成

        輪詢任務狀態直到完成（成功、失敗或取消）。

        參數：
            job_id: 訓練任務 ID
            poll_interval: 輪詢間隔（秒）
            timeout: 最大等待時間（秒），None 表示無限等待
            on_progress: 進度回調函數，每次輪詢時調用

        返回：
            最終的 TuningJob 對象

        異常：
            TimeoutError: 等待超時
            TuningAPIError: 任務失敗

        範例：
            ```python
            def print_progress(job):
                if job.metrics:
                    print(f"Progress: {job.metrics.progress * 100:.1f}%")

            job = client.tunings.wait(
                "job-abc123",
                poll_interval=5,
                on_progress=print_progress
            )
            print(f"Final status: {job.status}")
            ```
        """
        start_time = time.time()

        logger.info(f"Waiting for job {job_id} to complete...")

        while True:
            job = self.get(job_id)

            # 調用進度回調
            if on_progress:
                try:
                    on_progress(job)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

            # 檢查是否完成
            if job.is_terminal:
                if job.is_successful:
                    logger.info(f"Job {job_id} completed successfully")
                elif job.is_failed:
                    logger.error(f"Job {job_id} failed: {job.error}")
                elif job.is_cancelled:
                    logger.info(f"Job {job_id} was cancelled")
                return job

            # 檢查超時
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Waiting for job {job_id} timed out after {timeout}s")

            # 等待下一次輪詢
            time.sleep(poll_interval)

    def stream_logs(
        self,
        job_id: str,
        poll_interval: int = 2,
        timeout: Optional[int] = None
    ):
        """流式輸出訓練進度

        持續輸出訓練任務的進度信息，直到任務完成。

        參數：
            job_id: 訓練任務 ID
            poll_interval: 輪詢間隔（秒）
            timeout: 最大等待時間（秒）

        返回：
            生成器，每次返回更新後的 TuningJob 對象

        範例：
            ```python
            for job in client.tunings.stream_logs("job-abc123"):
                if job.metrics:
                    print(f"Epoch {job.metrics.current_epoch}/{job.metrics.total_epochs}, "
                          f"Loss: {job.metrics.train_loss:.4f}")
                if job.is_terminal:
                    break
            ```
        """
        start_time = time.time()

        while True:
            job = self.get(job_id)
            yield job

            if job.is_terminal:
                break

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Streaming job {job_id} timed out after {timeout}s")

            time.sleep(poll_interval)


# ============================================================================
# 第六部分：主客戶端類
# ============================================================================

class TuningClient:
    """微調訓練服務客戶端

    這是 SDK 的主要入口類，提供對所有 API 資源的訪問。

    使用範例：
    ---------
    ```python
    from client import TuningClient

    # 初始化客戶端
    client = TuningClient(
        api_key="test-key-001",
        base_url="http://localhost:8000"
    )

    # 創建訓練任務
    job = client.tunings.create(
        model="meta-llama/Llama-2-7b-hf",
        training_file="data/train.jsonl",
        hyperparameters={"epochs": 3}
    )

    # 等待完成
    final_job = client.tunings.wait(job.id)
    print(f"Training completed! Model: {final_job.fine_tuned_model}")
    ```
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """初始化客戶端

        參數：
            api_key: API 密鑰（必填）
            base_url: API 基礎 URL（默認 localhost:8000）
            timeout: 請求超時時間（秒）
            max_retries: 最大重試次數
        """
        self._http_client = HTTPClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )

        # 初始化資源
        self.tunings = TuningResource(self._http_client)

        logger.info(f"Initialized TuningClient for {base_url}")

    def health_check(self) -> Dict[str, Any]:
        """檢查服務健康狀態

        返回：
            包含服務狀態的字典

        範例：
            ```python
            health = client.health_check()
            print(f"Status: {health['status']}, Uptime: {health['uptime']:.0f}s")
            ```
        """
        return self._http_client.get("/health")

    def close(self):
        """關閉客戶端

        釋放底層的 HTTP 連接資源。
        建議在使用完客戶端後調用此方法。

        範例：
            ```python
            client = TuningClient(api_key="...")
            try:
                # 使用客戶端
                pass
            finally:
                client.close()
            ```
        """
        self._http_client.close()
        logger.info("TuningClient closed")

    def __enter__(self):
        """支持 context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 context manager"""
        self.close()


# ============================================================================
# 第七部分：便捷函數
# ============================================================================

def create_client(api_key: str, **kwargs) -> TuningClient:
    """創建客戶端的便捷函數

    參數：
        api_key: API 密鑰
        **kwargs: TuningClient 的其他參數

    返回：
        TuningClient 實例

    範例：
        ```python
        from client import create_client

        client = create_client("test-key-001")
        ```
    """
    return TuningClient(api_key=api_key, **kwargs)


# ============================================================================
# 第八部分：示例代碼
# ============================================================================

if __name__ == "__main__":
    """
    SDK 使用示例

    運行此腳本可以測試 SDK 的基本功能。
    """
    import sys

    # 配置日誌級別
    logging.basicConfig(level=logging.INFO)

    # 初始化客戶端
    client = TuningClient(
        api_key="test-key-001",
        base_url="http://localhost:8000"
    )

    try:
        # 1. 健康檢查
        print("=" * 60)
        print("1. Health Check")
        print("=" * 60)
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Uptime: {health['uptime']:.0f}s")
        print()

        # 2. 創建訓練任務
        print("=" * 60)
        print("2. Create Training Job")
        print("=" * 60)
        job = client.tunings.create(
            model="meta-llama/Llama-2-7b-hf",
            training_file="data/train.jsonl",
            validation_file="data/val.jsonl",
            hyperparameters={
                "epochs": 3,
                "learning_rate": 2e-4,
                "batch_size": 4,
                "lora_r": 8
            },
            suffix="demo"
        )
        print(f"Job ID: {job.id}")
        print(f"Model: {job.model}")
        print(f"Status: {job.status}")
        print()

        # 3. 等待訓練完成（帶進度顯示）
        print("=" * 60)
        print("3. Wait for Training to Complete")
        print("=" * 60)

        def show_progress(j: TuningJob):
            """顯示訓練進度"""
            if j.metrics:
                progress_bar = "=" * int(j.metrics.progress * 50)
                print(f"\rEpoch {j.metrics.current_epoch}/{j.metrics.total_epochs} | "
                      f"Loss: {j.metrics.train_loss:.4f} | "
                      f"Progress: [{progress_bar:<50}] {j.metrics.progress * 100:.1f}%",
                      end="", flush=True)

        final_job = client.tunings.wait(
            job.id,
            poll_interval=2,
            timeout=300,
            on_progress=show_progress
        )
        print()  # 換行
        print(f"\nFinal Status: {final_job.status}")
        if final_job.fine_tuned_model:
            print(f"Fine-tuned Model: {final_job.fine_tuned_model}")
        if final_job.result_files:
            print(f"Result Files: {', '.join(final_job.result_files)}")
        print()

        # 4. 列出所有任務
        print("=" * 60)
        print("4. List All Jobs")
        print("=" * 60)
        job_list = client.tunings.list(limit=5)
        print(f"Total jobs: {job_list.total}")
        for j in job_list.data:
            duration = f"{j.duration}s" if j.duration else "N/A"
            print(f"  - {j.id}: {j.status} (duration: {duration})")
        print()

    except TuningAPIError as e:
        print(f"\nAPI Error: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)

    finally:
        client.close()

    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
