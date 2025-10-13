"""
FastAPI 微調訓練服務
===================

這是一個完整的企業級 FastAPI 服務，提供 LoRA/QLoRA 微調訓練的 RESTful API。

主要功能：
---------
1. 訓練任務管理（CRUD 操作）
2. 異步訓練執行與狀態追蹤
3. API Key 認證與授權
4. 請求驗證與錯誤處理
5. 結構化日誌記錄
6. 健康檢查與監控
7. API 版本管理
8. 優雅關機

技術棧：
--------
- FastAPI: 現代化的 Python Web 框架
- Pydantic: 資料驗證與序列化
- asyncio: 異步任務處理
- uvicorn: ASGI 伺服器

學習目標：
----------
1. 理解 RESTful API 設計原則
2. 掌握 FastAPI 框架的高級功能
3. 學習異步編程與任務管理
4. 實踐企業級服務的最佳實踐

作者：EdgeAI Lab
版本：1.0.0
更新日期：2025-10-13
"""

import asyncio
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Header, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, ValidationError
import uvicorn


# ============================================================================
# 第一部分：配置與日誌設置
# ============================================================================

# 配置日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tuning_service.log')
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """服務配置類"""
    APP_NAME = "Tuning Service API"
    VERSION = "1.0.0"
    API_PREFIX = "/v1"

    # 認證配置
    VALID_API_KEYS = {
        "test-key-001": {"user_id": "user-001", "name": "Test User"},
        "test-key-002": {"user_id": "user-002", "name": "Demo User"},
    }

    # 訓練配置
    MAX_TRAINING_TIME = 3600  # 最大訓練時間（秒）
    TRAINING_SIMULATION_TIME = 30  # 模擬訓練時間（秒）

    # 服務配置
    MAX_CONCURRENT_JOBS = 10
    REQUEST_TIMEOUT = 60


# ============================================================================
# 第二部分：資料模型定義
# ============================================================================

class JobStatus(str, Enum):
    """訓練任務狀態枚舉

    狀態轉換規則：
    pending → running → succeeded/failed
    任何狀態 → cancelled（用戶主動取消）
    """
    PENDING = "pending"      # 等待執行
    RUNNING = "running"      # 正在執行
    SUCCEEDED = "succeeded"  # 成功完成
    FAILED = "failed"        # 執行失敗
    CANCELLED = "cancelled"  # 用戶取消


class Hyperparameters(BaseModel):
    """訓練超參數模型

    定義所有可配置的訓練參數，並設置合理的預設值和驗證規則。
    """
    learning_rate: float = Field(default=2e-4, ge=1e-6, le=1e-2, description="學習率")
    epochs: int = Field(default=3, ge=1, le=100, description="訓練輪數")
    batch_size: int = Field(default=4, ge=1, le=128, description="批次大小")
    lora_r: int = Field(default=8, ge=1, le=64, description="LoRA 秩")
    lora_alpha: int = Field(default=16, ge=1, le=128, description="LoRA alpha")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="LoRA dropout")
    max_seq_length: int = Field(default=512, ge=128, le=4096, description="最大序列長度")
    warmup_steps: int = Field(default=100, ge=0, le=1000, description="預熱步數")

    @validator('lora_alpha')
    def validate_lora_alpha(cls, v, values):
        """驗證 lora_alpha 通常是 lora_r 的 2 倍"""
        if 'lora_r' in values and v < values['lora_r']:
            logger.warning(f"lora_alpha ({v}) is less than lora_r ({values['lora_r']})")
        return v

    class Config:
        schema_extra = {
            "example": {
                "learning_rate": 2e-4,
                "epochs": 3,
                "batch_size": 4,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05
            }
        }


class TuningRequest(BaseModel):
    """創建訓練任務的請求模型

    此模型定義了啟動一個新訓練任務所需的所有參數。
    """
    model: str = Field(..., description="基礎模型名稱", min_length=1, max_length=100)
    training_file: str = Field(..., description="訓練資料檔案路徑", min_length=1)
    validation_file: Optional[str] = Field(None, description="驗證資料檔案路徑")
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters, description="訓練超參數")
    suffix: Optional[str] = Field(None, max_length=40, description="模型名稱後綴")

    @validator('training_file', 'validation_file')
    def validate_file_path(cls, v):
        """驗證檔案路徑格式"""
        if v and not (v.endswith('.json') or v.endswith('.jsonl') or v.endswith('.csv')):
            raise ValueError("訓練檔案必須是 .json, .jsonl 或 .csv 格式")
        return v

    class Config:
        schema_extra = {
            "example": {
                "model": "meta-llama/Llama-2-7b-hf",
                "training_file": "data/train.jsonl",
                "validation_file": "data/val.jsonl",
                "hyperparameters": {
                    "learning_rate": 2e-4,
                    "epochs": 3,
                    "batch_size": 4
                },
                "suffix": "customer-support"
            }
        }


class TrainingMetrics(BaseModel):
    """訓練指標模型

    記錄訓練過程中的各項指標，用於監控訓練進度和效果。
    """
    current_epoch: int = Field(description="當前訓練輪數")
    total_epochs: int = Field(description="總訓練輪數")
    current_step: int = Field(default=0, description="當前訓練步數")
    total_steps: int = Field(default=0, description="總訓練步數")
    train_loss: float = Field(description="訓練損失")
    val_loss: Optional[float] = Field(None, description="驗證損失")
    learning_rate: float = Field(description="當前學習率")
    progress: float = Field(ge=0.0, le=1.0, description="訓練進度（0-1）")
    throughput: Optional[float] = Field(None, description="訓練吞吐量（樣本/秒）")
    estimated_finish_time: Optional[int] = Field(None, description="預計完成時間（Unix 時間戳）")


class TuningJob(BaseModel):
    """訓練任務完整模型

    表示一個訓練任務的完整狀態，包含所有相關資訊。
    這是 API 返回的主要資料結構。
    """
    id: str = Field(description="任務唯一標識符")
    object: str = Field(default="tuning.job", description="對象類型")
    model: str = Field(description="基礎模型名稱")
    status: JobStatus = Field(description="任務狀態")
    created_at: int = Field(description="創建時間（Unix 時間戳）")
    started_at: Optional[int] = Field(None, description="開始時間（Unix 時間戳）")
    finished_at: Optional[int] = Field(None, description="完成時間（Unix 時間戳）")
    training_file: str = Field(description="訓練資料檔案")
    validation_file: Optional[str] = Field(None, description="驗證資料檔案")
    hyperparameters: Hyperparameters = Field(description="訓練超參數")
    fine_tuned_model: Optional[str] = Field(None, description="微調後的模型名稱")
    metrics: Optional[TrainingMetrics] = Field(None, description="訓練指標")
    error: Optional[str] = Field(None, description="錯誤訊息")
    result_files: Optional[List[str]] = Field(default_factory=list, description="結果檔案列表")


class JobListResponse(BaseModel):
    """任務列表響應模型"""
    object: str = Field(default="list", description="對象類型")
    data: List[TuningJob] = Field(description="任務列表")
    has_more: bool = Field(default=False, description="是否有更多資料")
    total: int = Field(description="總任務數")


class ErrorResponse(BaseModel):
    """錯誤響應模型"""
    error: Dict[str, Any] = Field(description="錯誤詳情")

    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "message": "Job not found",
                    "type": "not_found_error",
                    "code": "job_not_found"
                }
            }
        }


class HealthResponse(BaseModel):
    """健康檢查響應模型"""
    status: str = Field(description="服務狀態")
    version: str = Field(description="服務版本")
    uptime: float = Field(description="運行時間（秒）")
    active_jobs: int = Field(description="活躍任務數")
    total_jobs: int = Field(description="總任務數")


# ============================================================================
# 第三部分：全局狀態管理
# ============================================================================

class ApplicationState:
    """應用程式全局狀態管理

    使用單例模式管理應用程式的全局狀態，包括任務儲存、背景任務等。
    """
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.background_tasks: Dict[str, asyncio.Task] = {}
        self.start_time = time.time()
        self.shutdown_event = asyncio.Event()
        logger.info("Application state initialized")

    def get_uptime(self) -> float:
        """獲取服務運行時間"""
        return time.time() - self.start_time

    def get_active_jobs_count(self) -> int:
        """獲取活躍任務數量"""
        return sum(1 for job in self.jobs.values()
                  if job["status"] in [JobStatus.PENDING, JobStatus.RUNNING])

    def cleanup_finished_tasks(self):
        """清理已完成的背景任務"""
        finished_tasks = [
            job_id for job_id, task in self.background_tasks.items()
            if task.done()
        ]
        for job_id in finished_tasks:
            del self.background_tasks[job_id]
        if finished_tasks:
            logger.info(f"Cleaned up {len(finished_tasks)} finished tasks")


# 全局應用狀態實例
app_state = ApplicationState()


# ============================================================================
# 第四部分：生命週期管理
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理

    處理應用程式啟動和關閉時的資源管理。
    """
    # 啟動邏輯
    logger.info(f"Starting {Config.APP_NAME} v{Config.VERSION}")
    logger.info(f"API documentation available at: http://localhost:8000/docs")

    yield

    # 關閉邏輯
    logger.info("Shutting down gracefully...")
    app_state.shutdown_event.set()

    # 取消所有正在運行的任務
    for job_id, task in app_state.background_tasks.items():
        if not task.done():
            logger.info(f"Cancelling job: {job_id}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    logger.info("Shutdown complete")


# ============================================================================
# 第五部分：FastAPI 應用程式初始化
# ============================================================================

app = FastAPI(
    title=Config.APP_NAME,
    version=Config.VERSION,
    description="""
    # LoRA/QLoRA 微調訓練服務 API

    這是一個企業級的 RESTful API 服務，提供大語言模型微調訓練功能。

    ## 主要功能

    - **訓練任務管理**：創建、查詢、列出、取消訓練任務
    - **異步訓練**：後台執行訓練，即時更新狀態
    - **API 認證**：基於 API Key 的安全認證
    - **完整監控**：訓練指標追蹤與健康檢查

    ## 快速開始

    1. 獲取 API Key（測試環境使用 `test-key-001`）
    2. 在請求頭中添加：`Authorization: Bearer <your-api-key>`
    3. 調用 API 端點進行訓練任務操作

    ## 範例

    ```bash
    curl -X POST http://localhost:8000/v1/tunings.create \\
      -H "Authorization: Bearer test-key-001" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "training_file": "data/train.jsonl",
        "hyperparameters": {"epochs": 3}
      }'
    ```
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 添加 CORS 中間件（開發環境使用，生產環境需要限制來源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境應該限制為特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 第六部分：認證與授權
# ============================================================================

async def verify_api_key(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """驗證 API Key

    從請求頭中提取並驗證 API Key，返回用戶資訊。

    參數：
        authorization: Authorization 請求頭，格式為 "Bearer <api-key>"

    返回：
        用戶資訊字典，包含 user_id 和 name

    異常：
        HTTPException: 當 API Key 無效或缺失時拋出 401 錯誤
    """
    if not authorization:
        logger.warning("Missing authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 解析 Bearer token
    try:
        scheme, credentials = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Invalid authentication scheme")
    except ValueError:
        logger.warning(f"Invalid authorization format: {authorization}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Expected: Bearer <api-key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 驗證 API Key
    user_info = Config.VALID_API_KEYS.get(credentials)
    if not user_info:
        logger.warning(f"Invalid API key: {credentials[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info(f"Authenticated user: {user_info['user_id']}")
    return user_info


# ============================================================================
# 第七部分：異步訓練模擬器
# ============================================================================

async def simulate_training(job_id: str):
    """模擬訓練過程

    這個函數模擬實際的訓練過程，包括：
    1. 狀態轉換（pending → running → succeeded/failed）
    2. 訓練指標更新（loss、進度等）
    3. 錯誤處理與恢復

    在實際應用中，這裡會調用真實的訓練腳本（如 task01 和 task02 的訓練代碼）。

    參數：
        job_id: 訓練任務 ID
    """
    job = app_state.jobs[job_id]

    try:
        # 階段 1：初始化訓練
        logger.info(f"Starting training job: {job_id}")
        job["status"] = JobStatus.RUNNING
        job["started_at"] = int(datetime.now().timestamp())

        # 獲取訓練參數
        hyperparams = job["hyperparameters"]
        total_epochs = hyperparams.get("epochs", 3)
        batch_size = hyperparams.get("batch_size", 4)
        learning_rate = hyperparams.get("learning_rate", 2e-4)

        # 模擬每個 epoch 的訓練
        for epoch in range(1, total_epochs + 1):
            # 檢查是否被取消
            if app_state.shutdown_event.is_set() or job["status"] == JobStatus.CANCELLED:
                logger.info(f"Job {job_id} cancelled at epoch {epoch}")
                return

            # 模擬訓練時間
            epoch_duration = Config.TRAINING_SIMULATION_TIME / total_epochs
            steps_per_epoch = 10

            for step in range(1, steps_per_epoch + 1):
                await asyncio.sleep(epoch_duration / steps_per_epoch)

                # 檢查取消狀態
                if job["status"] == JobStatus.CANCELLED:
                    return

                # 計算訓練損失（模擬下降趨勢）
                progress = (epoch - 1) / total_epochs + step / (steps_per_epoch * total_epochs)
                train_loss = 2.5 * (1 - progress) + 0.5  # 從 2.5 降到 0.5
                val_loss = train_loss + 0.1  # 驗證損失略高

                # 計算當前學習率（模擬 warmup 和 decay）
                warmup_steps = hyperparams.get("warmup_steps", 100)
                current_step = (epoch - 1) * steps_per_epoch + step
                total_steps = total_epochs * steps_per_epoch

                if current_step < warmup_steps:
                    current_lr = learning_rate * (current_step / warmup_steps)
                else:
                    current_lr = learning_rate * (1 - (current_step - warmup_steps) / (total_steps - warmup_steps))

                # 更新訓練指標
                job["metrics"] = {
                    "current_epoch": epoch,
                    "total_epochs": total_epochs,
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4) if job.get("validation_file") else None,
                    "learning_rate": round(current_lr, 6),
                    "progress": round(progress, 3),
                    "throughput": round(batch_size * 10 / epoch_duration, 2),
                    "estimated_finish_time": int(time.time() + (1 - progress) * Config.TRAINING_SIMULATION_TIME)
                }

                logger.debug(f"Job {job_id}: Epoch {epoch}/{total_epochs}, Step {step}/{steps_per_epoch}, Loss: {train_loss:.4f}")

        # 階段 2：訓練完成
        job["status"] = JobStatus.SUCCEEDED
        job["finished_at"] = int(datetime.now().timestamp())

        # 生成微調後的模型名稱
        suffix = job.get("suffix", "finetuned")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job["fine_tuned_model"] = f"{job['model']}-{suffix}-{timestamp}"

        # 生成結果檔案列表
        job["result_files"] = [
            f"models/{job['fine_tuned_model']}/adapter_model.bin",
            f"models/{job['fine_tuned_model']}/adapter_config.json",
            f"models/{job['fine_tuned_model']}/training_args.json",
            f"logs/{job_id}/training_log.txt",
        ]

        logger.info(f"Job {job_id} completed successfully. Model: {job['fine_tuned_model']}")

    except asyncio.CancelledError:
        # 任務被取消
        job["status"] = JobStatus.CANCELLED
        job["finished_at"] = int(datetime.now().timestamp())
        logger.info(f"Job {job_id} was cancelled")
        raise

    except Exception as e:
        # 訓練失敗
        job["status"] = JobStatus.FAILED
        job["finished_at"] = int(datetime.now().timestamp())
        job["error"] = str(e)
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)


# ============================================================================
# 第八部分：全局異常處理器
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """處理 HTTP 異常"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "http_error",
                "code": f"http_{exc.status_code}"
            }
        }
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """處理 Pydantic 驗證異常"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Request validation failed",
                "type": "validation_error",
                "code": "invalid_request",
                "details": exc.errors()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """處理未捕獲的異常"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "internal_error"
            }
        }
    )


# ============================================================================
# 第九部分：API 端點實現
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """根路徑重定向到文檔"""
    return {
        "message": f"Welcome to {Config.APP_NAME}",
        "version": Config.VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """健康檢查端點

    用於監控服務狀態，不需要認證。
    Kubernetes/Docker 可以使用此端點進行 liveness 和 readiness 探測。
    """
    app_state.cleanup_finished_tasks()

    return HealthResponse(
        status="healthy",
        version=Config.VERSION,
        uptime=app_state.get_uptime(),
        active_jobs=app_state.get_active_jobs_count(),
        total_jobs=len(app_state.jobs)
    )


@app.post(
    f"{Config.API_PREFIX}/tunings.create",
    response_model=TuningJob,
    status_code=status.HTTP_201_CREATED,
    tags=["Training Jobs"],
    summary="創建訓練任務",
    description="創建一個新的微調訓練任務。任務將在後台異步執行。"
)
async def create_tuning(
    request: TuningRequest,
    user_info: Dict = Depends(verify_api_key)
):
    """創建訓練任務

    這個端點會：
    1. 驗證請求參數
    2. 檢查並發任務限制
    3. 創建新任務並分配 ID
    4. 啟動後台訓練任務
    5. 返回任務資訊
    """
    # 檢查並發任務限制
    active_jobs = app_state.get_active_jobs_count()
    if active_jobs >= Config.MAX_CONCURRENT_JOBS:
        logger.warning(f"Max concurrent jobs reached: {active_jobs}/{Config.MAX_CONCURRENT_JOBS}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Maximum concurrent jobs ({Config.MAX_CONCURRENT_JOBS}) reached. Please wait for existing jobs to complete."
        )

    # 生成任務 ID
    job_id = f"job-{uuid.uuid4().hex[:12]}"

    # 創建任務記錄
    job_data = {
        "id": job_id,
        "object": "tuning.job",
        "model": request.model,
        "status": JobStatus.PENDING,
        "created_at": int(datetime.now().timestamp()),
        "started_at": None,
        "finished_at": None,
        "training_file": request.training_file,
        "validation_file": request.validation_file,
        "hyperparameters": request.hyperparameters.dict(),
        "suffix": request.suffix,
        "fine_tuned_model": None,
        "metrics": None,
        "error": None,
        "result_files": [],
        "user_id": user_info["user_id"]
    }

    app_state.jobs[job_id] = job_data

    # 啟動後台訓練任務
    task = asyncio.create_task(simulate_training(job_id))
    app_state.background_tasks[job_id] = task

    logger.info(f"Created job {job_id} for user {user_info['user_id']}, model: {request.model}")

    return TuningJob(**job_data)


@app.get(
    f"{Config.API_PREFIX}/tunings.get/{{job_id}}",
    response_model=TuningJob,
    tags=["Training Jobs"],
    summary="查詢訓練任務",
    description="根據任務 ID 查詢訓練任務的詳細資訊和當前狀態。"
)
async def get_tuning(
    job_id: str,
    user_info: Dict = Depends(verify_api_key)
):
    """查詢訓練任務

    返回指定任務的完整資訊，包括當前狀態、訓練指標等。
    """
    if job_id not in app_state.jobs:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    job_data = app_state.jobs[job_id]

    # 檢查用戶權限（簡化版本，實際應用需要更完善的權限控制）
    # if job_data["user_id"] != user_info["user_id"]:
    #     raise HTTPException(status_code=403, detail="Access denied")

    logger.info(f"Retrieved job {job_id} for user {user_info['user_id']}")

    return TuningJob(**job_data)


@app.get(
    f"{Config.API_PREFIX}/tunings.list",
    response_model=JobListResponse,
    tags=["Training Jobs"],
    summary="列出訓練任務",
    description="列出所有訓練任務，支持分頁和過濾。"
)
async def list_tunings(
    limit: int = 10,
    status_filter: Optional[JobStatus] = None,
    user_info: Dict = Depends(verify_api_key)
):
    """列出訓練任務

    參數：
        limit: 返回的最大任務數量
        status_filter: 按狀態過濾任務
    """
    # 過濾任務（可以根據用戶 ID、狀態等條件）
    filtered_jobs = []
    for job_data in app_state.jobs.values():
        # 根據狀態過濾
        if status_filter and job_data["status"] != status_filter:
            continue
        filtered_jobs.append(TuningJob(**job_data))

    # 按創建時間倒序排序
    filtered_jobs.sort(key=lambda x: x.created_at, reverse=True)

    # 分頁
    total = len(filtered_jobs)
    jobs_page = filtered_jobs[:limit]
    has_more = total > limit

    logger.info(f"Listed {len(jobs_page)}/{total} jobs for user {user_info['user_id']}")

    return JobListResponse(
        object="list",
        data=jobs_page,
        has_more=has_more,
        total=total
    )


@app.post(
    f"{Config.API_PREFIX}/tunings.cancel/{{job_id}}",
    response_model=TuningJob,
    tags=["Training Jobs"],
    summary="取消訓練任務",
    description="取消正在執行或等待中的訓練任務。"
)
async def cancel_tuning(
    job_id: str,
    user_info: Dict = Depends(verify_api_key)
):
    """取消訓練任務

    只能取消狀態為 pending 或 running 的任務。
    """
    if job_id not in app_state.jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    job_data = app_state.jobs[job_id]

    # 檢查任務狀態
    if job_data["status"] not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in status: {job_data['status']}"
        )

    # 取消背景任務
    if job_id in app_state.background_tasks:
        task = app_state.background_tasks[job_id]
        if not task.done():
            task.cancel()

    # 更新任務狀態
    job_data["status"] = JobStatus.CANCELLED
    job_data["finished_at"] = int(datetime.now().timestamp())

    logger.info(f"Cancelled job {job_id} by user {user_info['user_id']}")

    return TuningJob(**job_data)


@app.delete(
    f"{Config.API_PREFIX}/tunings.delete/{{job_id}}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Training Jobs"],
    summary="刪除訓練任務",
    description="刪除已完成或已取消的訓練任務記錄。"
)
async def delete_tuning(
    job_id: str,
    user_info: Dict = Depends(verify_api_key)
):
    """刪除訓練任務

    只能刪除已完成（succeeded、failed、cancelled）的任務。
    """
    if job_id not in app_state.jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    job_data = app_state.jobs[job_id]

    # 檢查任務狀態
    if job_data["status"] in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete job in status: {job_data['status']}. Cancel it first."
        )

    # 刪除任務
    del app_state.jobs[job_id]

    # 清理背景任務
    if job_id in app_state.background_tasks:
        del app_state.background_tasks[job_id]

    logger.info(f"Deleted job {job_id} by user {user_info['user_id']}")

    return None


# ============================================================================
# 第十部分：開發工具端點（生產環境應禁用）
# ============================================================================

@app.get(
    "/dev/jobs",
    tags=["Development"],
    summary="[開發] 查看所有任務",
    description="開發工具：查看所有任務的內部狀態（生產環境應禁用）"
)
async def dev_list_all_jobs():
    """開發工具：列出所有任務的原始資料"""
    return {
        "jobs": app_state.jobs,
        "background_tasks": {
            job_id: {"done": task.done(), "cancelled": task.cancelled()}
            for job_id, task in app_state.background_tasks.items()
        }
    }


@app.post(
    "/dev/reset",
    tags=["Development"],
    summary="[開發] 重置服務",
    description="開發工具：清除所有任務和背景任務（生產環境應禁用）"
)
async def dev_reset():
    """開發工具：重置服務狀態"""
    # 取消所有背景任務
    for task in app_state.background_tasks.values():
        if not task.done():
            task.cancel()

    # 清除所有資料
    app_state.jobs.clear()
    app_state.background_tasks.clear()

    logger.warning("Service state has been reset (dev mode)")

    return {"message": "Service reset successfully", "timestamp": int(time.time())}


# ============================================================================
# 第十一部分：應用程式入口
# ============================================================================

if __name__ == "__main__":
    """
    直接運行此檔案啟動服務：

    python app.py

    或使用 uvicorn 命令：

    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    """
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 開發模式自動重載
        log_level="info",
        access_log=True
    )
