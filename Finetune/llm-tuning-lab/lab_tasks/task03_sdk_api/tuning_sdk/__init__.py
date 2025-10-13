"""
EdgeAI Tuning SDK
=================

Python SDK for EdgeAI Tuning Service API.

基本使用：
---------
```python
from tuning_sdk import TuningClient

client = TuningClient(api_key="your-api-key")
job = client.tunings.create(
    model="meta-llama/Llama-2-7b-hf",
    training_file="data/train.jsonl"
)
```
"""

from .client import (
    TuningClient,
    create_client,
    TuningResource,
    HTTPClient,

    # 數據模型
    TuningJob,
    JobList,
    Hyperparameters,
    TrainingMetrics,
    JobStatus,

    # 異常
    TuningAPIError,
    AuthenticationError,
    PermissionError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ConnectionError,
)

__version__ = "1.0.0"
__author__ = "EdgeAI Lab"

__all__ = [
    # 主類
    "TuningClient",
    "create_client",
    "TuningResource",
    "HTTPClient",

    # 數據模型
    "TuningJob",
    "JobList",
    "Hyperparameters",
    "TrainingMetrics",
    "JobStatus",

    # 異常
    "TuningAPIError",
    "AuthenticationError",
    "PermissionError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "ConnectionError",
]
