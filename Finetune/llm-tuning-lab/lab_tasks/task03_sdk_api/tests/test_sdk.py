"""
SDK 客戶端測試
==============

測試 sdk_client/client.py 中的 SDK 功能。

運行測試：
    pytest tests/test_sdk.py -v

注意：這些測試需要 FastAPI 服務正在運行。
"""

import pytest
import time

from tuning_sdk import (
    TuningClient,
    JobStatus,
    TuningAPIError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    TimeoutError,
    Hyperparameters,
    TuningJob
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """創建測試客戶端"""
    return TuningClient(api_key="test-key-001", base_url="http://localhost:8000")


@pytest.fixture
def invalid_client():
    """創建無效認證的客戶端"""
    return TuningClient(api_key="invalid-key", base_url="http://localhost:8000")


# ============================================================================
# 客戶端初始化測試
# ============================================================================

def test_client_initialization():
    """測試客戶端初始化"""
    client = TuningClient(api_key="test-key")
    assert client._http_client.api_key == "test-key"
    assert client._http_client.base_url == "http://localhost:8000"
    assert client.tunings is not None
    client.close()


def test_client_with_custom_url():
    """測試自定義 URL"""
    client = TuningClient(
        api_key="test-key",
        base_url="http://example.com:8080"
    )
    assert client._http_client.base_url == "http://example.com:8080"
    client.close()


def test_client_context_manager():
    """測試 Context Manager"""
    with TuningClient(api_key="test-key") as client:
        assert client is not None
    # 客戶端應該已關閉


# ============================================================================
# 健康檢查測試
# ============================================================================

def test_health_check(client):
    """測試健康檢查"""
    try:
        health = client.health_check()
        assert health["status"] == "healthy"
        assert "version" in health
        assert "uptime" in health
    finally:
        client.close()


# ============================================================================
# 認證測試
# ============================================================================

def test_authentication_error(invalid_client):
    """測試認證失敗"""
    try:
        with pytest.raises(AuthenticationError):
            invalid_client.tunings.create(
                model="test-model",
                training_file="data/train.jsonl"
            )
    finally:
        invalid_client.close()


# ============================================================================
# 創建任務測試
# ============================================================================

def test_create_job_basic(client):
    """測試基本任務創建"""
    try:
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl"
        )

        assert job.id is not None
        assert job.model == "test-model"
        assert job.status == JobStatus.PENDING
        assert job.training_file == "data/train.jsonl"
    finally:
        client.close()


def test_create_job_with_hyperparameters(client):
    """測試帶超參數的任務創建"""
    try:
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl",
            hyperparameters={
                "epochs": 5,
                "learning_rate": 1e-4,
                "batch_size": 8
            }
        )

        assert job.hyperparameters.epochs == 5
        assert job.hyperparameters.learning_rate == 1e-4
        assert job.hyperparameters.batch_size == 8
    finally:
        client.close()


def test_create_job_with_validation_file(client):
    """測試帶驗證檔案的任務創建"""
    try:
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl",
            validation_file="data/val.jsonl",
            suffix="test-suffix"
        )

        assert job.validation_file == "data/val.jsonl"
    finally:
        client.close()


def test_create_job_invalid_file_format(client):
    """測試無效檔案格式"""
    try:
        with pytest.raises(ValidationError):
            client.tunings.create(
                model="test-model",
                training_file="data/train.txt"  # 不支持的格式
            )
    finally:
        client.close()


# ============================================================================
# 查詢任務測試
# ============================================================================

def test_get_job(client):
    """測試查詢任務"""
    try:
        # 先創建任務
        created_job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl"
        )

        # 查詢任務
        job = client.tunings.get(created_job.id)

        assert job.id == created_job.id
        assert job.model == created_job.model
        assert job.status in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED]
    finally:
        client.close()


def test_get_job_not_found(client):
    """測試查詢不存在的任務"""
    try:
        with pytest.raises(NotFoundError):
            client.tunings.get("non-existent-job-id")
    finally:
        client.close()


# ============================================================================
# 列出任務測試
# ============================================================================

def test_list_jobs(client):
    """測試列出任務"""
    try:
        # 創建幾個任務
        for i in range(3):
            client.tunings.create(
                model=f"test-model-{i}",
                training_file="data/train.jsonl"
            )

        # 列出任務
        job_list = client.tunings.list(limit=10)

        assert job_list.total >= 3
        assert len(job_list.data) >= 3
        assert all(isinstance(job, TuningJob) for job in job_list.data)
    finally:
        client.close()


def test_list_jobs_with_limit(client):
    """測試分頁限制"""
    try:
        # 創建任務
        for i in range(5):
            client.tunings.create(
                model=f"test-model-{i}",
                training_file="data/train.jsonl"
            )

        # 限制返回數量
        job_list = client.tunings.list(limit=2)

        assert len(job_list.data) <= 2
        assert job_list.total >= 5
    finally:
        client.close()


def test_list_jobs_with_status_filter(client):
    """測試狀態過濾"""
    try:
        # 創建並取消一個任務
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl"
        )
        time.sleep(1)
        client.tunings.cancel(job.id)

        # 過濾已取消的任務
        job_list = client.tunings.list(status_filter=JobStatus.CANCELLED)

        assert all(j.status == JobStatus.CANCELLED for j in job_list.data)
    finally:
        client.close()


# ============================================================================
# 取消任務測試
# ============================================================================

def test_cancel_job(client):
    """測試取消任務"""
    try:
        # 創建任務
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl"
        )

        # 等待任務開始
        time.sleep(1)

        # 取消任務
        cancelled_job = client.tunings.cancel(job.id)

        assert cancelled_job.status == JobStatus.CANCELLED
        assert cancelled_job.id == job.id
    finally:
        client.close()


def test_cancel_nonexistent_job(client):
    """測試取消不存在的任務"""
    try:
        with pytest.raises(NotFoundError):
            client.tunings.cancel("non-existent-id")
    finally:
        client.close()


# ============================================================================
# 刪除任務測試
# ============================================================================

def test_delete_job(client):
    """測試刪除任務"""
    try:
        # 創建並取消任務
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl"
        )
        time.sleep(1)
        client.tunings.cancel(job.id)
        time.sleep(1)

        # 刪除任務
        client.tunings.delete(job.id)

        # 驗證任務已刪除
        with pytest.raises(NotFoundError):
            client.tunings.get(job.id)
    finally:
        client.close()


# ============================================================================
# 等待任務完成測試
# ============================================================================

def test_wait_for_job_completion(client):
    """測試等待任務完成"""
    try:
        # 創建短時間任務
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl",
            hyperparameters={"epochs": 1}
        )

        # 等待完成（設置較短超時用於測試）
        final_job = client.tunings.wait(
            job.id,
            poll_interval=2,
            timeout=60  # 60秒超時
        )

        assert final_job.is_terminal
        assert final_job.status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]
    finally:
        client.close()


def test_wait_with_progress_callback(client):
    """測試帶進度回調的等待"""
    progress_updates = []

    def track_progress(job):
        if job.metrics:
            progress_updates.append(job.metrics.progress)

    try:
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl",
            hyperparameters={"epochs": 1}
        )

        final_job = client.tunings.wait(
            job.id,
            poll_interval=2,
            timeout=60,
            on_progress=track_progress
        )

        # 應該有多個進度更新
        assert len(progress_updates) > 0
    finally:
        client.close()


def test_wait_timeout(client):
    """測試等待超時"""
    try:
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl",
            hyperparameters={"epochs": 10}  # 長時間訓練
        )

        # 設置很短的超時
        with pytest.raises(TimeoutError):
            client.tunings.wait(job.id, poll_interval=1, timeout=2)
    finally:
        client.close()


# ============================================================================
# 流式日誌測試
# ============================================================================

def test_stream_logs(client):
    """測試流式日誌"""
    try:
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl",
            hyperparameters={"epochs": 1}
        )

        updates = []
        for updated_job in client.tunings.stream_logs(job.id, poll_interval=2, timeout=60):
            updates.append(updated_job)
            if updated_job.is_terminal:
                break

        assert len(updates) > 0
        assert updates[-1].is_terminal
    finally:
        client.close()


# ============================================================================
# 資料模型測試
# ============================================================================

def test_hyperparameters_model():
    """測試超參數模型"""
    hp = Hyperparameters(
        learning_rate=1e-4,
        epochs=5,
        lora_r=16
    )

    assert hp.learning_rate == 1e-4
    assert hp.epochs == 5
    assert hp.lora_r == 16

    # 測試轉換為字典
    hp_dict = hp.to_dict()
    assert hp_dict["learning_rate"] == 1e-4

    # 測試從字典創建
    hp2 = Hyperparameters.from_dict(hp_dict)
    assert hp2.learning_rate == hp.learning_rate


def test_job_status_enum():
    """測試任務狀態枚舉"""
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.SUCCEEDED.value == "succeeded"
    assert JobStatus.FAILED.value == "failed"
    assert JobStatus.CANCELLED.value == "cancelled"

    # 測試 is_terminal
    assert JobStatus.SUCCEEDED.is_terminal()
    assert JobStatus.FAILED.is_terminal()
    assert not JobStatus.RUNNING.is_terminal()


def test_tuning_job_properties(client):
    """測試 TuningJob 屬性"""
    try:
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl"
        )

        assert job.is_terminal == False
        assert job.is_successful == False
        assert job.is_failed == False

        # 等待完成
        final_job = client.tunings.wait(job.id, timeout=60)

        if final_job.status == JobStatus.SUCCEEDED:
            assert final_job.is_successful
            assert final_job.is_terminal
            assert final_job.duration is not None
    finally:
        client.close()


# ============================================================================
# 錯誤處理測試
# ============================================================================

def test_http_client_error_handling(client):
    """測試 HTTP 錯誤處理"""
    try:
        # 測試不同類型的錯誤
        with pytest.raises(NotFoundError):
            client.tunings.get("invalid-id")

        with pytest.raises(ValidationError):
            client.tunings.create(
                model="",  # 空模型名
                training_file="data/train.jsonl"
            )
    finally:
        client.close()


# ============================================================================
# 運行測試
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
