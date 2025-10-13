"""
FastAPI 服務測試
================

測試 tuning_service/app.py 中的所有 API 端點。

運行測試：
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --cov=tuning_service

依賴：
    pip install pytest pytest-asyncio httpx
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi import status

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tuning_service.app import app, app_state


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def reset_app_state():
    """每個測試前重置應用狀態"""
    # 取消所有背景任務
    for task in app_state.background_tasks.values():
        if not task.done():
            task.cancel()

    # 清空數據
    app_state.jobs.clear()
    app_state.background_tasks.clear()

    yield

    # 測試後再次清理
    for task in app_state.background_tasks.values():
        if not task.done():
            task.cancel()
    app_state.jobs.clear()
    app_state.background_tasks.clear()


@pytest.fixture
def valid_headers():
    """有效的認證頭"""
    return {"Authorization": "Bearer test-key-001"}


@pytest.fixture
def invalid_headers():
    """無效的認證頭"""
    return {"Authorization": "Bearer invalid-key"}


# ============================================================================
# 健康檢查測試
# ============================================================================

@pytest.mark.asyncio
async def test_health_check():
    """測試健康檢查端點"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "uptime" in data
    assert "active_jobs" in data
    assert "total_jobs" in data


@pytest.mark.asyncio
async def test_root_endpoint():
    """測試根路徑"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


# ============================================================================
# 認證測試
# ============================================================================

@pytest.mark.asyncio
async def test_create_job_without_auth(reset_app_state):
    """測試沒有認證頭的創建請求"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/tunings.create",
            json={
                "model": "test-model",
                "training_file": "data/train.jsonl"
            }
        )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "error" in response.json()


@pytest.mark.asyncio
async def test_create_job_with_invalid_auth(reset_app_state, invalid_headers):
    """測試使用無效認證頭的創建請求"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/tunings.create",
            headers=invalid_headers,
            json={
                "model": "test-model",
                "training_file": "data/train.jsonl"
            }
        )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ============================================================================
# 創建任務測試
# ============================================================================

@pytest.mark.asyncio
async def test_create_job_success(reset_app_state, valid_headers):
    """測試成功創建訓練任務"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/tunings.create",
            headers=valid_headers,
            json={
                "model": "meta-llama/Llama-2-7b-hf",
                "training_file": "data/train.jsonl",
                "hyperparameters": {
                    "epochs": 2,
                    "learning_rate": 2e-4
                }
            }
        )

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert "id" in data
    assert data["model"] == "meta-llama/Llama-2-7b-hf"
    assert data["status"] == "pending"
    assert data["training_file"] == "data/train.jsonl"
    assert data["hyperparameters"]["epochs"] == 2


@pytest.mark.asyncio
async def test_create_job_with_validation_file(reset_app_state, valid_headers):
    """測試創建帶驗證資料的任務"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/tunings.create",
            headers=valid_headers,
            json={
                "model": "test-model",
                "training_file": "data/train.jsonl",
                "validation_file": "data/val.jsonl",
                "suffix": "test-run"
            }
        )

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["validation_file"] == "data/val.jsonl"


@pytest.mark.asyncio
async def test_create_job_invalid_file_format(reset_app_state, valid_headers):
    """測試無效的檔案格式"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/tunings.create",
            headers=valid_headers,
            json={
                "model": "test-model",
                "training_file": "data/train.txt"  # 不支持的格式
            }
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ============================================================================
# 查詢任務測試
# ============================================================================

@pytest.mark.asyncio
async def test_get_job_success(reset_app_state, valid_headers):
    """測試成功查詢任務"""
    # 先創建一個任務
    async with AsyncClient(app=app, base_url="http://test") as ac:
        create_response = await ac.post(
            "/v1/tunings.create",
            headers=valid_headers,
            json={
                "model": "test-model",
                "training_file": "data/train.jsonl"
            }
        )
        job_id = create_response.json()["id"]

        # 查詢任務
        get_response = await ac.get(
            f"/v1/tunings.get/{job_id}",
            headers=valid_headers
        )

    assert get_response.status_code == 200
    data = get_response.json()
    assert data["id"] == job_id
    assert "status" in data


@pytest.mark.asyncio
async def test_get_job_not_found(reset_app_state, valid_headers):
    """測試查詢不存在的任務"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get(
            "/v1/tunings.get/non-existent-id",
            headers=valid_headers
        )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "error" in response.json()


# ============================================================================
# 列出任務測試
# ============================================================================

@pytest.mark.asyncio
async def test_list_jobs_empty(reset_app_state, valid_headers):
    """測試列出空任務列表"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get(
            "/v1/tunings.list",
            headers=valid_headers
        )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["total"] == 0
    assert len(data["data"]) == 0


@pytest.mark.asyncio
async def test_list_jobs_with_data(reset_app_state, valid_headers):
    """測試列出任務"""
    # 創建 3 個任務
    async with AsyncClient(app=app, base_url="http://test") as ac:
        for i in range(3):
            await ac.post(
                "/v1/tunings.create",
                headers=valid_headers,
                json={
                    "model": f"test-model-{i}",
                    "training_file": "data/train.jsonl"
                }
            )

        # 列出任務
        response = await ac.get(
            "/v1/tunings.list?limit=10",
            headers=valid_headers
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["data"]) == 3


@pytest.mark.asyncio
async def test_list_jobs_with_limit(reset_app_state, valid_headers):
    """測試分頁限制"""
    # 創建 5 個任務
    async with AsyncClient(app=app, base_url="http://test") as ac:
        for i in range(5):
            await ac.post(
                "/v1/tunings.create",
                headers=valid_headers,
                json={
                    "model": f"test-model-{i}",
                    "training_file": "data/train.jsonl"
                }
            )

        # 限制返回 2 個
        response = await ac.get(
            "/v1/tunings.list?limit=2",
            headers=valid_headers
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["data"]) == 2
    assert data["has_more"] is True


# ============================================================================
# 取消任務測試
# ============================================================================

@pytest.mark.asyncio
async def test_cancel_job_success(reset_app_state, valid_headers):
    """測試成功取消任務"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # 創建任務
        create_response = await ac.post(
            "/v1/tunings.create",
            headers=valid_headers,
            json={
                "model": "test-model",
                "training_file": "data/train.jsonl"
            }
        )
        job_id = create_response.json()["id"]

        # 等待任務開始
        await asyncio.sleep(1)

        # 取消任務
        cancel_response = await ac.post(
            f"/v1/tunings.cancel/{job_id}",
            headers=valid_headers
        )

    assert cancel_response.status_code == 200
    data = cancel_response.json()
    assert data["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_job_not_found(reset_app_state, valid_headers):
    """測試取消不存在的任務"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/tunings.cancel/non-existent-id",
            headers=valid_headers
        )

    assert response.status_code == status.HTTP_404_NOT_FOUND


# ============================================================================
# 刪除任務測試
# ============================================================================

@pytest.mark.asyncio
async def test_delete_job_success(reset_app_state, valid_headers):
    """測試成功刪除任務"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # 創建任務
        create_response = await ac.post(
            "/v1/tunings.create",
            headers=valid_headers,
            json={
                "model": "test-model",
                "training_file": "data/train.jsonl"
            }
        )
        job_id = create_response.json()["id"]

        # 取消任務
        await ac.post(f"/v1/tunings.cancel/{job_id}", headers=valid_headers)

        # 等待取消完成
        await asyncio.sleep(1)

        # 刪除任務
        delete_response = await ac.delete(
            f"/v1/tunings.delete/{job_id}",
            headers=valid_headers
        )

    assert delete_response.status_code == status.HTTP_204_NO_CONTENT


@pytest.mark.asyncio
async def test_delete_running_job_fails(reset_app_state, valid_headers):
    """測試無法刪除正在運行的任務"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # 創建任務
        create_response = await ac.post(
            "/v1/tunings.create",
            headers=valid_headers,
            json={
                "model": "test-model",
                "training_file": "data/train.jsonl"
            }
        )
        job_id = create_response.json()["id"]

        # 等待任務開始
        await asyncio.sleep(1)

        # 嘗試刪除（應該失敗）
        delete_response = await ac.delete(
            f"/v1/tunings.delete/{job_id}",
            headers=valid_headers
        )

    assert delete_response.status_code == status.HTTP_400_BAD_REQUEST


# ============================================================================
# 訓練模擬測試
# ============================================================================

@pytest.mark.asyncio
async def test_job_lifecycle(reset_app_state, valid_headers):
    """測試任務完整生命週期"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # 創建任務
        create_response = await ac.post(
            "/v1/tunings.create",
            headers=valid_headers,
            json={
                "model": "test-model",
                "training_file": "data/train.jsonl",
                "hyperparameters": {"epochs": 1}  # 快速完成
            }
        )
        job_id = create_response.json()["id"]

        # 等待任務開始
        await asyncio.sleep(2)
        job = (await ac.get(f"/v1/tunings.get/{job_id}", headers=valid_headers)).json()
        assert job["status"] in ["pending", "running"]

        # 等待任務完成（簡化測試，不等待實際完成）
        # 在實際環境中可以等待更長時間
        await asyncio.sleep(5)
        job = (await ac.get(f"/v1/tunings.get/{job_id}", headers=valid_headers)).json()

        # 任務應該在運行或已完成
        assert job["status"] in ["running", "succeeded"]

        # 如果有指標，檢查指標
        if job.get("metrics"):
            assert "train_loss" in job["metrics"]
            assert "progress" in job["metrics"]


# ============================================================================
# 開發端點測試
# ============================================================================

@pytest.mark.asyncio
async def test_dev_list_all_jobs(reset_app_state):
    """測試開發端點：列出所有任務"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/dev/jobs")

    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert "background_tasks" in data


@pytest.mark.asyncio
async def test_dev_reset(reset_app_state):
    """測試開發端點：重置服務"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/dev/reset")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "timestamp" in data


# ============================================================================
# 運行測試
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
