# SDK 使用範例

本目錄包含 SDK 的各種使用範例，從基礎到進階。

## 目錄結構

```
examples/
├── README.md           # 本文件
├── basic_usage.py      # 基礎使用範例
└── advanced_usage.py   # 進階使用範例
```

## 前置準備

### 1. 安裝依賴

```bash
cd lab_tasks/task03_sdk_api
pip install -r requirements.txt
```

### 2. 啟動 FastAPI 服務

在一個終端窗口中：

```bash
cd tuning_service
python app.py
```

服務會在 `http://localhost:8000` 啟動。

### 3. 驗證服務運行

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

## 範例說明

### basic_usage.py - 基礎使用

展示 SDK 的基本操作：

1. **初始化客戶端**
2. **健康檢查**
3. **創建訓練任務**
4. **查詢任務狀態**
5. **等待任務完成**（帶進度顯示）
6. **顯示訓練結果**
7. **列出所有任務**

**運行方式：**

```bash
cd examples
python basic_usage.py
```

**預期輸出：**

```
======================================================================
SDK 基本使用範例
======================================================================

1. 初始化客戶端...
   ✓ 客戶端初始化完成

2. 檢查服務健康狀態...
   ✓ 服務狀態: healthy
   ✓ 服務版本: 1.0.0
   ✓ 運行時間: 45 秒

3. 創建訓練任務...
   ✓ 任務已創建
   - 任務 ID: job-abc123def456
   - 模型: meta-llama/Llama-2-7b-hf
   - 狀態: pending

4. 查詢任務狀態...
   ✓ 當前狀態: running
   - 進度: 15.5%
   - 訓練損失: 1.8543

5. 等待任務完成...
   (這可能需要 30 秒左右)

   [████████████████████████████] 100.0% | Epoch 3/3 | Loss: 0.5234

6. 訓練完成！
   ✓ 最終狀態: succeeded
   ✓ 微調模型: meta-llama/Llama-2-7b-hf-basic-example-20251013-154530
   ✓ 訓練時長: 32 秒
   ✓ 結果檔案:
      - models/.../adapter_model.bin
      - models/.../adapter_config.json
      - models/.../training_args.json
      - logs/.../training_log.txt

7. 列出最近的任務...
   ✓ 共有 5 個任務
   ✓ job-abc123: succeeded (meta-llama/Llama-2-7b-hf)
   ⋯ job-def456: running (test-model)
   ✗ job-ghi789: failed (another-model)

======================================================================
範例執行完成！
======================================================================
```

### advanced_usage.py - 進階使用

展示 SDK 的進階功能，包含 6 個子範例：

#### 範例 1：錯誤處理

- 測試無效 API Key
- 測試查詢不存在的任務
- 展示各種異常類型的處理

#### 範例 2：Context Manager

- 使用 `with` 語句自動管理資源
- 確保客戶端正確關閉

#### 範例 3：取消訓練任務

- 創建長時間訓練任務
- 在執行中取消任務
- 驗證取消狀態

#### 範例 4：流式輸出訓練日誌

- 使用 `stream_logs()` 方法
- 實時顯示訓練進度
- 展示訓練指標變化

#### 範例 5：批量操作

- 批量創建多個訓練任務
- 批量查詢任務狀態
- 批量取消任務

#### 範例 6：自定義進度條

- 自定義進度顯示格式
- 使用回調函數處理進度
- 計算預計完成時間

**運行方式：**

```bash
cd examples
python advanced_usage.py
```

**互動式選單：**

```
╔════════════════════════════════════════════════════════════════════╗
║                    SDK 進階使用範例                    ║
╚════════════════════════════════════════════════════════════════════╝

請選擇要執行的範例：
1. 錯誤處理
2. Context Manager
3. 取消任務
4. 流式日誌
5. 批量操作
6. 自定義進度條
0. 執行所有範例

請輸入選項 (0-6):
```

## 常見問題

### Q: 範例執行失敗，顯示連接錯誤

**A:** 確保 FastAPI 服務已啟動：

```bash
# 在另一個終端窗口
cd tuning_service
python app.py
```

### Q: 認證失敗

**A:** 測試環境使用的 API Key 是 `test-key-001` 或 `test-key-002`。檢查代碼中的 API Key 是否正確。

### Q: 訓練任務一直處於 pending 狀態

**A:** 檢查服務日誌，可能是：
- 達到並發任務上限（默認 10）
- 後台任務執行出錯

### Q: 如何加快測試速度？

**A:** 修改訓練參數：

```python
job = client.tunings.create(
    model="test-model",
    training_file="data/train.jsonl",
    hyperparameters={
        "epochs": 1,  # 減少訓練輪數
    }
)
```

或修改服務配置（`tuning_service/app.py`）：

```python
class Config:
    TRAINING_SIMULATION_TIME = 10  # 從 30 改為 10 秒
```

## 擴展學習

### 1. 添加錯誤重試

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def create_job_with_retry(client):
    return client.tunings.create(
        model="test-model",
        training_file="data/train.jsonl"
    )
```

### 2. 並行處理多個任務

```python
import concurrent.futures

def create_and_wait(client, model_name):
    job = client.tunings.create(
        model=model_name,
        training_file="data/train.jsonl"
    )
    return client.tunings.wait(job.id)

with TuningClient(api_key="test-key-001") as client:
    models = ["model-1", "model-2", "model-3"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(create_and_wait, client, model) for model in models]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
```

### 3. 保存訓練記錄

```python
import json
from datetime import datetime

def save_training_record(job):
    record = {
        "timestamp": datetime.now().isoformat(),
        "job_id": job.id,
        "model": job.model,
        "status": job.status.value,
        "duration": job.duration,
        "final_loss": job.metrics.train_loss if job.metrics else None,
        "fine_tuned_model": job.fine_tuned_model
    }

    with open(f"training_records/{job.id}.json", "w") as f:
        json.dump(record, f, indent=2)
```

### 4. 實現自動重啟失敗任務

```python
def train_with_auto_retry(client, max_retries=3, **create_params):
    for attempt in range(max_retries):
        try:
            job = client.tunings.create(**create_params)
            final_job = client.tunings.wait(job.id)

            if final_job.is_successful:
                return final_job
            elif final_job.is_failed:
                print(f"訓練失敗 (嘗試 {attempt+1}/{max_retries}): {final_job.error}")
                if attempt == max_retries - 1:
                    raise Exception(f"訓練失敗 {max_retries} 次")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"錯誤 (嘗試 {attempt+1}/{max_retries}): {e}")
```

## 相關資源

- **API 文檔**: http://localhost:8000/docs（啟動服務後訪問）
- **SDK 源碼**: `tuning_sdk/client.py`
- **服務源碼**: `tuning_service/app.py`
- **測試代碼**: `tests/test_sdk.py`, `tests/test_api.py`

## 反饋與貢獻

如果您發現問題或有改進建議，請：

1. 查看 `README.md` 中的故障排除部分
2. 查看 `GUIDE.md` 中的實作指南
3. 檢查服務日誌 `tuning_service.log`

Happy coding! 🚀
