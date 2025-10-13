"""
進階使用範例
============

這個範例展示了 SDK 的進階功能：
- 錯誤處理
- 任務取消
- 流式日誌
- Context Manager

執行前請確保 FastAPI 服務已啟動。
"""

import sys
import os
import time

from tuning_sdk import (
    TuningClient,
    JobStatus,
    TuningAPIError,
    AuthenticationError,
    NotFoundError,
    TimeoutError
)


def example_1_error_handling():
    """範例 1：錯誤處理"""
    print("=" * 70)
    print("範例 1：錯誤處理")
    print("=" * 70)
    print()

    # 測試無效的 API Key
    print("1.1 測試無效的 API Key...")
    client = TuningClient(api_key="invalid-key")
    try:
        client.health_check()  # 這個不需要認證
        print("   ✓ 健康檢查成功（不需要認證）")

        # 嘗試創建任務（需要認證）
        client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl"
        )
    except AuthenticationError as e:
        print(f"   ✓ 成功捕獲認證錯誤: {e}")
    except Exception as e:
        print(f"   ✗ 意外錯誤: {e}")
    finally:
        client.close()
    print()

    # 測試查詢不存在的任務
    print("1.2 測試查詢不存在的任務...")
    client = TuningClient(api_key="test-key-001")
    try:
        client.tunings.get("non-existent-job-id")
    except NotFoundError as e:
        print(f"   ✓ 成功捕獲 NotFound 錯誤: {e}")
    except Exception as e:
        print(f"   ✗ 意外錯誤: {e}")
    finally:
        client.close()
    print()


def example_2_context_manager():
    """範例 2：使用 Context Manager"""
    print("=" * 70)
    print("範例 2：Context Manager（自動資源管理）")
    print("=" * 70)
    print()

    # 使用 with 語句自動管理資源
    with TuningClient(api_key="test-key-001") as client:
        health = client.health_check()
        print(f"✓ 服務狀態: {health['status']}")
        print(f"✓ 活躍任務: {health['active_jobs']}")
        print(f"✓ 總任務數: {health['total_jobs']}")

    print("\n✓ 客戶端已自動關閉")
    print()


def example_3_cancel_job():
    """範例 3：取消訓練任務"""
    print("=" * 70)
    print("範例 3：取消訓練任務")
    print("=" * 70)
    print()

    with TuningClient(api_key="test-key-001") as client:
        # 創建訓練任務
        print("3.1 創建訓練任務...")
        job = client.tunings.create(
            model="test-model",
            training_file="data/train.jsonl",
            hyperparameters={"epochs": 5}  # 較長的訓練時間
        )
        print(f"   ✓ 任務已創建: {job.id}")
        print()

        # 等待一段時間
        print("3.2 等待任務開始執行...")
        time.sleep(3)
        job = client.tunings.get(job.id)
        print(f"   ✓ 當前狀態: {job.status}")
        print()

        # 取消任務
        if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            print("3.3 取消任務...")
            cancelled_job = client.tunings.cancel(job.id)
            print(f"   ✓ 任務已取消: {cancelled_job.status}")
        print()


def example_4_stream_logs():
    """範例 4：流式輸出訓練日誌"""
    print("=" * 70)
    print("範例 4：流式輸出訓練日誌")
    print("=" * 70)
    print()

    with TuningClient(api_key="test-key-001") as client:
        # 創建訓練任務
        print("4.1 創建訓練任務...")
        job = client.tunings.create(
            model="meta-llama/Llama-2-7b-hf",
            training_file="data/train.jsonl",
            hyperparameters={"epochs": 3}
        )
        print(f"   ✓ 任務已創建: {job.id}")
        print()

        # 流式輸出訓練進度
        print("4.2 流式輸出訓練進度:")
        print("-" * 70)

        try:
            for updated_job in client.tunings.stream_logs(job.id, poll_interval=2, timeout=60):
                if updated_job.metrics:
                    m = updated_job.metrics
                    print(f"Epoch {m.current_epoch}/{m.total_epochs} | "
                          f"Step {m.current_step}/{m.total_steps} | "
                          f"Loss: {m.train_loss:.4f} | "
                          f"LR: {m.learning_rate:.6f} | "
                          f"Progress: {m.progress * 100:.1f}%")

                if updated_job.is_terminal:
                    break

        except TimeoutError:
            print("\n⚠ 超時，但任務仍在後台執行")

        print("-" * 70)
        print(f"\n✓ 最終狀態: {updated_job.status}")
        print()


def example_5_batch_operations():
    """範例 5：批量操作"""
    print("=" * 70)
    print("範例 5：批量創建與管理任務")
    print("=" * 70)
    print()

    with TuningClient(api_key="test-key-001") as client:
        # 批量創建任務
        print("5.1 批量創建 3 個訓練任務...")
        jobs = []
        for i in range(3):
            job = client.tunings.create(
                model=f"test-model-{i+1}",
                training_file=f"data/train_{i+1}.jsonl",
                hyperparameters={"epochs": 2},
                suffix=f"batch-{i+1}"
            )
            jobs.append(job)
            print(f"   ✓ 任務 {i+1}: {job.id}")
        print()

        # 等待一段時間
        print("5.2 等待任務執行...")
        time.sleep(5)
        print()

        # 批量查詢狀態
        print("5.3 查詢所有任務狀態...")
        for i, job in enumerate(jobs):
            updated_job = client.tunings.get(job.id)
            progress = f"{updated_job.metrics.progress * 100:.1f}%" if updated_job.metrics else "N/A"
            print(f"   任務 {i+1}: {updated_job.status} | 進度: {progress}")
        print()

        # 批量取消
        print("5.4 批量取消任務...")
        for i, job in enumerate(jobs):
            try:
                client.tunings.cancel(job.id)
                print(f"   ✓ 已取消任務 {i+1}")
            except TuningAPIError as e:
                print(f"   ⚠ 任務 {i+1} 無法取消: {e}")
        print()


def example_6_custom_progress():
    """範例 6：自定義進度顯示"""
    print("=" * 70)
    print("範例 6：自定義進度顯示")
    print("=" * 70)
    print()

    class ProgressBar:
        """簡單的進度條"""
        def __init__(self, total_width=50):
            self.total_width = total_width
            self.last_progress = 0

        def update(self, job):
            if not job.metrics:
                return

            m = job.metrics
            progress = m.progress

            # 只在進度變化時更新
            if abs(progress - self.last_progress) < 0.01:
                return

            self.last_progress = progress

            # 繪製進度條
            filled = int(progress * self.total_width)
            bar = "█" * filled + "░" * (self.total_width - filled)

            # 計算剩餘時間
            if m.estimated_finish_time:
                remaining = m.estimated_finish_time - int(time.time())
                eta = f"ETA: {remaining}s"
            else:
                eta = "ETA: --"

            print(f"\r[{bar}] {progress * 100:5.1f}% | "
                  f"Epoch {m.current_epoch}/{m.total_epochs} | "
                  f"Loss: {m.train_loss:6.4f} | "
                  f"{eta}",
                  end="", flush=True)

    with TuningClient(api_key="test-key-001") as client:
        job = client.tunings.create(
            model="meta-llama/Llama-2-7b-hf",
            training_file="data/train.jsonl",
            hyperparameters={"epochs": 3}
        )

        progress_bar = ProgressBar()
        final_job = client.tunings.wait(
            job.id,
            poll_interval=1,
            on_progress=progress_bar.update
        )

        print()  # 換行
        print(f"\n✓ 訓練完成: {final_job.fine_tuned_model}")
        print()


def main():
    """運行所有範例"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "SDK 進階使用範例" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    examples = [
        ("錯誤處理", example_1_error_handling),
        ("Context Manager", example_2_context_manager),
        ("取消任務", example_3_cancel_job),
        ("流式日誌", example_4_stream_logs),
        ("批量操作", example_5_batch_operations),
        ("自定義進度條", example_6_custom_progress),
    ]

    print("請選擇要執行的範例：")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("0. 執行所有範例")
    print()

    try:
        choice = input("請輸入選項 (0-6): ").strip()
        print()

        if choice == "0":
            for name, func in examples:
                try:
                    func()
                    time.sleep(2)  # 範例之間間隔
                except Exception as e:
                    print(f"✗ 範例執行失敗: {e}\n")
        elif choice in [str(i) for i in range(1, len(examples) + 1)]:
            idx = int(choice) - 1
            examples[idx][1]()
        else:
            print("無效的選項")
            return

        print("=" * 70)
        print("所有範例執行完成！")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n中斷執行")
        sys.exit(0)


if __name__ == "__main__":
    main()
