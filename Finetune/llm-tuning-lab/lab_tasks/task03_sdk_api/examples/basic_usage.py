"""
基本使用範例
============

這個範例展示了 SDK 的基本使用方式。

執行前請確保：
1. FastAPI 服務已啟動（python tuning_service/app.py）
2. 已安裝 SDK 依賴（pip install requests）
"""

import sys
import os

from tuning_sdk import TuningClient, JobStatus


def main():
    print("=" * 70)
    print("SDK 基本使用範例")
    print("=" * 70)
    print()

    # 1. 初始化客戶端
    print("1. 初始化客戶端...")
    client = TuningClient(
        api_key="test-key-001",
        base_url="http://localhost:8000"
    )
    print("   ✓ 客戶端初始化完成")
    print()

    try:
        # 2. 健康檢查
        print("2. 檢查服務健康狀態...")
        health = client.health_check()
        print(f"   ✓ 服務狀態: {health['status']}")
        print(f"   ✓ 服務版本: {health['version']}")
        print(f"   ✓ 運行時間: {health['uptime']:.0f} 秒")
        print()

        # 3. 創建訓練任務
        print("3. 創建訓練任務...")
        job = client.tunings.create(
            model="meta-llama/Llama-2-7b-hf",
            training_file="data/train.jsonl",
            validation_file="data/val.jsonl",
            hyperparameters={
                "epochs": 3,
                "learning_rate": 2e-4,
                "batch_size": 4
            },
            suffix="basic-example"
        )
        print(f"   ✓ 任務已創建")
        print(f"   - 任務 ID: {job.id}")
        print(f"   - 模型: {job.model}")
        print(f"   - 狀態: {job.status}")
        print()

        # 4. 查詢任務狀態
        print("4. 查詢任務狀態...")
        job = client.tunings.get(job.id)
        print(f"   ✓ 當前狀態: {job.status}")
        if job.metrics:
            print(f"   - 進度: {job.metrics.progress * 100:.1f}%")
            print(f"   - 訓練損失: {job.metrics.train_loss:.4f}")
        print()

        # 5. 等待任務完成
        print("5. 等待任務完成...")
        print("   (這可能需要 30 秒左右)")
        print()

        def show_progress(j):
            """進度回調函數"""
            if j.metrics:
                bar_length = 30
                filled = int(j.metrics.progress * bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                print(f"\r   [{bar}] {j.metrics.progress * 100:.1f}% | "
                      f"Epoch {j.metrics.current_epoch}/{j.metrics.total_epochs} | "
                      f"Loss: {j.metrics.train_loss:.4f}",
                      end="", flush=True)

        final_job = client.tunings.wait(
            job.id,
            poll_interval=2,
            on_progress=show_progress
        )
        print()  # 換行
        print()

        # 6. 顯示結果
        print("6. 訓練完成！")
        print(f"   ✓ 最終狀態: {final_job.status}")
        if final_job.is_successful:
            print(f"   ✓ 微調模型: {final_job.fine_tuned_model}")
            print(f"   ✓ 訓練時長: {final_job.duration} 秒")
            print(f"   ✓ 結果檔案:")
            for file in final_job.result_files:
                print(f"      - {file}")
        elif final_job.is_failed:
            print(f"   ✗ 錯誤訊息: {final_job.error}")
        print()

        # 7. 列出所有任務
        print("7. 列出最近的任務...")
        job_list = client.tunings.list(limit=5)
        print(f"   ✓ 共有 {job_list.total} 個任務")
        for j in job_list.data:
            status_icon = "✓" if j.is_successful else "✗" if j.is_failed else "⋯"
            print(f"   {status_icon} {j.id}: {j.status} ({j.model})")
        print()

    finally:
        # 關閉客戶端
        client.close()

    print("=" * 70)
    print("範例執行完成！")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中斷執行")
        sys.exit(0)
    except Exception as e:
        print(f"\n錯誤: {e}")
        sys.exit(1)
