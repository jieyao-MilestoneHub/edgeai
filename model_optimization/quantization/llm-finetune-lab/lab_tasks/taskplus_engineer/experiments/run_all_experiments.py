"""
Task 03: 實驗自動化 - 批次運行所有實驗配置

功能：
1. 自動運行 6 個實驗配置
2. 記錄每個實驗的結果
3. 監控訓練進度
4. 保存實驗摘要

實驗清單：
- exp1: rank=8, Attention only
- exp2: rank=8, Attention + MLP
- exp3: rank=16, Attention only
- exp4: rank=16, Attention + MLP  (基準)
- exp5: rank=32, Attention only
- exp6: rank=32, Attention + MLP

作者: Joel
"""

import os
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime


def run_experiment(exp_config: str, exp_name: str):
    """
    運行單個實驗

    Args:
        exp_config: 實驗配置檔案路徑
        exp_name: 實驗名稱
    """
    print("\n" + "="*80)
    print(f"🚀 Starting Experiment: {exp_name}")
    print("="*80)
    print(f"Config: {exp_config}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # 構建命令
    cmd = [
        "python",
        "../model_training/train_qwen_instruct.py",
        "--config", "../config.yaml",
        "--exp_config", exp_config,
    ]

    print(f"\nCommand: {' '.join(cmd)}\n")

    # 運行訓練
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # 直接顯示輸出
            text=True
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✅ Experiment completed successfully!")
        print(f"   Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

        return {
            "name": exp_name,
            "config": exp_config,
            "status": "success",
            "duration": duration,
            "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n❌ Experiment failed!")
        print(f"   Error: {e}")

        return {
            "name": exp_name,
            "config": exp_config,
            "status": "failed",
            "duration": duration,
            "error": str(e),
            "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def main():
    """主函數：運行所有實驗"""
    print("\n" + "="*80)
    print("🧪 Task 03: Running All Experiments")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 實驗配置清單
    experiments = [
        ("../configs/exp1_r8_attention.yaml", "Exp1: rank=8, Attention"),
        ("../configs/exp2_r8_full.yaml", "Exp2: rank=8, Full"),
        ("../configs/exp3_r16_attention.yaml", "Exp3: rank=16, Attention"),
        ("../configs/exp4_r16_full.yaml", "Exp4: rank=16, Full (Baseline)"),
        ("../configs/exp5_r32_attention.yaml", "Exp5: rank=32, Attention"),
        ("../configs/exp6_r32_full.yaml", "Exp6: rank=32, Full"),
    ]

    print(f"\nTotal experiments: {len(experiments)}\n")

    # 確認
    response = input("Start running all experiments? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # 運行所有實驗
    results = []
    overall_start = time.time()

    for exp_config, exp_name in experiments:
        result = run_experiment(exp_config, exp_name)
        results.append(result)

        # 實驗間休息（清理GPU記憶體）
        print("\n⏸️  Waiting 30 seconds before next experiment...")
        time.sleep(30)

    overall_end = time.time()
    overall_duration = overall_end - overall_start

    # 保存結果摘要
    print("\n" + "="*80)
    print("📊 Saving Experiment Summary")
    print("="*80)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    summary_path = results_dir / "experiments_summary.json"

    summary = {
        "total_experiments": len(experiments),
        "total_duration": overall_duration,
        "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "results": results
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Summary saved to: {summary_path}")

    # 顯示統計
    print("\n" + "="*80)
    print("📈 Experiment Statistics")
    print("="*80)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')

    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nTotal duration: {overall_duration:.2f} seconds ({overall_duration/3600:.2f} hours)")

    print("\n" + "-"*80)
    print("Individual results:")
    for r in results:
        status_icon = "✅" if r['status'] == 'success' else "❌"
        print(f"\n{status_icon} {r['name']}")
        print(f"   Duration: {r['duration']:.2f}s ({r['duration']/60:.2f} min)")
        print(f"   Status: {r['status']}")

    # 下一步提示
    print("\n" + "="*80)
    print("✅ All experiments completed!")
    print("="*80)
    print("\n💡 Next steps:")
    print("   1. Compare results: python compare_results.py")
    print("   2. Evaluate models: cd ../4_evaluation && python evaluate_model.py")
    print("   3. Generate report: cd ../5_report_generation && python generate_report.py")


if __name__ == "__main__":
    main()
