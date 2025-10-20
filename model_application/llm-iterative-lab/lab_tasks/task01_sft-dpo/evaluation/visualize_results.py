"""
可視化評估結果
生成柱狀圖、雷達圖和案例對比分析
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 設定中文字體
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """評估結果可視化工具"""

    def __init__(self, results_dir: Path):
        """
        初始化可視化工具

        Args:
            results_dir: 評估結果目錄
        """
        self.results_dir = results_dir
        self.metrics_dir = results_dir / "metrics"
        self.responses_dir = results_dir / "responses"
        self.charts_dir = results_dir / "charts"

        # 確保圖表目錄存在
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def load_comparison_data(self) -> Dict[str, Any]:
        """載入比較數據"""
        comparison_file = self.metrics_dir / "comparison_summary.json"

        if not comparison_file.exists():
            raise FileNotFoundError(f"找不到比較結果文件: {comparison_file}")

        with open(comparison_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def plot_bar_chart(self, comparison_data: Dict[str, Any]):
        """繪製柱狀圖比較"""
        stages = comparison_data["stages"]
        metrics_names = [
            "avg_keyword_coverage",
            "avg_format_correctness",
            "avg_length_appropriateness",
            "avg_structure_score",
            "avg_total_score"
        ]

        metrics_labels = {
            "avg_keyword_coverage": "關鍵詞覆蓋率",
            "avg_format_correctness": "格式正確性",
            "avg_length_appropriateness": "長度適當性",
            "avg_structure_score": "結構分數",
            "avg_total_score": "總分"
        }

        # 準備數據
        data = {metric: [] for metric in metrics_names}

        for stage in stages:
            stage_metrics = comparison_data["comparison"][stage]
            for metric in metrics_names:
                data[metric].append(stage_metrics.get(metric, 0))

        # 創建子圖
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('三階段模型評估指標對比', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        x = np.arange(len(stages))
        width = 0.6

        colors = ['#3498db', '#2ecc71', '#e74c3c']

        for idx, metric in enumerate(metrics_names):
            ax = axes[idx]

            bars = ax.bar(x, data[metric], width, color=[colors[i] for i in range(len(stages))])

            # 添加數值標籤
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

            ax.set_ylabel('分數', fontsize=10)
            ax.set_title(metrics_labels[metric], fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([s.upper() for s in stages])
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 刪除多餘的子圖
        fig.delaxes(axes[-1])

        plt.tight_layout()
        output_file = self.charts_dir / "metrics_comparison_bar.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 柱狀圖已保存: {output_file}")
        plt.close()

    def plot_radar_chart(self, comparison_data: Dict[str, Any]):
        """繪製雷達圖"""
        stages = comparison_data["stages"]

        # 選擇展示的指標
        metrics_names = [
            "avg_keyword_coverage",
            "avg_format_correctness",
            "avg_length_appropriateness",
            "avg_structure_score",
            "avg_total_score"
        ]

        metrics_labels = [
            "關鍵詞\n覆蓋率",
            "格式\n正確性",
            "長度\n適當性",
            "結構\n分數",
            "總分"
        ]

        # 準備數據
        data_by_stage = {}
        for stage in stages:
            values = []
            stage_metrics = comparison_data["comparison"][stage]
            for metric in metrics_names:
                values.append(stage_metrics.get(metric, 0))
            data_by_stage[stage] = values

        # 創建雷達圖
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # 閉合圖形

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = ['#3498db', '#2ecc71', '#e74c3c']
        line_styles = ['-', '--', '-.']

        for idx, stage in enumerate(stages):
            values = data_by_stage[stage]
            values += values[:1]  # 閉合數據

            ax.plot(angles, values, line_styles[idx], linewidth=2,
                   label=stage.upper(), color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        plt.title('三階段模型能力雷達圖', fontsize=16, fontweight='bold', pad=20)

        output_file = self.charts_dir / "metrics_comparison_radar.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 雷達圖已保存: {output_file}")
        plt.close()

    def plot_improvement_curve(self, comparison_data: Dict[str, Any]):
        """繪製提升曲線"""
        stages = comparison_data["stages"]

        # 獲取總分
        total_scores = []
        for stage in stages:
            score = comparison_data["comparison"][stage].get("avg_total_score", 0)
            total_scores.append(score)

        # 創建折線圖
        fig, ax = plt.subplots(figsize=(10, 6))

        x = list(range(len(stages)))
        ax.plot(x, total_scores, marker='o', linewidth=2.5,
               markersize=10, color='#2ecc71')

        # 添加數值標籤
        for i, score in enumerate(total_scores):
            ax.text(i, score + 0.02, f'{score:.3f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 計算提升幅度
        if len(total_scores) >= 2:
            for i in range(1, len(total_scores)):
                improvement = ((total_scores[i] - total_scores[i-1]) / total_scores[i-1] * 100)
                mid_x = (x[i-1] + x[i]) / 2
                mid_y = (total_scores[i-1] + total_scores[i]) / 2
                ax.text(mid_x, mid_y - 0.05, f'+{improvement:.1f}%',
                       ha='center', fontsize=10, color='#e74c3c',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('訓練階段', fontsize=12, fontweight='bold')
        ax.set_ylabel('平均總分', fontsize=12, fontweight='bold')
        ax.set_title('模型能力提升曲線', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in stages], fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, linestyle='--')

        output_file = self.charts_dir / "improvement_curve.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 提升曲線已保存: {output_file}")
        plt.close()

    def generate_case_study(self, num_cases: int = 5):
        """
        生成案例對比分析

        Args:
            num_cases: 生成的案例數量
        """
        print("\n生成案例對比分析...")

        # 載入所有階段的回答
        responses_by_stage = {}

        for stage_file in self.responses_dir.glob("*_responses.json"):
            stage = stage_file.stem.replace("_responses", "")
            with open(stage_file, 'r', encoding='utf-8') as f:
                responses_by_stage[stage] = json.load(f)

        if len(responses_by_stage) < 2:
            print("  ⚠ 需要至少 2 個階段的回答才能生成案例對比")
            return

        # 選擇代表性案例（基於總分差異）
        stages = sorted(responses_by_stage.keys())
        first_stage = stages[0]
        last_stage = stages[-1]

        first_responses = {r["question_id"]: r for r in responses_by_stage[first_stage]}
        last_responses = {r["question_id"]: r for r in responses_by_stage[last_stage]}

        # 計算每個問題的分數差異
        score_diffs = []
        for q_id in first_responses:
            if q_id in last_responses:
                first_score = first_responses[q_id]["metrics"]["total_score"]
                last_score = last_responses[q_id]["metrics"]["total_score"]
                diff = last_score - first_score
                score_diffs.append((q_id, diff, first_score, last_score))

        # 選擇改進最大的案例
        score_diffs.sort(key=lambda x: x[1], reverse=True)
        selected_cases = score_diffs[:num_cases]

        # 生成 Markdown 報告
        report_lines = [
            "# 模型訓練前後案例對比分析\n",
            f"比較階段: **{first_stage.upper()}** vs **{last_stage.upper()}**\n",
            f"選取改進最明顯的 {num_cases} 個案例進行對比分析。\n",
            "---\n"
        ]

        for idx, (q_id, diff, first_score, last_score) in enumerate(selected_cases, 1):
            first_resp = first_responses[q_id]
            last_resp = last_responses[q_id]

            report_lines.append(f"\n## 案例 {idx}: {q_id}\n")
            report_lines.append(f"**問題類別**: {first_resp['category']}\n")
            report_lines.append(f"**分數提升**: {first_score:.3f} → {last_score:.3f} (+{diff:.3f})\n")
            report_lines.append(f"\n### 問題\n")
            report_lines.append(f"```\n{first_resp['prompt']}\n```\n")

            report_lines.append(f"\n### {first_stage.upper()} 回答\n")
            report_lines.append(f"```\n{first_resp['response']}\n```\n")
            report_lines.append(f"**評分**: {first_score:.3f}\n")

            report_lines.append(f"\n### {last_stage.upper()} 回答\n")
            report_lines.append(f"```\n{last_resp['response']}\n```\n")
            report_lines.append(f"**評分**: {last_score:.3f}\n")

            report_lines.append(f"\n### 改進分析\n")

            # 分析具體改進點
            improvements = []
            for metric in ["keyword_coverage", "format_correctness", "structure_score"]:
                first_val = first_resp["metrics"][metric]
                last_val = last_resp["metrics"][metric]
                if last_val > first_val:
                    metric_names = {
                        "keyword_coverage": "關鍵詞覆蓋率",
                        "format_correctness": "格式正確性",
                        "structure_score": "結構分數"
                    }
                    improvements.append(
                        f"- **{metric_names[metric]}**: {first_val:.3f} → {last_val:.3f}"
                    )

            if improvements:
                report_lines.extend(improvements)
            else:
                report_lines.append("- 整體品質提升\n")

            report_lines.append("\n---\n")

        # 保存報告
        output_file = self.charts_dir.parent / "case_study.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)

        print(f"✓ 案例對比分析已保存: {output_file}")

    def generate_summary_report(self, comparison_data: Dict[str, Any]):
        """生成總結報告"""
        stages = comparison_data["stages"]

        report_lines = [
            "# 三階段模型評估總結報告\n",
            f"\n**評估時間**: {comparison_data['timestamp']}\n",
            f"**評估階段**: {', '.join([s.upper() for s in stages])}\n",
            "\n---\n",
            "\n## 整體表現\n",
            "\n| 階段 | 關鍵詞覆蓋率 | 格式正確性 | 長度適當性 | 結構分數 | 總分 |\n",
            "|------|------------|-----------|-----------|---------|-----|\n"
        ]

        for stage in stages:
            metrics = comparison_data["comparison"][stage]
            report_lines.append(
                f"| {stage.upper()} | "
                f"{metrics.get('avg_keyword_coverage', 0):.3f} | "
                f"{metrics.get('avg_format_correctness', 0):.3f} | "
                f"{metrics.get('avg_length_appropriateness', 0):.3f} | "
                f"{metrics.get('avg_structure_score', 0):.3f} | "
                f"{metrics.get('avg_total_score', 0):.3f} |\n"
            )

        # 計算改進幅度
        if len(stages) >= 2:
            report_lines.append("\n## 改進幅度\n\n")

            for i in range(1, len(stages)):
                prev_stage = stages[i-1]
                curr_stage = stages[i]

                prev_score = comparison_data["comparison"][prev_stage]["avg_total_score"]
                curr_score = comparison_data["comparison"][curr_stage]["avg_total_score"]

                improvement = ((curr_score - prev_score) / prev_score * 100)

                report_lines.append(
                    f"- **{prev_stage.upper()} → {curr_stage.upper()}**: "
                    f"{prev_score:.3f} → {curr_score:.3f} "
                    f"({'+' if improvement >= 0 else ''}{improvement:.2f}%)\n"
                )

        report_lines.append("\n## 可視化結果\n\n")
        report_lines.append("- [柱狀圖](charts/metrics_comparison_bar.png)\n")
        report_lines.append("- [雷達圖](charts/metrics_comparison_radar.png)\n")
        report_lines.append("- [提升曲線](charts/improvement_curve.png)\n")
        report_lines.append("- [案例對比分析](case_study.md)\n")

        # 保存報告
        output_file = self.results_dir / "evaluation_summary.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)

        print(f"✓ 總結報告已保存: {output_file}")

    def visualize_all(self):
        """生成所有可視化結果"""
        print("\n開始生成可視化結果...")

        # 載入數據
        comparison_data = self.load_comparison_data()

        # 生成圖表
        self.plot_bar_chart(comparison_data)
        self.plot_radar_chart(comparison_data)
        self.plot_improvement_curve(comparison_data)

        # 生成案例分析
        self.generate_case_study(num_cases=5)

        # 生成總結報告
        self.generate_summary_report(comparison_data)

        print("\n✓ 所有可視化結果已生成完成!")


def main():
    parser = argparse.ArgumentParser(description="可視化評估結果")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="evaluation/results",
        help="評估結果目錄"
    )

    parser.add_argument(
        "--num-cases",
        type=int,
        default=5,
        help="案例對比數量"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"✗ 找不到結果目錄: {results_dir}")
        print("請先執行 evaluate_models.py 生成評估結果")
        return

    visualizer = ResultVisualizer(results_dir)
    visualizer.visualize_all()

    print(f"\n查看結果:")
    print(f"  - 圖表: {results_dir / 'charts'}")
    print(f"  - 總結報告: {results_dir / 'evaluation_summary.md'}")
    print(f"  - 案例分析: {results_dir / 'case_study.md'}")


if __name__ == "__main__":
    main()
