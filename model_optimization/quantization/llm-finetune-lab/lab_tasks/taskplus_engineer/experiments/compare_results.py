"""
Task 03: 實驗結果比較

功能：
1. 載入所有實驗的訓練日誌
2. 比較不同配置的效果
3. 生成對比圖表
4. 輸出參數效率對比表

作者: Joel
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_experiment_results():
    """載入所有實驗結果"""
    print("\n" + "="*60)
    print("📂 Loading Experiment Results...")
    print("="*60)

    experiments = [
        {"name": "Exp1_r8_att", "path": "../output_task03_exp1_r8_attention", "rank": 8, "modules": "Attention"},
        {"name": "Exp2_r8_full", "path": "../output_task03_exp2_r8_full", "rank": 8, "modules": "Full"},
        {"name": "Exp3_r16_att", "path": "../output_task03_exp3_r16_attention", "rank": 16, "modules": "Attention"},
        {"name": "Exp4_r16_full", "path": "../output_task03_exp4_r16_full", "rank": 16, "modules": "Full"},
        {"name": "Exp5_r32_att", "path": "../output_task03_exp5_r32_attention", "rank": 32, "modules": "Attention"},
        {"name": "Exp6_r32_full", "path": "../output_task03_exp6_r32_full", "rank": 32, "modules": "Full"},
    ]

    results = []

    for exp in experiments:
        exp_path = Path(exp['path'])

        if not exp_path.exists():
            print(f"⚠️  {exp['name']}: Not found")
            continue

        # 讀取 adapter_config.json 獲取參數信息
        config_path = exp_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                adapter_config = json.load(f)

            exp['lora_rank'] = adapter_config.get('r', exp['rank'])
            exp['lora_alpha'] = adapter_config.get('lora_alpha', exp['rank'] * 2)

        # 讀取訓練日誌
        log_path = exp_path / "training_log.txt"
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_content = f.read()

            # 解析日誌（簡化版，實際可能需要更複雜的解析）
            if "Final Train Loss:" in log_content:
                train_loss = float(log_content.split("Final Train Loss:")[1].split()[0])
                exp['train_loss'] = train_loss

            if "Final Eval Loss:" in log_content:
                eval_loss = float(log_content.split("Final Eval Loss:")[1].split()[0])
                exp['eval_loss'] = eval_loss

        results.append(exp)
        print(f"✅ {exp['name']}: Loaded")

    print(f"\nTotal results loaded: {len(results)}")

    return results


def create_comparison_table(results):
    """創建對比表"""
    print("\n" + "="*60)
    print("📊 Creating Comparison Table...")
    print("="*60)

    df = pd.DataFrame(results)

    # 選擇要顯示的列
    columns = ['name', 'rank', 'modules', 'train_loss', 'eval_loss']
    df_display = df[columns].copy()

    # 格式化
    df_display.columns = ['Experiment', 'Rank', 'Modules', 'Train Loss', 'Eval Loss']

    print("\n" + df_display.to_string(index=False))

    # 保存為 CSV
    output_path = Path("results/comparison_table.csv")
    output_path.parent.mkdir(exist_ok=True)
    df_display.to_csv(output_path, index=False)

    print(f"\n✅ Table saved to: {output_path}")

    return df


def plot_comparison_charts(df):
    """繪製對比圖表"""
    print("\n" + "="*60)
    print("📈 Creating Comparison Charts...")
    print("="*60)

    # 設定樣式
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss 對比（按 rank）
    ax1 = axes[0, 0]
    df_plot = df[['rank', 'modules', 'eval_loss']].copy()
    for modules in df_plot['modules'].unique():
        data = df_plot[df_plot['modules'] == modules]
        ax1.plot(data['rank'], data['eval_loss'], marker='o', label=modules, linewidth=2, markersize=8)

    ax1.set_xlabel('LoRA Rank', fontsize=12)
    ax1.set_ylabel('Eval Loss', fontsize=12)
    ax1.set_title('Evaluation Loss vs LoRA Rank', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Loss 對比（條形圖）
    ax2 = axes[0, 1]
    df_plot = df[['name', 'eval_loss']].copy()
    colors = ['#1f77b4' if 'att' in name else '#ff7f0e' for name in df_plot['name']]
    ax2.barh(df_plot['name'], df_plot['eval_loss'], color=colors)
    ax2.set_xlabel('Eval Loss', fontsize=12)
    ax2.set_ylabel('Experiment', fontsize=12)
    ax2.set_title('Evaluation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Train vs Eval Loss
    ax3 = axes[1, 0]
    x = range(len(df))
    width = 0.35
    ax3.bar([i - width/2 for i in x], df['train_loss'], width, label='Train Loss', alpha=0.8)
    ax3.bar([i + width/2 for i in x], df['eval_loss'], width, label='Eval Loss', alpha=0.8)
    ax3.set_xlabel('Experiment', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Train vs Eval Loss', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['name'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 參數效率（eval_loss vs rank）
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['rank'], df['eval_loss'], s=200, alpha=0.6, c=range(len(df)), cmap='viridis')
    for i, name in enumerate(df['name']):
        ax4.annotate(name, (df.iloc[i]['rank'], df.iloc[i]['eval_loss']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    ax4.set_xlabel('LoRA Rank (Parameter Count)', fontsize=12)
    ax4.set_ylabel('Eval Loss (Performance)', fontsize=12)
    ax4.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    output_path = Path("results/comparison_charts.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"✅ Charts saved to: {output_path}")

    plt.close()


def find_best_configuration(df):
    """找出最佳配置"""
    print("\n" + "="*60)
    print("🏆 Finding Best Configuration...")
    print("="*60)

    # 按 eval_loss 排序
    best = df.loc[df['eval_loss'].idxmin()]

    print(f"\n🥇 Best Configuration:")
    print(f"   Experiment: {best['name']}")
    print(f"   LoRA Rank: {best['rank']}")
    print(f"   Target Modules: {best['modules']}")
    print(f"   Eval Loss: {best['eval_loss']:.4f}")

    # 找出最高效配置（考慮參數量與性能平衡）
    # 簡單策略：Loss 最低的 Attention-only 配置（參數較少）
    attention_only = df[df['modules'] == 'Attention']
    if not attention_only.empty:
        efficient = attention_only.loc[attention_only['eval_loss'].idxmin()]

        print(f"\n⚡ Most Efficient Configuration (Attention-only):")
        print(f"   Experiment: {efficient['name']}")
        print(f"   LoRA Rank: {efficient['rank']}")
        print(f"   Eval Loss: {efficient['eval_loss']:.4f}")


def main():
    """主函數"""
    print("\n" + "="*60)
    print("🔍 Task 03: Comparing Experiment Results")
    print("="*60)

    # 載入結果
    results = load_experiment_results()

    if not results:
        print("\n❌ No experiment results found!")
        print("   Please run experiments first:")
        print("   python run_all_experiments.py")
        return

    # 創建對比表
    df = create_comparison_table(results)

    # 繪製圖表
    if 'train_loss' in df.columns and 'eval_loss' in df.columns:
        plot_comparison_charts(df)

        # 找出最佳配置
        find_best_configuration(df)
    else:
        print("\n⚠️  Insufficient data for plotting")

    print("\n" + "="*60)
    print("✅ Comparison completed!")
    print("="*60)


if __name__ == "__main__":
    main()
