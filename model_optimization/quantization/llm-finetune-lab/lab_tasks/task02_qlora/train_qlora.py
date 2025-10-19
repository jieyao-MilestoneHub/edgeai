"""
QLoRA Training Script - Qwen2.5-3B Language Model Fine-tuning

完整的 QLoRA 訓練流程，支援：
- 從 config.yaml 讀取配置
- 自動套用 4-bit NF4 量化與 LoRA
- Wikitext-2 語言模型訓練（可替換為自訂資料集）
- 完整的訓練與評估循環
- Loss/Perplexity 評估指標
- 訓練進度視覺化
- LoRA adapter 儲存與載入
- 記憶體使用監控

記憶體需求：
- GTX 4060 (8GB): ✅ 可訓練（batch_size=1, gradient_accumulation=16）
- RTX 3090 (24GB): ✅ 可訓練（batch_size=4, gradient_accumulation=4）
- RTX 4090 (24GB): ✅ 可訓練（batch_size=4, gradient_accumulation=4）

作者: Joel
授權: MIT
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# 從同目錄導入我們實作的 QLoRA 模組
from quantization_utils import (
    load_model_and_tokenizer,
    create_lora_config,
    prepare_model_for_training,
    print_memory_stats,
)


# ====================================================================================
# 1) 載入配置
# ====================================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    載入 YAML 配置檔

    Args:
        config_path: 配置檔案路徑（默認 config.yaml）

    Returns:
        配置字典

    範例：
        >>> config = load_config("config.yaml")
        >>> print(config['model']['name'])
        Qwen/Qwen2.5-3B-Instruct
    """
    print(f"📋 Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded successfully!")
    return config


# ====================================================================================
# 2) 準備資料集
# ====================================================================================

def prepare_dataset(config: Dict[str, Any], tokenizer):
    """
    準備並預處理資料集

    Args:
        config: 配置字典
        tokenizer: Tokenizer 物件

    Returns:
        tokenized_datasets: Tokenize 後的資料集

    處理流程：
        1. 載入資料集（Wikitext-2）
        2. Tokenization（轉換為 input_ids）
        3. 設定 labels（語言模型任務：labels = input_ids）
        4. 移除未使用的欄位

    注意：
        - Wikitext-2 是語言模型基準資料集
        - 可替換為自訂資料集（修改 dataset_name 和 tokenize_function）
    """
    print(f"\n📚 Loading dataset: {config['data']['dataset_name']}/{config['data']['dataset_config']}")

    # 載入資料集
    dataset = load_dataset(
        config['data']['dataset_name'],
        config['data']['dataset_config']
    )

    print(f"   Train samples: {len(dataset['train'])}")
    print(f"   Validation samples: {len(dataset['validation'])}")

    # Tokenization 函數
    def tokenize_function(examples):
        """
        將文本轉換為 token IDs

        語言模型任務：
            - input_ids: 模型輸入
            - labels: 預測目標（語言模型中 labels = input_ids）
        """
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config['data']['max_length'],
            padding="max_length",  # 固定長度（訓練效率）
        )
        # 語言模型任務：labels = input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    # 批次處理資料集
    print(f"🔄 Tokenizing dataset (max_length={config['data']['max_length']})...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config['data']['preprocessing_num_proc'],
        remove_columns=dataset["train"].column_names,  # 移除原始文本欄位
        desc="Tokenizing",
    )

    print(f"✅ Dataset tokenized successfully!")

    return tokenized_datasets


# ====================================================================================
# 3) 計算評估指標（Perplexity）
# ====================================================================================

def compute_metrics(eval_pred):
    """
    計算評估指標（用於 Trainer）

    Args:
        eval_pred: (predictions, labels) tuple

    Returns:
        metrics: 包含 perplexity 的字典

    Perplexity (困惑度)：
        - 語言模型的標準評估指標
        - 公式：perplexity = exp(loss)
        - 越低越好（表示模型預測越準確）
        - 例：perplexity=10 表示每步平均在 10 個候選中選擇
    """
    predictions, labels = eval_pred
    # 注意：Trainer 會自動計算 loss，這裡只需計算衍生指標
    # 實際上 compute_metrics 在這裡不會被呼叫，因為我們使用默認的 loss
    # 保留此函數作為範例
    return {}


# ====================================================================================
# 4) 繪製訓練曲線
# ====================================================================================

def plot_training_curves(
    log_history: list,
    save_path: str = "training_curves.png"
):
    """
    繪製訓練曲線（Loss 和 Learning Rate）

    Args:
        log_history: Trainer 的 log history
        save_path: 儲存路徑

    繪製內容：
        - 左圖：Training Loss 和 Validation Loss
        - 右圖：Learning Rate 變化

    注意：
        - Trainer 會自動記錄 log_history
        - 可從 trainer.state.log_history 取得
    """
    print(f"\n📊 Plotting training curves...")

    # 提取資料
    train_loss = []
    eval_loss = []
    learning_rates = []
    steps = []

    for entry in log_history:
        if 'loss' in entry:  # Training loss
            train_loss.append((entry['step'], entry['loss']))
        if 'eval_loss' in entry:  # Validation loss
            eval_loss.append((entry['step'], entry['eval_loss']))
        if 'learning_rate' in entry:  # Learning rate
            learning_rates.append((entry['step'], entry['learning_rate']))

    # 繪製
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲線
    if train_loss:
        steps_train, losses_train = zip(*train_loss)
        ax1.plot(steps_train, losses_train, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    if eval_loss:
        steps_eval, losses_eval = zip(*eval_loss)
        ax1.plot(steps_eval, losses_eval, 'r-s', label='Eval Loss', linewidth=2, markersize=4)

    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Learning Rate 曲線
    if learning_rates:
        steps_lr, lrs = zip(*learning_rates)
        ax2.plot(steps_lr, lrs, 'g-', label='Learning Rate', linewidth=2)
        ax2.set_xlabel('Steps', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to: {save_path}")


# ====================================================================================
# 5) 主訓練函數
# ====================================================================================

def main(args):
    # ========== 1. 載入配置 ==========
    config_path = args.config if args.config else "config.yaml"
    config = load_config(config_path)

    # 命令列參數覆寫配置
    if args.rank is not None:
        config['lora']['rank'] = args.rank
    if args.alpha is not None:
        config['lora']['alpha'] = args.alpha
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    if args.output_dir is not None:
        config['output']['dir'] = args.output_dir

    # ========== 2. 設定裝置與隨機種子 ==========
    if not torch.cuda.is_available():
        print("❌ CUDA not available. QLoRA requires GPU.")
        print("   Please check your CUDA installation and GPU availability.")
        return

    device = torch.device('cuda')
    print(f"\n🖥️  Device Information:")
    print(f"   Device: {device}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 設定隨機種子（可重現性）
    torch.manual_seed(config['advanced']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['advanced']['seed'])

    # 啟用 TF32 加速（Ampere 架構及以上）
    if config['advanced']['allow_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"   TF32 acceleration: Enabled")

    # ========== 3. 載入模型與 Tokenizer ==========
    print("\n" + "="*60)
    print("🚀 QLoRA Training - Qwen2.5-3B")
    print("="*60)

    model, tokenizer = load_model_and_tokenizer(config['model']['name'])
    print_memory_stats("After loading model")

    # ========== 4. 套用 LoRA ==========
    print(f"\n🔧 Applying LoRA:")
    print(f"   Rank: {config['lora']['rank']}")
    print(f"   Alpha: {config['lora']['alpha']}")
    print(f"   Dropout: {config['lora']['dropout']}")
    print(f"   Target modules: {', '.join(config['lora']['target_modules'])}")

    lora_config = create_lora_config(
        r=config['lora']['rank'],
        alpha=config['lora']['alpha'],
        dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules'],
    )

    model = prepare_model_for_training(
        model,
        lora_config,
        use_gradient_checkpointing=config['training']['gradient_checkpointing']
    )

    print_memory_stats("After applying LoRA")

    # ========== 5. 準備資料集 ==========
    tokenized_datasets = prepare_dataset(config, tokenizer)

    # Data Collator（處理動態 padding 和 labels）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果語言模型（不使用 masked language modeling）
    )

    # ========== 6. 設定訓練參數 ==========
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        # 輸出目錄
        output_dir=str(output_dir),
        overwrite_output_dir=config['output']['overwrite_output_dir'],

        # 訓練參數
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],

        # 學習率
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],

        # 精度設定
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],

        # 優化器
        optim=config['training']['optim'],  # paged_adamw_8bit
        max_grad_norm=config['training']['max_grad_norm'],

        # 日誌與儲存
        logging_dir=config['output']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],

        # 評估
        eval_strategy="steps",
        eval_steps=config['training']['eval_steps'],

        # 其他
        report_to="tensorboard",
        seed=config['advanced']['seed'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        ddp_find_unused_parameters=config['advanced']['ddp_find_unused_parameters'],
    )

    print(f"\n🎓 Training Configuration:")
    print(f"   Epochs: {config['training']['num_epochs']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Warmup ratio: {config['training']['warmup_ratio']}")
    print(f"   Optimizer: {config['training']['optim']}")
    print(f"   Precision: {'BF16' if config['training']['bf16'] else 'FP16' if config['training']['fp16'] else 'FP32'}")
    print(f"   Gradient checkpointing: {config['training']['gradient_checkpointing']}")

    # ========== 7. 建立 Trainer ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets[config['data']['train_split']],
        eval_dataset=tokenized_datasets[config['data']['eval_split']],
        data_collator=data_collator,
    )

    # ========== 8. 開始訓練 ==========
    print("\n" + "="*60)
    print("🚀 Starting Training...")
    print("="*60)

    print_memory_stats("Before training")

    try:
        # 訓練
        train_result = trainer.train()

        print("\n" + "="*60)
        print("✅ Training Completed!")
        print("="*60)

        # 顯示訓練結果
        print(f"\n📊 Training Results:")
        print(f"   Final Train Loss: {train_result.training_loss:.4f}")
        print(f"   Training Time: {train_result.metrics['train_runtime']:.2f} seconds")
        print(f"   Samples/Second: {train_result.metrics['train_samples_per_second']:.2f}")

        # ========== 9. 最終評估 ==========
        print(f"\n📈 Running Final Evaluation...")
        eval_result = trainer.evaluate()
        print(f"   Final Eval Loss: {eval_result['eval_loss']:.4f}")

        # 計算 Perplexity
        perplexity = torch.exp(torch.tensor(eval_result['eval_loss']))
        print(f"   Final Perplexity: {perplexity:.2f}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("   Saving current state...")

    # ========== 10. 儲存結果 ==========
    print("\n" + "="*60)
    print("💾 Saving Results...")
    print("="*60)

    # 10.1 儲存 LoRA adapter
    print(f"📦 Saving LoRA adapter...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ LoRA adapter saved to: {output_dir}")

    # 10.2 繪製訓練曲線
    if config['output']['save_training_curves']:
        plot_training_curves(
            trainer.state.log_history,
            save_path=str(output_dir / "training_curves.png")
        )

    # 10.3 儲存訓練日誌
    if config['output']['save_training_log']:
        log_path = output_dir / "training_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("QLoRA Training Log - Qwen2.5-3B\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: {config['model']['name']}\n")
            f.write(f"Dataset: {config['data']['dataset_name']}/{config['data']['dataset_config']}\n")
            f.write(f"Task: Causal Language Modeling\n\n")

            f.write(f"LoRA Config:\n")
            f.write(f"  Rank: {config['lora']['rank']}\n")
            f.write(f"  Alpha: {config['lora']['alpha']}\n")
            f.write(f"  Dropout: {config['lora']['dropout']}\n")
            f.write(f"  Target modules: {', '.join(config['lora']['target_modules'])}\n\n")

            f.write(f"Training Config:\n")
            f.write(f"  Epochs: {config['training']['num_epochs']}\n")
            f.write(f"  Batch size: {config['training']['batch_size']}\n")
            f.write(f"  Gradient accumulation: {config['training']['gradient_accumulation_steps']}\n")
            f.write(f"  Learning rate: {config['training']['learning_rate']}\n\n")

            if 'train_result' in locals():
                f.write(f"Results:\n")
                f.write(f"  Final Train Loss: {train_result.training_loss:.4f}\n")
                f.write(f"  Final Eval Loss: {eval_result['eval_loss']:.4f}\n")
                f.write(f"  Final Perplexity: {perplexity:.2f}\n")
                f.write(f"  Training Time: {train_result.metrics['train_runtime']:.2f}s\n")

        print(f"✅ Training log saved to: {log_path}")

    print_memory_stats("After training")

    # ========== 11. 完成 ==========
    print("\n" + "="*60)
    print("🎉 All Done!")
    print("="*60)
    print(f"\n📁 Output Directory: {output_dir.absolute()}")
    print(f"\n✨ Files created:")
    print(f"   - adapter_config.json (LoRA 配置)")
    print(f"   - adapter_model.safetensors (LoRA 權重)")
    print(f"   - training_curves.png (訓練曲線)")
    print(f"   - training_log.txt (訓練日誌)")
    print(f"\n💡 Next steps:")
    print(f"   - Run inference: python inference_example.py")
    print(f"   - View TensorBoard: tensorboard --logdir {config['output']['logging_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5-3B with QLoRA on Language Modeling Task"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="LoRA rank (overrides config)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="LoRA alpha (overrides config)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )

    args = parser.parse_args()
    main(args)
