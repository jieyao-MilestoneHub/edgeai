"""
LoRA Basic Training Script - SST-2 Sentiment Classification

完整的 LoRA 訓練流程，支援：
- 從 config.yaml 讀取配置
- 自動套用 LoRA 到 BERT 模型
- GLUE SST-2 情感分類任務
- 完整的訓練與評估循環
- 準確率評估指標
- 訓練進度視覺化
- LoRA 權重儲存與載入

作者: Jiao
授權: MIT
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import evaluate

# 從同目錄導入我們實作的 LoRA 模組
from lora_linear import (
    apply_lora_to_model,
    mark_only_lora_as_trainable,
    count_lora_parameters,
    get_lora_state_dict,
    merge_lora_weights,
)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """載入 YAML 配置檔"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(config: Dict[str, Any], tokenizer):
    """準備並預處理 SST-2 資料集"""
    print(f"Loading dataset: {config['data']['dataset']}/{config['data']['subset']}")

    # 載入 GLUE SST-2 資料集
    dataset = load_dataset(
        config['data']['dataset'],
        config['data']['subset']
    )

    # Tokenization 函數 (SST-2 格式: {sentence, label})
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=config['training']['max_length'],
            return_tensors=None,
        )

    # 對資料集進行 tokenize
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
    )

    # 設定格式為 PyTorch tensor
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    return tokenized_datasets


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> tuple[float, float]:
    """訓練一個 epoch"""
    model.train()
    total_loss = 0.0

    # 使用 evaluate 庫來計算 accuracy
    metric = evaluate.load("accuracy")

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        # 將資料移到裝置
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 前向傳播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits

        # 收集預測結果用於計算準確率
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions.cpu(), references=labels.cpu())

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip
            )

        # 更新參數
        optimizer.step()
        scheduler.step()

        # 記錄
        total_loss += loss.item()

        # 更新進度條
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
        })

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = metric.compute()['accuracy']

    return avg_loss, avg_accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """評估模型"""
    model.eval()
    total_loss = 0.0

    # 使用 evaluate 庫來計算 accuracy
    metric = evaluate.load("accuracy")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            # 收集預測結果用於計算準確率
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions.cpu(), references=labels.cpu())

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = metric.compute()['accuracy']

    return avg_loss, avg_accuracy


def plot_training_curves(
    train_losses: list,
    eval_losses: list,
    train_accuracies: list,
    eval_accuracies: list,
    save_path: str = "training_curves.png"
):
    """繪製訓練曲線（Loss 和 Accuracy）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    # Loss 曲線
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, eval_losses, 'r-s', label='Eval Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy 曲線
    ax2.plot(epochs, train_accuracies, 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, eval_accuracies, 'r-s', label='Eval Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to: {save_path}")


def main(args):
    # ========== 1. 載入配置 ==========
    config_path = args.config if args.config else "config.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # 允許命令列覆寫配置
    if args.rank is not None:
        config['lora']['rank'] = args.rank
    if args.alpha is not None:
        config['lora']['alpha'] = args.alpha
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs

    # ========== 2. 設定裝置 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ========== 3. 載入模型與 Tokenizer ==========
    print(f"\nLoading model: {config['model_name']}")
    print(f"Task: {config['task_type']} with {config['num_labels']} labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=config['num_labels']
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # ========== 4. 套用 LoRA ==========
    print(f"\nApplying LoRA:")
    print(f"  - rank: {config['lora']['rank']}")
    print(f"  - alpha: {config['lora']['alpha']}")
    print(f"  - dropout: {config['lora']['dropout']}")
    print(f"  - target_modules: {config['lora']['target_modules']}")

    apply_lora_to_model(
        model,
        target_modules=config['lora']['target_modules'],
        rank=config['lora']['rank'],
        alpha=config['lora']['alpha'],
        dropout=config['lora']['dropout'],
    )

    # 只訓練 LoRA 參數
    mark_only_lora_as_trainable(model)

    # 顯示參數統計
    param_stats = count_lora_parameters(model)
    print("\n" + "="*60)
    print("📊 Parameter Statistics:")
    print(f"  Total parameters:     {param_stats['total']:>12,}")
    print(f"  Trainable parameters: {param_stats['trainable']:>12,}")
    print(f"  Frozen parameters:    {param_stats['frozen']:>12,}")
    print(f"  Trainable percentage: {param_stats['percentage']:>11.4f}%")
    print("="*60)

    # 將模型移到裝置
    model = model.to(device)

    # ========== 5. 準備資料集 ==========
    tokenized_datasets = prepare_dataset(config, tokenizer)

    # 使用動態 padding（更節省記憶體）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets[config['data']['train_split']],
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        shuffle=True,
    )

    eval_dataloader = DataLoader(
        tokenized_datasets[config['data']['eval_split']],
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        shuffle=False,
    )

    print(f"\n📚 Dataset Info:")
    print(f"  Train samples: {len(tokenized_datasets[config['data']['train_split']])}")
    print(f"  Eval samples:  {len(tokenized_datasets[config['data']['eval_split']])}")
    print(f"  Batch size:    {config['training']['batch_size']}")
    print(f"  Task:          SST-2 Sentiment Classification (Binary)")

    # ========== 6. 設定優化器與 Scheduler ==========
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    num_training_steps = len(train_dataloader) * config['training']['num_epochs']
    num_warmup_steps = int(config['training']['warmup_ratio'] * num_training_steps)

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"\n🎓 Training Config:")
    print(f"  Epochs:          {config['training']['num_epochs']}")
    print(f"  Learning rate:   {config['training']['learning_rate']}")
    print(f"  Weight decay:    {config['training']['weight_decay']}")
    print(f"  Gradient clip:   {config['training']['gradient_clip']}")
    print(f"  Warmup ratio:    {config['training']['warmup_ratio']}")
    print(f"  Warmup steps:    {num_warmup_steps}")
    print(f"  Total steps:     {num_training_steps}")

    # ========== 7. 訓練循環 ==========
    print("\n" + "="*60)
    print("🚀 Starting Training...")
    print("="*60 + "\n")

    train_losses = []
    eval_losses = []
    train_accuracies = []
    eval_accuracies = []
    best_eval_accuracy = 0.0

    # 建立輸出目錄
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['training']['num_epochs']):
        print(f"\n📍 Epoch {epoch + 1}/{config['training']['num_epochs']}")
        print("-" * 60)

        # 訓練
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_clip=config['training']['gradient_clip'],
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 評估
        eval_loss, eval_acc = evaluate(
            model=model,
            dataloader=eval_dataloader,
            device=device,
        )
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_acc)

        # 顯示結果
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Eval Loss:  {eval_loss:.4f} | Eval Acc:  {eval_acc:.4f}")

        # 儲存最佳模型
        if eval_acc > best_eval_accuracy:
            best_eval_accuracy = eval_acc
            best_checkpoint_path = output_dir / "best_lora_model.pt"

            torch.save({
                'epoch': epoch + 1,
                'lora_state_dict': get_lora_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'train_acc': train_acc,
                'eval_acc': eval_acc,
                'config': config,
            }, best_checkpoint_path)

            print(f"  💾 Best model saved! (eval_acc: {eval_acc:.4f})")

    # ========== 8. 儲存最終結果 ==========
    print("\n" + "="*60)
    print("💾 Saving Results...")
    print("="*60)

    # 8.1 儲存最終 LoRA 權重
    final_checkpoint_path = output_dir / "final_lora_model.pt"
    torch.save({
        'epoch': config['training']['num_epochs'],
        'lora_state_dict': get_lora_state_dict(model),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'train_accuracies': train_accuracies,
        'eval_accuracies': eval_accuracies,
    }, final_checkpoint_path)
    print(f"✅ Final LoRA weights saved to: {final_checkpoint_path}")

    # 8.2 儲存只有 LoRA 參數的精簡版本
    lora_only_path = output_dir / "lora_adapter.pt"
    torch.save({
        'lora_state_dict': get_lora_state_dict(model),
        'config': {
            'rank': config['lora']['rank'],
            'alpha': config['lora']['alpha'],
            'target_modules': config['lora']['target_modules'],
            'model_name': config['model_name'],
            'num_labels': config['num_labels'],
        }
    }, lora_only_path)
    print(f"✅ LoRA adapter saved to: {lora_only_path}")

    # 8.3 繪製訓練曲線
    plot_training_curves(
        train_losses,
        eval_losses,
        train_accuracies,
        eval_accuracies,
        save_path=str(output_dir / "training_curves.png")
    )

    # 8.4 儲存訓練日誌
    log_path = output_dir / "training_log.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("LoRA Training Log - SST-2 Sentiment Classification\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Dataset: {config['data']['dataset']}/{config['data']['subset']}\n")
        f.write(f"Task: Binary Sentiment Classification\n\n")
        f.write(f"LoRA Config:\n")
        f.write(f"  rank: {config['lora']['rank']}\n")
        f.write(f"  alpha: {config['lora']['alpha']}\n")
        f.write(f"  dropout: {config['lora']['dropout']}\n\n")
        f.write(f"Parameter Statistics:\n")
        f.write(f"  Total: {param_stats['total']:,}\n")
        f.write(f"  Trainable: {param_stats['trainable']:,}\n")
        f.write(f"  Percentage: {param_stats['percentage']:.4f}%\n\n")
        f.write(f"Training Results:\n")
        for i, (tl, el, ta, ea) in enumerate(zip(train_losses, eval_losses, train_accuracies, eval_accuracies), 1):
            f.write(f"  Epoch {i}:\n")
            f.write(f"    Train Loss: {tl:.4f}, Train Acc: {ta:.4f}\n")
            f.write(f"    Eval Loss:  {el:.4f}, Eval Acc:  {ea:.4f}\n")
        f.write(f"\nBest Eval Accuracy: {best_eval_accuracy:.4f}\n")

    print(f"✅ Training log saved to: {log_path}")

    # ========== 9. 上傳到 Hugging Face Hub (可選) ==========
    if args.push_to_hub:
        print("\n" + "="*60)
        print("📤 Uploading to Hugging Face Hub...")
        print("="*60)

        try:
            # 合併 LoRA 權重到基礎模型
            print("Merging LoRA weights...")
            merge_lora_weights(model)

            # 準備 model card
            model_card = f"""---
language: en
license: mit
tags:
- text-classification
- sentiment-analysis
- lora
- bert
- sst2
datasets:
- glue
metrics:
- accuracy
---

# BERT-base LoRA Fine-tuned on SST-2

LoRA fine-tuned BERT model for binary sentiment classification on the SST-2 dataset.

## Model Details

- **Base Model**: `{config['model_name']}`
- **Task**: Binary Sentiment Classification (SST-2)
- **Method**: LoRA (Low-Rank Adaptation)
- **Language**: English

## Training Configuration

- **LoRA rank**: {config['lora']['rank']}
- **LoRA alpha**: {config['lora']['alpha']}
- **LoRA dropout**: {config['lora']['dropout']}
- **Target modules**: {', '.join(config['lora']['target_modules'])}
- **Training epochs**: {config['training']['num_epochs']}
- **Learning rate**: {config['training']['learning_rate']}
- **Batch size**: {config['training']['batch_size']}

## Performance

- **Best Validation Accuracy**: {best_eval_accuracy:.4f}
- **Trainable Parameters**: {param_stats['trainable']:,} ({param_stats['percentage']:.4f}% of total)
- **Total Parameters**: {param_stats['total']:,}

## Training Results

| Epoch | Train Loss | Train Acc | Eval Loss | Eval Acc |
|-------|-----------|-----------|-----------|----------|
"""
            for i, (tl, el, ta, ea) in enumerate(zip(train_losses, eval_losses, train_accuracies, eval_accuracies), 1):
                model_card += f"| {i} | {tl:.4f} | {ta:.4f} | {el:.4f} | {ea:.4f} |\n"

            model_card += f"""
## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("{args.hub_model_id}")
tokenizer = AutoTokenizer.from_pretrained("{args.hub_model_id}")

# Prepare input
text = "This movie is fantastic!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Predict
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "Positive" if predictions[0][1] > 0.5 else "Negative"
    confidence = predictions[0][1].item() if predictions[0][1] > 0.5 else predictions[0][0].item()

print(f"Sentiment: {{sentiment}} (Confidence: {{confidence:.2%}})")
```

## Dataset

This model was trained on the [SST-2 (Stanford Sentiment Treebank)](https://huggingface.co/datasets/glue) dataset:
- **Training samples**: 67,349
- **Validation samples**: 872
- **Task**: Binary sentiment classification (positive/negative)

## Training Procedure

The model was fine-tuned using LoRA (Low-Rank Adaptation), which only updates a small number of parameters:
- Only {param_stats['percentage']:.4f}% of the model's parameters were trained
- The base BERT weights remain frozen
- LoRA adapters were applied to: {', '.join(config['lora']['target_modules'])}

## Limitations

- This model is specifically trained for sentiment classification on movie reviews
- Performance may vary on other domains or text types
- The model is based on BERT-base and has the same limitations (max sequence length: 512 tokens)

## Citation

If you use this model, please cite:

```bibtex
@misc{{bert-lora-sst2,
  author = {{LLM Tuning Lab}},
  title = {{BERT-base LoRA Fine-tuned on SST-2}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{args.hub_model_id}}}}},
}}
```

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
"""

            # 儲存 model card
            readme_path = output_dir / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(model_card)
            print(f"✅ Model card created: {readme_path}")

            # 上傳模型
            print(f"Uploading model to: {args.hub_model_id}")
            model.push_to_hub(
                repo_id=args.hub_model_id,
                commit_message=f"Upload LoRA fine-tuned BERT (Acc: {best_eval_accuracy:.4f})",
                private=args.hub_private
            )

            # 上傳 tokenizer
            print(f"Uploading tokenizer...")
            tokenizer.push_to_hub(
                repo_id=args.hub_model_id,
                commit_message="Upload tokenizer",
                private=args.hub_private
            )

            # 上傳訓練曲線圖
            print(f"Uploading training curves...")
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=str(output_dir / "training_curves.png"),
                path_in_repo="training_curves.png",
                repo_id=args.hub_model_id,
                repo_type="model",
                commit_message="Upload training curves"
            )

            print(f"\n✅ Model successfully uploaded!")
            print(f"🌐 View at: https://huggingface.co/{args.hub_model_id}")

        except Exception as e:
            print(f"\n❌ Upload failed: {e}")
            print("Please check:")
            print("  1. Hugging Face token is set (run: huggingface-cli login)")
            print("  2. Model ID format is correct (username/model-name)")
            print("  3. You have write permissions to the repository")

    # ========== 10. 完成 ==========
    print("\n" + "="*60)
    print("🎉 Training Completed!")
    print("="*60)
    print(f"\n📁 Output Directory: {output_dir.absolute()}")
    print(f"📈 Best Eval Accuracy: {best_eval_accuracy:.4f}")
    print(f"📉 Final Train Loss: {train_losses[-1]:.4f}")
    print(f"📉 Final Eval Loss: {eval_losses[-1]:.4f}")
    print(f"🎯 Final Train Accuracy: {train_accuracies[-1]:.4f}")
    print(f"🎯 Final Eval Accuracy: {eval_accuracies[-1]:.4f}")
    print("\n✨ Files created:")
    print(f"  - {final_checkpoint_path.name}")
    print(f"  - {lora_only_path.name}")
    print(f"  - best_lora_model.pt")
    print(f"  - training_curves.png")
    print(f"  - training_log.txt")
    print("\n💡 Next steps:")
    print(f"  - Run inference: python inference_example.py")
    if args.push_to_hub:
        print(f"  - View model: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BERT with LoRA on SST-2 Sentiment Classification"
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
        "--push_to_hub",
        action="store_true",
        help="Push model to Hugging Face Hub after training"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="your-username/bert-lora-sst2",
        help="Hugging Face Hub model ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Make the uploaded model private on Hugging Face Hub"
    )

    args = parser.parse_args()
    main(args)
