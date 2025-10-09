"""
LoRA 基礎訓練腳本

這個腳本展示如何使用手寫的 LoRA 實作來微調語言模型。

使用方式：
    python train_lora_basic.py --rank 8 --alpha 16 --epochs 3

作者：LLM Tuning Lab
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import json
import os
from datetime import datetime

# 導入我們手寫的 LoRA 模組
from lora_linear import (
    apply_lora_to_model,
    count_lora_parameters,
    get_lora_parameters,
)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    訓練一個 epoch

    Args:
        model: 模型
        dataloader: 訓練資料
        optimizer: 優化器
        scheduler: 學習率調度器
        device: 計算設備
        epoch: 當前 epoch 編號

    Returns:
        平均 loss
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        leave=True,
    )

    for batch_idx, batch in enumerate(progress_bar):
        # 移動資料到 device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 前向傳播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # 語言模型：輸入即標籤
        )
        loss = outputs.loss

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # 更新參數
        optimizer.step()
        scheduler.step()

        # 記錄
        total_loss += loss.item()
        current_lr = scheduler.get_last_lr()[0]

        # 更新進度條
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
            'lr': f'{current_lr:.2e}',
        })

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    評估模型

    Args:
        model: 模型
        dataloader: 評估資料
        device: 計算設備

    Returns:
        平均 loss
    """
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(
        dataloader,
        desc="Evaluating",
        leave=False,
    )

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        total_loss += outputs.loss.item()
        progress_bar.set_postfix({
            'loss': f'{outputs.loss.item():.4f}'
        })

    avg_loss = total_loss / num_batches
    return avg_loss


def plot_training_curve(train_losses, eval_losses, save_path):
    """繪製訓練曲線"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    plt.plot(epochs, eval_losses, 'r-s', label='Eval Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('LoRA Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 添加數值標註
    for i, (train_loss, eval_loss) in enumerate(zip(train_losses, eval_losses)):
        plt.text(i+1, train_loss, f'{train_loss:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i+1, eval_loss, f'{eval_loss:.3f}', ha='center', va='top', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss curve saved to: {save_path}")


def main(args):
    """主訓練函數"""

    # ========== 1. 環境設定 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"🚀 LoRA Training Script")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")

    # ========== 2. 載入模型與 Tokenizer ==========
    print("📥 Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    # 設定 pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # ========== 3. 應用 LoRA ==========
    print(f"\n🔧 Applying LoRA (rank={args.rank}, alpha={args.alpha})...")

    # 對 GPT-2 的 attention 層應用 LoRA
    # GPT-2 使用 'c_attn' (Q,K,V 合併) 和 'c_proj' (O)
    model = apply_lora_to_model(
        model,
        target_modules=['c_attn', 'c_proj'],  # GPT-2 特定
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.lora_dropout,
    )

    model = model.to(device)

    # ========== 4. 參數統計 ==========
    print("\n📊 Parameter Statistics:")
    print("─" * 60)
    param_stats = count_lora_parameters(model)
    print(f"Total parameters:     {param_stats['total']:>15,}")
    print(f"Trainable parameters: {param_stats['trainable']:>15,}")
    print(f"Frozen parameters:    {param_stats['frozen']:>15,}")
    print(f"Trainable percentage: {param_stats['percentage']:>14.4f}%")
    print(f"Parameter reduction:  {param_stats['total']/param_stats['trainable']:>14.1f}×")
    print("─" * 60)

    # ========== 5. 載入數據集 ==========
    print("\n📚 Loading dataset...")

    # 使用 WikiText-2 作為示範
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        """Tokenize 文本"""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Tokenize 資料集
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    # 過濾掉空序列
    def filter_empty(example):
        return len(example['input_ids']) > 0

    tokenized_datasets = tokenized_datasets.filter(filter_empty)

    print(f"Train samples: {len(tokenized_datasets['train'])}")
    print(f"Validation samples: {len(tokenized_datasets['validation'])}")

    # 創建 DataLoader
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows 相容性
    )

    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=args.batch_size,
        num_workers=0,
    )

    # ========== 6. 優化器與調度器 ==========
    print("\n⚙️ Setting up optimizer and scheduler...")

    # 只優化 LoRA 參數
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 學習率調度器
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")

    # ========== 7. 訓練循環 ==========
    print(f"\n🏋️ Starting training for {args.num_epochs} epochs...\n")

    train_losses = []
    eval_losses = []
    best_eval_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        # 訓練
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, device, epoch
        )
        train_losses.append(train_loss)

        # 評估
        eval_loss = evaluate(model, eval_dataloader, device)
        eval_losses.append(eval_loss)

        # 打印統計
        print(f"\n📈 Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Eval Loss:  {eval_loss:.4f}")

        # 保存最佳模型
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            print(f"  🌟 New best model! (Eval Loss: {eval_loss:.4f})")

            # 保存 LoRA 權重
            lora_params = get_lora_parameters(model)
            torch.save({
                'lora_state_dict': lora_params,
                'config': {
                    'rank': args.rank,
                    'alpha': args.alpha,
                    'model_name': args.model_name,
                    'target_modules': ['c_attn', 'c_proj'],
                },
                'training': {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                },
            }, os.path.join(args.output_dir, 'best_adapter_model.bin'))

    # ========== 8. 保存最終結果 ==========
    print(f"\n{'='*60}")
    print("💾 Saving results...")
    print(f"{'='*60}")

    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存最終 LoRA 權重
    lora_params = get_lora_parameters(model)
    torch.save({
        'lora_state_dict': lora_params,
        'config': {
            'rank': args.rank,
            'alpha': args.alpha,
            'model_name': args.model_name,
            'target_modules': ['c_attn', 'c_proj'],
        },
        'training': {
            'final_train_loss': train_losses[-1],
            'final_eval_loss': eval_losses[-1],
            'best_eval_loss': best_eval_loss,
        },
    }, os.path.join(args.output_dir, 'final_adapter_model.bin'))
    print(f"✅ Saved: {os.path.join(args.output_dir, 'final_adapter_model.bin')}")

    # 保存訓練 metrics
    metrics = {
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'best_eval_loss': best_eval_loss,
        'hyperparameters': vars(args),
        'param_stats': param_stats,
    }

    with open(os.path.join(args.output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved: {os.path.join(args.output_dir, 'training_metrics.json')}")

    # 繪製訓練曲線
    plot_training_curve(
        train_losses,
        eval_losses,
        os.path.join(args.output_dir, 'training_loss_curve.png')
    )

    # ========== 9. 總結 ==========
    print(f"\n{'='*60}")
    print("🎉 Training Completed!")
    print(f"{'='*60}")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Eval Loss:  {eval_losses[-1]:.4f}")
    print(f"Best Eval Loss:   {best_eval_loss:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Script")

    # 模型參數
    parser.add_argument("--model_name", type=str, default="gpt2",
                       help="Pretrained model name")

    # LoRA 參數
    parser.add_argument("--rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0,
                       help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                       help="LoRA dropout probability")

    # 訓練參數
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")

    # 輸出參數
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")

    args = parser.parse_args()

    # 運行訓練
    main(args)
