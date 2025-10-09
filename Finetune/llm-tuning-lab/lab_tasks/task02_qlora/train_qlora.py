"""
QLoRA 訓練腳本

使用 4-bit 量化 + LoRA 訓練大型語言模型

使用方式：
    python train_qlora.py --model meta-llama/Llama-2-7b-hf --epochs 3

作者：LLM Tuning Lab
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from datasets import load_dataset
import argparse
import os

def create_bnb_config():
    """創建 4-bit 量化配置"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def create_lora_config(args):
    """創建 LoRA 配置"""
    return LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

def load_model_and_tokenizer(args):
    """載入量化模型和 tokenizer"""
    print(f"Loading model: {args.model_name}")

    # 量化配置
    bnb_config = create_bnb_config()

    # 載入模型（自動量化）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def prepare_model_for_training(model, args):
    """準備模型並添加 LoRA"""
    print("Preparing model for k-bit training...")

    # 啟用梯度檢查點等優化
    model = prepare_model_for_kbit_training(model)

    # 啟用梯度檢查點（節省記憶體）
    model.gradient_checkpointing_enable()

    # 添加 LoRA
    lora_config = create_lora_config(args)
    model = get_peft_model(model, lora_config)

    # 顯示參數統計
    model.print_trainable_parameters()

    return model

def print_memory_stats():
    """顯示記憶體使用情況"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")

def main(args):
    print("=" * 60)
    print("🚀 QLoRA Training Script")
    print("=" * 60)

    # 檢查 CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. QLoRA requires GPU.")
        return

    # 載入模型
    model, tokenizer = load_model_and_tokenizer(args)
    print_memory_stats()

    # 準備訓練
    model = prepare_model_for_training(model, args)
    print_memory_stats()

    # 載入數據
    print("\nLoading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # 訓練配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=False,  # QLoRA 使用 bf16
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        optim="paged_adamw_32bit",  # QLoRA 推薦的優化器
        warmup_steps=100,
        report_to="tensorboard",
    )

    # 創建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # 開始訓練
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    # 保存模型
    print("\nSaving model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 模型參數
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")

    # LoRA 參數
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 訓練參數
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)

    # 輸出
    parser.add_argument("--output_dir", type=str, default="./output_qlora")

    args = parser.parse_args()
    main(args)
