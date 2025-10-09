"""
超簡化的微調腳本 - 使用 PEFT 庫

這個腳本讓你用最少的程式碼完成模型微調

使用方式：
    python simple_finetune.py --data_file my_data.csv --output_dir ./my_model

作者：LLM Tuning Lab
"""

import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

def load_data(file_path):
    """
    載入數據 - 支援 CSV, JSON, TXT

    CSV 格式: input,output
    JSON 格式: [{"input": "...", "output": "..."}]
    TXT 格式: 每行一個對話
    """
    print(f"📊 載入數據: {file_path}")

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        texts = [f"{row['input']}\n{row['output']}" for _, row in df.iterrows()]
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
        texts = [f"{row['input']}\n{row['output']}" for _, row in df.iterrows()]
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        raise ValueError("支援的格式: .csv, .json, .txt")

    print(f"✅ 載入了 {len(texts)} 個訓練樣本")
    return Dataset.from_dict({"text": texts})

def main(args):
    print("=" * 60)
    print("🚀 開始微調小模型")
    print("=" * 60)

    # ========== 1. 載入模型與 Tokenizer ==========
    print(f"\n📥 載入模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # 設定 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ========== 2. 應用 LoRA（使用 PEFT）==========
    print(f"\n🔧 添加 LoRA (rank={args.lora_rank})")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"] if "gpt2" in args.model_name.lower() else ["q_proj", "v_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ========== 3. 載入數據 ==========
    dataset = load_data(args.data_file)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # ========== 4. 訓練配置 ==========
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",  # 不使用 wandb 等
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    # ========== 5. 訓練 ==========
    print(f"\n🏋️ 開始訓練 {args.epochs} epochs...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # ========== 6. 保存模型 ==========
    print(f"\n💾 保存模型到: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("✅ 訓練完成！")
    print("=" * 60)
    print(f"\n模型已保存到: {args.output_dir}")
    print("\n下一步：")
    print(f"1. 測試模型: python test_from_hf.py --model_path {args.output_dir}")
    print(f"2. 上傳到 HF: python upload_to_hf.py --model_path {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="簡化的微調腳本")

    # 模型參數
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="模型名稱 (distilgpt2, gpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0)")

    # 數據參數
    parser.add_argument("--data_file", type=str, required=True,
                       help="訓練數據檔案 (.csv, .json, .txt)")

    # LoRA 參數
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16,
                       help="LoRA alpha")

    # 訓練參數
    parser.add_argument("--epochs", type=int, default=3,
                       help="訓練 epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="學習率")
    parser.add_argument("--max_length", type=int, default=256,
                       help="最大序列長度")

    # 輸出參數
    parser.add_argument("--output_dir", type=str, default="./my_model",
                       help="輸出目錄")

    args = parser.parse_args()
    main(args)
