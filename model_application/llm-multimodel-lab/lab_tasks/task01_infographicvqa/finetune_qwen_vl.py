#!/usr/bin/env python3
"""
Qwen2.5-VL-3B InfographicVQA微調腳本 (QLoRA)

使用方法:
python finetune_qwen_vl.py \\
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \\
  --output_dir ./outputs/qwen_vl \\
  --num_epochs 3 \\
  --batch_size 2 \\
  --learning_rate 2e-4
"""

import argparse
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL微調腳本")

    # 模型相關
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="HuggingFace模型ID")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen_vl",
                        help="輸出目錄")

    # 訓練參數
    parser.add_argument("--num_epochs", type=int, default=3, help="訓練epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="每設備batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累積步數")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="學習率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="權重衰減")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup步數")

    # QLoRA參數
    parser.add_argument("--use_qlora", action="store_true", default=True,
                        help="使用QLoRA(4-bit量化)")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # 資料集
    parser.add_argument("--dataset_name", type=str, default="MMInstruction/InfographicVQA",
                        help="資料集名稱")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列長度")

    # 記憶體優化
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="啟用gradient checkpointing")

    # 評估與儲存
    parser.add_argument("--eval_steps", type=int, default=500, help="評估步數")
    parser.add_argument("--save_steps", type=int, default=500, help="儲存步數")
    parser.add_argument("--logging_steps", type=int, default=50, help="日誌步數")

    return parser.parse_args()


def preprocess_function(examples, processor, max_length=2048):
    """
    將資料轉換為模型輸入格式
    """
    images = examples['image']
    questions = examples['question']
    answers = [ans[0] if isinstance(ans, list) else ans for ans in examples['answers']]

    # 構建Qwen-VL對話格式
    conversations = []
    for q, a in zip(questions, answers):
        conversations.append([
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Question: {q}\nAnswer:"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": a}]
            }
        ])

    # 應用chat template
    texts = processor.apply_chat_template(
        conversations,
        add_generation_prompt=False,
        tokenize=False
    )

    # 處理圖片和文字
    model_inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Labels = input_ids
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs


def main():
    args = parse_args()

    print("=" * 60)
    print("Qwen2.5-VL-3B InfographicVQA 微調")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"輸出目錄: {args.output_dir}")
    print(f"使用QLoRA: {args.use_qlora}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)

    # ===== 1. 配置QLoRA =====
    if args.use_qlora:
        print("\n[1/6] 配置QLoRA 4-bit量化...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("✓ QLoRA配置完成")
    else:
        bnb_config = None
        print("\n[1/6] 使用標準精度訓練(不量化)")

    # ===== 2. 載入模型 =====
    print("\n[2/6] 載入模型...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ 模型載入完成")

    # 準備QLoRA訓練
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # ===== 3. 添加LoRA adapters =====
    print("\n[3/6] 配置LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✓ LoRA配置完成")

    # ===== 4. 載入資料集 =====
    print(f"\n[4/6] 載入資料集: {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name)
    print(f"✓ 訓練集: {len(dataset['train'])} 樣本")
    print(f"✓ 驗證集: {len(dataset['validation'])} 樣本")

    # 載入processor
    processor = Qwen2VLProcessor.from_pretrained(args.model_name)

    # 預處理資料
    print("   處理訓練資料...")
    processed_train = dataset['train'].map(
        lambda x: preprocess_function(x, processor, args.max_length),
        batched=True,
        batch_size=8,
        remove_columns=dataset['train'].column_names
    )

    print("   處理驗證資料...")
    processed_val = dataset['validation'].map(
        lambda x: preprocess_function(x, processor, args.max_length),
        batched=True,
        batch_size=8,
        remove_columns=dataset['validation'].column_names
    )
    print("✓ 資料預處理完成")

    # ===== 5. 訓練參數 =====
    print("\n[5/6] 配置訓練參數...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # 訓練設定
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # 優化器
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,

        # 記憶體優化
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=False,
        bf16=True,

        # 評估與儲存
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # 日誌
        logging_steps=args.logging_steps,
        report_to="tensorboard",

        # 其他
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    print("✓ 訓練參數配置完成")

    # ===== 6. 建立Trainer =====
    print("\n[6/6] 建立Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_val,
        tokenizer=processor.tokenizer,
    )
    print("✓ Trainer建立完成")

    # ===== 開始訓練 =====
    print("\n" + "=" * 60)
    print("開始訓練...")
    print("=" * 60)
    print(f"有效batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"總訓練步數: {len(processed_train) // (args.batch_size * args.gradient_accumulation_steps) * args.num_epochs}")
    print("=" * 60 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n訓練被中斷!")
    except Exception as e:
        print(f"\n訓練錯誤: {e}")
        raise

    # ===== 儲存模型 =====
    print(f"\n儲存模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("✓ 訓練完成!")
    print("=" * 60)
    print(f"模型已儲存到: {args.output_dir}")
    print(f"TensorBoard日誌: {os.path.join(args.output_dir, 'logs')}")
    print("\n使用以下命令進行推論:")
    print(f"python inference.py --model_path {args.output_dir}")


if __name__ == "__main__":
    main()
