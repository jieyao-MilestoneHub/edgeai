"""
Task 03: Engineering Agent - QLoRA 指令微調訓練腳本

功能：
1. 載入 Qwen2.5-3B-Instruct 並應用 4-bit NF4 量化
2. 套用 LoRA 到 Attention 和 MLP 層
3. 載入準備好的 instruction-following 數據集
4. 使用 SFTTrainer 進行 supervised fine-tuning
5. 保存 LoRA adapter 和訓練日誌

使用套件：
- transformers: 模型載入、Trainer
- peft: LoRA/QLoRA 實作
- bitsandbytes: 4-bit 量化
- trl: SFTTrainer (Supervised Fine-tuning)

硬體需求：
- RTX 4060 8GB: ✅ 可運行（batch_size=1, gradient_accumulation=16）
- RTX 3090 24GB: ✅ 可運行（batch_size=4, gradient_accumulation=4）

作者: Joel
授權: MIT
"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
from typing import Dict, Any

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig
import matplotlib.pyplot as plt


# ====================================================================================
# 1) 載入配置
# ====================================================================================

def load_config(config_path: str = "../config.yaml") -> Dict[str, Any]:
    """載入配置檔案"""
    print(f"📋 Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded successfully!")
    return config


def merge_configs(base_config: Dict, exp_config_path: str = None) -> Dict:
    """合併基礎配置和實驗配置"""
    if exp_config_path and Path(exp_config_path).exists():
        print(f"📋 Merging with experiment config: {exp_config_path}")
        with open(exp_config_path, 'r', encoding='utf-8') as f:
            exp_config = yaml.safe_load(f)

        # 深度合併
        for key, value in exp_config.items():
            if isinstance(value, dict) and key in base_config:
                base_config[key].update(value)
            else:
                base_config[key] = value

    return base_config


# ====================================================================================
# 2) 模型與 Tokenizer 載入
# ====================================================================================

def create_bnb_config(config: Dict) -> BitsAndBytesConfig:
    """
    建立 BitsAndBytes 量化配置

    QLoRA 核心配置：
    - 4-bit NF4 量化
    - BF16 計算精度
    - 雙重量化
    """
    return BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_quant_type=config['quantization']['quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, config['quantization']['compute_dtype']),
        bnb_4bit_use_double_quant=config['quantization']['use_double_quant'],
    )


def load_model_and_tokenizer(config: Dict):
    """
    載入模型和 Tokenizer

    使用 4-bit 量化載入模型
    """
    print("\n" + "="*60)
    print("🤖 Loading Model and Tokenizer...")
    print("="*60)

    model_name = config['model']['name']
    print(f"   Model: {model_name}")

    # 建立量化配置
    bnb_config = create_bnb_config(config)

    # 載入 Tokenizer
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 設定 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 載入模型（4-bit 量化）
    print("   Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("✅ Model and tokenizer loaded successfully!")

    # 顯示記憶體使用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")

    return model, tokenizer


# ====================================================================================
# 3) LoRA 配置與套用
# ====================================================================================

def create_lora_config(config: Dict) -> LoraConfig:
    """建立 LoRA 配置"""
    return LoraConfig(
        r=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type'],
        target_modules=config['lora']['target_modules'],
    )


def prepare_model_for_training(model, config: Dict):
    """
    準備模型進行訓練

    步驟：
    1. prepare_model_for_kbit_training (啟用梯度檢查點等)
    2. 套用 LoRA
    3. 顯示可訓練參數統計
    """
    print("\n" + "="*60)
    print("🔧 Preparing Model for Training...")
    print("="*60)

    # 準備 k-bit 訓練
    print("   Preparing for k-bit training...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config['training']['gradient_checkpointing']
    )

    # 建立並套用 LoRA
    print(f"   Applying LoRA (rank={config['lora']['rank']})...")
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)

    # 顯示可訓練參數
    model.print_trainable_parameters()

    print("✅ Model prepared for training!")

    return model


# ====================================================================================
# 4) 數據集載入
# ====================================================================================

def load_dataset(data_path: str = "../data_preparation/final_dataset"):
    """
    載入準備好的數據集

    數據格式：
    {
        "instruction": "指令",
        "input": "輸入（可選）",
        "output": "輸出"
    }
    """
    print("\n" + "="*60)
    print("📚 Loading Dataset...")
    print("="*60)

    print(f"   Dataset path: {data_path}")

    try:
        dataset = load_from_disk(data_path)

        # Filter to only keep required columns (remove 'image' and other unused columns)
        required_columns = ['instruction', 'input', 'output']
        dataset = dataset.select_columns(required_columns)

        print("✅ Dataset loaded successfully!")
        print(f"   Train samples: {len(dataset['train'])}")
        print(f"   Test samples: {len(dataset['test'])}")
        print(f"   Columns: {dataset['train'].column_names}")

        # 顯示範例
        print("\n📝 Sample data:")
        sample = dataset['train'][0]
        print(f"   Instruction: {sample['instruction'][:100]}...")
        print(f"   Output: {sample['output'][:100]}...")

        return dataset

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"   Please run data preparation first:")
        print(f"   cd ../data_preparation && python merge_datasets.py")
        sys.exit(1)


def formatting_prompts_func(example):
    """
    格式化 prompt 函數

    將 instruction-input-output 格式轉換為訓練用的 text

    格式：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {instruction}
    {input}<|im_end|>
    <|im_start|>assistant
    {output}<|im_end|>
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')

    # 構建 prompt
    prompt = "<|im_start|>system\nYou are a helpful AI assistant specialized in engineering and technical domains.<|im_end|>\n"
    prompt += f"<|im_start|>user\n{instruction}"

    if input_text:
        prompt += f"\n\n{input_text}"

    prompt += "<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n{output}<|im_end|>"

    return prompt


# ====================================================================================
# 5) 訓練配置與執行
# ====================================================================================

def create_training_arguments(config: Dict) -> SFTConfig:
    """建立訓練參數"""
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    return SFTConfig(
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

        # 精度
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],

        # 優化器
        optim=config['training']['optim'],
        max_grad_norm=config['training']['max_grad_norm'],

        # 日誌與儲存
        logging_dir=config['output']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        save_total_limit=config['training']['save_total_limit'],

        # 評估
        eval_strategy="steps",

        # 其他
        report_to="tensorboard",
        seed=config['advanced']['seed'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],

        # SFT specific - max sequence length
        max_length=config['data']['max_length'],
    )


def plot_training_curves(log_history, save_path: str):
    """繪製訓練曲線"""
    print(f"\n📊 Plotting training curves...")

    # 提取資料
    train_loss = [(entry['step'], entry['loss']) for entry in log_history if 'loss' in entry]
    eval_loss = [(entry['step'], entry['eval_loss']) for entry in log_history if 'eval_loss' in entry]
    learning_rates = [(entry['step'], entry['learning_rate']) for entry in log_history if 'learning_rate' in entry]

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
# 6) 主訓練函數
# ====================================================================================

def main(args):
    """主訓練函數"""
    # 載入配置
    config = load_config(args.config)

    # 合併實驗配置（如果有）
    if args.exp_config:
        config = merge_configs(config, args.exp_config)

    # 命令列覆寫
    if args.output_dir:
        config['output']['dir'] = args.output_dir

    # 顯示裝置信息
    if not torch.cuda.is_available():
        print("❌ CUDA not available. QLoRA requires GPU.")
        sys.exit(1)

    device = torch.device('cuda')
    print(f"\n🖥️  Device Information:")
    print(f"   Device: {device}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 設定隨機種子
    torch.manual_seed(config['advanced']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['advanced']['seed'])

    # 啟用 TF32
    if config['advanced']['allow_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("\n" + "="*60)
    print("🚀 QLoRA Training - Qwen2.5-3B Engineering Agent")
    print("="*60)

    # 載入模型與 tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # 準備模型
    model = prepare_model_for_training(model, config)

    # 載入數據集
    dataset = load_dataset(args.data_path)

    # 建立訓練參數
    training_args = create_training_arguments(config)

    # 建立 SFTTrainer
    print("\n" + "="*60)
    print("🎓 Creating Trainer...")
    print("="*60)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
    )

    # 開始訓練
    print("\n" + "="*60)
    print("🚀 Starting Training...")
    print("="*60)

    try:
        train_result = trainer.train()

        print("\n" + "="*60)
        print("✅ Training Completed!")
        print("="*60)

        # 顯示結果
        print(f"\n📊 Training Results:")
        print(f"   Final Train Loss: {train_result.training_loss:.4f}")
        print(f"   Training Time: {train_result.metrics['train_runtime']:.2f} seconds")

        # 最終評估
        print(f"\n📈 Running Final Evaluation...")
        eval_result = trainer.evaluate()
        print(f"   Final Eval Loss: {eval_result['eval_loss']:.4f}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")

    # 保存結果
    print("\n" + "="*60)
    print("💾 Saving Results...")
    print("="*60)

    output_dir = Path(config['output']['dir'])

    # 保存 LoRA adapter
    print("📦 Saving LoRA adapter...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved to: {output_dir}")

    # 繪製訓練曲線
    if config['output']['save_training_curves']:
        plot_training_curves(
            trainer.state.log_history,
            str(output_dir / "training_curves.png")
        )

    # 保存訓練日誌
    if config['output']['save_training_log']:
        log_path = output_dir / "training_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("QLoRA Training Log - Engineering Agent\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: {config['model']['name']}\n")
            f.write(f"LoRA Rank: {config['lora']['rank']}\n")
            f.write(f"Target Modules: {', '.join(config['lora']['target_modules'])}\n\n")

            if 'train_result' in locals():
                f.write(f"Final Train Loss: {train_result.training_loss:.4f}\n")
                f.write(f"Final Eval Loss: {eval_result['eval_loss']:.4f}\n")

        print(f"📝 Training log saved to: {log_path}")

    print("\n" + "="*60)
    print("🎉 All Done!")
    print("="*60)
    print(f"\n💡 Next steps:")
    print(f"   - Test model: python inference_example.py --model_path {output_dir}")
    print(f"   - View logs: tensorboard --logdir {config['output']['logging_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen2.5-3B with QLoRA")
    parser.add_argument("--config", type=str, default="../config.yaml", help="Base config file")
    parser.add_argument("--exp_config", type=str, default=None, help="Experiment config file")
    parser.add_argument("--data_path", type=str, default="../data_preparation/final_dataset", help="Dataset path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    main(args)
