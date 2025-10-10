"""
LoRA Inference Example - SST-2 Sentiment Classification

示範如何使用訓練好的 LoRA 模型進行情感分析推論

作者: Jiao
授權: MIT
"""

import argparse
from pathlib import Path
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from lora_linear import (
    apply_lora_to_model,
    load_lora_state_dict,
)


def load_model_with_lora(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """載入帶有 LoRA 權重的模型"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # 1. 載入基礎模型
    print(f"Loading base model: {config['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=config['num_labels']
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # 2. 套用 LoRA 結構
    print(f"Applying LoRA structure (rank={config['lora']['rank']})")
    apply_lora_to_model(
        model,
        target_modules=config['lora']['target_modules'],
        rank=config['lora']['rank'],
        alpha=config['lora']['alpha'],
        dropout=0.0,  # 推論時不需要 dropout
    )

    # 3. 載入 LoRA 權重
    print(f"Loading LoRA weights...")
    load_lora_state_dict(model, checkpoint['lora_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully!")
    print(f"   Device: {device}")
    if 'eval_acc' in checkpoint:
        print(f"   Validation Accuracy: {checkpoint['eval_acc']:.4f}")

    return model, tokenizer, device


def batch_predict(
    texts: list[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = 16
) -> list[tuple[str, float]]:
    """批次預測多個文本的情感

    Args:
        texts: 要預測的文本列表
        model: 模型
        tokenizer: Tokenizer
        device: 設備 (cpu/cuda)
        batch_size: 批次大小（默認 16）

    Returns:
        list[tuple[str, float]]: 每個文本的 (情感標籤, 信心分數)
    """
    label_map = {0: "Negative", 1: "Positive"}
    results = []

    # 分批處理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # 批次 tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)

        # 批次推論
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1)
            confidences = torch.gather(probs, 1, pred_labels.unsqueeze(1)).squeeze(1)

        # 收集結果
        for pred_label, confidence in zip(pred_labels.cpu().tolist(), confidences.cpu().tolist()):
            sentiment = label_map[pred_label]
            results.append((sentiment, confidence))

    return results


def interactive_mode(model, tokenizer, device):
    """互動模式：讓使用者輸入句子進行即時預測"""
    print("\n" + "="*60)
    print("🎭 Interactive Sentiment Analysis")
    print("="*60)
    print("輸入句子進行情感分析，輸入 'quit' 或 'exit' 結束")
    print("-"*60)

    while True:
        try:
            text = input("\n請輸入句子: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not text:
                print("⚠️  請輸入有效的句子")
                continue

            # 預測（使用 batch_predict 處理單一文本）
            results = batch_predict([text], model, tokenizer, device)
            sentiment, confidence = results[0]

            # 顯示結果
            emoji = "😊" if sentiment == "Positive" else "😞"
            print(f"\n{emoji} 情感: {sentiment}")
            print(f"📊 信心分數: {confidence:.2%}")

            # 顯示信心等級
            if confidence > 0.9:
                level = "非常確定"
            elif confidence > 0.7:
                level = "確定"
            elif confidence > 0.5:
                level = "較為確定"
            else:
                level = "不太確定"
            print(f"🎯 等級: {level}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")


def demo_examples(model, tokenizer, device):
    """示範範例（使用批次處理提升效率）"""
    print("\n" + "="*60)
    print("📝 Demo Examples")
    print("="*60)

    examples = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The worst film I've ever seen. Totally disappointing.",
        "It was okay, nothing special but not terrible either.",
        "Brilliant performance by the lead actor!",
        "I fell asleep halfway through. So boring.",
        "A masterpiece! Highly recommended.",
        "Waste of time and money.",
        "Pretty good, would watch again.",
    ]

    # 批次預測所有範例（更高效）
    results = batch_predict(examples, model, tokenizer, device)

    # 顯示結果
    for i, (text, (sentiment, confidence)) in enumerate(zip(examples, results), 1):
        emoji = "😊" if sentiment == "Positive" else "😞"
        print(f"\n{i}. \"{text}\"")
        print(f"   {emoji} {sentiment} ({confidence:.2%})")


def main(args):
    # 載入模型
    checkpoint_path = args.checkpoint if args.checkpoint else "output/best_lora_model.pt"

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("請先訓練模型: python train_lora_basic.py")
        return

    model, tokenizer, device = load_model_with_lora(checkpoint_path)

    # 根據模式執行
    if args.mode == "interactive":
        interactive_mode(model, tokenizer, device)

    elif args.mode == "demo":
        demo_examples(model, tokenizer, device)

    elif args.mode == "text":
        if not args.text:
            print("❌ 請使用 --text 參數提供輸入文本")
            return

        # 使用 batch_predict 處理單一文本
        results = batch_predict([args.text], model, tokenizer, device)
        sentiment, confidence = results[0]
        emoji = "😊" if sentiment == "Positive" else "😞"

        print(f"\n輸入: \"{args.text}\"")
        print(f"{emoji} 情感: {sentiment}")
        print(f"📊 信心分數: {confidence:.2%}")

    else:
        print(f"❌ 未知模式: {args.mode}")
        print("可用模式: interactive, demo, text")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA Sentiment Classification Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/best_lora_model.pt",
        help="Path to checkpoint file (default: output/best_lora_model.pt)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "demo", "text"],
        help="Inference mode: interactive (互動模式), demo (範例展示), text (單次預測)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text for sentiment analysis (when mode=text)"
    )

    args = parser.parse_args()
    main(args)
