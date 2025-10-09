"""測試模型（本地或 HF Hub）"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

def main(args):
    print("=" * 60)
    print("🧪 測試模型")
    print("=" * 60)

    # 載入模型
    if args.model_path:
        print(f"\n📥 載入本地模型: {args.model_path}")
        model_name = args.model_path
    else:
        print(f"\n📥 從 HF Hub 載入: {args.model_name}")
        model_name = args.model_name

    # 使用 pipeline（最簡單）
    try:
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )

        print(f"\n🤖 輸入: {args.prompt}")
        outputs = generator(
            args.prompt,
            max_length=args.max_length,
            num_return_sequences=1,
            temperature=0.7,
        )

        print(f"💬 輸出: {outputs[0]['generated_text']}")

    except Exception as e:
        print(f"❌ 錯誤: {e}")
        print("\n提示：如果是 LoRA 模型，可能需要手動載入")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="本地模型路徑")
    parser.add_argument("--model_name", type=str, help="HF Hub 模型名稱")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="測試 prompt")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成長度")
    args = parser.parse_args()
    
    if not args.model_path and not args.model_name:
        print("❌ 請提供 --model_path 或 --model_name")
        exit(1)
    
    main(args)
