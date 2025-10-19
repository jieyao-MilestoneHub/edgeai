"""
Task 03: Engineering Agent - 推論測試腳本

功能：
1. 載入訓練好的 QLoRA 模型
2. 進行文本生成推論
3. 支援三種模式：互動、演示、單次

使用套件：
- transformers: 模型載入與生成
- peft: LoRA adapter 載入

作者: Joel
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """載入訓練好的模型"""
    print(f"\n🤖 Loading model from: {model_path}")
    print(f"   Base model: {base_model}")

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 載入基礎模型
    print("   Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 載入 LoRA adapter
    print("   Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, model_path)

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("✅ Model loaded successfully!")

    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, input_text: str = "", max_new_tokens: int = 256):
    """生成回應"""
    # 構建 prompt
    prompt = "<|im_start|>system\nYou are a helpful AI assistant specialized in engineering and technical domains.<|im_end|>\n"
    prompt += f"<|im_start|>user\n{instruction}"

    if input_text:
        prompt += f"\n\n{input_text}"

    prompt += "<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 提取 assistant 回應
    assistant_start = response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    assistant_end = response.find("<|im_end|>", assistant_start)

    if assistant_end == -1:
        assistant_response = response[assistant_start:].strip()
    else:
        assistant_response = response[assistant_start:assistant_end].strip()

    return assistant_response


def demo_mode(model, tokenizer):
    """演示模式：運行預設案例"""
    print("\n" + "="*60)
    print("🎯 Demo Mode - Engineering Test Cases")
    print("="*60)

    test_cases = [
        {
            "instruction": "Explain the difference between synchronous and asynchronous circuits in digital design.",
            "input": ""
        },
        {
            "instruction": "What are the key considerations when designing a low-power IC?",
            "input": ""
        },
        {
            "instruction": "Describe the VLSI design flow from RTL to GDSII.",
            "input": ""
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}")
        print(f"{'='*60}")
        print(f"Q: {case['instruction']}")
        print(f"\nGenerating response...")

        response = generate_response(model, tokenizer, case['instruction'], case['input'])

        print(f"\nA: {response}")


def interactive_mode(model, tokenizer):
    """互動模式"""
    print("\n" + "="*60)
    print("💬 Interactive Mode")
    print("="*60)
    print("Enter your questions (type 'quit' to exit)\n")

    while True:
        instruction = input("Q: ")

        if instruction.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if not instruction.strip():
            continue

        print("\nGenerating response...")
        response = generate_response(model, tokenizer, instruction)

        print(f"\nA: {response}\n")
        print("-" * 60)


def text_mode(model, tokenizer, instruction: str, input_text: str = ""):
    """單次文本模式"""
    print("\n" + "="*60)
    print("📝 Text Mode")
    print("="*60)
    print(f"Q: {instruction}")

    if input_text:
        print(f"Input: {input_text}")

    print("\nGenerating response...")
    response = generate_response(model, tokenizer, instruction, input_text)

    print(f"\nA: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Inference with trained QLoRA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained LoRA model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "demo", "text"], help="Inference mode")
    parser.add_argument("--instruction", type=str, default="", help="Instruction for text mode")
    parser.add_argument("--input", type=str, default="", help="Input for text mode")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")

    args = parser.parse_args()

    # 載入模型
    model, tokenizer = load_model(args.model_path, args.base_model)

    # 運行對應模式
    if args.mode == "demo":
        demo_mode(model, tokenizer)
    elif args.mode == "interactive":
        interactive_mode(model, tokenizer)
    elif args.mode == "text":
        if not args.instruction:
            print("❌ Error: --instruction is required for text mode")
            return
        text_mode(model, tokenizer, args.instruction, args.input)


if __name__ == "__main__":
    main()
