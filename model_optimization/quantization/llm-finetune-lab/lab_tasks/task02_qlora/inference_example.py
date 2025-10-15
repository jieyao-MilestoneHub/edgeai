"""
QLoRA Inference Example - Qwen2.5-3B Text Generation

示範如何使用訓練好的 QLoRA 模型進行文本生成推論

支援三種模式：
1. Interactive (互動模式): 即時輸入 prompt 並生成文本
2. Demo (範例模式): 展示預設範例的生成結果
3. Text (單次生成): 對單一 prompt 進行生成

作者: LLM Tuning Lab
授權: MIT
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from quantization_utils import create_bnb_config


# ====================================================================================
# 1) 載入模型與 LoRA Adapter
# ====================================================================================

def load_model_with_lora(
    base_model_name: str,
    adapter_dir: str,
    use_4bit: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    載入帶有 LoRA adapter 的量化模型

    Args:
        base_model_name: 基礎模型名稱（例：\"Qwen/Qwen2.5-3B-Instruct\"）
        adapter_dir: LoRA adapter 目錄
        use_4bit: 是否使用 4-bit 量化（推論時也可節省顯存）
        device: 設備（cuda/cpu）

    Returns:
        (model, tokenizer): 載入好的模型與 tokenizer

    載入流程：
        1. 載入量化的基礎模型（4-bit NF4）
        2. 套用 LoRA adapter 權重
        3. 載入 tokenizer

    範例：
        >>> model, tokenizer = load_model_with_lora(
        ...     \"Qwen/Qwen2.5-3B-Instruct\",
        ...     \"./output_qlora_qwen_3b\"
        ... )
    """
    print(f"📥 Loading base model: {base_model_name}")

    if use_4bit and torch.cuda.is_available():
        # 使用 4-bit 量化（節省顯存）
        bnb_config = create_bnb_config()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"   ✅ Base model loaded (4-bit NF4)")
    else:
        # 不使用量化（CPU 或測試用）
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        print(f"   ✅ Base model loaded (FP16)")

    # 載入 LoRA adapter
    print(f"🔧 Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    print(f"   ✅ LoRA adapter loaded")

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    # 設定 padding token（某些模型需要）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Model and tokenizer ready!")

    return model, tokenizer


# ====================================================================================
# 2) 文本生成函數
# ====================================================================================

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """
    使用模型生成文本

    Args:
        model: 模型
        tokenizer: Tokenizer
        prompt: 輸入 prompt
        max_new_tokens: 最大生成長度（默認 128）
        temperature: 溫度參數（控制隨機性，默認 0.7）
            - 越高越隨機（1.0+）
            - 越低越確定（0.1-0.5）
        top_p: Nucleus sampling 參數（默認 0.9）
            - 累積概率閾值
            - 只從累積概率達到 top_p 的 tokens 中採樣
        top_k: Top-K sampling 參數（默認 50）
            - 只從概率最高的 k 個 tokens 中採樣
        repetition_penalty: 重複懲罰（默認 1.1）
            - > 1.0: 懲罰重複
            - = 1.0: 不懲罰
        do_sample: 是否使用採樣（默認 True）
            - True: 隨機採樣（更多樣化）
            - False: Greedy decoding（更確定）

    Returns:
        生成的文本

    生成策略說明：
        - temperature: 控制分布平滑度
          * 高溫（1.0+）: 更隨機，更有創意
          * 低溫（0.1-0.5）: 更確定，更保守
        - top_p (nucleus sampling): 動態截斷
          * 0.9: 從累積概率 90% 的 tokens 中採樣
          * 更適合開放式生成
        - top_k: 固定截斷
          * 50: 只考慮概率最高的 50 個 tokens
          * 防止選到極低概率的 tokens

    範例：
        >>> text = generate_text(
        ...     model, tokenizer,
        ...     prompt=\"Explain quantum computing in simple terms.\",
        ...     temperature=0.7,
        ...     max_new_tokens=100
        ... )
    """
    # 設定模型為評估模式
    model.eval()

    # Tokenize 輸入
    inputs = tokenizer(prompt, return_tensors="pt")

    # 移到模型所在設備
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成
    with torch.inference_mode():  # 推論模式（節省記憶體）
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode 輸出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


# ====================================================================================
# 3) 互動模式
# ====================================================================================

def interactive_mode(model, tokenizer, config):
    """
    互動模式：讓使用者輸入 prompt 進行即時生成

    Args:
        model: 模型
        tokenizer: Tokenizer
        config: 配置字典（包含生成參數）

    使用說明：
        - 輸入 prompt 進行生成
        - 輸入 'quit', 'exit', 'q' 結束
        - 輸入 'config' 查看當前配置
        - 輸入 'reset' 重置配置

    範例對話：
        >>> 請輸入 prompt: Explain machine learning
        >>> [生成結果]
        >>> 請輸入 prompt: quit
        >>> 👋 Goodbye!
    """
    print("\n" + "="*60)
    print("💬 Interactive Text Generation")
    print("="*60)
    print("輸入 prompt 進行文本生成")
    print("指令:")
    print("  - 'quit', 'exit', 'q': 結束")
    print("  - 'config': 查看當前配置")
    print("  - 'reset': 重置配置為默認值")
    print("-"*60)

    # 默認配置
    gen_config = {
        'max_new_tokens': config.get('max_new_tokens', 128),
        'temperature': config.get('temperature', 0.7),
        'top_p': config.get('top_p', 0.9),
        'top_k': config.get('top_k', 50),
        'repetition_penalty': config.get('repetition_penalty', 1.1),
    }

    while True:
        try:
            prompt = input("\n請輸入 prompt: ").strip()

            # 檢查退出指令
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            # 檢查配置指令
            if prompt.lower() == 'config':
                print("\n📋 當前配置:")
                for key, value in gen_config.items():
                    print(f"   {key}: {value}")
                continue

            # 檢查重置指令
            if prompt.lower() == 'reset':
                gen_config = {
                    'max_new_tokens': 128,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.1,
                }
                print("\n✅ 配置已重置為默認值")
                continue

            # 檢查空輸入
            if not prompt:
                print("⚠️  請輸入有效的 prompt")
                continue

            # 生成
            print(f"\n🤖 生成中...")
            generated = generate_text(
                model, tokenizer, prompt,
                **gen_config
            )

            # 顯示結果
            print(f"\n{'='*60}")
            print(f"📝 生成結果:")
            print(f"{'='*60}")
            print(generated)
            print(f"{'='*60}")

            # 顯示統計
            input_length = len(tokenizer.encode(prompt))
            output_length = len(tokenizer.encode(generated))
            print(f"\n📊 統計:")
            print(f"   輸入長度: {input_length} tokens")
            print(f"   輸出長度: {output_length} tokens")
            print(f"   新生成: {output_length - input_length} tokens")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")
            print("   請重試或輸入 'quit' 結束")


# ====================================================================================
# 4) 範例模式
# ====================================================================================

def demo_mode(model, tokenizer, config):
    """
    範例模式：展示預設範例的生成結果

    Args:
        model: 模型
        tokenizer: Tokenizer
        config: 配置字典

    展示內容：
        - 多個不同類型的 prompt
        - 展示模型的生成能力
        - 涵蓋不同領域（解釋、創作、翻譯等）
    """
    print("\n" + "="*60)
    print("📝 Demo Examples")
    print("="*60)

    # 預設範例（涵蓋不同類型）
    examples = [
        {
            "prompt": "Explain quantum computing in simple terms:",
            "category": "解釋 (Explanation)"
        },
        {
            "prompt": "Write a short poem about artificial intelligence:",
            "category": "創作 (Creative Writing)"
        },
        {
            "prompt": "Translate the following to French: 'Hello, how are you?'",
            "category": "翻譯 (Translation)"
        },
        {
            "prompt": "What are the benefits of regular exercise?",
            "category": "知識問答 (Q&A)"
        },
        {
            "prompt": "Once upon a time in a distant galaxy,",
            "category": "續寫 (Story Continuation)"
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"範例 {i}/{len(examples)} - {example['category']}")
        print(f"{'='*60}")
        print(f"\n📥 Prompt:")
        print(f"{example['prompt']}")

        # 生成
        print(f"\n🤖 生成中...")
        generated = generate_text(
            model, tokenizer,
            example['prompt'],
            max_new_tokens=config.get('max_new_tokens', 128),
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.9),
        )

        # 顯示結果
        print(f"\n📤 生成結果:")
        print(f"{'-'*60}")
        print(generated)
        print(f"{'-'*60}")

        # 等待用戶（除了最後一個範例）
        if i < len(examples):
            input("\n按 Enter 繼續下一個範例...")


# ====================================================================================
# 5) 單次生成模式
# ====================================================================================

def text_mode(model, tokenizer, prompt: str, config):
    """
    單次生成模式：對單一 prompt 進行生成

    Args:
        model: 模型
        tokenizer: Tokenizer
        prompt: 輸入 prompt
        config: 配置字典

    適用場景：
        - 腳本調用
        - API 整合
        - 批次處理
    """
    print("\n" + "="*60)
    print("📝 Text Generation")
    print("="*60)

    print(f"\n📥 Prompt:")
    print(f"{prompt}")

    # 生成
    print(f"\n🤖 生成中...")
    generated = generate_text(
        model, tokenizer, prompt,
        max_new_tokens=config.get('max_new_tokens', 128),
        temperature=config.get('temperature', 0.7),
        top_p=config.get('top_p', 0.9),
    )

    # 顯示結果
    print(f"\n📤 生成結果:")
    print(f"{'='*60}")
    print(generated)
    print(f"{'='*60}")

    # 顯示統計
    input_length = len(tokenizer.encode(prompt))
    output_length = len(tokenizer.encode(generated))
    print(f"\n📊 統計:")
    print(f"   輸入長度: {input_length} tokens")
    print(f"   輸出長度: {output_length} tokens")
    print(f"   新生成: {output_length - input_length} tokens")


# ====================================================================================
# 6) 主函數
# ====================================================================================

def main(args):
    # 檢查 adapter 目錄
    adapter_path = Path(args.adapter_dir)
    if not adapter_path.exists():
        print(f"❌ Adapter directory not found: {args.adapter_dir}")
        print("   請先訓練模型: python train_qlora.py")
        return

    # 載入模型
    print("="*60)
    print("🚀 QLoRA Inference - Qwen2.5-3B")
    print("="*60)

    model, tokenizer = load_model_with_lora(
        args.model_name,
        args.adapter_dir,
        use_4bit=args.use_4bit
    )

    # 生成配置
    gen_config = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'repetition_penalty': args.repetition_penalty,
    }

    # 根據模式執行
    if args.mode == "interactive":
        interactive_mode(model, tokenizer, gen_config)

    elif args.mode == "demo":
        demo_mode(model, tokenizer, gen_config)

    elif args.mode == "text":
        if not args.prompt:
            print("❌ 請使用 --prompt 參數提供輸入文本")
            print("   範例: python inference_example.py --mode text --prompt \"Explain AI\"")
            return
        text_mode(model, tokenizer, args.prompt, gen_config)

    else:
        print(f"❌ 未知模式: {args.mode}")
        print("   可用模式: interactive, demo, text")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA Text Generation Inference"
    )

    # 模型相關
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="./output_qlora_qwen_3b",
        help="LoRA adapter directory (default: ./output_qlora_qwen_3b)"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization for inference (default: True)"
    )

    # 模式選擇
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "demo", "text"],
        help="Inference mode (default: interactive)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for text mode"
    )

    # 生成參數
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (default: 128)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)"
    )

    args = parser.parse_args()
    main(args)
