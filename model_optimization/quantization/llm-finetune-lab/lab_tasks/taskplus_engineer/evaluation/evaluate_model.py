"""
Task 03: 模型評估腳本

功能：
1. 載入測試案例
2. 對訓練好的模型進行評估
3. 計算評估指標（關鍵字匹配、ROUGE等）
4. 生成評估報告

作者: Joel
"""

import argparse
import yaml
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer


def load_test_cases(test_cases_path: str = "test_cases.yaml"):
    """載入測試案例"""
    print(f"\n📋 Loading test cases from: {test_cases_path}")

    with open(test_cases_path, 'r', encoding='utf-8') as f:
        test_cases = yaml.safe_load(f)

    # 展平所有測試案例
    all_cases = []
    for category, cases in test_cases.items():
        if isinstance(cases, list):
            for case in cases:
                case['test_category'] = category
                all_cases.append(case)

    print(f"✅ Loaded {len(all_cases)} test cases")

    return all_cases


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """載入模型"""
    print(f"\n🤖 Loading model from: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("✅ Model loaded")

    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, input_text: str = ""):
    """生成回應"""
    prompt = "<|im_start|>system\nYou are a helpful AI assistant specialized in engineering and technical domains.<|im_end|>\n"
    prompt += f"<|im_start|>user\n{instruction}"

    if input_text:
        prompt += f"\n\n{input_text}"

    prompt += "<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_start = response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    assistant_end = response.find("<|im_end|>", assistant_start)

    if assistant_end == -1:
        return response[assistant_start:].strip()
    else:
        return response[assistant_start:assistant_end].strip()


def evaluate_keywords(response: str, expected_keywords: list):
    """評估關鍵字匹配"""
    response_lower = response.lower()
    matched = []
    missing = []

    for keyword in expected_keywords:
        if keyword.lower() in response_lower:
            matched.append(keyword)
        else:
            missing.append(keyword)

    score = len(matched) / len(expected_keywords) if expected_keywords else 0

    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "total": len(expected_keywords)
    }


def run_evaluation(model, tokenizer, test_cases):
    """運行評估"""
    print("\n" + "="*60)
    print("🧪 Running Evaluation...")
    print("="*60)

    results = []

    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {case['id']} - {case['category']}")
        print(f"Q: {case['instruction'][:80]}...")

        # 生成回應
        response = generate_response(model, tokenizer, case['instruction'], case.get('input', ''))

        # 評估關鍵字
        keyword_eval = evaluate_keywords(response, case.get('expected_keywords', []))

        result = {
            "id": case['id'],
            "category": case['category'],
            "test_category": case.get('test_category', ''),
            "instruction": case['instruction'],
            "response": response,
            "keyword_score": keyword_eval['score'],
            "matched_keywords": keyword_eval['matched'],
            "missing_keywords": keyword_eval['missing'],
        }

        results.append(result)

        print(f"   Keyword score: {keyword_eval['score']:.2%} ({len(keyword_eval['matched'])}/{keyword_eval['total']} matched)")

    return results


def generate_report(results, model_name: str, output_path: str = "evaluation_report.json"):
    """生成評估報告"""
    print("\n" + "="*60)
    print("📊 Generating Report...")
    print("="*60)

    # 計算統計
    avg_keyword_score = sum(r['keyword_score'] for r in results) / len(results) if results else 0

    # 按類別統計
    category_stats = {}
    for r in results:
        cat = r['test_category']
        if cat not in category_stats:
            category_stats[cat] = {'count': 0, 'total_score': 0}

        category_stats[cat]['count'] += 1
        category_stats[cat]['total_score'] += r['keyword_score']

    for cat in category_stats:
        category_stats[cat]['avg_score'] = category_stats[cat]['total_score'] / category_stats[cat]['count']

    # 構建報告
    report = {
        "model_name": model_name,
        "total_test_cases": len(results),
        "avg_keyword_score": avg_keyword_score,
        "category_stats": category_stats,
        "results": results
    }

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Report saved to: {output_path}")

    # 顯示摘要
    print("\n" + "="*60)
    print("📈 Evaluation Summary")
    print("="*60)
    print(f"\nModel: {model_name}")
    print(f"Total test cases: {len(results)}")
    print(f"Avg keyword score: {avg_keyword_score:.2%}")

    print("\n" + "-"*60)
    print("Category breakdown:")
    for cat, stats in category_stats.items():
        print(f"  {cat}: {stats['avg_score']:.2%} ({stats['count']} cases)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model")
    parser.add_argument("--test_cases", type=str, default="test_cases.yaml", help="Test cases file")
    parser.add_argument("--output", type=str, default="evaluation_report.json", help="Output report path")

    args = parser.parse_args()

    # 載入測試案例
    test_cases = load_test_cases(args.test_cases)

    # 載入模型
    model, tokenizer = load_model(args.model_path, args.base_model)

    # 運行評估
    results = run_evaluation(model, tokenizer, test_cases)

    # 生成報告
    generate_report(results, args.model_path, args.output)

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
