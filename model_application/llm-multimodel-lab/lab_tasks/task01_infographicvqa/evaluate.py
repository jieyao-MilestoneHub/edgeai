#!/usr/bin/env python3
"""
InfographicVQA評估腳本 - 計算ANLS分數

使用方法:
python evaluate.py \\
  --model_path ./outputs/qwen_vl \\
  --dataset_split validation \\
  --output_file results.json
"""

import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from datasets import load_dataset
from Levenshtein import distance
from tqdm import tqdm
import numpy as np
import json
from typing import List, Dict, Tuple


def calculate_anls(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    計算ANLS分數

    Args:
        prediction: 模型預測答案
        ground_truths: 參考答案列表
        threshold: NLS閾值,低於此值視為0

    Returns:
        ANLS分數 (0-1之間)
    """
    if not ground_truths:
        return 0.0

    max_score = 0.0

    for gt in ground_truths:
        # 正規化
        pred = prediction.lower().strip()
        gt = gt.lower().strip()

        # 特殊情況:都是空字串
        if len(pred) == 0 and len(gt) == 0:
            return 1.0

        # 空字串情況
        if len(pred) == 0 or len(gt) == 0:
            continue

        # 計算編輯距離
        lev_dist = distance(pred, gt)
        max_len = max(len(pred), len(gt))

        # NLS分數
        nls = 1 - (lev_dist / max_len)

        # 應用閾值
        score = nls if nls >= threshold else 0.0
        max_score = max(max_score, score)

    return max_score


def predict(model, processor, image, question: str, max_new_tokens: int = 128) -> str:
    """
    對單個樣本進行預測
    """
    # 構建對話
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {question}\nAnswer:"}
            ]
        }
    ]

    # 處理輸入
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    # 生成答案
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            temperature=None,
            top_p=None,
        )

    # 解碼
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    answer = processor.decode(generated_ids, skip_special_tokens=True)

    return answer.strip()


def analyze_errors(results: List[Dict]) -> Dict:
    """
    分析錯誤類型
    """
    total = len(results)
    perfect_match = sum(1 for r in results if r['anls'] == 1.0)
    partial_match = sum(1 for r in results if 0.5 <= r['anls'] < 1.0)
    no_match = sum(1 for r in results if r['anls'] == 0.0)

    # 找出最差的案例
    worst_cases = sorted(results, key=lambda x: x['anls'])[:10]

    return {
        'total_samples': total,
        'perfect_match': {
            'count': perfect_match,
            'percentage': perfect_match / total * 100
        },
        'partial_match': {
            'count': partial_match,
            'percentage': partial_match / total * 100
        },
        'no_match': {
            'count': no_match,
            'percentage': no_match / total * 100
        },
        'worst_cases': [
            {
                'question': case['question'],
                'prediction': case['prediction'],
                'ground_truth': case['ground_truth'],
                'anls': case['anls']
            }
            for case in worst_cases
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="InfographicVQA評估")
    parser.add_argument("--model_path", type=str, required=True, help="微調後的模型路徑")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="基礎模型名稱")
    parser.add_argument("--dataset_name", type=str, default="MMInstruction/InfographicVQA",
                        help="資料集名稱")
    parser.add_argument("--dataset_split", type=str, default="validation",
                        choices=['train', 'validation', 'test'],
                        help="評估哪個split")
    parser.add_argument("--max_samples", type=int, default=None, help="最大評估樣本數(用於快速測試)")
    parser.add_argument("--output_file", type=str, default="results.json", help="輸出檔案")
    parser.add_argument("--analyze_errors", action="store_true", help="進行錯誤分析")

    args = parser.parse_args()

    print("=" * 60)
    print("InfographicVQA 評估")
    print("=" * 60)
    print(f"模型路徑: {args.model_path}")
    print(f"資料集: {args.dataset_name}")
    print(f"評估split: {args.dataset_split}")
    print("=" * 60)

    # ===== 1. 載入模型 =====
    print("\n[1/3] 載入模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 載入base model
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 載入LoRA adapters
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()
    print("✓ 模型載入完成")

    # 載入processor
    processor = Qwen2VLProcessor.from_pretrained(args.model_path)

    # ===== 2. 載入資料集 =====
    print(f"\n[2/3] 載入資料集: {args.dataset_split}...")
    dataset = load_dataset(args.dataset_name)
    eval_data = dataset[args.dataset_split]

    if args.max_samples:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    print(f"✓ 評估樣本數: {len(eval_data)}")

    # ===== 3. 評估 =====
    print("\n[3/3] 開始評估...")
    results = []
    scores = []

    for idx, sample in enumerate(tqdm(eval_data, desc="評估進度")):
        try:
            # 預測
            prediction = predict(
                model, processor,
                sample['image'],
                sample['question']
            )

            # 計算ANLS
            ground_truths = sample['answers'] if isinstance(sample['answers'], list) else [sample['answers']]
            anls_score = calculate_anls(prediction, ground_truths)

            scores.append(anls_score)
            results.append({
                'index': idx,
                'question': sample['question'],
                'prediction': prediction,
                'ground_truth': ground_truths[0],
                'all_ground_truths': ground_truths,
                'anls': anls_score
            })

        except Exception as e:
            print(f"\n警告: 樣本 {idx} 評估失敗: {e}")
            continue

    # ===== 4. 計算統計 =====
    avg_anls = np.mean(scores)
    median_anls = np.median(scores)
    std_anls = np.std(scores)

    print("\n" + "=" * 60)
    print("評估結果")
    print("=" * 60)
    print(f"平均ANLS:  {avg_anls:.4f}")
    print(f"中位數ANLS: {median_anls:.4f}")
    print(f"標準差:    {std_anls:.4f}")
    print(f"\n完全匹配(1.0):      {(np.array(scores) == 1.0).sum():4d} ({(np.array(scores) == 1.0).sum() / len(scores) * 100:.1f}%)")
    print(f"部分匹配(0.5-1.0): {((np.array(scores) >= 0.5) & (np.array(scores) < 1.0)).sum():4d} ({((np.array(scores) >= 0.5) & (np.array(scores) < 1.0)).sum() / len(scores) * 100:.1f}%)")
    print(f"不匹配(0):         {(np.array(scores) == 0.0).sum():4d} ({(np.array(scores) == 0.0).sum() / len(scores) * 100:.1f}%)")
    print("=" * 60)

    # ===== 5. 錯誤分析 =====
    if args.analyze_errors:
        print("\n錯誤分析:")
        error_analysis = analyze_errors(results)

        print("\n最差的10個預測:")
        for i, case in enumerate(error_analysis['worst_cases'], 1):
            print(f"\n{i}. ANLS={case['anls']:.3f}")
            print(f"   問題: {case['question']}")
            print(f"   預測: {case['prediction']}")
            print(f"   正確: {case['ground_truth']}")

    # ===== 6. 儲存結果 =====
    output = {
        'model_path': args.model_path,
        'dataset': args.dataset_name,
        'split': args.dataset_split,
        'num_samples': len(eval_data),
        'metrics': {
            'avg_anls': float(avg_anls),
            'median_anls': float(median_anls),
            'std_anls': float(std_anls)
        },
        'predictions': results
    }

    if args.analyze_errors:
        output['error_analysis'] = analyze_errors(results)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 結果已儲存到: {args.output_file}")


if __name__ == "__main__":
    main()
