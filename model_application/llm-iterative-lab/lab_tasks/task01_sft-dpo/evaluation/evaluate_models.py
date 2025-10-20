"""
三階段模型評估腳本
評估 Pre-trained / SFT / DPO 模型在測試集上的表現
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

# 添加專案根目錄到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.metrics import EvaluationMetrics


class ModelEvaluator:
    """三階段模型評估器"""

    def __init__(
        self,
        base_model_path: str,
        sft_model_path: str = None,
        dpo_model_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化評估器

        Args:
            base_model_path: 基礎模型路徑
            sft_model_path: SFT 模型路徑（LoRA adapter）
            dpo_model_path: DPO 模型路徑（LoRA adapter）
            device: 計算裝置
        """
        self.base_model_path = base_model_path
        self.sft_model_path = sft_model_path
        self.dpo_model_path = dpo_model_path
        self.device = device
        self.evaluator = EvaluationMetrics()

        print(f"使用裝置: {self.device}")

    def load_model(self, stage: str):
        """
        載入指定階段的模型

        Args:
            stage: 模型階段 ('pretrained', 'sft', 'dpo')

        Returns:
            model, tokenizer
        """
        print(f"\n載入 {stage} 模型...")

        # 載入 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True
        )

        # 載入基礎模型
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        # 根據階段載入適配器
        if stage == "sft" and self.sft_model_path:
            print(f"  載入 SFT adapter: {self.sft_model_path}")
            model = PeftModel.from_pretrained(model, self.sft_model_path)
        elif stage == "dpo" and self.dpo_model_path:
            print(f"  載入 DPO adapter: {self.dpo_model_path}")
            model = PeftModel.from_pretrained(model, self.dpo_model_path)

        model.eval()
        print(f"  ✓ {stage} 模型載入完成")

        return model, tokenizer

    def generate_response(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        生成模型回答

        Args:
            model: 模型
            tokenizer: tokenizer
            prompt: 輸入提示
            max_new_tokens: 最大生成 token 數
            temperature: 溫度參數
            top_p: top-p 採樣參數

        Returns:
            生成的回答
        """
        # 構建對話格式（根據 Qwen 格式）
        messages = [
            {"role": "system", "content": "你是一個專業的 AI 助手,請根據問題提供準確、清晰的回答。"},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 解碼（只取新生成的部分）
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def evaluate_stage(
        self,
        stage: str,
        test_questions: List[Dict[str, Any]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        評估單個階段的模型

        Args:
            stage: 模型階段
            test_questions: 測試問題列表
            output_dir: 輸出目錄

        Returns:
            評估結果
        """
        print(f"\n{'='*60}")
        print(f"開始評估 {stage.upper()} 模型")
        print(f"{'='*60}")

        # 載入模型
        model, tokenizer = self.load_model(stage)

        # 準備結果存儲
        all_responses = []
        all_metrics = []

        # 逐題評估
        for i, question in enumerate(test_questions, 1):
            q_id = question["id"]
            prompt = question["prompt"]
            category = question["category"]

            print(f"\n[{i}/{len(test_questions)}] {q_id} ({category})")
            print(f"  問題: {prompt[:50]}...")

            # 生成回答
            try:
                response = self.generate_response(model, tokenizer, prompt)
                print(f"  回答: {response[:80]}...")
            except Exception as e:
                print(f"  ✗ 生成失敗: {e}")
                response = f"[ERROR] {str(e)}"

            # 計算指標
            metrics = self.evaluator.evaluate_response(
                response=response,
                expected_keywords=question.get("expected_keywords", []),
                expected_format=question.get("expected_format", "structured"),
                category=category
            )

            print(f"  總分: {metrics['total_score']:.3f}")

            # 保存結果
            all_responses.append({
                "question_id": q_id,
                "prompt": prompt,
                "response": response,
                "category": category,
                "metrics": metrics
            })

            all_metrics.append(metrics)

        # 釋放模型記憶體
        del model, tokenizer
        torch.cuda.empty_cache()

        # 聚合指標
        aggregated = self.evaluator.aggregate_metrics(all_metrics)

        # 保存結果
        self._save_results(stage, all_responses, aggregated, output_dir)

        print(f"\n{'='*60}")
        print(f"{stage.upper()} 模型評估完成")
        print(f"平均總分: {aggregated.get('avg_total_score', 0):.3f}")
        print(f"{'='*60}")

        return {
            "stage": stage,
            "responses": all_responses,
            "aggregated_metrics": aggregated
        }

    def _save_results(
        self,
        stage: str,
        responses: List[Dict],
        aggregated: Dict,
        output_dir: Path
    ):
        """保存評估結果"""
        # 保存詳細回答
        responses_file = output_dir / "responses" / f"{stage}_responses.json"
        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)

        # 保存聚合指標
        metrics_file = output_dir / "metrics" / f"{stage}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)

        print(f"  ✓ 結果已保存:")
        print(f"    - {responses_file}")
        print(f"    - {metrics_file}")

    def compare_stages(
        self,
        test_questions: List[Dict[str, Any]],
        output_dir: Path,
        stages: List[str] = ["pretrained", "sft", "dpo"]
    ) -> Dict[str, Any]:
        """
        比較多個階段的模型

        Args:
            test_questions: 測試問題列表
            output_dir: 輸出目錄
            stages: 要評估的階段列表

        Returns:
            比較結果
        """
        comparison_results = {}

        for stage in stages:
            if stage == "pretrained" or \
               (stage == "sft" and self.sft_model_path) or \
               (stage == "dpo" and self.dpo_model_path):
                result = self.evaluate_stage(stage, test_questions, output_dir)
                comparison_results[stage] = result
            else:
                print(f"\n跳過 {stage} 階段（模型路徑未提供）")

        # 保存比較結果
        comparison_file = output_dir / "metrics" / "comparison_summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "stages": list(comparison_results.keys()),
            "comparison": {
                stage: data["aggregated_metrics"]
                for stage, data in comparison_results.items()
            }
        }

        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 比較結果已保存: {comparison_file}")

        return comparison_results


def load_test_suite(test_file: Path) -> List[Dict[str, Any]]:
    """載入測試問題集"""
    questions = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def main():
    parser = argparse.ArgumentParser(description="評估三階段模型")

    parser.add_argument(
        "--test-set",
        type=str,
        default="evaluation/test_suite.jsonl",
        help="測試問題集路徑"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="基礎模型路徑"
    )

    parser.add_argument(
        "--sft-model",
        type=str,
        default=None,
        help="SFT 模型路徑（LoRA adapter）"
    )

    parser.add_argument(
        "--dpo-model",
        type=str,
        default=None,
        help="DPO 模型路徑（LoRA adapter）"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results",
        help="結果輸出目錄"
    )

    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=["pretrained", "sft", "dpo"],
        help="要評估的階段"
    )

    args = parser.parse_args()

    # 準備路徑
    test_file = Path(args.test_set)
    output_dir = Path(args.output)

    # 確保輸出目錄存在
    (output_dir / "responses").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # 載入測試集
    print("載入測試問題集...")
    test_questions = load_test_suite(test_file)
    print(f"✓ 載入 {len(test_questions)} 個測試問題")

    # 創建評估器
    evaluator = ModelEvaluator(
        base_model_path=args.base_model,
        sft_model_path=args.sft_model,
        dpo_model_path=args.dpo_model
    )

    # 執行評估
    evaluator.compare_stages(
        test_questions=test_questions,
        output_dir=output_dir,
        stages=args.stages
    )

    print("\n✓ 所有評估完成!")
    print(f"  結果保存在: {output_dir}")
    print("\n下一步:")
    print("  1. 查看詳細結果: evaluation/results/responses/")
    print("  2. 查看指標統計: evaluation/results/metrics/")
    print("  3. 生成可視化報告:")
    print("     python evaluation/visualize_results.py")


if __name__ == "__main__":
    main()
