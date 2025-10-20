"""
評估指標模組
用於計算模型輸出的各項自動指標
"""

import re
import json
from typing import Dict, List, Any
from collections import Counter


class EvaluationMetrics:
    """自動評估指標計算器"""

    def __init__(self):
        self.metrics_history = []

    def evaluate_response(
        self,
        response: str,
        expected_keywords: List[str],
        expected_format: str,
        category: str
    ) -> Dict[str, Any]:
        """
        評估單個回答

        Args:
            response: 模型生成的回答
            expected_keywords: 期望出現的關鍵詞
            expected_format: 期望的格式類型
            category: 問題類別

        Returns:
            包含各項指標的字典
        """
        metrics = {
            "keyword_coverage": self._keyword_coverage(response, expected_keywords),
            "format_correctness": self._format_correctness(response, expected_format),
            "length_appropriateness": self._length_appropriateness(response, category),
            "refusal_rate": self._refusal_detection(response),
            "code_quality": self._code_quality(response) if category == "coding" else None,
            "structure_score": self._structure_score(response, expected_format),
        }

        # 計算總分（加權平均）
        weights = {
            "keyword_coverage": 0.30,
            "format_correctness": 0.25,
            "length_appropriateness": 0.15,
            "structure_score": 0.20,
            "code_quality": 0.10 if category == "coding" else 0,
        }

        # 調整權重（如果沒有 code_quality）
        if category != "coding":
            weights["keyword_coverage"] = 0.35
            weights["format_correctness"] = 0.30
            weights["structure_score"] = 0.20

        total_score = sum(
            metrics[key] * weight
            for key, weight in weights.items()
            if metrics[key] is not None
        )

        metrics["total_score"] = round(total_score, 3)

        return metrics

    def _keyword_coverage(self, response: str, keywords: List[str]) -> float:
        """計算關鍵詞覆蓋率"""
        if not keywords:
            return 1.0

        response_lower = response.lower()
        matched = sum(1 for kw in keywords if kw.lower() in response_lower)
        return round(matched / len(keywords), 3)

    def _format_correctness(self, response: str, expected_format: str) -> float:
        """檢查格式正確性"""
        format_checks = {
            "code": self._check_code_format,
            "list": self._check_list_format,
            "structured": self._check_structured_format,
            "comparison": self._check_comparison_format,
            "short": self._check_short_format,
            "narrative": self._check_narrative_format,
            "troubleshooting": self._check_troubleshooting_format,
        }

        checker = format_checks.get(expected_format, lambda x: 0.5)
        return round(checker(response), 3)

    def _check_code_format(self, response: str) -> float:
        """檢查代碼格式"""
        score = 0.0

        # 包含代碼塊標記
        if "```" in response or "def " in response or "class " in response:
            score += 0.4

        # 包含常見代碼關鍵字
        code_keywords = ["def", "class", "import", "return", "if", "for", "while"]
        if any(kw in response for kw in code_keywords):
            score += 0.3

        # 有適當的縮排（檢查多個空格開頭的行）
        if re.search(r'\n    \w+', response) or re.search(r'\n\t\w+', response):
            score += 0.3

        return min(score, 1.0)

    def _check_list_format(self, response: str) -> float:
        """檢查列表格式"""
        score = 0.0

        # 包含數字編號或項目符號
        has_numbers = bool(re.search(r'\n\s*\d+[\.\)、]', response))
        has_bullets = bool(re.search(r'\n\s*[-\*\+•]', response))

        if has_numbers or has_bullets:
            score += 0.5

        # 至少有 2 個項目
        items = re.findall(r'\n\s*[\d\-\*\+•]', response)
        if len(items) >= 2:
            score += 0.3

        # 項目之間有換行
        if '\n' in response.strip():
            score += 0.2

        return min(score, 1.0)

    def _check_structured_format(self, response: str) -> float:
        """檢查結構化格式"""
        score = 0.0

        # 包含段落分隔
        paragraphs = response.strip().split('\n\n')
        if len(paragraphs) >= 2:
            score += 0.4

        # 包含標題或強調
        if re.search(r'(^|\n)#{1,3}\s', response) or re.search(r'\*\*.+?\*\*', response):
            score += 0.3

        # 長度適中（不過短不過長）
        word_count = len(response.split())
        if 30 <= word_count <= 300:
            score += 0.3

        return min(score, 1.0)

    def _check_comparison_format(self, response: str) -> float:
        """檢查對比格式"""
        score = 0.0

        # 包含對比關鍵詞
        comparison_words = ["相比", "不同", "而", "但", "vs", "比較", "優點", "缺點"]
        if any(word in response for word in comparison_words):
            score += 0.5

        # 有結構化呈現（表格或列表）
        if "|" in response or re.search(r'\n\s*[\d\-\*]', response):
            score += 0.5

        return min(score, 1.0)

    def _check_short_format(self, response: str) -> float:
        """檢查簡短格式"""
        word_count = len(response.split())

        # 非常簡短（< 20 字）得高分
        if word_count <= 20:
            return 1.0
        elif word_count <= 50:
            return 0.7
        else:
            return 0.3

    def _check_narrative_format(self, response: str) -> float:
        """檢查敘事格式"""
        score = 0.0

        # 有連貫的句子
        sentences = re.split(r'[。.!?]', response)
        if len(sentences) >= 3:
            score += 0.5

        # 長度適中
        word_count = len(response.split())
        if 50 <= word_count <= 200:
            score += 0.5

        return min(score, 1.0)

    def _check_troubleshooting_format(self, response: str) -> float:
        """檢查故障排除格式"""
        score = 0.0

        # 包含問題分析關鍵詞
        keywords = ["原因", "可能", "解決", "檢查", "建議", "方法"]
        if any(kw in response for kw in keywords):
            score += 0.5

        # 有列表或步驟
        if re.search(r'\n\s*[\d\-\*]', response):
            score += 0.5

        return min(score, 1.0)

    def _length_appropriateness(self, response: str, category: str) -> float:
        """評估長度適當性"""
        word_count = len(response.split())
        char_count = len(response)

        # 根據類別設定期望長度範圍
        expected_ranges = {
            "factual": (30, 200),
            "coding": (50, 500),
            "reasoning": (50, 300),
            "creative": (50, 400),
            "technical": (50, 300),
            "comparison": (50, 300),
            "debug": (50, 250),
            "application": (50, 300),
            "math": (30, 200),
            "safety": (50, 250),
            "practice": (50, 250),
            "edge_case": (5, 100),
            "instruction_following": (20, 150),
        }

        min_words, max_words = expected_ranges.get(category, (30, 300))

        # 計算偏離程度
        if min_words <= word_count <= max_words:
            return 1.0
        elif word_count < min_words:
            # 過短
            ratio = word_count / min_words
            return max(0.3, ratio)
        else:
            # 過長
            ratio = max_words / word_count
            return max(0.3, ratio)

    def _refusal_detection(self, response: str) -> float:
        """檢測拒答情況（返回 0 表示正常，1 表示拒答）"""
        refusal_patterns = [
            r"我無法",
            r"我不能",
            r"抱歉.*無法",
            r"sorry.*cannot",
            r"i cannot",
            r"unable to",
            r"不適合",
            r"不應該",
        ]

        response_lower = response.lower()
        for pattern in refusal_patterns:
            if re.search(pattern, response_lower):
                return 1.0

        return 0.0

    def _code_quality(self, response: str) -> float:
        """評估代碼品質（僅用於 coding 類別）"""
        score = 0.0

        # 包含函數定義
        if "def " in response:
            score += 0.25

        # 包含註釋或文檔字串
        if "#" in response or '"""' in response or "'''" in response:
            score += 0.25

        # 包含 return 語句
        if "return" in response:
            score += 0.25

        # 有適當的結構（縮排）
        if re.search(r'\n    ', response):
            score += 0.25

        return min(score, 1.0)

    def _structure_score(self, response: str, expected_format: str) -> float:
        """評估整體結構分數"""
        score = 0.0

        # 有清晰的段落
        if '\n\n' in response:
            score += 0.3

        # 有適當的標點符號
        punctuation_count = len(re.findall(r'[。.!?,;:、]', response))
        if punctuation_count >= 3:
            score += 0.3

        # 不是純粹的流水文字（有結構性元素）
        if any(marker in response for marker in ['**', '##', '```', '-', '*', '1.', '2.']):
            score += 0.4

        return min(score, 1.0)

    def aggregate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        聚合多個問題的評估結果

        Args:
            all_metrics: 所有問題的評估結果列表

        Returns:
            聚合後的平均指標
        """
        if not all_metrics:
            return {}

        keys = ["keyword_coverage", "format_correctness", "length_appropriateness",
                "refusal_rate", "structure_score", "total_score"]

        aggregated = {}
        for key in keys:
            values = [m[key] for m in all_metrics if m.get(key) is not None]
            if values:
                aggregated[f"avg_{key}"] = round(sum(values) / len(values), 3)

        # 計算代碼品質平均（僅針對有代碼的問題）
        code_quality_values = [m["code_quality"] for m in all_metrics if m.get("code_quality") is not None]
        if code_quality_values:
            aggregated["avg_code_quality"] = round(sum(code_quality_values) / len(code_quality_values), 3)

        return aggregated


def calculate_preference_alignment(
    responses_a: List[str],
    responses_b: List[str],
    human_preferences: List[int]
) -> float:
    """
    計算偏好對齊率

    Args:
        responses_a: 模型 A 的回答列表
        responses_b: 模型 B 的回答列表
        human_preferences: 人工偏好評分 (0: A 更好, 1: B 更好, 2: 相同)

    Returns:
        對齊率（0-1）
    """
    if not human_preferences:
        return 0.0

    # 這裡可以擴展為使用自動化方法預測偏好
    # 目前僅作為人工評分的輔助
    return len([p for p in human_preferences if p in [0, 1]]) / len(human_preferences)


if __name__ == "__main__":
    # 測試示例
    evaluator = EvaluationMetrics()

    test_response = """
    LoRA (Low-Rank Adaptation) 是一種參數高效的微調方法。

    主要原理：
    1. 使用低秩分解來近似權重更新
    2. 只訓練新增的適配器層
    3. 保持原始模型參數凍結

    這樣可以大幅減少訓練參數量。
    """

    metrics = evaluator.evaluate_response(
        response=test_response,
        expected_keywords=["低秩分解", "適配器", "參數高效"],
        expected_format="structured",
        category="factual"
    )

    print("評估結果:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
