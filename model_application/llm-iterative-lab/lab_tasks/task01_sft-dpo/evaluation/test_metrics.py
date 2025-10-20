"""
快速測試評估指標系統
用於驗證指標計算邏輯是否正常工作
"""

from metrics import EvaluationMetrics


def test_metrics():
    """測試評估指標計算"""
    print("=" * 60)
    print("測試評估指標系統")
    print("=" * 60)

    evaluator = EvaluationMetrics()

    # 測試案例 1: 結構化回答
    print("\n測試案例 1: 結構化回答 (LoRA 解釋)")
    response1 = """
LoRA (Low-Rank Adaptation) 是一種參數高效的微調方法。

主要原理：
1. 使用低秩分解來近似權重更新
2. 只訓練新增的適配器層
3. 保持原始模型參數凍結

這樣可以大幅減少訓練參數量，同時保持良好的性能。
    """

    metrics1 = evaluator.evaluate_response(
        response=response1,
        expected_keywords=["低秩分解", "適配器", "參數高效", "微調"],
        expected_format="structured",
        category="factual"
    )

    print("評估結果:")
    for key, value in metrics1.items():
        if value is not None:
            print(f"  {key}: {value}")

    # 測試案例 2: 代碼回答
    print("\n" + "=" * 60)
    print("測試案例 2: 代碼回答 (Python 函數)")
    response2 = """
以下是計算列表最大值的函數：

```python
def find_max(numbers):
    \"\"\"找到列表中的最大值\"\"\"
    if not numbers:
        return None

    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num

    return max_val
```

這個函數遍歷列表，記錄當前最大值。
    """

    metrics2 = evaluator.evaluate_response(
        response=response2,
        expected_keywords=["def", "for", "if", "return"],
        expected_format="code",
        category="coding"
    )

    print("評估結果:")
    for key, value in metrics2.items():
        if value is not None:
            print(f"  {key}: {value}")

    # 測試案例 3: 列表回答
    print("\n" + "=" * 60)
    print("測試案例 3: 列表回答 (過擬合原因)")
    response3 = """
模型在訓練集上表現好但測試集差的可能原因：

1. 過擬合 - 模型記住了訓練數據而非學習模式
2. 訓練/測試數據分布不一致
3. 模型過於複雜，自由度太高
4. 訓練時間過長
5. 缺乏正則化
    """

    metrics3 = evaluator.evaluate_response(
        response=response3,
        expected_keywords=["過擬合", "泛化", "數據", "正則化"],
        expected_format="list",
        category="reasoning"
    )

    print("評估結果:")
    for key, value in metrics3.items():
        if value is not None:
            print(f"  {key}: {value}")

    # 測試案例 4: 簡短回答
    print("\n" + "=" * 60)
    print("測試案例 4: 簡短回答")
    response4 = "42"

    metrics4 = evaluator.evaluate_response(
        response=response4,
        expected_keywords=[],
        expected_format="short",
        category="edge_case"
    )

    print("評估結果:")
    for key, value in metrics4.items():
        if value is not None:
            print(f"  {key}: {value}")

    # 測試案例 5: 拒答檢測
    print("\n" + "=" * 60)
    print("測試案例 5: 拒答檢測")
    response5 = "抱歉，我無法回答這個問題，因為它涉及敏感內容。"

    metrics5 = evaluator.evaluate_response(
        response=response5,
        expected_keywords=["內容", "問題"],
        expected_format="structured",
        category="factual"
    )

    print("評估結果:")
    for key, value in metrics5.items():
        if value is not None:
            print(f"  {key}: {value}")
    print(f"  拒答檢測: {'是' if metrics5['refusal_rate'] > 0 else '否'}")

    # 聚合測試
    print("\n" + "=" * 60)
    print("測試聚合指標")

    all_metrics = [metrics1, metrics2, metrics3, metrics4]
    aggregated = evaluator.aggregate_metrics(all_metrics)

    print("聚合結果:")
    for key, value in aggregated.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("✓ 所有測試完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_metrics()
