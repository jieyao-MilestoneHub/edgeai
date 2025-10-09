# Task 08: 自動化評測 (AutoEval)

> 建立自動化模型評測框架

## 🎯 學習目標

- ✅ 實作評測框架
- ✅ 多種評測指標（Rouge, BLEU, GPT-Eval）
- ✅ 基準測試集管理
- ✅ 評測報告生成

## 評測流程

```python
# 訓練完成後自動觸發
evaluator = AutoEvaluator(
    model=model,
    test_set="benchmarks/test.jsonl"
)

results = evaluator.evaluate()
# => {rouge-1: 0.45, bleu: 0.38, ...}

evaluator.generate_report("report.html")
```

詳見 [GUIDE.md](GUIDE.md)
