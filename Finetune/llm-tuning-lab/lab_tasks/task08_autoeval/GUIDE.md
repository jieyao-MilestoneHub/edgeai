# Task 08: 自動化評測

## 評測框架

```python
from evaluate import load

# Rouge, BLEU, etc.
rouge = load("rouge")
results = rouge.compute(predictions=preds, references=refs)
```

## 自動化流程

訓練完成 → 自動評測 → 生成報告

詳見實作檔案
