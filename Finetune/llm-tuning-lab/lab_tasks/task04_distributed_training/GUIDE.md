# Task 04: 分散式訓練

## DeepSpeed ZeRO

### ZeRO-2: Optimizer State Partitioning
### ZeRO-3: Parameter Partitioning

## 配置範例

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

## 啟動

```bash
deepspeed --num_gpus=4 train_distributed.py --deepspeed ds_config.json
```

詳見實作檔案
