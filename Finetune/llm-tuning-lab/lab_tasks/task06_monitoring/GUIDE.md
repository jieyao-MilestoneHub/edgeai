# Task 06: 訓練監控

## Prometheus + Grafana

### Metrics 收集

```python
from prometheus_client import Gauge, Counter

gpu_memory = Gauge('gpu_memory_used', 'GPU memory usage')
training_loss = Gauge('training_loss', 'Current training loss')
```

### Dashboard

Grafana 可視化：
- GPU 使用率
- Loss 曲線
- 訓練速度

詳見實作檔案
