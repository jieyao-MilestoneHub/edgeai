"""Prometheus Metrics 收集器"""
from prometheus_client import Gauge, start_http_server
import torch
import time

# 定義 metrics
gpu_memory = Gauge('gpu_memory_used_bytes', 'GPU memory usage')
training_loss = Gauge('training_loss', 'Current training loss')

def collect_metrics():
    """收集訓練 metrics"""
    while True:
        if torch.cuda.is_available():
            memory = torch.cuda.memory_allocated()
            gpu_memory.set(memory)
        time.sleep(5)

if __name__ == "__main__":
    start_http_server(8000)
    collect_metrics()
