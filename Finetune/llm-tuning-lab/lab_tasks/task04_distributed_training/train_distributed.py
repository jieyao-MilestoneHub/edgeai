"""DeepSpeed 分散式訓練腳本"""
import torch
import deepspeed
from transformers import AutoModelForCausalLM

def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # DeepSpeed 初始化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="deepspeed_config.json"
    )
    
    # 訓練循環
    for step, batch in enumerate(train_dataloader):
        loss = model_engine(**batch).loss
        model_engine.backward(loss)
        model_engine.step()

if __name__ == "__main__":
    main()
