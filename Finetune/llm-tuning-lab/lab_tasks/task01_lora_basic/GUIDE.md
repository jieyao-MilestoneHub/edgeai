# Task 01 詳細教學指引

> 手把手教你實作 LoRA - 從理論到程式碼

## 📖 導讀

本指引將帶你完成以下步驟：

1. **理解 LoRA 原理** - 數學推導與視覺化
2. **實作 LoRA 模組** - 逐行解釋程式碼
3. **整合到模型** - 應用到 Transformer
4. **訓練與評估** - 完整訓練流程
5. **實驗與分析** - 參數影響與調優

---

## 第一部分：LoRA 原理深入理解

### 1.1 為什麼需要 LoRA？

假設你要微調 GPT-2 (124M 參數)：

```python
# 全參數微調
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.train()  # 所有 124M 參數都可訓練

# 問題：
# - 記憶體：需要儲存 124M 個梯度
# - 儲存：每個任務都需要完整模型副本 (500MB+)
# - 效率：訓練速度慢，容易過擬合
```

**LoRA 解法：**

```python
# 只訓練額外的小矩陣
# 凍結原始 124M 參數
# 只訓練 ~300K 新參數 (rank=8)

# 優勢：
# ✅ 記憶體減少 4-10×
# ✅ 每個任務只需儲存 ~2MB
# ✅ 訓練更快，更不易過擬合
```

### 1.2 數學原理拆解

#### 原始線性層

```python
# 線性變換
y = Wx

其中：
- x ∈ ℝ^k: 輸入向量
- W ∈ ℝ^(d×k): 權重矩陣
- y ∈ ℝ^d: 輸出向量
```

#### 全參數微調

```python
# 訓練後權重變為
W' = W₀ + ΔW

其中：
- W₀: 預訓練權重 (凍結)
- ΔW: 訓練中學到的變化 (d×k，全部可訓練)
```

#### LoRA 低秩近似

```python
# 核心假設：ΔW 可以用低秩矩陣近似
ΔW ≈ BA

其中：
- B ∈ ℝ^(d×r): 下投影矩陣
- A ∈ ℝ^(r×k): 上投影矩陣
- r << min(d, k): rank，通常 4-64

# 參數量對比：
# 全參數：d × k
# LoRA：  d×r + r×k = r(d+k)

# 例如 d=k=1024, r=8:
# 全參數：1024 × 1024 = 1,048,576
# LoRA：  8 × (1024 + 1024) = 16,384
# 減少：  ~64 倍！
```

### 1.3 視覺化理解

```
輸入 x (batch, seq, k)
│
├─────────────────┬──────────────────┐
│                 │                  │
│ 路徑1: 基礎輸出  │  路徑2: LoRA 輸出 │
│                 │                  │
│   W₀ (凍結)      │     A (r×k)      │
│   (d×k)         │       ↓          │
│     ↓           │    x @ Aᵀ        │
│   W₀x           │   (batch,seq,r)  │
│                 │       ↓          │
│                 │     B (d×r)      │
│                 │       ↓          │
│                 │    (·) @ Bᵀ      │
│                 │   (batch,seq,d)  │
│                 │       ↓          │
│                 │   × (α/r)        │
│                 │                  │
└────────┬────────┴────────┬─────────┘
         │                 │
         └────── + ────────┘
                 │
            y = W₀x + (α/r)·BAx
```

---

## 第二部分：LoRA 模組實作

### 2.1 核心 LoRA Layer

讓我們逐步實作 `LoRALayer`：

```python
import torch
import torch.nn as nn
import math
from typing import Optional

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 層實作

    數學形式：
        h = Wx + (α/r) · BAx

    其中：
        - W: 凍結的預訓練權重
        - B, A: 可訓練的低秩矩陣
        - α: 縮放超參數
        - r: rank
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ========== 超參數儲存 ==========
        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features

        # ========== LoRA 權重定義 ==========
        # 注意維度！A 是 r×in，B 是 out×r
        # 這樣才能正確計算：x @ A^T @ B^T
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        # ========== Dropout ==========
        # 防止過擬合
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # ========== 縮放因子 ==========
        # 這很重要！確保不同 rank 的訓練穩定性
        self.scaling = alpha / rank

        # ========== 權重初始化 ==========
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化策略：
        - A: Kaiming uniform (類似預訓練權重)
        - B: 零初始化 (確保初始輸出為 0)

        為什麼 B 初始化為 0？
        - 這樣 BA 初始就是零矩陣
        - 訓練一開始，LoRA 不影響原始模型
        - 逐步學習，慢慢加入 adapter 的影響
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入 tensor，shape = (batch, seq_len, in_features)

        Returns:
            LoRA 輸出，shape = (batch, seq_len, out_features)

        計算流程：
            x → dropout → A → B → scaling → output
        """
        # 步驟 1: Dropout (可選)
        x_drop = self.dropout(x)

        # 步驟 2: 第一次投影 (降維到 rank)
        # x_drop: (B, S, in_features)
        # lora_A: (rank, in_features)
        # result: (B, S, rank)
        lora_intermediate = x_drop @ self.lora_A.T

        # 步驟 3: 第二次投影 (升維到 out_features)
        # lora_intermediate: (B, S, rank)
        # lora_B: (out_features, rank)
        # result: (B, S, out_features)
        lora_output = lora_intermediate @ self.lora_B.T

        # 步驟 4: 縮放
        lora_output = lora_output * self.scaling

        return lora_output
```

### 2.2 整合到 Linear Layer

```python
class LinearWithLoRA(nn.Module):
    """
    將 LoRA 整合到標準 Linear 層

    實現方式：
        output = Linear(x) + LoRA(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ========== 基礎 Linear 層 (凍結) ==========
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # 凍結原始權重！非常重要！
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # ========== LoRA 層 ==========
        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # ========== 標記 ==========
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播：並行計算兩條路徑

        Args:
            x: shape = (batch, seq_len, in_features)

        Returns:
            shape = (batch, seq_len, out_features)
        """
        if self.merged:
            # 如果已合併，直接使用 linear
            return self.linear(x)
        else:
            # 並行計算：基礎輸出 + LoRA 輸出
            base_output = self.linear(x)
            lora_output = self.lora(x)
            return base_output + lora_output

    def merge_weights(self):
        """
        訓練後合併權重：W' = W₀ + BA

        優點：
        - 推論時零額外開銷
        - 可以直接部署原始模型格式
        """
        if not self.merged:
            # 計算 BA
            delta_w = self.lora.lora_B @ self.lora.lora_A
            delta_w = delta_w * self.lora.scaling

            # 更新權重
            self.linear.weight.data += delta_w

            self.merged = True

    def unmerge_weights(self):
        """反向操作：將權重拆分回去"""
        if self.merged:
            delta_w = self.lora.lora_B @ self.lora.lora_A
            delta_w = delta_w * self.lora.scaling

            self.linear.weight.data -= delta_w

            self.merged = False
```

### 2.3 應用到 GPT-2

```python
from transformers import GPT2LMHeadModel
import torch.nn as nn

def apply_lora_to_gpt2(
    model: GPT2LMHeadModel,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list = None,
) -> GPT2LMHeadModel:
    """
    將 LoRA 應用到 GPT-2 模型

    Args:
        model: 預訓練的 GPT-2 模型
        rank: LoRA rank
        alpha: LoRA alpha
        target_modules: 要應用 LoRA 的模組名稱

    Returns:
        修改後的模型
    """

    # 預設對 Q, K, V, O 應用 LoRA
    if target_modules is None:
        target_modules = ['c_attn', 'c_proj']

    # 遍歷所有層
    for name, module in model.named_modules():
        # 檢查是否是目標模組
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 獲取父模組
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = model.get_submodule(parent_name)

                # 替換為 LoRA 版本
                lora_module = LinearWithLoRA(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=alpha,
                    bias=module.bias is not None,
                )

                # 複製原始權重
                lora_module.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_module.linear.bias.data = module.bias.data.clone()

                # 替換模組
                setattr(parent_module, child_name, lora_module)

    return model
```

---

## 第三部分：訓練腳本實作

### 3.1 完整訓練流程

```python
"""
train_lora_basic.py - LoRA 訓練腳本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# 假設 lora_linear.py 在同一目錄
from lora_linear import apply_lora_to_gpt2


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # 移動到 GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 前向傳播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # 語言模型自回歸任務
        )
        loss = outputs.loss

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新參數
        optimizer.step()
        scheduler.step()

        # 記錄
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """評估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def count_parameters(model):
    """計算參數量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'percentage': 100 * trainable_params / total_params,
    }


def main(args):
    # ========== 1. 設定 Device ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========== 2. 載入模型與 Tokenizer ==========
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ========== 3. 應用 LoRA ==========
    print(f"Applying LoRA (rank={args.rank}, alpha={args.alpha})...")
    model = apply_lora_to_gpt2(
        model,
        rank=args.rank,
        alpha=args.alpha,
    )
    model = model.to(device)

    # ========== 4. 顯示參數統計 ==========
    param_stats = count_parameters(model)
    print("\n" + "="*50)
    print("Parameter Statistics:")
    print(f"  Total:     {param_stats['total']:,}")
    print(f"  Trainable: {param_stats['trainable']:,}")
    print(f"  Frozen:    {param_stats['frozen']:,}")
    print(f"  Percentage: {param_stats['percentage']:.4f}%")
    print("="*50 + "\n")

    # ========== 5. 載入數據 ==========
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=args.batch_size,
    )

    # ========== 6. 設定 Optimizer 與 Scheduler ==========
    # 只優化可訓練參數（LoRA 權重）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ========== 7. 訓練循環 ==========
    print("Start training...")
    train_losses = []
    eval_losses = []

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # 訓練
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, device
        )
        train_losses.append(train_loss)

        # 評估
        eval_loss = evaluate(model, eval_dataloader, device)
        eval_losses.append(eval_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Eval Loss:  {eval_loss:.4f}")

    # ========== 8. 儲存結果 ==========
    # 8.1 儲存 LoRA 權重
    torch.save({
        'lora_state_dict': {
            name: param for name, param in model.named_parameters()
            if 'lora' in name
        },
        'config': {
            'rank': args.rank,
            'alpha': args.alpha,
            'model_name': args.model_name,
        }
    }, 'adapter_model.bin')
    print("✅ Saved adapter_model.bin")

    # 8.2 繪製 Loss 曲線
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(eval_losses, label='Eval Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LoRA Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Saved training_loss_curve.png")

    print("\n🎉 Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt2", type=str)
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--alpha", default=16.0, type=float)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--max_length", default=512, type=int)

    args = parser.parse_args()
    main(args)
```

---

## 第四部分：實驗與分析

### 4.1 驗證 LoRA 實作正確性

```python
def test_lora_layer():
    """單元測試：驗證 LoRA 層"""
    from lora_linear import LoRALayer

    # 創建測試資料
    batch_size, seq_len = 4, 10
    in_features, out_features = 512, 512
    rank = 8

    lora = LoRALayer(in_features, out_features, rank=rank)
    x = torch.randn(batch_size, seq_len, in_features)

    # 前向傳播
    output = lora(x)

    # 檢查輸出形狀
    assert output.shape == (batch_size, seq_len, out_features)

    # 檢查初始輸出為零（因為 B 初始化為 0）
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    print("✅ LoRA Layer 測試通過！")

test_lora_layer()
```

### 4.2 參數量驗證

```python
def verify_parameter_reduction():
    """驗證參數減少效果"""
    from transformers import GPT2LMHeadModel
    from lora_linear import apply_lora_to_gpt2

    # 原始模型
    original_model = GPT2LMHeadModel.from_pretrained('gpt2')
    original_params = sum(p.numel() for p in original_model.parameters())

    # LoRA 模型
    lora_model = GPT2LMHeadModel.from_pretrained('gpt2')
    lora_model = apply_lora_to_gpt2(lora_model, rank=8)
    lora_trainable = sum(
        p.numel() for p in lora_model.parameters() if p.requires_grad
    )

    # 計算減少比例
    reduction = original_params / lora_trainable

    print(f"原始參數：{original_params:,}")
    print(f"LoRA 可訓練參數：{lora_trainable:,}")
    print(f"減少：{reduction:.1f}×")

verify_parameter_reduction()
```

---

## 第五部分：常見問題除錯

### 問題 1：LoRA 沒有生效

**症狀**：訓練 loss 不下降

**除錯步驟**：

```python
# 1. 檢查 LoRA 參數是否可訓練
for name, param in model.named_parameters():
    if 'lora' in name:
        print(f"{name}: requires_grad={param.requires_grad}")  # 應該是 True

# 2. 檢查原始權重是否凍結
for name, param in model.named_parameters():
    if 'lora' not in name and 'weight' in name:
        print(f"{name}: requires_grad={param.requires_grad}")  # 應該是 False

# 3. 檢查梯度是否傳播
loss.backward()
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is None:
        print(f"⚠️ {name} 沒有梯度！")
```

### 問題 2：記憶體溢出

**解決方案**：

```python
# 方案 1：減少 batch size
train_dataloader = DataLoader(dataset, batch_size=4)  # 原本 8

# 方案 2：梯度累積
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(**batch).loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 方案 3：混合精度訓練
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(**batch)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 🎓 學習檢查清單

完成後，確認你能夠：

- [ ] 解釋 LoRA 的數學原理
- [ ] 手寫 LoRALayer 類別
- [ ] 計算參數量減少比例
- [ ] 應用 LoRA 到任意 Linear 層
- [ ] 完整訓練一個 LoRA 模型
- [ ] 繪製並分析 loss 曲線
- [ ] 調整 rank 和 alpha 超參數
- [ ] 合併 LoRA 權重到原始模型

---

<div align="center">

**恭喜完成 Task 01！🎉**

準備好挑戰下一個任務了嗎？

[Task 02: QLoRA 實戰 →](../task02_qlora/)

</div>
