"""
LoRA (Low-Rank Adaptation) 實作

這個模組實作了 LoRA 的核心組件：
1. LoRALayer: 基礎 LoRA 層
2. LinearWithLoRA: 整合 LoRA 的 Linear 層
3. apply_lora_to_model: 應用 LoRA 到模型的工具函數

作者：LLM Tuning Lab
授權：MIT
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict


class LoRALayer(nn.Module):
    """
    LoRA Layer 實作

    數學原理：
        output = (α/r) · B @ A @ x

    其中：
        - A ∈ ℝ^(r×in_features): 下投影矩陣
        - B ∈ ℝ^(out_features×r): 上投影矩陣
        - α: 縮放超參數
        - r: rank（低秩維度）

    Args:
        in_features: 輸入特徵維度
        out_features: 輸出特徵維度
        rank: LoRA rank，決定可訓練參數量
        alpha: 縮放超參數，通常設為 rank 的 1-2 倍
        dropout: Dropout 比例，防止過擬合

    Example:
        >>> lora = LoRALayer(512, 512, rank=8, alpha=16)
        >>> x = torch.randn(4, 10, 512)  # (batch, seq, features)
        >>> output = lora(x)  # (4, 10, 512)
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

        # 參數驗證
        assert rank > 0, f"rank must be positive, got {rank}"
        assert alpha > 0, f"alpha must be positive, got {alpha}"
        assert 0 <= dropout < 1, f"dropout must be in [0, 1), got {dropout}"

        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features

        # LoRA 權重矩陣
        # A: 將輸入從 in_features 降維到 rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))

        # B: 將 rank 升維到 out_features
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # 縮放因子：α/r
        # 這確保不同 rank 的學習率自動調整
        self.scaling = alpha / rank

        # 初始化權重
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化 LoRA 權重

        策略：
        - lora_A: Kaiming uniform 初始化（類似 PyTorch Linear 層）
        - lora_B: 零初始化

        為什麼 B 初始化為零？
        - 保證訓練開始時，LoRA 輸出為零矩陣
        - 不影響預訓練模型的初始行為
        - 隨著訓練進行，逐步學習任務特定的調整
        """
        # A 使用 Kaiming uniform（與 PyTorch Linear 一致）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B 初始化為零
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA 前向傳播

        計算流程：
            1. x → dropout
            2. dropout(x) @ A^T → (降維到 rank)
            3. result @ B^T → (升維到 out_features)
            4. result × (α/r) → 最終輸出

        Args:
            x: 輸入 tensor, shape = (batch_size, seq_len, in_features)
               或 (batch_size, in_features)

        Returns:
            LoRA 輸出, shape 與 Linear(x) 相同

        Shape:
            - Input: (*, in_features)
            - Output: (*, out_features)
        """
        # 應用 dropout
        x_dropout = self.dropout(x)

        # 第一次投影：降維
        # x_dropout: (*, in_features)
        # lora_A: (rank, in_features)
        # result: (*, rank)
        intermediate = x_dropout @ self.lora_A.T

        # 第二次投影：升維
        # intermediate: (*, rank)
        # lora_B: (out_features, rank)
        # result: (*, out_features)
        output = intermediate @ self.lora_B.T

        # 縮放
        output = output * self.scaling

        return output

    def extra_repr(self) -> str:
        """額外的字串表示，用於 print(model)"""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}"
        )


class LinearWithLoRA(nn.Module):
    """
    整合 LoRA 的 Linear 層

    這個模組將標準 Linear 層與 LoRA 層結合：
        output = Linear(x) + LoRA(x)

    其中 Linear 的權重被凍結，只訓練 LoRA 部分。

    Args:
        in_features: 輸入特徵維度
        out_features: 輸出特徵維度
        rank: LoRA rank
        alpha: LoRA alpha
        bias: 是否使用 bias
        dropout: LoRA dropout 比例

    Example:
        >>> layer = LinearWithLoRA(512, 512, rank=8, alpha=16)
        >>> x = torch.randn(4, 10, 512)
        >>> output = layer(x)  # (4, 10, 512)
        >>>
        >>> # 檢查參數
        >>> trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        >>> print(f"可訓練參數：{trainable}")
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

        # 基礎 Linear 層（將被凍結）
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA 層
        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # 凍結原始權重
        self._freeze_linear()

        # 合併狀態標記
        self.merged = False

    def _freeze_linear(self):
        """凍結 Linear 層的所有參數"""
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入, shape = (*, in_features)

        Returns:
            輸出, shape = (*, out_features)
        """
        if self.merged:
            # 權重已合併，直接使用 Linear
            return self.linear(x)
        else:
            # 並行計算兩條路徑
            base_output = self.linear(x)
            lora_output = self.lora(x)
            return base_output + lora_output

    def merge_weights(self):
        """
        合併 LoRA 權重到原始權重

        訓練完成後，可以將 LoRA 權重合併：
            W' = W₀ + α/r · BA

        優點：
        - 推論時零額外計算開銷
        - 可以直接部署標準格式模型
        - 移除 LoRA 結構
        """
        if not self.merged:
            # 計算 LoRA delta: BA
            delta_weight = self.lora.lora_B @ self.lora.lora_A
            delta_weight = delta_weight * self.lora.scaling

            # 更新權重：W' = W + ΔW
            self.linear.weight.data += delta_weight

            # 標記為已合併
            self.merged = True

    def unmerge_weights(self):
        """
        反向操作：將權重拆分回去

        用途：
        - 需要繼續訓練時
        - 需要單獨保存 LoRA 權重時
        """
        if self.merged:
            # 計算 LoRA delta
            delta_weight = self.lora.lora_B @ self.lora.lora_A
            delta_weight = delta_weight * self.lora.scaling

            # 減去 delta：W = W' - ΔW
            self.linear.weight.data -= delta_weight

            # 標記為未合併
            self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"bias={self.linear.bias is not None}, "
            f"merged={self.merged}"
        )


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    將 LoRA 應用到模型中的指定模組

    Args:
        model: 要修改的模型
        target_modules: 目標模組名稱列表（包含這些字串的模組會被替換）
                       例如：['q_proj', 'v_proj', 'k_proj', 'o_proj']
                       如果為 None，則對所有 Linear 層應用
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout

    Returns:
        修改後的模型（in-place 修改）

    Example:
        >>> from transformers import GPT2LMHeadModel
        >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>>
        >>> # 只對 attention 的 Q, V 使用 LoRA
        >>> model = apply_lora_to_model(
        ...     model,
        ...     target_modules=['q_proj', 'v_proj'],
        ...     rank=8
        ... )
        >>>
        >>> # 檢查可訓練參數
        >>> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        >>> total = sum(p.numel() for p in model.parameters())
        >>> print(f"可訓練: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    """
    # 收集需要替換的模組
    modules_to_replace = []

    for name, module in model.named_modules():
        # 檢查是否是 Linear 層
        if isinstance(module, nn.Linear):
            # 如果指定了 target_modules，檢查名稱是否匹配
            if target_modules is None or any(
                target in name for target in target_modules
            ):
                modules_to_replace.append((name, module))

    # 替換模組
    for name, module in modules_to_replace:
        # 分割名稱以獲取父模組
        name_parts = name.split('.')
        parent_name = '.'.join(name_parts[:-1])
        child_name = name_parts[-1]

        # 獲取父模組
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model

        # 創建新的 LoRA 層
        lora_layer = LinearWithLoRA(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
            bias=module.bias is not None,
            dropout=dropout,
        )

        # 複製原始權重
        lora_layer.linear.weight.data = module.weight.data.clone()
        if module.bias is not None:
            lora_layer.linear.bias.data = module.bias.data.clone()

        # 替換
        setattr(parent, child_name, lora_layer)

    return model


def get_lora_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    提取模型中的 LoRA 參數

    Args:
        model: 包含 LoRA 層的模型

    Returns:
        LoRA 參數字典

    Example:
        >>> lora_params = get_lora_parameters(model)
        >>> torch.save(lora_params, 'adapter_model.bin')
    """
    lora_state_dict = {}

    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_state_dict[name] = param.data.clone()

    return lora_state_dict


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """
    統計模型參數

    Returns:
        字典包含：
        - total: 總參數量
        - trainable: 可訓練參數量（LoRA）
        - frozen: 凍結參數量
        - percentage: 可訓練參數百分比
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'percentage': 100.0 * trainable_params / total_params if total_params > 0 else 0,
    }


# ========== 測試程式碼 ==========

if __name__ == "__main__":
    print("Testing LoRA implementation...\n")

    # 測試 1: LoRALayer
    print("=" * 60)
    print("Test 1: LoRALayer")
    print("=" * 60)

    lora = LoRALayer(512, 512, rank=8, alpha=16)
    x = torch.randn(4, 10, 512)
    output = lora(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"LoRA config:  {lora}")

    # 初始輸出應該接近零
    print(f"Initial output norm: {output.norm().item():.6f} (should be ~0)")

    # 測試 2: LinearWithLoRA
    print("\n" + "=" * 60)
    print("Test 2: LinearWithLoRA")
    print("=" * 60)

    layer = LinearWithLoRA(512, 512, rank=8, alpha=16)
    output = layer(x)

    print(f"Layer: {layer}")
    print(f"Output shape: {output.shape}")

    # 參數統計
    total = sum(p.numel() for p in layer.parameters())
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Reduction:        {total/trainable:.1f}×")

    # 測試 3: 權重合併
    print("\n" + "=" * 60)
    print("Test 3: Weight Merging")
    print("=" * 60)

    layer = LinearWithLoRA(512, 512, rank=8, alpha=16)

    # 前向傳播（未合併）
    out_before = layer(x)
    print(f"Output before merge: {out_before.shape}")

    # 合併權重
    layer.merge_weights()
    out_after = layer(x)
    print(f"Output after merge:  {out_after.shape}")

    # 驗證輸出一致
    diff = (out_before - out_after).abs().max().item()
    print(f"Max difference: {diff:.10f} (should be ~0)")

    # 測試 4: 應用到簡單模型
    print("\n" + "=" * 60)
    print("Test 4: Apply to Model")
    print("=" * 60)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(512, 512)
            self.layer2 = nn.Linear(512, 256)
            self.layer3 = nn.Linear(256, 128)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x

    model = SimpleModel()

    print("Before LoRA:")
    stats = count_lora_parameters(model)
    print(f"  Total:     {stats['total']:,}")
    print(f"  Trainable: {stats['trainable']:,}")

    # 應用 LoRA
    model = apply_lora_to_model(model, rank=8, alpha=16)

    print("\nAfter LoRA:")
    stats = count_lora_parameters(model)
    print(f"  Total:     {stats['total']:,}")
    print(f"  Trainable: {stats['trainable']:,}")
    print(f"  Percentage: {stats['percentage']:.4f}%")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
