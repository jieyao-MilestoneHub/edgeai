"""
LoRA (Low-Rank Adaptation) 實作（精簡、專注、可直接整合 transformers）

本模組只負責 LoRA 的核心：
1) LoRALayer: 低秩增量路徑  y = (alpha/r) * B(Ax)
2) LinearWithLoRA: 凍結 Linear + 並聯 LoRA，支援合併/解除
3) apply_lora_to_model: 以名稱/正則/predicate 選層套用 LoRA（含常見架構預設）
4) 工具函數：只開 LoRA 參數訓練、統計參數、存/載 LoRA 權重、全模型合併/解除

作者：LLM Tuning Lab
授權：MIT
"""

from __future__ import annotations
import math
import re
from typing import Optional, List, Dict, Callable, Union, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 1) LoRA 路徑：y = (α/r) * B(Ax)
# ---------------------------
class LoRALayer(nn.Module):
    """
    LoRA 低秩增量路徑（不含基底 Linear）
    - A: (rank, in_features)   先降維
    - B: (out_features, rank)  再升維
    - scaling = alpha / rank

    NOTE:
    * A 用 Kaiming，B 用 0 初始化 => 初始增量為 0，不干擾預訓練行為
    * 前向使用 F.linear：更快、更適配 AMP
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
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)

        # 參數：A, B
        # 注意：F.linear(x, weight) 期望 weight 形狀為 (out, in)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (*, in_features) -> (*, rank)
        x = self.dropout(x)
        lx = F.linear(x, self.lora_A)               # A @ x
        # (*, rank) -> (*, out_features)
        lx = F.linear(lx, self.lora_B)              # B @ (A @ x)
        return lx * self.scaling

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"r={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}")


# --------------------------------------
# 2) 凍結 Linear + 並聯 LoRA（可合併/解除）
# --------------------------------------
class LinearWithLoRA(nn.Module):
    """
    output = Linear(x) + LoRA(x)
    - 凍結底層 Linear 的權重/偏置
    - 支援 merge/unmerge，便於部署/續訓
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        dropout: float = 0.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype, device=device)
        self.lora = LoRALayer(in_features, out_features, rank=rank, alpha=alpha, dropout=dropout)
        self._freeze_linear()
        self.merged = False

    def _freeze_linear(self):
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)
        base = self.linear(x)
        return base + self.lora(x)

    @torch.no_grad()
    def merge_weights(self):
        if self.merged:
            return
        delta = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling   # (out, in)
        # 對齊 dtype/device
        delta = delta.to(self.linear.weight.dtype).to(self.linear.weight.device)
        self.linear.weight.add_(delta)
        self.merged = True

    @torch.no_grad()
    def unmerge_weights(self):
        if not self.merged:
            return
        delta = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
        delta = delta.to(self.linear.weight.dtype).to(self.linear.weight.device)
        self.linear.weight.sub_(delta)
        self.merged = False

    def extra_repr(self) -> str:
        return (f"in={self.linear.in_features}, out={self.linear.out_features}, "
                f"bias={self.linear.bias is not None}, merged={self.merged}")


# ------------------------------------------------------
# 3) 套用 LoRA 到模型：字串 / 正則 / predicate 皆可
#    內建常見架構的預設掛點
# ------------------------------------------------------
_TargetSpec = Union[List[str], str, re.Pattern, Callable[[str, nn.Module], bool], None]

DEFAULT_TARGETS: Dict[str, List[str]] = {
    # BERT / RoBERTa / DeBERTa 系列（HF 命名）
    "bert": ["query", "key", "value", "output.dense"],
    "roberta": ["query", "key", "value", "output.dense"],
    "deberta": ["query_proj", "key_proj", "value_proj", "dense"],
    # GPT-2
    "gpt2": ["c_attn", "c_proj"],
    # LLaMA / LLaMA2 / Mistral 系列（HF 命名）
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

def _name_matches(name: str, module: nn.Module, spec: _TargetSpec) -> bool:
    if spec is None:
        return True
    if isinstance(spec, list):
        return any(tok in name for tok in spec)
    if isinstance(spec, str):
        return spec in name
    if isinstance(spec, re.Pattern):
        return bool(spec.search(name))
    if callable(spec):
        return bool(spec(name, module))
    return False


def apply_lora_to_model(
    model: nn.Module,
    target_modules: _TargetSpec = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    *,
    arch_hint: Optional[str] = None,
    skip_if_already_lora: bool = True,
) -> nn.Module:
    """
    將模型中的 nn.Linear 以 LinearWithLoRA 替換（in-place）
    - target_modules:
        * None: 對所有 Linear 套用（不建議，除非很小模型）
        * List[str]/str: 名稱包含任一字串
        * 正則: re.Pattern
        * predicate(name, module) -> bool
    - arch_hint: "bert" | "roberta" | "gpt2" | "llama" | "mistral" ...
      若未指定 target_modules，會用 DEFAULT_TARGETS[arch_hint]
    - skip_if_already_lora: 避免重複替換

    回傳 model（同一物件）
    """
    if target_modules is None and arch_hint:
        key = arch_hint.lower()
        if key in DEFAULT_TARGETS:
            target_modules = DEFAULT_TARGETS[key]

    replacements: List[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA) and skip_if_already_lora:
            continue
        if isinstance(module, nn.Linear) and _name_matches(name, module, target_modules):
            replacements.append((name, module))

    for name, lin in replacements:
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        lora_lin = LinearWithLoRA(
            in_features=lin.in_features,
            out_features=lin.out_features,
            rank=rank,
            alpha=alpha,
            bias=(lin.bias is not None),
            dropout=dropout,
            dtype=lin.weight.dtype,
            device=lin.weight.device,
        )
        # 複製底層 Linear 權重/偏置（保留 dtype/device）
        with torch.no_grad():
            lora_lin.linear.weight.copy_(lin.weight)
            if lin.bias is not None:
                lora_lin.linear.bias.copy_(lin.bias)

        setattr(parent, child_name, lora_lin)

    return model


# ---------------------------
# 4) 常用工具函式
# ---------------------------
def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """
    只開啟 LoRA 參數 requires_grad=True；其餘全部關閉。
    """
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    只取 LoRA 參數（適合存 adapter）
    """
    out: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name):
            out[name] = param.detach().clone()
    return out


@torch.no_grad()
def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor], strict: bool = False):
    """
    將 LoRA 權重載入到已套用 LoRA 的模型
    - strict=False：略過找不到的鍵，並打印提示
    """
    missing = []
    for name, tensor in lora_state.items():
        try:
            module = dict(model.named_parameters())[name]
            module.copy_(tensor.to(module.device).to(module.dtype))
        except KeyError:
            missing.append(name)
    if strict and missing:
        raise KeyError(f"Missing LoRA params in model: {missing}")
    if missing:
        print(f"[LoRA] Skipped {len(missing)} unmatched keys (strict={strict}): {missing[:3]}{'...' if len(missing)>3 else ''}")


def count_lora_parameters(model: nn.Module) -> Dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "percentage": 100.0 * trainable / total if total > 0 else 0.0,
    }


@torch.no_grad()
def merge_lora_weights(model: nn.Module):
    """
    將全模型的 LinearWithLoRA 權重合併（部署前使用）
    """
    for m in model.modules():
        if isinstance(m, LinearWithLoRA):
            m.merge_weights()


@torch.no_grad()
def unmerge_lora_weights(model: nn.Module):
    """
    解除全模型的 LinearWithLoRA 權重合併（續訓時使用）
    """
    for m in model.modules():
        if isinstance(m, LinearWithLoRA):
            m.unmerge_weights()
