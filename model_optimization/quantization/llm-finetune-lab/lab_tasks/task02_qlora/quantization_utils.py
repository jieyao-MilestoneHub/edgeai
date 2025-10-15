"""
QLoRA 量化與 LoRA 工具函數

本模組提供 QLoRA 訓練所需的核心功能：
1) BitsAndBytesConfig: 4-bit NF4 量化配置
2) LoRAConfig: LoRA 超參數配置
3) load_model_and_tokenizer: 載入量化模型與 tokenizer
4) prepare_model_for_training: 準備 k-bit 訓練並套用 LoRA
5) print_memory_stats: 顯示 GPU 記憶體使用情況

關鍵技術：
- NF4 (4-bit NormalFloat): 針對正態分布優化的量化格式
- Double Quantization: 對量化常數再量化，額外節省 8% 記憶體
- BF16 Compute: 計算時使用 BFloat16，訓練更穩定
- Gradient Checkpointing: 節省記憶體的反向傳播技術
- Paged Optimizer: CPU-GPU 分頁管理，降低記憶體峰值

作者: LLM Tuning Lab
授權: MIT
"""

from __future__ import annotations
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ====================================================================================
# 1) 量化配置：4-bit NF4 + Double Quantization + BF16 Compute
# ====================================================================================

def create_bnb_config(
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.bfloat16,
    use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    建立 BitsAndBytes 量化配置（4-bit QLoRA 標準設定）

    Args:
        load_in_4bit: 是否使用 4-bit 量化（默認 True）
        quant_type: 量化類型，"nf4" 或 "fp4"（默認 "nf4"，針對正態分布優化）
        compute_dtype: 計算時的數據類型（默認 torch.bfloat16，訓練更穩定）
        use_double_quant: 是否啟用雙重量化（默認 True，額外節省 8% 記憶體）

    Returns:
        BitsAndBytesConfig: 量化配置對象

    關鍵設計：
        - NF4 vs FP4:
          * NF4 使用正態分布的分位數進行非均勻量化
          * 權重通常服從正態分布，NF4 精度更高（相同記憶體下提升 1.8×）
        - BF16 vs FP16:
          * BF16 範圍大（10³⁸ vs 10⁴），梯度累積不易溢位
          * FP16 精度稍高，但大模型訓練容易出現 NaN
        - Double Quantization:
          * 第一次：權重 → 4-bit
          * 第二次：量化常數 (FP32) → INT8
          * 額外節省約 8% 記憶體，精度損失 < 0.1%

    範例：
        >>> config = create_bnb_config()
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "Qwen/Qwen2.5-3B",
        ...     quantization_config=config
        ... )
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,              # 啟用 4-bit 量化
        bnb_4bit_quant_type=quant_type,         # NF4 格式（正態分布優化）
        bnb_4bit_compute_dtype=compute_dtype,   # 計算用 BF16（穩定性）
        bnb_4bit_use_double_quant=use_double_quant,  # 雙重量化（節省 8%）
    )


# ====================================================================================
# 2) LoRA 配置：Attention + MLP 層
# ====================================================================================

def create_lora_config(
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    建立 LoRA 配置（針對 LLaMA/Qwen 系列架構）

    Args:
        r: LoRA rank（低秩維度，默認 16）
            - 越大能力越強，但參數量越多
            - 建議：分類任務 8-16，生成任務 16-64
        alpha: LoRA alpha（縮放因子，默認 32.0）
            - 控制 LoRA 更新的影響強度
            - 通常設為 2×rank 或 1×rank
        dropout: LoRA dropout（默認 0.05）
            - 防止過擬合
            - 小資料集建議 0.1，大資料集 0.05
        target_modules: 要套用 LoRA 的層（默認 None，自動選擇）
            - None: 自動選擇 Attention + MLP 層
            - 可手動指定層名稱列表
        task_type: 任務類型（默認 "CAUSAL_LM"，因果語言模型）

    Returns:
        LoraConfig: LoRA 配置對象

    Target Modules 選擇策略：
        - Attention 層（必選）：
          * q_proj, k_proj, v_proj: Query/Key/Value 投影
          * o_proj: Output 投影
        - MLP 層（可選，效果提升 10-20%）：
          * gate_proj: Gate 投影（LLaMA/Qwen 架構）
          * up_proj: Up 投影
          * down_proj: Down 投影

    參數量計算：
        - 單層 LoRA 參數量 = r × (d_in + d_out)
        - 例：d=4096, r=16
          * 單個 Attention 投影：16 × (4096+4096) = 131K
          * 四個投影 (Q,K,V,O)：131K × 4 = 524K
          * 三個 MLP 投影：根據隱藏層維度計算

    範例：
        >>> config = create_lora_config(r=16, alpha=32.0)
        >>> model = get_peft_model(model, config)
    """
    # 默認使用 LLaMA/Qwen 架構的標準層
    if target_modules is None:
        target_modules = [
            # Attention 層（核心，必須包含）
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP 層（增強效果，建議包含）
            "gate_proj", "up_proj", "down_proj"
        ]

    return LoraConfig(
        r=r,                            # LoRA rank（低秩維度）
        lora_alpha=alpha,               # 縮放因子 α
        lora_dropout=dropout,           # Dropout 比例
        target_modules=target_modules,  # 目標層
        bias="none",                    # 不訓練 bias（標準做法）
        task_type=task_type,            # 任務類型
    )


# ====================================================================================
# 3) 載入量化模型與 Tokenizer
# ====================================================================================

def load_model_and_tokenizer(
    model_name: str,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    載入量化模型與 tokenizer

    Args:
        model_name: 模型名稱（Hugging Face model ID）
            - 例：\"Qwen/Qwen2.5-3B-Instruct\"
        quantization_config: 量化配置（默認 None，自動建立 4-bit NF4 配置）
        device_map: 設備映射策略（默認 "auto"，自動分配）
        trust_remote_code: 是否信任遠端程式碼（Qwen 等模型需要）

    Returns:
        (model, tokenizer): 量化模型與 tokenizer

    記憶體節省範例（Qwen2.5-3B）：
        - FP16: 6 GB
        - 4-bit NF4: 1.5 GB（節省 75%）
        - 4-bit NF4 + Double Quant: 1.4 GB（節省 77%）

    注意事項：
        - 首次執行會下載模型（約 1.5 GB for 3B model）
        - 需要網路連線
        - 模型會快取到 ~/.cache/huggingface/

    範例：
        >>> model, tokenizer = load_model_and_tokenizer(\"Qwen/Qwen2.5-3B\")
        >>> print(f\"Model loaded: {model.num_parameters()/1e9:.2f}B parameters\")
    """
    # 如果未提供量化配置，使用默認 4-bit NF4
    if quantization_config is None:
        quantization_config = create_bnb_config()

    print(f"📥 Loading model: {model_name}")
    print(f"   Quantization: 4-bit NF4 + Double Quant + BF16 Compute")

    # 載入量化模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,              # 自動分配到 GPU
        trust_remote_code=trust_remote_code,  # Qwen 等模型需要
        torch_dtype=torch.bfloat16,         # 非量化部分使用 BF16
    )

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )

    # 設定 padding 相關參數（重要！）
    tokenizer.padding_side = "right"  # 右側 padding（標準做法）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 使用 EOS 作為 PAD

    print(f"✅ Model loaded successfully!")

    return model, tokenizer


# ====================================================================================
# 4) 準備 k-bit 訓練並套用 LoRA
# ====================================================================================

def prepare_model_for_training(
    model: AutoModelForCausalLM,
    lora_config: LoraConfig,
    use_gradient_checkpointing: bool = True,
) -> AutoModelForCausalLM:
    """
    準備量化模型進行訓練並套用 LoRA

    Args:
        model: 量化後的模型
        lora_config: LoRA 配置
        use_gradient_checkpointing: 是否啟用梯度檢查點（默認 True，節省記憶體）

    Returns:
        套用 LoRA 後的 PEFT 模型

    處理流程：
        1. prepare_model_for_kbit_training: 準備量化模型訓練
           - 凍結量化權重
           - 設定輸入 requires_grad
           - 啟用梯度累積
        2. Gradient Checkpointing: 啟用梯度檢查點
           - 節省記憶體約 30-50%
           - 訓練速度略降 10-20%
           - 權衡：記憶體 vs 速度
        3. get_peft_model: 套用 LoRA 結構
           - 插入 LoRA 層到目標模組
           - 只有 LoRA 參數可訓練

    記憶體節省（Qwen2.5-3B, batch_size=1, seq_len=512）：
        - 無 Checkpointing: ~6 GB
        - 有 Checkpointing: ~4 GB（節省 33%）

    範例：
        >>> lora_config = create_lora_config(r=16)
        >>> model = prepare_model_for_training(model, lora_config)
        >>> model.print_trainable_parameters()
        trainable params: 2,097,152 || all params: 3,002,097,152 || trainable%: 0.0698
    """
    print("\n🔧 Preparing model for QLoRA training...")

    # Step 1: 準備 k-bit 訓練
    # - 凍結量化權重（4-bit 部分）
    # - 啟用輸入的梯度計算（某些架構需要）
    model = prepare_model_for_kbit_training(model)
    print("   ✅ k-bit training preparation done")

    # Step 2: 啟用梯度檢查點（節省記憶體）
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("   ✅ Gradient checkpointing enabled")

        # 某些環境需要額外設定（穩定性）
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # 設定梯度檢查點參數（PyTorch 2.0+）
        if hasattr(model, "gradient_checkpointing_kwargs"):
            model.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # Step 3: 套用 LoRA
    # - 在目標層插入 LoRA 模組
    # - 凍結所有預訓練權重
    # - 只訓練 LoRA 參數
    model = get_peft_model(model, lora_config)
    print("   ✅ LoRA adapters applied")

    # 顯示可訓練參數統計
    print("\n📊 Trainable Parameters:")
    model.print_trainable_parameters()

    return model


# ====================================================================================
# 5) GPU 記憶體統計工具
# ====================================================================================

def print_memory_stats(prefix: str = "") -> None:
    """
    顯示當前 GPU 記憶體使用情況

    Args:
        prefix: 輸出前綴字串（例如 \"After loading model\"）

    輸出格式：
        GPU Memory: Allocated=1.50 GB | Reserved=2.00 GB | Free=6.00 GB

    記憶體類型說明：
        - Allocated: 實際使用的記憶體（張量佔用）
        - Reserved: PyTorch 預留的記憶體（包含快取）
        - Free: 可用記憶體（總顯存 - Reserved）

    範例：
        >>> print_memory_stats(\"After loading model\")
        [After loading model] GPU Memory: Allocated=1.50 GB | Reserved=2.00 GB | Free=6.00 GB
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, cannot show memory stats")
        return

    # 取得記憶體統計（單位：bytes）
    allocated = torch.cuda.memory_allocated() / 1e9   # 已分配
    reserved = torch.cuda.memory_reserved() / 1e9     # 已預留
    total = torch.cuda.get_device_properties(0).total_memory / 1e9  # 總顯存
    free = total - reserved                            # 可用

    # 格式化輸出
    prefix_str = f"[{prefix}] " if prefix else ""
    print(f"{prefix_str}💾 GPU Memory: "
          f"Allocated={allocated:.2f} GB | "
          f"Reserved={reserved:.2f} GB | "
          f"Free={free:.2f} GB")


# ====================================================================================
# 6) 快捷函數：一鍵準備 QLoRA 訓練
# ====================================================================================

def setup_qlora_model(
    model_name: str,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    一鍵設置 QLoRA 模型（整合所有步驟）

    Args:
        model_name: 模型名稱
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: 目標模組列表

    Returns:
        (model, tokenizer): 準備好訓練的模型與 tokenizer

    這個函數整合了：
        1. 建立量化配置
        2. 載入量化模型與 tokenizer
        3. 建立 LoRA 配置
        4. 準備訓練並套用 LoRA
        5. 顯示記憶體統計

    範例（最簡單的使用方式）：
        >>> model, tokenizer = setup_qlora_model(\"Qwen/Qwen2.5-3B\")
        >>> # 直接開始訓練！
    """
    print("="*60)
    print("🚀 Setting up QLoRA Model")
    print("="*60)

    # 1. 載入量化模型與 tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    print_memory_stats("After loading model")

    # 2. 建立 LoRA 配置並準備訓練
    lora_config = create_lora_config(
        r=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = prepare_model_for_training(model, lora_config)
    print_memory_stats("After applying LoRA")

    print("\n" + "="*60)
    print("✅ QLoRA Model Ready for Training!")
    print("="*60)

    return model, tokenizer
