"""Hybrid Trainer implementing true closed-loop: 數據收集 → SFT → DPO(條件觸發) → 評估 → 數據收集 → ..."""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from copy import deepcopy

from trl import SFTConfig, SFTTrainer as HF_SFTTrainer, DPOConfig as HF_DPOConfig, DPOTrainer as HF_DPOTrainer

from src.trainers.base_trainer import QwenTrainerBase
from src.config.schemas import ExperimentConfig
from src.callbacks.base_callback import Callback
from src.data.preference.preference_pool import PreferencePool
from src.data.instruction_processor import InstructionDataProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QwenHybridWrapper(QwenTrainerBase):
    """混合訓練包裝器：實現 SFT 為主、條件觸發 DPO 的閉環訓練。

    Named QwenHybridWrapper to avoid conflicts with TRL trainers.

    閉環流程：
    ┌─────────────────────────────────┐
    │ 迭代 N                           │
    │  1. 數據收集（加載偏好對）        │
    │  2. SFT 訓練（固定 steps）        │
    │  3. 檢查偏好池                   │
    │      └─ 若達閾值 → Micro-DPO ✅   │
    │  4. 評估模型                     │
    │  5. 判斷是否繼續                 │
    └─────────────────────────────────┘

    與 Iterative 模式的差異：
    - Iterative: 固定 SFT epochs → 完整 DPO → 重複
    - Hybrid: 小步 SFT → 條件觸發輕量 DPO → 持續循環

    Example:
        >>> config = ConfigManager.load_config("hybrid_config.yaml")
        >>> trainer = QwenHybridWrapper(config)
        >>> results = trainer.train()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        callbacks: Optional[List[Callback]] = None
    ):
        """初始化混合訓練器。

        Args:
            config: 實驗配置
            callbacks: 可選的回調函數列表
        """
        super().__init__(config, callbacks)

        # 初始化偏好池
        self.preference_pool = PreferencePool(
            batch_size=self.config.hybrid.preference_batch_size,
            quality_threshold=self.config.hybrid.quality_threshold,
            data_file=self.config.hybrid.preference_data_file
        )

        # 追蹤訓練狀態
        self.iteration_logs = []
        self.micro_dpo_runs = []
        self.total_sft_steps = 0
        self.loaded_preference_count = 0

        # SFT 訓練器（持續使用同一個實例）
        self.sft_trainer = None
        self.sft_dataset = None

    def prepare_data(self) -> Any:
        """準備 SFT 數據集（僅執行一次）。

        Returns:
            處理後的數據集
        """
        if self.sft_dataset is not None:
            return self.sft_dataset

        # 確定數據路徑
        data_path = self.config.data.train_file or "data_preparation/final_dataset"

        # 創建處理器並加載數據
        processor = InstructionDataProcessor(data_path)
        dataset = processor.load_data()
        dataset = processor.process(dataset)

        self.sft_dataset = dataset
        logger.info(f"✅ SFT 數據準備完成: {len(dataset['train'])} 訓練樣本")

        return dataset

    def _collect_preferences(self, iteration: int) -> int:
        """收集偏好對數據（閉環第一步）。

        Args:
            iteration: 當前迭代次數

        Returns:
            本次新增的偏好對數量
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📥 迭代 {iteration + 1} - 數據收集階段")
        logger.info(f"{'='*60}")

        if not self.config.hybrid.preference_data_file:
            logger.warning("⚠️  未指定偏好對數據文件，跳過數據收集")
            return 0

        pref_file = Path(self.config.hybrid.preference_data_file)
        if not pref_file.exists():
            logger.warning(f"⚠️  偏好對文件不存在: {pref_file}")
            return 0

        # 讀取文件中的總行數
        with open(pref_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        # 計算本次應加載的數量（增量加載）
        to_load = min(
            self.config.hybrid.preference_batch_size,
            total_lines - self.loaded_preference_count
        )

        if to_load <= 0:
            logger.info("   所有偏好對已加載完畢")
            return 0

        # 增量加載（跳過已加載的行）
        loaded = 0
        with open(pref_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < self.loaded_preference_count:
                    continue
                if loaded >= to_load:
                    break

                try:
                    data = json.loads(line.strip())
                    if self.preference_pool.add_pair(
                        prompt=data.get('prompt', ''),
                        chosen=data.get('chosen', ''),
                        rejected=data.get('rejected', ''),
                        confidence=data.get('confidence'),
                        source=data.get('source', pref_file.name)
                    ):
                        loaded += 1
                except Exception as e:
                    logger.warning(f"   解析偏好對失敗: {e}")

        self.loaded_preference_count += loaded
        logger.info(f"✅ 本次新增 {loaded} 筆偏好對（累計已加載 {self.loaded_preference_count}/{total_lines}）")

        return loaded

    def _run_sft_iteration(self, iteration: int) -> Dict[str, Any]:
        """執行一次 SFT 迭代訓練（閉環第二步）。

        Args:
            iteration: 當前迭代次數

        Returns:
            SFT 訓練結果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 迭代 {iteration + 1} - SFT 訓練階段")
        logger.info(f"   訓練步數: {self.config.hybrid.sft_steps_per_iteration}")
        logger.info(f"{'='*60}")

        # 首次創建 SFT trainer
        if self.sft_trainer is None:
            # 準備數據集
            dataset = self.prepare_data()

            # 創建 SFT 配置
            sft_output = Path(self.config.training.output_dir) / "sft"
            sft_output.mkdir(parents=True, exist_ok=True)

            training_args = SFTConfig(
                output_dir=str(sft_output),
                overwrite_output_dir=True,
                max_steps=self.config.hybrid.sft_steps_per_iteration,  # 使用 max_steps 而非 epochs
                per_device_train_batch_size=self.config.training.batch_size,
                gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                warmup_ratio=self.config.training.warmup_ratio,
                lr_scheduler_type=self.config.training.lr_scheduler_type,
                fp16=self.config.training.fp16,
                bf16=self.config.training.bf16,
                optim=self.config.training.optim,
                max_grad_norm=self.config.training.max_grad_norm,
                logging_steps=max(self.config.hybrid.sft_steps_per_iteration // 10, 1),
                save_steps=99999,  # 不在中途保存
                eval_steps=99999,
                eval_strategy="no",
                max_seq_length=self.config.data.max_length,
                packing=False,
                report_to="none",  # 暫時關閉 tensorboard
                seed=self.config.training.seed,
                gradient_checkpointing=self.config.training.gradient_checkpointing,
            )

            # 創建格式化函數
            processor = InstructionDataProcessor(self.config.data.train_file or "data_preparation/final_dataset")

            def formatting_func(example):
                return processor.format_example(example)

            # 創建 HF SFTTrainer
            self.sft_trainer = HF_SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=None,
                processing_class=self.tokenizer,
                formatting_func=formatting_func,
            )

        else:
            # 更新訓練步數（繼續訓練）
            self.sft_trainer.args.max_steps = self.total_sft_steps + self.config.hybrid.sft_steps_per_iteration

        # 執行訓練
        train_result = self.sft_trainer.train(resume_from_checkpoint=False)

        # 更新總步數
        self.total_sft_steps += self.config.hybrid.sft_steps_per_iteration

        results = {
            "iteration": iteration + 1,
            "steps": self.config.hybrid.sft_steps_per_iteration,
            "total_steps": self.total_sft_steps,
            "train_loss": train_result.training_loss,
        }

        logger.info(f"✅ SFT 訓練完成 - Loss: {results['train_loss']:.4f}")

        return results

    def _run_micro_dpo(self, iteration: int, preference_batch: List[Dict]) -> Dict[str, Any]:
        """執行微型 DPO 訓練（閉環第三步 - 條件觸發）。

        Args:
            iteration: 當前迭代次數
            preference_batch: 偏好對批次

        Returns:
            Micro-DPO 訓練結果
        """
        run_id = len(self.micro_dpo_runs) + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 迭代 {iteration + 1} - Micro-DPO 階段（第 {run_id} 次觸發）")
        logger.info(f"   偏好對數量: {len(preference_batch)}")
        logger.info(f"   訓練步數: {self.config.hybrid.micro_dpo_steps}")
        logger.info(f"{'='*60}")

        # 保存偏好批次到臨時文件
        temp_data_dir = Path(self.config.training.output_dir) / f"micro_dpo_{run_id}"
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        temp_data_file = temp_data_dir / "preference_batch.jsonl"

        with open(temp_data_file, 'w', encoding='utf-8') as f:
            for pair in preference_batch:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        # 創建 DPO 配置
        training_args = HF_DPOConfig(
            output_dir=str(temp_data_dir),
            overwrite_output_dir=True,
            max_steps=self.config.hybrid.micro_dpo_steps,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.hybrid.micro_dpo_lr,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=0.05,  # 較短的 warmup
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            optim=self.config.training.optim,
            max_grad_norm=self.config.training.max_grad_norm,
            logging_steps=max(self.config.hybrid.micro_dpo_steps // 5, 1),
            save_steps=99999,
            eval_steps=99999,
            eval_strategy="no",
            beta=self.config.dpo.beta,
            label_smoothing=self.config.dpo.label_smoothing,
            loss_type=self.config.dpo.loss_type,
            max_length=self.config.dpo.max_length,
            max_prompt_length=self.config.dpo.max_prompt_length,
            report_to="none",
            seed=self.config.training.seed,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
        )

        # 加載偏好數據
        from src.data.preference_processor import PreferenceDataProcessor
        processor = PreferenceDataProcessor(str(temp_data_file))
        dataset = processor.load_data()
        dataset = processor.process(dataset)
        train_dataset = dataset['train'] if isinstance(dataset, dict) else dataset

        # 創建 DPO Trainer
        dpo_trainer = HF_DPOTrainer(
            model=self.sft_trainer.model,  # 使用當前 SFT 模型
            ref_model=None,  # 自動創建 reference model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            processing_class=self.tokenizer,
        )

        # 執行訓練
        train_result = dpo_trainer.train()

        # 記錄結果
        results = {
            "run_id": run_id,
            "iteration": iteration + 1,
            "num_pairs": len(preference_batch),
            "steps": self.config.hybrid.micro_dpo_steps,
            "train_loss": train_result.training_loss,
        }

        self.micro_dpo_runs.append(results)
        logger.info(f"✅ Micro-DPO 完成 - Loss: {results['train_loss']:.4f}")

        return results

    def _check_and_trigger_dpo(self, iteration: int) -> Optional[Dict[str, Any]]:
        """檢查偏好池並在條件滿足時觸發 DPO（閉環第三步）。

        Args:
            iteration: 當前迭代次數

        Returns:
            DPO 結果（若觸發），否則 None
        """
        stats = self.preference_pool.get_statistics()

        logger.info(f"\n📊 偏好池狀態檢查:")
        logger.info(f"   當前數量: {stats['current_size']}/{stats['batch_size']}")
        logger.info(f"   累計加載: {stats['total_added']}")
        logger.info(f"   過濾數量: {stats['total_filtered']}")

        if not self.preference_pool.is_ready():
            logger.info("   ⏸️  未達觸發閾值，繼續下一迭代")
            return None

        # 觸發 Micro-DPO
        logger.info("\n" + "🔔"*30)
        logger.info("偏好池達到閾值 - 觸發 Micro-DPO！")
        logger.info("🔔"*30)

        preference_batch = self.preference_pool.drain()
        dpo_results = self._run_micro_dpo(iteration, preference_batch)

        return dpo_results

    def _evaluate_iteration(self, iteration: int) -> Dict[str, Any]:
        """評估當前迭代的模型（閉環第四步）。

        Args:
            iteration: 當前迭代次數

        Returns:
            評估結果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 迭代 {iteration + 1} - 評估階段")
        logger.info(f"{'='*60}")

        # TODO: 實現真實評估邏輯
        # 目前返回佔位符結果
        eval_results = {
            "iteration": iteration + 1,
            "eval_score": 0.0,  # 佔位符
        }

        logger.info(f"   評估分數: {eval_results['eval_score']:.4f}")

        return eval_results

    def _should_continue(self, iteration: int, eval_results: Dict[str, Any]) -> bool:
        """判斷是否應該繼續訓練（閉環第五步）。

        Args:
            iteration: 當前迭代次數
            eval_results: 評估結果

        Returns:
            True 表示繼續訓練
        """
        # 檢查是否達到最大迭代次數
        if iteration + 1 >= self.config.hybrid.num_iterations:
            logger.info(f"✅ 已達最大迭代次數 ({self.config.hybrid.num_iterations})")
            return False

        # 檢查評估閾值
        eval_score = eval_results.get('eval_score', 0.0)
        if eval_score >= self.config.hybrid.evaluation_threshold:
            logger.info(f"✅ 評估分數達標 ({eval_score:.4f} >= {self.config.hybrid.evaluation_threshold})")
            return False

        logger.info(f"➡️  繼續下一迭代...")
        return True

    def _train_impl(self) -> Dict[str, Any]:
        """執行混合訓練的閉環邏輯。

        Closed Loop 流程：
        for iteration in range(num_iterations):
            1. 數據收集（加載偏好對）
            2. SFT 訓練（固定 steps）
            3. 檢查偏好池 → 條件觸發 DPO
            4. 評估模型
            5. 判斷是否繼續

        Returns:
            完整訓練結果
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"# 🔄 Hybrid Closed-Loop Training")
        logger.info(f"#   迭代次數: {self.config.hybrid.num_iterations}")
        logger.info(f"#   SFT 步數/迭代: {self.config.hybrid.sft_steps_per_iteration}")
        logger.info(f"#   DPO 觸發閾值: {self.config.hybrid.preference_batch_size} 筆")
        logger.info(f"#   Micro-DPO 步數: {self.config.hybrid.micro_dpo_steps}")
        logger.info(f"{'#'*60}\n")

        # 閉環迭代
        for iteration in range(self.config.hybrid.num_iterations):
            logger.info(f"\n{'█'*60}")
            logger.info(f"█  迭代 {iteration + 1}/{self.config.hybrid.num_iterations}")
            logger.info(f"{'█'*60}")

            iteration_log = {"iteration": iteration + 1}

            # 步驟 1: 數據收集
            new_prefs = self._collect_preferences(iteration)
            iteration_log["new_preferences"] = new_prefs

            # 步驟 2: SFT 訓練
            sft_results = self._run_sft_iteration(iteration)
            iteration_log["sft"] = sft_results

            # 步驟 3: 檢查並條件觸發 DPO
            dpo_results = self._check_and_trigger_dpo(iteration)
            iteration_log["dpo_triggered"] = dpo_results is not None
            if dpo_results:
                iteration_log["dpo"] = dpo_results

            # 步驟 4: 評估
            if self.config.hybrid.evaluation_interval > 0 and (iteration + 1) % self.config.hybrid.evaluation_interval == 0:
                eval_results = self._evaluate_iteration(iteration)
                iteration_log["evaluation"] = eval_results
            else:
                eval_results = {"eval_score": 0.0}

            # 保存迭代日誌
            self.iteration_logs.append(iteration_log)
            self._save_iteration_log(iteration, iteration_log)

            # 步驟 5: 判斷是否繼續
            if not self._should_continue(iteration, eval_results):
                break

        # 編譯最終結果
        results = {
            "total_iterations": len(self.iteration_logs),
            "total_sft_steps": self.total_sft_steps,
            "num_micro_dpo_runs": len(self.micro_dpo_runs),
            "iterations": self.iteration_logs,
            "micro_dpo_runs": self.micro_dpo_runs,
            "preference_pool_stats": self.preference_pool.get_statistics(),
        }

        # 列印總結
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Hybrid Training 總結")
        logger.info(f"{'='*60}")
        logger.info(f"   完成迭代數: {results['total_iterations']}")
        logger.info(f"   總 SFT 步數: {results['total_sft_steps']}")
        logger.info(f"   Micro-DPO 觸發次數: {results['num_micro_dpo_runs']}")
        if self.micro_dpo_runs:
            avg_dpo_loss = sum(r['train_loss'] for r in self.micro_dpo_runs) / len(self.micro_dpo_runs)
            logger.info(f"   平均 DPO Loss: {avg_dpo_loss:.4f}")
        logger.info(f"{'='*60}\n")

        return results

    def _save_iteration_log(self, iteration: int, log: Dict[str, Any]) -> None:
        """保存迭代日誌。

        Args:
            iteration: 迭代編號
            log: 日誌字典
        """
        log_dir = Path(self.config.training.output_dir) / "iterations"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"iteration_{iteration + 1}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        logger.debug(f"📝 迭代日誌已保存: {log_file}")
