"""
資料準備 - 步驟 3: 合併數據集

功能：
1. 載入 OpenHermes-2.5 通用指令集
2. 載入過濾後的 ScienceQA 和 PubMedQA
3. 統一格式為 instruction-following 格式
4. 合併並分割訓練/驗證集
5. 保存最終數據集

數據格式統一為：
{
    "instruction": "指令或問題",
    "input": "額外輸入（可選）",
    "output": "期望輸出"
}

使用套件：
- datasets: 數據處理與合併

作者: Joel
"""

import os
import yaml
import random
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from tqdm import tqdm


def load_config(config_path: str = "../config.yaml"):
    """載入配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def format_openhermes(dataset, num_samples: int):
    """
    格式化 OpenHermes-2.5 數據

    原始格式：
    {
        "conversations": [
            {"from": "human", "value": "問題"},
            {"from": "gpt", "value": "回答"}
        ]
    }

    目標格式：
    {
        "instruction": "問題",
        "input": "",
        "output": "回答"
    }
    """
    print("\n" + "="*60)
    print("🔄 Formatting OpenHermes-2.5...")
    print("="*60)

    print(f"   Sampling {num_samples} from {len(dataset)} samples...")

    # 隨機取樣
    sampled = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    # 格式化函數
    def format_example(example):
        conversations = example.get('conversations', [])

        # 找到第一個 human-gpt 對話
        instruction = ""
        output = ""

        for i, conv in enumerate(conversations):
            if conv.get('from') == 'human':
                instruction = conv.get('value', '')
                # 找下一個 gpt 回覆
                if i + 1 < len(conversations) and conversations[i + 1].get('from') == 'gpt':
                    output = conversations[i + 1].get('value', '')
                break

        return {
            "instruction": instruction,
            "input": "",
            "output": output
        }

    # 應用格式化
    formatted = sampled.map(format_example, desc="Formatting OpenHermes-2.5")

    # 過濾掉空數據
    formatted = formatted.filter(
        lambda x: len(x['instruction']) > 0 and len(x['output']) > 0,
        desc="Filtering empty samples"
    )

    print(f"✅ Formatted {len(formatted)} samples")

    return formatted


def format_scienceqa(dataset_path: str):
    """
    格式化 ScienceQA 數據

    原始格式：
    {
        "question": "問題",
        "choices": ["A", "B", "C", "D"],
        "answer": 0,
        "lecture": "相關知識",
        "solution": "解答"
    }

    目標格式：
    {
        "instruction": "問題 + 選項",
        "input": "相關知識（lecture）",
        "output": "答案 + 解答"
    }
    """
    print("\n" + "="*60)
    print("🔄 Formatting ScienceQA...")
    print("="*60)

    # 載入過濾後的數據
    try:
        dataset = Dataset.load_from_disk(dataset_path)
        print(f"   Loaded {len(dataset)} samples from {dataset_path}")
    except Exception as e:
        print(f"❌ Error loading ScienceQA: {e}")
        return None

    # 格式化函數
    def format_example(example):
        question = example.get('question', '')
        choices = example.get('choices', [])
        answer_idx = example.get('answer', 0)
        lecture = example.get('lecture', '')
        solution = example.get('solution', '')

        # 構建指令（問題 + 選項）
        instruction = question
        if choices:
            instruction += "\n\nChoices:\n"
            for i, choice in enumerate(choices):
                instruction += f"{chr(65+i)}. {choice}\n"

        # 構建輸出（答案 + 解答）
        output = ""
        if choices and 0 <= answer_idx < len(choices):
            output = f"Answer: {chr(65+answer_idx)}. {choices[answer_idx]}\n\n"

        if solution:
            output += f"Explanation: {solution}"

        return {
            "instruction": instruction,
            "input": lecture if lecture else "",
            "output": output
        }

    # 應用格式化
    formatted = dataset.map(format_example, desc="Formatting ScienceQA")

    # 過濾空數據
    formatted = formatted.filter(
        lambda x: len(x['instruction']) > 0 and len(x['output']) > 0,
        desc="Filtering empty samples"
    )

    print(f"✅ Formatted {len(formatted)} samples")

    return formatted


def format_pubmedqa(dataset_path: str):
    """
    格式化 PubMedQA 數據

    原始格式：
    {
        "question": "問題",
        "context": {"contexts": ["段落1", "段落2"]},
        "long_answer": "詳細答案",
        "final_decision": "yes/no/maybe"
    }

    目標格式：
    {
        "instruction": "問題",
        "input": "上下文段落",
        "output": "決策 + 詳細答案"
    }
    """
    print("\n" + "="*60)
    print("🔄 Formatting PubMedQA...")
    print("="*60)

    # 載入過濾後的數據
    try:
        dataset = Dataset.load_from_disk(dataset_path)
        print(f"   Loaded {len(dataset)} samples from {dataset_path}")
    except Exception as e:
        print(f"❌ Error loading PubMedQA: {e}")
        return None

    # 格式化函數
    def format_example(example):
        question = example.get('question', '')
        contexts = example.get('context', {}).get('contexts', [])
        long_answer = example.get('long_answer', '')
        decision = example.get('final_decision', '')

        # 構建指令
        instruction = f"Based on the provided context, answer the following question:\n\n{question}"

        # 構建輸入（上下文）
        input_text = "\n\n".join(contexts) if contexts else ""

        # 構建輸出
        output = ""
        if decision:
            output = f"Decision: {decision.upper()}\n\n"
        if long_answer:
            output += f"Explanation: {long_answer}"

        return {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }

    # 應用格式化
    formatted = dataset.map(format_example, desc="Formatting PubMedQA")

    # 過濾空數據
    formatted = formatted.filter(
        lambda x: len(x['instruction']) > 0 and len(x['output']) > 0,
        desc="Filtering empty samples"
    )

    print(f"✅ Formatted {len(formatted)} samples")

    return formatted


def merge_and_split(datasets_list, config):
    """
    合併所有數據集並分割為訓練/驗證集

    策略：
    1. 合併所有數據集
    2. 打亂順序
    3. 按比例分割訓練/驗證集
    """
    print("\n" + "="*60)
    print("🔗 Merging datasets...")
    print("="*60)

    # 過濾掉 None
    valid_datasets = [d for d in datasets_list if d is not None]

    if not valid_datasets:
        print("❌ No valid datasets to merge!")
        return None

    # 顯示統計
    for i, dataset in enumerate(valid_datasets):
        print(f"   Dataset {i+1}: {len(dataset)} samples")

    # 合併
    merged = concatenate_datasets(valid_datasets)
    print(f"\n✅ Merged total: {len(merged)} samples")

    # 打亂
    print("   Shuffling...")
    merged = merged.shuffle(seed=42)

    # 分割
    test_size = config['data']['test_size']
    print(f"   Splitting (test_size={test_size})...")

    split_dataset = merged.train_test_split(test_size=test_size, seed=42)

    print(f"✅ Split completed:")
    print(f"   Train: {len(split_dataset['train'])} samples")
    print(f"   Test: {len(split_dataset['test'])} samples")

    return split_dataset


def save_dataset(dataset_dict, output_dir: str = "final_dataset"):
    """保存最終數據集"""
    print("\n" + "="*60)
    print("💾 Saving final dataset...")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 保存為 Hugging Face 格式
    dataset_dict.save_to_disk(str(output_path))

    print(f"✅ Dataset saved to: {output_path}")

    # 顯示範例
    print("\n📝 Sample from train set:")
    sample = dataset_dict['train'][0]
    print(f"   Instruction: {sample['instruction'][:100]}...")
    print(f"   Input: {sample['input'][:100] if sample['input'] else '(empty)'}...")
    print(f"   Output: {sample['output'][:100]}...")

    # 保存統計信息
    stats_path = output_path / "dataset_stats.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Dataset Statistics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Train samples: {len(dataset_dict['train'])}\n")
        f.write(f"Test samples: {len(dataset_dict['test'])}\n")
        f.write(f"Total samples: {len(dataset_dict['train']) + len(dataset_dict['test'])}\n\n")

        f.write("Columns:\n")
        for col in dataset_dict['train'].column_names:
            f.write(f"  - {col}\n")

    print(f"📊 Statistics saved to: {stats_path}")

    return output_path


def main():
    """主函數：合併數據集"""
    print("\n" + "="*60)
    print("🚀 Data Preparation - Step 3: Merge Datasets")
    print("="*60)

    # 載入配置
    config = load_config()

    # 1. 格式化 OpenHermes-2.5
    openhermes = load_dataset(config['data']['general_dataset'], split="train")
    openhermes_formatted = format_openhermes(openhermes, config['data']['general_samples'])

    # 2. 格式化 ScienceQA（如果存在）
    scienceqa_formatted = None
    scienceqa_path = Path("filtered_data/scienceqa_engineering")
    if scienceqa_path.exists():
        scienceqa_formatted = format_scienceqa(str(scienceqa_path))

    # 3. 格式化 PubMedQA（如果存在）
    pubmedqa_formatted = None
    pubmedqa_path = Path("filtered_data/pubmedqa_engineering")
    if pubmedqa_path.exists():
        pubmedqa_formatted = format_pubmedqa(str(pubmedqa_path))

    # 4. 合併並分割
    datasets_to_merge = [openhermes_formatted, scienceqa_formatted, pubmedqa_formatted]
    final_dataset = merge_and_split(datasets_to_merge, config)

    # 5. 保存
    if final_dataset:
        save_dataset(final_dataset)

        print("\n" + "="*60)
        print("✅ Data preparation completed!")
        print("="*60)
        print("\n💡 Next step:")
        print("   Move to: ../2_model_training/")
        print("   Run: python train_qwen_instruct.py")
    else:
        print("\n❌ Data preparation failed!")


if __name__ == "__main__":
    main()
