"""
資料準備 - 步驟 2: 過濾工程相關數據

功能：
1. 從 ScienceQA 和 PubMedQA 過濾 engineering 相關問題
2. 使用關鍵字匹配
3. 控制取樣數量
4. 保存過濾後的數據

使用套件：
- datasets: 數據處理
- re: 正則表達式匹配

作者: Joel
"""

import os
import re
import yaml
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm


def load_config(config_path: str = "../config.yaml"):
    """載入配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def check_engineering_relevance(text: str, keywords: list) -> bool:
    """
    檢查文本是否與工程相關

    Args:
        text: 要檢查的文本
        keywords: 關鍵字列表

    Returns:
        bool: 是否包含工程相關關鍵字
    """
    if not text:
        return False

    # 轉換為小寫以進行不區分大小寫的匹配
    text_lower = text.lower()

    # 檢查是否包含任何關鍵字
    for keyword in keywords:
        if re.search(r'\b' + keyword.lower() + r'\b', text_lower):
            return True

    return False


def filter_scienceqa(config):
    """
    過濾 ScienceQA 中的工程相關問題

    ScienceQA 包含 subject 欄位，可以用來過濾：
    - 'natural science': 自然科學
    - 'social science': 社會科學（可能包含工程管理）

    也會檢查 question 和 lecture 欄位中的關鍵字
    """
    print("\n" + "="*60)
    print("🔍 Filtering ScienceQA for engineering content...")
    print("="*60)

    # 載入數據集
    dataset_name = config['data']['tech_dataset_1']
    dataset = load_dataset(dataset_name, split="train")
    print(f"   Original samples: {len(dataset)}")

    # 過濾關鍵字
    keywords = config['data']['filter_keywords']
    print(f"   Filter keywords: {', '.join(keywords)}")

    # 過濾函數
    def is_engineering_related(example):
        # 檢查問題
        if check_engineering_relevance(example.get('question', ''), keywords):
            return True

        # 檢查講義內容
        if check_engineering_relevance(example.get('lecture', ''), keywords):
            return True

        # 檢查解答
        if check_engineering_relevance(example.get('solution', ''), keywords):
            return True

        return False

    # 應用過濾
    print("   Filtering...")
    filtered = dataset.filter(is_engineering_related, desc="Filtering ScienceQA")

    print(f"✅ Filtered samples: {len(filtered)}")
    print(f"   Retention rate: {len(filtered)/len(dataset)*100:.2f}%")

    return filtered


def filter_pubmedqa(config):
    """
    過濾 PubMedQA 中的工程相關問題

    由於 PubMedQA 主要是生物醫學領域，與 engineering 相關的可能較少。
    但我們仍然可以找到：
    - 生物工程 (bioengineering)
    - 組織工程 (tissue engineering)
    - 醫療設備設計等相關內容
    """
    print("\n" + "="*60)
    print("🔍 Filtering PubMedQA for engineering content...")
    print("="*60)

    # 載入數據集
    dataset_name = config['data']['tech_dataset_2']
    dataset_config = config['data']['tech_dataset_2_config']

    try:
        dataset = load_dataset(dataset_name, dataset_config, split="train")
        print(f"   Original samples: {len(dataset)}")
    except Exception as e:
        print(f"❌ Error loading PubMedQA: {e}")
        return None

    # 過濾關鍵字
    keywords = config['data']['filter_keywords']

    # 過濾函數
    def is_engineering_related(example):
        # 檢查問題
        if check_engineering_relevance(example.get('question', ''), keywords):
            return True

        # 檢查上下文
        contexts = example.get('context', {}).get('contexts', [])
        for context in contexts:
            if check_engineering_relevance(context, keywords):
                return True

        # 檢查詳細答案
        if check_engineering_relevance(example.get('long_answer', ''), keywords):
            return True

        return False

    # 應用過濾
    print("   Filtering...")
    filtered = dataset.filter(is_engineering_related, desc="Filtering PubMedQA")

    print(f"✅ Filtered samples: {len(filtered)}")
    print(f"   Retention rate: {len(filtered)/len(dataset)*100:.2f}%")

    return filtered


def sample_and_save(scienceqa_filtered, pubmedqa_filtered, config):
    """
    取樣並保存過濾後的數據

    策略：
    1. 計算每個數據集應取樣的數量
    2. 隨機取樣
    3. 保存到本地
    """
    print("\n" + "="*60)
    print("📊 Sampling and saving...")
    print("="*60)

    target_samples = config['data']['tech_samples']
    print(f"   Target total samples: {target_samples}")

    # 計算可用樣本
    scienceqa_available = len(scienceqa_filtered) if scienceqa_filtered else 0
    pubmedqa_available = len(pubmedqa_filtered) if pubmedqa_filtered else 0
    total_available = scienceqa_available + pubmedqa_available

    print(f"   ScienceQA available: {scienceqa_available}")
    print(f"   PubMedQA available: {pubmedqa_available}")
    print(f"   Total available: {total_available}")

    if total_available == 0:
        print("❌ No filtered samples available!")
        return None

    # 計算取樣數量（按比例）
    if total_available < target_samples:
        print(f"⚠️  Available samples ({total_available}) less than target ({target_samples})")
        print(f"   Using all available samples")
        scienceqa_samples = scienceqa_available
        pubmedqa_samples = pubmedqa_available
    else:
        # 按比例分配
        ratio = target_samples / total_available
        scienceqa_samples = int(scienceqa_available * ratio)
        pubmedqa_samples = target_samples - scienceqa_samples

    print(f"\n   Sampling plan:")
    print(f"   - ScienceQA: {scienceqa_samples} samples")
    print(f"   - PubMedQA: {pubmedqa_samples} samples")

    # 取樣
    sampled_datasets = []

    if scienceqa_samples > 0 and scienceqa_filtered:
        scienceqa_sampled = scienceqa_filtered.shuffle(seed=42).select(range(min(scienceqa_samples, len(scienceqa_filtered))))
        sampled_datasets.append(scienceqa_sampled)
        print(f"✅ Sampled {len(scienceqa_sampled)} from ScienceQA")

    if pubmedqa_samples > 0 and pubmedqa_filtered:
        pubmedqa_sampled = pubmedqa_filtered.shuffle(seed=42).select(range(min(pubmedqa_samples, len(pubmedqa_filtered))))
        sampled_datasets.append(pubmedqa_sampled)
        print(f"✅ Sampled {len(pubmedqa_sampled)} from PubMedQA")

    # 合併（暫時保存為單獨數據集，下一步 merge_datasets.py 會統一格式）
    # 這裡只是儲存過濾後的結果
    output_dir = Path("filtered_data")
    output_dir.mkdir(exist_ok=True)

    if scienceqa_filtered and scienceqa_samples > 0:
        scienceqa_sampled.save_to_disk(str(output_dir / "scienceqa_engineering"))
        print(f"💾 Saved ScienceQA to: {output_dir / 'scienceqa_engineering'}")

    if pubmedqa_filtered and pubmedqa_samples > 0:
        pubmedqa_sampled.save_to_disk(str(output_dir / "pubmedqa_engineering"))
        print(f"💾 Saved PubMedQA to: {output_dir / 'pubmedqa_engineering'}")

    return sampled_datasets


def main():
    """主函數：過濾工程相關數據"""
    print("\n" + "="*60)
    print("🚀 Data Preparation - Step 2: Filter Engineering Data")
    print("="*60)

    # 載入配置
    config = load_config()

    # 過濾數據集
    scienceqa_filtered = filter_scienceqa(config)
    pubmedqa_filtered = filter_pubmedqa(config)

    # 取樣並保存
    if scienceqa_filtered or pubmedqa_filtered:
        sample_and_save(scienceqa_filtered, pubmedqa_filtered, config)

        print("\n" + "="*60)
        print("✅ Filtering completed!")
        print("="*60)
        print("\n💡 Next step:")
        print("   Run: python merge_datasets.py")
    else:
        print("\n" + "="*60)
        print("❌ Filtering failed!")
        print("="*60)
        print("   No engineering-related data found.")


if __name__ == "__main__":
    main()
