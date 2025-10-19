"""
資料準備 - 步驟 1: 下載數據集

功能：
1. 下載 OpenHermes-2.5 通用指令集
2. 下載 ScienceQA 科學問答數據集
3. 下載 PubMedQA 生物醫學問答數據集
4. 儲存到本地快取

使用套件：
- datasets: Hugging Face Datasets 庫，自動下載與快取

作者: Joel
"""

import os
import yaml
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def load_config(config_path: str = "../config.yaml"):
    """載入配置檔案"""
    print(f"📋 Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def download_openhermes(config):
    """
    下載 OpenHermes-2.5 數據集

    OpenHermes-2.5 是高品質的通用指令集，整合了：
    - Alpaca: Stanford 的指令遵循數據
    - ShareGPT: 用戶與 ChatGPT 的對話
    - FLAN: Google 的指令微調數據
    - 等多個來源

    數據格式：
    {
        "conversations": [
            {"from": "human", "value": "問題"},
            {"from": "gpt", "value": "回答"}
        ]
    }
    """
    print("\n" + "="*60)
    print("📥 Downloading OpenHermes-2.5...")
    print("="*60)

    dataset_name = config['data']['general_dataset']
    print(f"   Dataset: {dataset_name}")

    try:
        # 使用 datasets 庫自動下載
        dataset = load_dataset(dataset_name, split="train")
        print(f"✅ Downloaded successfully!")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Columns: {dataset.column_names}")

        # 顯示範例
        print("\n📝 Sample data:")
        print(f"   {dataset[0]}")

        return dataset

    except Exception as e:
        print(f"❌ Error downloading OpenHermes-2.5: {e}")
        return None


def download_scienceqa(config):
    """
    下載 ScienceQA 數據集

    ScienceQA 包含：
    - 21,208 個多模態科學問題
    - 涵蓋物理、化學、生物、工程等領域
    - 包含問題、選項、答案、解釋

    數據格式：
    {
        "question": "問題文本",
        "choices": ["選項A", "選項B", "選項C", "選項D"],
        "answer": 0,  # 正確答案索引
        "lecture": "相關知識",
        "solution": "解答說明"
    }
    """
    print("\n" + "="*60)
    print("📥 Downloading ScienceQA...")
    print("="*60)

    dataset_name = config['data']['tech_dataset_1']
    print(f"   Dataset: {dataset_name}")

    try:
        # ScienceQA 包含多個分割
        dataset = load_dataset(dataset_name)
        print(f"✅ Downloaded successfully!")
        print(f"   Splits: {list(dataset.keys())}")

        # 使用訓練集
        train_data = dataset['train']
        print(f"   Train samples: {len(train_data)}")
        print(f"   Columns: {train_data.column_names}")

        # 顯示範例
        print("\n📝 Sample data:")
        sample = train_data[0]
        print(f"   Question: {sample.get('question', 'N/A')}")
        print(f"   Choices: {sample.get('choices', 'N/A')}")

        return dataset

    except Exception as e:
        print(f"❌ Error downloading ScienceQA: {e}")
        return None


def download_pubmedqa(config):
    """
    下載 PubMedQA 數據集

    PubMedQA 包含：
    - 1,000 個專家標註的生物醫學問題
    - 基於 PubMed 摘要的問答
    - 是/否/可能 三種答案類型

    數據格式：
    {
        "question": "問題文本",
        "context": {"contexts": ["段落1", "段落2", ...]},
        "long_answer": "詳細答案",
        "final_decision": "yes/no/maybe"
    }
    """
    print("\n" + "="*60)
    print("📥 Downloading PubMedQA...")
    print("="*60)

    dataset_name = config['data']['tech_dataset_2']
    dataset_config = config['data']['tech_dataset_2_config']
    print(f"   Dataset: {dataset_name}")
    print(f"   Config: {dataset_config}")

    try:
        # 使用 pqa_labeled 配置（有標註的數據）
        dataset = load_dataset(dataset_name, dataset_config)
        print(f"✅ Downloaded successfully!")
        print(f"   Splits: {list(dataset.keys())}")

        # 使用訓練集
        train_data = dataset['train']
        print(f"   Train samples: {len(train_data)}")
        print(f"   Columns: {train_data.column_names}")

        # 顯示範例
        print("\n📝 Sample data:")
        sample = train_data[0]
        print(f"   Question: {sample.get('question', 'N/A')}")
        print(f"   Decision: {sample.get('final_decision', 'N/A')}")

        return dataset

    except Exception as e:
        print(f"❌ Error downloading PubMedQA: {e}")
        print(f"   Note: PubMedQA might require authentication or have access restrictions")
        return None


def main():
    """主函數：下載所有數據集"""
    print("\n" + "="*60)
    print("🚀 Data Preparation - Step 1: Download Datasets")
    print("="*60)

    # 載入配置
    config = load_config()

    # 下載數據集
    openhermes = download_openhermes(config)
    scienceqa = download_scienceqa(config)
    pubmedqa = download_pubmedqa(config)

    # 統計結果
    print("\n" + "="*60)
    print("📊 Download Summary")
    print("="*60)

    results = {
        "OpenHermes-2.5": openhermes is not None,
        "ScienceQA": scienceqa is not None,
        "PubMedQA": pubmedqa is not None,
    }

    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {name}: {status}")

    # 保存信息
    if all(results.values()):
        print("\n✅ All datasets downloaded successfully!")
        print("\n💡 Next step:")
        print("   Run: python filter_engineering.py")
    else:
        print("\n⚠️  Some datasets failed to download.")
        print("   Please check the error messages above.")

    print("\n📁 Datasets cached in: ~/.cache/huggingface/datasets/")


if __name__ == "__main__":
    main()
