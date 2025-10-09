"""上傳模型到 Hugging Face Hub"""
import argparse
from huggingface_hub import HfApi, create_repo

def main(args):
    print("=" * 60)
    print("🚀 上傳模型到 Hugging Face")
    print("=" * 60)

    # 創建 repo（如果不存在）
    try:
        create_repo(
            repo_id=args.repo_name,
            private=args.private,
            exist_ok=True,
        )
        print(f"✅ Repo 創建成功: {args.repo_name}")
    except Exception as e:
        print(f"ℹ️  Repo 已存在或創建失敗: {e}")

    # 上傳模型
    api = HfApi()
    print(f"\n📤 上傳模型檔案...")

    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo_name,
        repo_type="model",
    )

    print("\n" + "=" * 60)
    print("✅ 上傳完成！")
    print("=" * 60)
    print(f"\n🌐 模型連結: https://huggingface.co/{args.repo_name}")
    print("\n現在任何人都可以使用你的模型：")
    print(f"  from transformers import pipeline")
    print(f'  model = pipeline("text-generation", model="{args.repo_name}")')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="模型路徑")
    parser.add_argument("--repo_name", required=True, help="HF repo 名稱 (username/model-name)")
    parser.add_argument("--private", action="store_true", help="設為私有")
    args = parser.parse_args()
    main(args)
