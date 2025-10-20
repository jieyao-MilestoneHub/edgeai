#!/usr/bin/env python3
"""
Interactive CLI for Qwen2.5-3B AlignLoop Training Framework
互動式命令列工具 - Qwen2.5-3B 多路徑訓練框架

Usage:
    python cli.py
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Try to import questionary for better UI, fallback to basic input
try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False
    print("⚠️  提示: 安裝 questionary 可獲得更好的互動體驗: pip install questionary\n")


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.ENDC}\n")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return Path(file_path).exists()


def get_route_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all training routes."""
    base_dir = Path(__file__).parent / "lab_tasks" / "task01_sft-dpo"

    return {
        "A": {
            "name": "SFT 單次訓練",
            "description": "快速指令微調,建立基礎能力",
            "script": str(base_dir / "scripts" / "run_sft.py"),
            "config": str(base_dir / "config.yaml"),
            "override": str(base_dir / "configs" / "sft" / "sft_r16.yaml"),
            "required_data": "SFT 數據集 (data_preparation/final_dataset)",
            "command_template": 'python "{script}" --config "{config}"'
        },
        "B": {
            "name": "DPO 單次訓練",
            "description": "已有 SFT 模型,需要偏好對齊",
            "script": str(base_dir / "scripts" / "run_dpo.py"),
            "config": str(base_dir / "config.yaml"),
            "override": str(base_dir / "configs" / "dpo" / "dpo_r16.yaml"),
            "required_data": "偏好對數據集 (data/preference)",
            "command_template": 'python "{script}" --config "{config}" --override "{override}"'
        },
        "C": {
            "name": "Iterative 固定迭代訓練",
            "description": "多輪數據收集與訓練,固定迭代次數",
            "script": str(base_dir / "scripts" / "run_iteration.py"),
            "config": str(base_dir / "config.yaml"),
            "override": str(base_dir / "configs" / "iterative" / "iteration_1.yaml"),
            "required_data": "SFT 數據集 + 偏好對數據集",
            "command_template": 'python "{script}" --config "{config}" --override "{override}" --iterations {iterations}'
        },
        "D": {
            "name": "Hybrid 閉環訓練",
            "description": "持續數據累積,條件觸發 DPO,自動優化",
            "script": str(base_dir / "scripts" / "run_hybrid.py"),
            "config": str(base_dir / "config.yaml"),
            "override": str(base_dir / "configs" / "hybrid" / "default.yaml"),
            "required_data": "SFT 數據集 + 偏好對數據集 (JSONL格式)",
            "command_template": 'python "{script}" --config "{config}" --override "{override}"'
        }
    }


def select_route_interactive() -> Optional[str]:
    """Interactive route selection using questionary."""
    routes = get_route_info()

    choices = [
        f"路線 {key}: {info['name']} - {info['description']}"
        for key, info in routes.items()
    ]

    answer = questionary.select(
        "請選擇訓練路線:",
        choices=choices
    ).ask()

    if answer is None:
        return None

    # Extract route key (A/B/C/D) from selection
    route_key = answer.split(":")[0].replace("路線 ", "").strip()
    return route_key


def select_route_basic() -> Optional[str]:
    """Basic route selection using input()."""
    routes = get_route_info()

    print(f"\n{Colors.BOLD}請選擇訓練路線:{Colors.ENDC}")
    for key, info in routes.items():
        print(f"  {Colors.CYAN}{key}{Colors.ENDC}. {info['name']}")
        print(f"     {Colors.YELLOW}{info['description']}{Colors.ENDC}")

    while True:
        choice = input(f"\n{Colors.BOLD}輸入選項 (A/B/C/D) 或 'q' 退出: {Colors.ENDC}").strip().upper()
        if choice == 'Q':
            return None
        if choice in routes:
            return choice
        print_error("無效選項,請重新輸入!")


def verify_config_files(route_key: str) -> bool:
    """Verify that required config files exist."""
    routes = get_route_info()
    route_info = routes[route_key]

    print_header("驗證設定檔")

    all_exist = True

    # Check main config
    if check_file_exists(route_info["config"]):
        print_success(f"主設定檔: {route_info['config']}")
    else:
        print_error(f"主設定檔不存在: {route_info['config']}")
        all_exist = False

    # Check override config if specified
    if route_info.get("override"):
        if check_file_exists(route_info["override"]):
            print_success(f"覆蓋設定檔: {route_info['override']}")
        else:
            print_warning(f"覆蓋設定檔不存在: {route_info['override']}")
            print_info("將使用主設定檔執行")

    # Check script
    if check_file_exists(route_info["script"]):
        print_success(f"訓練腳本: {route_info['script']}")
    else:
        print_error(f"訓練腳本不存在: {route_info['script']}")
        all_exist = False

    return all_exist


def display_route_summary(route_key: str, extra_args: Dict[str, Any] = None):
    """Display summary of the selected route and configuration."""
    routes = get_route_info()
    route_info = routes[route_key]

    print_header(f"路線 {route_key}: {route_info['name']}")

    print(f"{Colors.BOLD}說明:{Colors.ENDC} {route_info['description']}")
    print(f"{Colors.BOLD}所需數據:{Colors.ENDC} {route_info['required_data']}")

    print(f"\n{Colors.BOLD}設定檔:{Colors.ENDC}")
    print(f"  • 主設定: {route_info['config']}")
    if route_info.get("override"):
        print(f"  • 覆蓋設定: {route_info['override']}")

    # Build command
    cmd_template = route_info["command_template"]
    cmd_vars = {
        "script": route_info["script"],
        "config": route_info["config"],
        "override": route_info.get("override", ""),
        "iterations": extra_args.get("iterations", 3) if extra_args else 3
    }

    # Add extra arguments
    if extra_args:
        if extra_args.get("output"):
            cmd_template += ' --output "{output}"'
            cmd_vars["output"] = extra_args["output"]
        if extra_args.get("data"):
            cmd_template += ' --data "{data}"'
            cmd_vars["data"] = extra_args["data"]

    command = cmd_template.format(**cmd_vars)

    print(f"\n{Colors.BOLD}執行命令:{Colors.ENDC}")
    print(f"  {Colors.CYAN}{command}{Colors.ENDC}")

    return command


def confirm_execution() -> bool:
    """Ask user to confirm execution."""
    if HAS_QUESTIONARY:
        return questionary.confirm(
            "\n確認執行訓練?",
            default=False
        ).ask()
    else:
        while True:
            choice = input(f"\n{Colors.BOLD}確認執行訓練? (y/n): {Colors.ENDC}").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            print_error("請輸入 y 或 n")


def run_training(command: str):
    """Execute the training command."""
    print_header("開始訓練")
    print_info(f"執行命令: {command}")
    print()

    try:
        # Change to the correct directory
        script_dir = Path(__file__).parent / "lab_tasks" / "task01_sft-dpo" / "scripts"

        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(script_dir),
            check=True
        )

        if result.returncode == 0:
            print_success("\n訓練完成!")

    except subprocess.CalledProcessError as e:
        print_error(f"\n訓練執行失敗: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print_warning("\n\n訓練被使用者中斷")
        sys.exit(0)


def get_additional_args() -> Dict[str, Any]:
    """Get additional arguments from user."""
    args = {}

    if HAS_QUESTIONARY:
        # Ask for optional parameters
        if questionary.confirm("是否指定自訂輸出目錄?", default=False).ask():
            output = questionary.text("輸出目錄路徑:").ask()
            if output:
                args["output"] = output

        if questionary.confirm("是否指定自訂數據路徑?", default=False).ask():
            data = questionary.text("數據路徑:").ask()
            if data:
                args["data"] = data
    else:
        # Basic input
        output = input(f"{Colors.BOLD}自訂輸出目錄 (留空使用預設): {Colors.ENDC}").strip()
        if output:
            args["output"] = output

        data = input(f"{Colors.BOLD}自訂數據路徑 (留空使用預設): {Colors.ENDC}").strip()
        if data:
            args["data"] = data

    return args


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-3B AlignLoop 互動式訓練工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
路線說明:
  A - SFT 單次訓練: 快速指令微調,建立基礎能力
  B - DPO 單次訓練: 已有 SFT 模型,需要偏好對齊
  C - Iterative 固定迭代訓練: 多輪數據收集與訓練
  D - Hybrid 閉環訓練: 持續數據累積,條件觸發 DPO

範例:
  python cli.py              # 互動式選擇
  python cli.py --route A    # 直接執行 SFT 訓練
  python cli.py --route C --iterations 5  # 執行 5 次迭代訓練
        """
    )

    parser.add_argument(
        "--route",
        type=str,
        choices=["A", "B", "C", "D"],
        help="直接指定訓練路線 (A/B/C/D)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="迭代次數 (僅適用於路線 C)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="自訂輸出目錄"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="自訂數據路徑"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳過確認,直接執行"
    )

    args = parser.parse_args()

    # Welcome message
    print_header("Qwen2.5-3B AlignLoop 訓練框架")
    print_info("歡迎使用互動式命令列工具!")

    # Select route
    if args.route:
        route_key = args.route
        print_info(f"使用指定路線: {route_key}")
    else:
        if HAS_QUESTIONARY:
            route_key = select_route_interactive()
        else:
            route_key = select_route_basic()

    if route_key is None:
        print_warning("已取消")
        sys.exit(0)

    # Verify config files
    if not verify_config_files(route_key):
        print_error("\n設定檔驗證失敗!")
        print_warning("請確保以下項目已完成:")
        print("  1. 執行數據準備腳本")
        print("  2. 確認設定檔存在且正確")
        print("  3. 檢查數據路徑是否正確")
        sys.exit(1)

    # Get additional arguments
    extra_args = {
        "iterations": args.iterations,
        "output": args.output,
        "data": args.data
    }

    # If not specified via CLI, ask interactively
    if not args.output and not args.data and not args.yes:
        interactive_args = get_additional_args()
        extra_args.update(interactive_args)

    # Display summary
    command = display_route_summary(route_key, extra_args)

    # Important reminders
    print(f"\n{Colors.BOLD}{Colors.YELLOW}重要提醒:{Colors.ENDC}")
    routes = get_route_info()
    print(f"  ⚠️  請確認已完成: {routes[route_key]['required_data']}")
    print(f"  ⚠️  請確認設定檔中的參數已正確配置")
    print(f"  ⚠️  訓練將佔用大量 GPU 資源,請確保沒有其他任務在執行")

    # Confirm execution
    if args.yes or confirm_execution():
        run_training(command)
    else:
        print_warning("已取消執行")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\n程式被使用者中斷")
        sys.exit(0)
    except Exception as e:
        print_error(f"\n發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
