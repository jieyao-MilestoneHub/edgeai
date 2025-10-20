# ============================================
# 套件安裝腳本 - LLM Iterative Lab
# ============================================
# 描述: 自動安裝專案所需的 Python 套件
# 使用方式: .\install_requirements.ps1
# ============================================

param(
    [switch]$Force,
    [switch]$SkipTorch,
    [string]$CudaVersion = "cu121"
)

# 顏色輸出函數
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Header($message) {
    Write-Host ""
    Write-ColorOutput Green "═══════════════════════════════════════════════════"
    Write-ColorOutput Green $message
    Write-ColorOutput Green "═══════════════════════════════════════════════════"
    Write-Host ""
}

function Write-Step($message) {
    Write-ColorOutput Cyan "[$(Get-Date -Format 'HH:mm:ss')] $message"
}

function Write-Success($message) {
    Write-ColorOutput Green "✓ $message"
}

function Write-Warning($message) {
    Write-ColorOutput Yellow "⚠ $message"
}

function Write-Error-Custom($message) {
    Write-ColorOutput Red "✗ $message"
}

# 開始安裝
Write-Header "LLM Iterative Lab - 套件安裝腳本"

# 1. 檢查 Python
Write-Step "檢查 Python 安裝..."
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "找到 Python: $pythonVersion"
    } else {
        throw "Python 未安裝或無法執行"
    }
} catch {
    Write-Error-Custom "Python 未安裝，請先安裝 Python 3.9 或更高版本"
    Write-Host "下載連結: https://www.python.org/downloads/"
    exit 1
}

# 檢查 Python 版本
$pythonVersionNumber = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$majorVersion = [int]($pythonVersionNumber.Split('.')[0])
$minorVersion = [int]($pythonVersionNumber.Split('.')[1])

if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 9)) {
    Write-Error-Custom "Python 版本過低 (當前: $pythonVersionNumber)，需要 3.9 或更高版本"
    exit 1
}

# 2. 檢查 pip
Write-Step "檢查 pip..."
try {
    $pipVersion = python -m pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "找到 pip: $pipVersion"
    } else {
        throw "pip 無法執行"
    }
} catch {
    Write-Warning "pip 未安裝，正在安裝..."
    python -m ensurepip --upgrade
}

# 3. 升級 pip
Write-Step "升級 pip 到最新版本..."
python -m pip install --upgrade pip
if ($LASTEXITCODE -eq 0) {
    Write-Success "pip 已升級"
} else {
    Write-Warning "pip 升級失敗，繼續安裝..."
}

# 4. 安裝 PyTorch
if (-not $SkipTorch) {
    Write-Header "安裝 PyTorch (CUDA $CudaVersion)"
    Write-Step "這可能需要幾分鐘時間..."

    $torchUrl = "https://download.pytorch.org/whl/$CudaVersion"
    python -m pip install torch torchvision torchaudio --index-url $torchUrl

    if ($LASTEXITCODE -eq 0) {
        Write-Success "PyTorch 安裝完成"

        # 驗證 CUDA
        Write-Step "驗證 CUDA 可用性..."
        $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>&1
        if ($cudaAvailable -eq "True") {
            $cudaDevice = python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
            Write-Success "CUDA 可用: $cudaDevice"
        } else {
            Write-Warning "CUDA 不可用，將使用 CPU 模式"
        }
    } else {
        Write-Error-Custom "PyTorch 安裝失敗"
        exit 1
    }
} else {
    Write-Warning "跳過 PyTorch 安裝 (--SkipTorch 已設定)"
}

# 5. 檢查 requirements.txt
$requirementsPath = Join-Path $PSScriptRoot "..\requirements.txt"
if (Test-Path $requirementsPath) {
    Write-Header "安裝專案依賴套件"
    Write-Step "從 requirements.txt 安裝..."

    python -m pip install -r $requirementsPath

    if ($LASTEXITCODE -eq 0) {
        Write-Success "所有依賴套件安裝完成"
    } else {
        Write-Error-Custom "部分套件安裝失敗"
        Write-Host "請檢查錯誤訊息並手動安裝失敗的套件"
    }
} else {
    Write-Warning "未找到 requirements.txt，跳過依賴安裝"
    Write-Host "位置: $requirementsPath"
}

# 6. 安裝額外推薦套件
Write-Header "安裝推薦套件"

$recommendedPackages = @(
    "jupyter",
    "ipywidgets",
    "tensorboard",
    "wandb"
)

foreach ($package in $recommendedPackages) {
    Write-Step "安裝 $package..."
    python -m pip install $package --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Success "$package 已安裝"
    } else {
        Write-Warning "$package 安裝失敗 (非必要)"
    }
}

# 7. 列出已安裝的關鍵套件
Write-Header "已安裝的關鍵套件版本"

$keyPackages = @(
    "torch",
    "transformers",
    "datasets",
    "peft",
    "accelerate",
    "bitsandbytes",
    "trl"
)

foreach ($pkg in $keyPackages) {
    $version = python -c "try:
    import $pkg
    print($pkg.__version__)
except:
    print('未安裝')" 2>&1

    if ($version -ne "未安裝") {
        Write-Host "$pkg : $version" -ForegroundColor White
    } else {
        Write-Host "$pkg : 未安裝" -ForegroundColor Red
    }
}

# 完成
Write-Header "安裝完成"
Write-Success "所有套件安裝流程已完成！"
Write-Host ""
Write-Host "下一步:" -ForegroundColor Cyan
Write-Host "  1. 執行環境檢查: .\check_environment.ps1"
Write-Host "  2. 準備訓練數據: cd ..\data_preparation"
Write-Host "  3. 開始訓練: python run_sft.py --config ..\config.yaml"
Write-Host ""
