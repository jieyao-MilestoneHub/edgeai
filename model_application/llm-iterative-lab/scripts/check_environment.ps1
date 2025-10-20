# ============================================
# 環境檢查腳本 - LLM Iterative Lab
# ============================================
# 描述: 檢查訓練環境是否正確配置
# 使用方式: .\check_environment.ps1
# ============================================

param(
    [switch]$Verbose
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

function Write-Section($message) {
    Write-Host ""
    Write-ColorOutput Cyan "─── $message ───"
}

function Write-CheckItem($name, $status, $detail = "") {
    $symbol = if ($status -eq "OK") { "✓" } elseif ($status -eq "WARN") { "⚠" } else { "✗" }
    $color = if ($status -eq "OK") { "Green" } elseif ($status -eq "WARN") { "Yellow" } else { "Red" }

    Write-Host -NoNewline "$symbol " -ForegroundColor $color
    Write-Host -NoNewline "$name : " -ForegroundColor White
    Write-Host "$detail" -ForegroundColor $color
}

# 統計變數
$script:totalChecks = 0
$script:passedChecks = 0
$script:warningChecks = 0
$script:failedChecks = 0

function Update-CheckStats($status) {
    $script:totalChecks++
    if ($status -eq "OK") { $script:passedChecks++ }
    elseif ($status -eq "WARN") { $script:warningChecks++ }
    else { $script:failedChecks++ }
}

# 開始檢查
Write-Header "LLM Iterative Lab - 環境檢查"

# ============================================
# 1. Python 環境檢查
# ============================================
Write-Section "Python 環境"

# 檢查 Python
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $pythonVersionNumber = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
        Write-CheckItem "Python" "OK" "版本 $pythonVersionNumber"
        Update-CheckStats "OK"
    } else {
        Write-CheckItem "Python" "FAIL" "未安裝"
        Update-CheckStats "FAIL"
    }
} catch {
    Write-CheckItem "Python" "FAIL" "無法執行"
    Update-CheckStats "FAIL"
}

# 檢查 pip
try {
    $pipVersion = python -m pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-CheckItem "pip" "OK" "已安裝"
        Update-CheckStats "OK"
    } else {
        Write-CheckItem "pip" "FAIL" "未安裝"
        Update-CheckStats "FAIL"
    }
} catch {
    Write-CheckItem "pip" "FAIL" "無法執行"
    Update-CheckStats "FAIL"
}

# ============================================
# 2. 核心套件檢查
# ============================================
Write-Section "核心 Python 套件"

$corePackages = @{
    "torch" = "PyTorch"
    "transformers" = "Transformers"
    "datasets" = "Datasets"
    "peft" = "PEFT"
    "accelerate" = "Accelerate"
    "trl" = "TRL"
}

foreach ($pkg in $corePackages.Keys) {
    $result = python -c "
try:
    import $pkg
    print($pkg.__version__)
    exit(0)
except ImportError:
    print('未安裝')
    exit(1)
" 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-CheckItem $corePackages[$pkg] "OK" "版本 $result"
        Update-CheckStats "OK"
    } else {
        Write-CheckItem $corePackages[$pkg] "FAIL" "未安裝"
        Update-CheckStats "FAIL"
    }
}

# ============================================
# 3. CUDA 和 GPU 檢查
# ============================================
Write-Section "CUDA 和 GPU"

# 檢查 CUDA 可用性
$cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cudaAvailable -eq "True") {
    Write-CheckItem "CUDA" "OK" "可用"
    Update-CheckStats "OK"

    # GPU 數量
    $gpuCount = python -c "import torch; print(torch.cuda.device_count())" 2>&1
    Write-CheckItem "GPU 數量" "OK" "$gpuCount 個"
    Update-CheckStats "OK"

    # GPU 詳細資訊
    for ($i = 0; $i -lt [int]$gpuCount; $i++) {
        $gpuName = python -c "import torch; print(torch.cuda.get_device_name($i))" 2>&1
        $gpuMemory = python -c "import torch; print(f'{torch.cuda.get_device_properties($i).total_memory / 1024**3:.1f} GB')" 2>&1
        Write-CheckItem "  GPU $i" "OK" "$gpuName ($gpuMemory)"
        Update-CheckStats "OK"
    }

    # CUDA 版本
    $cudaVersion = python -c "import torch; print(torch.version.cuda)" 2>&1
    Write-CheckItem "CUDA 版本" "OK" "$cudaVersion"
    Update-CheckStats "OK"

    # cuDNN 版本
    $cudnnVersion = python -c "import torch; print(torch.backends.cudnn.version())" 2>&1
    Write-CheckItem "cuDNN 版本" "OK" "$cudnnVersion"
    Update-CheckStats "OK"

} else {
    Write-CheckItem "CUDA" "WARN" "不可用，將使用 CPU 模式"
    Update-CheckStats "WARN"
}

# ============================================
# 4. 記憶體檢查
# ============================================
Write-Section "系統資源"

# 系統 RAM
$ram = Get-CimInstance Win32_ComputerSystem
$totalRamGB = [math]::Round($ram.TotalPhysicalMemory / 1GB, 2)
if ($totalRamGB -ge 16) {
    Write-CheckItem "系統記憶體" "OK" "${totalRamGB} GB"
    Update-CheckStats "OK"
} elseif ($totalRamGB -ge 8) {
    Write-CheckItem "系統記憶體" "WARN" "${totalRamGB} GB (建議 16GB 以上)"
    Update-CheckStats "WARN"
} else {
    Write-CheckItem "系統記憶體" "FAIL" "${totalRamGB} GB (不足，建議 16GB 以上)"
    Update-CheckStats "FAIL"
}

# 可用磁碟空間
$drive = Get-PSDrive C
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
if ($freeSpaceGB -ge 50) {
    Write-CheckItem "磁碟空間" "OK" "${freeSpaceGB} GB 可用"
    Update-CheckStats "OK"
} elseif ($freeSpaceGB -ge 20) {
    Write-CheckItem "磁碟空間" "WARN" "${freeSpaceGB} GB 可用 (建議 50GB 以上)"
    Update-CheckStats "WARN"
} else {
    Write-CheckItem "磁碟空間" "FAIL" "${freeSpaceGB} GB 可用 (不足)"
    Update-CheckStats "FAIL"
}

# ============================================
# 5. 專案結構檢查
# ============================================
Write-Section "專案結構"

$projectRoot = Split-Path $PSScriptRoot -Parent
$requiredDirs = @(
    "scripts",
    "configs",
    "src",
    "data_preparation"
)

foreach ($dir in $requiredDirs) {
    $dirPath = Join-Path $projectRoot $dir
    if (Test-Path $dirPath) {
        Write-CheckItem "目錄 $dir" "OK" "存在"
        Update-CheckStats "OK"
    } else {
        Write-CheckItem "目錄 $dir" "FAIL" "不存在"
        Update-CheckStats "FAIL"
    }
}

# 檢查配置檔案
$configPath = Join-Path $projectRoot "config.yaml"
if (Test-Path $configPath) {
    Write-CheckItem "config.yaml" "OK" "存在"
    Update-CheckStats "OK"
} else {
    Write-CheckItem "config.yaml" "WARN" "不存在 (需要時再建立)"
    Update-CheckStats "WARN"
}

# ============================================
# 6. 可選套件檢查
# ============================================
Write-Section "可選套件"

$optionalPackages = @{
    "jupyter" = "Jupyter Notebook"
    "tensorboard" = "TensorBoard"
    "wandb" = "Weights & Biases"
    "bitsandbytes" = "BitsAndBytes (量化)"
}

foreach ($pkg in $optionalPackages.Keys) {
    $result = python -c "
try:
    import $pkg
    print($pkg.__version__)
    exit(0)
except ImportError:
    print('未安裝')
    exit(1)
" 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-CheckItem $optionalPackages[$pkg] "OK" "版本 $result"
        Update-CheckStats "OK"
    } else {
        Write-CheckItem $optionalPackages[$pkg] "WARN" "未安裝 (可選)"
        Update-CheckStats "WARN"
    }
}

# ============================================
# 7. 詳細測試 (Verbose)
# ============================================
if ($Verbose) {
    Write-Section "詳細測試"

    # 測試模型加載
    Write-Host "測試 Transformers 模型加載..." -ForegroundColor Cyan
    $modelTest = python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print('OK: 可以加載模型')
except Exception as e:
    print(f'FAIL: {str(e)}')
" 2>&1
    Write-Host $modelTest

    # 測試 CUDA 張量操作
    if ($cudaAvailable -eq "True") {
        Write-Host "測試 CUDA 張量操作..." -ForegroundColor Cyan
        $cudaTest = python -c "
import torch
try:
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.mm(x, y)
    print('OK: CUDA 張量運算正常')
except Exception as e:
    print(f'FAIL: {str(e)}')
" 2>&1
        Write-Host $cudaTest
    }
}

# ============================================
# 8. 總結報告
# ============================================
Write-Header "檢查總結"

Write-Host "總檢查項目: $script:totalChecks" -ForegroundColor White
Write-Host "通過: $script:passedChecks" -ForegroundColor Green
Write-Host "警告: $script:warningChecks" -ForegroundColor Yellow
Write-Host "失敗: $script:failedChecks" -ForegroundColor Red
Write-Host ""

# 判斷整體狀態
if ($script:failedChecks -eq 0 -and $script:warningChecks -eq 0) {
    Write-ColorOutput Green "✓ 環境完全就緒，可以開始訓練！"
    exit 0
} elseif ($script:failedChecks -eq 0) {
    Write-ColorOutput Yellow "⚠ 環境基本就緒，但有部分警告項目"
    Write-Host "  建議處理警告項目以獲得最佳性能"
    exit 0
} else {
    Write-ColorOutput Red "✗ 環境檢查失敗，請修復以下問題："
    Write-Host "  1. 檢查上方失敗的項目"
    Write-Host "  2. 執行 .\install_requirements.ps1 安裝缺少的套件"
    Write-Host "  3. 重新執行此檢查腳本"
    exit 1
}
