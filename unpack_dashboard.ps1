<#
    unpack_dashboard.ps1  --  unpack the TradingLab dashboard zip into the repo
    -------------------------------------------------------------------------
    The zip contains a top-level folder `tradinglab-dashboard/`. This script
    extracts it into a subfolder of your project so it stays self-contained and
    does NOT clutter or collide with your repo root.

    USAGE (PowerShell, from anywhere):
        .\unpack_dashboard.ps1                       # dry run (shows what it will do)
        .\unpack_dashboard.ps1 -Execute              # actually extract

    Optional params:
        -Zip   "path\to\tradinglab-dashboard.zip"    # default: .\tradinglab-dashboard.zip
        -Dest  "D:\Luke Files\...\TradingResearch"   # default: current directory
#>

param(
    [string]$Zip  = ".\tradinglab-dashboard.zip",
    [string]$Dest = ".",
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'

# Resolve paths
if (-not (Test-Path $Zip)) {
    Write-Host "Zip not found: $Zip" -ForegroundColor Red
    Write-Host "Pass the right path with -Zip 'C:\Users\you\Downloads\tradinglab-dashboard.zip'"
    exit 1
}
$Zip  = (Resolve-Path $Zip).Path
$Dest = (Resolve-Path $Dest).Path
$target = Join-Path $Dest 'tradinglab-dashboard'

Write-Host "Zip    : $Zip"
Write-Host "Dest   : $Dest"
Write-Host "Target : $target"
Write-Host ""

if (Test-Path $target) {
    Write-Host "WARNING: '$target' already exists." -ForegroundColor Yellow
    Write-Host "This script will NOT overwrite it. Rename/remove it first, or extract elsewhere."
    if ($Execute) { exit 1 }
}

if (-not $Execute) {
    Write-Host "DRY RUN. Would extract the zip's 'tradinglab-dashboard/' folder into:" -ForegroundColor Cyan
    Write-Host "    $target"
    Write-Host ""
    Write-Host "Re-run with -Execute to extract." -ForegroundColor Green
    exit 0
}

# Extract (Expand-Archive preserves the top-level tradinglab-dashboard/ folder)
Write-Host "Extracting..." -ForegroundColor Cyan
Expand-Archive -Path $Zip -DestinationPath $Dest -Force

if (Test-Path $target) {
    Write-Host "Done." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "    cd `"$target`""
    Write-Host "    npm install"
    Write-Host "    npm run dev            # fixtures (DEV FIXTURE badges)"
    Write-Host ""
    Write-Host "To use real data, in another terminal from the REPO ROOT:"
    Write-Host "    conda activate quant2"
    Write-Host "    python `"$target\bridge\sqlite_bridge.py`" --root `"$Dest`" --port 8799"
    Write-Host "  then:"
    Write-Host "    `$env:VITE_BRIDGE_URL = 'http://127.0.0.1:8799'; npm run dev"
} else {
    Write-Host "Extraction finished but '$target' not found — check the zip's structure." -ForegroundColor Yellow
}
