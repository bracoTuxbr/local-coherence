# pin_and_run.ps1 — runs bench with the least possible noise on Windows
#
# Usage:
#   ./tools/pin_and_run.ps1                         # default: 1M cells, 30 runs, core 0
#   ./tools/pin_and_run.ps1 -NCells 65536 -Runs 50 -Core 2
#   ./tools/pin_and_run.ps1 -OutCsv runs/e01_lp.csv
#
# Requires (ideally, not mandatory):
#   - PowerShell 5+ with administrator privileges (for SeLockMemoryPrivilege and power plan)
#   - bench.exe already built under ./build/

[CmdletBinding()]
param(
    [int]$NCells = 1048576,
    [int]$Runs   = 30,
    [int]$Core   = 0,
    [string]$Exe = "$PSScriptRoot/../build/bench.exe",
    [string]$OutCsv = "$PSScriptRoot/../runs/e01_$(Get-Date -Format yyyyMMdd_HHmmss).csv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Exe)) {
    Write-Host "ERROR: $Exe not found. Build with:" -ForegroundColor Red
    Write-Host "  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
    Write-Host "  cmake --build build --config Release"
    exit 1
}

# ensure output directory exists
$runsDir = Split-Path $OutCsv -Parent
if (-not (Test-Path $runsDir)) { New-Item -ItemType Directory -Path $runsDir -Force | Out-Null }

# check admin
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(`
    [Security.Principal.WindowsBuiltInRole] 'Administrator')
Write-Host "Admin: $isAdmin" -ForegroundColor Cyan

# save current power plan and switch to High Performance
$prevPlan = $null
try {
    $cur = (powercfg /getactivescheme) 2>$null
    if ($cur -match '\(([^)]+)\)\s*$') { $prevPlanName = $matches[1] }
    if ($cur -match 'GUID:\s*([0-9a-f-]+)') { $prevPlan = $matches[1] }
    Write-Host "Current plan: $prevPlanName ($prevPlan)" -ForegroundColor Cyan

    $highPerf = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    powercfg /setactive $highPerf 2>$null
    Write-Host "Plan switched to High Performance" -ForegroundColor Yellow
} catch {
    Write-Host "WARN: could not change power plan: $_" -ForegroundColor Yellow
}

# inspect general load before measuring
Write-Host "`n--- current load ---" -ForegroundColor Cyan
Get-Process | Sort-Object -Property WS -Descending | Select-Object -First 5 Name, Id, WS, CPU | Format-Table -AutoSize

# affinity mask: bit `Core` set
$mask = [int64](1 -shl $Core)
Write-Host "Pin core: $Core  (mask=0x$($mask.ToString('X')))" -ForegroundColor Cyan
Write-Host "n_cells=$NCells  runs=$Runs"
Write-Host "Output CSV: $OutCsv`n"

# arguments: the exe also tries to pin internally, but we add an extra layer
$argList = @("$NCells", "$Runs", "$Core")

# Start-Process with affinity + high priority
$si = New-Object System.Diagnostics.ProcessStartInfo
$si.FileName = $Exe
$si.Arguments = $argList -join " "
$si.UseShellExecute = $false
$si.RedirectStandardOutput = $true
$si.RedirectStandardError  = $true
$si.WorkingDirectory = (Resolve-Path "$PSScriptRoot/..").Path

$p = New-Object System.Diagnostics.Process
$p.StartInfo = $si
$null = $p.Start()

try {
    $p.PriorityClass = 'High'
    $p.ProcessorAffinity = [int64]$mask
} catch {
    Write-Host "WARN: priority/affinity adjustment failed: $_" -ForegroundColor Yellow
}

$stdout = $p.StandardOutput.ReadToEnd()
$stderr = $p.StandardError.ReadToEnd()
$p.WaitForExit()

# stderr -> screen. stdout -> CSV.
if ($stderr) { Write-Host $stderr -ForegroundColor DarkGray }
$stdout | Out-File -FilePath $OutCsv -Encoding utf8

Write-Host "`n--- CSV ---" -ForegroundColor Green
Write-Host $stdout
Write-Host "Saved to: $OutCsv" -ForegroundColor Green

# restore power plan
if ($prevPlan) {
    powercfg /setactive $prevPlan 2>$null
    Write-Host "Plan restored: $prevPlan" -ForegroundColor Cyan
}

exit $p.ExitCode
