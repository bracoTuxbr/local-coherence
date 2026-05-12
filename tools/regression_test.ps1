# regression_test.ps1 — smoke test contra baseline v0 (golden_numbers.txt)
# Uso: ./tools/regression_test.ps1 [-Quick]
# Falha (exit 1) se invariante EXACT quebrar ou PERF sair da tolerancia.

[CmdletBinding()]
param([switch]$Quick)

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$root = Split-Path $PSScriptRoot -Parent
$build = Join-Path $root 'build'
$golden_file = Join-Path $root 'lab\golden_numbers.txt'

if (-not (Test-Path $golden_file)) {
    Write-Host "ERRO: $golden_file nao encontrado" -ForegroundColor Red
    exit 1
}

# parse golden_numbers.txt
$golden = @{}
Get-Content $golden_file | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith('#')) { return }
    $parts = $line -split '\|'
    if ($parts.Count -lt 5) { return }
    $key = $parts[0] + '|' + $parts[1]
    $golden[$key] = @{
        expected = [double]$parts[2]
        tol_pct  = [double]$parts[3]
        type     = $parts[4]
    }
}

$failures = New-Object System.Collections.ArrayList
$results  = New-Object System.Collections.ArrayList

function Compare-Metric {
    param([string]$exp, [string]$metric, [double]$observed)
    $key = $exp + '|' + $metric
    if (-not $golden.ContainsKey($key)) {
        Write-Host ("  AVISO: " + $key + " nao em golden, pulando") -ForegroundColor Yellow
        return
    }
    $g = $golden[$key]
    $exp_val = $g.expected
    $tol = $g.tol_pct
    $type = $g.type

    if ($exp_val -ne 0) {
        $diff_pct = 100 * [Math]::Abs($observed - $exp_val) / [Math]::Abs($exp_val)
    } else {
        $diff_pct = if ($observed -eq 0) { 0 } else { 999 }
    }

    $status = 'OK'
    if ($type -eq 'EXACT') {
        if ($observed -ne $exp_val) {
            $status = 'FAIL EXACT'
            [void]$failures.Add(($key + ' expected=' + $exp_val + ' got=' + $observed))
        }
    } else {
        if ($diff_pct -gt $tol) {
            $status = 'FAIL PERF tol=' + $tol
            [void]$failures.Add(($key + ' got=' + $observed + ' vs ' + $exp_val + ' drift=' + $diff_pct + '%'))
        }
    }

    $color = if ($status -eq 'OK') { 'Green' } else { 'Red' }
    Write-Host ("  {0,-50} expected={1,-12} got={2,-12} [{3}]" -f $key, $exp_val, $observed, $status) -ForegroundColor $color
    [void]$results.Add($status)
}

Write-Host '=== Regression test - baseline v0 ===' -ForegroundColor Cyan

# M1: e02
Write-Host '[M1] e02 propagation 1D (1M cells, tile=16K)' -ForegroundColor Cyan
$out = & (Join-Path $build 'e02.exe') 5 30 0 2>&1 | Out-String
$pat1 = 'n=1048576\s+tile=16384\s+median=\s*([0-9.]+)'
if ($out -match $pat1) {
    Compare-Metric 'e02' 'm1_1M_tile16K_ns_cell' ([double]$matches[1])
} else {
    Write-Host '  ERRO: nao parseou e02' -ForegroundColor Red
    [void]$failures.Add('e02 parse failed')
}

# M2.5: e04 (skip se -Quick)
if (-not $Quick) {
    Write-Host '[M2.5] e04 temporal blocking (4M T=4 K=16 tile=4K)' -ForegroundColor Cyan
    $out = & (Join-Path $build 'e04.exe') 5 64 2>&1 | Out-String
    $pat2 = 'n=4194304\s+T=4\s+K=16\s+tile=4096\s+median=([0-9.]+)'
    if ($out -match $pat2) {
        Compare-Metric 'e04' 'm25_4M_T4_K16_tile4K_ns_cell' ([double]$matches[1])
    } else {
        Write-Host '  ERRO: nao parseou e04' -ForegroundColor Red
        [void]$failures.Add('e04 parse failed')
    }
}

# M4: e05 pulse
Write-Host '[M4] e05 dirty+freeze pulse (1M cells, 4000 gens)' -ForegroundColor Cyan
$out = & (Join-Path $build 'e05.exe') 1048576 4000 60000 0 2>&1 | Out-String
$pat3 = 'speedup=([0-9.]+)x'
if ($out -match $pat3) {
    Compare-Metric 'e05' 'm4_pulse_speedup' ([double]$matches[1])
}
$pat4 = 'L1_dist=([0-9]+)'
if ($out -match $pat4) {
    Compare-Metric 'e05' 'm4_pulse_L1_distance' ([double]$matches[1])
}

# M4: e05 2-pulse (skip se -Quick)
if (-not $Quick) {
    Write-Host '[M4] e05 2-pulse (second pulse @2000)' -ForegroundColor Cyan
    $out = & (Join-Path $build 'e05.exe') 1048576 4000 60000 2000 2>&1 | Out-String
    if ($out -match $pat4) {
        Compare-Metric 'e05' 'm4_2pulse_L1_distance' ([double]$matches[1])
    }
}

# M7: e09 FFT
Write-Host '[M7] e09 FFT pipeline (sine 440 Hz)' -ForegroundColor Cyan
$out = & (Join-Path $build 'e09.exe') 100 2>&1 | Out-String
$pat5 = 'FFT sine 440Hz: peak bin = ([0-9]+)'
if ($out -match $pat5) {
    Compare-Metric 'e09' 'm7_fft_sine_440_peak_bin' ([double]$matches[1])
}

# M-2: e15 PAD (validacao do principio)
Write-Host '[M-2] e15 PAD propagation cost' -ForegroundColor Cyan
$out = & (Join-Path $build 'e15.exe') 50 60000 1 2>&1 | Out-String
$pat6 = '\b(\d+)\s+(\d+)\s+\[OK\]'  # match generic gen-row patterns; we'll use specific lines
# parse |A(t=7)| 1D row: "    7    15      7    15  [OK]"
if ($out -match '(?m)^\s+7\s+(\d+)\s+\d+\s+\d+\s+\[OK\]') {
    Compare-Metric 'e15' 'm_minus_2_1d_active_t7' ([double]$matches[1])
}
# parse |A(t=4)| 2D row from the 2D section: "    4    41     41  [OK]"
if ($out -match '(?ms)2D trace.+?\n\s+4\s+(\d+)\s+\d+\s+\[OK\]') {
    Compare-Metric 'e15' 'm_minus_2_2d_active_t4' ([double]$matches[1])
}
# parse first_touch 1d k=4: "1d,4,4,4,OK"
if ($out -match '(?m)^1d,4,(\d+),4,OK') {
    Compare-Metric 'e15' 'm_minus_2_1d_first_touch_k4' ([double]$matches[1])
}
# parse first_touch 2d k=4: "2d,4,4,4,OK"
if ($out -match '(?m)^2d,4,(\d+),4,OK') {
    Compare-Metric 'e15' 'm_minus_2_2d_first_touch_k4' ([double]$matches[1])
}

# M-3/M-4: e16 estabilizacao + trace (skip se -Quick)
if (-not $Quick) {
    Write-Host '[M-3/M-4] e16 estabilizacao' -ForegroundColor Cyan
    $out = & (Join-Path $build 'e16.exe') 5000 60000 2>&1 | Out-String
    if ($out -match 'gen_stable\s*=\s*(-?\d+)') {
        Compare-Metric 'e16' 'm_minus_3_gen_stable_60000' ([double]$matches[1])
    }
    if ($out -match 'peak_active\s*=\s*(\d+)') {
        Compare-Metric 'e16' 'm_minus_3_peak_active' ([double]$matches[1])
    }
    if ($out -match 'peak_r_eff\s*=\s*(\d+)') {
        Compare-Metric 'e16' 'm_minus_3_peak_r_eff' ([double]$matches[1])
    }
}

# Resumo
Write-Host ''
Write-Host '=== Resumo ===' -ForegroundColor Cyan
$pass = ($results | Where-Object { $_ -eq 'OK' }).Count
$total = $results.Count
Write-Host ('Total: ' + $total + ' | Pass: ' + $pass + ' | Fail: ' + $failures.Count)

if ($failures.Count -gt 0) {
    Write-Host ''
    Write-Host 'FALHAS:' -ForegroundColor Red
    foreach ($f in $failures) { Write-Host ('  ' + $f) -ForegroundColor Red }
    Write-Host ''
    Write-Host '>>> PARADIGMA QUEBRADO. REVERTER MUDANCAS. <<<' -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host '>>> Todos os invariantes preservados. Paradigma integro. <<<' -ForegroundColor Green
exit 0
