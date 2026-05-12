# build.ps1 — builds bench.exe directly with g++ (w64devkit / mingw)
# avoids CMake; target is Release with AVX2 (Zen 2 Mendocino, no AVX-512)

[CmdletBinding()]
param(
    [string]$Compiler = "C:\w64devkit\bin\g++.exe",
    [string]$Source   = "benchmarks/bench.cpp",
    [string]$Out      = "",
    [switch]$Dbg,
    [switch]$Lib       # builds as liblc.dll + liblc.a
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/.." | Select-Object -ExpandProperty Path
$src  = Join-Path $root $Source
if ([string]::IsNullOrEmpty($Out)) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Source)
    $Out = Join-Path $root "build/$stem.exe"
}
$buildDir = Split-Path $Out -Parent
if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir -Force | Out-Null }

if (-not (Test-Path $Compiler)) {
    Write-Host "ERROR: compiler not found: $Compiler" -ForegroundColor Red
    exit 1
}

$flags = @(
    "-std=c++17",
    "-O3", "-DNDEBUG",
    "-mavx2", "-mfma",
    "-fno-omit-frame-pointer",
    "-Wall", "-Wextra",
    "-pipe",
    "-Iinclude","-Isrc"
)
if ($Dbg) {
    $flags = @("-std=c++17","-O0","-g","-mavx2","-mfma","-Wall","-Wextra","-Iinclude","-Isrc")
}

$libs = @("-ladvapi32","-static-libgcc","-static-libstdc++")

if ($Lib) {
    # builds shared library (liblc.dll + liblc.a)
    $dllOut = Join-Path $buildDir "liblc.dll"
    $impOut = Join-Path $buildDir "liblc.a"
    $flags += @("-shared","-fPIC","-Wl,--out-implib=$impOut")
    $argList = $flags + @("-o", $dllOut, $src) + $libs
} else {
    $argList = $flags + @("-o", $Out, $src) + $libs
}
Write-Host ">> $Compiler $($argList -join ' ')" -ForegroundColor Cyan

# w64devkit needs its own directory on PATH so g++ can find as.exe, ld.exe etc
$compilerDir = Split-Path $Compiler -Parent
$origPath = $env:PATH
$env:PATH = "$compilerDir;$origPath"
try {
    $p = Start-Process -FilePath $Compiler -ArgumentList $argList -NoNewWindow -PassThru -Wait `
            -WorkingDirectory $root
} finally {
    $env:PATH = $origPath
}
if ($p.ExitCode -ne 0) {
    Write-Host "BUILD FAILED (exit $($p.ExitCode))" -ForegroundColor Red
    exit $p.ExitCode
}

if ($Lib) {
    $dllOut = Join-Path $buildDir "liblc.dll"
    $impOut = Join-Path $buildDir "liblc.a"
    $sz = (Get-Item $dllOut).Length
    $isz = (Get-Item $impOut).Length
    Write-Host "OK: $dllOut ($sz bytes) + $impOut ($isz bytes)" -ForegroundColor Green
} else {
    $sz = (Get-Item $Out).Length
    Write-Host "OK: $Out  ($sz bytes)" -ForegroundColor Green
}
