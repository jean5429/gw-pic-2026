param(
    [string]$PythonExecutable = "$env:USERPROFILE\anaconda3\envs\gw-v7\python.exe",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [ValidateRange(1, 64)][int]$BatchSize = 16,
    [string]$RunDirectory = "",
    [switch]$Resume
)

# No installation, download, training or canonical-output replacement.
# Run this file yourself. It was prepared without executing it.
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python nao encontrado: $PythonExecutable. Informe -PythonExecutable com o ambiente gw-v7."
}
if ($Resume -and [string]::IsNullOrWhiteSpace($RunDirectory)) {
    throw "Para retomar, informe -Resume -RunDirectory com a pasta exata da execucao anterior."
}

$PreviousPythonPath = $env:PYTHONPATH
$PreviousNoUserSite = $env:PYTHONNOUSERSITE
$PreviousCublas = $env:CUBLAS_WORKSPACE_CONFIG
$PreviousOmp = $env:OMP_NUM_THREADS
$PreviousMkl = $env:MKL_NUM_THREADS
$TranscriptStarted = $false
Push-Location -LiteralPath $ProjectRoot
try {
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    $env:OMP_NUM_THREADS = "4"
    $env:MKL_NUM_THREADS = "4"
    if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
        $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $Suffix = [guid]::NewGuid().ToString("N").Substring(0, 8)
        $RunDirectory = Join-Path $ProjectRoot "outputs\research\paired_intensity_${Stamp}_${Suffix}"
    } else {
        if (-not [System.IO.Path]::IsPathRooted($RunDirectory)) {
            $RunDirectory = Join-Path $ProjectRoot $RunDirectory
        }
        $RunDirectory = [System.IO.Path]::GetFullPath($RunDirectory)
    }
    $AllowedParent = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "outputs\research"))
    if ((Split-Path $RunDirectory -Parent) -ne $AllowedParent -or
        (Split-Path $RunDirectory -Leaf) -notlike "paired_intensity_*") {
        throw "Use uma pasta direta outputs\research\paired_intensity_*, separada dos resultados anteriores."
    }
    $LogRoot = Join-Path $AllowedParent "logs"
    New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
    $LogStamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $LogPath = Join-Path $LogRoot ((Split-Path $RunDirectory -Leaf) + "_${LogStamp}.log")
    Start-Transcript -Path $LogPath | Out-Null
    $TranscriptStarted = $true
    Write-Host "Pasta desta execucao: $RunDirectory"
    Write-Host "Log: $LogPath"
    Write-Host "[1/2] Testes locais (sem benchmark e sem rede). Falhas interrompem o experimento."
    & $PythonExecutable -m pytest -m "not benchmark and not network"
    if ($LASTEXITCODE -ne 0) { throw "Testes falharam. Envie o log; a inferencia nao foi iniciada." }

    Write-Host "[2/2] Experimento pareado: 218 eventos / 475 canais / 1177 imagens de validacao."
    $Arguments = @("-m", "gwproject.research_cli", "paired-intensity",
        "--config", "configs/paired_intensity.toml", "--output-dir", $RunDirectory,
        "--device", $Device, "--batch-size", "$BatchSize")
    if ($Resume) { $Arguments += "--resume" }
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Experimento interrompido. Preserve a pasta e envie o log. Nao interprete arquivos parciais."
    }
    Write-Host "Concluido: $RunDirectory\REPORT.md"
    Write-Host "Envie REPORT.md, summary.json, paired_contrasts.csv e este log."
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
    $env:PYTHONPATH = $PreviousPythonPath
    $env:PYTHONNOUSERSITE = $PreviousNoUserSite
    $env:CUBLAS_WORKSPACE_CONFIG = $PreviousCublas
    $env:OMP_NUM_THREADS = $PreviousOmp
    $env:MKL_NUM_THREADS = $PreviousMkl
    Pop-Location
}
