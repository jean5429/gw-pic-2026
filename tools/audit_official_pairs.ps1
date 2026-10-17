param([string]$PythonExecutable = "$env:USERPROFILE\anaconda3\envs\gw-v7\python.exe")

# Prepared only; execution and validation are performed by the project owner.
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$PreviousPythonPath = $env:PYTHONPATH
$PreviousNoUserSite = $env:PYTHONNOUSERSITE
$TranscriptStarted = $false
Push-Location -LiteralPath $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Python nao encontrado." }
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    $env:PYTHONNOUSERSITE = "1"
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Suffix = [guid]::NewGuid().ToString("N").Substring(0, 8)
    $Name = "official_pairs_${Stamp}_${Suffix}"
    $OutputDirectory = Join-Path $ProjectRoot "outputs\research\$Name"
    $LogRoot = Join-Path $ProjectRoot "outputs\research\logs"
    New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
    $LogPath = Join-Path $LogRoot "$Name.log"
    Start-Transcript -Path $LogPath | Out-Null
    $TranscriptStarted = $true
    Write-Host "Log: $LogPath"
    Write-Host "[1/2] Testes locais; nenhum download ou treinamento."
    & $PythonExecutable -m pytest -m "not benchmark and not network"
    if ($LASTEXITCODE -ne 0) { throw "Testes falharam; auditoria nao iniciada. Envie o log." }
    Write-Host "[2/2] Auditoria completa de chaves e pixels somente da validacao."
    & $PythonExecutable -m gwproject.research_cli audit-official-pairs --output-dir $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw "Auditoria interrompida. Preserve os arquivos e envie o log." }
    Write-Host "Envie: $OutputDirectory\summary.json"
    Write-Host "E o log: $LogPath"
    Write-Host "coverage_gate=false indica pendencias de pareamento, mesmo se a auditoria concluiu."
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
    $env:PYTHONPATH = $PreviousPythonPath
    $env:PYTHONNOUSERSITE = $PreviousNoUserSite
    Pop-Location
}
