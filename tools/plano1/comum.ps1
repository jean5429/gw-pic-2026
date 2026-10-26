# Funcoes e caminhos compartilhados pelos lotes do Plano 1 ("mesma camera").
# Nao execute este arquivo sozinho: os lotes o carregam com ". $PSScriptRoot\comum.ps1".

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

if (-not $Python) {
    $Python = Join-Path $env:USERPROFILE "miniforge3\envs\gwproject\python.exe"
}
if (-not (Test-Path $Python)) {
    throw "Python do ambiente gwproject nao encontrado em $Python. Informe -Python <caminho do python.exe>."
}

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo

# Entradas existentes
$CleanMetadata   = ".\data\gravity_spy_harmonized\clean_metadata.csv"
$CurrentModels   = ".\models\gravity_spy_harmonized_v1d1"
$CurrentOod      = ".\outputs\research\gwtc4_218\ood_reference.npz"
$ApprovedCatalog = ".\outputs\gwtc4_218\approved_channels.csv"
$GsImages        = ".\data\gravity_spy\trainingset"
$GsMetadata      = ".\data\gravity_spy\trainingset_v1d1_metadata.csv"

# Saidas do Plano 1 (fora do controle de versao, como todo outputs/)
$Plano1     = ".\outputs\research\plano1"
$Closure    = "$Plano1\fechamento"
$ClosureCsv = "$Closure\closure_channels.csv"
$BandPanels = ".\data\gravity_spy_band20_512"
$BandModels = ".\models\gravity_spy_band20_512"
$BandOod    = "$Plano1\ood_reference_band20_512.npz"

New-Item -ItemType Directory -Force "$Plano1\logs" | Out-Null

function Invoke-Gw {
    # Roda "gwproject research <comando> ..." no Python do ambiente e para no primeiro erro.
    & $Python -m gwproject.research_cli @args
    if ($LASTEXITCODE -ne 0) { throw "Falhou: gwproject research $($args -join ' ')" }
}

function Start-Lote([string]$Nome) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = Join-Path $Plano1 "logs\${Nome}_$stamp.log"
    Start-Transcript -Path $log | Out-Null
    Write-Host "== $Nome  ($(Get-Date -Format 'dd/MM/yyyy HH:mm'))  log: $log" -ForegroundColor Cyan
}

function Stop-Lote([string]$Mensagem) {
    Write-Host $Mensagem -ForegroundColor Green
    Stop-Transcript | Out-Null
}

function Get-Checkpoints([string]$Root) {
    $found = @(Get-ChildItem $Root -Recurse -Filter best_model.pt | Sort-Object FullName | ForEach-Object { $_.FullName })
    if ($found.Count -ne 5) { throw "Esperados 5 checkpoints em $Root; encontrados $($found.Count)." }
    return $found
}

function Invoke-ClosureCondition(
    [string]$Nome, [string]$Models, [string]$Ood, [string]$Render,
    [string]$Csv = $ClosureCsv, [string]$Root = $Plano1, [string]$Config = ".\configs\plano1.toml"
) {
    # Uma condicao = (modelos, renderer) aplicada ao conjunto de fechamento + relatorio.
    $out = "$Root\condicoes\$Nome"
    $checkpoints = Get-Checkpoints $Models
    Write-Host "-- Condicao $Nome (renderer $Render)"
    Invoke-Gw classify-ensemble @checkpoints --approved $Csv --ood-reference $Ood `
        --render-mode $Render --output-dir $out
    Invoke-Gw closure-report $out --closure $Csv --label $Nome --config $Config
}
