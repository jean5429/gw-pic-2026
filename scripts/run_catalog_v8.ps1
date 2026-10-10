param(
    [ValidateSet("Smoke", "Full")]
    [string]$Mode = "Smoke",
    [int]$RidgeWorkers = 2,
    [int]$MatchedWorkers = 1,
    [switch]$SkipMatchedFilter
)

$ErrorActionPreference = "Stop"

if ($env:CONDA_DEFAULT_ENV -notin @("gw-v7", "gw-v8")) {
    Write-Warning "Ambiente atual: $env:CONDA_DEFAULT_ENV. Ative gw-v7 ou gw-v8 antes de continuar."
}

Write-Host "[1/4] Reconstruindo indice do catalogo..."
python .\ondasgravitacionais_catalogo_v8.py index
if ($LASTEXITCODE -ne 0) { throw "Falha ao construir o indice" }

Write-Host "[2/4] Verificando os arquivos aprovados..."
python -m scripts.check_catalog_ready
if ($LASTEXITCODE -ne 0) { throw "Ha caminhos ausentes em approved_channels.csv" }

if ($Mode -eq "Smoke") {
    Write-Host "[3/4] Smoke test do ridge em dois eventos..."
    python .\ondasgravitacionais_catalogo_v8.py ridge -- --limit 2 --workers $RidgeWorkers --no-resume
} else {
    Write-Host "[3/4] Ridge completo do catalogo..."
    python .\ondasgravitacionais_catalogo_v8.py ridge -- --workers $RidgeWorkers
}
if ($LASTEXITCODE -ne 0) { throw "Falha na analise de ridge" }

if (-not $SkipMatchedFilter) {
    if ($Mode -eq "Smoke") {
        Write-Host "[4/4] Smoke test do matched filter em dois eventos..."
        python .\ondasgravitacionais_catalogo_v8.py matched-filter -- --limit 2 --workers $MatchedWorkers --no-resume
    } else {
        Write-Host "[4/4] Matched filter completo do catalogo..."
        python .\ondasgravitacionais_catalogo_v8.py matched-filter -- --workers $MatchedWorkers
    }
    if ($LASTEXITCODE -ne 0) { throw "Falha no matched filter do catalogo" }
} else {
    Write-Host "[4/4] Matched filter ignorado por solicitacao."
}

Write-Host "Pipeline concluido."
