param(
    [ValidateSet("Smoke", "Full")]
    [string]$Mode = "Smoke",
    [int]$RidgeWorkers = 2,
    [int]$MatchedWorkers = 1,
    [switch]$SkipMatchedFilter
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

gwproject catalog index
if ($LASTEXITCODE -ne 0) { throw "Falha ao construir os indices" }

python -m gwproject.commands.check_catalog_ready
if ($LASTEXITCODE -ne 0) { throw "Catalogo nao esta pronto" }

if ($Mode -eq "Smoke") {
    gwproject catalog ridge -- --limit 2 --workers $RidgeWorkers --no-resume
} else {
    gwproject catalog ridge -- --workers $RidgeWorkers
}
if ($LASTEXITCODE -ne 0) { throw "Falha na analise de ridge" }

if (-not $SkipMatchedFilter) {
    if ($Mode -eq "Smoke") {
        gwproject catalog matched-filter -- --limit 2 --workers $MatchedWorkers --no-resume
    } else {
        gwproject catalog matched-filter -- --workers $MatchedWorkers
    }
    if ($LASTEXITCODE -ne 0) { throw "Falha no matched filtering" }
}

gwproject catalog diagnose-ridge
