# Lote 3 - paineis em 20-512 Hz e retreino das 5 seeds (GPU, ~2 h)
#
# Recorta os PNGs Gravity Spy na mesma faixa dos Q-grams do catalogo (20-512 Hz), treina
# as 5 seeds, constroi a referencia OOD (ajuste no treino, calibracao na validacao) e
# confere a pre-condicao C0 no teste oficial. Nao toca nos modelos atuais.
#   powershell -ExecutionPolicy Bypass -File .\tools\plano1\lote3_treino_banda.ps1
#   (-PularTreino reaproveita modelos ja treinados em models\gravity_spy_band20_512)
param([string]$Python, [switch]$PularTreino)
. "$PSScriptRoot\comum.ps1"
Start-Lote "lote3_treino_banda"

$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"  # treino deterministico, como na linha 2.0

if (-not $PularTreino) {
    Write-Host "[1/5] Paineis recortados em 20-512 Hz"
    Invoke-Gw prepare-panels $GsImages $BandPanels --metadata $GsMetadata --apply-frequency-crop --overwrite

    Write-Host "[2/5] Treino das 5 seeds (~20 min cada)"
    Invoke-Gw train-seeds "$BandPanels\clean_metadata.csv" $BandPanels --profile cuda_balanced --output-dir $BandModels
} else {
    Write-Host "[1-2/5] Treino ignorado (-PularTreino)."
}

Write-Host "[3/5] Agregando as seeds"
Invoke-Gw aggregate-seeds $BandModels
$checkpoints = Get-Checkpoints $BandModels

Write-Host "[4/5] Referencia OOD por checkpoint"
Invoke-Gw build-ood @checkpoints --metadata "$BandPanels\clean_metadata.csv" --images-root $BandPanels --output $BandOod
Invoke-Gw validate-ood $BandOod --metadata "$BandPanels\clean_metadata.csv" --images-root $BandPanels `
    --output "$Plano1\ood_test_control_band20_512.json"

Write-Host "[5/5] Pre-condicao C0 (teste oficial, paineis recortados)"
$criteria = & $Python -c "import json,tomllib; print(json.dumps(tomllib.load(open('configs/plano1.toml','rb'))['criteria']))" | ConvertFrom-Json
$aggregate = Get-Content "$BandModels\aggregate_summary.json" -Raw | ConvertFrom-Json
$control = Get-Content "$Plano1\ood_test_control_band20_512.json" -Raw | ConvertFrom-Json
$f1 = [double]$aggregate.metrics.macro_f1.mean
$tail = ($control.models | ForEach-Object { [math]::Max([double]$_.mahalanobis_fraction_ge_0_99, [double]$_.energy_fraction_ge_0_99) } | Measure-Object -Maximum).Maximum
$okF1 = $f1 -ge [double]$criteria.band_model_test_macro_f1_min
$okTail = $tail -le [double]$criteria.band_model_ood_control_99_max
Write-Host ("  macro-F1 medio no teste: {0:N3} (minimo {1}) -> {2}" -f $f1, $criteria.band_model_test_macro_f1_min, $(if ($okF1) { "ok" } else { "NAO" }))
Write-Host ("  pior cauda OOD q99 no teste: {0:N3} (maximo {1}) -> {2}" -f $tail, $criteria.band_model_ood_control_99_max, $(if ($okTail) { "ok" } else { "NAO" }))
if (-not ($okF1 -and $okTail)) {
    Stop-Transcript | Out-Null
    throw "C0 nao atendida: o modelo em 20-512 Hz nao serve de base. Registre e discuta antes do lote 4."
}
Stop-Lote "Lote 3 ok (C0 atendida). Proximo: lote4_fechamento_banda.ps1"
