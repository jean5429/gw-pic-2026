# Lote 5 - catalogo com a camera validada (exploratorio; GPU, ~5 min)
#
# SO rode se o lote 4 deu "camera validada". Aplica os modelos 20-512 Hz + renderer
# gravity_spy aos 475 canais dos 218 eventos, numa pasta propria. Resultado exploratorio:
# nao substitui a saida canonica nem e deteccao de ondas (a rede ainda nao conhece sinais;
# isso e o Plano 2).
#   powershell -ExecutionPolicy Bypass -File .\tools\plano1\lote5_catalogo_exploratorio.ps1
param([string]$Python)
. "$PSScriptRoot\comum.ps1"
Start-Lote "lote5_catalogo_exploratorio"

$report = "$Plano1\condicoes\B1_banda_gravity_spy\fechamento\report.json"
if (-not (Test-Path $report)) { throw "Rode o lote 4 antes." }
if (-not (Get-Content $report -Raw | ConvertFrom-Json).criteria.all_passed) {
    throw "A camera nao foi validada no lote 4; o lote 5 nao deve ser executado."
}

$out = "$Plano1\catalogo_banda_gravity_spy"
$checkpoints = Get-Checkpoints $BandModels
Invoke-Gw classify-ensemble @checkpoints --approved $ApprovedCatalog --ood-reference $BandOod `
    --render-mode gravity_spy --output-dir $out
$summary = Get-Content "$out\summary.json" -Raw | ConvertFrom-Json
Write-Host "Decisoes open-world nos 475 canais:"
$summary.open_world_counts | Format-List
Stop-Lote "Lote 5 ok. Resultado em $out (exploratorio)."
