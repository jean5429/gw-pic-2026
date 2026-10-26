# Lote 4 - teste de fechamento dos modelos 20-512 Hz e veredito (GPU, ~10 min)
#
# Condicao principal: B1 = modelos 20-512 Hz + renderer gravity_spy (mesma intensidade dos
# PNGs). B2 (percentile) e secundaria. Imprime a tabela das quatro condicoes e o veredito
# dos criterios C1/C2 de configs/plano1.toml.
#   powershell -ExecutionPolicy Bypass -File .\tools\plano1\lote4_fechamento_banda.ps1
param([string]$Python)
. "$PSScriptRoot\comum.ps1"
Start-Lote "lote4_fechamento_banda"

if (-not (Test-Path $BandOod)) { throw "Rode o lote 3 antes (falta $BandOod)." }
Invoke-ClosureCondition "B1_banda_gravity_spy" $BandModels $BandOod "gravity_spy"
Invoke-ClosureCondition "B2_banda_percentile"  $BandModels $BandOod "percentile"

Write-Host ""
Write-Host "Condicao               n    acuracia  macro-F1  reject_ood  veredito"
foreach ($name in @("A1_atual_percentile", "A2_atual_gravity_spy", "B1_banda_gravity_spy", "B2_banda_percentile")) {
    $path = "$Plano1\condicoes\$name\fechamento\report.json"
    if (-not (Test-Path $path)) { Write-Host ("{0,-22} (nao executada)" -f $name); continue }
    $r = Get-Content $path -Raw | ConvertFrom-Json
    Write-Host ("{0,-22} {1,4}  {2,8:N3}  {3,8:N3}  {4,10:N3}  {5}" -f $name, $r.evaluation.n, `
        $r.evaluation.accuracy, $r.evaluation.macro_f1, $r.evaluation.open_world_fractions.reject_ood, `
        $(if ($r.criteria.all_passed) { "PASSOU" } else { "nao passou" }))
}
$primary = Get-Content "$Plano1\condicoes\B1_banda_gravity_spy\fechamento\report.json" -Raw | ConvertFrom-Json
Write-Host ""
if ($primary.criteria.all_passed) {
    Stop-Lote "VEREDITO: camera validada (B1 passou em C1 e C2). Pode seguir para o lote 5."
} else {
    Stop-Lote "VEREDITO: camera NAO validada. Nao rode o lote 5; leia os REPORT.md por classe."
}
