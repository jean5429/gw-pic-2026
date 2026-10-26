# Plano 1 v1.1 - Lote B: fechamento com a variante escolhida e veredito
#
# Remonta o conjunto de fechamento (validacao + teste de O1) com a receita, a janela e a
# taxa escolhidas no piloto, e avalia sem retreino:
#   B1v11 = modelos 20-512 Hz da v1.0 + renderer gravity_spy   (decide)
#   A2v11 = modelos atuais + renderer gravity_spy              (comparacao)
# Rede: ate ~19,5 GB a 4 kHz. Se o piloto escolher 16 kHz, ate ~78 GB: exige -Aceito16k.
#   powershell -ExecutionPolicy Bypass -File .\tools\plano1\v1_1\loteB_fechamento.ps1
param([string]$Python, [switch]$Aceito16k)
. "$PSScriptRoot\..\comum.ps1"
$Plano11 = ".\outputs\research\plano1_v1_1"
$Closure11 = "$Plano11\fechamento"
$Csv11 = "$Closure11\closure_channels.csv"
$Config11 = ".\configs\plano1_v1_1.toml"
Start-Lote "v11_loteB_fechamento"

$escolhaPath = "$Plano11\piloto\escolha.json"
if (-not (Test-Path $escolhaPath)) { throw "Rode o lote A antes (falta $escolhaPath)." }
$escolha = Get-Content $escolhaPath -Raw | ConvertFrom-Json
if ([int]$escolha.sample_rate -gt 4096 -and -not $Aceito16k) {
    Stop-Transcript | Out-Null
    throw "O piloto escolheu 16 kHz (download de ate ~78 GB). Rode de novo com -Aceito16k para confirmar."
}
Write-Host ("Variante: {0} ({1} Hz, {2} s, receita {3})" -f $escolha.name, $escolha.sample_rate, $escolha.seconds, $escolha.recipe)

for ($tentativa = 1; $tentativa -le 4; $tentativa++) {
    Write-Host "-- Rodada $tentativa de 4 do conjunto de fechamento"
    & $Python -m gwproject.research_cli closure-set --config $Config11 --output-root $Closure11 `
        --sample-rate $escolha.sample_rate --window-seconds $escolha.seconds --recipe $escolha.recipe
    if ($LASTEXITCODE -eq 0) { break }
    if ($tentativa -eq 4) { throw "Conjunto de fechamento incompleto apos 4 rodadas; rode o lote B de novo." }
    Start-Sleep -Seconds 60
}

Invoke-ClosureCondition "B1v11_banda_gravity_spy" $BandModels $BandOod "gravity_spy" $Csv11 $Plano11 $Config11
Invoke-ClosureCondition "A2v11_atual_gravity_spy" $CurrentModels $CurrentOod "gravity_spy" $Csv11 $Plano11 $Config11

Write-Host ""
Write-Host "Condicao                   n    acuracia  macro-F1  reject_ood  veredito"
foreach ($name in @("B1v11_banda_gravity_spy", "A2v11_atual_gravity_spy")) {
    $r = Get-Content "$Plano11\condicoes\$name\fechamento\report.json" -Raw | ConvertFrom-Json
    Write-Host ("{0,-26} {1,4}  {2,8:N3}  {3,8:N3}  {4,10:N3}  {5}" -f $name, $r.evaluation.n, `
        $r.evaluation.accuracy, $r.evaluation.macro_f1, $r.evaluation.open_world_fractions.reject_ood, `
        $(if ($r.criteria.all_passed) { "PASSOU" } else { "nao passou" }))
}
$primary = Get-Content "$Plano11\condicoes\B1v11_banda_gravity_spy\fechamento\report.json" -Raw | ConvertFrom-Json
Write-Host ""
if ($primary.criteria.all_passed) {
    Stop-Lote "VEREDITO v1.1: camera validada (B1v11 passou em C1 e C2)."
} else {
    Stop-Lote "VEREDITO v1.1: camera NAO validada. Leia os REPORT.md por classe."
}
