# Plano 1 v1.1 - Lote A: piloto de pixels (rede: ate ~7,8 GB; ~1 h; sem GPU)
#
# Para 82 glitches do split de TREINO em 12 arquivos de O1, gera a imagem por 4 variantes
# (receita atual; receita Gravity Spy a 4 kHz com 32 e 64 s; e a 16 kHz com 64 s) e mede
# a semelhanca com o painel oficial do mesmo glitch. Escolhe a variante pela regra de
# configs/plano1_v1_1.toml e grava outputs\research\plano1_v1_1\piloto\escolha.json.
#   powershell -ExecutionPolicy Bypass -File .\tools\plano1\v1_1\loteA_piloto.ps1
#   (-SoPlano mostra os arquivos e glitches do piloto, sem download)
param([string]$Python, [switch]$SoPlano)
. "$PSScriptRoot\..\comum.ps1"
$Plano11 = ".\outputs\research\plano1_v1_1"
New-Item -ItemType Directory -Force "$Plano11\piloto" | Out-Null
Start-Lote "v11_loteA_piloto"

if ($SoPlano) {
    Invoke-Gw closure-pilot --config .\configs\plano1_v1_1.toml --output-root "$Plano11\piloto" --dry-run
    Stop-Lote "Plano do piloto mostrado (nada foi baixado)."
    return
}
for ($tentativa = 1; $tentativa -le 3; $tentativa++) {
    & $Python -m gwproject.research_cli closure-pilot --config .\configs\plano1_v1_1.toml --output-root "$Plano11\piloto"
    if ($LASTEXITCODE -eq 0) { break }
    if ($tentativa -eq 3) { throw "Piloto falhou 3 vezes (veja o log)." }
    Write-Host "Falha (provavelmente conexao com o GWOSC); nova tentativa em 60 s."
    Start-Sleep -Seconds 60
}
$escolha = Get-Content "$Plano11\piloto\escolha.json" -Raw | ConvertFrom-Json
Stop-Lote ("Variante escolhida: {0} ({1} Hz, {2} s, receita {3}). Proximo: loteB_fechamento.ps1" -f `
    $escolha.name, $escolha.sample_rate, $escolha.seconds, $escolha.recipe)
