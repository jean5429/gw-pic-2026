# Lote 1 - conjunto de fechamento (rede: ~19,5 GB baixados, um arquivo por vez; 4-8 h)
#
# Baixa do GWOSC os 150 arquivos de O1 escolhidos em configs/plano1.toml, recorta 32 s em
# torno de cada glitch Gravity Spy rotulado e gera o Q-gram pela mesma cadeia dos eventos.
# Cada arquivo baixado e apagado logo depois. Se cair a internet, rode de novo: continua
# de onde parou.
#   powershell -ExecutionPolicy Bypass -File .\tools\plano1\lote1_conjunto_fechamento.ps1
#   (-SoPlano mostra quais arquivos seriam baixados, sem rede)
param([string]$Python, [switch]$SoPlano)
. "$PSScriptRoot\comum.ps1"
Start-Lote "lote1_conjunto_fechamento"

if ($SoPlano) {
    Invoke-Gw closure-set --output-root $Closure --dry-run
    Stop-Lote "Plano gravado em $Closure\plan_files.csv (nada foi baixado)."
    return
}

# O GWOSC as vezes corta a conexao; cada rodada refaz so os arquivos que falharam.
for ($tentativa = 1; $tentativa -le 4; $tentativa++) {
    Write-Host "-- Rodada $tentativa de 4"
    & $Python -m gwproject.research_cli closure-set --output-root $Closure
    $code = $LASTEXITCODE
    if ($code -eq 0) { break }
    Start-Sleep -Seconds 60
}
$summary = Get-Content "$Closure\summary.json" -Raw | ConvertFrom-Json
Write-Host ("  arquivos concluidos: {0} de {1} | Q-grams ok: {2} de {3}" -f `
    $summary.files_done, $summary.files, $summary.qgrams_ok, $summary.glitches_processed)
if ($code -ne 0) {
    throw "Lote 1 incompleto (falha de rede ou de arquivo). Rode o lote 1 de novo para retomar."
}
Stop-Lote "Lote 1 ok. Proximo: lote2_linha_de_base.ps1"
