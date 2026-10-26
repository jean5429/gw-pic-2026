# Lote 2 - linha de base (GPU, ~10 min)
#
# Modelos ATUAIS (treinados no painel 10-2048 Hz) sobre o conjunto de fechamento, nos dois
# renderers. Mede quanto a "camera" atual perde quando a classe verdadeira e conhecida.
# Espera-se que falhe: a faixa de frequencia ainda nao foi igualada.
#   powershell -ExecutionPolicy Bypass -File .\tools\plano1\lote2_linha_de_base.ps1
param([string]$Python)
. "$PSScriptRoot\comum.ps1"
Start-Lote "lote2_linha_de_base"

if (-not (Test-Path $ClosureCsv)) { throw "Rode o lote 1 antes (falta $ClosureCsv)." }
Invoke-ClosureCondition "A1_atual_percentile"  $CurrentModels $CurrentOod "percentile"
Invoke-ClosureCondition "A2_atual_gravity_spy" $CurrentModels $CurrentOod "gravity_spy"

Stop-Lote "Lote 2 ok. Relatorios em $Plano1\condicoes\A*\fechamento\REPORT.md. Proximo: lote3_treino_banda.ps1"
