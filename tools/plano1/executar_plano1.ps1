# Executa os lotes 0 a 4 do Plano 1 em sequencia, parando no primeiro erro.
# O lote 5 (catalogo) fica de fora de proposito: so roda depois de ler o veredito.
#
#   powershell -ExecutionPolicy Bypass -File .\tools\plano1\executar_plano1.ps1
#   -APartirDe 2   recomeca de um lote (ex.: depois de uma queda no lote 1, use 1)
param([string]$Python, [int]$APartirDe = 0)
$ErrorActionPreference = "Stop"

$lotes = @(
    "lote0_verificacao.ps1",
    "lote1_conjunto_fechamento.ps1",
    "lote2_linha_de_base.ps1",
    "lote3_treino_banda.ps1",
    "lote4_fechamento_banda.ps1"
)
for ($i = $APartirDe; $i -lt $lotes.Count; $i++) {
    $arguments = @{}
    if ($Python) { $arguments["Python"] = $Python }
    & (Join-Path $PSScriptRoot $lotes[$i]) @arguments
}
