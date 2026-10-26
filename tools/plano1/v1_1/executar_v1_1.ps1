# Executa os lotes A (piloto) e B (fechamento + veredito) do Plano 1 v1.1 em sequencia.
#   powershell -ExecutionPolicy Bypass -File .\tools\plano1\v1_1\executar_v1_1.ps1
#   -Aceito16k   autoriza o lote B a baixar a 16 kHz se o piloto escolher essa variante
param([string]$Python, [switch]$Aceito16k)
$ErrorActionPreference = "Stop"
$common = @{}
if ($Python) { $common["Python"] = $Python }
& (Join-Path $PSScriptRoot "loteA_piloto.ps1") @common
$extra = $common.Clone()
if ($Aceito16k) { $extra["Aceito16k"] = $true }
& (Join-Path $PSScriptRoot "loteB_fechamento.ps1") @extra
