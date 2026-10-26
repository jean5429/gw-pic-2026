# Lote 0 - verificacao (5 min, sem rede, sem GPU obrigatoria)
#
# Confere ambiente, entradas, espaco em disco e roda os testes. Nada e gravado alem do log.
#   powershell -ExecutionPolicy Bypass -File .\tools\plano1\lote0_verificacao.ps1
param([string]$Python)
. "$PSScriptRoot\comum.ps1"
Start-Lote "lote0_verificacao"

Write-Host "[1/4] Ambiente"
& $Python -c "import gwproject, gwpy, gwosc, torch; print('gwpy', gwpy.__version__, '| gwosc', gwosc.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Ambiente incompleto." }

Write-Host "[2/4] Entradas"
foreach ($path in @($CleanMetadata, $CurrentModels, $CurrentOod, $ApprovedCatalog, $GsImages, $GsMetadata)) {
    if (-not (Test-Path $path)) { throw "Falta: $path" }
    Write-Host "  ok  $path"
}
Get-Checkpoints $CurrentModels | Out-Null

Write-Host "[3/4] Espaco em disco"
$drive = (Get-Item $Repo).PSDrive
$freeGb = [math]::Round($drive.Free / 1GB, 1)
# Um arquivo de O1 (~130 MB) por vez + recortes/Q-grams (~3 GB) + paineis e modelos novos (~1 GB).
Write-Host "  livre em ${drive}: $freeGb GB (minimo recomendado: 8 GB)"
if ($freeGb -lt 8) { throw "Espaco insuficiente." }

Write-Host "[4/4] Testes (sem rede)"
& $Python -m pytest -m "not benchmark and not network" -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw "Testes falharam; nao siga para o lote 1." }

Stop-Lote "Lote 0 ok. Proximo: lote1_conjunto_fechamento.ps1"
