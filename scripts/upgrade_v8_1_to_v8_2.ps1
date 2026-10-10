$ErrorActionPreference = "Stop"

$python = (Get-Command python -ErrorAction Stop).Source
$prefix = python -c "import sys; print(sys.prefix)"
$version = python -c "import sys, struct; print('{}.{}.{} ({} bits)'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro, struct.calcsize('P')*8))"

Write-Host "Python efetivo: $python"
Write-Host "Prefixo Python: $prefix"
Write-Host "Python: $version"

python -m pip uninstall -y gwproject-v8
python -m pip install -e ".[gw,ml,test]"

Write-Host "Executando testes funcionais..."
python -m pytest -q

Write-Host "Reconstruindo indices para incluir redshift..."
python .\ondasgravitacionais_catalogo_v8.py index
python -m scripts.check_catalog_ready

Write-Host "Atualizacao para gwproject-v8 0.8.2 concluida."
Write-Host "Reexecute ridge e matched filter com --no-resume por causa da correcao de frame."
