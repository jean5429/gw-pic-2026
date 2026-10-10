$ErrorActionPreference = "Stop"

$pythonExe = (Get-Command python -ErrorAction Stop).Source
$pythonPrefix = python -c "import sys; print(sys.prefix)"
$pythonVersion = python -c "import sys, struct; print('{}.{}.{} ({} bits)'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro, struct.calcsize('P') * 8))"
Write-Host "Python efetivo: $pythonExe"
Write-Host "Prefixo Python: $pythonPrefix"
Write-Host "Python: $pythonVersion"

python -m pip uninstall -y gwproject-v8
python -m pip install -e ".[gw,ml,test]"
Write-Host "Executando testes funcionais..."
python -m pytest
Write-Host "Atualizacao para gwproject-v8 0.8.3 concluida."
Write-Host "Proximo passo: python .\ondasgravitacionais_catalogo_v8.py refresh-metadata -- --workers 4"
