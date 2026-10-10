$ErrorActionPreference = "Stop"

$pythonInfo = python -c "import sys,struct; print(sys.executable); print(sys.prefix); print(sys.version.split()[0]); print(struct.calcsize('P')*8)"
$lines = @($pythonInfo)
Write-Host "Python efetivo: $($lines[0])"
Write-Host "Prefixo Python: $($lines[1])"
Write-Host "Python: $($lines[2]) ($($lines[3]) bits)"

python -m pip uninstall -y gwproject-v8
python -m pip install -e ".[ml,gw,test]"

Write-Host "Executando testes funcionais..."
python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "Os testes da v9.0 falharam."
}

$version = python -c "import importlib.metadata as m; print(m.version('gwproject-v8'))"
if ($version.Trim() -ne "0.9.0") {
    throw "Versao instalada inesperada: $version"
}

Write-Host "Atualizacao para gwproject-v8 0.9.0 concluida."
Write-Host "Leia research\GUIA_DE_EXECUCAO.md antes de executar o treinamento harmonizado."
