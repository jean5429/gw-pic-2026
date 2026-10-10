$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Description,

        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host $Description
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description falhou com codigo $LASTEXITCODE."
    }
}

# Nao dependa somente de CONDA_DEFAULT_ENV. Ao iniciar um PowerShell filho,
# essa variavel pode nao refletir o ambiente que fornece o executavel Python.
$pythonCommand = Get-Command python -ErrorAction Stop
$pythonExecutable = (& python -c "import sys; print(sys.executable)").Trim()
$pythonPrefix = (& python -c "import sys; print(sys.prefix)").Trim()
$pythonVersion = (& python -c "import platform; print(platform.python_version())").Trim()
$pythonBits = (& python -c "import struct; print(struct.calcsize('P') * 8)").Trim()

if ($LASTEXITCODE -ne 0 -or -not $pythonExecutable) {
    throw "Nao foi possivel executar o Python ativo. Ative o ambiente Conda e tente novamente."
}

Write-Host "Ambiente informado pelo Conda: $env:CONDA_DEFAULT_ENV"
Write-Host "Python efetivo: $pythonExecutable"
Write-Host "Prefixo Python: $pythonPrefix"
Write-Host "Python: $pythonVersion ($pythonBits bits)"

if ($pythonBits -ne "64") {
    throw "A v8 requer Python de 64 bits."
}

if (-not $pythonVersion.StartsWith("3.11")) {
    Write-Warning "A entrega foi validada com Python 3.11. O Python ativo e $pythonVersion."
}

$normalizedExecutable = $pythonExecutable.ToLowerInvariant().Replace('/', '\\')
$expectedEnvironment = (
    $normalizedExecutable -like "*\\envs\\gw-v7\\python.exe" -or
    $normalizedExecutable -like "*\\envs\\gw-v8\\python.exe"
)

if (-not $expectedEnvironment) {
    Write-Warning "O Python ativo nao pertence a gw-v7/gw-v8. A instalacao continuara no ambiente mostrado acima."
}

Invoke-Checked "Removendo metadados editaveis antigos, sem apagar codigo ou outputs..." {
    python -m pip uninstall -y gwproject-v7 gwproject-v8
}

Invoke-Checked "Instalando o pacote v8 em modo editavel, sem apagar outputs..." {
    python -m pip install -e ".[gw,ml,test]"
}

Invoke-Checked "Executando testes funcionais..." {
    python -m pytest
}

Invoke-Checked "Reconstruindo indices a partir dos dados existentes..." {
    python .\ondasgravitacionais_catalogo_v8.py index
}

Invoke-Checked "Verificando arquivos aprovados..." {
    python -m scripts.check_catalog_ready
}

Write-Host "Atualizacao concluida. Os dados em outputs\gwtc foram preservados."
