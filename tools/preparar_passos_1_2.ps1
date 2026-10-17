# Passos 1 e 2 da reproducao da linha gwproject 2.0 com modelos proprios.
#
#   Passo 1: ambiente conda "gwproject" + PyTorch 2.5.1/CUDA 12.1 + testes.
#   Passo 2: paineis Gravity Spy com controle de qualidade (remove o painel vazio)
#            e comparacao com os artefatos canonicos do professor.
#
# Uso (PowerShell com conda disponivel, na raiz do repositorio):
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   & .\tools\preparar_passos_1_2.ps1
#
# Opcoes: -SkipEnv (ambiente ja pronto), -SkipTests, -EnvName <nome>.
param(
    [string]$EnvName = "gwproject",
    [string]$GravitySpyImages = ".\data\gravity_spy\trainingset",
    [string]$GravitySpyMetadata = ".\data\gravity_spy\trainingset_v1d1_metadata.csv",
    [string]$HarmonizedRoot = ".\data\gravity_spy_harmonized",
    [switch]$SkipEnv,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

function Invoke-InEnv {
    # Funcao simples (sem bloco param): repassa todos os argumentos, inclusive -e e -m.
    & conda run -n $EnvName --no-capture-output @args
    if ($LASTEXITCODE -ne 0) { throw "Falhou: $($args -join ' ')" }
}

if (-not (Test-Path ".\pyproject.toml") -or -not (Test-Path ".\environment.yml")) {
    throw "Execute este script na raiz do repositorio (onde estao pyproject.toml e environment.yml)."
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda nao encontrado. Abra o 'Miniforge Prompt' ou rode 'conda init powershell' e reabra o terminal."
}

# ---------------------------------------------------------------- Passo 1
if (-not $SkipEnv) {
    $envExists = (conda env list) -match "^\s*$([regex]::Escape($EnvName))\s"
    if ($envExists) {
        Write-Host "[1/5] Ambiente '$EnvName' ja existe; atualizando o pacote..."
        Invoke-InEnv python -m pip install -e ".[all,test]"
    } else {
        Write-Host "[1/5] Criando o ambiente '$EnvName' (environment.yml)..."
        & conda env create -n $EnvName -f environment.yml
        if ($LASTEXITCODE -ne 0) { throw "Falha ao criar o ambiente '$EnvName'." }
    }

    Write-Host "[2/5] Instalando PyTorch 2.5.1 + CUDA 12.1 (versao do treino canonico)..."
    & nvidia-smi | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi falhou. Verifique o driver NVIDIA." }
    Invoke-InEnv python -m pip uninstall -y torch torchvision torchaudio
    Invoke-InEnv python -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 `
        --index-url https://download.pytorch.org/whl/cu121
    Invoke-InEnv python -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponivel'; print('Torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"
} else {
    Write-Host "[1-2/5] Ambiente ignorado (-SkipEnv)."
}

if (-not $SkipTests) {
    Write-Host "[3/5] Rodando os testes..."
    Invoke-InEnv python -m pytest -m "not benchmark and not network" -q
} else {
    Write-Host "[3/5] Testes ignorados (-SkipTests)."
}

# ---------------------------------------------------------------- Passo 2
if (-not (Test-Path $GravitySpyImages) -or -not (Test-Path $GravitySpyMetadata)) {
    throw "Gravity Spy nao encontrado em $GravitySpyImages. Baixe com: conda run -n $EnvName python -m gwproject.commands.download_gravity_spy --extract"
}

Write-Host "[4/5] Preparando os paineis com controle de qualidade (alguns minutos)..."
Invoke-InEnv gwproject research prepare-panels $GravitySpyImages $HarmonizedRoot `
    --metadata $GravitySpyMetadata --overwrite

Write-Host "[5/5] Comparando com os artefatos canonicos do professor..."
$qc = Get-Content (Join-Path $HarmonizedRoot "qc_summary.json") -Raw | ConvertFrom-Json
$expected = @{ "metadata_rows" = 7966; "clean_samples" = 7807 }
$ok = $true
foreach ($key in $expected.Keys) {
    $value = $qc.$key
    $match = ($value -eq $expected[$key])
    if (-not $match) { $ok = $false }
    Write-Host ("  {0,-22} {1,6}  (esperado {2})  {3}" -f $key, $value, $expected[$key], $(if ($match) { "OK" } else { "DIFERENTE" }))
}
Write-Host ("  {0,-22} {1,6}" -f "excluded_samples", $qc.excluded_samples)
Write-Host ("  {0,-22} {1,6}" -f "blank_samples", $qc.blank_samples)
Write-Host ("  {0,-22} {1,6}" -f "qc_passed", $qc.qc_passed)

$canonical = @{}
foreach ($line in Get-Content ".\ARTIFACT_HASHES.sha256") {
    $parts = $line -split "\s+", 2
    if ($parts.Count -eq 2) { $canonical[$parts[1].Trim().Replace("\", "/")] = $parts[0].ToLower() }
}
foreach ($name in @("clean_metadata.csv", "qc_summary.json")) {
    $key = "data/gravity_spy_harmonized/$name"
    $actual = (Get-FileHash (Join-Path $HarmonizedRoot $name) -Algorithm SHA256).Hash.ToLower()
    $match = $canonical.ContainsKey($key) -and $canonical[$key] -eq $actual
    if ($name -eq "clean_metadata.csv" -and -not $match) { $ok = $false }
    Write-Host ("  sha256 {0,-20} {1}" -f $name, $(if ($match) { "IGUAL ao do professor" } else { "diferente" }))
}

Write-Host ""
if ($ok) {
    Write-Host "Passos 1 e 2 concluidos: a base de treino e identica a da linha 2.0." -ForegroundColor Green
    Write-Host "Proximo passo: treino das 5 seeds (gwproject research train-seeds)."
} else {
    Write-Host "Passos 1 e 2 concluidos, mas a base de treino difere da canonica. Envie a saida acima para revisao." -ForegroundColor Yellow
}
Write-Host "Obs.: qc_summary.json registra o caminho da pasta de saida; seu hash pode diferir sem que o conteudo mude."
