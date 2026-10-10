param(
    [string]$GravitySpyMetadata = ".\data\gravity_spy\trainingset_v1d1_metadata.csv",
    [string]$GravitySpyImages = ".\data\gravity_spy\trainingset",
    [string]$ApprovedChannels = ".\outputs\gwtc\approved_channels.csv",
    [string]$Device = "cuda",
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONNOUSERSITE = "1"

$HarmonizedRoot = ".\data\gravity_spy\trainingset_harmonized_20_512"
$TrainingRoot = ".\outputs\gravity_spy_harmonized"
$ResearchRoot = ".\outputs\research"

New-Item $ResearchRoot -ItemType Directory -Force | Out-Null

Write-Host "[1/6] Harmonizando imagens Gravity Spy..."
python .\ondasgravitacionais_research_v9.py prepare-panels -- `
    $GravitySpyImages `
    $HarmonizedRoot `
    --overwrite
if ($LASTEXITCODE -ne 0) { throw "Falha no preparo dos paineis" }

if (-not $SkipTraining) {
    Write-Host "[2/6] Treinando cinco sementes no dominio harmonizado..."
    python -m scripts.run_gravity_spy_seeds `
        $GravitySpyMetadata `
        $HarmonizedRoot `
        --profile cuda_balanced `
        --output-dir $TrainingRoot
    if ($LASTEXITCODE -ne 0) { throw "Falha no treinamento multiseed" }
} else {
    Write-Host "[2/6] Treinamento ignorado por solicitacao."
}

Write-Host "[3/6] Agregando sementes..."
python -m scripts.aggregate_gravity_spy $TrainingRoot
if ($LASTEXITCODE -ne 0) { throw "Falha na agregacao" }

$Checkpoint42 = Get-ChildItem $TrainingRoot -Recurse -Filter best_model.pt |
    Where-Object { $_.FullName -match "seed_42" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $Checkpoint42) { throw "Checkpoint seed 42 nao encontrado" }

Write-Host "[4/6] Construindo referencia OOD..."
python .\ondasgravitacionais_research_v9.py build-ood -- `
    $Checkpoint42.FullName `
    $GravitySpyMetadata `
    $HarmonizedRoot `
    --device $Device `
    --output "$ResearchRoot\ood_reference_seed42.npz"
if ($LASTEXITCODE -ne 0) { throw "Falha na referencia OOD" }

Write-Host "[5/6] Classificando catalogo com ensemble..."
python .\ondasgravitacionais_research_v9.py classify-ensemble -- `
    $TrainingRoot `
    --approved $ApprovedChannels `
    --device $Device `
    --ood-reference "$ResearchRoot\ood_reference_seed42.npz" `
    --output-dir "$ResearchRoot\gravity_spy_ensemble"
if ($LASTEXITCODE -ne 0) { throw "Falha na inferencia ensemble" }

Write-Host "[6/6] Concluido. Consulte:"
Write-Host "$ResearchRoot\gravity_spy_ensemble\summary.json"
