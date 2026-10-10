param(
    [string]$GravitySpyMetadata = ".\data\gravity_spy\trainingset_v1d1_metadata.csv",
    [string]$GravitySpyImages = ".\data\gravity_spy\trainingset",
    [string]$ApprovedChannels = ".\outputs\gwtc4_218\approved_channels.csv",
    [string]$Device = "cuda",
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONNOUSERSITE = "1"

$HarmonizedRoot = ".\data\gravity_spy_harmonized"
$CleanMetadata = "$HarmonizedRoot\clean_metadata.csv"
$TrainingRoot = ".\models\gravity_spy_harmonized_v1d1"
$ResearchRoot = ".\outputs\research\gwtc4_218"
$OodReference = "$ResearchRoot\ood_reference.npz"

New-Item $ResearchRoot -ItemType Directory -Force | Out-Null

Write-Host "[1/10] Harmonizando imagens Gravity Spy..."
gwproject research prepare-panels `
    $GravitySpyImages `
    $HarmonizedRoot `
    --metadata $GravitySpyMetadata `
    --overwrite
if ($LASTEXITCODE -ne 0) { throw "Falha no preparo dos paineis" }

if (-not $SkipTraining) {
    Write-Host "[2/10] Treinando ensemble multiseed..."
    gwproject research train-seeds `
        $CleanMetadata `
        $HarmonizedRoot `
        --profile cuda_balanced `
        --output-dir $TrainingRoot
    if ($LASTEXITCODE -ne 0) { throw "Falha no treinamento multiseed" }
} else {
    Write-Host "[2/10] Treinamento ignorado por solicitacao."
}

Write-Host "[3/10] Agregando sementes..."
gwproject research aggregate-seeds $TrainingRoot
if ($LASTEXITCODE -ne 0) { throw "Falha na agregacao" }

$Checkpoints = Get-ChildItem $TrainingRoot -Recurse -Filter best_model.pt | Sort-Object FullName
if (-not $Checkpoints -or $Checkpoints.Count -lt 2) {
    throw "Sao necessarios ao menos dois checkpoints para o ensemble"
}
$CheckpointPaths = @($Checkpoints | ForEach-Object { $_.FullName })

Write-Host "[4/10] Construindo referencias OOD por checkpoint..."
& gwproject research build-ood @CheckpointPaths `
    --metadata $CleanMetadata `
    --images-root $HarmonizedRoot `
    --device $Device `
    --output $OodReference
if ($LASTEXITCODE -ne 0) { throw "Falha na referencia OOD" }

Write-Host "[5/10] Validando calibracao OOD no teste independente..."
gwproject research validate-ood `
    $OodReference `
    --metadata $CleanMetadata `
    --images-root $HarmonizedRoot `
    --device $Device `
    --output "$ResearchRoot\ood_test_control.json"
if ($LASTEXITCODE -ne 0) { throw "Falha no controle independente OOD" }

Write-Host "[6/10] Classificando catalogo com ensemble..."
& gwproject research classify-ensemble @CheckpointPaths `
    --approved $ApprovedChannels `
    --device $Device `
    --ood-reference $OodReference `
    --output-dir "$ResearchRoot\gravity_spy_ensemble"
if ($LASTEXITCODE -ne 0) { throw "Falha na inferencia ensemble" }

Write-Host "[7/10] Calculando estatisticas da entrada renderizada..."
gwproject research catalog-statistics `
    "$ResearchRoot\gravity_spy_ensemble\per_channel.csv" `
    --output "$ResearchRoot\catalog_rendered_statistics.csv"
if ($LASTEXITCODE -ne 0) { throw "Falha nas estatisticas catalogais" }

Write-Host "[8/10] Auditando atalhos de baixo nivel..."
gwproject research audit-domain `
    "$ResearchRoot\catalog_rendered_statistics.csv" `
    --output-dir "$ResearchRoot\domain_shift_audit" `
    --target-column forced_closed_set_class `
    --positive-label None_of_the_Above `
    --group-column event
if ($LASTEXITCODE -ne 0) { throw "Falha na auditoria de dominio" }

Write-Host "[9/10] Consolidando os 218 eventos..."
gwproject research consolidate-events `
    --quality ".\outputs\gwtc4_218\event_quality_summary.csv" `
    --ensemble "$ResearchRoot\gravity_spy_ensemble\per_event.csv" `
    --ridge "$ResearchRoot\ridge_diagnostic\per_event.csv" `
    --matched-filter "$ResearchRoot\matched_filter_diagnostic\per_event.csv" `
    --expected-events 218 `
    --output "$ResearchRoot\all_218_events.csv"
if ($LASTEXITCODE -ne 0) { throw "Falha na consolidacao dos eventos" }

Write-Host "[10/10] Concluido. Consulte:"
Write-Host "$ResearchRoot\gravity_spy_ensemble\summary.json"
Write-Host "$ResearchRoot\ood_test_control.json"
Write-Host "$ResearchRoot\domain_shift_audit\domain_shift_audit.json"
Write-Host "$ResearchRoot\all_218_events.csv"
