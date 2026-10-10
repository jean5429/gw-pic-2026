$ErrorActionPreference = "Stop"

Write-Host "Verificando NVIDIA..."
& nvidia-smi
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi falhou. Verifique o driver NVIDIA."
}

Write-Host "Removendo wheels CPU/antigos do PyTorch..."
python -m pip uninstall -y torch torchvision torchaudio

$base = "https://download.pytorch.org/whl/cu121"
$wheels = @(
    "$base/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl",
    "$base/torchvision-0.20.1%2Bcu121-cp311-cp311-win_amd64.whl",
    "$base/torchaudio-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl"
)

Write-Host "Instalando wheels CUDA 12.1 por URL direta..."
foreach ($wheel in $wheels) {
    python -m pip install --no-cache-dir $wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao instalar $wheel"
    }
}

python -c "import torch, torchvision; print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA wheel:', torch.version.cuda); print('CUDA disponivel:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma')"
if ($LASTEXITCODE -ne 0) {
    throw "A verificacao CUDA falhou."
}
