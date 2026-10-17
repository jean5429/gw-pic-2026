param(
    [ValidateSet("cu126", "cu128", "cu130")]
    [string]$CudaIndex = "cu128",
    [string]$TorchVersion = "2.11.0",
    [string]$TorchvisionVersion = "0.26.0",
    [string]$TorchaudioVersion = "2.11.0"
)

$ErrorActionPreference = "Stop"

Write-Host "Verificando driver NVIDIA..."
& nvidia-smi
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi falhou. Verifique o driver NVIDIA antes de instalar PyTorch CUDA."
}

$indexUrl = "https://download.pytorch.org/whl/$CudaIndex"
$packages = @(
    "torch==$TorchVersion",
    "torchvision==$TorchvisionVersion",
    "torchaudio==$TorchaudioVersion"
)

Write-Host "Removendo instalacoes anteriores de PyTorch..."
python -m pip uninstall -y torch torchvision torchaudio

Write-Host "Instalando PyTorch pelo indice oficial $indexUrl..."
python -m pip install --no-cache-dir @packages --index-url $indexUrl
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao instalar PyTorch pelo indice $indexUrl."
}

python -c "import torch, torchvision; print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA disponivel:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma')"
if ($LASTEXITCODE -ne 0) {
    throw "A verificacao CUDA falhou."
}
