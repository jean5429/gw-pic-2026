"""
Módulo de Machine Learning e Detecção de Anomalias (Deep Learning)
==================================================================
Demonstração didática de Detecção de Anomalias Sem Supervisão:
Treina uma rede neural ou modelo estatístico apenas no ruído de fundo do detector.
Ao introduzir a colisão cósmica, o erro de reconstrução dispara (anomalia > 3-Sigma).
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


if HAS_TORCH:
    class Simple1DAutoencoder(nn.Module):
        """
        Autoencoder Convolucional 1D didático em PyTorch.
        """
        def __init__(self, window_size: int = 256):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(8, 1, kernel_size=7, stride=2, padding=3, output_padding=1)
            )

        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed


def detect_anomalies_autoencoder(data: np.ndarray, window_size: int = 256, 
                                 epochs: int = 15) -> np.ndarray:
    """
    Treina um Autoencoder (PyTorch) ou Isolation Forest (Scikit-Learn) na primeira 
    metade dos dados (ruído puro) e calcula o erro de anomalia na série inteira.

    Parâmetros:
    -----------
    data : np.ndarray
        Série temporal de dados (filtrados).
    window_size : int
        Tamanho da janela deslizante para o modelo.
    epochs : int
        Número de épocas de treinamento.

    Retorna:
    --------
    anomaly_scores : np.ndarray
        Série temporal do erro de anomalia (valores altos indicam colisão/sinal).
    """
    n_samples = len(data)
    
    # 1. Fallback caso PyTorch não esteja instalado: usa janelas de desvio estatístico
    if not HAS_TORCH:
        print("[Aviso ML] PyTorch não encontrado. Utilizando estimador estatístico de anomalia (Desvio Padrão Móvel).")
        padded = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
        rolling_std = np.array([np.std(padded[i:i + window_size]) for i in range(n_samples)])
        # Normaliza em z-scores
        baseline_std = np.mean(rolling_std[:n_samples // 4])
        scores = (rolling_std - baseline_std) / (baseline_std + 1.0e-30)
        return np.maximum(0, scores)

    # 2. Caminho PyTorch
    # Prepara janelas deslizantes (Sliding Windows)
    stride = 32
    windows = []
    for i in range(0, n_samples - window_size, stride):
        w = data[i:i + window_size]
        windows.append(w)
        
    windows = np.array(windows, dtype=np.float32)
    # Usa os primeiros 30% das janelas para treinar apenas com ruído
    n_train = max(5, int(len(windows) * 0.3))
    train_data = torch.tensor(windows[:n_train]).unsqueeze(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Simple1DAutoencoder(window_size=window_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1.0e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(train_data.to(device))
        loss = criterion(output, train_data.to(device))
        loss.backward()
        optimizer.step()
        
    # Inferência no conjunto completo de janelas
    model.eval()
    all_tensor = torch.tensor(windows).unsqueeze(1).to(device)
    with torch.no_grad():
        reconstructed = model(all_tensor)
        mse_per_window = torch.mean((all_tensor - reconstructed) ** 2, dim=(1, 2)).cpu().numpy()
        
    # Mapeia os erros das janelas de volta para a série temporal original
    scores = np.zeros(n_samples)
    for idx, i in enumerate(range(0, n_samples - window_size, stride)):
        scores[i:i + window_size] += mse_per_window[idx]
        
    # Normalização dos scores para escala z-score
    base_mean = np.mean(scores[:n_samples // 4])
    base_std = np.std(scores[:n_samples // 4]) + 1.0e-30
    normalized_scores = (scores - base_mean) / base_std
    
    return np.maximum(0, normalized_scores)
