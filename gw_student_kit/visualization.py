"""
Módulo de Visualização Gráfica para Ondas Gravitacionais
======================================================
Geração de gráficos didáticos comparativos:
- Strain Cru vs Strain Filtrado (Butterworth)
- Espectrograma Tempo-Frequência (STFT / Transformada Q)
- Série Temporal de SNR com picos de detecção destacados.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal


def plot_strain_comparison(time: np.ndarray, raw_data: np.ndarray, 
                           filtered_data: np.ndarray, save_path: str = None, 
                           title: str = "Strain do Detector: Bruto vs Filtrado",
                           xlim: tuple = None):
    """
    Plota o sinal cru comparado lado a lado com o sinal filtrado.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    axes[0].plot(time, raw_data, color='gray', alpha=0.7, label='Sinal Bruto (Com Ruído)')
    axes[0].set_ylabel('Strain h(t)')
    axes[0].set_title(title)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    
    axes[1].plot(time, filtered_data, color='crimson', label='Passa-Banda (30Hz - 400Hz)')
    axes[1].set_xlabel('Tempo (s)')
    axes[1].set_ylabel('Strain Filtrado')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')
    
    if xlim:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def relative_spectrogram(data: np.ndarray, fs: float, nperseg: int = 256,
                         noverlap: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT com a potência de cada frequência dividida pela sua mediana no tempo.

    O ruído dos detectores é "colorido": as baixas frequências têm ordens de grandeza
    mais potência e escondem o chirp. Normalizar cada linha pela mediana (como o
    ``norm="median"`` da transformada Q do gwpy) mostra quanto cada ponto se destaca do
    ruído típico daquela frequência, e o resultado não depende da escala do strain
    (~1e-21). Valor 1 = ruído típico.
    """
    f, t_spec, sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    median = np.median(sxx, axis=1, keepdims=True)
    floor = np.finfo(float).tiny
    # Linhas sem potência (ex.: abaixo do passa-banda) ficam em zero, sem divisão por zero.
    relative = np.where(median > floor, sxx / np.maximum(median, floor), 0.0)
    return f, t_spec, relative


def plot_spectrogram(data: np.ndarray, fs: float, save_path: str = None,
                     title: str = "Espectrograma Tempo-Frequência (Chirp Signal)",
                     xlim: tuple = None):
    """
    Plota o Espectrograma (Short-Time Fourier Transform) no domínio tempo-frequência,
    com a potência relativa à mediana de cada frequência (ver ``relative_spectrogram``).

    A versão original somava um piso fixo de 1e-30 antes do logaritmo; como a potência
    do strain real é ~1e-46/Hz, o mapa virava uma constante (-300 dB) e saía preto.
    """
    f, t_spec, relative = relative_spectrogram(data, fs)

    plt.figure(figsize=(10, 5))
    # Escala linear em unidades da mediana: 1 = ruído típico; o chirp aparece acima.
    vmax = max(float(np.percentile(relative, 99.9)), 5.0)
    plt.pcolormesh(t_spec, f, relative, shading='gouraud', cmap='inferno', vmin=0.0, vmax=vmax)
    plt.colorbar(label='Potência relativa à mediana')
    plt.ylim(20, 500)
    if xlim:
        plt.xlim(xlim)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência (Hz)')
    plt.title(title)
    plt.grid(False)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_snr_time_series(time: np.ndarray, snr_series: np.ndarray,
                         threshold: float = 8.0, t_event: float = None,
                         save_path: str = None, title: str = "Série Temporal de SNR (Filtro Casado)",
                         xlim: tuple = None, xlabel: str = 'Tempo (s)'):
    """
    Plota a evolução temporal da Razão Sinal-Ruído (SNR).

    Depois de ``align_to_merger``, o eixo x é o instante de fusão hipotético do template.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time, snr_series, color='navy', lw=1.5, label='SNR(t)')
    plt.axhline(threshold, color='crimson', linestyle='--', linewidth=1.5, label=f'Limiar de Detecção (SNR >= {threshold})')
    
    max_idx = np.argmax(snr_series)
    max_snr = snr_series[max_idx]
    t_peak = time[max_idx]
    
    plt.plot(t_peak, max_snr, 'ro', markersize=8, label=f'Pico Máximo: SNR={max_snr:.2f} em t={t_peak:.3f}s')
    
    if t_event is not None:
        plt.axvline(t_event, color='darkgreen', linestyle=':', label=f'Instante de Injeção (t={t_event:.3f}s)')

    plt.xlabel(xlabel)
    plt.ylabel('Razão Sinal-Ruído (SNR)')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if xlim:
        plt.xlim(xlim)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
