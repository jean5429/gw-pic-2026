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


def plot_spectrogram(data: np.ndarray, fs: float, save_path: str = None, 
                     title: str = "Espectrograma Tempo-Frequência (Chirp Signal)",
                     xlim: tuple = None):
    """
    Plota o Espectrograma (Short-Time Fourier Transform) no domínio tempo-frequência.
    """
    f, t_spec, Sxx = signal.spectrogram(data, fs=fs, nperseg=256, noverlap=220)
    
    plt.figure(figsize=(10, 5))
    # Escala logarítmica para evidenciar energias fracas
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1.0e-30), shading='gouraud', cmap='inferno')
    plt.colorbar(label='Potência (dB)')
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
