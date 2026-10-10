"""
Módulo de Geração de Sinais Sintéticos e Ruído para Ondas Gravitacionais
========================================================================
Este módulo permite que alunos sem acesso prévio aos gigabytes de dados do LIGO
gerem séries temporais realistas com ruído de detector e injeção de sinais Chirp.
"""

import numpy as np

def generate_synthetic_chirp(fs: float = 4096.0, duration: float = 4.0, 
                             f_min: float = 30.0, f_max: float = 350.0, 
                             t_merger: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera uma forma de onda 'Chirp' simplificada representando a espiralização 
    e fusão (coalescência) de dois objetos compactos (buracos negros / estrelas de nêutrons).

    Matemática didática:
    f(t) cresce com a aproximação dos corpos: f(t) = f_min + (f_max - f_min) * (t / t_merger)^3
    h(t) = A(t) * sin(2 * pi * integral(f(t) dt))

    Parâmetros:
    -----------
    fs : float
        Frequência de amostragem em Hz (padrão LIGO: 4096 Hz).
    duration : float
        Duração total do sinal em segundos.
    f_min : float
        Frequência inicial da varredura (Hz).
    f_max : float
        Frequência máxima no instante de fusão (Hz).
    t_merger : float
        Instante exato em segundos onde ocorre a colisão.

    Retorna:
    --------
    time : np.ndarray
        Vetor de tempo em segundos.
    h_t : np.ndarray
        Amplitude do strain h(t) adimensional.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    h_t = np.zeros_like(t)
    
    # Seleciona apenas a janela de tempo antes da fusão
    mask = t <= t_merger
    t_rel = t[mask] / t_merger
    
    # Frequência instantânea aumentando com o tempo (modelo aproximado de chirp)
    freq = f_min + (f_max - f_min) * (t_rel ** 3.0)
    
    # Fase acumulada: integral de 2*pi*f(t) dt
    phase = 2.0 * np.pi * np.cumsum(freq) / fs
    
    # Amplitude crescendo até a fusão e decaindo rapidamente (Ringdown)
    amp = 1.0e-21 * (t_rel ** 2.0)
    
    signal = amp * np.sin(phase)
    
    # Adiciona pequeno ringdown (amortecimento pós-fusão)
    ringdown_mask = (t > t_merger) & (t <= t_merger + 0.05)
    t_ring = t[ringdown_mask] - t_merger
    ringdown_amp = 1.0e-21 * np.exp(-t_ring / 0.005) * np.sin(2.0 * np.pi * f_max * t_ring)
    
    h_t[mask] = signal
    h_t[ringdown_mask] = ringdown_amp
    
    return t, h_t


def generate_detector_noise(fs: float = 4096.0, duration: float = 4.0, 
                            std_dev: float = 1.0e-20) -> np.ndarray:
    """
    Gera ruído gaussiano simulando a perturbação térmica/sísmica de um interferômetro.

    Parâmetros:
    -----------
    fs : float
        Frequência de amostragem (Hz).
    duration : float
        Duração em segundos.
    std_dev : float
        Desvio padrão do ruído (típico do LIGO: ~1e-20 strain).

    Retorna:
    --------
    noise : np.ndarray
        Série temporal de ruído.
    """
    n_samples = int(fs * duration)
    return np.random.normal(loc=0.0, scale=std_dev, size=n_samples)


def inject_signal(noise: np.ndarray, signal: np.ndarray, target_snr: float = 12.0) -> np.ndarray:
    """
    Escala a amplitude do sinal de onda gravitacional e o injeta no ruído 
    para obter uma Razão Sinal-Ruído (SNR) desejada.

    Parâmetros:
    -----------
    noise : np.ndarray
        Série temporal do ruído.
    signal : np.ndarray
        Série temporal do sinal puro h(t).
    target_snr : float
        SNR de injeção desejado (ex: SNR=12 é uma detecção bem clara).

    Retorna:
    --------
    combined_strain : np.ndarray
        Strain combinado (Ruído + Sinal escalado).
    """
    sig_power = np.sqrt(np.mean(signal ** 2))
    noise_power = np.sqrt(np.mean(noise ** 2))
    
    if sig_power == 0:
        return noise.copy()
        
    scale_factor = (target_snr * noise_power) / sig_power
    scaled_signal = signal * scale_factor
    
    return noise + scaled_signal
