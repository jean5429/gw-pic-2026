"""
Módulo de Geração de Sinais Sintéticos e Ruído para Ondas Gravitacionais
========================================================================
Este módulo permite que alunos sem acesso prévio aos gigabytes de dados do LIGO
gerem séries temporais realistas com ruído de detector e injeção de sinais Chirp.
"""

import numpy as np

# G·M_sol/c³ em segundos (massa solar nominal da IAU).
SOLAR_MASS_SECONDS = 4.925490947641267e-6


def generate_synthetic_chirp(fs: float = 4096.0, duration: float = 4.0, 
                             f_min: float = 30.0, f_max: float = 350.0, 
                             t_merger: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera uma forma de onda 'Chirp' simplificada representando a espiralização
    e fusão (coalescência) de dois objetos compactos (buracos negros / estrelas de nêutrons).

    É um brinquedo didático: a lei f(t) abaixo não vem da relatividade geral. Para
    filtrar dados reais, use ``generate_newtonian_chirp``.

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


def generate_newtonian_chirp(fs: float = 4096.0, duration: float = 4.0,
                             chirp_mass_solar: float = 30.0, t_merger: float = 3.0,
                             f_min: float = 30.0, f_max: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera o chirp de ordem dominante (quadrupolar, "newtoniano") de uma binária.

    Com τ = t_merger - t e 𝓜 = G·M_chirp/c³ (em segundos):
        f(τ) = (1/π) · (5 / (256 τ))^{3/8} · 𝓜^{-5/8}
        Φ(τ) = -2 · (τ / (5 𝓜))^{5/8}
        h(t) ∝ f^{2/3} · cos Φ
    O sinal é cortado em f_min e f_max e não modela fusão nem ringdown. Use a massa
    chirp no referencial do detector (coluna ``chirp_mass_detector`` do catálogo).

    Retorna:
    --------
    time : np.ndarray
        Vetor de tempo em segundos.
    h_t : np.ndarray
        Strain adimensional, com pico 1e-21 (a escala não altera a SNR do filtro casado).
    """
    if chirp_mass_solar <= 0 or not 0 < f_min < f_max:
        raise ValueError("massa chirp e banda de frequência devem ser positivas")
    t = np.arange(int(round(fs * duration))) / fs
    tau = t_merger - t
    script_m = chirp_mass_solar * SOLAR_MASS_SECONDS
    h_t = np.zeros_like(t)
    inspiral = tau > 0
    freq = np.zeros_like(t)
    freq[inspiral] = (5.0 / (256.0 * tau[inspiral])) ** (3.0 / 8.0) * script_m ** (-5.0 / 8.0) / np.pi
    band = inspiral & (freq >= f_min) & (freq <= f_max)
    if not np.any(band):
        raise ValueError("o chirp não cruza a banda pedida dentro da janela")
    phase = -2.0 * (tau[band] / (5.0 * script_m)) ** (5.0 / 8.0)
    h_t[band] = 1.0e-21 * (freq[band] / freq[band].max()) ** (2.0 / 3.0) * np.cos(phase)
    return t, h_t


def generate_detector_noise(fs: float = 4096.0, duration: float = 4.0,
                            std_dev: float = 1.0e-20,
                            rng: np.random.Generator = None) -> np.ndarray:
    """
    Gera ruído gaussiano branco, um modelo simplificado do ruído de um interferômetro.
    O ruído real é colorido (PSD dependente da frequência) e não estacionário.

    Parâmetros:
    -----------
    fs : float
        Frequência de amostragem (Hz).
    duration : float
        Duração em segundos.
    std_dev : float
        Desvio padrão do ruído (típico do LIGO: ~1e-20 strain).
    rng : np.random.Generator, opcional
        Gerador para resultados reprodutíveis (ex.: ``np.random.default_rng(42)``).

    Retorna:
    --------
    noise : np.ndarray
        Série temporal de ruído.
    """
    n_samples = int(fs * duration)
    if rng is None:
        return np.random.normal(loc=0.0, scale=std_dev, size=n_samples)
    return rng.normal(loc=0.0, scale=std_dev, size=n_samples)


def inject_signal(noise: np.ndarray, signal: np.ndarray, target_snr: float = 12.0) -> np.ndarray:
    """
    Escala o sinal para uma SNR ótima de filtro casado e o injeta no ruído.

    Para ruído branco gaussiano com desvio padrão σ, a SNR ótima de um template
    perfeitamente casado é ρ = sqrt(Σ h²) / σ. O fator de escala é, portanto,
    ρ_alvo · σ / sqrt(Σ h²). (A versão original usava a razão entre RMS, que com
    16 mil amostras produzia ρ efetivo ~1.800 para "SNR = 14".)

    Parâmetros:
    -----------
    noise : np.ndarray
        Série temporal do ruído (suposto branco).
    signal : np.ndarray
        Série temporal do sinal puro h(t).
    target_snr : float
        SNR ótima desejada (ex: 12 é uma detecção clara em um detector).

    Retorna:
    --------
    combined_strain : np.ndarray
        Strain combinado (Ruído + Sinal escalado).
    """
    signal_norm = np.sqrt(np.sum(signal ** 2))
    noise_std = np.std(noise)

    if signal_norm == 0:
        return noise.copy()

    scale_factor = target_snr * noise_std / signal_norm
    return noise + signal * scale_factor
