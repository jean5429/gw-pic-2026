"""
Módulo de Filtro Casado (Matched Filtering) para Ondas Gravitacionais
====================================================================
Implementa a busca ótima por correlação cruzada no domínio da frequência,
pesada pelo ruído do detector (PSD), gerando a série temporal de SNR (Signal-to-Noise Ratio).
"""

import numpy as np
from scipy.fft import fft, ifft

def compute_matched_filter(data: np.ndarray, template: np.ndarray, fs: float, 
                           psd: np.ndarray = None) -> tuple[np.ndarray, float, int]:
    """
    Calcula a Razão Sinal-Ruído (SNR) complexa via Filtro Casado no domínio da frequência.

    Fórmula do Filtro Casado:
    SNR(t) = | F^{-1} [ (Data_fft * Template_fft*) / PSD ] | / Normalização

    Parâmetros:
    -----------
    data : np.ndarray
        Série temporal de dados do detector (filtrados).
    template : np.ndarray
        Forma de onda teórica esperada (Chirp).
    fs : float
        Frequência de amostragem em Hz.
    psd : np.ndarray, opcional
        Vetor de Densidade Espectral de Potência. Se None, assume ruído branco plano.

    Retorna:
    --------
    snr_series : np.ndarray
        Série temporal da amplitude do SNR (valores reais >= 0).
    max_snr : float
        Pico máximo de SNR encontrado.
    peak_idx : int
        Índice do ponto temporal onde ocorreu o pico máximo.
    """
    N = len(data)
    
    # Ajusta o tamanho do template se necessário para casar com a extensão dos dados
    if len(template) < N:
        template_padded = np.pad(template, (0, N - len(template)))
    else:
        template_padded = template[:N]
        
    # Transformadas de Fourier
    data_fft = fft(data)
    template_fft = fft(template_padded)
    
    # Se a PSD for fornecida, interpola para a grade da FFT
    if psd is not None and len(psd) > 0:
        freqs_psd = np.linspace(0, fs / 2, len(psd))
        freqs_fft = np.linspace(0, fs / 2, N // 2 + 1)
        psd_half = np.interp(freqs_fft, freqs_psd, psd)
        psd_full = np.concatenate([psd_half, psd_half[-2:0:-1]])
        psd_full[psd_full <= 0] = 1.0e-40
    else:
        psd_full = np.ones(N)

    # Integração de Fourier com dF = fs / N
    df = fs / N
    
    # Produto de convolução no domínio da frequência pesado pela PSD (One-sided factor = 4 * df)
    # Seleciona frequências positivas e aplica o peso da PSD
    matched_filter_fft = np.zeros(N, dtype=complex)
    half_N = N // 2 + 1
    matched_filter_fft[:half_N] = 4.0 * df * (data_fft[:half_N] * np.conj(template_fft[:half_N])) / psd_full[:half_N]
    
    # Inversa de Fourier no tempo
    snr_complex = ifft(matched_filter_fft) * N
    
    # Variância (sigma) do template normalizado
    template_power = 4.0 * df * np.sum((np.abs(template_fft[:half_N])**2) / psd_full[:half_N])
    sigma = np.sqrt(template_power)
    
    if sigma > 0:
        snr_series = np.abs(snr_complex) / sigma
    else:
        snr_series = np.zeros(N)
        
    max_snr = float(np.max(snr_series))
    peak_idx = int(np.argmax(snr_series))
    
    return snr_series, max_snr, peak_idx


def find_snr_peaks(snr_series: np.ndarray, time: np.ndarray, 
                   threshold: float = 8.0) -> list[dict]:
    """
    Identifica picos significativos de SNR na série temporal acima de um limiar.

    Parâmetros:
    -----------
    snr_series : np.ndarray
        Série temporal de SNR.
    time : np.ndarray
        Vetor de tempo correspondente.
    threshold : float
        Limiar mínimo de SNR para considerar como evento candidato (ex: 8.0).

    Retorna:
    --------
    candidates : list[dict]
        Lista de dicionários contendo {'time': t, 'snr': valor}.
    """
    candidates = []
    peaks = np.where(snr_series >= threshold)[0]
    
    if len(peaks) == 0:
        return candidates
        
    # Agrupa picos vizinhos para evitar múltiplas contagens do mesmo evento
    group = [peaks[0]]
    for idx in peaks[1:]:
        if idx - group[-1] <= 50:
            group.append(idx)
        else:
            best_idx = group[np.argmax(snr_series[group])]
            candidates.append({
                'time': float(time[best_idx]),
                'snr': float(snr_series[best_idx]),
                'index': int(best_idx)
            })
            group = [idx]
            
    if group:
        best_idx = group[np.argmax(snr_series[group])]
        candidates.append({
            'time': float(time[best_idx]),
            'snr': float(snr_series[best_idx]),
            'index': int(best_idx)
        })
        
    return candidates
