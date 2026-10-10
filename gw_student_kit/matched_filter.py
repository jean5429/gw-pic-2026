"""
Módulo de Filtro Casado (Matched Filtering) para Ondas Gravitacionais
====================================================================
Implementa a busca ótima por correlação no domínio da frequência, ponderada pelo
ruído do detector (PSD), gerando a série temporal de SNR (Signal-to-Noise Ratio).

Convenções (as mesmas de Allen et al. 2012, FINDCHIRP):

    h̃(f) ≈ Δt · FFT[h]                          (transformada contínua discretizada)
    z(t)  = 4 Σ_{f>0} d̃(f) h̃*(f) / S(f) · e^{2πift} Δf
    σ²    = 4 Σ_{f>0} |h̃(f)|² / S(f) · Δf
    ρ(t)  = |z(t)| / σ

Em ruído gaussiano estacionário, ρ² segue uma χ² com 2 graus de liberdade (média 2).

Correções feitas na incorporação ao gwproject (10/10/2026): a versão original omitia
os fatores Δt da transformada (SNR inflada por ~fs), dividia por uma PSD quase nula
fora da banda filtrada e tratava o atraso da correlação como instante de fusão.
"""

import numpy as np
from scipy.fft import fft, ifft


def compute_matched_filter(data: np.ndarray, template: np.ndarray, fs: float,
                           psd: np.ndarray = None, f_low: float = None,
                           f_high: float = None) -> tuple[np.ndarray, float, int]:
    """
    Calcula a série temporal da Razão Sinal-Ruído (SNR) pelo Filtro Casado.

    Parâmetros:
    -----------
    data : np.ndarray
        Série temporal de dados do detector.
    template : np.ndarray
        Forma de onda teórica esperada (Chirp), amostrada como os dados.
    fs : float
        Frequência de amostragem em Hz.
    psd : np.ndarray, opcional
        PSD unilateral (strain²/Hz) em frequências igualmente espaçadas de 0 a fs/2,
        como devolve ``scipy.signal.welch``. Se None, supõe ruído branco com a
        variância dos próprios dados: S = 2·var(dados)·Δt.
        Se os dados foram filtrados, filtre o template com o mesmo filtro e use a PSD
        dos dados filtrados; assim a SNR não muda dentro da banda.
    f_low, f_high : float, opcionais
        Banda de análise em Hz. Use a banda do filtro passa-banda: fora dela a PSD dos
        dados filtrados é quase nula e a divisão amplifica erros numéricos.

    Retorna:
    --------
    snr_series : np.ndarray
        ρ(t) para cada atraso do template (valores >= 0).
    max_snr : float
        Pico máximo de SNR encontrado.
    peak_idx : int
        Atraso, em amostras, em que ocorre o pico. Use ``align_to_merger`` para
        convertê-lo em instante de fusão.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    template = np.asarray(template, dtype=float)
    if len(template) < n:
        template_padded = np.pad(template, (0, n - len(template)))
    else:
        template_padded = template[:n]

    dt = 1.0 / fs
    df = fs / n
    data_fft = dt * fft(data)
    template_fft = dt * fft(template_padded)
    half_n = n // 2 + 1
    freqs = np.arange(half_n) * df

    if psd is not None and len(psd) > 0:
        freqs_psd = np.linspace(0.0, fs / 2.0, len(psd))
        psd_half = np.interp(freqs, freqs_psd, np.asarray(psd, dtype=float))
    else:
        psd_half = np.full(half_n, 2.0 * np.var(data) * dt)

    band = freqs > 0.0
    if f_low is not None:
        band &= freqs >= f_low
    if f_high is not None:
        band &= freqs <= f_high
    band &= np.isfinite(psd_half) & (psd_half > 0)
    if not np.any(band):
        raise ValueError("banda de análise vazia ou PSD inválida")
    weight = np.zeros(half_n)
    weight[band] = 1.0 / psd_half[band]

    # Apenas frequências positivas: |z| maximiza sobre a fase do template.
    integrand = np.zeros(n, dtype=complex)
    integrand[:half_n] = 4.0 * df * data_fft[:half_n] * np.conj(template_fft[:half_n]) * weight
    z = ifft(integrand) * n
    sigma = np.sqrt(4.0 * df * np.sum(np.abs(template_fft[:half_n]) ** 2 * weight))

    if sigma > 0:
        snr_series = np.abs(z) / sigma
    else:
        snr_series = np.zeros(n)

    max_snr = float(np.max(snr_series))
    peak_idx = int(np.argmax(snr_series))
    return snr_series, max_snr, peak_idx


def align_to_merger(snr_series: np.ndarray, template_merger_index: int) -> np.ndarray:
    """
    Reindexa a série de SNR para que o índice i corresponda à fusão no instante i/fs.

    O pico da correlação ocorre no atraso j em que os dados reproduzem o template
    deslocado de j amostras. Se a fusão do template está no índice m, a fusão nos dados
    fica em (j + m) mod N. A correlação é circular: picos a menos de uma duração de
    template das bordas não são confiáveis.
    """
    return np.roll(np.asarray(snr_series), int(template_merger_index))


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
