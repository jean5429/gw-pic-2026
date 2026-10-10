"""
Módulo de Processamento Digital de Sinais (DSP) para Ondas Gravitacionais
========================================================================
Filtros Passa-Banda (Butterworth), Remoção de Tendências/Ruídos de Baixa Frequência,
Cálculo da Densidade Espectral de Potência (PSD) e Leitor HDF5 do LIGO.
"""

import numpy as np
import scipy.signal as signal

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def bandpass_filter(data: np.ndarray, fs: float, lowcut: float = 30.0, 
                    highcut: float = 400.0, order: float = 4) -> np.ndarray:
    """
    Aplica um Filtro Passa-Banda Butterworth IIR sem distorção de fase (filtfilt).

    Conceito Didático:
    - O LIGO é "cego" abaixo de ~20-30 Hz devido ao ruído sísmico da Terra.
    - Acima de ~400-500 Hz domina o ruído de disparo de fótons (shot noise).
    - O filtro limita os dados à janela útil de sensibilidade astrofísica.

    Parâmetros:
    -----------
    data : np.ndarray
        Série temporal crua (strain).
    fs : float
        Frequência de amostragem em Hz.
    lowcut : float
        Frequência de corte inferior (padrão 30 Hz).
    highcut : float
        Frequência de corte superior (padrão 400 Hz).
    order : int
        Ordem do filtro Butterworth.

    Retorna:
    --------
    filtered_data : np.ndarray
        Dados filtrados no mesmo formato da entrada.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Coeficientes a e b do filtro Butterworth
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Aplica o filtro nos dois sentidos (forward-backward) para cancelar desfasagem
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def compute_psd(data: np.ndarray, fs: float, nperseg: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula a Densidade Espectral de Potência (PSD) usando o método de Welch.

    Conceito Didático:
    A PSD Sn(f) mede o nível de ruído do interferômetro em cada frequência f.
    É fundamental para ponderar o Filtro Casado (dando menos peso a frequências ruidosas).

    Parâmetros:
    -----------
    data : np.ndarray
        Série temporal (strain).
    fs : float
        Frequência de amostragem (Hz).
    nperseg : int
        Tamanho do segmento de janela para a FFT.

    Retorna:
    --------
    freqs : np.ndarray
        Vetor de frequências (Hz).
    psd : np.ndarray
        Densidade espectral de potência (strain^2 / Hz).
    """
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg)
    return freqs, psd


def load_lwtc_hdf5(file_path: str) -> tuple[np.ndarray, float, float, str]:
    """
    Carrega arquivos brutos no formato oficial .hdf5 disponibilizados pelo LIGO/GWOSC.

    Parâmetros:
    -----------
    file_path : str
        Caminho para o arquivo .hdf5.

    Retorna:
    --------
    strain : np.ndarray
        Dados da série temporal do detector.
    fs : float
        Frequência de amostragem (Hz).
    gps_start : float
        Tempo inicial do trecho em segundos GPS (x0).
    detector_name : str
        Nome do canal/detector (ex: H1, L1, V1).
    """
    if not HAS_H5PY:
        raise ImportError("A biblioteca 'h5py' é necessária para ler arquivos .hdf5 do LIGO. Instale com 'pip install h5py'.")
        
    with h5py.File(file_path, 'r') as f:
        if 'strain' in f and 'Strain' in f['strain']:
            # Arquivo oficial do GWOSC (grupos meta/quality/strain)
            key = 'strain/Strain'
            ds = f[key]
            fs = 1.0 / float(ds.attrs['Xspacing'])
            gps_start = float(ds.attrs['Xstart'])
        else:
            # Arquivo gravado pelo gwpy (TimeSeries.write), como outputs/gwtc/raw
            key = list(f.keys())[0]
            ds = f[key]
            fs = 1.0 / float(ds.attrs['dx'])
            gps_start = float(ds.attrs['x0'])
        strain = ds[:]
        
        # Limpeza de NaNs/Infs se houver
        valid_mask = ~np.isnan(strain) & ~np.isinf(strain)
        strain = np.where(valid_mask, strain, 0.0)
        
        return strain, fs, gps_start, key


def load_event_folder(folder_path: str) -> dict[str, dict]:
    """
    Varre uma pasta de evento do catálogo GWTC e carrega todos os arquivos .hdf5 encontrados (ex: H1.hdf5, L1.hdf5).

    Parâmetros:
    -----------
    folder_path : str
        Caminho para o diretório do evento (ex: 'GW150914-v4').

    Retorna:
    --------
    detectors_data : dict
        Dicionário com chave sendo o nome do detector ('H1', 'L1', etc) contendo:
        {'strain': array, 'fs': float, 'gps_start': float, 'file_name': str}
    """
    import os
    import glob
    
    h5_files = glob.glob(os.path.join(folder_path, "*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"Nenhum arquivo .hdf5 foi encontrado na pasta: '{folder_path}'")
        
    results = {}
    for fpath in sorted(h5_files):
        det_name = os.path.splitext(os.path.basename(fpath))[0]
        strain, fs, gps_start, channel_key = load_lwtc_hdf5(fpath)
        results[det_name] = {
            'strain': strain,
            'fs': fs,
            'gps_start': gps_start,
            'channel_key': channel_key,
            'file_path': fpath
        }
    return results
