"""
GW Student Kit - Toolkit Educacional para Análise de Ondas Gravitacionais
========================================================================
Desenvolvido para ensino de Processamento Digital de Sinais, Filtro Casado e
Machine Learning em dados de interferômetros (LIGO/Virgo/KAGRA).
"""

from .synthetic import generate_synthetic_chirp, generate_detector_noise, inject_signal
from .dsp import bandpass_filter, compute_psd, load_lwtc_hdf5, load_event_folder
from .matched_filter import compute_matched_filter, find_snr_peaks
from .visualization import plot_strain_comparison, plot_spectrogram, plot_snr_time_series
from .ml_anomaly import detect_anomalies_autoencoder

__version__ = "1.0.0"
__all__ = [
    "generate_synthetic_chirp",
    "generate_detector_noise",
    "inject_signal",
    "bandpass_filter",
    "compute_psd",
    "load_lwtc_hdf5",
    "load_event_folder",
    "compute_matched_filter",
    "find_snr_peaks",
    "plot_strain_comparison",
    "plot_spectrogram",
    "plot_snr_time_series",
    "detect_anomalies_autoencoder",
]
