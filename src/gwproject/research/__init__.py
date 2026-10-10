"""Ferramentas de pesquisa para robustez, domain shift e inferencia open-world."""

from .domain_shift import grouped_domain_audit, pixel_statistics
from .panel_preprocessing import (
    PanelCropConfig,
    crop_frequency_band,
    detect_spectrogram_panel,
    prepare_gravity_spy_panel,
)
from .phase import randomize_fourier_phase
from .uncertainty import (
    energy_score,
    ensemble_statistics,
    expected_calibration_error,
    fit_mahalanobis_reference,
    mahalanobis_score,
)

__all__ = [
    "PanelCropConfig",
    "crop_frequency_band",
    "detect_spectrogram_panel",
    "prepare_gravity_spy_panel",
    "randomize_fourier_phase",
    "energy_score",
    "ensemble_statistics",
    "expected_calibration_error",
    "fit_mahalanobis_reference",
    "mahalanobis_score",
    "pixel_statistics",
    "grouped_domain_audit",
]
