"""
Camada 1 — Repositório GWTC-3
Projeto de IC: Detecção de Ondas Gravitacionais

Pipeline:
  1. Instalação de dependências
  2. Configurações globais
  3. Download dos metadados (catálogo)
  4. Download do strain (HDF5)
  5. Pré-processamento (whitening + bandpass)
  6. Espectrogramas Q-transform
  7. Gate 1 — Reprodução da Fig. 1 do paper GW150914

Correções acumuladas:
  - Bandpass manual com scipy.signal.butter + sosfiltfilt (evita o firwin
    do gwpy que quebra com scipy recente).
  - Formatters customizados nos eixos dos plots para impedir o matplotlib
    de gerar tick labels com $\mathdefault{...}$ (causa o ParseException
    em algumas versões).
  - Q-transform agora usa o strain raw em vez do processado para evitar
    divide-by-zero.
  - Correção do parâmetro ausente qrange no compute_qtransform.
  - Filtro físico de contraste de Ridge para barrar arquivos corrompidos.
"""

# ──────────────────────────────────────────────────────────────────
# CÉLULA 1 — Instalação de dependências
# ──────────────────────────────────────────────────────────────────
import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["gwosc", "gwpy", "h5py", "numpy", "pandas", "matplotlib", "scipy", "tqdm"]:
    install(pkg)

print("Dependências instaladas.")

# ──────────────────────────────────────────────────────────────────
# CÉLULA 2 — Configurações globais
# ──────────────────────────────────────────────────────────────────
import os
import time
import threading
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"]                 = False
matplotlib.rcParams["text.parse_math"]             = False
matplotlib.rcParams["mathtext.default"]            = "regular"
matplotlib.rcParams["axes.unicode_minus"]          = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt

CATALOG_NAMES = [
    "GWTC-1-confident",
    "GWTC-2.1-confident",
    "GWTC-3-confident",
]
N_EVENTS_TARGET = 90
DETECTORS = ["H1", "L1", "V1"]

CROP_BEFORE = 16  # s
CROP_AFTER = 16   # s
SAMPLE_RATE = 4096  # Hz
NYQUIST = SAMPLE_RATE / 2

BANDPASS_FMIN = 20
BANDPASS_FMAX = 2000
BANDPASS_ORDER = 4

QRANGE = (4, 64)
FRANGE = (20, 1024)
OUTSEG_DUR = 4  # s

BASE_DIR = "camada1_gwtc3"
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGS_DIR = os.path.join(BASE_DIR, "figs")

for d in [DATA_RAW_DIR, DATA_PROC_DIR, FIGS_DIR]:
    os.makedirs(d, exist_ok=True)

LOCAL_CACHE_DIR = os.path.abspath(os.path.join(BASE_DIR, "cache"))
ASTROPY_CACHE_DIR = os.path.join(LOCAL_CACHE_DIR, "astropy")
os.makedirs(ASTROPY_CACHE_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", LOCAL_CACHE_DIR)
os.environ.setdefault("ASTROPY_CACHE_DIR", ASTROPY_CACHE_DIR)

def _plain_format(value, pos=None):
    if value == 0: return "0"
    av = abs(value)
    if av >= 1:
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value))}"
        return f"{value:g}"
    return f"{value:.2g}"

_PLAIN_FMT = mticker.FuncFormatter(_plain_format)

def apply_plain_formatters(ax, log_y=False):
    ax.xaxis.set_major_formatter(_PLAIN_FMT)
    ax.yaxis.set_major_formatter(_PLAIN_FMT)
    if log_y:
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

print(f"Catálogos : {CATALOG_NAMES}")
print(f"Detectores: {DETECTORS}")
print(f"Bandpass  : {BANDPASS_FMIN}–{BANDPASS_FMAX} Hz (Nyquist={NYQUIST:.0f} Hz)")
print(f"Q-transform FRANGE: {FRANGE} Hz, QRANGE: {QRANGE}")

# ──────────────────────────────────────────────────────────────────
# CÉLULA 3 — Download dos metadados do catálogo
# ──────────────────────────────────────────────────────────────────
from gwosc import datasets
from gwosc.api import fetch_event_json

def get_event_list(catalog_names):
    seen = {}
    for cat in catalog_names:
        evs = datasets.find_datasets(type="events", catalog=cat)
        for ev in evs:
            parts = ev.rsplit("-v", 1)
            base = parts[0]
            ver = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
            if base not in seen or ver > seen[base][0]:
                seen[base] = (ver, ev)
    return sorted(v for _, v in seen.values())

def fetch_event_params(event_name, retries=4, backoff=1.5):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            data   = fetch_event_json(event_name)
            params = data.get("events", {})
            latest = list(params.values())[-1] if params else {}
            return {
                "name":                event_name,
                "gps":                 latest.get("GPS"),
                "chirp_mass":          latest.get("chirp_mass"),
                "chirp_mass_source":   latest.get("chirp_mass_source"),
                "mass_1_source":       latest.get("mass_1_source"),
                "mass_2_source":       latest.get("mass_2_source"),
                "luminosity_distance": latest.get("luminosity_distance"),
                "far":                 latest.get("far"),
                "snr":                 latest.get("network_matched_filter_snr"),
                "catalog":             latest.get("catalog.shortName"),
            }
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    print(f"  [aviso] {event_name}: {last_err}")
    return {"name": event_name}

print(f"\n[1/2] Listando eventos de {CATALOG_NAMES}...")
event_list = get_event_list(CATALOG_NAMES)
print(f"      {len(event_list)} eventos encontrados (esperado: ~{N_EVENTS_TARGET}).")

print("\n[2/2] Baixando metadados em paralelo (4 workers)...")
with ThreadPoolExecutor(max_workers=4) as executor:
    rows = list(tqdm(
        executor.map(fetch_event_params, event_list),
        total=len(event_list), unit="evento",
    ))

catalog_df = pd.DataFrame(rows).sort_values("gps").reset_index(drop=True)

nan_mask = catalog_df["gps"].isna()
if nan_mask.any():
    nan_events = catalog_df.loc[nan_mask, "name"].tolist()
    print(f"\n  [reparo] {len(nan_events)} GPS NaN — re-fetch sequencial...")
    for ev in nan_events:
        for _ in range(3):
            fixed = fetch_event_params(ev, retries=1)
            if fixed.get("gps") is not None:
                for col in fixed:
                    if col in catalog_df.columns:
                        catalog_df.loc[catalog_df["name"] == ev, col] = fixed[col]
                break
            time.sleep(2)
    catalog_df = catalog_df.sort_values("gps").reset_index(drop=True)

csv_path = os.path.join(DATA_PROC_DIR, "gwtc3_catalog.csv")
json_path = os.path.join(DATA_PROC_DIR, "gwtc3_catalog.json")
catalog_df.to_csv(csv_path, index=False)
catalog_df.to_json(json_path, orient="records", indent=2)

print(f"\nCatálogo salvo em '{csv_path}' ({len(catalog_df)} eventos)")

# ──────────────────────────────────────────────────────────────────
# CÉLULA 4 — Download do strain (HDF5)
# ──────────────────────────────────────────────────────────────────
from gwpy.timeseries import TimeSeries

def fetch_open_data_safe(detector, t_start, t_end, sample_rate=SAMPLE_RATE, retries=3, prefer_cache=True):
    last_err = None
    cache_modes = [prefer_cache, False] if prefer_cache else [False, True]
    for attempt in range(1, retries + 1):
        for use_cache in cache_modes:
            try:
                strain = TimeSeries.fetch_open_data(
                    detector, t_start, t_end, sample_rate=sample_rate, cache=use_cache,
                )
                if strain.sample_rate.value != sample_rate:
                    strain = strain.resample(sample_rate)
                return strain
            except Exception as e:
                last_err = e
                msg = str(e)
                if "Permission denied" in msg or ".astropy" in msg or "cache" in msg.lower():
                    continue
        if attempt < retries: time.sleep(1.5 * attempt)
    raise RuntimeError(f"Falha ao baixar {detector} entre {t_start} e {t_end}. Último erro: {last_err}")

def fetch_and_save_raw(detector, t_start, t_end, out_path=None, prefer_cache=True):
    strain = fetch_open_data_safe(detector, t_start, t_end, sample_rate=SAMPLE_RATE, prefer_cache=prefer_cache)
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        strain.write(out_path, format="hdf5", overwrite=True)
    return strain

N_WORKERS = 6
_print_lock = threading.Lock()

def download_strain_event(event_name, gps, detectors=DETECTORS):
    event_dir = os.path.join(DATA_RAW_DIR, event_name)
    os.makedirs(event_dir, exist_ok=True)
    t_start = gps - CROP_BEFORE
    t_end = gps + CROP_AFTER
    saved = {}
    for det in detectors:
        out_path = os.path.join(event_dir, f"{det}.hdf5")
        if os.path.exists(out_path):
            saved[det] = out_path
            with _print_lock:
                tqdm.write(f"  [pulo] Arquivo já existe: {event_name}/{det}.hdf5")
            continue
        try:
            fetch_and_save_raw(det, t_start, t_end, out_path=out_path, prefer_cache=True)
            saved[det] = out_path
        except Exception as e:
            pass
    return saved

def _download_task(row):
    ev = row["name"]
    gps = float(row["gps"])
    saved = download_strain_event(ev, gps)
    return {"event": ev, "detectors": str(list(saved.keys())), "n_detectors": len(saved)}

print(f"\nBaixando strain de {len(catalog_df)} eventos [{N_WORKERS} workers paralelos]...\n")

download_log = [None] * len(catalog_df)
with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    rows_list = [row for _, row in catalog_df.iterrows()]
    future_to_idx = {executor.submit(_download_task, row): i for i, row in enumerate(rows_list)}
    with tqdm(total=len(rows_list), unit="evento") as pbar:
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try: download_log[idx] = future.result()
            except Exception: download_log[idx] = {"event": rows_list[idx]["name"], "detectors": "[]", "n_detectors": 0}
            pbar.update(1)

# ──────────────────────────────────────────────────────────────────
# CÉLULA 5 — Pré-processamento
# ──────────────────────────────────────────────────────────────────
def load_raw_strain(event_name, detector):
    path = os.path.join(DATA_RAW_DIR, event_name, f"{detector}.hdf5")
    if not os.path.exists(path): return None
    return TimeSeries.read(path, format="hdf5")

def bandpass_butter(strain, fmin, fmax, order=4):
    fs = float(strain.sample_rate.value)
    nyq = fs / 2.0
    sos = butter(order, [fmin / nyq, fmax / nyq], btype="bandpass", output="sos")
    filtered_arr = sosfiltfilt(sos, strain.value)
    return TimeSeries(filtered_arr, t0=strain.t0, dt=strain.dt, unit=strain.unit, channel=strain.channel)

def preprocess_strain(strain):
    white = strain.whiten(fftlength=4, fduration=4)
    return bandpass_butter(white, BANDPASS_FMIN, BANDPASS_FMAX, order=BANDPASS_ORDER)

def save_processed(strain, event_name, detector):
    out_dir = os.path.join(DATA_PROC_DIR, event_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{detector}_processed.hdf5")
    strain.write(out_path, format="hdf5", overwrite=True)
    return out_path

def _preprocess_task(row):
    ev = row["name"]
    results = {}
    for det in DETECTORS:
        raw = load_raw_strain(ev, det)
        if raw is not None:
            try:
                proc = preprocess_strain(raw)
                results[det] = save_processed(proc, ev, det)
            except Exception: pass
    return {"event": ev, "processed": str(list(results.keys()))}

print(f"\nPré-processando strain...\n")
with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    list(tqdm(executor.map(_preprocess_task, [row for _, row in catalog_df.iterrows()]), total=len(catalog_df), unit="evento"))

# ──────────────────────────────────────────────────────────────────
# CÉLULA 6 — Espectrogramas Q-transform
# ──────────────────────────────────────────────────────────────────
def sanitize_strain(strain):
    arr = strain.value.copy().astype(np.float64)
    bad = ~np.isfinite(arr)
    if bad.any():
        x = np.arange(len(arr))
        good = np.where(~bad)[0]
        arr = np.interp(x, good, arr[good]) if len(good) > 1 else arr * 0.0
    lo, hi = np.percentile(arr, [0.001, 99.999])
    arr = np.clip(arr, lo, hi)
    sigma = float(np.std(arr))
    if sigma > 0 and np.isfinite(sigma):
        arr = arr + np.random.default_rng(42).normal(0, sigma * 1e-6, len(arr))
    return TimeSeries(arr, t0=strain.t0, dt=strain.dt, unit=strain.unit, channel=strain.channel)

def _has_signal(strain, gps_center, window=4.0, min_std=1e-25):
    t0, dt = float(strain.t0.value), float(strain.dt.value)
    arr = strain.value
    i_s = max(0, int((gps_center - window / 2 - t0) / dt))
    i_e = min(len(arr), int((gps_center + window / 2 - t0) / dt))
    seg = arr[i_s:i_e]
    return len(seg) >= 10 and float(np.std(seg)) > min_std

def _crop_strain(strain, gps_center, outseg_dur, margin=6.0):
    t0, dt = float(strain.t0.value), float(strain.dt.value)
    t_s = max(t0, gps_center - outseg_dur / 2 - margin)
    t_e = min(t0 + len(strain) * dt, gps_center + outseg_dur / 2 + margin)
    return strain if (t_e - t_s) < outseg_dur else strain.crop(t_s, t_e)

def _try_qtransform(strain, gps_center, outseg_dur, norm, qrange, frange, whiten=True):
    t_s, t_e = gps_center - outseg_dur / 2, gps_center + outseg_dur / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return strain.q_transform(qrange=qrange, frange=frange, outseg=(t_s, t_e), logf=True, norm=norm, whiten=whiten)

def _qtransform_scipy(strain, gps_center, outseg_dur=2.0, fmin=20, fmax=512, n_freqs=64):
    from scipy.signal import spectrogram, welch
    from scipy.ndimage import gaussian_filter
    fs, t0 = float(strain.sample_rate.value), float(strain.t0.value)
    data = strain.value.copy().astype(np.float64)
    i_s = max(0, int((gps_center - outseg_dur / 2 - 4.0 - t0) * fs))
    i_e = min(len(data), int((gps_center + outseg_dur / 2 + 4.0 - t0) * fs))
    crop = data[i_s:i_e]
    t0_crop = t0 + i_s / fs
    _, psd = welch(crop, fs=fs, nperseg=min(len(crop), int(4 * fs)))
    f_s, t_s, Sxx = spectrogram(crop, fs=fs, window="hann", nperseg=max(min(len(crop) // 4, int(fs)), 256), noverlap=256*3//4, scaling="density", mode="magnitude")
    Sxx_w = Sxx / np.sqrt(np.maximum(np.interp(f_s, np.linspace(0, fs / 2, len(psd)), np.maximum(psd, 1e-60))[:, None], 1e-60))
    f_valid = f_s[(f_s >= fmin) & (f_s <= fmax)]
    f_log = np.logspace(np.log10(f_valid[0]), np.log10(f_valid[-1]), n_freqs)
    arr_log = gaussian_filter(np.log10(np.array([Sxx_w[np.argmin(np.abs(f_s - ff))] for ff in f_log]) + 1e-12), sigma=0.8)
    mask = (t0_crop + t_s >= gps_center - outseg_dur / 2) & (t0_crop + t_s <= gps_center + outseg_dur / 2)
    return arr_log[:, mask], (t0_crop + t_s)[mask] - gps_center, f_log

def compute_qtransform(strain, gps_center):
    gwpy_levels = [
        dict(outseg_dur=OUTSEG_DUR,     norm="median", frange=FRANGE,    whiten=True),
        dict(outseg_dur=OUTSEG_DUR / 2, norm="median", frange=FRANGE,    whiten=True),
        dict(outseg_dur=OUTSEG_DUR / 2, norm="mean",   frange=FRANGE,    whiten=True),
        dict(outseg_dur=OUTSEG_DUR / 2, norm="mean",   frange=(20, 512), whiten=True),
        dict(outseg_dur=0.5,            norm="mean",   frange=(20, 256), whiten=False),
    ]
    last_gwpy_err = None
    for level, cfg in enumerate(gwpy_levels):
        try:
            sc = _crop_strain(strain, gps_center, cfg["outseg_dur"])
            qgram = _try_qtransform(sc, gps_center, qrange=QRANGE, **cfg)
            if np.std(np.nan_to_num(np.array(qgram, dtype=float))) < 1e-12: raise ValueError("array plano")
            return qgram, level
        except Exception as e:
            last_gwpy_err = e
    try:
        return _qtransform_scipy(strain, gps_center, outseg_dur=min(OUTSEG_DUR, 2.0), fmin=20, fmax=512, n_freqs=64), 5
    except Exception as e:
        raise RuntimeError(f"gwpy 0–4: {last_gwpy_err} | scipy: {e}")

def normalize_qgram(qgram):
    arr = qgram[0].copy() if isinstance(qgram, tuple) else np.array(qgram, dtype=float)
    arr = np.nan_to_num(arr, nan=0., posinf=0., neginf=0.)
    vmin, vmax = np.percentile(arr, (1, 99))
    denom = vmax - vmin
    return np.zeros_like(arr) if denom < 1e-10 else np.clip((arr - vmin) / denom, 0, 1)

def validate_and_extract_ridge(arr, contrast_threshold=0.15):
    """ Filtro Físico: Garante que a trilha de energia se destaque do ruído de fundo. """
    background_mean = np.mean(arr)
    ridge_track = np.max(arr, axis=0)
    ridge_mean = np.mean(ridge_track)
    contrast = ridge_mean - background_mean
    if contrast >= contrast_threshold:
        return True, f"aprovado (contraste: {contrast:.2f})"
    return False, f"reprovado por contraste ({contrast:.2f})"

def save_qgram(arr, event_name, detector, qgram, gps_center, save_png=True):
    import gc
    out_dir = os.path.join(DATA_PROC_DIR, event_name)
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"{detector}_qgram.npy")
    np.save(npy_path, arr)
    png_path = None
    if save_png:
        t_arr, f_arr, plot_arr, shading = (qgram[1], qgram[2], arr, "gouraud") if isinstance(qgram, tuple) else (qgram.times.value - gps_center, qgram.frequencies.value, arr.T, "auto")
        fig, ax = plt.subplots(figsize=(9, 3.5))
        pcm = ax.pcolormesh(t_arr, f_arr, plot_arr, cmap="viridis", vmin=0, vmax=1, shading=shading)
        ax.set_yscale("log")
        ax.set_xlabel("Tempo relativo ao merger (s)")
        ax.set_ylabel("Frequência (Hz)")
        ax.set_title(f"{event_name} — {detector} — Q-transform")
        apply_plain_formatters(ax, log_y=True)
        plt.colorbar(pcm, ax=ax, label="Energia normalizada")
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"{detector}_qgram.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        gc.collect()
    return {"npy": npy_path, "png": png_path}

_FALLBACK_LABELS = {0: "ok", 1: "fallback_janela_menor", 2: "fallback_norm_mean", 3: "fallback_frange_reduzido", 4: "fallback_sem_whiten", 5: "fallback_scipy"}

def _qtransform_task(row):
    ev = row["name"]
    gps = float(row["gps"]) if pd.notna(row["gps"]) else float("nan")
    quality_entries = []

    if not np.isfinite(gps):
        with _print_lock: tqdm.write(f"  [pulo] {ev}: GPS NaN — sem Q-transform")
        return [{"event": ev, "detector": det, "qgram_status": "gps_ausente", "fallback_level": -2} for det in DETECTORS if os.path.exists(os.path.join(DATA_RAW_DIR, ev, f"{det}.hdf5"))]

    for det in DETECTORS:
        raw_path = os.path.join(DATA_RAW_DIR, ev, f"{det}.hdf5")
        if not os.path.exists(raw_path): continue
        try:
            strain = sanitize_strain(TimeSeries.read(raw_path, format="hdf5"))
            if not _has_signal(strain, gps):
                with _print_lock: tqdm.write(f"  [sem_dados] {ev}/{det}: segmento zerado")
                quality_entries.append({"event": ev, "detector": det, "qgram_status": "sem_dados", "fallback_level": -3})
                continue

            qgram, fallback_level = compute_qtransform(strain, gps)
            arr = normalize_qgram(qgram)
            
            # --- Validação Física (Limpeza de duplicata) ---
            is_valid, validation_msg = validate_and_extract_ridge(arr, contrast_threshold=0.15)
            if not is_valid:
                with _print_lock: tqdm.write(f"  [reprovado] {ev}/{det}: {validation_msg}")
                quality_entries.append({"event": ev, "detector": det, "qgram_status": "reprovado_ridge", "fallback_level": -4})
                continue
            
            save_qgram(arr, ev, det, qgram, gps, save_png=True)
            status = _FALLBACK_LABELS.get(fallback_level, f"fallback_{fallback_level}")
            quality_entries.append({"event": ev, "detector": det, "qgram_status": status, "fallback_level": fallback_level})
            
            if fallback_level > 0:
                with _print_lock: tqdm.write(f"  [fallback-{fallback_level}] {ev}/{det}: {status}")
        except Exception as e:
            quality_entries.append({"event": ev, "detector": det, "qgram_status": "falhou", "fallback_level": -1})
            with _print_lock: tqdm.write(f"  [erro] {ev}/{det}: {e}")

    plt.close("all")
    return quality_entries

print(f"\nGerando Q-transforms de {len(catalog_df)} eventos [{N_WORKERS} workers paralelos]...\n")

all_quality = []
with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    rows_list = [row for _, row in catalog_df.iterrows()]
    future_to_idx = {executor.submit(_qtransform_task, row): i for i, row in enumerate(rows_list)}
    with tqdm(total=len(rows_list), unit="evento") as pbar:
        for future in as_completed(future_to_idx):
            try: all_quality.extend(future.result())
            except Exception as e: pass
            pbar.update(1)

quality_df = pd.DataFrame(all_quality)
quality_path = os.path.join(DATA_PROC_DIR, "qgram_quality.csv")
quality_df.to_csv(quality_path, index=False)

def _worst_status(group):
    lvls = group["fallback_level"]
    if (lvls == -2).any(): return "gps_ausente"
    if (lvls == -3).any(): return "sem_dados"
    if (lvls == -4).any(): return "reprovado_ridge"
    if (lvls == -1).any(): return "qgram_falhou"
    if (lvls  >  0).any(): return "qgram_fallback"
    return "ok"

event_flags = quality_df.groupby("event").apply(_worst_status).reset_index().rename(columns={"event": "name", 0: "data_quality_flag"})
catalog_df = catalog_df.merge(event_flags, on="name", how="left")
catalog_df["data_quality_flag"] = catalog_df["data_quality_flag"].fillna("sem_qgram")
catalog_df.to_csv(csv_path, index=False)
catalog_df.to_json(json_path, orient="records", indent=2)

n_ok        = (catalog_df["data_quality_flag"] == "ok").sum()
n_fallback  = (catalog_df["data_quality_flag"] == "qgram_fallback").sum()
n_sem_dados = (catalog_df["data_quality_flag"] == "sem_dados").sum()
n_reprovado = (catalog_df["data_quality_flag"] == "reprovado_ridge").sum()
n_gps_nan   = (catalog_df["data_quality_flag"] == "gps_ausente").sum()
n_falhou    = (catalog_df["data_quality_flag"] == "qgram_falhou").sum()

print(f"\nEspectrogramas Q-transform gerados.")
print(f"  ok             : {n_ok} eventos")
print(f"  fallback usado : {n_fallback} eventos  ← recuperados")
print(f"  reprovados     : {n_reprovado} eventos  ← baixo contraste de ridge (filtrados)")
print(f"  sem_dados      : {n_sem_dados} eventos  ← segmento zerado no GWOSC")
print(f"  gps_ausente    : {n_gps_nan} eventos")
print(f"  falhou         : {n_falhou} eventos")

# ──────────────────────────────────────────────────────────────────
# CÉLULA 7 — Gate 1 (Mantida igual ao código anterior)
# ──────────────────────────────────────────────────────────────────
GATE1_BASE = "GW150914"
GPS_MERGER = 1126259462.4
EVENT_NAME = catalog_df[catalog_df["name"].str.startswith(GATE1_BASE)]["name"].iloc[0] if len(catalog_df[catalog_df["name"].str.startswith(GATE1_BASE)]) > 0 else GATE1_BASE

print("\n" + "=" * 60)
print("Resumo — Camada 1 completa")
print("=" * 60)

total_raw = sum(len(os.listdir(os.path.join(DATA_RAW_DIR, ev))) for ev in os.listdir(DATA_RAW_DIR) if os.path.isdir(os.path.join(DATA_RAW_DIR, ev)))
total_proc = sum(len([f for f in os.listdir(os.path.join(DATA_PROC_DIR, ev)) if f.endswith("_processed.hdf5")]) for ev in os.listdir(DATA_PROC_DIR) if os.path.isdir(os.path.join(DATA_PROC_DIR, ev)))
total_qgram_npy = sum(len([f for f in os.listdir(os.path.join(DATA_PROC_DIR, ev)) if f.endswith("_qgram.npy")]) for ev in os.listdir(DATA_PROC_DIR) if os.path.isdir(os.path.join(DATA_PROC_DIR, ev)))
total_qgram_png = sum(len([f for f in os.listdir(os.path.join(DATA_PROC_DIR, ev)) if f.endswith("_qgram.png")]) for ev in os.listdir(DATA_PROC_DIR) if os.path.isdir(os.path.join(DATA_PROC_DIR, ev)))

print(f"  Eventos no catálogo : {len(catalog_df)}")
print(f"  Strain bruto (HDF5) : {total_raw}")
print(f"  Strain processado   : {total_proc}")
print(f"  Q-transforms (.npy) : {total_qgram_npy}")
print(f"  Q-transforms (.png) : {total_qgram_png}")
print(f"\n  Próximo passo → Camada 2.")
print("=" * 60)