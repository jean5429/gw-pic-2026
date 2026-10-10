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
    de gerar tick labels com $\\mathdefault{...}$ (causa o ParseException
    em algumas versões; unicode_minus=False sozinho não resolve, é o
    \\mathdefault em si que quebra o parser).
  - Q-transform agora usa o strain raw em vez do processado. O q_transform
    do gwpy faz whitening interno; passar um strain já whitened achata a
    PSD a ponto de aparecer divide-by-zero em alguns segmentos (mais comum
    em V1).
  - Gate 1 resolve o nome do evento dinamicamente (GW150914 → GW150914-v4)
    para reaproveitar o cache do download.
"""


# ──────────────────────────────────────────────────────────────────
# CÉLULA 1 — Instalação de dependências
# ──────────────────────────────────────────────────────────────────
import subprocess
import sys


def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Precisa ter instalado scipy
# pip install "scipy<1.12.0"
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

# Catálogos GWTC. Usar os três cumulativos cobre todos os 90 eventos.
# Alguns aparecem em múltiplos catálogos com versões diferentes; a
# deduplicação é feita lá embaixo, mantendo a versão mais recente.
CATALOG_NAMES = [
    "GWTC-1-confident",
    "GWTC-2.1-confident",
    "GWTC-3-confident",
]
N_EVENTS_TARGET = 90

DETECTORS = ["H1", "L1", "V1"]

# Janela em torno do merger
CROP_BEFORE = 16  # s
CROP_AFTER = 16   # s

SAMPLE_RATE = 4096  # Hz
NYQUIST = SAMPLE_RATE / 2

# Filtro bandpass. Nyquist=2048, então fica em 2000 com folga.
BANDPASS_FMIN = 20
BANDPASS_FMAX = 2000
BANDPASS_ORDER = 4

# Q-transform.
# FRANGE máxima fica em 1024 Hz. Com QRANGE=(4, 64), se você pede 2048 Hz
# o gwpy reduz automaticamente pra ~1291 Hz e cospe um UserWarning chato.
QRANGE = (4, 64)
FRANGE = (20, 1024)
OUTSEG_DUR = 4  # janela do espectrograma centrada no merger (s)

# Diretórios
BASE_DIR = "camada1_gwtc3"
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGS_DIR = os.path.join(BASE_DIR, "figs")

for d in [DATA_RAW_DIR, DATA_PROC_DIR, FIGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Cache local do projeto para evitar PermissionError no cache padrão:
# C:\Users\<usuario>\.astropy\cache\...
# Isso também deixa a execução mais portável entre máquinas/ambientes.
LOCAL_CACHE_DIR = os.path.abspath(os.path.join(BASE_DIR, "cache"))
ASTROPY_CACHE_DIR = os.path.join(LOCAL_CACHE_DIR, "astropy")
os.makedirs(ASTROPY_CACHE_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", LOCAL_CACHE_DIR)
os.environ.setdefault("ASTROPY_CACHE_DIR", ASTROPY_CACHE_DIR)


# Formatter sem mathtext, usado em todos os plots com escala log.
# Cobre bem tanto frequências (10, 100, 1000 Hz) quanto tempos
# (-2.0, 0.5, 1.5 s). Retorna sempre string pura, nunca com $ ou \mathdefault.
def _plain_format(value, pos=None):
    if value == 0:
        return "0"
    av = abs(value)
    if av >= 1:
        # Inteiro (ou quase) → sem casas decimais; senão %g compacto.
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value))}"
        return f"{value:g}"
    return f"{value:.2g}"


_PLAIN_FMT = mticker.FuncFormatter(_plain_format)


def apply_plain_formatters(ax, log_y=False):
    """Trava os formatters dos dois eixos no _plain_format. Em escala
    log o matplotlib enche o eixo de minor ticks; suprime os labels
    deles com NullFormatter para o gráfico não virar uma sopa."""
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
    """
    Junta eventos de vários catálogos e remove duplicatas pelo nome base
    (GW150914-v3 e GW150914-v2 viram só GW150914-v3, a versão mais nova).
    """
    seen = {}  # base_name -> (versao_int, nome_completo)
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
    """Puxa parâmetros físicos do evento com retry para RemoteDisconnected."""
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

# Reparo: re-fetch sequencial para eventos que ainda chegaram com GPS NaN
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
                print(f"    ✓ {ev}: GPS = {fixed['gps']}")
                break
            time.sleep(2)
        else:
            print(f"    ✗ {ev}: GPS ainda NaN")
    catalog_df = catalog_df.sort_values("gps").reset_index(drop=True)


csv_path = os.path.join(DATA_PROC_DIR, "gwtc3_catalog.csv")
json_path = os.path.join(DATA_PROC_DIR, "gwtc3_catalog.json")
catalog_df.to_csv(csv_path, index=False)
catalog_df.to_json(json_path, orient="records", indent=2)

if len(catalog_df) < N_EVENTS_TARGET:
    print(f"\n[aviso] Vieram {len(catalog_df)} eventos (esperado: {N_EVENTS_TARGET}).")
    print("        Pode ser indisponibilidade temporária de algum catálogo.")

print(f"\nCatálogo salvo em '{csv_path}' ({len(catalog_df)} eventos)")
print("\nPrimeiros 5:")
print(catalog_df[["name", "gps", "chirp_mass_source",
                  "mass_1_source", "mass_2_source",
                  "luminosity_distance"]].head().to_string(index=False))


# ──────────────────────────────────────────────────────────────────
# CÉLULA 4 — Download do strain (HDF5)
# ──────────────────────────────────────────────────────────────────
from gwpy.timeseries import TimeSeries


def fetch_open_data_safe(detector, t_start, t_end, sample_rate=SAMPLE_RATE,
                         retries=3, prefer_cache=True):
    """
    Baixa strain do GWOSC com proteção contra cache corrompido/bloqueado.

    O erro do log veio de PermissionError dentro do cache do Astropy/GWPy
    em C:\\Users\\...\\.astropy\\cache. Por isso a função tenta:
      1) cache normal/local;
      2) cache=False, para ignorar o cache problemático;
      3) pequenas novas tentativas para falhas temporárias de rede.
    """
    last_err = None
    cache_modes = [prefer_cache, False] if prefer_cache else [False, True]

    for attempt in range(1, retries + 1):
        for use_cache in cache_modes:
            try:
                strain = TimeSeries.fetch_open_data(
                    detector, t_start, t_end,
                    sample_rate=sample_rate,
                    cache=use_cache,
                )
                if strain.sample_rate.value != sample_rate:
                    strain = strain.resample(sample_rate)
                return strain
            except Exception as e:
                last_err = e
                msg = str(e)
                # Se for problema de permissão/cache, não adianta insistir no cache.
                if "Permission denied" in msg or ".astropy" in msg or "cache" in msg.lower():
                    continue

        if attempt < retries:
            time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"Falha ao baixar {detector} entre {t_start} e {t_end}. "
        f"Último erro: {last_err}"
    )


def fetch_and_save_raw(detector, t_start, t_end, out_path=None, prefer_cache=True):
    """Baixa strain com fetch_open_data_safe e opcionalmente salva em HDF5."""
    strain = fetch_open_data_safe(
        detector, t_start, t_end,
        sample_rate=SAMPLE_RATE,
        prefer_cache=prefer_cache,
    )
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        strain.write(out_path, format="hdf5", overwrite=True)
    return strain


# Workers paralelos. O gargalo aqui é I/O de rede, então threads servem.
# 6 é um meio-termo razoável: não estressa o servidor do GWOSC e usa banda.
N_WORKERS = 6

# Lock pra evitar que tqdm.write das threads se sobreponha.
_print_lock = threading.Lock()


def download_strain_event(event_name, gps, detectors=DETECTORS):
    """
    Baixa o strain de um evento em cada detector que tiver dados.
    Salva em data/raw/<evento>/<detector>.hdf5.
    Cada thread escreve no seu próprio diretório, sem corrida.
    """
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
            strain = fetch_and_save_raw(
                det, t_start, t_end,
                out_path=out_path,
                prefer_cache=True,
            )
            saved[det] = out_path
        except Exception as e:
            # Detector pode não ter dados nessa janela, ou pode haver falha temporária de rede. Segue.
            with _print_lock:
                tqdm.write(f"  [aviso] {event_name}/{det}: {e}")

    return saved


def _download_task(row):
    ev = row["name"]
    gps = float(row["gps"])
    saved = download_strain_event(ev, gps)
    with _print_lock:
        tqdm.write(f"  {ev}: {list(saved.keys())}")
    return {
        "event": ev,
        "detectors": str(list(saved.keys())),
        "n_detectors": len(saved),
    }


print(f"\nBaixando strain de {len(catalog_df)} eventos "
      f"[{N_WORKERS} workers paralelos]...\n")

rows_list = [row for _, row in catalog_df.iterrows()]
download_log = [None] * len(rows_list)

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    future_to_idx = {
        executor.submit(_download_task, row): i
        for i, row in enumerate(rows_list)
    }
    with tqdm(total=len(rows_list), unit="evento") as pbar:
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                download_log[idx] = future.result()
            except Exception as e:
                ev = rows_list[idx]["name"]
                download_log[idx] = {"event": ev, "detectors": "[]", "n_detectors": 0}
                tqdm.write(f"  [erro] {ev}: {e}")
            pbar.update(1)

dl_df = pd.DataFrame(download_log)
dl_df.to_csv(os.path.join(DATA_PROC_DIR, "download_summary.csv"), index=False)

ok = (dl_df["n_detectors"] > 0).sum()
print(f"\nDownload concluído: {ok}/{len(catalog_df)} eventos com strain disponível.")


# ──────────────────────────────────────────────────────────────────
# CÉLULA 5 — Pré-processamento: whitening + bandpass Butterworth
# ──────────────────────────────────────────────────────────────────

def load_raw_strain(event_name, detector):
    path = os.path.join(DATA_RAW_DIR, event_name, f"{detector}.hdf5")
    if not os.path.exists(path):
        return None
    return TimeSeries.read(path, format="hdf5")


def bandpass_butter(strain, fmin, fmax, order=4):
    """
    Bandpass Butterworth manual com scipy.

    Por que não usar strain.bandpass() do gwpy? Porque em versões recentes
    de scipy + gwpy esse caminho cai no firwin que não aceita mais o kwarg
    'analog'. Implementar direto com butter+sosfiltfilt evita o problema
    todo e ainda dá filtro de fase zero (o sosfiltfilt aplica forward-backward).
    """
    fs = float(strain.sample_rate.value)
    nyq = fs / 2.0
    low = fmin / nyq
    high = fmax / nyq

    if not (0 < low < 1) or not (0 < high < 1) or low >= high:
        raise ValueError(
            f"Banda inválida: fmin={fmin}, fmax={fmax}, fs={fs} "
            f"(low={low:.3f}, high={high:.3f})"
        )

    # output='sos' dá estabilidade numérica boa em ordens > 4.
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    filtered_arr = sosfiltfilt(sos, strain.value)

    return TimeSeries(
        filtered_arr,
        t0=strain.t0,
        dt=strain.dt,
        unit=strain.unit,
        channel=strain.channel,
        name=getattr(strain, "name", None),
    )


def preprocess_strain(strain):
    """Whitening espectral + bandpass Butterworth 20–2000 Hz."""
    white = strain.whiten(fftlength=4, fduration=4)
    bp = bandpass_butter(white, BANDPASS_FMIN, BANDPASS_FMAX, order=BANDPASS_ORDER)
    return bp


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
        if raw is None:
            continue
        try:
            proc = preprocess_strain(raw)
            path = save_processed(proc, ev, det)
            results[det] = path
        except Exception as e:
            with _print_lock:
                tqdm.write(f"  [erro] {ev}/{det}: {e}")
    with _print_lock:
        tqdm.write(f"  {ev}: {list(results.keys())}")
    return {"event": ev, "processed": str(list(results.keys()))}


print(f"\nPré-processando strain de {len(catalog_df)} eventos "
      f"[{N_WORKERS} workers paralelos]...\n")

rows_list = [row for _, row in catalog_df.iterrows()]
proc_log = [None] * len(rows_list)

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    future_to_idx = {
        executor.submit(_preprocess_task, row): i
        for i, row in enumerate(rows_list)
    }
    with tqdm(total=len(rows_list), unit="evento") as pbar:
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                proc_log[idx] = future.result()
            except Exception as e:
                ev = rows_list[idx]["name"]
                proc_log[idx] = {"event": ev, "processed": "[]"}
                tqdm.write(f"  [erro] {ev}: {e}")
            pbar.update(1)

pd.DataFrame(proc_log).to_csv(
    os.path.join(DATA_PROC_DIR, "processing_log.csv"), index=False
)
print("\nPré-processamento concluído.")


# ──────────────────────────────────────────────────────────────────
# CÉLULA 6 — Espectrogramas Q-transform
# ──────────────────────────────────────────────────────────────────

def sanitize_strain(strain):
    arr = strain.value.copy().astype(np.float64)
    bad = ~np.isfinite(arr)
    if bad.any():
        x    = np.arange(len(arr))
        good = np.where(~bad)[0]
        arr  = np.interp(x, good, arr[good]) if len(good) > 1 else arr * 0.0
    lo, hi = np.percentile(arr, [0.001, 99.999])
    arr = np.clip(arr, lo, hi)
    sigma = float(np.std(arr))
    if sigma > 0 and np.isfinite(sigma):
        arr = arr + np.random.default_rng(42).normal(0, sigma * 1e-6, len(arr))
    return TimeSeries(arr, t0=strain.t0, dt=strain.dt,
                      unit=strain.unit, channel=strain.channel)


def _has_signal(strain, gps_center, window=4.0, min_std=1e-25):
    """
    Verifica se o strain tem sinal real na janela ao redor do merger.

    Arquivos V1 do O3 frequentemente têm segmentos inteiros zerados
    (data gaps preenchidos com zeros pelo GWOSC). O gwpy e o scipy
    retornam arrays planos nesses casos — não é erro de parâmetro,
    é ausência real de dados. Detectar isso antes evita ciclos de
    fallback inúteis e registra o status correto ('sem_dados').
    """
    t0   = float(strain.t0.value)
    dt   = float(strain.dt.value)
    arr  = strain.value
    i_s  = max(0, int((gps_center - window / 2 - t0) / dt))
    i_e  = min(len(arr), int((gps_center + window / 2 - t0) / dt))
    seg  = arr[i_s:i_e]
    if len(seg) < 10:
        return False
    return float(np.std(seg)) > min_std


def _crop_strain(strain, gps_center, outseg_dur, margin=6.0):
    t0   = float(strain.t0.value)
    tend = t0 + len(strain) * float(strain.dt.value)
    t_s  = max(t0,   gps_center - outseg_dur / 2 - margin)
    t_e  = min(tend, gps_center + outseg_dur / 2 + margin)
    if (t_e - t_s) < outseg_dur:
        return strain
    return strain.crop(t_s, t_e)


def _try_qtransform(strain, gps_center, outseg_dur, norm, qrange, frange,
                    whiten=True):
    t_start = gps_center - outseg_dur / 2
    t_end   = gps_center + outseg_dur / 2
    if not (np.isfinite(t_start) and np.isfinite(t_end)):
        raise ValueError(f"GPS inválido: {gps_center}")
    t0   = float(strain.t0.value)
    tend = t0 + len(strain) * float(strain.dt.value)
    if t_start < t0 or t_end > tend:
        raise ValueError(f"outseg ({t_start:.1f}–{t_end:.1f}) fora do strain ({t0:.1f}–{tend:.1f})")
    with np.errstate(divide="ignore", invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return strain.q_transform(
                qrange=qrange, frange=frange,
                outseg=(t_start, t_end),
                logf=True, norm=norm, whiten=whiten,
            )


def _qtransform_scipy(strain, gps_center, outseg_dur=2.0, fmin=20, fmax=512, n_freqs=64):
    """
    Q-transform manual com scipy.signal.spectrogram + whitening Welch.
    Usado quando o gwpy levanta 'cannot access local variable out'.

    Retorna (arr, t_arr, f_arr) onde arr.shape == (n_freqs, n_times),
    t_arr é relativo ao gps_center, f_arr em Hz escala log.
    """
    from scipy.signal import spectrogram, welch
    from scipy.ndimage import gaussian_filter

    fs      = float(strain.sample_rate.value)
    t0      = float(strain.t0.value)
    data    = strain.value.copy().astype(np.float64)
    margin  = 4.0
    i_s     = max(0, int((gps_center - outseg_dur / 2 - margin - t0) * fs))
    i_e     = min(len(data), int((gps_center + outseg_dur / 2 + margin - t0) * fs))
    crop    = data[i_s:i_e]
    t0_crop = t0 + i_s / fs

    if len(crop) < 256:
        raise ValueError("Segmento muito curto para spectrogram")

    # Whitening via Welch: divide pelo sqrt da PSD estimada
    nperseg_w = min(len(crop), int(4 * fs))
    _, psd    = welch(crop, fs=fs, nperseg=nperseg_w)
    psd       = np.maximum(psd, 1e-60)

    # Spectrogram com janela Hann
    nperseg = max(min(len(crop) // 4, int(fs)), 256)
    f_s, t_s, Sxx = spectrogram(
        crop, fs=fs, window="hann",
        nperseg=nperseg, noverlap=nperseg * 3 // 4,
        scaling="density", mode="magnitude",
    )
    psd_i = np.interp(f_s, np.linspace(0, fs / 2, len(psd)), psd)
    Sxx_w = Sxx / np.sqrt(np.maximum(psd_i[:, None], 1e-60))

    # Re-grida para escala log em frequência → shape (n_freqs, n_times)
    f_valid = f_s[(f_s >= fmin) & (f_s <= fmax)]
    if len(f_valid) < 2:
        raise ValueError(f"Nenhuma frequência em [{fmin},{fmax}] Hz")
    f_log   = np.logspace(np.log10(f_valid[0]), np.log10(f_valid[-1]), n_freqs)
    arr_log = np.array([Sxx_w[np.argmin(np.abs(f_s - ff))] for ff in f_log])
    # arr_log.shape == (n_freqs, n_times) ✓

    # Log + suavização gaussian (melhora visual, reduz ruído granular)
    arr_log = np.log10(arr_log + 1e-12)
    arr_log = gaussian_filter(arr_log, sigma=0.8)

    # Seleciona janela outseg centrada no merger
    t_abs = t0_crop + t_s
    mask  = (t_abs >= gps_center - outseg_dur / 2) & \
            (t_abs <= gps_center + outseg_dur / 2)
    if mask.sum() < 2:
        mask = np.ones(len(t_abs), dtype=bool)

    arr_out = arr_log[:, mask]   # (n_freqs, n_times_selected)
    t_rel   = t_abs[mask] - gps_center

    if np.std(arr_out) < 1e-12:
        raise ValueError("scipy array plano após spectrogram")
    return (arr_out, t_rel, f_log)


def compute_qtransform(strain, gps_center):
    """
    Q-transform com cascata gwpy (níveis 0–4) + scipy (nível 5).

      0: outseg=4s,   norm='median', frange padrão,  whiten=True
      1: outseg=2s,   norm='median', frange padrão,  whiten=True
      2: outseg=2s,   norm='mean',   frange padrão,  whiten=True
      3: outseg=2s,   norm='mean',   frange=(20,512), whiten=True
      4: outseg=0.5s, norm='mean',   frange=(20,256), whiten=False
      5: scipy spectrogram manual — contorna o bug interno do gwpy

    Pré-condição: chamar _has_signal() antes. Se retornar False,
    não chamar esta função — registrar 'sem_dados' diretamente.
    """
    if not np.isfinite(gps_center):
        raise RuntimeError(f"GPS inválido: {gps_center}")

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
            sc    = _crop_strain(strain, gps_center, cfg["outseg_dur"])
            # Injetamos o parâmetro qrange=QRANGE explicitamente:
            qgram = _try_qtransform(sc, gps_center, qrange=QRANGE, **cfg)
            arr   = np.nan_to_num(np.array(qgram, dtype=float), nan=0., posinf=0., neginf=0.)
            if np.std(arr) < 1e-12:
                raise ValueError("array plano")
            return qgram, level
        except Exception as e:
            last_gwpy_err = e

    # Nível 5: scipy
    try:
        result = _qtransform_scipy(strain, gps_center,
                                   outseg_dur=min(OUTSEG_DUR, 2.0),
                                   fmin=20, fmax=512, n_freqs=64)
        return result, 5
    except Exception as e:
        raise RuntimeError(f"gwpy 0–4: {last_gwpy_err} | scipy: {e}")


def normalize_qgram(qgram):
    """Normaliza [0,1] via P1–P99. Aceita gwpy.Spectrogram ou tupla (arr,t,f)."""
    arr = qgram[0].copy() if isinstance(qgram, tuple) else np.array(qgram, dtype=float)
    arr = np.nan_to_num(arr, nan=0., posinf=0., neginf=0.)
    vmin, vmax = np.percentile(arr, (1, 99))
    denom = vmax - vmin
    return np.zeros_like(arr) if denom < 1e-10 else np.clip((arr - vmin) / denom, 0, 1)

def validate_and_extract_ridge(arr, contrast_threshold=0.25):
    """
    Testa a qualidade do espectrograma analisando o contraste do ridge.
    arr: Matriz normalizada do Q-transform (n_freqs, n_times) entre 0 e 1.
    """
    # 1. Calcula a energia média de todo o gráfico (ruído de fundo)
    background_mean = np.mean(arr)
    
    # 2. Encontra a trilha de maior energia (o suposto Ridge)
    ridge_track = np.max(arr, axis=0)
    ridge_mean = np.mean(ridge_track)
    
    # 3. Calcula o contraste real entre a onda e o fundo
    contrast = ridge_mean - background_mean
    
    # Se o whitening tiver falhado, o gráfico será um "borrão" com médias muito parecidas.
    # Se tiver funcionado, o fundo será escuro e a onda brilhante (alto contraste).
    if contrast >= contrast_threshold:
        return True, f"Aprovado (Contraste: {contrast:.2f})"
    else:
        return False, f"Reprovado: Whitening falhou, sinal achatado (Contraste: {contrast:.2f})"


def save_qgram(arr, event_name, detector, qgram, gps_center, save_png=True):
    """
    Salva .npy e PNG.

    Orientação esperada pelo pcolormesh(t, f, C):
      C.shape == (n_freqs, n_times)  →  eixo Y = frequência, eixo X = tempo

    - gwpy Spectrogram: array interno shape (n_times, n_freqs) → aplica .T
    - scipy tuple     : arr já vem (n_freqs, n_times)           → usa direto
    """
    import gc
    out_dir  = os.path.join(DATA_PROC_DIR, event_name)
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"{detector}_qgram.npy")
    np.save(npy_path, arr)

    png_path = None
    if save_png:
        if isinstance(qgram, tuple):
            # scipy: (arr, t_rel, f_arr) — arr já é (n_freqs, n_times)
            _, t_arr, f_arr = qgram
            plot_arr = arr          # (n_freqs, n_times) — correto para pcolormesh(t,f,C)
            shading  = "gouraud"   # robusto para poucos pontos temporais
        else:
            # gwpy Spectrogram: interno é (n_times, n_freqs) → transpõe
            t_arr    = qgram.times.value - gps_center
            f_arr    = qgram.frequencies.value
            plot_arr = arr.T        # (n_times,n_freqs).T → (n_freqs,n_times)
            shading  = "auto"

        fig, ax = plt.subplots(figsize=(9, 3.5))
        pcm = ax.pcolormesh(t_arr, f_arr, plot_arr, cmap="viridis",
                            vmin=0, vmax=1, shading=shading)
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
        gc.collect()   # evita leak de memória em execuções longas com muitos eventos

    return {"npy": npy_path, "png": png_path}



_FALLBACK_LABELS = {
    0: "ok",
    1: "fallback_janela_menor",
    2: "fallback_norm_mean",
    3: "fallback_frange_reduzido",
    4: "fallback_sem_whiten",
    5: "fallback_scipy",
}


def _qtransform_task(row):
    ev  = row["name"]
    gps = float(row["gps"]) if pd.notna(row["gps"]) else float("nan")
    quality_entries = []

    # GPS NaN: registra e pula
    if not np.isfinite(gps):
        with _print_lock:
            tqdm.write(f"  [pulo] {ev}: GPS NaN — sem Q-transform")
        for det in DETECTORS:
            if os.path.exists(os.path.join(DATA_RAW_DIR, ev, f"{det}.hdf5")):
                quality_entries.append({
                    "event": ev, "detector": det,
                    "qgram_status": "gps_ausente", "fallback_level": -2,
                })
        return quality_entries

    for det in DETECTORS:
        raw_path = os.path.join(DATA_RAW_DIR, ev, f"{det}.hdf5")
        if not os.path.exists(raw_path):
            continue
        try:
            strain = TimeSeries.read(raw_path, format="hdf5")
            strain = sanitize_strain(strain)

            # ── Verificação de dados antes de qualquer Q-transform ────────
            # Segmentos zerados (data gaps do GWOSC) causam 'array plano'
            # em todos os níveis gwpy e scipy. Detectar aqui economiza tempo
            # e registra o status correto no CSV.
            if not _has_signal(strain, gps):
                with _print_lock:
                    tqdm.write(f"  [sem_dados] {ev}/{det}: segmento zerado/sem sinal")
                quality_entries.append({
                    "event": ev, "detector": det,
                    "qgram_status": "sem_dados", "fallback_level": -3,
                })
                continue

            # Gera o Q-transform (que já tenta fazer o whitening internamente)
            qgram, fallback_level = compute_qtransform(strain, gps)
            arr = normalize_qgram(qgram)
            
            # --- NOVO: Teste de Validação ---
            is_valid, validation_msg = validate_and_extract_ridge(arr)
            
            if not is_valid:
                with _print_lock:
                    tqdm.write(f"  [reprovado] {ev}/{det}: {validation_msg}")
                quality_entries.append({
                    "event": ev, "detector": det,
                    "qgram_status": "reprovado_no_teste", "fallback_level": fallback_level,
                })
                continue # Pula o salvamento deste arquivo!
            # --------------------------------
            
            # Se passou no teste, salva o arquivo PNG e NPY
            save_qgram(arr, ev, det, qgram, gps, save_png=True)

            status = _FALLBACK_LABELS.get(fallback_level, f"fallback_{fallback_level}")

            qgram, fallback_level = compute_qtransform(strain, gps)
            arr    = normalize_qgram(qgram)
            save_qgram(arr, ev, det, qgram, gps, save_png=True)

            status = _FALLBACK_LABELS.get(fallback_level, f"fallback_{fallback_level}")
            quality_entries.append({
                "event": ev, "detector": det,
                "qgram_status": status, "fallback_level": fallback_level,
            })
            if fallback_level > 0:
                with _print_lock:
                    tqdm.write(f"  [fallback-{fallback_level}] {ev}/{det}: {status}")

        except Exception as e:
            quality_entries.append({
                "event": ev, "detector": det,
                "qgram_status": "falhou", "fallback_level": -1,
            })
            with _print_lock:
                tqdm.write(f"  [erro] {ev}/{det}: {e}")

    plt.close("all")
    return quality_entries


print(f"\nGerando Q-transforms de {len(catalog_df)} eventos "
      f"[{N_WORKERS} workers paralelos]...\n")

rows_list  = [row for _, row in catalog_df.iterrows()]
all_quality = []

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    future_to_idx = {
        executor.submit(_qtransform_task, row): i
        for i, row in enumerate(rows_list)
    }
    with tqdm(total=len(rows_list), unit="evento") as pbar:
        for future in as_completed(future_to_idx):
            try:
                all_quality.extend(future.result())
            except Exception as e:
                idx = future_to_idx[future]
                tqdm.write(f"  [erro] {rows_list[idx]['name']}: {e}")
            pbar.update(1)

quality_df   = pd.DataFrame(all_quality)
quality_path = os.path.join(DATA_PROC_DIR, "qgram_quality.csv")
quality_df.to_csv(quality_path, index=False)

def _worst_status(group):
    lvls = group["fallback_level"]
    if (lvls == -2).any(): return "gps_ausente"
    if (lvls == -3).any(): return "sem_dados"
    if (lvls == -1).any(): return "qgram_falhou"
    if (lvls  >  0).any(): return "qgram_fallback"
    return "ok"

event_flags = (
    quality_df.groupby("event").apply(_worst_status)
    .reset_index().rename(columns={"event": "name", 0: "data_quality_flag"})
)
catalog_df = catalog_df.merge(event_flags, on="name", how="left")
catalog_df["data_quality_flag"] = catalog_df["data_quality_flag"].fillna("sem_qgram")
catalog_df.to_csv(csv_path, index=False)
catalog_df.to_json(json_path, orient="records", indent=2)

n_ok       = (catalog_df["data_quality_flag"] == "ok").sum()
n_fallback = (catalog_df["data_quality_flag"] == "qgram_fallback").sum()
n_sem_dados= (catalog_df["data_quality_flag"] == "sem_dados").sum()
n_gps_nan  = (catalog_df["data_quality_flag"] == "gps_ausente").sum()
n_falhou   = (catalog_df["data_quality_flag"] == "qgram_falhou").sum()

print(f"\nEspectrogramas Q-transform gerados.")
print(f"  ok             : {n_ok} eventos")
print(f"  fallback usado : {n_fallback} eventos  ← recuperados (gwpy ou scipy)")
print(f"  sem_dados      : {n_sem_dados} eventos  ← segmento zerado no GWOSC (data gap)")
print(f"  gps_ausente    : {n_gps_nan} eventos   ← GPS NaN na API")
print(f"  falhou         : {n_falhou} eventos    ← erro inesperado")
print(f"  Log salvo em   : {quality_path}")
print(f"  Catálogo atualizado com coluna 'data_quality_flag': {csv_path}")



# ──────────────────────────────────────────────────────────────────
# CÉLULA 7 — Gate 1: Reprodução da Figura 1 do paper GW150914
#            Abbott et al. (2016), PRL 116, 061102
# ──────────────────────────────────────────────────────────────────

GATE1_BASE = "GW150914"   # nome base; resolvemos pra versão real abaixo
GPS_MERGER = 1126259462.4
T_START = GPS_MERGER - 16
T_END = GPS_MERGER + 16

# Janela dos painéis de strain, igual ao paper (foco no chirp).
T_PLOT_BEFORE = 0.45
T_PLOT_AFTER = 0.05

# Janela do Q-transform: 1 s centrado no merger.
OUTSEG_GATE1 = (GPS_MERGER - 0.5, GPS_MERGER + 0.5)

COLORS = {"H1": "#0d6efd", "L1": "#dc3545"}


def resolve_event_name(base_name, df):
    """
    Resolve 'GW150914' para o nome com versão usado no catálogo
    (ex: 'GW150914-v4'). Sem isso, o Gate 1 não acha os arquivos
    cacheados em data/raw/GW150914-v4/ e baixa tudo de novo.
    """
    matches = df[df["name"].str.startswith(base_name + "-")
                 | (df["name"] == base_name)]
    if len(matches) > 0:
        return matches.iloc[0]["name"]
    return base_name


EVENT_NAME = resolve_event_name(GATE1_BASE, catalog_df)
print(f"\nNome resolvido para o Gate 1: {EVENT_NAME}")


def load_or_download_gate1(detector):
    """
    Tenta usar o strain processado da etapa 5; se não, processa do raw;
    se não tiver raw, baixa do GWOSC.
    """
    proc_path = os.path.join(DATA_PROC_DIR, EVENT_NAME, f"{detector}_processed.hdf5")
    if os.path.exists(proc_path):
        print(f"  {detector}: carregando processado do disco...")
        return TimeSeries.read(proc_path, format="hdf5")

    raw_path = os.path.join(DATA_RAW_DIR, EVENT_NAME, f"{detector}.hdf5")
    if os.path.exists(raw_path):
        print(f"  {detector}: processando strain bruto...")
        raw = TimeSeries.read(raw_path, format="hdf5")
        return preprocess_strain(raw)

    print(f"  {detector}: baixando do GWOSC sem depender do cache global...")
    raw = fetch_and_save_raw(
        detector, T_START, T_END,
        out_path=raw_path,
        prefer_cache=False,
    )
    proc = preprocess_strain(raw)
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    proc.write(proc_path, format="hdf5", overwrite=True)
    return proc


print("\n" + "=" * 60)
print("Gate 1 — Reprodução da Figura 1 do paper GW150914")
print("Abbott et al. (2016), PRL 116, 061102")
print("=" * 60)

# Painéis temporais: bandpass extra 35–350 Hz, igual o paper.
print("\nCarregando strain processado...")
strains = {}
for det in ["H1", "L1"]:
    s = load_or_download_gate1(det)
    strains[det] = bandpass_butter(s, 35, 350, order=4)

# Q-transforms a partir do raw (mesma lógica da Célula 6).
print("Calculando Q-transforms...")
raw_strains = {}
for det in ["H1", "L1"]:
    raw_path = os.path.join(DATA_RAW_DIR, EVENT_NAME, f"{det}.hdf5")
    if os.path.exists(raw_path):
        raw_strains[det] = TimeSeries.read(raw_path, format="hdf5")
    else:
        raw_strains[det] = fetch_and_save_raw(
            det, T_START, T_END,
            out_path=raw_path,
            prefer_cache=False,
        )

qgrams = {}
for det in ["H1", "L1"]:
    strain_qt = sanitize_strain(raw_strains[det])
    with np.errstate(divide="ignore", invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qgrams[det] = strain_qt.q_transform(
                qrange=QRANGE,
                frange=(25, 512),  # banda do paper
                outseg=OUTSEG_GATE1,
                logf=True,
                norm="median",
            )

# Figura com 4 painéis: 2 strains temporais + 2 espectrogramas.
print("Gerando figura...")
fig = plt.figure(figsize=(12, 10), facecolor="white")
gs = gridspec.GridSpec(4, 1, hspace=0.5)

# Painéis 1-2: strain whitened+bandpass(35-350) por detector.
for i, det in enumerate(["H1", "L1"]):
    ax = fig.add_subplot(gs[i])
    s = strains[det]
    t = s.times.value - GPS_MERGER
    mask = (t >= -T_PLOT_BEFORE) & (t <= T_PLOT_AFTER)

    ax.plot(t[mask], s.value[mask],
            color=COLORS[det], lw=0.8,
            label=f"{det} — whitened, BP 35–350 Hz")
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Strain (u.a.)", fontsize=9)
    ax.set_xlabel("Tempo relativo ao merger (s)", fontsize=9)
    ax.set_title(f"Painel {i + 1}: {det}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    apply_plain_formatters(ax)

# Painéis 3-4: Q-transforms.
for i, det in enumerate(["H1", "L1"]):
    ax = fig.add_subplot(gs[2 + i])
    qg = qgrams[det]
    t_q = qg.times.value - GPS_MERGER
    f_q = qg.frequencies.value
    arr = np.array(qg)
    vmin, vmax = np.percentile(arr, (1, 99))

    im = ax.pcolormesh(
        t_q, f_q, arr.T,
        vmin=vmin, vmax=vmax,
        cmap="viridis", shading="auto",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Frequência (Hz)", fontsize=9)
    ax.set_xlabel("Tempo relativo ao merger (s)", fontsize=9)
    ax.set_title(f"Painel {i + 3}: {det} — Q-transform",
                 fontsize=10, fontweight="bold")
    ax.axvline(0, color="white", lw=0.8, ls="--", alpha=0.6)
    plt.colorbar(im, ax=ax, label="Energia normalizada")
    apply_plain_formatters(ax, log_y=True)

fig.suptitle(
    f"{EVENT_NAME} — Reprodução da Figura 1\n"
    "Abbott et al. (2016), PRL 116, 061102",
    fontsize=12, fontweight="bold",
)

gate1_dir = os.path.join(DATA_PROC_DIR, EVENT_NAME)
os.makedirs(gate1_dir, exist_ok=True)
gate1_path = os.path.join(gate1_dir, "gate1_figura1_reproducao.png")
fig.savefig(gate1_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n{'=' * 60}")
print(f"Gate 1 concluído.")
print(f"  Figura: {gate1_path}")
print(f"{'=' * 60}")


# ──────────────────────────────────────────────────────────────────
# Resumo final
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Resumo — Camada 1 completa")
print("=" * 60)

total_raw = sum(
    len(os.listdir(os.path.join(DATA_RAW_DIR, ev)))
    for ev in os.listdir(DATA_RAW_DIR)
    if os.path.isdir(os.path.join(DATA_RAW_DIR, ev))
)
total_proc = sum(
    len([f for f in os.listdir(os.path.join(DATA_PROC_DIR, ev))
         if f.endswith("_processed.hdf5")])
    for ev in os.listdir(DATA_PROC_DIR)
    if os.path.isdir(os.path.join(DATA_PROC_DIR, ev))
)
total_qgram_npy = sum(
    len([f for f in os.listdir(os.path.join(DATA_PROC_DIR, ev))
         if f.endswith("_qgram.npy")])
    for ev in os.listdir(DATA_PROC_DIR)
    if os.path.isdir(os.path.join(DATA_PROC_DIR, ev))
)
total_qgram_png = sum(
    len([f for f in os.listdir(os.path.join(DATA_PROC_DIR, ev))
         if f.endswith("_qgram.png")])
    for ev in os.listdir(DATA_PROC_DIR)
    if os.path.isdir(os.path.join(DATA_PROC_DIR, ev))
)

print(f"  Eventos no catálogo : {len(catalog_df)}")
print(f"  Strain bruto (HDF5) : {total_raw}")
print(f"  Strain processado   : {total_proc}")
print(f"  Q-transforms (.npy) : {total_qgram_npy}")
print(f"  Q-transforms (.png) : {total_qgram_png}")
print(f"  Gate 1              : {gate1_path}")
print(f"\n  Próximo passo → Camada 2.")
print("=" * 60)