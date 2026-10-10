"""
Script Principal de Demonstração (GW Student Kit)
=================================================
Este script executa a pipeline completa para análise de Ondas Gravitacionais:
1. Leitura de dados de detectores (.hdf5) de eventos reais do LIGO (ex: GW150914-v4)
   OU geração de sinal sintético se nenhum evento local for especificado.
2. Limpeza por Processamento Digital de Sinais (Filtro Passa-Banda Butterworth 30-400Hz).
3. Geração de Espectrogramas (STFT).
4. Filtro Casado (Matched Filtering) com cálculo da série temporal de SNR.
5. Detecção de Anomalias com Inteligência Artificial (Autoencoders 1D / ML).

Para rodar em modo automático (detecta pastas reais locais como GW150914-v4):
   python main_demo.py

Para rodar em um evento do gwproject, com a massa chirp do catálogo (referencial do detector):
   python gw_student_kit/main_demo.py --event outputs/gwtc/raw/GW150914-v4 --chirp-mass 30.69

Para forçar modo de dados sintéticos:
   python main_demo.py --synthetic
"""

import os
import sys
import glob
import argparse
import numpy as np
from scipy.signal.windows import tukey

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gw_student_kit.synthetic import (
    generate_synthetic_chirp,
    generate_newtonian_chirp,
    generate_detector_noise,
    inject_signal
)
from gw_student_kit.dsp import (
    bandpass_filter,
    compute_psd,
    load_event_folder
)
from gw_student_kit.matched_filter import (
    align_to_merger,
    compute_matched_filter
)
from gw_student_kit.visualization import (
    plot_strain_comparison,
    plot_spectrogram,
    plot_snr_time_series
)
from gw_student_kit.ml_anomaly import (
    detect_anomalies_autoencoder
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "results")
LOWCUT, HIGHCUT = 30.0, 400.0
MERGER_AXIS_LABEL = "Instante de fusão do template (s)"


def matched_filter_at_merger(strain_filtered, template, fs, merger_time, taper_fraction=0.1):
    """Filtro casado na banda do passa-banda, com a SNR indexada pelo instante de fusão.

    O template recebe o mesmo filtro dos dados; com a PSD dos dados filtrados, a SNR
    dentro da banda fica igual à que seria obtida sem o filtro. Uma janela de Tukey
    suprime os transientes do filtro nas bordas, que de outro modo dominam a correlação
    circular, e a PSD é estimada só na região central.
    """
    n = len(strain_filtered)
    edge = int(round(taper_fraction * n / 2.0))
    tapered = strain_filtered * tukey(n, alpha=taper_fraction)
    template_filtered = bandpass_filter(template, fs=fs, lowcut=LOWCUT, highcut=HIGHCUT)
    _, psd = compute_psd(strain_filtered[edge:n - edge], fs=fs)
    snr_series, _, _ = compute_matched_filter(
        tapered, template_filtered, fs=fs, psd=psd, f_low=LOWCUT, f_high=HIGHCUT
    )
    return align_to_merger(snr_series, int(round(merger_time * fs)))


def process_real_event(event_folder_path: str, chirp_mass: float):
    event_name = os.path.basename(os.path.normpath(event_folder_path))
    output_dir = os.path.join(RESULTS_BASE_DIR, event_name)
    os.makedirs(output_dir, exist_ok=True)

    print("==========================================================================")
    print(f"      PROCESSANDO EVENTO REAL DO LIGO/Virgo: {event_name}")
    print("==========================================================================")

    # 1. Carregar arquivos HDF5 do evento
    print(f"\n[PASSO 1] Carregando arquivos HDF5 do diretório '{event_folder_path}'...")
    detectors_data = load_event_folder(event_folder_path)
    print(f"  • Detectores Encontrados: {list(detectors_data.keys())}")

    for det_name, det_info in detectors_data.items():
        strain_raw = det_info['strain']
        fs = det_info['fs']
        gps_start = det_info['gps_start']
        n_samples = len(strain_raw)
        duration = n_samples / fs
        time = np.arange(n_samples) / fs

        print(f"\n--- Detector {det_name} ---")
        print(f"  • N° de Amostras: {n_samples:,} | f_s = {fs:.0f} Hz | Duração = {duration:.2f} s")
        print(f"  • Tempo GPS Inicial (x0): {gps_start:.4f} s")

        # 2. Filtragem Passa-Banda (30 - 400 Hz)
        print("  • [PASSO 2] Aplicando Filtro Passa-Banda (30Hz - 400Hz)...")
        strain_filtered = bandpass_filter(strain_raw, fs=fs, lowcut=LOWCUT, highcut=HIGHCUT)

        zoom_lims = (15.5, 16.5) # Os arquivos de evento do GWOSC centram o evento em 16 s

        comp_path = os.path.join(output_dir, f"{det_name}_01_strain_filtrado.png")
        plot_strain_comparison(
            time, strain_raw, strain_filtered,
            save_path=comp_path,
            title=f"Strain Detector {det_name} ({event_name})",
            xlim=zoom_lims
        )
        print(f"    -> Gráfico de Strain salvo: {comp_path}")

        # 3. Espectrograma Tempo-Frequência
        print("  • [PASSO 3] Gerando Espectrograma...")
        spec_path = os.path.join(output_dir, f"{det_name}_02_espectrograma.png")
        plot_spectrogram(
            strain_filtered, fs=fs,
            save_path=spec_path,
            title=f"Espectrograma {det_name} - {event_name}",
            xlim=zoom_lims
        )
        print(f"    -> Gráfico do Espectrograma salvo: {spec_path}")

        # 4. Filtro Casado com um chirp newtoniano da massa chirp informada
        print(f"  • [PASSO 4] Filtro Casado com chirp newtoniano (M_chirp = {chirp_mass:.2f} M_sol)...")
        template_merger = duration / 2.0
        _, template = generate_newtonian_chirp(
            fs=fs, duration=duration, chirp_mass_solar=chirp_mass,
            t_merger=template_merger, f_min=LOWCUT
        )
        snr_series = matched_filter_at_merger(strain_filtered, template, fs, template_merger)
        peak_idx = int(np.argmax(snr_series))
        print(f"    [+] PICO DE SNR = {snr_series[peak_idx]:.2f} para fusão em t = +{time[peak_idx]:.4f}s "
              f"(GPS {gps_start + time[peak_idx]:.4f}s)")
        print("        Template de ordem dominante e PSD estimada nos mesmos dados: SNR indicativa, "
              "sem significância estatística.")

        snr_path = os.path.join(output_dir, f"{det_name}_03_snr_series.png")
        plot_snr_time_series(
            time, snr_series, threshold=8.0,
            save_path=snr_path,
            title=f"Série Temporal de SNR - {det_name} ({event_name})",
            xlim=zoom_lims,
            xlabel=MERGER_AXIS_LABEL
        )
        print(f"    -> Gráfico de SNR salvo: {snr_path}")

        # 5. Detecção de Anomalias com IA
        print("  • [PASSO 5] Executando Detecção de Anomalias com Autoencoder 1D...")
        anomaly_scores = detect_anomalies_autoencoder(strain_filtered)
        max_anomaly_t = time[np.argmax(anomaly_scores)]
        max_anomaly_val = np.max(anomaly_scores)
        print(f"    [IA] Maior Desvio de Anomalia pela IA: Erro = {max_anomaly_val:.2f} Sigmas em t = +{max_anomaly_t:.4f}s")

    print("\n==========================================================================")
    print(f" EVENTO REAL '{event_name}' PROCESSADO COM SUCESSO!")
    print(f" Gráficos de saída salvos em: {output_dir}")
    print("==========================================================================")


def process_synthetic(seed: int = 2026):
    output_dir = os.path.join(RESULTS_BASE_DIR, "sintetico")
    os.makedirs(output_dir, exist_ok=True)

    print("==========================================================================")
    print("      EXECUTANDO DEMONSTRAÇÃO COM DADOS SINTÉTICOS (SIMULAÇÃO)")
    print("==========================================================================")

    fs = 4096.0
    duration = 4.0
    t_merger = 3.0
    target_snr = 14.0
    rng = np.random.default_rng(seed)

    print(f"\n[PASSO 1] Gerando Ruído e Injetando Chirp (SNR ótima = {target_snr})...")
    time, pure_signal = generate_synthetic_chirp(fs=fs, duration=duration, t_merger=t_merger)
    noise = generate_detector_noise(fs=fs, duration=duration, rng=rng)
    strain_raw = inject_signal(noise, pure_signal, target_snr=target_snr)

    strain_filtered = bandpass_filter(strain_raw, fs=fs, lowcut=LOWCUT, highcut=HIGHCUT)

    raw_path = os.path.join(output_dir, "01_strain_bruto_vs_filtrado.png")
    plot_strain_comparison(time, strain_raw, strain_filtered, save_path=raw_path)

    spec_path = os.path.join(output_dir, "02_espectrograma.png")
    plot_spectrogram(strain_filtered, fs=fs, save_path=spec_path)

    snr_series = matched_filter_at_merger(strain_filtered, pure_signal, fs, t_merger)
    peak_idx = int(np.argmax(snr_series))
    print(f"  [+] PICO DE SNR = {snr_series[peak_idx]:.2f} para fusão em t = {time[peak_idx]:.4f} s "
          f"(injeção em t = {t_merger:.4f} s)")
    print("      A SNR recuperada fica um pouco abaixo da injetada: parte do chirp está fora de 30-400 Hz.")

    snr_path = os.path.join(output_dir, "03_serie_temporal_snr.png")
    plot_snr_time_series(time, snr_series, threshold=8.0, t_event=t_merger, save_path=snr_path,
                         xlabel=MERGER_AXIS_LABEL)

    anomaly_scores = detect_anomalies_autoencoder(strain_filtered)
    max_anomaly_t = time[np.argmax(anomaly_scores)]
    max_anomaly_val = np.max(anomaly_scores)
    print(f"  [IA] Pico de Anomalia pela IA: Erro = {max_anomaly_val:.2f} Sigmas em t = {max_anomaly_t:.4f} s")
    print("       Com SNR realista, o sinal fica abaixo do ruído amostra a amostra; o filtro casado")
    print("       o encontra porque integra o template inteiro, e o autoencoder não tem essa informação.")

    print("\n==========================================================================")
    print(" DEMONSTRAÇÃO SINTÉTICA CONCLUÍDA COM SUCESSO!")
    print(f" Gráficos salvos em: {output_dir}")
    print("==========================================================================")


def main():
    parser = argparse.ArgumentParser(description="Kit Didático de Análise de Ondas Gravitacionais (LIGO/GWTC)")
    parser.add_argument("--event", type=str, help="Caminho ou nome da pasta do evento (ex: GW150914-v4)")
    parser.add_argument("--chirp-mass", type=float, default=30.0,
                        help="Massa chirp no referencial do detector, em massas solares (padrão: 30)")
    parser.add_argument("--synthetic", action="store_true", help="Força o modo de dados sintéticos")
    parser.add_argument("--seed", type=int, default=2026, help="Semente do ruído sintético")
    args = parser.parse_args()

    if args.synthetic:
        process_synthetic(seed=args.seed)
        return

    # Se uma pasta de evento foi fornecida no CLI
    if args.event:
        target_path = args.event
        if not os.path.exists(target_path):
            target_path = os.path.join(BASE_DIR, args.event)

        if os.path.exists(target_path):
            process_real_event(target_path, chirp_mass=args.chirp_mass)
        else:
            print(f"⚠️ Erro: Pasta do evento '{args.event}' não foi encontrada.")
            sys.exit(1)
        return

    # Auto-busca de eventos reais no diretório base
    event_folders = sorted(glob.glob(os.path.join(BASE_DIR, "GW*")))
    if event_folders:
        print(f"🔍 Auto-detectadas {len(event_folders)} pastas de eventos no diretório local!")
        first_event = event_folders[0]
        print(f"▶️ Executando pipeline no primeiro evento encontrado: '{os.path.basename(first_event)}'...\n")
        process_real_event(first_event, chirp_mass=args.chirp_mass)
    else:
        print("ℹ️ Nenhuma pasta de evento 'GW*' encontrada localmente. Executando modo Sintético...")
        process_synthetic(seed=args.seed)


if __name__ == "__main__":
    main()
