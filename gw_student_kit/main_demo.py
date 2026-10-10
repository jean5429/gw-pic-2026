"""
Script Principal de Demonstração (GW Student Kit)
=================================================
Este script executa a pipeline completa para análise de Ondas Gravitacionais:
1. Leitura de dados de detectores (.hdf5) de eventos reais do LIGO (ex: GW150914-v4)
   OU geração de sinal sintético se nenhum evento local for especificado.
2. Limpeza por Processamento Digital de Sinais (Filtro Passa-Banda Butterworth 30-400Hz).
3. Geração de Espectrogramas (Transformada Q / STFT).
4. Filtro Casado (Matched Filtering) com cálculo da série temporal de SNR.
5. Detecção de Anomalias com Inteligência Artificial (Autoencoders 1D / ML).

Para rodar em modo automático (detecta pastas reais locais como GW150914-v4):
   python main_demo.py

Para rodar em um evento específico:
   python main_demo.py --event GW150914-v4

Para forçar modo de dados sintéticos:
   python main_demo.py --synthetic
"""

import os
import sys
import glob
import argparse
import numpy as np

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gw_student_kit.synthetic import (
    generate_synthetic_chirp, 
    generate_detector_noise, 
    inject_signal
)
from gw_student_kit.dsp import (
    bandpass_filter, 
    compute_psd,
    load_event_folder
)
from gw_student_kit.matched_filter import (
    compute_matched_filter, 
    find_snr_peaks
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


def process_real_event(event_folder_path: str):
    event_name = os.path.basename(os.path.normpath(event_folder_path))
    output_dir = os.path.join(RESULTS_BASE_DIR, event_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("==========================================================================")
    print(f"      PROCESSANDO EVENTO REAL DO LWT/LIGO: {event_name}")
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
        time = np.linspace(0, duration, n_samples, endpoint=False)
        
        print(f"\n--- Detector {det_name} ---")
        print(f"  • N° de Amostras: {n_samples:,} | f_s = {fs:.0f} Hz | Duração = {duration:.2f} s")
        print(f"  • Tempo GPS Inicial (x0): {gps_start:.4f} s")
        
        # 2. Filtragem Passa-Banda (30 - 400 Hz)
        print("  • [PASSO 2] Aplicando Filtro Passa-Banda (30Hz - 400Hz)...")
        strain_filtered = bandpass_filter(strain_raw, fs=fs, lowcut=30.0, highcut=400.0)
        freqs, psd = compute_psd(strain_filtered, fs=fs)
        
        zoom_lims = (15.5, 16.5) # Zoom de 1 segundo ao redor do momento da colisão
        
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
        
        # 4. Filtro Casado
        print("  • [PASSO 4] Executando Filtro Casado com Template Chirp...")
        # Gera template teórico com massa representativa para buscar o pico de SNR
        _, template = generate_synthetic_chirp(fs=fs, duration=duration, t_merger=duration/2.0)
        snr_series, max_snr, peak_idx = compute_matched_filter(
            data=strain_filtered, 
            template=template, 
            fs=fs, 
            psd=psd
        )
        t_peak = time[peak_idx]
        gps_peak = gps_start + t_peak
        print(f"    [+] PICO MÁXIMO DE SNR DETECTADO: SNR = {max_snr:.2f} em t = +{t_peak:.4f}s (GPS {gps_peak:.4f}s)")
        
        snr_path = os.path.join(output_dir, f"{det_name}_03_snr_series.png")
        plot_snr_time_series(
            time, snr_series, threshold=8.0, 
            save_path=snr_path, 
            title=f"Série Temporal de SNR - {det_name} ({event_name})",
            xlim=zoom_lims
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


def process_synthetic():
    output_dir = os.path.join(RESULTS_BASE_DIR, "sintetico")
    os.makedirs(output_dir, exist_ok=True)
    
    print("==========================================================================")
    print("      EXECUTANDO DEMONSTRAÇÃO COM DADOS SINTÉTICOS (SIMULAÇÃO)")
    print("==========================================================================")
    
    fs = 4096.0
    duration = 4.0
    t_merger = 3.0
    target_snr = 14.0
    
    print(f"\n[PASSO 1] Gerando Ruído e Injetando Chirp (SNR Target = {target_snr})...")
    time, pure_signal = generate_synthetic_chirp(fs=fs, duration=duration, t_merger=t_merger)
    noise = generate_detector_noise(fs=fs, duration=duration)
    strain_raw = inject_signal(noise, pure_signal, target_snr=target_snr)
    
    strain_filtered = bandpass_filter(strain_raw, fs=fs, lowcut=30.0, highcut=400.0)
    freqs, psd = compute_psd(strain_filtered, fs=fs)
    
    raw_path = os.path.join(output_dir, "01_strain_bruto_vs_filtrado.png")
    plot_strain_comparison(time, strain_raw, strain_filtered, save_path=raw_path)
    
    spec_path = os.path.join(output_dir, "02_espectrograma.png")
    plot_spectrogram(strain_filtered, fs=fs, save_path=spec_path)
    
    snr_series, max_snr, peak_idx = compute_matched_filter(strain_filtered, pure_signal, fs=fs, psd=psd)
    t_peak = time[peak_idx]
    print(f"  [+] PICO MÁXIMO DE SNR DETECTADO: SNR = {max_snr:.2f} no instante t = {t_peak:.4f} s")
    
    snr_path = os.path.join(output_dir, "03_serie_temporal_snr.png")
    plot_snr_time_series(time, snr_series, threshold=8.0, t_event=t_merger, save_path=snr_path)
    
    anomaly_scores = detect_anomalies_autoencoder(strain_filtered)
    max_anomaly_t = time[np.argmax(anomaly_scores)]
    max_anomaly_val = np.max(anomaly_scores)
    print(f"  [IA] Pico de Anomalia pela IA: Erro = {max_anomaly_val:.2f} Sigmas em t = {max_anomaly_t:.4f} s")

    print("\n==========================================================================")
    print(" DEMONSTRAÇÃO SINTÉTICA CONCLUÍDA COM SUCESSO!")
    print(f" Gráficos salvos em: {output_dir}")
    print("==========================================================================")


def main():
    parser = argparse.ArgumentParser(description="Kit Didático de Análise de Ondas Gravitacionais (LIGO/GWTC)")
    parser.add_argument("--event", type=str, help="Caminho ou nome da pasta do evento (ex: GW150914-v4)")
    parser.add_argument("--synthetic", action="store_true", help="Força o modo de dados sintéticos")
    args = parser.parse_args()

    if args.synthetic:
        process_synthetic()
        return

    # Se uma pasta de evento foi fornecida no CLI
    if args.event:
        target_path = args.event
        if not os.path.exists(target_path):
            target_path = os.path.join(BASE_DIR, args.event)
            
        if os.path.exists(target_path):
            process_real_event(target_path)
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
        process_real_event(first_event)
    else:
        print("ℹ️ Nenhuma pasta de evento 'GW*' encontrada localmente. Executando modo Sintético...")
        process_synthetic()


if __name__ == "__main__":
    main()
