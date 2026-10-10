# 🌌 Kit Educacional: Análise de Ondas Gravitacionais com Python

Este kit foi desenvolvido para permitir que estudantes de física, astronomia, ciência da computação e engenharias experimentem na prática as principais técnicas modernas usadas na detecção de **Ondas Gravitacionais** (interferômetros LIGO, Virgo e KAGRA).

É material didático independente do pacote `gwproject` e não produz resultados científicos do projeto. Licença MIT (ver `LICENSE` na raiz do repositório). Em 10/10/2026 o filtro casado e a demonstração foram corrigidos; as correções estão listadas no fim deste guia.

---

## 📚 Conceitos Abordados no Kit

1. **Processamento Digital de Sinais (DSP):**
   - Filtragem passa-banda IIR (**Butterworth**) entre 30 Hz e 400 Hz. Abaixo de ~30 Hz domina o ruído sísmico; em altas frequências, o ruído quântico (*shot noise*).
   - Cálculo da Densidade Espectral de Potência (**PSD**) pelo método de Welch.
2. **Análise Tempo-Frequência:**
   - Espectrograma por Transformada de Fourier de Tempo Curto (**STFT**) para visualizar o *chirp* (aumento simultâneo de frequência e amplitude até a fusão). O `gwproject` usa a transformada Q; o kit usa a STFT por simplicidade.
3. **Filtro Casado (Matched Filtering):**
   - Correlação, no domínio da frequência, entre os dados e um template, ponderada pela PSD. O resultado é a série temporal da Razão Sinal-Ruído (**SNR**). Em ruído gaussiano, ρ² tem média 2.
   - Template físico de ordem dominante (*chirp* newtoniano) a partir da massa chirp, e um *chirp* de brinquedo para a simulação.
4. **Machine Learning e Detecção de Anomalias:**
   - Autoencoder convolucional 1D treinado no início da série. Ele só acusa sinais muito mais fortes que o ruído amostra a amostra; com SNR realista, o filtro casado é muito mais sensível, porque integra o template inteiro. Comparar os dois é um bom exercício.

---

## 🛠️ Requisitos de Instalação

No ambiente do `gwproject`, nada mais é necessário. Fora dele:

```bash
pip install numpy scipy matplotlib h5py
```

*(Opcional - para recursos de Aprendizado de Máquina/PyTorch)*:
```bash
pip install torch scikit-learn
```

---

## 🚀 Como Executar

Execute a partir da raiz do repositório.

### 1. Eventos reais do catálogo (dados baixados pelo pipeline)

Os arquivos brutos baixados por `scripts/run_catalog.py` ficam em `outputs/gwtc/raw/<evento>/` (`H1.hdf5`, `L1.hdf5`, ...). Informe a massa chirp **no referencial do detector**, coluna `chirp_mass_detector` de `outputs/gwtc/catalog_dataset.csv`:

```bash
python gw_student_kit/main_demo.py --event outputs/gwtc/raw/GW150914-v4 --chirp-mass 30.69
```

Resultado de referência (10/10/2026): H1 com ρ ≈ 15,0 e L1 com ρ ≈ 10,8 perto de t = 16,02 s, com L1 ~7 ms antes de H1. As análises oficiais, com formas de onda completas, obtêm valores maiores. Em eventos fracos o template de ordem dominante pode não se destacar do ruído: no GW151012 (massa chirp 18,72 M☉ no detector), L1 dá ρ ≈ 5,3 perto de t = 16,05 s, mas o pico global de H1 (ρ ≈ 5,0) cai em t ≈ 1,8 s, longe do evento.

O leitor aceita tanto arquivos gravados pelo gwpy quanto os HDF5 oficiais do GWOSC.

### 2. Modo Sintético (Sem dados reais)

Simulação autossuficiente com ruído branco e injeção de um *chirp* com SNR ótima 14:

```bash
python gw_student_kit/main_demo.py --synthetic --seed 2026
```

A SNR recuperada fica perto de 13 no instante da injeção (t = 3 s). Parte do *chirp* fica fora da banda de 30 a 400 Hz.

As figuras vão para `gw_student_kit/results/<evento>/` e `gw_student_kit/results/sintetico/`. As versionadas neste repositório foram geradas com o código corrigido, em 10/10/2026.

---

## 📂 Estrutura dos Módulos

```text
gw_student_kit/
├── __init__.py         # Exportação das funções principais
├── synthetic.py        # Ruído, chirp de brinquedo, chirp newtoniano e injeção por SNR
├── dsp.py              # Filtro Butterworth, PSD de Welch e Leitor HDF5
├── matched_filter.py   # SNR do filtro casado e alinhamento pelo instante de fusão
├── visualization.py    # Plots do strain, espectrogramas e séries temporais de SNR
├── ml_anomaly.py       # Autoencoders 1D e detecção de anomalias com IA
├── main_demo.py        # Script completo para eventos reais e sintéticos
├── results/            # Figuras geradas pela demonstração
└── README.md           # Este guia didático
```

---

## 🔬 Como Usar os Módulos no Seu Próprio Script Python

```python
import numpy as np
from gw_student_kit import (
    load_event_folder,
    bandpass_filter,
    compute_psd,
    generate_newtonian_chirp,
    compute_matched_filter,
    align_to_merger,
)

# 1. Carrega dados de um evento real baixado por scripts/run_catalog.py
data = load_event_folder("outputs/gwtc/raw/GW150914-v4")
strain, fs = data["H1"]["strain"], data["H1"]["fs"]
duration = len(strain) / fs

# 2. Filtra dados e template com o mesmo passa-banda
filtered = bandpass_filter(strain, fs=fs, lowcut=30.0, highcut=400.0)
_, template = generate_newtonian_chirp(fs=fs, duration=duration, chirp_mass_solar=30.69,
                                       t_merger=duration / 2)
template = bandpass_filter(template, fs=fs, lowcut=30.0, highcut=400.0)

# 3. Filtro casado na mesma banda e SNR indexada pelo instante de fusão
_, psd = compute_psd(filtered, fs=fs)
snr, _, _ = compute_matched_filter(filtered, template, fs=fs, psd=psd, f_low=30.0, f_high=400.0)
snr = align_to_merger(snr, int(duration / 2 * fs))
```

Em dados reais, aplique uma janela (por exemplo, Tukey) antes do filtro casado, como faz `main_demo.py`. Sem ela, os transientes do filtro nas bordas dominam a correlação circular.

---

## 🩹 Correções de 10/10/2026

A versão anterior produzia SNR de 1,6 × 10⁹ na simulação e de 3,9 × 10¹⁰ no GW150914, com o pico fora do instante da injeção. As causas e correções:

1. **Normalização do filtro casado.** Faltavam os fatores Δt da transformada de Fourier discreta, o que inflava a SNR por um fator ~f_s. Agora ⟨ρ²⟩ = 2 em ruído gaussiano.
2. **Banda de análise.** A PSD dos dados filtrados é quase nula fora de 30 a 400 Hz, e o template não era filtrado. A divisão amplificava o template fora da banda. Agora o template recebe o mesmo filtro e a soma se restringe à banda.
3. **Instante de fusão.** O índice do pico da correlação é um atraso, não o instante de fusão. `align_to_merger` faz a conversão.
4. **Definição da SNR injetada.** `inject_signal` usava a razão entre RMS de sinal e ruído, que com 16 mil amostras equivale a ρ ≈ 1.800 para "SNR = 14". Agora usa a SNR ótima em ruído branco, ρ = √(Σh²)/σ.
5. **Template para dados reais.** O *chirp* de brinquedo não descreve uma binária. Os eventos reais usam `generate_newtonian_chirp` com a massa chirp do catálogo.
6. **Bordas.** Janela de Tukey e PSD estimada na região central, para que os transientes do filtro não criem picos espúrios.
7. **Reprodutibilidade.** O ruído sintético aceita semente (`--seed`).

As figuras em `results/` foram regeneradas com o código corrigido. As anteriores mostravam esses valores inflados.

---

## ⚠️ Limites

- O template newtoniano não inclui correções pós-newtonianas, spins, fusão nem *ringdown*. A SNR recuperada fica abaixo da obtida com formas de onda completas.
- A PSD é estimada nos mesmos dados que contêm o sinal, e não há banco de templates, teste χ², *time slides* nem taxa de falso alarme. A SNR do kit é indicativa e não estabelece detecção.
- O ruído sintético é branco e estacionário; o ruído real é colorido e contém *glitches*.

Bons estudos e ótimas pesquisas! 🚀
