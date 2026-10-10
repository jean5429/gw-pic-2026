# 🌌 Kit Educacional: Análise de Ondas Gravitacionais com Python

Este kit foi desenvolvido para permitir que estudantes de física, astronomia, ciência da computação e engenharias experimentem na prática as principais técnicas modernas usadas na detecção de **Ondas Gravitacionais** (interferômetros LIGO, Virgo e KAGRA).

---

## 📚 Conceitos Abordados no Kit

1. **Processamento Digital de Sinais (DSP):**
   - Filtragem passa-banda IIR (**Butterworth**) entre 30 Hz e 400 Hz para remoção de ruído sísmico e ruído de disparo quântico (*shot noise*).
   - Cálculo da Densidade Espectral de Potência (**PSD**) pelo método de Welch.
2. **Análise Tempo-Frequência:**
   - Transformada Fourier de Tempo Curto (**STFT**) / Transformada Q para visualização do padrão tipo *"Chirp"* (aumento simultâneo de frequência e amplitude no momento da fusão dos corpos massivos).
3. **Filtro Casado (Matched Filtering):**
   - Convolução de formas de onda teóricas no domínio da frequência para extrair a Razão Sinal-Ruído (**SNR**).
4. **Machine Learning e Detecção de Anomalias:**
   - Redes neurais Convolucionais 1D (**Autoencoders**) treinadas no ruído de fundo que disparam alarmes de anomalia (> 3-Sigmas) no instante da colisão cósmica sem o uso de equações teóricas prévias.

---

## 🛠️ Requisitos de Instalação

O kit é extremamente leve e pode ser executado com bibliotecas padrão de análise de dados em Python:

```bash
pip install numpy scipy matplotlib h5py
```

*(Opcional - para recursos de Aprendizado de Máquina/PyTorch)*:
```bash
pip install torch scikit-learn
```

---

## 🚀 Como Executar

### 1. Testando nos Dados Reais dos Eventos (LIGO/GWTC)
Se os alunos já possuem as pastas de eventos baixadas no computador (ex: `GW150914-v4`, `GW170817-v3`, etc.):

O script detecta automaticamente as pastas de eventos reais e processa os dados brutos de strain dos detectores (`H1.hdf5` e `L1.hdf5`):

```bash
# Detecta automaticamente a primeira pasta de evento encontrada no diretório local:
python gw_student_kit/main_demo.py

# Ou especifica uma pasta de evento real desejada:
python gw_student_kit/main_demo.py --event GW150914-v4
```

### 2. Modo Sintético (Sem dados reais)
Caso queiram testar uma simulação autossuficiente com ruído numérico e injeção de chirp:

```bash
python gw_student_kit/main_demo.py --synthetic
```

---

## 📂 Estrutura dos Módulos

```text
gw_student_kit/
├── __init__.py         # Exportação das funções principais
├── synthetic.py        # Geração de ruído e injeção de ondas Chirp sintéticas
├── dsp.py              # Filtro Butterworth, PSD de Welch e Leitor HDF5
├── matched_filter.py   # Convolução de SNR e busca por picos
├── visualization.py    # Plots do strain, espectrogramas e séries temporais de SNR
├── ml_anomaly.py       # Autoencoders 1D e detecção de anomalias com IA
├── main_demo.py        # Script completo para eventos reais e sintéticos
└── README.md           # Este guia didático
```

---

## 🔬 Como Usar os Módulos no Seu Próprio Script Python

Você ou seus alunos podem importar o kit como uma biblioteca Python regular em seus próprios scripts ou Jupyter Notebooks:

```python
from gw_student_kit import (
    load_event_folder,
    bandpass_filter,
    compute_psd,
    compute_matched_filter,
    plot_spectrogram
)

# 1. Carrega dados de um evento real baixado (ex: GW150914-v4)
data_dict = load_event_folder("GW150914-v4")
h1_strain = data_dict['H1']['strain']
fs = data_dict['H1']['fs']

# 2. Filtra o sinal
filtered_strain = bandpass_filter(h1_strain, fs=fs, lowcut=30.0, highcut=400.0)

# 3. Plota o espectrograma
plot_spectrogram(filtered_strain, fs=fs, save_path="meu_espectrograma.png")
```

Bons estudos e ótimas pesquisas! 🚀
