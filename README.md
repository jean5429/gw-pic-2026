# Classificação de Glitches do LIGO com CNNs e Grad-CAM

Projeto 3 – Ondas Gravitacionais · **PIC Astronomia**

Repositório: <https://github.com/jean5429/gw-pic-2026>

Pipeline auditável que:

1. baixa e processa dados públicos do catálogo de ondas gravitacionais **GWTC** (GWOSC);
2. gera Q-grams (tempo × frequência) de cada evento e detector;
3. treina uma **ResNet-18** no conjunto **Gravity Spy** (22 classes de glitches);
4. aplica um ensemble de 5 modelos aos eventos reais, com detecção fora de distribuição (OOD);
5. usa **Grad-CAM** para ver onde a rede está olhando.

O relatório mais recente é o [Relatório V2 (PDF)](docs/relatorios/Relatorio_PIC_Ondas_Gravitacionais_V2.pdf). Ele registra a auditoria do pipeline e corrige conclusões do relatório V1.

---

## Estrutura

```text
src/gwproject/          pacote Python (pré-processamento, Q-grams, modelo, filtro casado, OOD)
scripts/                linhas de comando (download, treino, ensemble, Grad-CAM, ...)
configs/                parâmetros em TOML (processamento, Gravity Spy, ensemble)
tests/                  testes automatizados (pytest)
outputs/                resultados versionados (métricas, CSVs, figuras, modelos via LFS)
docs/relatorios/        relatórios PIC (V1, V2, verificação)
docs/historico/         notas de versão e migrações (v6 → v9)
docs/figuras/           figuras Grad-CAM das classes de treino
research/               programa científico, preregistro, protocolos
gw_student_kit/         kit didático independente (filtro casado e DSP)
ondasgravitacionais_*.py  pontos de entrada das versões v8/v9 e scripts camada1 v14/v15
```

---

## Instalação

Requer **Python 3.11** e, para treino ou inferência rápida, GPU NVIDIA com CUDA 12.1.

```powershell
git clone https://github.com/jean5429/gw-pic-2026.git
cd gw-pic-2026
conda env create -f environment-core.yml
conda activate gw-v8
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[gw,ml,test]"
```

PyTorch com CUDA (Windows):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_gpu_windows.ps1
```

Sem GPU, instale a versão CPU (`pip install torch==2.5.1 torchvision==0.20.1`) e use `--device cpu` nos scripts.

---

## Dados (baixados sob demanda)

Nenhum dado bruto fica no repositório. Cada conjunto é baixado apenas quando você precisa dele.

| Conjunto | Tamanho | Necessário para | Como obter |
|---|---|---|---|
| Modelos treinados (5 seeds) | ~215 MB | ensemble, Grad-CAM | Git LFS (abaixo) ou treino próprio |
| Gravity Spy v1.1 (Zenodo 1486046) | ~11 GB | treino, referência OOD, Grad-CAM em treino | `scripts/download_gravity_spy.py` |
| Strain GWTC (GWOSC) + Q-grams | ~540 MB | ensemble em eventos reais | `scripts/run_catalog.py` |

### 1. Modelos treinados (Git LFS)

Os checkpoints `outputs/gravity_spy_harmonized/seed_*/best_model.pt` usam [Git LFS](https://git-lfs.com/).

```powershell
git lfs install
git lfs pull
```

Para clonar sem baixar os modelos (PowerShell): `$env:GIT_LFS_SKIP_SMUDGE = "1"; git clone ...`. Se precisar deles depois, rode `git lfs pull`.

### 2. Gravity Spy

```powershell
python -m scripts.download_gravity_spy --extract
python .\ondasgravitacionais_research_v9.py prepare-panels -- `
    .\data\gravity_spy\trainingset .\data\gravity_spy\trainingset_harmonized_20_512
```

O download confere o checksum oficial do Zenodo e retoma transferências interrompidas.

### 3. Eventos GWTC

```powershell
python -m scripts.run_catalog --download-strain --process --qgrams --workers 4
python -m scripts.check_catalog_ready
```

Use `--max-events 2` para um teste rápido. Os scripts `ondasgravitacionais__camada1_v14/v15.py` mantêm um cache próprio em `camada1_gwtc3/`, que é recriado automaticamente.

---

## Uso

### Ensemble nos eventos reais

```powershell
$M = "outputs\gravity_spy_harmonized"
python -m scripts.classify_catalog_qgrams_ensemble `
    $M\seed_101_cuda_balanced\best_model.pt $M\seed_11_cuda_balanced\best_model.pt `
    $M\seed_23_cuda_balanced\best_model.pt  $M\seed_42_cuda_balanced\best_model.pt `
    $M\seed_71_cuda_balanced\best_model.pt `
    --ood-reference outputs\research\ood_reference_seed42.npz `
    --output-dir outputs\research\gravity_spy_ensemble_gsrender
```

O pré-processamento vem de `configs/research_program.toml` (`[harmonized_inference]`). O padrão é `render_mode = "gravity_spy"`, que reproduz a escala de intensidade das imagens de treino.

### Grad-CAM

```powershell
# imagem de treino
python scripts\run_gradcam.py --image data\gravity_spy\trainingset_harmonized_20_512\TrainingSet\Chirp\H1_0fp5wZUWD7_spectrogram_1.0.png --output-dir docs\figuras

# evento real: converte o Q-gram com o mesmo pré-processamento do ensemble
python scripts\temp_npy_to_png.py --event GW150914-v4 --detectors L1
python scripts\run_gradcam.py --image outputs\gwtc\qgrams\GW150914-v4\L1_GW150914-v4.png --output-dir outputs\research\gradcam_fixed
```

No mapa, **vermelho = atenção máxima**. As figuras são salvas em 300 DPI.

### Treino completo (opcional)

```powershell
& .\scripts\run_research_program.ps1
```

O script harmoniza os painéis, treina 5 seeds, agrega as métricas, constrói a referência OOD e roda o ensemble.

---

## Testes

```powershell
python -m pytest
```

Os testes que dependem de dados baixados (por exemplo, o Q-gram do GW150914) são pulados automaticamente quando os arquivos não existem.

---

## Resultados principais (V2)

| Item | Resultado |
|---|---|
| Gravity Spy, teste (5 seeds) | macro-F1 0,856 ± 0,006 · acurácia 0,909 |
| Grad-CAM nas classes de treino | atenção concentrada no próprio transiente |
| GWTC, renderização Gravity Spy | 15 classes previstas; 11 canais Chirp com confiança ≥ 0,95, todos de eventos intensos (GW150914, GW170814, …) |
| Domínio | os 150 canais ainda ficam acima do limiar OOD q99; resultados em dados reais são exploratórios |

Detalhes, errata do V1 e próximos passos estão no [Relatório V2](docs/relatorios/Relatorio_PIC_Ondas_Gravitacionais_V2.pdf).

---

## Dados de terceiros e citações

- **GWOSC** – LIGO/Virgo/KAGRA Open Science Center, <https://gwosc.org>. Cite os artigos do catálogo GWTC correspondentes.
- **Gravity Spy** – Zevin et al. (2017), *Class. Quantum Grav.* 34, 064003; conjunto em Zenodo, registro 1486046.

Consulte as licenças e as orientações de citação de cada fonte antes de redistribuir os dados.

## Licença

Código sob licença [MIT](LICENSE).
