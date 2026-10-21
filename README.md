# GWProject

Pipeline auditável para aquisição e condicionamento de strain dos 218 eventos
do GWTC-4, análise catalogal, diagnósticos tempo–frequência e estudo de validade
externa de classificadores Gravity Spy.

Repositório: <https://github.com/jean5429/gw-pic-2026>

O repositório possui uma única interface de linha de comando e uma única identidade de pacote. Resultados científicos são separados de artefatos de execução, e toda análise relevante deve registrar configuração, hashes e manifesto de ambiente.

## Escopo científico

O projeto oferece quatro fluxos:

1. aquisição e controle de qualidade de dados públicos do GWOSC;
2. análise catalogal por canal e por evento;
3. treinamento e inferência de classificadores de morfologia;
4. experimentos de robustez, ensemble e reconhecimento open-world.

As rotinas de ridge e matched filtering são diagnósticos locais. Elas não implementam uma busca cega de detecção e não produzem false-alarm rate, significância de descoberta ou identificação causal de glitches.

## Instalação

### Núcleo

```powershell
git clone https://github.com/jean5429/gw-pic-2026.git
cd gw-pic-2026
conda env create -f environment-core.yml
conda activate gwproject
python -m pip install -e .
```

### Ambiente completo

```powershell
conda env create -f environment.yml
conda activate gwproject
python -m pip install -e ".[all,test]"
```

Para GPUs NVIDIA no Windows, use `tools/install_gpu_windows.ps1` após ativar o ambiente. O script usa o índice oficial do PyTorch e aceita `-CudaIndex`, `-TorchVersion`, `-TorchvisionVersion` e `-TorchaudioVersion`; não depende de URLs de wheels ligadas a uma versão específica do Python.

## Interface principal

```powershell
gwproject --help
gwproject catalog --help
gwproject research --help
```

### Catálogo

```powershell
gwproject acquire -- --download-strain --process --qgrams
gwproject catalog index
gwproject catalog refresh-metadata -- --workers 4
gwproject catalog ridge -- --workers 2 --no-resume
gwproject catalog matched-filter -- --workers 1 --no-resume
gwproject catalog diagnose-ridge
```

### Aquisição e processamento

```powershell
gwproject acquire --help
gwproject process --help
gwproject validate-event --help
gwproject matched-filter-event --help
```

### Gravity Spy

```powershell
gwproject train-gravity-spy `
  .\data\gravity_spy_harmonized\clean_metadata.csv `
  .\data\gravity_spy_harmonized `
  --output-dir .\models\gravity_spy_harmonized_v1d1
```

### Programa open-world

```powershell
gwproject research prepare-panels `
  .\data\gravity_spy\trainingset `
  .\data\gravity_spy_harmonized `
  --metadata .\data\gravity_spy\trainingset_v1d1_metadata.csv

gwproject research train-seeds `
  .\data\gravity_spy_harmonized\clean_metadata.csv `
  .\data\gravity_spy_harmonized `
  --output-dir .\models\gravity_spy_harmonized_v1d1

gwproject research build-ood `
  .\models\gravity_spy_harmonized_v1d1 `
  --metadata .\data\gravity_spy_harmonized\clean_metadata.csv `
  --images-root .\data\gravity_spy_harmonized `
  --output .\outputs\research\gwtc4_218\ood_reference.npz

gwproject research validate-ood `
  .\outputs\research\gwtc4_218\ood_reference.npz `
  --metadata .\data\gravity_spy_harmonized\clean_metadata.csv `
  --images-root .\data\gravity_spy_harmonized `
  --output .\outputs\research\gwtc4_218\ood_test_control.json

gwproject research classify-ensemble `
  .\models\gravity_spy_harmonized_v1d1 `
  --approved .\outputs\gwtc4_218\approved_channels.csv `
  --ood-reference .\outputs\research\gwtc4_218\ood_reference.npz `
  --output-dir .\outputs\research\gwtc4_218\gravity_spy_ensemble
```

A referência OOD é construída separadamente para cada checkpoint: a geometria
Mahalanobis é ajustada no treino, os percentis são calibrados na validação e o
teste permanece controle independente. Os modelos são associados às respectivas
referências por SHA-256; embeddings, covariâncias e escalas brutas de energia
não são transferidos entre redes. A decisão agregada usa percentis empíricos por
modelo e uma fração mínima configurável de concordância. Cada canal registra
também `mahalanobis_log_excess_99`, escore contínuo que não satura como o
percentil.

### Renderização dos Q-grams

`[harmonized_inference].render_mode` escolhe a escala de intensidade da entrada
da CNN. `"percentile"` é o renderer canônico do preregistro. `"gravity_spy"` é
exploratório: reproduz a escala dos PNGs de treino (energia 0–25,5 em viridis,
convertida para cinza). Para testá-lo sem tocar na configuração nem na saída
canônica, use `--render-mode` com uma pasta própria e compare as execuções
pareando por evento:

```powershell
gwproject research classify-ensemble .\models\gravity_spy_harmonized_v1d1 `
  --ood-reference .\outputs\research\gwtc4_218\ood_reference.npz `
  --render-mode gravity_spy --output-dir .\outputs\research\ensemble_gravity_spy

gwproject research compare-ensembles `
  .\outputs\research\gwtc4_218\gravity_spy_ensemble `
  .\outputs\research\ensemble_gravity_spy `
  --output-dir .\outputs\research\comparacao_renderers
```

`compare-ensembles` exige os mesmos canais, checkpoints e referência OOD. O
resultado de 10/10/2026 e seus limites estão em
`research/RENDERIZADOR_GRAVITY_SPY.md`.

### Inspeção visual

```powershell
gwproject research render-qgram --event GW150914-v4 --render-modes percentile gravity_spy
gwproject research gradcam --event GW150914-v4 --detector L1 --render-mode gravity_spy
gwproject research gradcam --image .\data\gravity_spy_harmonized\TrainingSet\Chirp\H1_BTGpb9EV7L_spectrogram_1.0.png
gwproject research replot-confusion .\models\gravity_spy_harmonized_v1d1 --output-dir .\outputs\research\confusion_300dpi
```

`render-qgram` grava a entrada exata da rede (140 × 170). `gradcam` usa por
padrão o checkpoint de semente 42 e grava a figura em 300 DPI com um JSON de
proveniência. O mapa explica a decisão fechada de um único checkpoint e não
valida a classe. `replot-confusion` regenera as matrizes de confusão a partir
dos `.npy`; sem `--output-dir`, substitui os PNGs ao lado de cada `.npy`.

## Dados Gravity Spy

O fluxo aceita os PNGs oficiais de 1 s referenciados pelos metadados. Por
padrão, `prepare-panels` aplica o recorte documentado do conjunto: linhas
`66:532` e colunas `105:671`, seguido de resize para `140×170`. O comando cria
`qc_samples.csv`, elimina imagens vazias e grupos de duplicatas exatas que
atravessam splits, preserva os metadados originais e grava
`clean_metadata.csv`. A chave de junção é `(ifo, gravityspy_id)`, evitando
colisão entre detectores.

O treinamento canônico recusa imagens sem esses manifestos de QC. O parâmetro
`--allow-unharmonized` existe apenas para reprodução explícita do legado.

A representação numérica oficial em HDF5 deve ser mantida como referência primária para controle de domínio. Não se deve confundir desempenho interno no conjunto Gravity Spy com validade externa sobre Q-transforms produzidos por outro renderer.

## Estrutura

```text
configs/                 configurações TOML
research/                preregistro, política de alegações e planos experimentais
src/gwproject/           biblioteca científica
src/gwproject/commands/  implementações dos comandos
 tests/                   testes unitários e de contratos
tools/                   automação PowerShell
outputs/                 resultados locais, não versionados
models/                  checkpoints canônicos e legado preservado
gw_student_kit/          kit didático independente (filtro casado, DSP)
```

Veja [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) para a fronteira completa
entre resultados canônicos e históricos.

## Linha anterior (v9) e relatórios PIC

Até 17/10/2026 a branch `main` continha a linha v9 (90 eventos GWTC-3), com
modelos treinados versionados via Git LFS e as saídas usadas no Relatório V2.
Essa linha está preservada na branch [`v9`](https://github.com/jean5429/gw-pic-2026/tree/v9).
Os relatórios do PIC ficam em [`docs/relatorios/`](docs/relatorios/):

- [Relatório V3](docs/relatorios/Relatorio_PIC_Ondas_Gravitacionais_V3.pdf)
  (mais recente): reprodução independente e completa da linha 2.0. Base de
  treino e treino das 5 seeds idênticos aos canônicos (mesmo SHA-256 do
  `aggregate_summary.json`); catálogo de 218 eventos, ensemble (475/475 canais
  fora do domínio) e experimento de renderização reproduzidos, com diferença
  de no máximo 1 canal;
- [Relatório V2](docs/relatorios/Relatorio_PIC_Ondas_Gravitacionais_V2.pdf):
  auditoria da linha v9 (orientação dos Q-grams, cores do Grad-CAM, escala de
  intensidade) e correção do kit didático;
- [Errata do Relatório V2](docs/relatorios/Errata_Relatorio_V2.pdf):
  painel vazio repetido nos dados Gravity Spy, limiar OOD calibrado nas mesmas
  amostras do ajuste e mistura de modelos no escore OOD; com o cálculo
  corrigido, 60 dos 150 canais (e não 150) excedem o limiar OOD;
- relatório V1 (`Relatorio_PIC_Ondas_Gravitacionais.doc` e `.doc.pdf`) e a
  verificação do cálculo de filtragem.

Os resultados do V2 se referem aos modelos e à referência OOD da linha v9 e
devem ser lidos junto com a errata. Ver também `VALIDATION_REPORT.md` e
`MIGRACAO_GW_PIC_2026.md`.

## Kit didático

`gw_student_kit/` é material de ensino independente do pacote: filtro
passa-banda, PSD, espectrograma, filtro casado com chirp newtoniano e um
autoencoder simples. Veio do repositório jean5429/gw-pic-2026 e foi corrigido na
incorporação (normalização da SNR, banda e instante de fusão). O espectrograma
mostra a potência relativa à mediana de cada frequência, como a transformada Q.
Não produz resultados do projeto.

```powershell
python gw_student_kit\main_demo.py --synthetic
python gw_student_kit\main_demo.py --event outputs\gwtc4_218\raw\GW150914-v4 --chirp-mass 30.69
```

A origem e as diferenças em relação ao repositório de origem estão em
[MIGRACAO_GW_PIC_2026.md](MIGRACAO_GW_PIC_2026.md).

## Testes

```powershell
pytest
pytest --cov=gwproject --cov-report=term-missing
python -m compileall -q src
```

Testes unitários não substituem validação com dados reais, campanhas de injeção, comparação com bibliotecas estabelecidas e auditoria independente dos artefatos produzidos.

## Reprodutibilidade

Cada execução científica deve preservar:

- configuração integral;
- hashes dos dados e checkpoints;
- commit do código;
- versões das dependências;
- seed e perfil de hardware;
- arquivos intermediários suficientes para reconstruir tabelas e figuras.

Os manifestos (`manifest.json`, `*.manifest.json`) trocam a pasta do usuário por
`~` antes de gravar. `run_log.json` mantém os caminhos reais, porque é insumo do
índice do catálogo; revise-o antes de publicar.

Consulte `research/PLANO_DE_DADOS_E_REPRODUTIBILIDADE.md` e `research/POLITICA_DE_ALEGACOES.md`.

A reconstrução estrutural, as correções metodológicas e os limites da validação
estão documentados em `VALIDATION_REPORT.md`. O veredito científico, as
explicações concorrentes e o roteiro priorizado estão em
`research/PARECER_CIENTIFICO_E_ROTEIRO.md`.

O resultado consolidado por evento é
`outputs/research/gwtc4_218/all_218_events.csv`: 218 linhas com QC, CNN/OOD,
ridge e matched filter, unidos sob validação um-para-um.
