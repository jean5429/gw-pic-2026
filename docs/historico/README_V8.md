# Ondas gravitacionais — versão 8

A v8 amplia a v7 para operar formalmente sobre o catálogo GWTC inteiro. A aquisição já realizada não precisa ser repetida: os 90 eventos, os 250 canais obtidos e os 230 canais aprovados podem ser reutilizados diretamente.

A versão acrescenta quatro fluxos:

1. construção automática dos índices de qualidade;
2. estimação de chirp mass por ridge em todos os canais aprovados;
3. matched filtering local por evento;
4. aplicação de um checkpoint Gravity Spy aos Q-grams aprovados.

A v8 também corrige a semântica dos canais indisponíveis: falhas de download são registradas como `data_unavailable`, e não como rejeições científicas.

## Estado esperado para a base atual

Depois de reconstruir o índice, a base informada pelo usuário deve resultar em:

```text
90 eventos
270 combinações teóricas evento–detector
250 canais obtidos
230 canais aprovados
20 canais indisponíveis
15 canais rejeitados por NaN/Inf
5 canais rejeitados pelo whitening
88 eventos network_ready
2 eventos single_detector
0 eventos unusable
```

## Atualização no mesmo diretório da v7

O caminho mais simples é extrair o pacote de atualização sobre o diretório atual. A pasta `outputs/` não é incluída no overlay e, portanto, os dados existentes são preservados.

Com `gw-v7` ativado:

```powershell
python -m pip uninstall -y gwproject-v7 gwproject-v8
python -m pip install -e ".[gw,ml,test]"
python -m pytest
python .\ondasgravitacionais_catalogo_v8.py index
python -m scripts.check_catalog_ready
```

Também é possível executar:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
& .\scripts\upgrade_v7_in_place.ps1
```

O nome do ambiente Conda pode continuar sendo `gw-v7`. O pacote instalado passa a ser `gwproject-v8==0.8.0`.

## Projeto novo

```powershell
conda env create -f .\environment-core.yml
conda activate gw-v8
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[gw,ml,test]"
```

A instalação CUDA continua usando PyTorch 2.5.1 + CUDA 12.1. O script agora utiliza URLs diretas dos wheels, evitando o erro de listagem do índice observado na v7:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_gpu_windows.ps1
```

## Testes

```powershell
python -m pytest
python -m pytest -m benchmark
```

Resultado desta entrega:

```text
32 passed
1 benchmark passed
compileall: ok
```

## Camada 1 — aquisição e condicionamento

Para apenas reconstruir os índices a partir dos dados existentes:

```powershell
python .\ondasgravitacionais_camada1_v8.py --rebuild-index
```

Para baixar ou reutilizar dados e executar toda a Camada 1:

```powershell
python .\ondasgravitacionais_camada1_v8.py `
  --download-strain `
  --process `
  --qgrams
```

Na v8, um Q-gram não é aceito cientificamente quando o processamento anterior foi reprovado. A opção abaixo existe somente para investigação diagnóstica:

```powershell
--diagnostic-qgrams-for-rejected
```

### NaN e Inf nas bordas

A v8 permite um recorte genérico e conservador quando os valores não finitos ocupam somente as bordas e o evento permanece dentro de um trecho contínuo suficientemente longo. Nenhum valor é interpolado.

Os critérios ficam em `configs/processing.toml`:

```toml
allow_edge_nonfinite_trim = true
reject_internal_nonfinite = true
min_valid_before_event_seconds = 8.0
min_valid_after_event_seconds = 8.0
min_contiguous_valid_seconds = 20.0
```

Essa regra não é específica para um evento. Ela é aplicada igualmente a todo o catálogo.

## Índices de qualidade

```powershell
python .\ondasgravitacionais_catalogo_v8.py index
```

Arquivos produzidos:

```text
outputs/gwtc/channels_quality.csv
outputs/gwtc/approved_channels.csv
outputs/gwtc/rejected_nonfinite_channels.csv
outputs/gwtc/unavailable_channels.csv
outputs/gwtc/catalog_dataset.csv
outputs/gwtc/network_ready_events.csv
outputs/gwtc/single_detector_events.csv
outputs/gwtc/unusable_events.csv
outputs/gwtc/catalog_dataset_summary.json
```

Verificação física dos caminhos:

```powershell
python -m scripts.check_catalog_ready
```

## Ridge do catálogo

### Smoke test em dois eventos

```powershell
python .\ondasgravitacionais_catalogo_v8.py ridge -- `
  --limit 2 `
  --workers 2 `
  --no-resume
```

### Catálogo completo

```powershell
python .\ondasgravitacionais_catalogo_v8.py ridge -- `
  --workers 2
```

A execução grava um JSON por canal. Se for interrompida, a próxima execução retoma os resultados existentes. Para recalcular tudo, use `--no-resume`.

Saídas:

```text
outputs/gwtc/ridge_catalog/per_channel.csv
outputs/gwtc/ridge_catalog/per_event.csv
outputs/gwtc/ridge_catalog/failures.csv
outputs/gwtc/ridge_catalog/ridge_arrays/*.npz
outputs/gwtc/ridge_catalog/summary.json
```

O algoritmo:

- calcula uma janela adaptativa a partir da chirp mass do catálogo;
- limita a banda inspiral usando a ISCO quando as massas estão disponíveis;
- usa Q constante com `Q=5`;
- extrai um caminho quase monotônico por programação dinâmica;
- ajusta a relação newtoniana com Theil–Sen;
- registra dispersão bootstrap do ridge;
- combina os detectores pela mediana por evento.

A estimativa é diagnóstica e não substitui inferência Bayesiana.

## Matched filtering do catálogo

### Smoke test

```powershell
python .\ondasgravitacionais_catalogo_v8.py matched-filter -- `
  --limit 2 `
  --workers 1 `
  --no-resume
```

### Catálogo completo

```powershell
python .\ondasgravitacionais_catalogo_v8.py matched-filter -- `
  --workers 1
```

Para cada evento, o banco local contém:

- template central nas massas catalogadas;
- escalas de 90% e 110%;
- mesma massa total com razão de massas mais simétrica;
- mesma massa total com razão de massas mais assimétrica.

Saídas:

```text
outputs/gwtc/matched_filter_catalog/per_event.csv
outputs/gwtc/matched_filter_catalog/rankings.csv
outputs/gwtc/matched_filter_catalog/events/*.json
outputs/gwtc/matched_filter_catalog/summary.json
```

O fluxo usa TaylorF2 e o instante catalogado do evento. Não calcula FAR, fundo temporal, veto chi-quadrado ou busca cega. Quando apenas massas source-frame estão disponíveis, isso é marcado explicitamente.

## Pipeline PowerShell

Smoke test completo:

```powershell
& .\scripts\run_catalog_v8.ps1 -Mode Smoke
```

Execução completa:

```powershell
& .\scripts\run_catalog_v8.ps1 -Mode Full
```

Somente índice e ridge:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_catalog_v8.ps1 `
  -Mode Full `
  -SkipMatchedFilter
```

## Gravity Spy

O treinamento permanece separado:

```powershell
python .\ondasgravitacionais_camada3_v8.py gravity-spy `
  .\data\gravity_spy\trainingset_v1d1_metadata.csv `
  .\data\gravity_spy\trainingset `
  -- `
  --profile cuda_balanced `
  --device cuda `
  --batch-size 32
```

Depois do treinamento, aplique o melhor checkpoint aos 230 Q-grams. Como os Q-grams do catálogo cobrem 4 s e o conjunto Gravity Spy usa imagens de 1 s, a v8 recorta automaticamente o segundo central antes do redimensionamento:

```powershell
python .\ondasgravitacionais_catalogo_v8.py classify-qgrams `
  .\outputs\gravity_spy\seed_42_cuda_balanced\best_model.pt `
  -- `
  --device cuda
```

Saídas:

```text
outputs/gwtc/gravity_spy_predictions/per_channel.csv
outputs/gwtc/gravity_spy_predictions/per_event.csv
outputs/gwtc/gravity_spy_predictions/summary.json
```

A classificação mede semelhança com as classes Gravity Spy. A softmax não deve ser interpretada como probabilidade física de glitch nem usada isoladamente como veto de evento.

## Camadas específicas do GW150914

As rotinas da v7 foram preservadas:

```powershell
python .\ondasgravitacionais_camada2_v8.py
python .\ondasgravitacionais_camada3_v8.py matched-filter
```

Elas continuam sendo diagnósticos específicos do GW150914. A análise dos 90 eventos deve usar `ondasgravitacionais_catalogo_v8.py`.

## Estrutura principal

```text
ondas_gravitacionais_v8/
├── configs/
│   ├── processing.toml
│   ├── matched_filter.toml
│   ├── gravity_spy.toml
│   └── catalog_analysis.toml
├── scripts/
│   ├── build_catalog_index.py
│   ├── check_catalog_ready.py
│   ├── run_catalog.py
│   ├── run_catalog_ridge.py
│   ├── run_catalog_matched_filter.py
│   ├── classify_catalog_qgrams.py
│   ├── run_catalog_pipeline.py
│   ├── run_catalog_v8.ps1
│   └── upgrade_v7_in_place.ps1
├── src/gwproject/
│   ├── catalog_index.py
│   ├── catalog_analysis.py
│   ├── segments.py
│   └── qgram_inference.py
├── tests/
├── ondasgravitacionais_camada1_v8.py
├── ondasgravitacionais_camada2_v8.py
├── ondasgravitacionais_camada3_v8.py
└── ondasgravitacionais_catalogo_v8.py
```

## Correção 0.8.1 — ridge quantizado

A versão 0.8.1 trata um caso observado nos dados reais: o estimador Theil–Sen pode retornar inclinação exatamente zero quando o ridge ocupa poucos bins de frequência e contém muitos empates. A correção:

- mantém Theil–Sen como estimador principal;
- usa regressão linear apenas como fallback quando o caminho é crescente, mas quantizado;
- continua rejeitando ridges realmente planos ou com inclinação não física;
- exige cobertura temporal e frequencial mínima nos pontos de ajuste;
- tenta quantis de energia progressivamente menos restritivos;
- registra `fit_method`, `selection` e estados de falha específicos;
- inclui o subcomando `diagnose-ridge`.

Após aplicar a atualização, reexecute sem cache:

```powershell
python .\ondasgravitacionais_catalogo_v8.py ridge -- --workers 2 --no-resume
python .\ondasgravitacionais_catalogo_v8.py diagnose-ridge
```

## Nota v8.2 — massas observadas

As frequencias do sinal observado dependem das massas no frame do detector. A
v8.2 usa massas detector-frame publicadas ou reconstrui essas massas por
`m_det=(1+z)m_source`. O uso de massas source-frame sem redshift fica desativado
por padrao. Os gates do matched filtering sao exportados separadamente para
nao confundir deteccao por SNR/margem com consistencia do template central.

## Atualização de metadados detector-frame — v8.3

Projetos migrados da v7 devem enriquecer o `catalog.csv` legado antes de executar ridge ou matched filtering:

```powershell
python .\ondasgravitacionais_catalogo_v8.py refresh-metadata -- --workers 4
```

O comando consulta somente metadados da API GWOSC v2, cria backup do catálogo e preserva strain, processados, Q-grams e `run_log.json`.


## Programa científico v9.0 — robustez open-world

A v9.0 acrescenta um programa preregistrado para estudar domain shift, renderer shortcuts, fase, ensemble e OOD. Os resultados v8.3 permanecem preservados como baseline histórico.

Documentação principal:

- `research/PROGRAMA_CIENTIFICO_COMPLETO.md`;
- `research/PREREGISTRO.md`;
- `research/GUIA_DE_EXECUCAO.md`;
- `research/POLITICA_DE_ALEGACOES.md`.

Entrada de linha de comando:

```powershell
python .\ondasgravitacionais_research_v9.py --help
```

O fluxo completo local pode ser iniciado por:

```powershell
& .\scripts\run_research_program.ps1
```

A categoria `in_domain_candidate` não é rótulo físico nem veto de evento.
