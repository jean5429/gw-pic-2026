# Incorporação de jean5429/gw-pic-2026 ao GWProject

Data: 10 de outubro de 2026.

## Origem

- Repositório: <https://github.com/jean5429/gw-pic-2026>, commit
  `1447593a0f76e8f8b40dfb03d21abbcef03a6145` (09/10/2026), de Jean Gonçalves.
- O repositório deriva da versão 9.0 (`gwproject-v8` 0.9.0), a mesma linha de
  `../gwproject_v9_0_full`. Não contém a reconstrução 1.0/2.0.
- Licença de origem: MIT, "Copyright (c) 2026 PIC Astronomia". O texto está em
  `gw_student_kit/LICENSE` e vale também para as partes adaptadas em `src/`
  listadas abaixo. A licença do GWProject continua pendente
  (`RELEASE_METADATA_REQUIRED.md`); a decisão precisa preservar esse aviso.

## Incorporado

| Origem (jean5429) | Destino (1.0.1) | Diferenças |
|---|---|---|
| `render_mode="gravity_spy"` em `qgram_inference.py` | mesmo módulo, `configs/research_program.toml` e todos os chamadores | Cinza pela cadeia exata do treino (LUT de bytes do matplotlib + `convert("L")`). Mantém o contrato de eixos da 1.0.1, sem o parâmetro `layout`. Padrão continua `"percentile"`; na origem era `"gravity_spy"`. |
| `anonymize_paths` em `provenance.py` | mesmo módulo, aplicado em `write_manifest` | Não é aplicado a `run_log.json`, que é insumo do índice do catálogo. Não troca prefixos de outro usuário. Aceita objetos `Path`. |
| `plotting.py` | `src/gwproject/plotting.py` | API orientada a objetos do matplotlib; o diagrama de confiabilidade do treino também passa a 300 DPI. |
| `scripts/replot_confusion_matrices.py` | `gwproject research replot-confusion` | Raiz padrão `models/gravity_spy_harmonized_v1d1`; `--output-dir` evita sobrescrever os PNGs dos modelos. |
| `scripts/run_gradcam.py` | `gwproject research gradcam` (`research/gradcam.py`) | Gradiente por `torch.autograd.grad`, sem hooks de backward; camada `layer4[-1]` (origem: `layer4[-1].conv2`); proteção contra mapa nulo; sem OpenCV; entrada idêntica à do ensemble; aceita `--event/--detector`, `--qgram` ou `--image`; JSON de proveniência com probabilidades calibradas. |
| `scripts/temp_npy_to_png.py` | `gwproject research render-qgram` | Usa o mesmo `Resize` da inferência (a origem usava o padrão do PIL) e renderiza os dois modos. |
| `gw_student_kit/` | `gw_student_kit/` | Correções descritas abaixo; figuras antigas não copiadas. |

## Criado nesta incorporação

- `gwproject research compare-ensembles`: compara duas execuções do ensemble
  com os mesmos canais, checkpoints e referência OOD, com bootstrap por evento.
- `classify-ensemble --render-mode`: sobrepõe a configuração e recusa gravar na
  saída canônica. Cada canal registra `mahalanobis_log_excess_99`.
- 31 testes novos: `test_qgram_render_modes.py`, `test_gradcam.py`,
  `test_compare_ensembles.py`, `test_plotting.py`, `test_classify_ensemble_options.py`,
  `test_student_kit.py` e dois casos em `test_provenance_and_catalog.py`.
- `research/RENDERIZADOR_GRAVITY_SPY.md`, figuras em `research/figures/`, notas
  na seção 18 do `PREREGISTRO.md` e na `POLITICA_DE_ALEGACOES.md`.

## Correções no kit didático

O filtro casado de origem dava SNR de 1,6 × 10⁹ na simulação e 3,9 × 10¹⁰ no
GW150914, com o pico fora do instante da injeção.

1. Fatores Δt da transformada: ⟨ρ²⟩ = 2 em ruído gaussiano (antes, inflado por ~f_s).
2. Template filtrado como os dados e soma restrita à banda do passa-banda.
3. Atraso da correlação convertido em instante de fusão (`align_to_merger`).
4. `inject_signal` passa a usar a SNR ótima em ruído branco; antes, "SNR = 14"
   equivalia a ρ ≈ 1.800.
5. Eventos reais filtrados com chirp newtoniano da massa chirp do catálogo.
6. Janela de Tukey e PSD na região central contra os transientes de borda.
7. Semente para o ruído sintético; leitor HDF5 aceita o formato oficial do GWOSC.

Resultado de referência: GW150914 com ρ ≈ 15,0 (H1) e 10,8 (L1) perto de
t = 16,02 s, L1 ~7 ms antes de H1; simulação com ρ ≈ 13 no instante injetado.

## Não incorporado

- Checkpoints, referências OOD e saídas da origem: 90 eventos do GWTC-3, treino
  sem o filtro de metadados v1d1 (macro-F1 0,856 contra 0,945) e referência OOD
  no formato anterior ao 4. Não são comparáveis nem citáveis sob a política atual.
- Tolerância de até 49 imagens com falha em `prepare_gravity_spy_panels`:
  contraria o controle de qualidade explícito da 1.0.1.
- `export_word.py`, relatórios V1/V2, `docs/historico` (já presentes nas
  pastas locais das versões anteriores), scripts `ondasgravitacionais_*` e
  `configs/gravity_spy_60epochs.toml`.

## Verificação

- `pytest`: 122 aprovados (91 existentes + 31 novos), 1 benchmark desmarcado.
  Ambiente: Python 3.13.1, `torch 2.14.0+cpu`, `torchvision 0.29.0+cpu`.
- `sha256sum -c ARTIFACT_HASHES.sha256`: os 25 artefatos canônicos conferem.
  Nenhuma saída canônica nem PNG dos modelos foi alterado.
- O ensemble em CPU com `render_mode="percentile"` reproduz a execução canônica
  em GPU: mesmas decisões e classes nos 475 canais.

## Pendências

- Os ambientes conda `gw-v7` e `gwproject`, usados por `tools/*.ps1` e pelos
  manifestos de julho, não existem mais em `%USERPROFILE%\anaconda3`. Recrie o
  ambiente com `environment.yml` (e `tools/install_gpu_windows.ps1` para GPU)
  antes de usar os roteiros PowerShell.
- Commitado em 10/10/2026 em dois commits: o trabalho de 18/09/2026
  (experimento pareado e auditoria do HDF5) e esta migração. As correções do kit
  didático foram enviadas ao repositório de origem como pull request a partir do
  fork `prestotedesco/gw-pic-2026`.
