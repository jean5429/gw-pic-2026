# Guia de execução do programa de pesquisa

> **Retomada em 18/09/2026:** se os modelos, a referência OOD v4 e os 475 canais
> dos 218 eventos já estão prontos, siga
> [RETOMADA_EXPERIMENTO_PAREADO.md](RETOMADA_EXPERIMENTO_PAREADO.md).
> O código novo foi preparado, mas não executado pelo assistente. O roteiro
> abaixo documenta a reconstrução completa; não é necessário repeti-la para
> iniciar o experimento pareado.

## Ambiente

```powershell
conda activate gwproject
python -m pip install -e ".[all,test]"
pytest
```

## 1. Harmonização experimental dos painéis

```powershell
gwproject research prepare-panels `
  .\data\gravity_spy\trainingset `
  .\data\gravity_spy_harmonized `
  --metadata .\data\gravity_spy\trainingset_v1d1_metadata.csv `
  --overwrite
```

O comando usa por padrão o recorte oficial `LEFT=105`, `TOP=66`, `RIGHT=671`, `BOTTOM=532`. A variante `--crop-mode auto` e a opção `--apply-frequency-crop` são estritamente exploratórias e devem produzir domínios de avaliação separados. A saída deve ser comparada com os arrays oficiais em HDF5 antes de qualquer interpretação científica.

## 2. Treinamento multiseed

```powershell
gwproject research train-seeds `
  .\data\gravity_spy_harmonized\clean_metadata.csv `
  .\data\gravity_spy_harmonized `
  --profile cuda_balanced `
  --output-dir .\models\gravity_spy_harmonized_v1d1

gwproject research aggregate-seeds .\models\gravity_spy_harmonized_v1d1
```

## 3. Referência OOD por checkpoint

```powershell
gwproject research build-ood `
  .\models\gravity_spy_harmonized_v1d1 `
  --metadata .\data\gravity_spy_harmonized\clean_metadata.csv `
  --images-root .\data\gravity_spy_harmonized `
  --device cuda `
  --output .\outputs\research\gwtc4_218\ood_reference.npz
```

O arquivo formato 4 ajusta médias e precisão Mahalanobis no treino e calibra as
distribuições empíricas de energia e distância na validação, separadamente para
cada checkpoint. O vínculo é verificado pelo hash SHA-256 do arquivo.

Antes de qualquer inferência externa, executar o controle no teste oficial:

```powershell
gwproject research validate-ood `
  .\outputs\research\gwtc4_218\ood_reference.npz `
  --metadata .\data\gravity_spy_harmonized\clean_metadata.csv `
  --images-root .\data\gravity_spy_harmonized `
  --device cuda `
  --output .\outputs\research\gwtc4_218\ood_test_control.json
```

O teste é apenas controle final. Não pode selecionar threshold, renderer ou
configuração.

## 4. Inferência ensemble

```powershell
gwproject research classify-ensemble `
  .\models\gravity_spy_harmonized_v1d1 `
  --approved .\outputs\gwtc4_218\approved_channels.csv `
  --device cuda `
  --ood-reference .\outputs\research\gwtc4_218\ood_reference.npz `
  --output-dir .\outputs\research\gwtc4_218\gravity_spy_ensemble
```

Interpretação dos estados:

- `reject_ood`: incompatibilidade forte com o domínio de referência;
- `review`: resultado limítrofe ou ensemble instável;
- `in_domain_candidate`: compatibilidade estatística suficiente para análise posterior.

Nenhum desses estados determina a origem física do transiente.
`reported_class` fica vazio para `review` e `reject_ood`; o rótulo fechado
forçado permanece apenas em `forced_closed_set_class`.

## 5. Auditoria de mudança de domínio

```powershell
gwproject research catalog-statistics `
  .\outputs\research\gwtc4_218\gravity_spy_ensemble\per_channel.csv `
  --output .\outputs\research\gwtc4_218\catalog_rendered_statistics.csv

gwproject research audit-domain `
  .\outputs\research\gwtc4_218\catalog_rendered_statistics.csv `
  --output-dir .\outputs\research\gwtc4_218\domain_shift_audit `
  --target-column forced_closed_set_class `
  --positive-label None_of_the_Above `
  --group-column event
```

As estatísticas são calculadas depois da mesma renderização usada pela CNN. A
auditoria atual é pós-hoc e mede previsibilidade das escolhas forçadas; não
identifica causalmente um shortcut.

## 6. Tabela mestra dos 218 eventos

```powershell
gwproject research consolidate-events `
  --quality .\outputs\gwtc4_218\event_quality_summary.csv `
  --ensemble .\outputs\research\gwtc4_218\gravity_spy_ensemble\per_event.csv `
  --ridge .\outputs\research\gwtc4_218\ridge_diagnostic\per_event.csv `
  --matched-filter .\outputs\research\gwtc4_218\matched_filter_diagnostic\per_event.csv `
  --expected-events 218 `
  --output .\outputs\research\gwtc4_218\all_218_events.csv
```

O comando falha se qualquer fluxo tiver eventos duplicados ou um conjunto de
eventos diferente. As colunas recebem prefixos `qc_`, `cnn_`, `ridge_` e `mf_`.

## 7. Contrafactuais de fase

```powershell
gwproject research phase-counterfactuals --help
```

## 8. Renderer exploratório e inspeção visual

```powershell
gwproject research classify-ensemble `
  .\models\gravity_spy_harmonized_v1d1 `
  --ood-reference .\outputs\research\gwtc4_218\ood_reference.npz `
  --render-mode gravity_spy `
  --output-dir .\outputs\research\renderer_gravity_spy_NOVA_DATA\ensemble_gravity_spy

gwproject research compare-ensembles `
  .\outputs\research\gwtc4_218\gravity_spy_ensemble `
  .\outputs\research\renderer_gravity_spy_NOVA_DATA\ensemble_gravity_spy `
  --output-dir .\outputs\research\renderer_gravity_spy_NOVA_DATA\comparacao

gwproject research render-qgram --event GW150914-v4 --render-modes percentile gravity_spy
gwproject research gradcam --event GW150914-v4 --detector L1 --render-mode gravity_spy
```

`--render-mode` não pode gravar na saída canônica. A comparação recusa execuções
com canais, checkpoints ou referência OOD diferentes. O renderer `gravity_spy` é
exploratório (seção 18 do `PREREGISTRO.md`); resultados e limites estão em
`RENDERIZADOR_GRAVITY_SPY.md`.

## Automação PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_research_program.ps1
```
