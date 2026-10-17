# Plano de dados e reprodutibilidade

## Estrutura

- `data/gravity_spy`: arquivo e imagens oficiais, preservados;
- `data/gravity_spy_harmonized`: painéis recortados e `clean_metadata.csv`;
- `models/gravity_spy_harmonized_v1d1`: cinco checkpoints canônicos;
- `models/legacy`: modelos anteriores, não canônicos;
- `outputs/gwtc4_218`: dados e índices do catálogo confirmatório;
- `outputs/research/gwtc4_218`: OOD, ensemble e diagnósticos;
- `outputs/legacy`: snapshots históricos;
- `research`: preregistro, referências e relatórios.

## Provenance obrigatório

Cada execução deve registrar:

- commit Git;
- versão do pacote;
- hash do config;
- hashes dos dados de entrada;
- ambiente Python/CUDA;
- sementes;
- dispositivo;
- timestamp UTC;
- comandos completos;
- critérios de exclusão.

O catálogo confirmatório exige 218 eventos e registra explicitamente a exclusão
de `GW230630_070659`. A integridade de todos os arquivos brutos, processados,
PSDs e Q-grams é verificada pelos hashes do `run_log.json`.

## Licenças

Verificar separadamente GWOSC, Gravity Spy, Hugging Face e outputs derivados. Não redistribuir dados quando a licença permitir apenas uso local. Model cards e dataset cards devem documentar origem, processamento e limitações.

## Prevenção de leakage

- grupos por evento/trigger;
- variantes de renderer no mesmo split;
- normalização estimada apenas no treino;
- médias/covariância Mahalanobis ajustadas apenas no treino;
- OOD thresholds estimados apenas em validação;
- teste final aberto uma única vez.

## Artefatos persistentes

- `code-release`: código, configurações e preregistro;
- `reference-data`: snapshot imutável dos dados permitidos para redistribuição;
- `model-results`: checkpoints, logits, probabilidades e calibração;
- `open-benchmark`: benchmark publicado somente após auditoria de licença e leakage.

Referências OOD formato 3 são inválidas e permanecem apenas em
`outputs/legacy/research_invalid_ood_v3_20260727`. Artefatos publicáveis devem
usar formato 4 e incluir `ood_test_control.json`, as estatísticas renderizadas e
o resultado do probe agrupado por evento/trigger.
