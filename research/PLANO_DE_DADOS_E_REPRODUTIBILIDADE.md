# Plano de dados e reprodutibilidade

## Estrutura

- `data/raw`: nunca modificado;
- `data/derived`: derivados versionados por hash de configuração;
- `outputs/baseline_v8_3`: snapshot imutável;
- `outputs/gravity_spy_harmonized`: novos checkpoints;
- `outputs/research`: OOD, ensemble e ablações;
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

## Licenças

Verificar separadamente GWOSC, Gravity Spy, Hugging Face e outputs derivados. Não redistribuir dados quando a licença permitir apenas uso local. Model cards e dataset cards devem documentar origem, processamento e limitações.

## Prevenção de leakage

- grupos por evento/trigger;
- variantes de renderer no mesmo split;
- normalização estimada apenas no treino;
- OOD thresholds estimados apenas em validação;
- teste final aberto uma única vez.

## Releases

- `v8.3-baseline`: resultados históricos;
- `v9.0-research`: código e preregistro;
- `v9.1-results`: resultados harmonizados;
- `v10.0-benchmark`: benchmark aberto após auditoria/licenças.
