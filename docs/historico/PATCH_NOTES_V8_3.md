# Patch v8.3 — atualização de metadados detector-frame

## Problema corrigido

A v8.2 passou a exigir massas no frame do detector, mas `index` apenas reconstruía os índices a partir do `catalog.csv` legado. Como esse arquivo havia sido criado pela v7 sem a coluna `redshift`, todos os 230 canais eram classificados antes da análise como sem metadados detector-frame.

## Alterações

- novo comando `refresh-metadata`, que consulta a API oficial GWOSC v2 para os mesmos 90 eventos;
- nenhuma nova transferência de strain, nenhum novo processamento e nenhuma alteração nos Q-grams;
- backup automático do `catalog.csv`;
- colunas explícitas `chirp_mass_detector`, `mass_1_detector` e `mass_2_detector`;
- proveniência da conversão por `(1+z)`;
- preflight em ridge e matched filtering, evitando sobrescrever resultados quando nenhum evento possui metadados adequados;
- `diagnose-ridge` tolera CSVs sem coluna `reason`;
- estados sem massa agora registram motivo explícito.

## Ordem de execução

```powershell
python .\ondasgravitacionais_catalogo_v8.py refresh-metadata -- --workers 4
python .\ondasgravitacionais_catalogo_v8.py ridge -- --workers 2 --no-resume
python .\ondasgravitacionais_catalogo_v8.py diagnose-ridge
python .\ondasgravitacionais_catalogo_v8.py matched-filter -- --workers 1 --no-resume
```
