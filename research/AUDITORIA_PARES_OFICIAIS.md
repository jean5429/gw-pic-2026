# Auditoria dos pares oficiais — preparação em 18/09/2026

O responsável executou o download e confirmou o MD5 oficial
`a6665b4ebee7b4ea57164ee743c2bd11`. O relatório
`outputs/research/official_hdf5_20260918_195840_03e906df.schema.json`
tem `truncated: true`: o limite de 2.000 objetos foi atingido. O trecho
inspecionado contém datasets como
`/1080Lines/test/2Xn58S1QCj/1.0.png`, de shape `(1, 140, 170)` e dtype
`float32`. Isso é compatível com imagens preparadas; não comprova presença
de strain, unidades físicas, escala de pixels ou equivalência de renderização.

## Próxima execução

Na raiz do projeto:

```powershell
& .\tools\audit_official_pairs.ps1
```

O roteiro e os novos testes foram escritos e revisados por leitura, **sem
execução pelo assistente**. O roteiro executa primeiro os testes locais e
interrompe se houver falha. Não baixa novamente o HDF5, não treina e não
executa inferência da CNN. A auditoria não usa GPU; lê um array por vez.

## Contrato

- Conferir tamanho e MD5 do HDF5 e MD5 dos metadados oficiais; calcular hashes de procedência.
- Percorrer toda a estrutura acessível por hard links, sem limite de 2.000 objetos. Links externos/soft links não são seguidos; aliases e esquemas inesperados são registrados como pendências.
- Indexar os datasets de 1 segundo e verificar o pareamento de todos os 7.807 registros limpos do Gravity Spy. Não parear por posição de linha.
- Recuperar o detector dos metadados apenas se o ID for globalmente único nos metadados oficiais completos. Exigir concordância de detector, classe e split entre os registros aplicáveis. O detector não está demonstrado no caminho HDF5 observado.
- Ler valores somente dos pares válidos de validação, com meta de 1.177 pares. Treino e teste têm apenas suas chaves/estrutura inspecionadas.
- Registrar mínimo, máximo, média e desvio dos pixels oficiais. Comparar diretamente com o PNG local nas escalas fixas `u8` (0–255), `unit` (0–1) e `signed` (−1–1), reportando MAE e erro absoluto máximo.
- Não ajustar normalização por imagem, não escolher automaticamente a escala de menor erro e não aplicar clipping, interpolação ou inversão espacial.

As três comparações são diagnósticas. Mesmo a coincidência dos pixels não
prova equivalência física entre processamentos de strain. Divergência pode
refletir escala, recorte, interpolação ou outro processamento; requer
investigação antes de escolher alterações na CNN.

## Produtos e interpretação

A execução cria uma pasta nova `outputs/research/official_pairs_<data>_<id>`:

- `summary.json`: cobertura, contagens, pendências, estatísticas de pixels e procedência.
- `hdf5_index_1s.csv`: índice completo dos datasets de 1 segundo identificados.
- `pairing_all_clean.csv`: resultado do pareamento de cada registro limpo, incluindo falhas.
- `validation_pixel_comparison.csv`: diagnóstico por imagem de validação, incluindo falhas.

O log fica em `outputs/research/logs`. Envie primeiro `summary.json` e a
saída do terminal. `status: complete` significa que a auditoria terminou;
**não significa que o pareamento passou**. `coverage_gate` exige 7.807
pareamentos válidos, 1.177 comparações de pixels e nenhuma pendência estrutural.
Se for falso, os arquivos detalhados identificam o motivo; não serão descartadas
linhas silenciosamente para produzir um resultado aprovado.

Os resultados anteriores dos 218 eventos/475 canais permanecem inalterados.
Esta etapa audita a representação da base Gravity Spy, não reclassifica os
eventos do catálogo e não substitui controles off-source ou injeções.
