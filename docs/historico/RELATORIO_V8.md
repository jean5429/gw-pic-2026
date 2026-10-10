# Relatório de implementação — versão 8

## Veredito

A v8 transforma a base validada da Camada 1 em um pipeline de catálogo. Os 90 eventos e 230 canais aprovados passam a ser entradas formais para ridge, matched filtering local e classificação experimental dos Q-grams.

## Dados de entrada assumidos

A implementação foi construída para o estado observado:

- 90 eventos;
- 270 combinações evento–detector;
- 250 downloads bem-sucedidos;
- 230 canais aprovados;
- 20 canais indisponíveis;
- 15 canais não finitos;
- 5 canais com whitening fora do gate;
- 88 eventos com pelo menos dois detectores;
- 2 eventos com apenas L1;
- nenhum evento totalmente inutilizável.

## Correções de engenharia

### Índice reproduzível

Os blocos PowerShell usados na investigação foram convertidos em `gwproject.catalog_index`. A classificação dos canais agora é parte testada do projeto.

### Falhas de download

`data_unavailable` é separado de rejeições científicas. Um detector indisponível não é descrito como se tivesse falhado no whitening ou na qualidade física.

### Dependência entre gates

O Q-transform não é aceito quando o processamento anterior reprova. Q-grams diagnósticos ainda podem ser solicitados explicitamente, mas recebem `scientific_eligibility=false`.

### Dados não finitos

O novo módulo `segments.py` seleciona somente um trecho finito contínuo contendo o evento. O recorte é permitido apenas nas bordas, exige contexto antes e depois e não interpola amostras.

## Ridge do catálogo

A janela é calculada por evento usando a chirp mass de referência. Quando as massas componentes estão disponíveis, a banda superior é limitada por uma fração da ISCO.

A extração usa:

- banco gaussiano de Q constante;
- redução temporal por máximos em blocos;
- programação dinâmica com penalidade de saltos;
- caminho aproximadamente crescente em frequência;
- seleção dos pontos de maior energia;
- ajuste robusto Theil–Sen;
- bootstrap em blocos;
- combinação por mediana entre detectores.

A saída distingue ausência de massa de referência, banda inspiral insuficiente, janela curta, ridge insuficiente e erro numérico.

## Matched filtering do catálogo

Cada evento recebe um banco local construído em torno das massas catalogadas. O template central é acompanhado por perturbações de escala e de razão de massas.

A série padrão é inspiral. A rotina:

- estima PSD por detector;
- calcula pico em torno do GPS catalogado;
- combina SNRs em quadratura;
- informa rank do template central;
- informa razão top-2;
- marca massas source-frame quando não há massas detector-frame.

O resultado continua pedagógico: TaylorF2, tempo conhecido, sem FAR, sem chi-quadrado e sem busca cega.

## Gravity Spy

Um checkpoint treinado pode ser aplicado aos Q-grams aprovados. Os arrays são recortados do intervalo central de 1 s, compatível com a duração usada no treinamento, e depois convertidos em imagens por log, percentis e normalização robusta.

A saída inclui classe principal, segunda classe, confiança e entropia. O relatório proíbe interpretar a softmax como probabilidade física de glitch.

## Adequação ao notebook

- ridge: CPU, dois workers recomendados;
- matched filtering: CPU, um worker inicial recomendado;
- treinamento e inferência Gravity Spy: RTX 3070 Ti com `cuda_balanced` e batch inicial 32;
- retomada automática evita perder execuções longas.

## Validação realizada

```text
python -m pytest
32 passed

python -m pytest -m benchmark
1 passed

python -m compileall
compile_ok
```

Também foi executado um smoke test sintético de ponta a ponta do script de ridge, com geração dos arquivos `per_channel.csv`, `per_event.csv`, `failures.csv`, JSON por canal e NPZ do caminho.

## Limites

Não foram executados neste ambiente:

- ridge dos 230 canais reais;
- matched filtering dos 90 eventos reais;
- treinamento integral Gravity Spy;
- inferência dos 230 Q-grams em checkpoint treinado.

Essas etapas dependem dos dados locais já baixados e da GPU física do usuário. O código grava manifestos e resultados intermediários suficientes para auditoria.
