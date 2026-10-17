---
title: "Preregistro do estudo de invariância de representação e domain shift"
date: "27 de julho de 2026"
lang: pt-BR
---

# 1. Título

Invariância de representação, atalhos de renderer e detecção open-world em classificadores de transientes de detectores de ondas gravitacionais.

# 2. Status

Preregistro prospectivo para os experimentos confirmatórios. Resultados exploratórios já observados não podem ser usados como confirmação das hipóteses preregistradas.

# 3. Pergunta primária

Depois de harmonizar geometria, faixa de frequência e transformação de intensidade, o desempenho e a calibração do classificador permanecem estáveis em renderer holdout e detector holdout?

# 4. Hipóteses confirmatórias

## H1 primária

A balanced accuracy de um probe logístico baseado em nove estatísticas escalares será menor ou igual a 0,70 depois da harmonização.

## H2 primária

O classificador harmonizado obterá macro-$F_1\ge0{,}85$ no split oficial e macro-$F_1\ge0{,}65$ em renderer holdout.

## H3 primária

Um score OOD combinado alcançará AUROC $\ge0{,}80$ para separar o teste
Gravity Spy, mantido intocado, de contrafactuais/renderers deliberadamente
deslocados e definidos antes da análise.

## H4 secundária

A informação mútua do ensemble será maior em dados deslocados que na validação in-domain.

## H5 secundária

A randomização de fase afetará modelos temporal/phase-aware mais fortemente que o classificador visual.

# 5. Dados

- Gravity Spy v1d1 com splits oficiais;
- painéis harmonizados derivados dos PNGs;
- 475 Q-grams aprovados, correspondentes aos 218 eventos catalogais;
- DeepExtractor reconstructions para experimentos temporais;
- dados O4/O4b apenas como holdout externo;
- injeções simuladas e contrafactuais registrados separadamente.

# 6. Exclusões

Serão excluídos apenas:

- arquivos ausentes/corrompidos;
- PNGs que não possuam as dimensões e coordenadas necessárias para o recorte oficial preregistrado;
- arrays não finitos;
- exemplos duplicados entre splits;
- dados cuja licença impeça a análise.

Não serão removidos exemplos por “aparência ruim” ou por prejudicarem as métricas.

# 7. Pré-processamento fixado

## Gravity Spy

1. recorte oficial dos PNGs nas linhas 66:532 e colunas 105:671;
2. ausência de recorte físico em frequência no fluxo confirmatório;
3. resize $140\times170$;
4. grayscale;
5. transformação afim fixa de intensidade de `[0,1]` para `[-1,1]`.

A detecção automática do painel e qualquer mapeamento de pixels para uma banda física são variantes exploratórias. Elas devem ser avaliadas como renderers separados e não podem substituir silenciosamente o fluxo confirmatório.

## Catálogo

1. recorte temporal central 4 s $\to$ 1 s;
2. baseline `per_frequency_median`;
3. excesso relativo $\max(Q/b-1,0)$;
4. $\log(1+x)$;
5. inversão vertical validada por teste de linha conhecida;
6. percentis 1–99,5%;
7. resize $140\times170$.

# 8. Modelos

- ResNet18 grayscale, cinco sementes: 11, 23, 42, 71, 101;
- baseline regressão logística 32×32;
- probe de nove estatísticas;
- baseline externo ViT/ResNet apenas como análise exploratória;
- modelos temporal/phase-aware em fase posterior.

# 9. Métricas primárias

- macro-$F_1$;
- balanced accuracy;
- ECE;
- OOD AUROC;
- probe balanced accuracy;
- renderer holdout macro-$F_1$.

# 10. Métricas secundárias

- Brier score;
- FPR@95TPR;
- AUPR-in e AUPR-out;
- mutual information do ensemble;
- vote fraction;
- Mahalanobis distance;
- energy score;
- per-class $F_1$;
- detector predictability;
- classe dominante e entropia da distribuição de classes.

# 11. Splits e dependência

Amostras associadas ao mesmo evento ou trigger não podem aparecer em treino e teste. Bootstrap e intervalos serão agrupados por evento/trigger. Renderer variants da mesma amostra permanecem no mesmo fold.

# 12. Testes estatísticos

- intervalos bootstrap de 95%, agrupados;
- diferenças de métricas com bootstrap pareado;
- correção Benjamini–Hochberg para famílias de testes secundários;
- tamanhos de efeito além de $p$-values;
- nenhuma seleção de threshold no conjunto final de teste.

# 13. Análises exploratórias

- clustering de `None_of_the_Above`;
- microestados instrumentais;
- correlação com ridge e matched filter;
- estudos de GW200129/GW191109;
- descoberta de candidatos coerentes.

Essas análises serão rotuladas como exploratórias e exigirão validação independente.

# 14. Regras de decisão

## Evidência favorável à invariância

- probe baixo;
- renderer holdout forte;
- calibration aceitável;
- baixa sensibilidade a transformações equivalentes;
- diversidade plausível de classes;
- OOD detectável.

## Evidência contra

- probe $>0{,}75$;
- colapso em uma ou duas classes;
- confiança fechada elevada em amostras rejeitadas por controles OOD
  independentes;
- queda $>30\%$ em holdout;
- predição dominada por detector/renderer sem relação instrumental independente.

# 15. Política de transparência

Todos os resultados serão reportados, inclusive falhas. O pipeline de referência será congelado. Mudanças de preprocessing gerarão novos manifests e não sobrescreverão resultados anteriores.

# 16. Nota de correção e validação (26 de julho de 2026)

Até esta data, a implementação de `src/gwproject/qgram_inference.py::qgram_to_uint8` **não
correspondia** ao pré-processamento descrito na seção 7 ("Catálogo"), apesar de o texto
descrever corretamente a intenção:

- item 1 ("recorte temporal central 4 s → 1 s") era aplicado sobre o eixo 1 do array bruto
  (frequência), não o eixo 0 (tempo). O recorte real mantinha os 4 s inteiros e descartava
  ~75% da banda de frequência de 20–512 Hz;
- item 5 ("inversão vertical validada por teste de linha conhecida") tinha sua validação
  comprometida por circularidade: o teste de contrato (`tests/test_research_qgram.py`)
  usava uma fixture sintética `(4, 8)` que já pressupunha eixo 0 = frequência — a mesma
  convenção, não verificada, que a função deveria implementar. Nenhum teste comparava a
  função contra um artefato `.npy` real.

A causa raiz e as evidências completas estão registradas na auditoria de 25/07/2026
(`bug_eixo_qgram`, memória de projeto). A correção foi implementada e validada nesta data:

1. `qgram_to_uint8` agora recorta a duração sobre o eixo 0 (tempo) do array de entrada,
   reduz `baseline_mode="per_frequency_median"` sobre o eixo do tempo (produzindo um
   escalar por bin de frequência, não o inverso), transpõe o resultado para
   `(n_frequencia, n_tempo)` e só então aplica `flip_frequency` sobre o eixo de frequência
   já transposto. O contrato de entrada/saída está documentado no docstring da função.
2. Validação contra artefato real:
   `outputs/gwtc4_218/qgrams/GW150914-v4/H1_qgram.npy`
   (shape `[1000, 500]`, confirmado em `H1_qgram.json`) processado com a configuração de
   produção (`configs/research_program.toml [harmonized_inference]`) produz uma imagem
   `(500, 250)` cujo ridge de potência (`ridge_from_energy`, `ridge.py`) sobe de ~38 Hz em
   t=−0,062 s para ~260 Hz em t=+0,030 s relativo ao GPS do evento — a assinatura de chirp
   esperada para GW150914 (correlação tempo×frequência pré-merger = +0,83), consistente com
   os parâmetros publicados do evento. Antes da correção, o mesmo artefato produzia uma
   imagem sem estrutura de chirp reconhecível e classificação `Scattered_Light` com
   confiança 0,999984.
3. `tests/test_qgram_inference.py` e `tests/test_research_qgram.py` foram reescritos para
   verificar o contrato de eixo explicitamente (posição de um pixel marcado, não apenas
   shape), incluindo um teste de regressão nas dimensões reais medidas (1000×500 → 500×250).

Nenhum resultado catalogal (classificação, OOD, ensemble) produzido antes desta nota é
válido. Os 475 Q-grams catalogais existentes não precisam ser regerados (o artefato `.npy`
bruto sempre esteve correto); apenas precisam ser reprocessados com a função corrigida.

# 17. Nota de correção da referência OOD (27 de julho de 2026)

A referência OOD formato 3 reutilizava o split de validação tanto para ajustar
as médias/covariância Mahalanobis quanto para calibrar os próprios percentis.
Esse procedimento foi invalidado após um controle no teste oficial: o limiar
nominal de 99% rejeitava entre 26% e 30% das 1.158 imagens in-domain.

O formato 4 separa as funções:

1. ajuste Mahalanobis nas 5.472 amostras de treino;
2. calibração empírica nas 1.177 amostras de validação;
3. controle final nas 1.158 amostras de teste.

No controle independente, as frações Mahalanobis acima de 0,95 ficaram entre
3,4% e 5,4%, e as frações acima de 0,99 entre 0,4% e 1,2% nos cinco
checkpoints. Depois dessa correção, os 475 canais dos 218 eventos foram
reprocessados. Todos permanecem `reject_ood`, porque os cinco modelos os colocam
acima do percentil 0,99 de Mahalanobis. O energy score, contudo, quase nunca os
rejeita; essa discordância metodológica deve ser preservada e investigada.

O probe pós-hoc calculado sobre as imagens efetivamente apresentadas à CNN
obteve balanced accuracy 0,769 e AUROC 0,817 para reproduzir a escolha fechada
`None_of_the_Above` versus as demais, com folds agrupados por evento. Como foi
observado antes do experimento pareado por trigger, esse número é diagnóstico,
não confirmação prospectiva de H1.

# 18. Nota sobre o renderer exploratório Gravity Spy (10 de outubro de 2026)

`qgram_to_uint8` ganhou a opção `render_mode = "gravity_spy"`, incorporada de
jean5429/gw-pic-2026. Ela renderiza a energia normalizada na escala dos PNGs
Gravity Spy (0–25,5 em viridis, convertida para cinza). É um renderer separado,
como exige a seção 7, e não substitui o fluxo confirmatório: o padrão em
`configs/research_program.toml` continua `render_mode = "percentile"`.

Uma comparação exploratória e pareada nos 475 canais, com os mesmos modelos e a
mesma referência OOD, reduziu a fração de canais `reject_ood` por evento de 1,00
para 0,54. Os canais acima de q99 em Mahalanobis caíram de 475 para 49, e os
acima de q99 em energy subiram de 0 para 227 (`research/RENDERIZADOR_GRAVITY_SPY.md`).
Esse resultado foi observado sem protocolo prospectivo. Não confirma H1 nem
justifica trocar o renderer canônico. A adoção de qualquer renderer exige
avaliação prévia em triggers rotulados e pareados (Fase 1 do parecer), com a
banda de frequência também harmonizada.
