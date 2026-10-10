---
title: "Preregistro do estudo de invariância de representação e domain shift"
date: "25 de julho de 2026"
lang: pt-BR
---

# 1. Título

Invariância de representação, atalhos de renderer e detecção open-world em classificadores de transientes de detectores de ondas gravitacionais.

# 2. Status

Preregistro prospectivo para os experimentos da v9. Os resultados v8.3 são considerados dados exploratórios já observados. Eles não podem ser usados como confirmação das hipóteses preregistradas.

# 3. Pergunta primária

Depois de harmonizar geometria, faixa de frequência e transformação de intensidade, o desempenho e a calibração do classificador permanecem estáveis em renderer holdout e detector holdout?

# 4. Hipóteses confirmatórias

## H1 primária

A balanced accuracy de um probe logístico baseado em nove estatísticas escalares será menor ou igual a 0,70 depois da harmonização.

## H2 primária

O classificador harmonizado obterá macro-$F_1\ge0{,}85$ no split oficial e macro-$F_1\ge0{,}65$ em renderer holdout.

## H3 primária

Um score OOD combinado alcançará AUROC $\ge0{,}80$ para separar validação Gravity Spy de contrafactuais/renderers deliberadamente deslocados.

## H4 secundária

A informação mútua do ensemble será maior em dados deslocados que na validação in-domain.

## H5 secundária

A randomização de fase afetará modelos temporal/phase-aware mais fortemente que o classificador visual.

# 5. Dados

- Gravity Spy v1d1 com splits oficiais;
- painéis harmonizados derivados dos PNGs;
- 230 Q-grams catalogais;
- DeepExtractor reconstructions para experimentos temporais;
- dados O4/O4b apenas como holdout externo;
- injeções simuladas e contrafactuais registrados separadamente.

# 6. Exclusões

Serão excluídos apenas:

- arquivos ausentes/corrompidos;
- imagens cujo painel não puder ser detectado pelo algoritmo preregistrado;
- arrays não finitos;
- exemplos duplicados entre splits;
- dados cuja licença impeça a análise.

Não serão removidos exemplos por “aparência ruim” ou por prejudicarem as métricas.

# 7. Pré-processamento fixado

## Gravity Spy

1. detecção automática da maior região escura contígua;
2. remoção de eixos, títulos e colorbar;
3. recorte log-frequency de 20–512 Hz, condicionado à confirmação da banda original;
4. resize $140\times170$;
5. grayscale;
6. normalização usada no treino.

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
- softmax extrema em OOD;
- queda $>30\%$ em holdout;
- predição dominada por detector/renderer sem relação instrumental independente.

# 15. Política de transparência

Todos os resultados serão reportados, inclusive falhas. O pipeline antigo será preservado como baseline. Mudanças de preprocessing gerarão novos manifests e não sobrescreverão resultados v8.3.
