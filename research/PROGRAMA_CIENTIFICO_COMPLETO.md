---
title: "Programa científico completo: representações, glitches e fragilidade da inferência em ondas gravitacionais"
author: "Projeto GW Open-World Robustness"
date: "25 de julho de 2026"
lang: pt-BR
---

# Sumário executivo

Este documento transforma os resultados do projeto `ondas_gravitacionais_v7/v8.3` em um programa científico falsificável. O ponto de partida não é a tentativa de salvar retrospectivamente as previsões do classificador Gravity Spy. O ponto de partida é o fato empírico de que uma rede com excelente desempenho interno falhou de maneira altamente estruturada quando transferida para Q-grams produzidos por outro pipeline.

A observação central é:

- cinco sementes da ResNet18 apresentaram macro-$F_1$ médio de aproximadamente $0{,}939$ no teste Gravity Spy;
- a aplicação aos 230 canais do catálogo usou somente duas das 22 classes: 145 `Scattered_Light` e 85 `None_of_the_Above`;
- 60 dos 90 eventos tiveram discordância entre detectores;
- a mediana da confiança softmax foi aproximadamente $0{,}9994$;
- nove estatísticas escalares do Q-gram bruto reproduzem os rótulos transferidos com balanced accuracy agrupada de $0{,}910$ e ROC-AUC de $0{,}964$;
- apenas o percentil inferior $p_{01}$ alcança balanced accuracy agrupada de $0{,}884$ e $|d|\approx2{,}35$.

Esses resultados mostram que a classificação transferida é dominada por propriedades de representação de baixo nível. Eles não mostram que 145 canais contêm scattered light. O resultado negativo, entretanto, sustenta uma pergunta científica relevante: **quais propriedades de uma representação tempo–frequência são invariantes, causalmente informativas e suficientes para distinguir sinal, glitch e estado instrumental?**

O programa propõe cinco eixos principais:

1. **invariância de representação e shortcut learning**;
2. **fase e identificabilidade física dos glitches**;
3. **open-set recognition e detecção de OOD**;
4. **coerência de rede e disentanglement detector–morfologia–estado**;
5. **fragilidade conjunta da inferência astrofísica ao waveform e ao modelo de glitch**.

![Mapa das hipóteses controversas](figures/mapa_hipoteses_controversas.png)

# 1. Estado científico do projeto

## 1.1 Infraestrutura validada

A infraestrutura de aquisição e processamento contém:

- 90 eventos GWTC;
- 270 combinações evento–detector;
- 230 canais aprovados;
- arquivos brutos, séries processadas e Q-grams auditados;
- redshifts e massas detector-frame para todos os 90 eventos;
- manifestos de ambiente e configuração;
- testes funcionais e execução em CUDA.

Essa camada constitui um ativo científico independente. Ela permite reconstruir experimentos, registrar provenance e realizar ablações controladas.

## 1.2 Ridge

O ridge temporal foi útil para localizar regimes em que uma trajetória inspiral é ou não mensurável. Não foi validado como estimador catalogal de chirp mass. Na v8.3:

- 140 de 230 canais produziram uma estimativa;
- 60 foram classificados como `insufficient_inspiral_band`;
- 25 como `flat_ridge`;
- cinco como `insufficient_frequency_span`;
- o erro relativo mediano permaneceu em aproximadamente 52,2%;
- somente 36 estimativas ficaram abaixo de 30% de erro.

Uso permitido: diagnóstico tempo–frequência, estudo de identificabilidade e construção de um diagrama de regimes metodológicos.

Uso não permitido: estimativa de massa competitiva com inferência Bayesiana ou pipelines LVK.

## 1.3 Matched filtering pedagógico

O matched filter local foi construído ao redor das massas catalogadas. Na v8.3:

- 85 eventos foram processados;
- cinco ficaram sem template ativo;
- 30 passaram o limiar de SNR;
- 38 passaram a margem top-2;
- 13 passaram ambos os gates;
- somente cinco tiveram o template central em primeiro lugar.

O procedimento não possui background, false-alarm rate, time slides, teste $\chi^2$, reweighted SNR, busca cega ou trials factor. Portanto, mede compatibilidade local dentro de um banco pequeno; não mede significância de detecção.

## 1.4 Gravity Spy

A validação interna multiseed foi forte e estável. O problema está na validade externa. A folha de contato revelou diferenças profundas entre:

- PNGs Gravity Spy completos, com margens, eixos, textos e colorbar;
- Q-grams numéricos do catálogo ocupando todo o frame;
- faixas de frequência e escalas de intensidade distintas;
- transformação do piso normalizado pela mediana;
- possível inversão vertical entre a ordem das frequências no array e a convenção visual de alta frequência no topo.

![Entradas efetivas do modelo antigo](figures/model_input_contact_sheet.png)

# 2. Pergunta-mãe

> Quais características aprendidas por classificadores de glitches são invariantes à representação e correspondem a propriedades físicas do transiente ou do detector, e quais são atalhos específicos do renderer, do run ou do instrumento?

A pergunta envolve quatro níveis que devem ser separados:

1. **morfologia observada:** aparência em uma representação escolhida;
2. **forma de onda:** estrutura temporal, espectral e de fase;
3. **causa instrumental:** mecanismo físico que gerou o glitch;
4. **efeito inferencial:** viés produzido sobre parâmetros astrofísicos.

Uma classe visual pode ser útil sem ser uma espécie ontológica. Uma morfologia pode ter várias causas. Uma causa pode produzir várias morfologias, dependendo do estado do detector. Um rótulo com alta softmax não identifica necessariamente nenhuma dessas entidades.

# 3. Hipóteses centrais

## H1 — A taxonomia é dependente da representação

**Mecanismo.** As fronteiras entre classes Gravity Spy são parcialmente definidas por escolhas de janela temporal, faixa de frequência, colormap, normalização, elementos gráficos e processo histórico de rotulagem. A rede aprende uma combinação de morfologia e estilo do dataset.

**Predição principal.** O rótulo de uma fração não desprezível dos exemplos mudará sob transformações fisicamente equivalentes: remoção de eixos, mudança de renderer, equalização de intensidade ou mudança entre magnitude-Q e strain temporal.

**Falsificação.** A hipótese enfraquece se as classes permanecerem estáveis sob renderer holdout, detector holdout, run holdout e transformações que preservam a informação física.

## H2 — A fase é necessária para identificabilidade física

**Mecanismo.** A magnitude do Q-transform não determina unicamente a forma de onda. Pares com magnitude semelhante podem ter fases, coerência e mecanismos físicos distintos.

**Predição principal.** Randomização controlada da fase preservará parte da aparência espectral, mas destruirá a consistência para modelos temporal/phase-aware. Um classificador visual poderá manter alta confiança mesmo quando a forma de onda se tornar fisicamente implausível.

**Falsificação.** Modelos phase-aware não melhoram generalização, OOD ou validação causal, e a randomização de fase não altera tarefas fisicamente motivadas.

## H3 — O domain shift contém informação instrumental real

**Mecanismo.** Algumas características dependentes do detector não são meros nuisances. Elas podem codificar acoplamentos, configurações de controle ou microestados instrumentais.

**Predição principal.** O embedding permitirá prever detector, run e período temporal mesmo após controlar classe e SNR. Uma decomposição explícita poderá separar conteúdo morfológico, identidade do detector e estado latente.

**Falsificação.** A informação de detector desaparece após harmonização de renderer e não melhora nenhuma tarefa de detector characterization.

## H4 — A discordância entre métodos é um observável de fragilidade

Defina

$$
\mathbf D=(D_{\rm ridge},D_{\rm template},D_{\rm detector},D_{\rm class},D_{\rm residual},D_{\rm waveform}).
$$

**Mecanismo.** Física omitida e glitches sobrepostos deixam assinaturas diferentes de inconsistência entre detectores, modelos e representações.

**Predição principal.** Injeções com precessão, excentricidade, modos superiores ou glitches produzirão distribuições distintas de $\mathbf D$ depois de controlar massa e SNR.

**Falsificação.** $\mathbf D$ é totalmente explicado por massa total, SNR, duração e número de detectores.

## H5 — Propriedades astrofísicas sutis são conjuntas ao modelo de sinal e ruído

**Mecanismo.** A posteriori de spins, precessão, excentricidade ou desvios de GR depende do waveform, do modelo de glitch, da PSD, da calibração e da janela. Essa dependência pode ser forte em eventos contaminados.

**Predição principal.** Eventos com alta fragilidade no vetor $\mathbf D$ também mostrarão maior variação posterior entre combinações $(W_i,G_j)$ de waveform e modelo de glitch.

**Falsificação.** Parâmetros permanecem estáveis sob uma família razoável de waveforms, PSDs, janelas e modelos de glitch.

# 4. Hipóteses secundárias de alto risco

## H6 — `None_of_the_Above` é uma mistura de estados, não uma classe

Clusters estáveis dentro de `None_of_the_Above` podem corresponder a novas famílias, misturas, baixa SNR, mudanças de run ou entradas OOD.

## H7 — Populações de glitches antecipam transições do detector

Mudanças no espaço de embeddings podem preceder degradação de sensibilidade, aumento de falsos alarmes ou intervenções de commissioning.

## H8 — Alguns transientes classificados como glitches são coerentes e não modelados

Essa hipótese não autoriza alegações de nova física. Ela exige coerência de rede, time slides, ausência de correlação auxiliar e reconstrução por modelos agnósticos.

## H9 — A falha do ridge define um diagrama de regimes observacionais

O sucesso ou fracasso do ridge pode mapear a transição entre regime inspiral mensurável, regime IMR e regime burst/wavelet.

## H10 — Coerência de rede pode fornecer auto-supervisão

Pares de detectores do mesmo evento, depois de correções geométricas, podem fornecer positivos contrastivos; janelas aleatórias e glitches locais fornecem negativos.

# 5. Matriz experimental

![Matriz experimental](figures/matriz_experimental_5x5.png)

As representações são:

1. PNG Gravity Spy completo;
2. PNG recortado para o painel físico;
3. magnitude $|Q|$ harmonizada;
4. Q-transform complexo ou magnitude+fase;
5. strain temporal $h(t)$.

Os regimes de validação são:

1. split oficial/aleatório;
2. holdout temporal;
3. detector holdout;
4. observing-run holdout;
5. renderer holdout.

Cada célula deve registrar:

- macro-$F_1$ e balanced accuracy;
- ECE e Brier score;
- OOD AUROC, AUPR e FPR@95TPR;
- predictability do detector;
- estabilidade entre sementes;
- estabilidade sob contrafactuais;
- custo computacional;
- intervalo de confiança por bootstrap agrupado.

# 6. Experimentos prioritários

## E1 — Auditoria de atalhos

**Objetivo.** Medir quanto dos rótulos pode ser previsto por características de baixo nível.

**Controles.** Split agrupado por evento; regressão logística; thresholds univariados treinados apenas no fold de treino; permutação de rótulos; bootstrap por evento.

**Resultado atual.** O probe de nove estatísticas atingiu balanced accuracy $0{,}910$ e AUC $0{,}964$. Isso é evidência contra uma interpretação puramente morfológica das previsões antigas.

![Separação por p01](figures/p01_class_separation.png)

![ROC do probe](figures/pixel_probe_roc.png)

## E2 — Harmonização do renderer

1. detectar automaticamente o painel escuro nos PNGs;
2. remover margens, títulos, eixos, ticks e colorbar;
3. recortar a banda 20–512 Hz supondo eixo logarítmico;
4. preservar ou documentar a convenção de frequência alta no topo;
5. aplicar a mesma geometria $140\times170$;
6. treinar cinco sementes novamente.

**Alerta.** A faixa original 10–2048 Hz inferida dos PNGs deve ser confirmada contra metadados oficiais. O código novo marca essa suposição no manifesto.

## E3 — Transformação do Q-gram

Comparar:

- `none`: $\log(1+Q)$;
- `subtract_one`: $\log[1+\max(Q-1,0)]$;
- `global_median`: $\log[1+\max(Q/{\rm med}(Q)-1,0)]$;
- `per_frequency_median`: $\log[1+\max(Q/{\rm med}_t Q-1,0)]$.

A última opção é o default do programa v9, pois o Q-transform foi normalizado pela mediana e o objetivo é representar excesso relativo de energia.

## E4 — Inversão vertical

A ordem de frequências no array deve ser auditada. Arrays numéricos normalmente armazenam a frequência em ordem crescente; uma imagem direta coloca a primeira linha no topo, o que pode inverter o eixo físico em relação ao renderer científico. O programa v9 inclui `flip_frequency=true` como hipótese explícita. Deve ser validada por injeção de linha em frequência conhecida.

## E5 — Ensemble e open-set

- média das probabilidades de cinco sementes;
- entropia preditiva;
- entropia esperada;
- informação mútua;
- fração de votos;
- energy score;
- distância de Mahalanobis no embedding;
- categorias `in_domain_candidate`, `review`, `reject_ood`.

Nenhuma dessas categorias é uma probabilidade causal de glitch.

## E6 — Contrafactuais de fase

Para cada waveform $h(t)$, gerar

$$
\tilde h_\alpha(t)=\mathcal F^{-1}\left[|H(f)|e^{i(\phi(f)+\alpha\Delta\phi(f))}\right],
$$

com $\alpha\in\{0,0.25,0.5,0.75,1\}$. Medir estabilidade de classe, embedding e confiança.

## E7 — Coerência de rede

Treinar uma representação contrastiva condicionada por detector e geometria. Positivos devem ser pares fisicamente associados; negativos incluem time slides e janelas off-source.

## E8 — Inferência conjunta sinal–glitch

Começar por GW200129 e GW191109, usando `bilby_glitch`, wavelets, chirplets e slow scattering, além de mais de um approximant. O produto não deve ser uma afirmação binária sobre precessão, mas um mapa de robustez da conclusão à escolha de modelos.

# 7. Dados e recursos

## 7.1 Recursos já disponíveis

- catálogo local de 90 eventos e 230 canais;
- dataset Gravity Spy v1d1;
- cinco checkpoints ResNet18;
- resultados de ridge e matched filter;
- CSVs de auditoria e Q-grams.

## 7.2 Recursos externos prioritários

- reconstruções temporais DeepExtractor no Hugging Face: 35 mil waveforms de 2 s, 4096 Hz, sete classes;
- modelos ViT/ResNet externos como baseline independente, com ressalva de que os resultados da model card são auto-relatados;
- dados públicos O4/O4b para holdout temporal e de run;
- modelos de glitches BayesWave publicados para eventos selecionados;
- GWOSC como fonte primária de strain e metadados.

## 7.3 Estratégia de uso do Hugging Face

Usar `datasets` em modo streaming para conjuntos grandes. O dataset O4b V1 em Parquet possui escala de dezenas de GB e mais de um bilhão de linhas; deve ser consultado por segmentos e projeção de colunas, não carregado integralmente em memória.

# 8. Critérios de sucesso e de parada

## Sucesso mínimo da harmonização

- macro-$F_1$ interno $\ge0{,}85$;
- ECE $\le0{,}08$;
- balanced accuracy do probe de pixels $\le0{,}70$;
- OOD AUROC $\ge0{,}80$ em conjuntos sintéticos e reais definidos a priori;
- nenhuma classe dominante acima de 80% no catálogo sem justificativa independente;
- estabilidade de classe sob pequenas mudanças de contraste;
- auditoria visual cega com concordância interavaliadores.

## Critérios de parada

Interromper a interpretação física se:

- o renderer puder prever a classe melhor que a morfologia;
- a validação cair drasticamente em detector/run holdout;
- OOD não for separável de in-domain;
- a classe mudar sob transformações fisicamente equivalentes;
- a conclusão depender de thresholds escolhidos após observar o catálogo.

# 9. Produtos científicos

## Artigo 1 — Representação e shortcut learning

**Título provisório:** *When a glitch image stops being a physical representation: renderer shortcuts and domain shift in gravitational-wave transient classification*.

Contribuição: benchmark controlado de renderers, probe de baixo nível, detector/run/renderer holdout e guidelines de validade externa.

## Artigo 2 — Fase e identificabilidade

**Título provisório:** *Phase blindness in gravitational-wave glitch classification*.

Contribuição: contrafactuais que preservam magnitude espectral, comparação entre modelos visuais, complexos e temporais.

## Artigo 3 — Fragilidade sinal–glitch

**Título provisório:** *A signal–glitch fragility atlas for compact-binary inference*.

Contribuição: marginalização sobre famílias de waveform e glitch; índice de fragilidade; estudos de caso.

## Recurso aberto

- benchmark de domínio;
- manifests e splits;
- scripts de harmonização;
- Q-grams e embeddings permitidos por licença;
- model cards;
- datasheets;
- resultados negativos.

# 10. Roteiro

![Roteiro do programa](figures/roadmap_programa.png)

## Fase 0 — concluída no pacote v9

- congelamento do baseline v8.3;
- auditoria agrupada dos atalhos;
- código para harmonização de painéis;
- transformações de baseline e inversão de frequência;
- ensemble e métricas OOD;
- contrafactuais de fase;
- testes unitários;
- preregistro e política de alegações.

## Fase 1 — execução local necessária

- preparar o dataset harmonizado;
- inspecionar amostra visual;
- retreinar cinco sementes;
- construir referência OOD;
- classificar com ensemble;
- repetir auditoria de pixels.

## Fase 2 — extensão científica

- adicionar strain temporal e reconstruções DeepExtractor;
- implementar magnitude+fase;
- renderer/run/detector holdouts;
- auto-supervisão de coerência.

## Fase 3 — inferência astrofísica

- estudos GW200129 e GW191109;
- atlas de fragilidade;
- validação cruzada com BayesWave e `bilby_glitch`.

# 11. Conclusão

O projeto deve abandonar a meta estreita de produzir mais rótulos Gravity Spy e adotar uma meta científica mais forte: **determinar quais conclusões permanecem invariantes quando a representação, o detector, o run e o modelo de ruído mudam**.

A hipótese mais controversa, mas rigorosa, é que uma parte do que chamamos de “classe de glitch” e uma parte do que chamamos de “propriedade sutil da fonte” são propriedades conjuntas do sistema de medição, da representação e do modelo inferencial. O programa v9 não assume que isso seja verdadeiro; ele constrói testes capazes de demonstrar quando é verdadeiro e quando é falso.
