# Matriz experimental e ablações

## A. Representações

| ID | Representação | Informação preservada | Risco principal |
|---|---|---|---|
| R1 | PNG completo | estilo + painel | shortcut gráfico |
| R2 | painel recortado | magnitude renderizada | renderer/colormap |
| R3 | Q harmonizado | magnitude numérica | perda de fase |
| R4 | magnitude+fase | Q complexo | complexidade/calibração |
| R5 | strain temporal | waveform | sensibilidade a PSD/whitening |

## B. Holdouts

| ID | Holdout | Pergunta |
|---|---|---|
| V1 | split oficial | capacidade interna |
| V2 | temporal | drift |
| V3 | detector | generalização instrumental |
| V4 | run | generalização histórica |
| V5 | renderer | dependência visual |

## C. Ablations preregistradas

1. eixos presentes/ausentes;
2. colorbar presente/ausente;
3. banda 20–512 versus banda completa;
4. frequência linear/log;
5. frequência alta no topo/baixo;
6. baseline none/subtract-one/global/per-frequency;
7. percentis 0–100, 1–99,5 e normalização global;
8. 1 s versus quatro escalas temporais;
9. grayscale versus colormap original;
10. uma seed versus ensemble.

## D. Negative controls

- rótulos permutados;
- imagens com painel mascarado e margens preservadas;
- margens preservadas e painel randomizado;
- phase randomization;
- frequency shuffling;
- time reversal;
- time slides entre detectores;
- ruído gaussiano colorido pela PSD.

## E. Positive controls

- exemplos Gravity Spy de alta confiança in-domain;
- injeções de linhas estacionárias;
- slow scattering simulado;
- chirps CBC simulados;
- glitches temporais do DeepExtractor.

## F. Critério de interpretação

Um modelo só será chamado de “morfologicamente robusto” se superar os controles negativos de estilo e generalizar em renderer holdout. Um modelo só será chamado de “fisicamente informativo” se seus scores se relacionarem a uma variável independente: waveform temporal, canal auxiliar, coerência de rede ou mecanismo simulado.
