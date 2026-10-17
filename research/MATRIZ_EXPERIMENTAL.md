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

Nenhum renderer holdout é válido se usar populações subjacentes distintas nos
dois lados. A comparação primária deve renderizar o mesmo trigger por todas as
cadeias e manter suas variantes no mesmo fold.

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

As ablações não devem ser executadas apenas uma por vez. O confirmatório deve
estimar pelo menos as interações renderer × tipo de amostra, renderer ×
detector/run e renderer × baseline. Se o fatorial completo for proibitivo, usar
fração de resolução documentada, depois de piloto para variância e poder.

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
- o mesmo trigger Gravity Spy nos renderers oficial numérico, painel recortado e
  Q-transform local;
- segmentos off-source renderizados pelas mesmas três cadeias;
- injeções de linhas estacionárias;
- slow scattering simulado;
- chirps CBC simulados;
- glitches temporais do DeepExtractor.

## F. Critério de interpretação

Um modelo só será chamado de “morfologicamente robusto” se superar os controles negativos de estilo e generalizar em renderer holdout. Um modelo só será chamado de “fisicamente informativo” se seus scores se relacionarem a uma variável independente: waveform temporal, canal auxiliar, coerência de rede ou mecanismo simulado.

Os 218 eventos CBC formam um _stress test_ externo, não ground truth de classes
de glitch. Eles devem permanecer completos na análise, mas não substituem
triggers rotulados e off-source.
