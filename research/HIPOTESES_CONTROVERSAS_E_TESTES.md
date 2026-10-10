---
title: "Hipóteses controversas e testes falsificáveis"
date: "25 de julho de 2026"
lang: pt-BR
---

# Princípio

Hipótese controversa não é licença para especulação sem controle. Cada hipótese abaixo inclui mecanismo, observações compatíveis, alternativas, previsão, teste e falsificação.

# H1. As fronteiras Gravity Spy são condicionadas à representação

## Mecanismo

As classes podem ser partições úteis e estáveis dentro de um domínio, mas suas
fronteiras dependem de janela, renderer, detector e run. Isso é mais restrito e
testável do que afirmar que as classes “não existem”.

## Evidências compatíveis

- evolução da taxonomia entre runs;
- classes novas e `None_of_the_Above`;
- interpolação morfológica em modelos generativos;
- diferenças entre detectores;
- rejeição Mahalanobis dos 475 canais catalogais após calibração independente;
- previsibilidade pós-hoc das escolhas fechadas por estatísticas globais.

## Alternativas

- a taxonomia pode ser estável, e o problema local resultar da mudança semântica
  entre triggers de glitch e janelas centradas em CBC;
- a rejeição pode ser fragilidade específica de Mahalanobis;
- classes podem ser estáveis em um domínio bem definido, sem pretensão universal.

## Predição

A estabilidade de classe cairá quando o mesmo trigger atravessar renderers,
runs ou detectores, mais que no split oficial in-domain.

## Falsificação

As classes permanecem separáveis e consistentes para o mesmo trigger em
$h(t)$, magnitude+fase, painéis recortados e runs distintos.

# H2. A fase é uma variável causal oculta

## Mecanismo

Magnitude-Q comprime muitas waveforms diferentes. Fase e coerência temporal distinguem mecanismos que parecem semelhantes em potência.

## Teste

Gerar contrafactuais com magnitude espectral preservada e fase randomizada. Comparar CNN visual, modelo temporal e modelo complexo.

## Predição quantitativa

A estabilidade de classe visual permanecerá alta para $\alpha\to1$, enquanto mismatch temporal e scores phase-aware mudarão fortemente.

## Falsificação

Todos os modelos e métricas físicas permanecem estáveis, sem vantagem phase-aware.

# H3. O nuisance de detector é um sinal científico

## Mecanismo

O embedding codifica microestado, acoplamentos e configurações do detector. Remover identidade pode apagar informação de detector characterization.

## Teste

Decompor embedding em $z_m$ (morfologia), $z_d$ (detector) e $z_s$ (estado temporal). Avaliar previsão de run, detector e changepoints.

## Predição

$z_s$ antecipa mudanças de taxa de glitches ou sensibilidade.

## Falsificação

A informação desaparece após harmonização e não antecipa nenhum indicador independente.

# H4. A discordância é um observável de física omitida

## Mecanismo

Diferentes métodos falham de modos complementares. A geometria das discordâncias distingue glitch, baixa SNR e waveform inadequada.

## Teste

Injeções fatoriais: circular/precessing/eccentric/higher modes × sem glitch/com glitch × SNR × detector.

## Predição

Um classificador de $\mathbf D$ separa física omitida de glitch acima de um baseline que usa apenas massa e SNR.

## Falsificação

O ganho desaparece depois de controlar variáveis triviais.

# H5. Precessão e outros parâmetros sutis são propriedades conjuntas da análise

## Mecanismo

A conclusão depende de $(W,G,P,C,T)$: waveform, glitch, PSD, calibração e janela.

## Teste

Grade de modelos e model averaging. Medir divergência de Jensen–Shannon entre posteriores e mudança de Bayes factor.

## Predição

Eventos com maior fragilidade local terão maior variabilidade posterior.

## Falsificação

Parâmetros e evidências são estáveis em toda a grade plausível.

# H6. `None_of_the_Above` contém múltiplas famílias estáveis

## Teste

Clustering multiview e phase-aware, bootstrap, estabilidade por seed, validação humana cega e correlação com canais auxiliares.

## Critério forte

Um candidato a nova família requer recorrência, cluster estável, assinatura independente e ausência de explicação por renderer.

# H7. Populações de glitches são um sistema dinâmico com transições de fase

## Mecanismo

O detector alterna entre microestados. Taxas e embeddings funcionam como parâmetros de ordem observacionais.

## Teste

HMM, changepoint Bayesiano e state-space models em séries temporais longas.

## Predição

Changepoints precedem degradação ou intervenção.

# H8. Uma minoria de glitches aparentes pode ser coerente

## Controle anti-sensacionalismo

- amostra off-source;
- time slides;
- background empírico;
- blind injections;
- canais auxiliares;
- BayesWave/GWAK;
- resposta de antena.

A ausência de classe não é evidência de sinal.

# H9. O ridge mapeia identificabilidade, não massa

## Produto

Diagrama de regimes em $(M_{\rm det},q,\rho,T_{\rm inspiral})$ indicando método apropriado: ridge, IMR, wavelet ou não identificável.

# H10. A coerência pode substituir rótulos humanos em parte do treino

## Teste

Aprendizado contrastivo com pares de detectores alinhados, negativos por time slides e hard negatives de glitches coincidentes.

# Priorização

| Hipótese | Novidade | Viabilidade | Risco | Prioridade |
|---|---:|---:|---:|---:|
| H1 representação, pareada pelo mesmo trigger | moderada/alta | alta | moderado | 1 |
| H4 discordância, com injeções e off-source | moderada/alta | alta | moderado | 2 |
| H9 identificabilidade do ridge | moderada | moderada | moderado | 3 |
| H2 fase | moderada | moderada | alto | 4 |
| H5 fragilidade | alta | baixa | alto | 5 |
| H3 microestados | moderada/alta | baixa/moderada | alto | 6 |
| H6 open-world/nova classe | baixa/moderada | moderada | alto | 7 |
| H7 prognóstico | alta | baixa | alto | 8 |
| H8 coerência anômala | moderada | baixa | muito alto | 9 |

As notas de novidade são condicionais a uma busca de escopo dirigida, não a uma
revisão sistemática. H6 foi rebaixada porque trabalhos recentes já cobrem
similarity learning, clustering multitemporal e DINOv2 em O4a, incluindo
resultado nulo para novas morfologias no conjunto analisado.
