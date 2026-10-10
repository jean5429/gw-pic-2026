---
title: "Hipóteses controversas e testes falsificáveis"
date: "25 de julho de 2026"
lang: pt-BR
---

# Princípio

Hipótese controversa não é licença para especulação sem controle. Cada hipótese abaixo inclui mecanismo, observações compatíveis, alternativas, previsão, teste e falsificação.

# H1. As classes Gravity Spy não são espécies naturais

## Mecanismo

As classes seriam partições úteis de um manifold contínuo de estados instrumentais e representações. As fronteiras surgiriam de decisões humanas, janelas e renderer.

## Evidências compatíveis

- evolução da taxonomia entre runs;
- classes novas e `None_of_the_Above`;
- interpolação morfológica em modelos generativos;
- diferenças entre detectores;
- colapso local sob mudança de renderer.

## Alternativas

- a taxonomia pode ser fisicamente estável, e o problema local ser apenas preprocessing incorreto;
- classes podem ser estáveis em um domínio bem definido, sem pretensão universal.

## Predição

A estabilidade de classe cairá em renderer/run/detector holdout mais que em split aleatório.

## Falsificação

As classes permanecem separáveis e consistentes em $h(t)$, magnitude+fase, painéis recortados e runs distintos.

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
| H1 representação | alta | alta | moderado | 1 |
| H2 fase | muito alta | moderada | moderado | 2 |
| H5 fragilidade | muito alta | baixa/moderada | alto | 3 |
| H3 microestados | muito alta | moderada | alto | 4 |
| H4 discordância | alta | alta com injeções | moderado | 5 |
| H6 open-world | alta | moderada | alto | 6 |
| H7 prognóstico | muito alta | baixa | alto | 7 |
| H8 coerência anômala | alta | baixa | muito alto | 8 |
