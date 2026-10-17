# Política de alegações científicas

## Permitido com os resultados atuais

- o modelo tem forte validação interna no dataset Gravity Spy usado;
- os 475 Q-grams catalogais apresentam mudança extrema no espaço latente
  Mahalanobis dos cinco checkpoints;
- as escolhas fechadas forçadas são parcialmente previsíveis por nove
  estatísticas globais da imagem (balanced accuracy 0,769; análise pós-hoc);
- energy e Mahalanobis discordam no domínio externo;
- os dados atuais são compatíveis com mudança de renderer, mudança semântica e
  drift de detector/run, sem permitir atribuição causal entre elas;
- ridge e matched filter são diagnósticos pedagógicos;
- o projeto oferece uma infraestrutura reprodutível para ablações.

## Proibido com os resultados atuais

- “145 canais contêm scattered light”;
- “os eventos foram vetados por glitch”;
- “o matched filter detectou 13 eventos”;
- “o ridge mediu a massa de 140 canais”;
- “discordância implica nova física”;
- “None_of_the_Above é uma nova classe”;
- “alta softmax é alta probabilidade física”.
- “o renderer sozinho causou a rejeição dos 475 canais”;
- “todos os 218 eventos contêm glitches desconhecidos”;
- “o renderer Gravity Spy resolve a mudança de domínio”;
- “os candidatos em domínio não contêm glitch” ou “a CNN identificou os chirps”.

## Permitido sobre o renderer exploratório (10/10/2026)

- a rejeição Mahalanobis dos 475 canais é fortemente sensível à renderização da
  intensidade (comparação exploratória e pareada);
- com a intensidade na escala Gravity Spy, energy e Mahalanobis discordam em
  sentido oposto ao do renderer canônico;
- a classe forçada `Chirp` se associa à SNR de rede publicada (análise pós-hoc,
  Spearman por evento +0,50).

## Linguagem recomendada

- “similaridade morfológica exploratória”;
- “compatibilidade estatística com o domínio de referência”;
- “candidato OOD”;
- “resultado condicionado ao preprocessing e ao modelo”;
- “não constitui veto nem significância de detecção”.

## Nota de correção (26/07/2026)

Resultados catalogais (classificação Gravity Spy, OOD, ensemble/open-world) produzidos por
qualquer cópia deste pacote anterior à correção registrada em `PREREGISTRO.md` seção 16 não
podem ser citados. O bug de eixo em `qgram_inference.py` produzia entradas sem estrutura
física reconhecível; qualquer "candidato OOD" ou classe prevista sobre o catálogo antes da
correção mede o artefato de pré-processamento, não deslocamento de domínio real ou
similaridade morfológica genuína.

A referência OOD formato 3 também foi invalidada porque ajustava Mahalanobis e
calibrava seus percentis na mesma validação. Somente resultados gerados com o
formato 4 — ajuste no treino, calibração na validação e controle no teste — podem
ser citados. O resultado v4 permite afirmar incompatibilidade latente com o
domínio de referência; não permite identificar a causa como renderer, glitch ou
novidade física.

## Regra de robustez

Uma alegação física exige convergência de pelo menos dois tipos independentes de evidência, por exemplo:

- morfologia + canal auxiliar;
- forma temporal + simulação causal;
- coerência de rede + background por time slides;
- posterior robusta + múltiplos modelos de glitch/waveform.
