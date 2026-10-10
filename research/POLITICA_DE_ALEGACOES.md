# Política de alegações científicas

## Permitido com os resultados atuais

- o modelo tem forte validação interna no dataset Gravity Spy usado;
- a transferência v8.3 apresenta domain shift severo;
- os rótulos transferidos são altamente previsíveis por estatísticas de baixo nível;
- a softmax está sobreconfiante para o domínio externo;
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

## Linguagem recomendada

- “similaridade morfológica exploratória”;
- “compatibilidade estatística com o domínio de referência”;
- “candidato OOD”;
- “resultado condicionado ao preprocessing e ao modelo”;
- “não constitui veto nem significância de detecção”.

## Regra de robustez

Uma alegação física exige convergência de pelo menos dois tipos independentes de evidência, por exemplo:

- morfologia + canal auxiliar;
- forma temporal + simulação causal;
- coerência de rede + background por time slides;
- posterior robusta + múltiplos modelos de glitch/waveform.
