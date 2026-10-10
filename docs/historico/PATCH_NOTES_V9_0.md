# Patch notes — v9.0 Research

## Objetivo

Transformar a v8.3 em uma plataforma de pesquisa sobre validade externa, open-set recognition, fase e fragilidade sinal–glitch.

## Alterações

- versão 0.9.0;
- novo módulo `gwproject.research`;
- preparação axis-free de painéis Gravity Spy;
- recorte de frequência logarítmico;
- modos de baseline para Q-grams;
- opção de inversão vertical de frequência;
- ensemble de cinco checkpoints;
- entropia, informação mútua, vote fraction e energy score;
- referência OOD por Mahalanobis;
- contrafactuais de fase;
- auditoria de domain shift agrupada por evento;
- preregistro, revisão, hipóteses e política de alegações;
- novos testes funcionais.

## Compatibilidade

Os defaults da função antiga `qgram_to_uint8` permanecem `baseline_mode="none"` e `flip_frequency=false`. O fluxo v9 usa explicitamente as novas opções no config de pesquisa. Resultados v8.3 não são sobrescritos.

## Limitações

- a banda original 10–2048 Hz dos PNGs precisa de confirmação;
- OOD por validação interna não garante detecção de toda forma externa;
- o OOD reference atual é baseado na seed 42;
- ainda não há modelo phase-aware treinado;
- não há inferência física conjunta executada neste pacote.
