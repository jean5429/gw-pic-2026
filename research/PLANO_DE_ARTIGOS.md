# Plano de artigos

## Manuscrito 1 — prioridade máxima

### Título

*Renderer shortcuts and domain shift in gravitational-wave glitch classification*

### Claim central

Acurácia in-domain não estabelece validade cross-renderer. O efeito causal do
renderer será estimado em comparações pareadas da mesma observação, separando-o
de mudança semântica, detector/run e método OOD.

### Figuras

1. pipeline e pontos de ruptura;
2. folha de contato;
3. probe de pixels;
4. matriz de holdouts;
5. calibration/OOD;
6. ablações.

Os 218 eventos entram como _stress test_ externo completo. O claim principal
depende adicionalmente de triggers Gravity Spy e off-source renderizados por
todas as cadeias; comparar apenas PNGs de glitches com Q-grams de CBC não basta.

### Resultado publicável mesmo se negativo

Se a harmonização resolver o problema no desenho pareado, o artigo identifica
um mecanismo de renderer e fornece protocolo. Se não resolver, o artigo só
poderá atribuir insuficiência à representação depois que população e tipo de
amostra forem mantidos fixos.

## Manuscrito 2 — sinal versus glitch open-world

### Título

*Open-world signal–glitch discrimination across representations and observing domains*

### Claim

Injeções CBC, glitches off-source e triggers rotulados permitem medir separadamente
mudança semântica e mudança de renderer. Esta linha deve ser diferenciada
explicitamente de GSpyNetTree e GSpyNetTreeS.

## Manuscrito 3 — fase e domínio temporal

### Título

*Phase blindness and physical identifiability of gravitational-wave detector glitches*

### Claim

Magnitude-Q não é suficiente para validar plausibilidade temporal; phase-aware models melhoram identificação ou rejeição.

## Manuscrito 4 — longo prazo

### Título

*A signal–glitch fragility atlas for compact-binary inference*

### Claim

A robustez de parâmetros sutis deve ser reportada sob famílias de waveform e glitch.

## Possíveis periódicos

- Classical and Quantum Gravity;
- Physical Review D;
- Machine Learning: Science and Technology;
- Journal of Open Source Software para o pacote, se houver documentação e testes suficientes;
- Data in Brief/Scientific Data apenas se o benchmark puder ser licenciado e curado adequadamente.

Um artigo autônomo do matched filter local ou das classes forçadas nos 218
eventos não é recomendado. Um artigo de ridge só se torna defensável após mapa
cego de identificabilidade por injeções.
