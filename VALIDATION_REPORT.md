# Relatório final de reconstrução e validação — GWProject 2.0

Data da execução: 27 de julho de 2026.

## Parecer executivo

O projeto foi reorganizado e reexecutado sobre o escopo canônico de 218 eventos.
A engenharia de dados, a reprodutibilidade do treinamento e os controles internos
da CNN ficaram em condição defensável para auditoria. Entretanto, o resultado
principal de validade externa é negativo: todos os 475 Q-grams dos eventos foram
rejeitados como fora do domínio Gravity Spy pela geometria Mahalanobis dos cinco
modelos. Assim, o projeto não sustenta classificação física de glitches nos
eventos GWTC-4. O resultado demonstra mudança latente robusta, mas ainda não
atribui causalmente a falha ao renderer, pois a população também muda de triggers
de glitch para janelas centradas em eventos CBC.

O ridge e o matched filter foram preservados como diagnósticos condicionados ao
catálogo. Seus resultados não sustentam medição cega de massa, detecção,
false-alarm rate ou confirmação independente de parâmetros publicados.

## Escopo catalogal

O catálogo combina os 90 eventos até o GWTC-3 com os registros do GWTC-4.0 e
exclui `GW230630_070659`, marcado pelo release como evidência de origem
instrumental. O contrato de execução exige exatamente 218 nomes únicos antes de
qualquer download.

Fontes de escopo:

- [GWTC-4.0 Data Release](https://gwosc.org/GWTC-4.0/);
- [GWOSC Event API — GWTC-4.0](https://gwosc.org/eventapi/html/GWTC-4.0/);
- [GWTC-4.0 catalog paper](https://arxiv.org/abs/2508.18082).

Resultados da aquisição e qualidade:

| Item | Resultado |
|---|---:|
| Eventos únicos | 218 |
| Combinações evento–detector indexadas | 654 |
| Canais brutos disponíveis | 501 |
| Canais processados e com Q-gram aprovado | 475 |
| Eventos com rede de ao menos dois detectores | 204 |
| Eventos com um detector aprovado | 14 |
| Eventos inutilizáveis | 0 |
| Dados públicos indisponíveis | 153 canais |
| Rejeições por não finitude | 21 canais |
| Rejeições no whitening | 5 canais |
| Eventos com massas detector-frame disponíveis | 176 |
| Eventos sem estimação de parâmetros publicada | 42 |

## Gravity Spy: limpeza e independência

Os 7.966 registros da metadata oficial foram associados às imagens pela chave
composta `(ifo, gravityspy_id)`. Isso elimina a colisão real observada para um ID
presente nos dois detectores. Todos os painéis foram recortados pelo bounding box
oficial e redimensionados para `140 x 170`.

O controle de qualidade excluiu 159 renderizações vazias; 139 delas formavam um
grupo de duplicatas exatas atravessando splits. Restaram 7.807 amostras, sem
falha de leitura e com todas as 22 classes presentes em treino, validação e
teste:

- treino: 5.472;
- validação: 1.177;
- teste: 1.158.

O treinamento canônico agora falha cedo se os manifestos de QC estiverem
ausentes, se houver falhas de preprocessamento ou se a metadata não coincidir
com o hash aprovado.

## CNN multiseed

Foram treinadas cinco ResNet-18 em GPU, com determinismo estrito, pesos de
classe, fine-tuning em duas fases, early stopping, baseline linear, calibração
por temperatura e 2.000 réplicas bootstrap por semente.

| Métrica no teste | Média | Desvio-padrão | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| Macro-F1 | 0,9449 | 0,0092 | 0,9337 | 0,9570 |
| Acurácia balanceada | 0,9459 | 0,0116 | 0,9325 | 0,9605 |
| Kappa | 0,9610 | 0,0041 | 0,9561 | 0,9646 |
| Acurácia | 0,9648 | 0,0037 | 0,9603 | 0,9680 |

Todos os checkpoints passaram os gates de macro-F1, acurácia balanceada, kappa,
F1 mínimo por classe e ECE calibrado. A baseline linear teve macro-F1 0,7963;
a vantagem da CNN variou de 0,1374 a 0,1607. Cada checkpoint contém
temperatura, estados do otimizador/scheduler/scaler, contrato da entrada e
hashes dos dados.

## Validade externa e open-world

A referência OOD formato 4 foi construída separadamente para cada checkpoint.
As médias e a matriz de precisão Mahalanobis foram ajustadas nas 5.472 amostras
de treino; energia e distância foram calibradas pelas 1.177 amostras de
validação; as 1.158 imagens de teste permaneceram como controle independente.
Os checkpoints são vinculados às referências por SHA-256.

A referência anterior, formato 3, foi invalidada: ela ajustava e calibrava
Mahalanobis nas mesmas amostras de validação. No teste, seu limiar nominal de 99%
rejeitava 26–30% das imagens in-domain. Com o formato 4, as taxas Mahalanobis
acima de 95% ficaram entre 3,4% e 5,4%, e as taxas acima de 99% entre 0,4% e
1,2%, comportamento compatível com os quantis nominais.

Nos 475 Q-grams dos 218 eventos:

- `reject_ood`: 475;
- `review`: 0;
- `in_domain_candidate`: 0;
- classes reportáveis: 0.

As escolhas fechadas forçadas foram 244 `None_of_the_Above`, 230
`Extremely_Loud` e 1 `Repeating_Blips`, mas permanecem apenas como diagnóstico.
Todos os canais excederam o percentil 99 de Mahalanobis para todos os modelos.
O energy score quase nunca rejeitou esses mesmos canais. Essa discordância
impede apresentar `reject_ood` como consenso universal entre métodos, mas
confirma que a CNN treinada nos painéis Gravity Spy não pode ser usada como
classificador válido dos Q-grams GWOSC produzidos pelo pipeline atual.

Uma auditoria pós-hoc das nove estatísticas globais das imagens efetivamente
apresentadas à CNN reproduziu `None_of_the_Above` versus as demais escolhas
fechadas com balanced accuracy 0,769 e AUROC 0,817, usando folds agrupados por
evento. O resultado evidencia dependência de propriedades globais, sem
identificar qual atalho foi usado.

## Ridge catálogo-condicionado

A monotonicidade deixou de ser imposta por `maximum.accumulate`. A massa
publicada continua condicionando a janela e a banda, logo o resultado é
explicitamente diagnóstico.

| Item | Resultado |
|---|---:|
| Canais solicitados | 475 |
| Ajustes aceitos | 201 |
| Eventos com estimativa em rede | 75 |
| Eventos com estimativa em um detector | 40 |
| Eventos sem estimativa | 103 |
| Erro relativo mediano contra o catálogo condicionante | 34,7% |
| Erro relativo médio | 40,1% |
| Percentil 90 do erro | 78,8% |
| Ajustes dentro de 30% | 90/201 |

As principais falhas foram banda inspiral insuficiente (123 canais), metadados
de massa ausentes (81) e inclinação não física (68). Esses números são
incompatíveis com uma alegação de estimador de massa validado.

## Matched filter local

A PSD passou a ser estimada off-source, excluindo um segundo em torno do evento.
O gate exige SNR, margem sobre o segundo template, coincidência entre
detectores em 30 ms e, no gate completo, vitória do template central.

| Item | Resultado |
|---|---:|
| Eventos solicitados | 218 |
| Eventos com banco ativo e análise concluída | 164 |
| Eventos sem template ativo | 12 |
| Eventos sem massas detector-frame | 42 |
| Template central vencedor | 9 |
| SNR local acima do limiar | 61 |
| Margem top-2 aprovada | 65 |
| Coincidência aprovada | 34 |
| Gate local SNR+margem+coincidência | 7 |
| Gate completo | 0 |

Sem background, FAR, chi-quadrado e busca cega, esses resultados não constituem
detecção. O fato de nenhum evento passar o gate completo também mostra que o
banco TaylorF2 local não reproduz de forma confiável a ordenação catalogal.

## Integridade e reprodutibilidade

Uma auditoria exaustiva recalculou os hashes de:

- 501 arquivos brutos;
- 475 séries processadas;
- 475 PSDs;
- 475 Q-grams;
- cinco checkpoints;
- referência OOD e tabelas finais.

Não foi encontrada divergência entre arquivo e hash registrado. Os CSVs finais
contêm 218 eventos no ensemble, ridge e matched filter, e 475 canais nos fluxos
por canal. A tabela `all_218_events.csv` une QC, CNN/OOD, ridge e matched filter
sob contrato estrito um-para-um, com 218 linhas e 72 colunas. A suíte encerrou
com `65 passed, 1 deselected`; o teste desmarcado é
um benchmark dependente de hardware.

## Decisão editorial

O material está pronto para:

1. servir de base a um artigo metodológico sobre auditoria de mudança de domínio;
2. um repositório reprodutível de processamento dos 218 eventos;
3. estudos posteriores de harmonização validados contra a representação
   numérica oficial.

O artigo do item 1 ainda requer o benchmark pareado do mesmo trigger em
múltiplos renderers, mais glitches off-source e injeções CBC para separar
renderer de mudança semântica.

Ainda não está pronto para:

1. afirmar classificação física de glitches nos eventos;
2. afirmar medição independente de chirp mass pelo ridge;
3. afirmar detecção ou significância pelo matched filter local;
4. publicar um release formal sem autores, ORCID, licença e `CITATION.cff`.

Os campos editoriais pendentes estão listados em
`RELEASE_METADATA_REQUIRED.md`.

## Adendo de 10/10/2026

O código de jean5429/gw-pic-2026 foi incorporado (`MIGRACAO_GW_PIC_2026.md`),
com 122 testes aprovados e os artefatos canônicos inalterados. Uma comparação
exploratória com o renderer na escala Gravity Spy reduziu os canais acima de q99
em Mahalanobis de 475 para 49, mas levou 227 canais acima de q99 em energy
(`research/RENDERIZADOR_GRAVITY_SPY.md`). O resultado canônico acima permanece
válido para o renderer preregistrado. A rejeição de todos os canais é, porém,
condicionada ao renderer e ao escore OOD, e deve ser relatada com essa análise
de sensibilidade.
