# Estrutura canônica do GWProject 2.0

Esta organização separa insumos oficiais, modelos, resultados vigentes e
artefatos históricos. Nada do legado foi apagado.

```text
configs/                                  parâmetros versionados
data/gravity_spy/                         arquivo e imagens oficiais intactos
data/gravity_spy_harmonized/              painéis 140x170 e metadados após QC
models/gravity_spy_harmonized_v1d1/       ensemble canônico de cinco sementes
models/legacy/gravity_spy_full_renderer/  checkpoints antigos, não canônicos
outputs/gwtc4_218/                        strain, Q-grams e índices dos 218 eventos
outputs/research/gwtc4_218/               OOD, CNN, ridge e matched filter
outputs/research/renderer_gravity_spy_20261010/ renderer exploratório (não canônico)
outputs/legacy/gwtc3_90/                  execução histórica de 90 eventos
outputs/legacy/research_unharmonized/      resultados históricos do renderer errado
outputs/legacy/research_invalid_ood_v3_20260727/ referência OOD com calibração inválida
research/                                 preregistro e política de alegações
src/gwproject/                            implementação
tests/                                    testes científicos e de software
tools/                                    automação
gw_student_kit/                           kit didático independente do pacote
```

`gw_student_kit/` veio de jean5429/gw-pic-2026 (ver `MIGRACAO_GW_PIC_2026.md`).
É material de ensino, não entra nos resultados científicos, e suas figuras em
`gw_student_kit/results/` não são versionadas.

As pastas `atenuacao_ridge/` e `ridge_atenuacao/` são ramificações externas a
este escopo. Elas permanecem fisicamente intactas, são ignoradas pelo Git e não
entram em inventários, hashes, testes ou alegações desta reconstrução.

## Fronteira entre produto e diagnóstico

- A CNN fornece semelhança morfológica com a taxonomia Gravity Spy. Uma saída
  só pode ser chamada de candidata em domínio após calibração do ensemble e
  comparação com a referência OOD de cada checkpoint.
- O ridge é um diagnóstico condicionado ao catálogo: a massa publicada informa
  a janela e a banda. Não é uma medição cega e independente de massa.
- O matched filter usa um banco local condicionado ao catálogo e PSD off-source.
  Sem FAR, chi-quadrado e busca cega, o resultado não é uma detecção.

## Escopo de 218 eventos

O conjunto canônico agrega os 90 eventos até o GWTC-3 e os registros do
GWTC-4.0, removendo `GW230630_070659`. O próprio release GWTC-4 documenta que
esse candidato, embora inicialmente significativo, apresentou evidência de
origem instrumental. O programa exige exatamente 218 nomes antes de aceitar o
catálogo.
