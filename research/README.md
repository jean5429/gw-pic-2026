# GW Open-World Robustness Program — v9.0

Este diretório contém o programa científico criado a partir da auditoria da v8.3.

## Documentos

- `PROGRAMA_CIENTIFICO_COMPLETO.md`: relatório principal;
- `REVISAO_DA_LITERATURA.md`: revisão de escopo;
- `HIPOTESES_CONTROVERSAS_E_TESTES.md`: hipóteses e falsificações;
- `PREREGISTRO.md`: protocolo confirmatório;
- `MATRIZ_EXPERIMENTAL.md`: ablações e controles;
- `GUIA_DE_EXECUCAO.md`: comandos locais;
- `POLITICA_DE_ALEGACOES.md`: linguagem permitida e proibida;
- `PLANO_DE_DADOS_E_REPRODUTIBILIDADE.md`: provenance, splits e releases;
- `PLANO_DE_ARTIGOS.md`: estratégia de publicação;
- `REFERENCIAS.bib`: bibliografia inicial.

## Código novo

- harmonização de painéis Gravity Spy;
- baseline subtraction e inversão de frequência para Q-grams;
- ensemble multiseed;
- energy score e Mahalanobis OOD;
- contrafactuais de fase;
- auditoria agrupada de atalhos de pixel;
- testes unitários.

## Limite importante

O pacote foi validado com testes unitários e análise dos CSVs fornecidos. O treinamento completo harmonizado e a inferência nos arquivos locais de 5,49 GB precisam ser executados no computador do projeto.
