# Programa de robustez open-world para dados de ondas gravitacionais

Este diretório contém a formulação científica do projeto: hipóteses falsificáveis, preregistro, matriz experimental, política de alegações, revisão da literatura e regras de reprodutibilidade.

O problema central é a validade externa de classificadores treinados em uma representação e aplicados a Q-transforms produzidos por outro pipeline. O programa separa desempenho interno, mudança de domínio, incerteza de ensemble, reconhecimento OOD e interpretação física.

## Documentos

- `PARECER_CIENTIFICO_E_ROTEIRO.md`: veredito sênior, evidência atual, hipóteses
  concorrentes e prioridades;
- `RENDERIZADOR_GRAVITY_SPY.md`: renderer exploratório na escala Gravity Spy e
  comparação pareada com o renderer canônico (10/10/2026);
- `PROGRAMA_CIENTIFICO_COMPLETO.md`: pergunta central, hipóteses e fases experimentais;
- `PREREGISTRO.md`: decisões analíticas fixadas antes dos experimentos confirmatórios;
- `MATRIZ_EXPERIMENTAL.md`: fatores, controles e resultados esperados;
- `HIPOTESES_CONTROVERSAS_E_TESTES.md`: testes adversariais;
- `POLITICA_DE_ALEGACOES.md`: linguagem permitida e proibida;
- `PLANO_DE_DADOS_E_REPRODUTIBILIDADE.md`: congelamento e rastreabilidade;
- `GUIA_DE_EXECUCAO.md`: comandos operacionais.

Os resultados de classificação são evidência sobre compatibilidade estatística
com o domínio de referência. Os 218 eventos formam um _stress test_, não ground
truth de classes de glitch. As saídas não constituem identificação causal de
glitch, veto detectorial ou evidência de origem astrofísica.
