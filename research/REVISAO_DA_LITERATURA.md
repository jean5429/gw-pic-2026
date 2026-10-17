---
title: "Revisão de escopo: glitches, representações e inferência robusta em ondas gravitacionais"
date: "27 de julho de 2026"
lang: pt-BR
---

# 1. Natureza da revisão

Esta é uma revisão de escopo estruturada, não uma revisão sistemática exaustiva. A busca foi atualizada em 25 de julho de 2026 e priorizou artigos revisados por pares, preprints recentes, documentos LVK/GWOSC e recursos científicos abertos. Os eixos de busca foram:

- Gravity Spy, taxonomia, multi-view e evolução entre observing runs;
- classificação supervisionada e descoberta não supervisionada de glitches;
- reconstrução temporal e modelos generativos;
- BayesWave, modelos conjuntos sinal–glitch e robustez da inferência;
- shortcut learning, domain shift, calibration e OOD;
- recursos Hugging Face/Hugging Science aplicáveis a ondas gravitacionais.

As afirmações sobre o projeto local derivam de seus próprios resultados. As afirmações sobre o estado da arte derivam das referências listadas em `REFERENCIAS.bib`. Preprints de 2025–2026 são tratados como evidência provisória.

# 2. Gravity Spy e a natureza das classes

Gravity Spy integra citizen science e machine learning para classificar transientes não gaussianos por morfologia tempo–frequência. O trabalho fundador estabeleceu uma taxonomia operacional que permite triagem rápida e investigação de causas instrumentais. O dataset formalizado por Bahaadini et al. consolidou imagens e metadados para tarefas de ML.

A literatura posterior mostra que a taxonomia não é estática. As distribuições de glitches diferem entre Hanford e Livingston, e novas morfologias emergem com mudanças do instrumento e do ambiente. A análise de O3 classificou centenas de milhares de glitches e concluiu que estudos de qualidade de dados podem precisar ser específicos para cada observatório. O artigo de lições aprendidas enfatiza o acoplamento contínuo entre voluntários e modelos, precisamente porque classes desconhecidas e mudanças de domínio permanecem relevantes.

A evolução para O4 adotou multi-view fusion e atenção, reconhecendo que uma única janela temporal perde informação. Isso sustenta duas conclusões:

1. as classes são úteis para detector characterization;
2. a estabilidade da classe depende da representação, do run e da janela.

# 3. Descoberta não supervisionada e open-world

Métodos de clustering, similarity learning, autoencoders e embeddings foundation-model procuram reduzir a dependência de rótulos fechados. O CTSAE combina múltiplas escalas temporais e componentes CNN/ViT. Trabalhos de similarity learning buscaram encontrar exemplos semelhantes e candidatos a novas classes. Um estudo O4a de 2026 com DINOv2 congelado relatou um resultado nulo para novas morfologias no conjunto analisado, mas observou diferenças estruturais entre os manifolds de H1 e L1.

Esse resultado nulo é cientificamente importante: discovery pipelines precisam controlar estabilidade de clustering, ablações, time slides e comparação com referência in-domain. Anomalia visual sem esses controles não é nova classe.

# 4. Limitação das imagens de magnitude

Imagens Q-transform são excelentes ferramentas diagnósticas, mas não são representações invertíveis da forma de onda. A magnitude perde fase e pode apagar distinções relevantes para causalidade e coerência. O projeto GlitchGAN, construído no domínio temporal e com derivadas, foi motivado pela necessidade de gerar waveforms fisicamente mais realistas. Sua análise indica que classificadores de imagens de magnitude podem atribuir alta confiança a formas temporalmente inadequadas.

DeepExtractor vai além da classificação e reconstrói sinal ou glitch no domínio temporal. O trabalho reporta comparação com BayesWave e generalização para dados Gravity Spy. Independentemente dos números específicos, sua contribuição conceitual é clara: avaliação por mismatch e reconstrução temporal é mais próxima da física do que apenas acurácia de classe visual.

# 5. Signal–glitch joint inference

BayesWave formulou sinais coerentes e glitches incoerentes em uma base transdimensional de wavelets. Isso permite comparar hipóteses sem exigir uma waveform CBC específica. Trabalhos recentes ampliam o problema para inferência conjunta de parâmetros astrofísicos e glitches.

`bilby_glitch`, publicado como preprint em julho de 2026, implementa modelos de slow scattering, sine-Gaussian e chirplet em uma estrutura modular. A reanálise de GW200129 é especialmente relevante: a evidência de precessão muda com a combinação entre approximant e modelo de glitch. A conclusão correta não é que precessão “não existe”, mas que a robustez da inferência deve ser testada no espaço produto

$$
\mathcal W\times\mathcal G,
$$

onde $\mathcal W$ é a família de waveforms e $\mathcal G$ a família de modelos de glitch.

GW191109 também tem sido discutido como exemplo de spins interessantes sob problemas de qualidade de dados. Esses casos tornam plausível uma agenda de “fragilidade inferencial”, desde que ela seja conduzida sem selecionar apenas eventos favoráveis à hipótese.

# 6. Shortcut learning e domain shift

Shortcut learning descreve regras que funcionam em benchmarks, mas falham quando o ambiente muda. Em imagens científicas, margens, textos, marcas de equipamento, protocolos de aquisição e estilos de renderer podem ser mais fáceis de aprender que o fenômeno de interesse.

O projeto local constitui um caso diagnóstico mais delimitado:

- o modelo canônico foi treinado nos painéis oficialmente recortados, não nos
  PNGs completos;
- as entradas externas são Q-grams GWOSC renderizados por outra cadeia;
- as escolhas fechadas se concentraram em duas classes, mas todos os 475 canais
  foram corretamente retidos como `reject_ood`;
- um probe de nove estatísticas das imagens efetivamente apresentadas à CNN
  reproduziu `None_of_the_Above` versus as demais escolhas com balanced accuracy
  0,769 e AUROC 0,817, usando folds agrupados por evento;
- Mahalanobis rejeitou os 475 canais em todos os checkpoints, enquanto energy
  quase nunca os rejeitou.

Isso é compatível com shortcut learning e mudança de representação, mas não
identifica qual atalho a rede utilizou. Também não separa renderer de mudança
semântica: Gravity Spy contém triggers de glitches, ao passo que as entradas
externas são janelas centradas em eventos CBC catalogados. São necessários
contrafactuais, renderer holdout pareado pelo mesmo trigger e conjuntos
off-source/injetados.

# 7. Incerteza, calibração e OOD

Redes modernas podem ser mal calibradas mesmo em dados in-domain. Temperature scaling é uma correção simples para calibration, mas não resolve OOD. Maximum softmax probability é um baseline, não uma garantia de validade externa.

A literatura oferece várias famílias:

- ODIN: perturbação de entrada e temperature scaling;
- Mahalanobis: distância a distribuições class-conditionais no embedding;
- energy scores: função dos logits mais apropriada que softmax para separação OOD em muitos benchmarks;
- Outlier Exposure: treino com exemplos externos;
- deep ensembles: dispersão entre modelos como incerteza epistemológica;
- conformal prediction sob covariate shift: cobertura com hipóteses explícitas de pesos/importância.

Nenhum método garante OOD universal. A referência local foi corrigida para
ajustar a geometria Mahalanobis no treino, calibrar percentis na validação e
reservar o teste como controle in-domain. Essa separação corrige leakage de
calibração, mas a discordância energy–Mahalanobis demonstra por que combinar
métricas não autoriza tratar a saída como probabilidade física.

# 8. Domain generalization

DANN tenta remover informação de domínio adversarialmente. IRM busca preditores invariantes entre ambientes. Group DRO minimiza risco no pior grupo. A literatura de robustez, contudo, mostra que “invariância” pode falhar se os ambientes não forem definidos de maneira causalmente informativa.

No caso GW, os ambientes naturais são:

- detector;
- observing run;
- período temporal;
- renderer;
- janela temporal;
- faixa de frequência;
- método de whitening/PSD.

Uma estratégia adequada não deve remover toda informação de detector. Parte dela pode ser o fenômeno instrumental que queremos estudar. O objetivo é disentanglement, não cegueira automática ao instrumento.

# 9. Recursos abertos para o programa

## 9.1 DeepExtractor reconstructions

O dataset público contém 35 mil waveforms balanceadas de sete classes, janelas de 2 s e amostragem de 4096 Hz. Ele permite experimentos phase-aware e comparação entre imagem e forma de onda.

## 9.2 Modelos externos de classificação

Uma model card pública oferece ViT e ResNet para Gravity Spy O3 e relata avaliação temporal/O4. Esses números são auto-relatados e o manuscrito estava submetido; o recurso deve ser usado como baseline externo, não como padrão-ouro.

## 9.3 Dados O4b

Há conversões Parquet de strain público O4b. Um exemplo V1 possui mais de um bilhão de linhas e 22,1 GB. O uso correto é streaming, seleção de segmentos e projeção de colunas.

# 10. Lacunas identificadas

1. Na busca de escopo dirigida, não foi encontrado um benchmark GW pareado que
   meça a estabilidade da mesma observação sob renderers fisicamente
   equivalentes, com splits por trigger e calibração OOD independente.
2. A fase é raramente incorporada em taxonomias visuais.
3. OOD costuma ser avaliado em benchmarks genéricos, não entre observing runs e pipelines GW.
4. A ligação entre embeddings de glitches e microestados instrumentais é pouco explorada publicamente.
5. A robustez de parâmetros astrofísicos à marginalização conjunta waveform–glitch ainda está emergindo.
6. Há pouca padronização de datasheets, manifests e renderer provenance para Q-transforms.
7. A categoria `None_of_the_Above` é frequentemente usada sem decomposição open-world rigorosa.

# 11. Síntese crítica

A literatura sustenta uma mudança de foco: de classificação fechada de imagens para modelagem multimodal, open-world e causalmente informada. O projeto local não deve competir com Gravity Spy em acurácia interna. Sua contribuição potencial está em demonstrar, de forma reprodutível, quais conclusões sobrevivem à mudança de representação e quais colapsam.

A evidência atual é forte para mudança no espaço latente e para associação
pós-hoc entre decisões fechadas e propriedades globais da imagem. É insuficiente
para atribuir causalmente o efeito a um renderer específico, porque população
CBC e população de glitches também diferem. Continua insuficiente para alegar
microestados instrumentais, nova taxonomia ou impacto astrofísico.

# Referências essenciais

Consulte `REFERENCIAS.bib`. As referências nucleares incluem Zevin et al. (2017, 2024), Bahaadini et al. (2018), Glanzer et al. (2023), Wu et al. (2025), DeepExtractor (2025), GlitchGAN (2026), BayesWave, `bilby_glitch` (2026), Geirhos et al. (2020), Lee et al. (2018), Liu et al. (2020), Lakshminarayanan et al. (2017) e Guo et al. (2017).
