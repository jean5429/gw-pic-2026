# Retomada — experimento pareado de intensidade

Preparado em 18/09/2026. **Código novo não executado nem testado nesta preparação**, por solicitação do responsável. Leitura e revisão estática não substituem os testes abaixo. Não foram executados downloads, treinamento ou inferência pelo assistente.

## Onde paramos

A saída de terminal fornecida pelo responsável registra 65 testes aprovados e um teste desmarcado, cobertura de 475 canais aprovados dos 218 eventos e reprodução da rejeição OOD dos 475 canais. Esses testes se referem à versão anterior, não ao código novo deste documento.

A referência OOD formato 4 separa treino para ajuste geométrico e validação para calibração. A nova reprodução confirma a falha de transferência observada, mas não identifica sozinha sua causa. Diferenças de representação, população física, detector e condições de observação continuam alternativas possíveis. Os rótulos fechados forçados não são identificações físicas dos eventos.

Não é necessário recomeçar o treinamento, refazer os painéis ou reconstruir a referência OOD para esta etapa. Não usar `tools/run_research_program.ps1` como retomada: esse roteiro contém reconstruções e sobrescritas de saídas canônicas.

## O que foi preparado

Um experimento exploratório fatorial 2 × 2 sobre a imagem final de entrada da CNN, mantendo os cinco modelos e os limiares existentes:

| Condição | Contraste adicional | Gama |
|---|---|---|
| `identity` | nenhum | 1 |
| `gamma` | nenhum | 0,7 |
| `contrast` | percentis 1–99,5 | 1 |
| `contrast_gamma` | percentis 1–99,5 | 0,7 |

O contraste, quando habilitado, precede gama; não há arredondamento intermediário. As operações são aplicadas após o redimensionamento para 140 × 170 pixels, antes de ToTensor/Normalize. A condição identidade conserva exatamente os pixels da entrada canônica. O contraste com intervalo degenerado é marcado e mantido sem alteração, sem excluir a imagem. A transformação de intensidade pode perder informação por clipping e quantização; não deve ser chamada de equivalência física.

Entram **todos os 475 canais aprovados, cobrindo os 218 eventos**, e as 1.177 imagens do split oficial de validação do Gravity Spy. Os 179 pares evento–detector reprovados/indisponíveis continuam fora da inferência; não serão fabricados ou readmitidos para completar uma grade. Os cinco checkpoints são reaproveitados. Nenhuma imagem do teste é usada para escolher a transformação, e não há novo ajuste de temperatura, rede ou limiares.

O teste principal é: quanto a entrada modificada altera os escores e decisões em relação à identidade da **mesma amostra**? A ordem das entradas/condições é pseudoaleatorizada com semente fixa, igual entre modelos. A análise primeiro agrega os detectores de cada evento e depois calcula contrastes pareados e bootstrap de eventos. Os cinco modelos não são tratados como cinco novos catálogos.

O controle Gravity Spy é descritivo, pois a validação também calibrou OOD. Ele não produz uma nova estimativa independente de falso positivo. Seus intervalos por trigger não removem possíveis dependências temporais entre triggers. Os intervalos de 95% são exploratórios, pontuais, condicionados a este ensemble; não há testes confirmatórios nem correção por multiplicidade.

## Execute primeiro este roteiro

Abra o PowerShell e copie:

```powershell
Set-Location '<pasta do gwproject-1.0.1>'
& .\tools\run_paired_intensity.ps1
```

O roteiro usa explicitamente `$env:USERPROFILE\anaconda3\envs\gw-v7\python.exe`, sem depender de `(base)` no prompt. Aponta para o código desta pasta, sem exigir reinstalação. Se uma dependência estiver ausente, envie o erro; o roteiro não instala nada automaticamente.

Ele executará os testes locais, excluindo os marcados como benchmark ou rede. **Só se os testes passarem** iniciará a inferência. Em seguida criará uma pasta exclusiva `outputs/research/paired_intensity_<data>_<identificador>`, sem alterar dados, modelos ou resultados anteriores. O log fica em `outputs/research/logs/` e seu caminho aparece no terminal.

Planejamento para Ryzen 7 6800H / 32 GB de RAM / RTX 3070 Ti Laptop de 8 GB: um checkpoint por vez na GPU, lote 16, quatro threads de CPU e zero workers auxiliares. As imagens de referência ocupam cerca de 39 MB em uint8; o consumo total inclui bibliotecas, modelos e intermediários e será maior. Há uma verificação de 2 GiB livres para a nova pasta. Não há garantia de tempo ou uso máximo de memória antes da execução real.

Se faltar memória de GPU, feche outros programas que a utilizem e faça uma **nova execução** com lote menor:

```powershell
& .\tools\run_paired_intensity.ps1 -BatchSize 8
```

Uma interrupção permite reaproveitar checkpoints inteiramente concluídos, informando a pasta exata impressa no terminal:

```powershell
# Substitua o nome abaixo pela pasta real. Mantenha lote, dispositivo e ambiente.
& .\tools\run_paired_intensity.ps1 -Resume `
  -RunDirectory '.\outputs\research\paired_intensity_DATA_IDENTIFICADOR'
```

O contrato registra versões, configuração, hashes do código, fontes, modelos e referência OOD. Se algum desses elementos ou o lote mudar, a retomada é recusada: use uma pasta nova. Um modelo incompleto será recalculado, não continuado a partir de um minibatch. Arquivos parciais não autorizam interpretação; o marcador final deve ser `STATUS.json` com `status: complete`.

## Resultados esperados como arquivos, não como conclusão

- `REPORT.md` e `summary.json`: cobertura e resumo com limites de interpretação.
- `inputs.csv` e `contract.json`: identidade e integridade dos dados usados.
- `inference_schedule.csv`: ordem reproduzível das avaliações.
- `per_sample_condition.csv`: 6.608 linhas (1.652 entradas × 4 condições).
- `per_event_condition.csv`: 872 linhas (218 eventos × 4 condições), detectores agregados por evento.
- `condition_summary.csv` e `paired_contrasts.csv`: médias por bloco, efeitos e intervalos exploratórios.
- `model_00.npz` a `model_04.npz`: logits e escores por amostra/condição, com recibos de integridade.
- `ARTIFACT_HASHES.sha256`: hashes dos produtos desta execução; não substitui o manifesto histórico. O marcador `STATUS.json` é gravado por último e contém o hash desse manifesto, evitando referência circular.

Além da rejeição Mahalanobis e energia em 99%, são registrados mudança do rótulo fechado e confiança máxima. Como percentis saturados em 1 podem esconder mudanças, também há `mahalanobis_log_excess_99`: mediana entre modelos de `log(1 + distância) − log(1 + quantil 99% da validação)`, calculada dentro de cada modelo. Não é um novo limiar nem uma probabilidade física.

Envie `REPORT.md`, `summary.json`, `paired_contrasts.csv` e o log de execução. Se houver falha, envie primeiro o log completo. Não é necessário transmitir as imagens, arrays grandes ou checkpoints.

## Inspeção oficial: etapa separada e leve

Para preparar a futura comparação entre renderizadores, foi criado um comando que consulta apenas a lista de arquivos do registro configurado do Gravity Spy. **Não baixa os arquivos nem pressupõe que um HDF5 contenha strain ou Q-grams brutos.** Esta consulta exige internet quando você a executar:

```powershell
Set-Location '<pasta do gwproject-1.0.1>'
$Py = "$env:USERPROFILE\anaconda3\envs\gw-v7\python.exe"
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
& $Py -m gwproject.research_cli inspect-official --record-id 1486046 `
  --output ".\outputs\research\official_inventory_$Stamp.json"
```

Envie também esse JSON. Se a consulta falhar, ela não invalida nem bloqueia o experimento local. Para um HDF5 que já exista, o mesmo comando aceita `--local-hdf5 'CAMINHO_REAL.h5' --output 'NOVO_RELATORIO.json'`, sem `--record-id`: nesse modo não consulta a rede, não carrega valores dos arrays e não segue links externos. Enumera apenas estrutura, formas, tipos e nomes de atributos, com limite de tamanho do inventário.

Ainda será necessário confirmar conteúdo, unidades, grade temporal/frequencial, duração e chaves explícitas IFO+ID/GPS antes de implementar o pareamento físico. **Não associar linhas por posição e não confundir arrays de imagens com strain.** A inspeção de esquema não demonstra que o pareamento é possível.

## Decisões após os resultados

### Inventário oficial recebido em 18/09/2026

O arquivo `outputs/research/official_inventory_20260918_195429.json` lista o
HDF5 `trainingsetv1d1.h5`, com 3.051.325.064 bytes e checksum oficial
`md5:a6665b4ebee7b4ea57164ee743c2bd11`. Não foi localizado HDF5 na árvore
`data` durante a consulta de nomes de arquivos. Isso não identifica ainda o
conteúdo numérico nem suas chaves de pareamento.

Foi preparado, **sem execução pelo assistente**, o roteiro:

```powershell
& .\tools\inspect_official_hdf5.ps1
```

Execute na raiz do projeto. Ele baixa somente esse HDF5 para
`data/gravity_spy_official`, mantém transferências parciais para retomada,
verifica tamanho e MD5 antes de promover o arquivo e chama a inspeção de
estrutura já existente. Não baixa o TAR, não usa GPU, não treina e não carrega
os valores dos arrays. O download é gravado em disco; exige o espaço restante
mais 2 GiB de reserva. Em caso de interrupção da transferência, repita o mesmo
roteiro. Em caso de checksum incorreto, o arquivo é preservado: envie o log
antes de repetir. Não há sobrescrita automática de um HDF5 existente.

Envie apenas o `.schema.json` e o `.log` cujos caminhos aparecerem ao final.
Uma estrutura truncada, ausência de identificadores ou existência apenas de
imagens normalizadas exigirá inspeção complementar antes de definir qualquer
comparação física. MD5 confere integridade contra o inventário, não valida a
ciência. O roteiro PowerShell novo ainda não foi testado por execução.

### Interpretação e etapas posteriores

1. Se a intensidade mudar fortemente os escores, haverá evidência de sensibilidade dos modelos a essas transformações. Isso não prova que ela explica toda a falha de transferência nem seleciona uma transformação válida para produção.
2. Se quase nada mudar, apenas estas intervenções terão resultado negativo. Persistem diferenças de faixa de frequência, discretização, normalização, população e detector; não se encerra a investigação de domínio.
3. Comparação entre renderizadores do mesmo trigger depende do inventário e das chaves de pareamento. Se o arquivo oficial não oferecer a representação necessária, a alternativa será obter strain público e reconstruir pares controlados, com parâmetros explícitos.
4. Controles off-source e injeções em ruído real continuam necessários para separar contexto instrumental e conteúdo do sinal. Ainda não foram implementados neste pacote; dependerão da definição de janelas, qualidade, amostragem e separação de dados.

Não se buscará uma redução artificial da rejeição dos 475 canais ajustando o limiar ao catálogo. O objetivo é explicar a resposta da rede e estabelecer quais avaliações sustentariam uma interpretação científica.

As pastas `atenuacao_ridge`, `ridge_atenuacao` e a ramificação adicional `ridge_atenuacao_v2` permaneceram fora deste trabalho.
