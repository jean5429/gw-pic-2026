# Relatório de implementação — versão 7

## Veredito

A v7 preserva a modularização da v6, mas altera os contratos científicos. Os gates deixam de ser campos decorativos e passam a controlar efetivamente o código de saída.

## Correspondência com a revisão da v6

### N1 — grandezas incompatíveis no gate morfológico

**Corrigido.** O módulo `morphology.py` produz strain físico com passa-banda e notches. O script `validate_gw150914.py` não chama `preprocess()` para essa comparação. O whitening permanece apenas na cadeia do Q-transform/ridge.

### N2/N2b — inversão de sinal e lag excessivo

**Corrigido.** O TOML define:

```toml
max_lag_seconds = 0.001
allow_sign_flip = false
```

A comparação H1×L1 mantém `allow_sign_flip=true` e janela de 20 ms, mas é marcada como diagnóstico entre detectores, não como gate LOSC.

### N3 — banco de templates e margem

**Corrigido parcialmente e tornado explícito.** O template `80+60` foi substituído por `30+10`. A configuração exige banda inspiral mínima e razão top-2. O programa reporta simultaneamente o ranking com corte em ISCO e o ranking com extensão suave.

A extensão ainda não é um aproximante IMR calibrado; a limitação aparece no JSON e na documentação.

### N4 — viés da chirp mass

**Corrigido operacionalmente.** Foram separados:

```toml
display_q = 12.0
ridge_q = 5.0
```

O gate usa tolerância relativa de 15%. O intervalo foi renomeado para `ridge_bootstrap_dispersion_solar`.

### M1 — configuração decorativa

**Corrigido.** Todas as chaves de gate são avaliadas. O `gravity_spy.toml` controla perfil, sementes, épocas, learning rates, AMP e métricas mínimas.

### M2 — gap parcial

**Corrigido.** `validate_data_segment()` mede fração de zeros e maior sequência contínua de zeros. A validação ocorre antes da PSD. O desvio padrão do strain branqueado é testado no núcleo temporal.

### M3 — adaptação da primeira convolução

**Corrigido.** `adapt_conv1_to_grayscale()` soma os canais e mantém o kernel 7×7. Não há recorte central 3×3.

### M4 — custo de CPU

**Corrigido por perfis, não por um único limiar universal.** Há três perfis: `cpu`, `cuda_balanced` e `cuda_highres`. O benchmark foi separado dos testes unitários.

### M5 — congelamento da conv1

**Corrigido.** Na primeira etapa treinam `conv1`, `bn1` e `fc`. Os blocos residuais ficam congelados.

### M6 — interpretação do bootstrap

**Corrigido semanticamente.** O JSON não usa mais o nome `ci95_solar`. O valor é identificado como dispersão do ridge.

### M7 — ausência de controles negativos

**Corrigido.** A suíte contém testes de:

- gap parcial;
- sinal invertido;
- extrapolação temporal;
- ruído puro no ridge;
- tar com path traversal;
- template inerte;
- margem de ranking;
- adaptação da conv1;
- congelamento seletivo.

### M8 — proveniência incompleta

**Corrigido.** O manifesto registra versões de NumPy, SciPy, pandas, GWPy, GWOSC, PyTorch, Torchvision, scikit-learn e Pillow, além de runtime CUDA, cuDNN, GPU e memória.

## Adequação ao notebook

O equipamento informado tem Ryzen 7 6800H, 32 GB de RAM e RTX 3070 Ti Laptop com 8 GB. A v7 adota `cuda_balanced` como perfil automático quando CUDA está disponível:

- AMP ativada;
- `channels_last`;
- `pin_memory`;
- quatro workers;
- batch 48;
- stride 2 sem max-pooling.

Esse perfil evita o custo extremo de stride 1 como padrão, mas mantém resolução espacial superior à ResNet convencional.

## Testes executados nesta entrega

```text
python -m pytest
19 passed

python -m pytest -m benchmark
1 passed

python -m compileall
compile_ok
```

## O que não foi validado neste ambiente

- download real do GWOSC;
- correlação real H1/L1 contra os arquivos LOSC;
- treinamento CUDA na RTX do usuário;
- download integral e treinamento do Gravity Spy;
- métricas finais de múltiplas sementes.

Esses itens exigem rede e a GPU física do notebook. Os scripts foram preparados para produzir manifestos suficientes para auditoria posterior.
