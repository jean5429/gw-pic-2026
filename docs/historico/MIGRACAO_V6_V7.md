# Migração da v6 para a v7

## Ambiente

Não altere o ambiente `gw-v6-clean`. Crie `gw-v7`:

```powershell
conda env create -f environment-core.yml
conda activate gw-v7
python -m pip install -e ".[gw,ml,test]"
powershell -ExecutionPolicy Bypass -File .\scripts\install_gpu_windows.ps1
```

## Por que a suíte antiga não foi copiada literalmente

Alguns testes da revisão eram especificações úteis, mas não testes de regressão isolados:

- N1 chamava diretamente `preprocess()` e, portanto, continuaria falhando mesmo depois de o script deixar de usar `preprocess()` no gate morfológico;
- N2 chamava `best_lag_alignment(..., allow_sign_flip=True)` e exigia que a própria opção deixasse de funcionar, embora ela seja necessária na comparação H1×L1;
- M7 contava expressões por regex, em vez de exercer comportamentos;
- M4 dependia do tempo absoluto da máquina.

Na v7, esses itens foram convertidos em testes comportamentais:

- `physical_morphology_series` é comparada ao passa-banda físico;
- a inversão é testada com `allow_sign_flip=False` no gate publicado;
- os controles negativos são funções de teste explícitas;
- o custo fica sob o marcador `benchmark`.

## Saídas incompatíveis

### Chirp mass

v6:

```json
"ci95_solar": [a, b]
```

v7:

```json
"ridge_bootstrap_dispersion_solar": [a, b]
```

### Matched filter

v6 retornava uma única lista `ranking`. A v7 retorna:

```json
"rankings": {
  "inspiral": [],
  "extended": []
}
```

### Gravity Spy

Os diretórios agora incluem o perfil:

```text
seed_42_cuda_balanced/
```

Isso impede misturar resultados produzidos por arquiteturas com downsampling diferente.
