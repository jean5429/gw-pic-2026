# Guia de execução da v9.0 Research

## 1. Instalação

No PowerShell, na raiz do projeto:

```powershell
conda activate gw-v7
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONNOUSERSITE = "1"
python -m pip install -e ".[ml,gw,test]"
python -m pytest -q
```

Esperado: todos os testes funcionais passam. Os testes não executam treinamento completo.

## 2. Preservar baseline

Não sobrescreva `outputs/gwtc/gravity_spy_predictions`. Crie um snapshot:

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item .\outputs\gwtc\gravity_spy_predictions `
  ".\outputs\gwtc\snapshots\gravity_spy_v8_3_$stamp" -Recurse
```

## 3. Executar programa completo

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
& .\scripts\run_research_program.ps1
```

O script:

1. prepara painéis Gravity Spy harmonizados;
2. treina cinco sementes em diretório novo;
3. agrega métricas;
4. constrói referência OOD da seed 42;
5. classifica os 230 canais por ensemble;
6. salva resultados em `outputs/research`.

## 4. Inspeção obrigatória antes do treino

Depois de preparar painéis, abra 50 imagens aleatórias e confirme:

- painel sem títulos/eixos/colorbar;
- frequência alta no topo;
- faixa física correta;
- ausência de crop excessivo;
- tamanho 170×140;
- nenhuma classe com processamento sistematicamente diferente.

O script assume que o painel original cobre aproximadamente 10–2048 Hz em escala logarítmica. Essa hipótese deve ser confirmada.

## 5. Auditoria antiga

```powershell
python .\ondasgravitacionais_research_v9.py audit-domain -- `
  .\outputs\gwtc\gravity_spy_predictions\qgram_domain_statistics.csv `
  --output-dir .\outputs\research\baseline_domain_audit
```

## 6. Preparar painéis separadamente

```powershell
python .\ondasgravitacionais_research_v9.py prepare-panels -- `
  .\data\gravity_spy\trainingset `
  .\data\gravity_spy\trainingset_harmonized_20_512 `
  --overwrite
```

## 7. Treinar sementes

```powershell
python -m scripts.run_gravity_spy_seeds `
  .\data\gravity_spy\trainingset_v1d1_metadata.csv `
  .\data\gravity_spy\trainingset_harmonized_20_512 `
  --profile cuda_balanced `
  --output-dir .\outputs\gravity_spy_harmonized
```

## 8. Referência OOD

```powershell
$checkpoint42 = Get-ChildItem `
  .\outputs\gravity_spy_harmonized `
  -Recurse -Filter best_model.pt |
  Where-Object {$_.FullName -match "seed_42"} |
  Select-Object -First 1

python .\ondasgravitacionais_research_v9.py build-ood -- `
  $checkpoint42.FullName `
  .\data\gravity_spy\trainingset_v1d1_metadata.csv `
  .\data\gravity_spy\trainingset_harmonized_20_512 `
  --device cuda `
  --output .\outputs\research\ood_reference_seed42.npz
```

## 9. Ensemble

```powershell
python .\ondasgravitacionais_research_v9.py classify-ensemble -- `
  .\outputs\gravity_spy_harmonized `
  --approved .\outputs\gwtc\approved_channels.csv `
  --device cuda `
  --ood-reference .\outputs\research\ood_reference_seed42.npz `
  --output-dir .\outputs\research\gravity_spy_ensemble
```

## 10. Interpretação dos outputs

- `in_domain_candidate`: compatível com a referência estatística; não prova origem física;
- `review`: discordância do ensemble ou OOD moderado;
- `reject_ood`: incompatibilidade forte; não significa sinal astrofísico;
- `vote_fraction`: consenso entre sementes;
- `mutual_information`: componente epistemológica aproximada;
- `energy_mean`: score de energia;
- `mahalanobis_distance`: distância ao embedding de validação.

## 11. Contrafactuais de fase

O script recebe NPY 1D de strain/reconstrução:

```powershell
python .\ondasgravitacionais_research_v9.py phase-counterfactuals -- `
  .\dados\exemplo_strain.npy `
  --output-dir .\outputs\research\phase_counterfactuals
```

Ele preserva o módulo do espectro de Fourier e altera a fase gradualmente.

## 12. Falhas que exigem interrupção

- painel detectado incorretamente em mais de 1% das imagens;
- classes com taxa de falha de crop diferente;
- probe de pixels acima de 0,75 após harmonização;
- colapso em uma ou duas classes novamente;
- OOD thresholds classificando quase tudo como in-domain;
- resultados sensíveis ao threshold escolhido depois da inspeção do teste.
