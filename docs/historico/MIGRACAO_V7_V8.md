# Migração da v7 para a v8

## Princípio

Os dados existentes em `outputs/gwtc` são preservados. Não é necessário baixar novamente os 90 eventos nem regenerar os 230 canais aprovados.

## Atualização no diretório atual

1. Faça um backup opcional dos índices:

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item ".\outputs\gwtc\snapshots\$stamp" -ItemType Directory -Force | Out-Null
Copy-Item .\outputs\gwtc\*.csv ".\outputs\gwtc\snapshots\$stamp"
Copy-Item .\outputs\gwtc\run_log.json ".\outputs\gwtc\snapshots\$stamp"
Copy-Item .\outputs\gwtc\manifest.json ".\outputs\gwtc\snapshots\$stamp"
```

2. Extraia `upgrade_v7_to_v8.zip` sobre a raiz atual com substituição dos arquivos.

3. Ative o ambiente já funcional:

```powershell
conda activate gw-v7
```

4. Reinstale o pacote:

```powershell
python -m pip uninstall -y gwproject-v7 gwproject-v8
python -m pip install -e ".[gw,ml,test]"
```

5. Execute a atualização automática:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
& .\scripts\upgrade_v7_in_place.ps1
```

## Primeiro teste científico

```powershell
& .\scripts\run_catalog_v8.ps1 -Mode Smoke
```

O smoke test usa dois eventos e não altera os dados brutos.

## Execução completa

```powershell
& .\scripts\run_catalog_v8.ps1 -Mode Full
```

## Mudanças de contrato

- `rejected_other` por falha de download passa a ser `data_unavailable`;
- o Q-transform é ignorado quando o processamento anterior reprova;
- NaN/Inf de borda podem ser recortados por regra geral e auditável;
- a análise de ridge deixa de usar uma janela fixa do GW150914;
- o matched filtering deixa de usar um banco fixo para todos os eventos;
- cada análise grava resultados por canal ou por evento e suporta retomada;
- a classificação Gravity Spy dos Q-grams é marcada como análise de semelhança morfológica.
