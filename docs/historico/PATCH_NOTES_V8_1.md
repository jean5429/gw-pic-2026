# Patch v8.1 — ridge do catálogo

## Motivo

Nos dados reais, alguns ridges tempo–frequência ocupam poucos bins e formam uma escada quantizada. O estimador Theil–Sen pode então devolver inclinação exatamente zero, apesar de existir uma tendência crescente global. Na v8.0 isso aparecia como `ValueError: a inclinacao deve ser negativa` e era registrado como erro genérico.

## Correções

1. Theil–Sen continua sendo o estimador principal.
2. Quando Theil–Sen zera por empates, uma regressão linear é usada como fallback somente se a inclinação continuar fisicamente negativa.
3. Ridges realmente planos continuam rejeitados.
4. O bootstrap usa a mesma regra robusta.
5. A seleção de pontos exige cobertura temporal e frequencial mínima.
6. O código tenta quantis de energia menos restritivos antes de desistir.
7. Falhas recebem estados específicos: `flat_ridge`, `nonphysical_ridge_slope`, `bootstrap_insufficient`, `insufficient_time_coverage` e `insufficient_frequency_span`.
8. Resultados aprovados registram `fit_method` e diagnósticos de seleção.
9. Novo comando `diagnose-ridge` resume falhas, detectores, métodos de ajuste e erros relativos.

## Aplicação

Extraia o patch na raiz do projeto e execute:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
& .\scripts\upgrade_v8_to_v8_1.ps1
```

Reexecute o ridge sem reutilizar os resultados da v8.0:

```powershell
python .\ondasgravitacionais_catalogo_v8.py ridge -- `
  --workers 2 `
  --no-resume
```

Depois:

```powershell
python .\ondasgravitacionais_catalogo_v8.py diagnose-ridge
```
