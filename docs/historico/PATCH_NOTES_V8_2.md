# Patch v8.2 — frame do detector e auditoria dos gates

## Correcoes cientificas

- O ridge e o matched filtering agora usam massas no frame do detector.
- Quando apenas massas source-frame estao presentes, aplica-se
  `m_detector = (1 + z) m_source` usando o redshift do catalogo.
- O fallback silencioso para massas source-frame foi desativado por padrao.
- Os hashes de retomada incluem a revisao do algoritmo; resultados v8.1 nao sao reutilizados.

## Auditoria do matched filtering

`per_event.csv` passa a registrar separadamente:

- `central_rank_passed`;
- `network_snr_passed`;
- `top2_ratio_passed`;
- `detection_gate_passed`;
- `catalog_consistency_passed`;
- `gate_requires_central_rank_1`;
- `gate_passed`.

Na configuracao padrao, `require_central_rank_1 = false`. Assim, `gate_passed`
mede SNR e margem top-2, enquanto a vitoria do template central e reportada em
`catalog_consistency_passed`.

## Reexecucao necessaria

```powershell
python .\ondasgravitacionais_catalogo_v8.py index
python .\ondasgravitacionais_catalogo_v8.py ridge -- --workers 2 --no-resume
python .\ondasgravitacionais_catalogo_v8.py matched-filter -- --workers 1 --no-resume
```
