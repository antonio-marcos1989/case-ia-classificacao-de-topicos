# Case IA — Classificação de tópicos (Vício / Método VDE)

Sistema **offline** de classificação de questões em uma árvore de temas (Penal, Processo Penal, Legislação Penal Especial, Criminologia). **Sem APIs pagas** e **sem LLM na decisão**.

**Repo:** https://github.com/antonio-marcos1989/case-ia-classificacao-de-topicos  
**Contato do case:** carla@viciodeumaestudante.com.br

## Entregáveis

| Arquivo | Descrição |
|---|---|
| `publicacao.csv` | 659 linhas: `id`, `decisao`, `tema_central_id`, `extras_ids`, `confianca`, `motivo` |
| `previsao.txt` | Precisão/cobertura previstas + intervalos e método |
| `docs/relatorio.md` | Relatório ≤ 4 páginas (PT) |
| `docs/memo-professor.md` | Memo de 1 página ao professor |
| `data/anexos/` | Questões, temas, folha-verdade (fonte) |
| `artifacts/` | Leis, léxico, calibrador, limiar, embeddings cacheados, fingerprint |

## Regeneração offline (um comando)

```bash
python -m pip install -e .
# Predict puro — sem rede, sem LLM (usa artifacts/ já versionados)
python -m classificador predict
# equivalentemente:
make predict
```

Isso regenera `outputs/publicacao.csv`, `outputs/previsao.txt` e cópias na raiz (`publicacao.csv`, `previsao.txt`), além de `artifacts/publicacao.sha256`.

### Build vs predict

| Comando | Quando usar | Rede? |
|---|---|---|
| `python -m classificador predict` | Regenerar entrega a partir dos artefatos | **Não** |
| `python -m classificador build` | Recalibrar τ / calibrador na folha (seed=42) | Não |
| `python scripts/build_embeddings.py` | (Opcional, 1×) regenerar `artifacts/*_embeddings.npy` | Só se o modelo local ainda não estiver em cache |
| `python -m classificador check` / `make check` | Schema + temas ativos + fingerprint | Não |

Tempo típico (notebook/CPU): **predict ~10–20 s** com embeddings cacheados; **build ~1–2 min**; embeddings na 1ª vez ~5–15 min.

## Arquitetura

```
anexos → limpeza HTML → TF-IDF/BM25 + embeddings cacheados + leis + kNN
      → score híbrido → calibração → política publicar/revisar/fora
      → publicacao.csv + previsao.txt
```

**Política:** precisão alvo ≥ **95%** (calibração interna ~97%); empate entre **irmãos** → `revisar`; temas `inactive` nunca escolhidos; piso de τ ≈ **0,74**.

## Artefatos (proveniência)

| Artefato | Origem | Predict |
|---|---|---|
| `artifacts/laws_v0.yaml` | Regras manuais lei/âncora → tema | Sim |
| `artifacts/lexicon_seeds.yaml` | Seeds lexicais curados | Sim (via lexicon no model) |
| `artifacts/model.joblib` | Calibrador + τ + lexicon fit (seed 42) | Sim |
| `artifacts/metrics.json` | Métricas hold-out / OOF | Documentação / previsao |
| `artifacts/*_embeddings.npy` | `scripts/build_embeddings.py` (MiniLM multilíngue) | Se presentes |
| `artifacts/publicacao.sha256` | Fingerprint do CSV de entrega | `check --strict-hash` |

## Métricas internas (última calibração)

| Métrica | Valor |
|---|---|
| Hold-out precisão / cobertura / τ | 1,00 / 0,50 / 0,74 |
| OOF top-signal | ~0,63 |
| Fora recall (folha) | 8/8 |
| Corpus (aprox.) | ~211 publicar / 421 revisar / 27 fora |

Detalhes em `artifacts/metrics.json`, `previsao.txt` e `docs/relatorio.md`.

## Checklist de submissão

- [x] `publicacao.csv` com 659 IDs Q0001–Q0659, schema válido  
- [x] Decisões ∈ {publicar, revisar, fora}; tema ativo; motivo em revisar/fora; conf ∈ [0,1]  
- [x] `previsao.txt` com 4 números + intervalos + método  
- [x] `predict` regenera offline, sem LLM  
- [x] Relatório (`docs/relatorio.md`) + memo (`docs/memo-professor.md`)  
- [x] README com artefatos e tempos  
- [x] `make check` / `python -m classificador check`  
- [x] Seed fixa (42) no treino/calibração  

Validar localmente:

```bash
make predict
make check
# opcional: hash estrito
python -m classificador check --strict-hash
PYTHONPATH=src pytest -q
```

## Testes / EDA

```bash
make test          # pytest
make eda           # estatísticas dos anexos
```

## Contato

carla@viciodeumaestudante.com.br
