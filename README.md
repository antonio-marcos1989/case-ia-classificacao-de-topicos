# Case IA — Classificação de tópicos (Vício / Método VDE)

Sistema **offline** de classificação de questões em uma árvore de temas (Penal, Processo Penal, Legislação Penal Especial, Criminologia). Sem APIs pagas e sem LLM na decisão.

## Entregáveis

| Arquivo | Descrição |
|---|---|
| `publicacao.csv` | 659 linhas: `id`, `decisao`, `tema_central_id`, `extras_ids`, `confianca`, `motivo` |
| `previsao.txt` | Precisão/cobertura previstas + intervalos e método |
| `data/anexos/` | Questões, temas, folha-verdade (fonte) |
| `artifacts/` | Dicionário de leis, calibrador, limiar, embeddings cacheados (opcional) |

## Como rodar (offline)

```bash
python -m pip install -e .
# ou: pip install -r requirements.txt && PYTHONPATH=src ...

# 1) (opcional, uma vez) embeddings locais — requer sentence-transformers; grava artifacts/*.npy
python scripts/build_embeddings.py

# 2) calibra limiar na folha-verdade e grava artifacts/model.joblib
python -m classificador build

# 3) regenera publicacao.csv + previsao.txt (sem rede / sem LLM)
python -m classificador predict
```

Tempo típico no notebook: build+predict **sem** embeddings ~1–2 min; **com** embeddings pré-cacheados ~mesmo; gerar embeddings na 1ª vez ~5–15 min (download do modelo local).

## Arquitetura

```
anexos → limpeza HTML → TF-IDF + regras de lei/termo (+ embeddings cacheados)
      → score híbrido → probabilidade isotônica → política publicar/revisar/fora
      → publicacao.csv + previsao.txt
```

**Política (travada):**
- Precisão alvo nas publicadas ≥ **95%**, calibração interna mirando **~97%**.
- Empate/ambiguidade entre **subtemas irmãos** → `revisar` (não chuta irmão; não sobe ao pai).
- Temas `inactive` nunca são escolhidos.

## Artefatos

| Artefato | Origem | Uso no predict |
|---|---|---|
| `artifacts/laws_v0.yaml` | Regras manuais (leis/âncoras → tema) | Sim |
| `artifacts/model.joblib` | Calibrador isotônico + `tau` (CV na folha) | Sim |
| `artifacts/metrics.json` | Métricas OOF da folha | Só documentação / previsao |
| `artifacts/*_embeddings.npy` | `scripts/build_embeddings.py` (opcional) | Se presentes |
| `artifacts/embeddings_meta.json` | Metadados do cache | Se presentes |

Predict **não** baixa modelos nem chama APIs: se os `.npy` não existirem, usa só TF-IDF + regras.

## Avaliação interna

Métrica alinhada ao enunciado (acerto tolerante: central, extra, pai/filho; **irmão = erro**). Números atuais após `build` ficam em `artifacts/metrics.json` e em `previsao.txt`.

## EDA

```bash
PYTHONPATH=src python scripts/eda.py
```

## Testes

```bash
PYTHONPATH=src pytest -q
```

## Contato do case

carla@viciodeumaestudante.com.br
