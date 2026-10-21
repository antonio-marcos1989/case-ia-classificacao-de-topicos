# Relatório — Classificação de tópicos (Case IA / Vício)

**Repositório:** https://github.com/antonio-marcos1989/case-ia-classificacao-de-topicos  
**Contato do case:** carla@viciodeumaestudante.com.br  
**Escopo:** 659 questões × 377 temas (375 ativos); decisão `publicar` / `revisar` / `fora` com precisão ≥ 95% nas publicadas.

---

## 1. Sistema

Pipeline **offline**, sem LLM na decisão e sem APIs pagas:

```
anexos → limpeza HTML → TF-IDF/BM25 + char n-grams + embeddings cacheados
      → leis/âncoras + kNN/protótipos da folha → score híbrido
      → calibração (isotônica + LR auxiliar) → política τ
      → publicacao.csv + previsao.txt
```

**Sinais:** recuperação lexical enriquecida (título + caminho + seeds); embeddings multilíngues pré-computados (`paraphrase-multilingual-MiniLM-L12-v2`); kNN ponderado e protótipos na folha; dicionário de leis com desambiguação (LEP material vs processual, Maria da Penha, JECRIM, etc.); detector de `fora` (disciplina OOD, enunciado incompleto/multi-tópico, vizinhos da folha).

**Política:** empate entre subtemas irmãos → `revisar`. Publicar só com \(P(\text{acerto tolerante}) \ge \tau\) e guards (margem, lei, knn-fora). Temas `inactive` nunca entram.

**Regeneração:** `python -m classificador predict` (ou `make predict`) a partir de `data/anexos/` + `artifacts/` — sem rede. `build` recalibra; embeddings são opcionais e já versionados em `.npy`.

---

## 2. Uso dos 100 rótulos

Protocolo **fit 70% / hold-out 30%** estratificado (disciplina + flag fora), seed fixa `42`:

| Parte | Uso |
|---|---|
| Fit (~70) | Léxico a partir da folha, kNN/protótipos, reranker pointwise, calibrador (OOF interno no fit) |
| Hold-out (~30) | Escolha de \(\tau\) e estimativa de precisão/cobertura → `previsao.txt` |
| Todos (sanidade) | Auditoria de `fora`, top-1 tolerante, fingerprint do CSV |

Os 100 **não entram** na métrica da banca; a previsão refere-se às demais questões. Seeds de léxico/leis foram curados manualmente (não gerados por LLM na decisão).

---

## 3. Diagnóstico

**Árvore:** descrições pobres (“Subtema de…”); ambiguidade estrutural — LEP e Maria da Penha (e afins) aparecem em ramos paralelos (Leg. Especial vs Penal/Processo Penal). Metadados `disciplina_origem` / `assunto_origem` são ruidosos; usamos como sinal fraco.

**Erros caros:** irmãos de nível 2 (regra de acerto). Mitigação: `revisar` no empate; lei específica no stem (sem distractors HTML); não deixar o reranker derrubar `lei_promovida`.

**Fora (~8% na folha):** modelado explicitamente; recall na folha chegou a **8/8** na rodada final, com zero falso `fora` in-taxonomy na folha.

**Shift:** folha só em Q0001–Q0359. Sanity banco vs recentes: confiança e taxa de publicação quase flat (Δconf ≈ −0,01) — sem rótulo nas recentes, só distribuição.

**História de \(\tau\):** calibração chegou a operar em **τ ≈ 0,84** com prec 1,00 e cov ~0,40. Com melhor sinal de ranking, o limiar que maximiza cobertura sob prec ≥ 0,97 caiu ao **piso operacional 0,74** (não baixamos abaixo do piso para “inflar” cobertura). Hold-out atual: **precisão 1,00 (15/15), cobertura 0,50**. OOF top-signal ~**0,63**. Corpus (659): ~**211 publicar / 421 revisar / 27 fora**.

---

## 4. Produção (50k questões/semana)

| Etapa | Tempo (ordem de grandeza) | Notas |
|---|---|---|
| Predict batch 659 | ~10–20 s | Artefatos em disco; CPU |
| Predict 50k | ~15–40 min | Linear no nº de questões; paralelizar por shard |
| Rebuild embeddings (opcional) | ~5–15 min / milhares | Só quando o corpus muda de domínio |
| Retreino `build` | ~1–2 min | Com folha ampliada |

**Custo:** zero API; máquina CPU suficiente. **Loop:** especialista corrige fila `revisar` → novos rótulos entram na folha → `build` → novo \(\tau\) → `predict`. Publicar só o que passa \(\tau\); o resto alimenta treino.

**O que não publicar agressivamente:** empates entre irmãos; knn-fora sem lei forte; stems incompletos/multi-tópico; temas “lixeira” sem âncora lexical/legal.

---

## 5. Métricas honestas (entrega)

| Métrica | Valor |
|---|---|
| Hold-out precisão / cobertura / τ | 1,00 / 0,50 / 0,74 |
| Bootstrap cov (hold-out) | 0,33–0,67 |
| OOF top-signal (fit) | ~0,63 |
| Fora recall (folha) | 8/8 |
| Corpus publicar / revisar / fora | ~211 / 421 / 27 |

`previsao.txt` usa hold-out + bootstrap (1000). Regeneração: `make predict` ou `python -m classificador predict`; validação: `make check`.
