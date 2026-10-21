# Memo ao professor — árvore, fila `revisar` e o que não publicar

**De:** Antonio Marcos ([antonio-marcos1989](https://github.com/antonio-marcos1989)) — repositório `case-ia-classificacao-de-topicos`  
**Para:** professor / especialista de conteúdo  
**Assunto:** onde a taxonomia atrapalha o classificador — e como usar a fila `revisar` no treino  
**Contato do case (banca):** carla@viciodeumaestudante.com.br

---

## 1. Problemas de árvore (desambiguação)

Dois ramos paralelos geram erro “entre irmãos” mesmo quando o conteúdo é claro:

| Tema | Onde aparece | Risco |
|---|---|---|
| **Lei de Execução Penal (LEP)** | Leg. Especial **e** Processo Penal (execução) | Questão material (remição, regime) vs processual (agravo, juízo da execução) |
| **Maria da Penha** | Leg. Especial **e** Penal | Mesmo padrão: lei especial vs tipificação/penal genérico |
| **Drogas / outras leis especiais** | Espelhos semelhantes | Distractors nas alternativas puxam o ramo errado |

**Pedido:** fundir ou marcar um ramo “canônico” por lei (ex.: LEP material → Leg. Especial; LEP processual → PP), com link cruzado no outro. Enquanto isso, o sistema **revisa** no empate em vez de chutar o subtema.

---

## 2. Feedback `revisar` → treino

1. Especialista corrige só a fila `revisar` (sugestão + motivo já vêm no CSV).  
2. Novos rótulos entram na `folha_verdade` (ou planilha equivalente).  
3. Rodar `python -m classificador build` (seed fixa) e `predict`.  
4. \(\tau\) é recalibrado para manter precisão ≥ 95% nas publicadas.

Priorize corrigir: (a) irmãos de nível 2; (b) LEP/Maria/Drogas; (c) casos marcados `fora` duvidosos. Isso rende mais que rotular aleatório.

---

## 3. O que **não** publicar com agressividade

- Empate ou dúvida entre **subtemas irmãos** → sempre revisar.  
- Vizinhos da folha sugerindo **fora**, sem lei/margem fortes.  
- Enunciados incompletos (“complete a frase”), multi-tópico (CPP + execução nas opções) ou disciplina de origem claramente fora do recorte.  
- Temas genéricos / “lixeira” sem âncora de lei ou léxico.

**Régua atual:** hold-out com precisão 1,00 nas publicadas; cobertura ~50% no hold-out; no corpus ~1/3 publicar, maioria revisar. Melhor cobrir menos e acertar do que publicar no ramo errado — o aluno estuda o tópico errado.

---

## 4. Uma linha de ação

Se puder mudar **uma** coisa na árvore esta semana: **desambiguar LEP** (material vs processual) com IDs canônicos. O restante o loop `revisar`→`build` absorve.
