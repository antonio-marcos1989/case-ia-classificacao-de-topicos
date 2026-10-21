# Anexos — Case "Classificação de tópicos"

Todos os arquivos estão em UTF-8. CSVs separados por vírgula, com aspas quando necessário.

| Arquivo | Linhas | O que é |
|---|---|---|
| `questoes.jsonl` | 659 | Uma questão por linha: `id`, `enunciado_html` (HTML cru), `comando`, `alternativas`, `disciplina_origem` e `assunto_origem` (como vieram do site de origem; podem estar errados). Q0001 a Q0359 vêm do nosso banco; Q0360 a Q0659 são de concursos recentes |
| `temas.csv` | 377 | Os temas em que as questões podem ser classificadas, em árvore: Penal, Processo Penal, Direito Penal - Legislação Especial e Criminologia. `nivel` 0 = disciplina, 1 = tema, 2 = subtema. Inclui `descricao` e `status` (não use temas inativos) |
| `folha_verdade.csv` | 100 | Questões classificadas pela equipe (formato abaixo) |
| `entrega_modelo.csv` | 2 | Exemplo do formato de `publicacao.csv` |

## `folha_verdade.csv`

| Coluna | Valor |
|---|---|
| `id` | Id da questão em `questoes.jsonl` |
| `tema_central_id` | `tema_id` de `temas.csv`. Vazio quando a questão é fora da taxonomia |
| `tema_central_caminho` | Caminho do tema na árvore, só para leitura |
| `extras_ids` | Outros temas que a questão também cobre, separados por `;`. Pode ficar vazio |
| `fora_da_taxonomia` | `1` quando nenhum tema da árvore serve; `0` nos demais casos |

A seleção é aleatória e estratificada por disciplina, então tem a mesma mistura das questões avaliadas, inclusive
questões fora da taxonomia. **Como contamos o acerto:** publicar no tema central, num extra ou num nó do mesmo ramo de
um deles (pai ou filho) conta como certo. Publicar em outro subtema do mesmo tema conta como erro.

## Formatos de entrega

### `publicacao.csv`: uma linha para **cada uma das 659 questões**

| Coluna | Valor |
|---|---|
| `id` | Q0001… |
| `decisao` | `publicar`, `revisar` ou `fora` |
| `tema_central_id` | `tema_id` de `temas.csv`. Obrigatório em `publicar` e em `revisar` (a sua sugestão). Vazio em `fora` |
| `extras_ids` | Outros `tema_id` que a questão também cobre, separados por `;`. Pode ficar vazio |
| `confianca` | De 0 a 1: a chance de o tema central estar certo |
| `motivo` | Curto. Obrigatório em `revisar` e em `fora` |

As 100 questões da folha-verdade podem ter qualquer decisão: não entram na avaliação.

### `previsao.txt` (para as questões fora da folha-verdade)

```
precisao_prevista: 0.00   # fração das publicadas que estará certa
precisao_intervalo: 0.00-0.00
cobertura_prevista: 0.00  # fração das questões que você publicou
cobertura_intervalo: 0.00-0.00
```
Depois dessas linhas, explique em texto livre como chegou nos números.

### Regeneração

O repositório precisa ter um comando que gera o `publicacao.csv` a partir destes anexos **sem chamar modelo de
linguagem** (ver seção 4 do PDF).
