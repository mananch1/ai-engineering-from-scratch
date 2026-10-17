# Avaliação: Indicadores de referência, Evals, Harness LM

> Lei de Goodhart: quando uma medida se torna um alvo, deixa de ser uma boa medida. Todos os jogos de laboratório fronteiriços são de referência. As pontuações da MMLU aumentam enquanto os modelos ainda não conseguem contar com confiança o número de Rs em "frango". A única avaliação que importa é a Vossa avaliação - em Vossa tarefa, com os Vossos dados.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-05 (LLMs from Scratch)
**Time:** ~90 minutes

## Objetivos de aprendizagem

- Construir um arame de avaliação personalizado que execute referências de escolha múltipla e de limite aberto contra um modelo de linguagem
- Explicar por que os valores-relatores padrão (MMLU, HumanEval) saturam e não diferenciam os modelos de fronteira
- Implementar avaliações específicas de tarefas com métricas adequadas: correspondência exacta, F1, BLEU e pontuação de LLM-as-judge
- Desenhar uma suíte de avaliação personalizada voltada para o seu caso de uso específico em vez de confiar apenas em quadros de classificação públicos

## O problema

O MMLU foi publicado em 2020 com 15.908 perguntas em 57 assuntos. Em três anos, os modelos de fronteira saturaram-no. GPT-4 obteve 86,4%. Claude 3 Opus obteve 86,8%. Llama 3 405B obteve 88,6%.

Enquanto isso, esses mesmos modelos falham em tarefas que uma criança de 10 anos lida sem pensar. Claude 3.5 Sonnet, com uma pontuação de 88,7% na MMLU, inicialmente não podia contar as letras em "frango" -- uma tarefa que requer zero conhecimento do mundo e zero raciocínio, apenas iteração de nível de personagens. O HumanEval testa a geração de código com 164 problemas. Os modelos têm 90% mais, enquanto ainda produzem código que cai em casos de borda que qualquer desenvolvedor mais novo apanha.

A diferença entre o desempenho dos índices de referência e a confiabilidade do mundo real é o problema central da avaliação do MLL. Os índices de referência dizem como um modelo funciona no índice de referência. Eles não dizem quase nada sobre como esse modelo irá executar em sua tarefa específica, com os seus dados específicos, sob os seus modos de falha específicos. Se você está a construir um bot de suporte ao cliente, o MMLU é irrelevante. Se estiver a construir um assistente de código, o HumanEval só abrange a geração de nível de função - não diz nada sobre depurar, refactorar ou explicar código em arquivos.

Precisamos de avaliações personalizadas. Não porque os benchmarks sejam inúteis - são úteis para a seleção aproximada de modelos - mas porque a avaliação final deve corresponder exatamente às condições de implantação.

## O conceito

### A paisagem de Eval

Existem três categorias de avaliação, cada uma com um custo e uma qualidade de sinal diferentes.

**Benchmarks**Os modelos de teste são padronizados. MMLU, HumanEval, SWE-bench, MATH, ARC, HellaSwag. Você executa um modelo contra o índice de referência e obtém uma pontuação. A vantagem: todos usam o mesmo teste, para que você possa comparar os modelos. A desvantagem: modelos e dados de treinamento contaminam cada vez mais esses índices de referência. Os laboratórios treinam em dados que incluem perguntas de referência. As pontuações aumentam. A capacidade pode não.

**Custom evals**Os dados de teste são os que você constrói para seu caso de uso específico. Você define as entradas, as saídas esperadas e a função de pontuação. Um resumo de documento legal é avaliado em documentos legais. Um gerador de SQL é avaliado em seu esquema de banco de dados. Estes são caros de criar, mas são a única avaliação que prevê o desempenho da produção.

**Human evals**O Chatbot Arena tem coletado mais de 2 milhões de votos de preferência humana em mais de 100 modelos.$0.10-$- 2,00 por acórdão) e velocidade (horas a dias).

```mermaid
graph TD
    subgraph Eval["Evaluation Landscape"]
        direction LR
        B["Benchmarks\n(MMLU, HumanEval)\nCheap, standardized\nGameable, stale"]
        C["Custom Evals\nYour task, your data\nHighest signal\nExpensive to build"]
        H["Human Evals\n(Chatbot Arena)\nGold standard\nSlow, costly"]
    end

    B -->|"rough model selection"| C
    C -->|"ambiguous cases"| H

    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
```

### Por que os índices de referência se quebram

Três mecanismos fazem com que as pontuações de referência deixem de refletir a capacidade real.

**Data contamination.**Os corpos de treinamento raspam a internet. As perguntas de referência estão em directo na internet. Os modelos veem as respostas durante o treinamento. Isto não é fazer trampa no sentido tradicional - os laboratórios não incluem intencionalmente dados de referência. Mas a raspação em escala web torna quase impossível excluir.

**Teaching to the test.**Os laboratórios otimizam as misturas de treinamento para o desempenho de referência. Se 5% da mistura de treinamento é uma escolha múltipla de estilo MMLU, o modelo aprende o formato e a distribuição da resposta. MMLU é uma escolha múltipla de quatro vias. Os modelos aprendem que a distribuição da resposta é aproximadamente uniforme em A / B / C / D, o que ajuda mesmo quando o modelo não conhece a resposta.

**Saturation.**Quando cada modelo de fronteira obtém uma pontuação de 85-90% em um índice de referência, o índice de referência deixa de discriminar. Os restantes 10-15% das perguntas podem ser ambíguas, erroneamente rotuladas ou exigir conhecimento obscuro do domínio. Melhorar de 87% para 89% no MMLU pode significar que o modelo memorizou duas perguntas mais obscuras, não que se tornou mais inteligente.

### Perplexidade: um rápido exame de saúde

A perplexidade mede o quão surpreendente um modelo é por uma sequência de tokens.

```math
\text{PPL} = \exp\left(-\frac{1}{N} \sum_i \log P(\text{token}_i \mid \text{context})\right)
```

Uma perplexidade de 10 significa que o modelo é, em média, tão incerto quanto escolher uniformemente entre 10 opções em cada posição de token. Baixo é melhor. GPT-2 obtém uma perplexidade de ~30 no WikiText-103. GPT-3 obtém ~20. Llama 3 8B obtém ~7.

A perplexidade é útil para comparar modelos no mesmo conjunto de testes, mas tem pontos cegos. Um modelo pode ter baixa perplexidade por ser bom em prever padrões comuns, enquanto ser terrível em padrões raros, mas importantes. Também não diz nada sobre a instrução seguindo, raciocínio ou precisão factual.

### Mestrado em Direito

Use um modelo forte para avaliar a produção de um modelo mais fraco. A ideia é simples: peça ao GPT-4o ou Claude Sonnet para avaliar uma resposta em uma escala de 1-5 para corretura, utilidade e segurança. Isso custa cerca de $0,01 por julgamento com o GPT-4o-mini e correlaciona-se surpreendentemente bem com os julgamentos humanos - cerca de 80% de acordo na maioria das tarefas.

O prompt de pontuação importa mais do que o modelo. Um prompt vaga ("Rate this response") produz pontuações barulhentas. Um prompt estruturado com uma rubrica ("Score 5 se a resposta é factualmente correta e cita uma fonte, 4 se correta, mas não citada, 3 se parcialmente correta...") produz pontuações consistentes e reprodutíveis.

Modos de falha: os modelos de juiz mostram preconceito de posição (preferem a primeira resposta em comparações em pares), preconceito de verbosidade (preferem respostas mais longas) e auto-preferência (GPT-4 avalia GPT-4 de saída superior a Claude equivalentes).

### Classificações ELO de comparações de pares

A abordagem do Chatbot Arena. Mostre duas respostas ao mesmo pedido de diferentes modelos. Um humano (ou juiz de LLM) escolhe o melhor. De milhares dessas comparações, calcula uma classificação ELO para cada modelo - o mesmo sistema usado em xadrez.

Vantagens do ELO: o ranking relativo é mais confiável do que a pontuação absoluta, lida com ligações graciosamente e converge com menos comparações do que marcar cada saída de forma independente.

```mermaid
graph LR
    subgraph ELO["ELO Rating Pipeline"]
        direction TB
        P["Prompt"] --> MA["Model A Output"]
        P --> MB["Model B Output"]
        MA --> J["Judge\n(Human or LLM)"]
        MB --> J
        J --> W["A Wins / B Wins / Tie"]
        W --> E["ELO Update\nK=32"]
    end

    style P fill:#1a1a2e,stroke:#0f3460,color:#fff
    style J fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### Estruturas equivalentes

**lm-evaluation-harness**(EleutherAI): o framework de avaliação de código aberto padrão. Suporta 200+ benchmarks. Execute qualquer modelo de Hugging Face contra MMLU, HellaSwag, ARC, etc. com um comando. usado pelo Open LLM Leaderboard.

**RAGAS**A avaliação da qualidade de informação e de informação é uma das principais medidas de avaliação da qualidade de informação e da informação.

**promptfoo**A avaliação é feita com base em configuração para engenharia de prompt. Defina casos de teste em YAML, corre contra vários modelos, obtenha um relatório de passagem/falha. Útil para instruções de teste de regressão - certifique-se de que uma alteração de prompt não quebra casos de teste existentes.

### Construir Evals à Custom

A única avaliação que importa para a produção.

1. **Define the task.**O que exatamente deve fazer o modelo? Seja preciso. "Responda às perguntas" é muito vaga. "Dado um e-mail de reclamação do cliente, extrair o nome do produto, categoria do problema e sentimento" é uma tarefa que você pode avaliar.

2. **Create test cases.**Minimo 50 para um prototipo de avaliação, 200+ para produção. Cada caso de teste é um par (input, expected_output). Incluir casos de borda: entradas vazias, entradas adversárias, entradas ambíguas, entradas em outras línguas.

3. **Define scoring.**A combinação exacta para as saídas estruturadas. BLEU/ROUGE para a semelhança de texto. LLM-as-judge para a qualidade aberta. F1 para tarefas de extracção. Combine múltiplas métricas com pesos.

4. **Automate.**Cada avaliação é executada com um comando, sem passos manuais, armazenando resultados em um formato que permite comparação ao longo do tempo.

5. **Track over time.**Uma pontuação de avaliação não tem sentido em isolamento. Você precisa da linha de tendência. A pontuação melhorou após a última mudança de aviso? Regressou após a mudança de modelos? Versionar sua avaliação ao lado de suas instruções.

| Eval Type | Cost per judgment | Agreement with humans | Best for |
|-----------|------------------|----------------------|----------|
| Exact match | ~$0 | 100% (when applicable) | Structured output, classification |
| BLEU/ROUGE | ~$0 | ~60% | Translation, summarization |
| LLM-as-judge | ~$0.01 | ~80% | Open-ended generation |
| Human eval | $0.10-$2.00 | N/A (is the ground truth) | Ambiguous, high-stakes tasks |

```figure
perplexity-loss
```

## Construí-lo

### Passo 1: Um quadro mínimo de igualdade

Defina as abstrações principais. Um caso de eval tem uma entrada, uma saída esperada e um ditado opcional de metadados. Um marcador toma uma previsão e uma referência e retorna uma pontuação entre 0 e 1.

```python
import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results
```

### Passo 2: pontuação das funções

Construir uma correspondência exata, um token F1, e um marcador de LLM como juiz simulado.

```python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def llm_judge_simulated(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)
```

### Passo 3: Sistema de classificação ELO

Implementar comparações em pares com atualizações ELO. Este é exatamente o sistema Chatbot Arena usa para classificar modelos.

```python
class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])
```

### Passo 4: Calculação de perplexidade

Compute perplexidade usando probabilidades de token. na prática você obtém estes dos logits do modelo. Aqui simulamos com uma distribuição de probabilidade.

```python
import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

def token_log_probs_simulated(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumerate(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs
```

### Passo 5: Resultados agregados

Compute estatísticas de resumo em uma execução de avaliação: média, média, taxa de aprovação em um limiar e desagregações por métrica.

```python
def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_rate": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f"  {suite_name} Summary")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n  {metric}:")
        print(f"    Mean:      {stats['mean']:.3f}")
        print(f"    Median:    {stats['median']:.3f}")
        print(f"    Std:       {stats['std']:.3f}")
        print(f"    Range:     [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Pass rate: {stats['pass_rate']:.1%} (threshold >= 0.8)")
        print(f"    N:         {stats['n']}")
```

### Passo 6: Caminhe o oleoduto completo

Defina uma tarefa, crie casos de teste, simule dois modelos, execute avaliações, compute o ELO a partir de comparações em pares e imprima o ranking.

```python
def demo_model_good(prompt):
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "What language is PyTorch written in?": "Python and C++",
        "What is the boiling point of water?": "100 degrees Celsius",
    }
    return responses.get(prompt, "I don't know")

def demo_model_bad(prompt):
    responses = {
        "What is the capital of France?": "Paris is the capital city of France",
        "What is 2 + 2?": "The answer is four",
        "Who wrote Hamlet?": "Shakespeare",
        "What language is PyTorch written in?": "Python",
        "What is the boiling point of water?": "212 Fahrenheit",
    }
    return responses.get(prompt, "Unknown")

cases = [
    EvalCase("What is the capital of France?", "Paris"),
    EvalCase("What is 2 + 2?", "4"),
    EvalCase("Who wrote Hamlet?", "William Shakespeare"),
    EvalCase("What language is PyTorch written in?", "Python and C++"),
    EvalCase("What is the boiling point of water?", "100 degrees Celsius"),
]

suite = EvalSuite(
    name="General Knowledge",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulated,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "Model A (concise)")
print_summary(summarize_results(results_bad), "Model B (verbose)")
```

O modelo "bom" dá respostas exatas. O modelo "mau" dá parafrases verbais. A correspondência exata puniu severamente o modelo verbais. O token F1 e o LLM como juiz são mais perdoadores. Isso ilustra por que a escolha métrica importa: o mesmo modelo parece ótimo ou terrível dependendo de como você o pontua.

### Passo 7: Torneio ELO

Faça comparações em pares entre modelos em várias rodadas.

```python
elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO Leaderboard:")
for name, rating in elo.leaderboard():
    print(f"  {name}: {rating:.0f}")
```

### Passo 8: Perplexidade Comparativa

Compare a perplexidade entre "modelos" de diferentes níveis de qualidade.

```python
test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "Strong model"), (0.7, "Medium model"), (0.4, "Weak model")]:
    log_probs = token_log_probs_simulated(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f"  {label} (quality={quality}): perplexity = {ppl:.2f}")
```

## Usá-lo

### - Valorização de uso (EleutherAI)

A ferramenta padrão para executar valores de referência em qualquer modelo.

```python
# pip install lm-eval
# Command line:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])
```

### promptfoo

Defina testes em YAML e execute contra vários provedores.

```yaml
# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "Answer in one word: {{question}}"

tests:
  - vars:
      question: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
  - vars:
      question: "What is 2 + 2?"
    assert:
      - type: equals
        value: "4"
```

### RAGAS para avaliação de RAG

```python
# pip install ragas
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)
```

O RAGAS mede o que os avaliadores genéricos não têm: se a resposta do modelo está fundamentada no contexto recuperado, não apenas se a resposta é "correcta" no abstracto.

## Envia-o

Esta lição produz`outputs/prompt-eval-designer.md`-- um prompt reutilizável que desenha conjuntos de avaliação personalizados para qualquer tarefa. Dê-lhe uma descrição da tarefa e ele gera casos de teste, funções de pontuação e uma recomendação de limiar de passagem / falha.

Também produz `outputs/skill-llm-evaluation.md`-- um quadro de decisão para escolher a estratégia de avaliação certa com base no tipo de tarefa, orçamento e requisitos de latência.

## Exercícios

1. Adicione um marcador de "consistência" que corre a mesma entrada através do modelo 5 vezes e mede a frequência com que as saídas coincidem.

2. Expanda o rastreador ELO para suportar múltiplas funções de juiz (paralelas exatas, F1, LLM-as-judge) e pesem-nas. Compare como o ranking muda quando você pesa paralelas exatas em relação a F1.

3. Crie um conjunto de avaliações para uma tarefa específica: classificação de e-mails em 5 categorias. Crie 100 casos de teste com exemplos diversos, incluindo casos de borda (e-mails que podem pertencer a várias categorias, e-mails vazios, e-mails em outras línguas). Messa como diferentes "modelos" (baseado em regras, correspondência de palavras-chave, LLM simulado) funcionam.

4. Implementar a detecção de contaminação: tendo em conta um conjunto de perguntas de avaliação e um corpo de formação, verifique a percentagem de perguntas de avaliação (ou parafrases próximas) que aparecem nos dados de formação.

5. Construir uma ferramenta de "modelo diferencial". Dados os resultados de avaliação de duas versões de modelo, salientar quais casos de teste específicos melhoraram, que regressaram e que permaneceram iguais. Este é o equivalente de avaliação de um código diferencial - essencial para entender se uma mudança ajudou ou prejudicou.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| MMLU | "The benchmark" | Massive Multitask Language Understanding -- 15,908 multiple choice questions across 57 subjects, saturated above 88% by 2025 |
| HumanEval | "Code eval" | 164 Python function-completion problems from OpenAI, tests only isolated function generation |
| SWE-bench | "Real coding eval" | 2,294 GitHub issues from 12 Python repos, measures end-to-end bug fixing including test generation |
| Perplexity | "How confused the model is" | $\exp(-\mathop{\mathrm{avg}}(\log P(\text{token}_i \text{ given context})))$ -- lower means the model assigns higher probability to the actual tokens |
| ELO rating | "Chess ranking for models" | A relative skill rating computed from pairwise win/loss records, used by Chatbot Arena to rank 100+ models |
| LLM-as-judge | "Using AI to grade AI" | A strong model scores a weaker model's outputs against a rubric, ~80% agreement with human judges at ~$0.01/judgment |
| Data contamination | "The model saw the test" | Training data includes benchmark questions, inflating scores without improving real capability |
| Eval suite | "A bunch of tests" | A versioned collection of (input, expected_output, scorer) triples that measure a specific capability |
| Pass rate | "What percentage it gets right" | Fraction of eval cases scoring above a threshold -- more actionable than mean score because it measures reliability |
| Chatbot Arena | "Model ranking website" | LMSYS platform with 2M+ human preference votes, producing the most trusted LLM leaderboard via ELO ratings |

## Mais leitura

- [Hendrycks et al., 2021 -- "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300)- o artigo da MMLU, que continua a ser o ponto de referência mais citado para o LLM, apesar da sua saturação
- [Chen et al., 2021 -- "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374)-- o artigo HumanEval da OpenAI, estabeleceu uma metodologia de avaliação da geração de código
- [Zheng et al., 2023 -- "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685)-- análise sistemática da utilização de LLM para avaliar LLM, incluindo as conclusões de viés de posição e viés de verbosidade
- [Arena (formerly LMSYS Chatbot Arena)](https://arena.ai/leaderboard)-- plataforma de comparação de modelos com crowdsourced com 2M+ votos, o ranking mais confiável do mundo real LLM
