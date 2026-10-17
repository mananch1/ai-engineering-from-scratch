# Probabilidade e distribuição

> A probabilidade é a linguagem que a IA usa para expressar incerteza.

**Type:** Learn
**Language:**Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Implementar PMFs e PDFs a partir do zero para distribuições Bernoulli, categóricas, Poisson, uniformes e normais
- Compute o valor esperado, a variância e use o Teorema do Limite Central para explicar por que os Gaussianos dominam
- Construir softmax e log-softmax funções com o truque de estabilidade numérica (subtrair logit max)
- Calcular a perda de entropia cruzada dos logits e conectá-la à probabilidade de log negativo

## O problema

Output de um classificador `[0.03, 0.91, 0.06]`Um modelo de linguagem escolhe a próxima palavra de 50.000 candidatos. Um modelo de difusão gera imagens através da amostragem de distribuições aprendidas.

Cada previsão que um modelo faz é uma distribuição de probabilidade. Cada função de perda mede a distância que a distribuição prevista está da verdadeira. Cada etapa de treinamento ajusta os parâmetros para fazer uma distribuição parecer mais parecida com outra. Sem probabilidade, você não pode ler um único artigo ML, depurar um único modelo ou entender por que sua perda de treinamento é NaN.

## O conceito

### Eventos, Espaços de Amostra e Probabilidade

O espaço de amostra $S$O conjunto de resultados possíveis é um conjunto de eventos. Um evento é um subconjunto do espaço de amostra.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Três axiomas definem toda a probabilidade:
1. $P(A) \ge 0$Para qualquer evento .$A$
2. $P(S) = 1$(algo acontece sempre)
3. $P(A \text{ or } B) = P(A) + P(B)$Quando ?$A$E ...$B$Não podem ocorrer ambas as coisas

Tudo o resto (teorema de Bayes, expectativas, distribuições) segue-se dessas três regras.

### Probabilidade condicional e independência

$P(A \mid B)$é a probabilidade de $A$Dado que$B$- O que aconteceu?

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Dois eventos são independentes quando saber um não diz nada sobre o outro:

```math
\begin{aligned}
\text{Independent:} \quad & P(A \mid B) = P(A) \\
\text{Equivalent to:} \quad & P(A \text{ and } B) = P(A) \cdot P(B)
\end{aligned}
```

As moedas são independentes, mas não os cartões sem substituição.

### Funções de massa de probabilidade vs funções de densidade de probabilidade

As variáveis aleatórias discretas têm uma função de massa de probabilidade (PMF). Cada resultado tem uma probabilidade específica que você pode ler diretamente.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

As variáveis aleatórias contínuas têm uma função de densidade de probabilidade (PDF). A densidade em um único ponto não é uma probabilidade.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Esta distinção é importante no ML. As saídas de classificação são PMFs (opções discretas).

### Distribuições comuns

**Bernoulli:**Um ensaio, dois resultados.

```math
\begin{aligned}
P(X = 1) &= p \\
P(X = 0) &= 1 - p \\
\text{Mean} = p, &\quad \text{Variance} = p(1-p)
\end{aligned}
```

**Categorical:**Um julgamento,$k$Os modelos são classificados em várias classes (output de softmax).

```math
\begin{aligned}
& P(X = i) = p_i, \quad \text{where } \textstyle\sum_i p_i = 1 \\
& \text{Example: } P(\text{cat}) = 0.7, \quad P(\text{dog}) = 0.2, \quad P(\text{bird}) = 0.1
\end{aligned}
```

**Uniform:**Todos os resultados são igualmente prováveis.

```math
\begin{aligned}
\text{Discrete:} \quad & P(X = k) = 1/n \text{ for } k \in \lbrace 1, \ldots, n \rbrace \\
\text{Continuous:} \quad & f(x) = 1/(b-a) \text{ for } x \in [a, b]
\end{aligned}
```

**Normal (Gaussian):**a curva da campanha. Parametrizada pela média ($\mu$) e variação ($\sigma^2$)).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:**Contas de eventos raros em um intervalo fixo.

```math
\begin{aligned}
P(X = k) &= \frac{\lambda^k e^{-\lambda}}{k!} \\
\text{Mean} = \lambda, &\quad \text{Variance} = \lambda
\end{aligned}
```

### Valor esperado e variação

O valor esperado é o resultado médio ponderado.

```math
\begin{aligned}
\text{Discrete:} \quad & \mathbb{E}[X] = \sum_i x_i \, P(X = x_i) \\
\text{Continuous:} \quad & \mathbb{E}[X] = \int x \, f(x) \, dx
\end{aligned}
```

Medidas de variação espalhadas em torno da média.

```math
\begin{aligned}
\mathrm{Var}(X) &= \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \\
\text{Standard deviation} &= \sqrt{\mathrm{Var}(X)}
\end{aligned}
```

Em ML, o valor esperado aparece como a função de perda (perda média sobre a distribuição de dados).

### Distribuições conjuntas e marginais

Uma distribuição conjunta $P(X, Y)$descreve duas variáveis aleatórias juntas.

Exemplo de PMF comum ($X$= tempo,$Y$= guarda-chuva):

| | $Y=0$ (no umbrella) | $Y=1$ (umbrella) | Marginal $P(X)$ |
|---|---|---|---|
| $X=0$ (sun) | 0.40 | 0.10 | $P(X=0) = 0.50$ |
| $X=1$ (rain) | 0.05 | 0.45 | $P(X=1) = 0.50$ |
| **Marginal $P(Y)$** | $P(Y=0) = 0.45$ | $P(Y=1) = 0.55$ | 1.00 |

A distribuição marginal soma a outra variável:

```math
P(X = x) = \sum_{y} P(X = x, Y = y)
```

Os números totais de filas e colunas da tabela acima são os marginais.

### Por que a distribuição normal aparece em todos os lugares

O Teorema do Limite Central: a soma (ou média) de muitas variáveis aleatórias independentes converge para uma distribuição normal, independentemente da distribuição original.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

É por isso que:
- Os erros de medição são aproximadamente normais (muitas pequenas fontes independentes)
- Inicializações de peso em redes neurais usam distribuições normais
- O ruído gradiente no SGD é aproximadamente normal (suma de muitos gradientes de amostra)
- A distribuição normal é a distribuição máxima de entropia para uma determinada média e variância

### Probabilidades de registro

As probabilidades crudas causam problemas numéricos. Multiplicar muitas probabilidades pequenas juntas rapidamente desce para zero.

```math
\begin{aligned}
P(\text{sentence}) &= P(\text{word}_1) \cdot P(\text{word}_2) \cdot \ldots \cdot P(\text{word}_n) \\
&= 0.01 \cdot 0.003 \cdot 0.02 \cdot \ldots \\
&\to 0.0 \quad \text{(underflow after } \sim 30 \text{ terms)}
\end{aligned}
```

As probabilidades de registro corrigem isto.

```math
\begin{aligned}
\log P(\text{sentence}) &= \log P(\text{word}_1) + \log P(\text{word}_2) + \ldots + \log P(\text{word}_n) \\
&= -4.6 + -5.8 + -3.9 + \ldots \\
&\to \text{finite number (no underflow)}
\end{aligned}
```

Regras:
- $\log(a \cdot b) = \log(a) + \log(b)$
- As probabilidades de log são sempre $\le 0$(depois de $0 < P \le 1$)
- Mais negativo = menos provável
- A perda de entropia cruzada é a probabilidade de registro negativo da classe correta

### Softmax como distribuição de probabilidade

As redes neurais emitem pontuações brutas (logits).

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

O truque softmax: subtrair o logit máximo antes de exponenciar para evitar o desbordamento.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax combina softmax e log para estabilidade numérica. PyTorch usa isso internamente para perda de entropia cruzada.

### Amostragem

"Este tipo de análise é o resultado de uma análise de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados de dados
- Deixe de lado as amostras aleatórias que os neurônios para zero
- Ampliação de dados amostras de transformações aleatórias
- Modelos de linguagem amostram o próximo token da distribuição prevista
- Modelos de difusão de amostra de ruído e denotação progressiva

A amostragem a partir de distribuições arbitrárias requer técnicas como amostragem de transformação inversa, amostragem de rejeição ou o truque de reparametrização (usado em VAEs).

```figure
gaussian-pdf
```

## Construí-lo

### Passo 1: Basics of Probability

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Passo 2: PMF e PDF a partir do zero

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Passo 3: Valor esperado e variação

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Passo 4: Amostragem a partir de distribuições

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Passo 5: Softmax e probabilidades de registro

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Passo 6: Teorema de Limite Central demonstração

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Passo 7: Visualização

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Implementações completas com todas as visualizações estão em `code/probability.py`- Não .

## Usá-lo

Com NumPy e SciPy, tudo acima é de uma linha:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Construíste isto do zero, agora sabes o que fazem as chamadas da biblioteca.

## Exercícios

1. Implementar a amostragem de transformação inversa para a distribuição exponencial. Verifique amostragem de valores de 10.000 e comparando o histograma com o PDF verdadeiro.

2. Construa uma tabela de distribuição conjunta para dois dados carregados.

3. Calcule a perda de entropia cruzada para um classificador de 5 classes que produz logits `[2.0, 0.5, -1.0, 3.0, 0.1]`Quando a classe correta é o índice 3. Então verifique sua resposta com PyTorch `nn.CrossEntropyLoss`- Não .

4. Escreva uma função que tome uma lista de probabilidades de registro e retorna a sequência mais provável, a probabilidade total de registro e a probabilidade bruta equivalente. Teste com uma frase de 50 palavras onde cada palavra tem probabilidade de 0,01.

### Indicios

**Exercise 1.**Desenhar`u`de uma distribuição e retorno uniformes $-\ln(1 - u) / \lambda$Verifique se a média da amostra está próxima de$1/\lambda$e que os canhões de histograma seguem $\lambda e^{-\lambda x}$- Não .

**Exercise 2.**Preencha uma tabela de 6x6 onde cada célula é uma probabilidade e a soma de todas as células é 1, então ligue `joint_to_marginals`- Usar .`check_independence`A probabilidade de um dado ser independente é de que cada célula seja igual ao produto de suas duas margens.

**Exercise 3.**Aplicar`softmax`Para as logites e levar $-\ln$O resultado deve ser próximo de 0,421, que é o valor `nn.CrossEntropyLoss`dá para os mesmos logits e alvo.

**Exercise 4.**Adicione as probabilidades de registro; não multiplica probabilidades brutas. Para 50 palavras com probabilidade 0.01, o total deve ser $50 \ln(0.01) \approx -230.26$, e a probabilidade bruta é $10^{-100}$- Não .

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sample space | "All the possibilities" | The set $S$ of every possible outcome of an experiment |
| PMF | "The probability function" | A function that gives the exact probability of each discrete outcome, summing to 1 |
| PDF | "The probability curve" | A density function for continuous variables. Integrate it over an interval to get probability |
| Conditional probability | "Probability given something" | $P(A \mid B) = P(A \text{ and } B) / P(B)$. The foundation of Bayesian thinking and Bayes' theorem |
| Independence | "They don't affect each other" | $P(A \text{ and } B) = P(A) \cdot P(B)$. Knowing one event tells you nothing about the other |
| Expected value | "The average" | The probability-weighted sum of all outcomes. The loss function is an expected value |
| Variance | "How spread out" | The expected squared deviation from the mean. High variance = noisy, unstable estimates |
| Normal distribution | "The bell curve" | $f(x) = (1/\sqrt{2\pi\sigma^2}) \cdot \exp(-(x-\mu)^2/(2\sigma^2))$. Appears everywhere due to the CLT |
| Central Limit Theorem | "Averages become normal" | The mean of many independent samples converges to a normal distribution regardless of the source |
| Joint distribution | "Two variables together" | $P(X, Y)$ describes the probability of every combination of $X$ and $Y$ outcomes |
| Marginal distribution | "Sum out the other variable" | $P(X) = \sum_y P(X, Y)$. Recovers one variable's distribution from the joint |
| Log probability | "Log of the probability" | $\log P(x)$. Turns products into sums, preventing numerical underflow in long sequences |
| Softmax | "Turn scores into probabilities" | $\text{softmax}(z_i) = \exp(z_i) / \sum_j \exp(z_j)$. Maps real-valued logits to a valid probability distribution |
| Cross-entropy | "The loss function" | $-\sum p_{\text{true}} \cdot \log(p_{\text{predicted}})$. Measures how different two distributions are. Lower is better |
| Logits | "Raw model outputs" | Unnormalized scores before softmax. Named after the logistic function |
| Sampling | "Drawing random values" | Generating values according to a probability distribution. How models generate output |

## Mais leitura

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo)- prova visual de que as médias se tornam normais
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf)- uma referência concisa que abrange tudo aqui e mais
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)- por que a estabilidade numérica é importante e como alcançá-la
