# Probabilité et répartition

> La probabilité est le langage utilisé par l'IA pour exprimer l'incertitude.

**Type:** Learn
**Language:**Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Implémenter des PMF et des PDF à partir de zéro pour les distributions Bernoulli, catégorique, Poisson, uniforme et normale
- Compute la valeur attendue, la variance et utilise le théorème de la limite centrale pour expliquer pourquoi les Gaussiens dominent
- Construire les fonctions softmax et log-softmax avec le truc de stabilité numérique (soustraire max logit)
- Calculer la perte d'entropie croisée des logits et la connecter à la probabilité négative de log

## Le problème

Les sorties d' un classifiateur `[0.03, 0.91, 0.06]`Un modèle de langage choisit le mot suivant parmi 50 000 candidats. Un modèle de diffusion génère des images en prélèvant des échantillons à partir de distributions apprises.

Chaque prédiction qu'un modèle fait est une distribution de probabilité. Chaque fonction de perte mesure à quelle distance la distribution prévue est de la vraie. Chaque étape de formation ajuste les paramètres pour que une distribution ressemble davantage à une autre. Sans probabilité, vous ne pouvez pas lire un seul document ML, déboguer un seul modèle ou comprendre pourquoi votre perte de formation est NaN.

## Le concept

### Les événements, les espaces d'échantillonnage et la probabilité

L' espace de l' échantillon $S$est l'ensemble de tous les résultats possibles. Un événement est un sous-ensemble de l'espace de l'échantillon.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Trois axiomes définissent toute probabilité:
1. $P(A) \ge 0$Pour tout événement $A$
2. $P(S) = 1$(quelque chose se passe toujours)
3. $P(A \text{ or } B) = P(A) + P(B)$quand ?$A$et $B$ne peut pas se produire les deux

Tout le reste (théorème de Bayes, attentes, distributions) découle de ces trois règles.

### Probabilité conditionnelle et indépendance

$P(A \mid B)$est la probabilité de $A$vu que $B$Ça s'est passé.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Deux événements sont indépendants quand le fait de connaître l' un ne vous dit rien de l' autre:

```math
\begin{aligned}
\text{Independent:} \quad & P(A \mid B) = P(A) \\
\text{Equivalent to:} \quad & P(A \text{ and } B) = P(A) \cdot P(B)
\end{aligned}
```

Les pièces de monnaie sont indépendantes, les cartes sans remplacement ne le sont pas.

### Les fonctions de masse de probabilité par rapport aux fonctions de densité de probabilité

Les variables aléatoires discrètes ont une fonction de masse de probabilité (PMF). Chaque résultat a une probabilité spécifique que vous pouvez lire directement.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Les variables aléatoires continues ont une fonction de densité de probabilité (PDF). La densité à un seul point n'est pas une probabilité.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Cette distinction est importante dans les ML. Les sorties de classification sont des PMF (choix discrètes).

### Distributions communes

**Bernoulli:**Un essai, deux résultats, des modèles de classification binaire.

```math
\begin{aligned}
P(X = 1) &= p \\
P(X = 0) &= 1 - p \\
\text{Mean} = p, &\quad \text{Variance} = p(1-p)
\end{aligned}
```

**Categorical:**un procès,$k$Les modèles sont classés en plusieurs classes (sortie maximale douce).

```math
\begin{aligned}
& P(X = i) = p_i, \quad \text{where } \textstyle\sum_i p_i = 1 \\
& \text{Example: } P(\text{cat}) = 0.7, \quad P(\text{dog}) = 0.2, \quad P(\text{bird}) = 0.1
\end{aligned}
```

**Uniform:**Il est utilisé pour initialiser au hasard.

```math
\begin{aligned}
\text{Discrete:} \quad & P(X = k) = 1/n \text{ for } k \in \lbrace 1, \ldots, n \rbrace \\
\text{Continuous:} \quad & f(x) = 1/(b-a) \text{ for } x \in [a, b]
\end{aligned}
```

**Normal (Gaussian):**la courbe de cloche. Paramétrisée par la moyenne ($\mu$) et la variance ($\sigma^2$)

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:**Les modèles de taux d'événements sont les suivants:

```math
\begin{aligned}
P(X = k) &= \frac{\lambda^k e^{-\lambda}}{k!} \\
\text{Mean} = \lambda, &\quad \text{Variance} = \lambda
\end{aligned}
```

### La valeur attendue et la variation

La valeur attendue est le résultat moyen pondéré.

```math
\begin{aligned}
\text{Discrete:} \quad & \mathbb{E}[X] = \sum_i x_i \, P(X = x_i) \\
\text{Continuous:} \quad & \mathbb{E}[X] = \int x \, f(x) \, dx
\end{aligned}
```

Les mesures de variance sont réparties autour de la moyenne.

```math
\begin{aligned}
\mathrm{Var}(X) &= \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \\
\text{Standard deviation} &= \sqrt{\mathrm{Var}(X)}
\end{aligned}
```

Dans ML, la valeur attendue apparaît comme la fonction de perte (perte moyenne sur la distribution des données).

### Distributions communes et marginales

Une distribution commune $P(X, Y)$décrive deux variables aléatoires ensemble.

Exemple de PMF commun ($X$= temps,$Y$= parapluie):

| | $Y=0$ (no umbrella) | $Y=1$ (umbrella) | Marginal $P(X)$ |
|---|---|---|---|
| $X=0$ (sun) | 0.40 | 0.10 | $P(X=0) = 0.50$ |
| $X=1$ (rain) | 0.05 | 0.45 | $P(X=1) = 0.50$ |
| **Marginal $P(Y)$** | $P(Y=0) = 0.45$ | $P(Y=1) = 0.55$ | 1.00 |

La distribution marginale résume l'autre variable:

```math
P(X = x) = \sum_{y} P(X = x, Y = y)
```

Les totaux de la rangée et de la colonne figurant dans le tableau ci-dessus sont les marges.

### Pourquoi la distribution normale est partout présente

Le théorème de la limite centrale: la somme (ou la moyenne) de nombreuses variables aléatoires indépendantes converge à une distribution normale, indépendamment de la distribution originale.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

C'est pour ça que:
- Les erreurs de mesure sont approximativement normales (plusieurs petites sources indépendantes)
- Les initialisations de poids dans les réseaux neuraux utilisent des distributions normales
- Le bruit de gradient dans le SGD est approximativement normal (summe de nombreux gradients d'échantillon)
- La distribution normale est la distribution d'entropie maximale pour une moyenne et une variance données

### Probabilités de log

Les probabilités brute causent des problèmes numériques. Multiplier de nombreuses petites probabilités ensemble diminue rapidement à zéro.

```math
\begin{aligned}
P(\text{sentence}) &= P(\text{word}_1) \cdot P(\text{word}_2) \cdot \ldots \cdot P(\text{word}_n) \\
&= 0.01 \cdot 0.003 \cdot 0.02 \cdot \ldots \\
&\to 0.0 \quad \text{(underflow after } \sim 30 \text{ terms)}
\end{aligned}
```

Les probabilités de logs corrigent cela.

```math
\begin{aligned}
\log P(\text{sentence}) &= \log P(\text{word}_1) + \log P(\text{word}_2) + \ldots + \log P(\text{word}_n) \\
&= -4.6 + -5.8 + -3.9 + \ldots \\
&\to \text{finite number (no underflow)}
\end{aligned}
```

Règles:
- $\log(a \cdot b) = \log(a) + \log(b)$
- Les probabilités de log sont toujours $\le 0$(depuis $0 < P \le 1$)
- Plus négatif = moins probable
- La perte de l'entropie croisée est la probabilité de log négatif de la classe correcte

### Softmax comme répartition de probabilité

Les réseaux neuronaux produisent des scores bruts (logits). Softmax les convertit en une distribution de probabilité valide.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Le truc de softmax: soustraire la logite max avant d'exposer pour éviter le débordement.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax combine softmax et log pour la stabilité numérique. PyTorch utilise cela en interne pour la perte d'entropie croisée.

### Prise d'échantillons

Prise d'échantillons: extraction de valeurs aléatoires à partir d'une distribution.
- Jetez des échantillons aléatoires qui ne sont pas des neurones
- Des échantillons d'augmentation des données transformations aléatoires
- Modèles de langage échantillonnent le prochain jeton de la distribution prévue
- Modèles de diffusion échantillonnent le bruit et dénoncent progressivement

L'échantillonnage à partir de distributions arbitraires nécessite des techniques telles que l'échantillonnage de transformation inverse, l'échantillonnage de rejet ou le truc de réparamétrisation (utilisé dans les VAE).

```figure
gaussian-pdf
```

## Faites-le

### Étape 1: Les bases de probabilité

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

### Étape 2: PMF et PDF à partir de zéro

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

### Étape 3: Value et variance attendues

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

### Étape 4: Prise d'échantillons à partir de distributions

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

### Étape 5: Softmax et probabilités de log

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

### Étape 6: démonstration du théorème des limites centrales

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Étape 7: Visualisation

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Des mises en œuvre complètes avec toutes les visualisations sont en `code/probability.py`- Je suis désolé .

## Utilisez-le

Avec NumPy et SciPy, tout ce qui est en haut est un-liners:

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

Vous avez construit ça à partir de zéro, maintenant vous savez ce que font les appels de la bibliothèque.

## Exercices

1. Implémenter l'échantillonnage inverse de transformation pour la distribution exponentielle. Vérifiez en prélèvant des échantillons de 10 000 valeurs et en comparant l'histogramme au vrai PDF.

2. Construisez une table de répartition commune pour deux dés chargés, comptez les répartitions marginales et vérifiez si les dés sont indépendants.

3. Calculer la perte d' entropie croisée pour un classificateur de 5 classes qui produit des logits `[2.0, 0.5, -1.0, 3.0, 0.1]`Lorsque la classe correcte est l'indice 3. Puis vérifiez votre réponse avec PyTorch `nn.CrossEntropyLoss`- Je suis désolé .

4. Écrivez une fonction qui prend une liste de probabilités de log et renvoie la séquence la plus probable, la probabilité totale de log et la probabilité brute équivalente. Testez-la avec une phrase de 50 mots où chaque mot a une probabilité de 0,01.

### Des indices

**Exercise 1.**Tirer`u`de la distribution et du retour uniformes $-\ln(1 - u) / \lambda$Vérifiez que la moyenne de l' échantillon est proche de$1/\lambda$et que les poubelles d' histogrammes suivent $\lambda e^{-\lambda x}$- Je suis désolé .

**Exercise 2.**Remplissez une table de 6x6 où chaque cellule est une probabilité et la somme de toutes les cellules est 1, puis appelez `joint_to_marginals`- Utilisez`check_independence`: les dés ne sont indépendants que si chaque cellule est égale au produit de ses deux marges.

**Exercise 3.**Appliquer `softmax`à la logite et prendre $-\ln$Le résultat doit être proche de 0,421, ce qui est la valeur `nn.CrossEntropyLoss`Il donne pour les mêmes logits et cible.

**Exercise 4.**Ajouter les probabilités de log; ne multipliez pas les probabilités brutes. Pour 50 mots avec probabilité 0.01, le total doit être $50 \ln(0.01) \approx -230.26$, et la probabilité brute est $10^{-100}$- Je suis désolé .

## Les termes clés

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

## Pour en savoir plus

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo)- preuve visuelle de la raison pour laquelle les moyennes deviennent normales
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf)- une référence concise couvrant tout ici et plus encore
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)- pourquoi la stabilité numérique est importante et comment y parvenir
