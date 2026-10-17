# Tratamento de dados desequilibrados

> Quando 99% dos seus dados são "normais", a precisão é uma mentira.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 2, Lessons 01-09 (especially evaluation metrics)
**Time:** ~90 minutes

## Objetivos de aprendizagem

- Implementar SMOTE a partir do zero e explicar como a sobre-amplificação sintética difere da duplicação aleatória
- Avaliação de classificadores desequilibrados usando o coeficiente de correlação F1, AUPRC e Matthews em vez de precisão
- Comparar as estratégias de ponderação de classes, ajuste de limiar e re-amostração e selecionar a abordagem correta para uma determinada relação de desequilíbrio
- Construir um pipeline de dados completamente desequilibrado que combina SMOTE, pesos de classe e otimização de limiar

## O problema

Construímos um modelo de detecção de fraude, obtemos uma precisão de 99,9%, celebramos e percebemos que prevê "não fraude" para cada transação.

Este não é um erro. É a coisa racional a fazer quando apenas 0,1% das transações são fraudulentas. O modelo aprende que sempre adivinhar a classe majoritária minimiza o erro geral. É tecnicamente correto e completamente inútil.

Isto acontece em todos os lugares que são importantes para a classificação real. Diagnóstico de doença: 1% taxa positiva. Intrusão de rede: 0,01% ataques. Defeitos de fabricação: 0,5% defeituosos. Filtragem de spam: 20% spam. Previsão de churn: 5% churners. Quanto mais consequente a classe minoritária, mais rara tendem a ser.

A precisão falha porque trata todas as previsões corretas de forma igual. A rotulagem correta de uma transação legítima e a captura correta de fraude contam como um ponto de precisão. Mas a captura de fraude é toda a razão pela qual o modelo existe. Precisamos de métricas, técnicas e estratégias de treinamento que forcem o modelo a prestar atenção à classe rara, mas importante.

## O conceito

### Por que a precisão falha

Considere um conjunto de dados com 1000 amostras: 990 negativas, 10 positivas. Um modelo que sempre prevê negativos:

|  | Predicted Positive | Predicted Negative |
|--|---|---|
| Actually Positive | 0 (TP) | 10 (FN) |
| Actually Negative | 0 (FP) | 990 (TN) |

```math
\text{Accuracy} = (0 + 990) / 1000 = 99.0\%
```

O modelo detecta zero fraudes, zero doenças, zero defeitos, mas a precisão diz 99%.

### Melhores métricas

**Precision**- Não .$\text{TP} / (\text{TP} + \text{FP})$De tudo que é marcado como positivo, quantos são realmente?

**Recall**- Não .$\text{TP} / (\text{TP} + \text{FN})$De tudo o que é positivo, quantos capturamos?

**F1 Score**- Não .$2 \cdot \text{precision} \cdot \text{recall} / (\text{precision} + \text{recall})$O meio armônico penaliza o desequilíbrio extremo entre precisão e recall mais do que o meio aritmético.

**F-beta Score**- Não .$(1 + \beta^2) \cdot \text{precision} \cdot \text{recall} / (\beta^2 \cdot \text{precision} + \text{recall})$Quando ?$\beta > 1$Quando é que o recordo é mais importante.$\beta < 1$A F2 é comum na detecção de fraude (a falta de fraude é pior do que um falso alarme).

**AUPRC**(Área sob a curva de recall de precisão). Como AUC-ROC, mas mais informativo para dados desequilibrados. Um classificador aleatório tem AUPRC igual à taxa de classe positiva (não 0,5 como ROC). Isso torna as melhorias mais fáceis de ver.

**Matthews Correlation Coefficient**- Não .$(\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}) / \sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}$. varia de -1 a +1. dá uma pontuação elevada apenas quando o modelo faz bem em ambas as classes.

Para o modelo "sempre prever negativo" acima: $\text{precision} = 0/0$(indefinido, muitas vezes definido em 0), $\text{recall} = 0/10 = 0$- Não .$\text{F1} = 0$- Não .$\text{MCC} = 0$Estas métricas identificam corretamente o modelo como inútil.

### O Pipeline de Dados Desbalançados

```mermaid
flowchart TD
    A[Imbalanced Dataset] --> B{Imbalance Ratio?}
    B -->|Mild: 80/20| C[Class Weights]
    B -->|Moderate: 95/5| D[SMOTE + Threshold Tuning]
    B -->|Severe: 99/1| E[SMOTE + Class Weights + Threshold]
    C --> F[Train Model]
    D --> F
    E --> F
    F --> G[Evaluate with F1 / AUPRC / MCC]
    G --> H{Good Enough?}
    H -->|No| I[Try Different Strategy]
    H -->|Yes| J[Deploy with Monitoring]
    I --> B
```

### SMOTE: Técnica de extração de amostras de minorias sintéticas

A sobre-análise aleatória duplica as amostras minoritárias existentes.

O SMOTE cria novas amostras sintéticas de minorias que são plausíveis mas não cópias.

1. Para cada amostra minoritária $x$, encontrar o seu$k$vizinhos mais próximos entre outras amostras de minorias
2. Escolha um vizinho aleatoriamente
3. Criar uma nova amostra no segmento de linha entre $x$E aquele vizinho

A fórmula: `new_sample = x + random(0, 1) * (neighbor - x)`

Isto interpola entre pontos de minoria reais, criando amostras na mesma região do espaço de recursos sem apenas copiar dados existentes.

```mermaid
flowchart LR
    subgraph Original["Original Minority Points"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE Generation"]
        direction TB
        S1["Pick x1, neighbor x2"]
        S2["random t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["Augmented Set"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### Estratégias de amostragem comparadas

**Random Oversampling**: duplicar as amostras de minorias para corresponder à maioria.
- Pros: simples, sem perda de informação
- Cons: duplicados exatos causam sobreajuste, aumenta o tempo de formação

**Random Undersampling**: remover amostras de maioria para corresponder à quantidade de minorias.
- Pros: treinamento rápido, simples
- Cons: descarta dados de maioria potencialmente úteis, maior variância

**SMOTE**A criação de amostras sintéticas de minorias através da interpolação.
- Pros: gera novos pontos de dados, reduz o excesso de adaptação em comparação com o excesso de amostragem aleatória
- Cons: pode criar amostras ruidosas perto do limite de decisão, não conta com a distribuição de classes majoritárias

| Strategy | Data Changed | Risk | When to Use |
|----------|-------------|------|-------------|
| Oversample | Minority duplicated | Overfitting | Small datasets, moderate imbalance |
| Undersample | Majority removed | Information loss | Large datasets, want fast training |
| SMOTE | Synthetic minority added | Boundary noise | Moderate imbalance, enough minority samples for k-NN |

### Peso de classe

Em vez de alterar os dados, alterar a forma como o modelo trata os erros.

Para um problema binário com 950 amostras negativas e 50 amostras positivas:
- Peso para classe negativa = $n_{\text{samples}} / (2 \cdot n_{\text{negative}}) = 1000 / (2 \cdot 950) = 0.526$
- Peso para classe positiva = $n_{\text{samples}} / (2 \cdot n_{\text{positive}}) = 1000 / (2 \cdot 50) = 10.0$

A classe positiva tem 19 vezes o peso. classificar mal uma amostra positiva custa tanto quanto classificar mal 19 amostras negativas. O modelo é forçado a prestar atenção à classe minoritária.

Na regressão logística, isso modifica a função de perda:

```math
\text{weighted\_loss} = -\sum \bigl( w_i \cdot [y_i \log(p_i) + (1-y_i) \log(1-p_i)] \bigr)
```

onde$w_i$depende da classe da amostra $i$- Não .

Os pesos de classe são matematicamente equivalentes ao excesso de amostragem na expectativa, mas sem criar novos pontos de dados.

### Apontação do limiar

A maioria dos classificadores emitem uma probabilidade. O limiar padrão é 0,5: se $P(\text{positive}) \ge 0.5$Quando as classes estão desequilibradas, o limiar ideal é geralmente muito menor.

O processo:
1. Treinar um modelo
2. Obter probabilidades previstas no conjunto de validação
3. Limitações de varredura de 0,0 a 1,0
4. Calcule F1 (ou a métrica escolhida) em cada limiar
5. Escolha o limite que maximize a sua métrica

```mermaid
flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]
```

Um modelo pode produzir$P(\text{fraud}) = 0.15$A diferença entre o valor de uma transação fraudulenta e o valor de uma transação fraudulenta é que, no limiar de 0,5, esta é classificada como não fraudulenta. No limiar de 0,10, é corretamente capturada. A calibração de probabilidade importa menos do que a classificação - desde que a probabilidade de fraude seja maior do que a de não fraude, existe um limiar que as separa.

### Aprendizagem sensível aos custos

Generalização dos pesos de classe. Em vez de custos uniformes, atribuir custos específicos de classificação errada:

| | Predict Positive | Predict Negative |
|--|---|---|
| Actually Positive | 0 (correct) | $C_{\text{FN}} = 100$ |
| Actually Negative | $C_{\text{FP}} = 1$ | 0 (correct) |

O modelo otimiza o custo total, não o número total de erros.

Esta é a abordagem mais básica quando se pode estimar os custos no mundo real. Um diagnóstico de câncer perdido tem um custo muito diferente de um falso alarme que leva a uma biópsia extra.

### Diagrama de fluxo de decisão

```mermaid
flowchart TD
    A[Start: Imbalanced Dataset] --> B{How imbalanced?}
    B -->|"< 70/30"| C["Mild: try class weights first"]
    B -->|"70/30 to 95/5"| D["Moderate: SMOTE + class weights"]
    B -->|"> 95/5"| E["Severe: combine multiple strategies"]
    C --> F{Enough data?}
    D --> F
    E --> F
    F -->|"< 1000 samples"| G["Oversample or SMOTE, avoid undersampling"]
    F -->|"1000-10000"| H["SMOTE + threshold tuning"]
    F -->|"> 10000"| I["Undersampling OK, or class weights"]
    G --> J[Train + Evaluate with F1/AUPRC]
    H --> J
    I --> J
    J --> K{Recall high enough?}
    K -->|No| L[Lower threshold]
    K -->|Yes| M{Precision acceptable?}
    M -->|No| N[Raise threshold or add features]
    M -->|Yes| O[Ship it]
```

```figure
class-imbalance
```

## Construí-lo

### Passo 1: Gerar um conjunto de dados desequilibrado

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### Passo 2: SMOTE a partir do zero

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### Passo 3: Superamento e sub-análise aleatórias

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### Passo 4: Regressão logística com pesos de classe

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### Passo 5: Ajuste de limiar

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### Passo 6: Funções de avaliação

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### Passo 7: Comparar todas as abordagens

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

O arquivo de código executa tudo isto num único script e imprime os resultados.

## Usá-lo

Com a aprendizagem escitada e a aprendizagem desequilibrada, estas técnicas são de linha única:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

As implementações do zero mostram exatamente o que cada técnica faz. SMOTE é apenas interpolação k-NN na classe minoritária. Pesos de classe multiplicam a perda.

## Envia-o

Esta lição produz:
- `outputs/skill-imbalanced-data.md`-- uma lista de controlo das decisões para lidar com os problemas de classificação desequilibrados

## Exercícios

1. **Borderline-SMOTE**A aplicação de SMOTE é modificada para gerar apenas amostras sintéticas para pontos minoritários próximos ao limite de decisão (aqueles cujos vizinhos k mais próximos incluem amostras de classes majoritárias).

2. **Cost matrix optimization**A função que assume uma matriz de custos e retorna previsões ótimas que minimizam o custo esperado. Teste com diferentes proporções de custos (1:10, 1:100, 1:1000) e trace como a diferença de precisão-recall muda.

3. **Threshold calibration**A calibração não altera a classificação (AUC permanece a mesma) mas torna as probabilidades mais significativas.

4. **Ensemble with balanced bagging**A partir da data de lançamento, a equipe de treinamento deve avaliar a quantidade de dados que podem ser utilizados para a execução de um modelo de bootstrap.

5. **Imbalance ratio experiment**A SMOTE é uma das principais ferramentas de desenvolvimento de dados para a análise de dados.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Class imbalance | "One class has way more samples" | The distribution of classes in the dataset is significantly skewed, causing models to favor the majority class |
| SMOTE | "Synthetic oversampling" | Creates new minority samples by interpolating between existing minority samples and their k-nearest minority neighbors |
| Class weights | "Making errors on rare classes more expensive" | Multiplying the loss function by class-specific weights so the model penalizes minority misclassification more heavily |
| Threshold tuning | "Moving the decision boundary" | Changing the probability cutoff for classification from the default 0.5 to a value that optimizes the desired metric |
| Precision-recall tradeoff | "You cannot have both" | Lowering the threshold catches more positives (higher recall) but also flags more false positives (lower precision), and vice versa |
| AUPRC | "Area under the PR curve" | Summarizes the precision-recall curve into a single number; more informative than AUC-ROC when classes are heavily imbalanced |
| Matthews Correlation Coefficient | "The balanced metric" | A correlation between predicted and actual labels that produces a high score only when the model performs well on both classes |
| Cost-sensitive learning | "Different mistakes cost different amounts" | Incorporating real-world misclassification costs into the training objective so the model optimizes for total cost, not error count |
| Random oversampling | "Duplicate the minority" | Repeating minority class samples to balance class counts; simple but risks overfitting to duplicated points |

## Mais leitura

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813)-- o artigo original do SMOTE, ainda o trabalho mais citado sobre aprendizagem desequilibrada
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907)-- uma pesquisa abrangente que abrange abordagens de amostragem, de custos e de algoritmos sensíveis
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)-- Biblioteca Python com variantes SMOTE, estratégias de sub-esampulagem e integração de pipeline
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)-- quando e por que preferir curvas de PR a curvas de ROC para problemas desequilibrados
