# Traiter les données déséquilibrées

> Quand 99% de vos données sont "normales", l'exactitude est un mensonge.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 2, Lessons 01-09 (especially evaluation metrics)
**Time:** ~90 minutes

## Objectifs d'apprentissage

- Mettre en œuvre SMOTE à partir de zéro et expliquer comment la suréchantillonnage synthétique diffère de la duplication aléatoire
- Évaluer les classifiants déséquilibrés en utilisant le coefficient de corrélation F1, AUPRC et Matthews au lieu de l'exactitude
- Comparer les stratégies de pondération de classe, de réglage des seuils et de reéchantillonnage et choisir la bonne approche pour un ratio de déséquilibre donné
- Construire un pipeline de données complètement déséquilibré qui combine SMOTE, poids de classe et optimisation des seuils

## Le problème

Vous construisez un modèle de détection de fraude, il a une précision de 99,9%, vous célébrez, et vous réalisez qu'il prédit " pas de fraude " pour chaque transaction.

Il s'agit d'une erreur qui n'est pas une erreur. C'est la chose rationnelle à faire lorsque seulement 0,1% des transactions sont frauduleuses. Le modèle apprend que deviner toujours la classe majoritaire réduit au minimum l'erreur globale. C'est techniquement correct et complètement inutile.

C'est le cas partout où il y a des questions de classification réelles. Diagnostic de maladie: taux positif de 1%. Intrusion au réseau: attaques de 0,01%. défauts de fabrication: 0,5% défectueux. Filtrage de spam: 20% de spam. Prédiction de churn: 5% de churners. Plus la classe minoritaire est conséquente, plus elle a tendance à être rare.

La précision échoue parce qu'elle traite toutes les prédictions correctes de la même manière. Étiqueter correctement une transaction légitime et attraper correctement la fraude comptent tous deux comme un point de précision. Mais attraper la fraude est la raison entière de l'existence du modèle. Nous avons besoin de mesures, de techniques et de stratégies de formation qui forcent le modèle à prêter attention à la classe rare mais importante.

## Le concept

### Pourquoi l'exactitude échoue

Considérez un ensemble de données avec 1000 échantillons: 990 négatifs, 10 positifs.

|  | Predicted Positive | Predicted Negative |
|--|---|---|
| Actually Positive | 0 (TP) | 10 (FN) |
| Actually Negative | 0 (FP) | 990 (TN) |

```math
\text{Accuracy} = (0 + 990) / 1000 = 99.0\%
```

Le modèle ne détecte aucune fraude, aucune maladie, aucun défaut, mais la précision est de 99%.

### Meilleures mesures

**Precision**- Je suis là.$\text{TP} / (\text{TP} + \text{FP})$De tout ce qui est marqué comme positif, combien sont-ils réellement ?

**Recall**- Je suis là.$\text{TP} / (\text{TP} + \text{FN})$De tout ce qui est positif, combien avons-nous attrapé ?

**F1 Score**- Je suis là.$2 \cdot \text{precision} \cdot \text{recall} / (\text{precision} + \text{recall})$La moyenne harmonieuse pénalise un déséquilibre extrême entre précision et rappel plus que la moyenne arithmétique.

**F-beta Score**- Je suis là.$(1 + \beta^2) \cdot \text{precision} \cdot \text{recall} / (\beta^2 \cdot \text{precision} + \text{recall})$- Quand ?$\beta > 1$Quand vous vous souvenez de ce qui s'est passé,$\beta < 1$La détection de la fraude est courante (la fraude manquante est pire qu'une fausse alarme).

**AUPRC**(Area Under Precision-Recall Curve). Comme AUC-ROC mais plus informatif pour les données déséquilibrées. Un classificateur aléatoire a AUPRC égal au taux de classe positive (pas 0,5 comme ROC). Cela facilite la perception des améliorations.

**Matthews Correlation Coefficient**- Je suis là.$(\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}) / \sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}$. varie de -1 à +1. ne donne un score élevé que lorsque le modèle fonctionne bien sur les deux classes.

Pour le modèle "prédire toujours négatif" ci-dessus: $\text{precision} = 0/0$(indéfinie, souvent définie à 0), $\text{recall} = 0/10 = 0$- Je suis là .$\text{F1} = 0$- Je suis là .$\text{MCC} = 0$Ces mesures identifient correctement le modèle comme sans valeur.

### Le pipeline de données déséquilibrées

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

### SMOTE: technique de dépistage des échantillons de la minorité synthétique

Le suréchantillonnage aléatoire duplique les échantillons existants de minorités. Cela fonctionne, mais risque de sur-adapter car le modèle voit à plusieurs reprises les mêmes points.

SMOTE crée de nouveaux échantillons synthétiques de minorités qui sont plausibles mais pas des copies.

1. Pour chaque échantillon minoritaire $x$, trouver son .$k$les voisins les plus proches parmi les autres échantillons de minorités
2. Choisissez un voisin au hasard
3. Créer un nouveau échantillon sur le segment de ligne entre $x$et ce voisin

La formule: `new_sample = x + random(0, 1) * (neighbor - x)`

Cela interpelle entre des points de minorité réels, créant des échantillons dans la même région de l'espace de fonctionnalités sans simplement copier les données existantes.

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

### Comparer les stratégies de prélèvement

**Random Oversampling**: dupliquer les échantillons minoritaires pour correspondre au nombre majoritaire.
- Avantages: simple, sans perte d'information
- Cons: les duplicates exactes provoquent un surmatch, augmentent le temps de formation

**Random Undersampling**: supprimer les échantillons majoritaires pour correspondre au nombre de minorités.
- Avantages: entraînement rapide, simple
- Les inconvénients: jeter des données majoritaires potentiellement utiles, plus grande variance

**SMOTE**: créer des échantillons de minorités synthétiques par interpolation.
- Avantages: génère de nouveaux points de données, réduit le surcodage par rapport à l'échantillonnage aléatoire
- Cons: peut créer des échantillons bruyants près de la limite de décision, ne tient pas compte de la distribution de la classe majoritaire

| Strategy | Data Changed | Risk | When to Use |
|----------|-------------|------|-------------|
| Oversample | Minority duplicated | Overfitting | Small datasets, moderate imbalance |
| Undersample | Majority removed | Information loss | Large datasets, want fast training |
| SMOTE | Synthetic minority added | Boundary noise | Moderate imbalance, enough minority samples for k-NN |

### Poids de classe

Au lieu de modifier les données, modifiez la façon dont le modèle traite les erreurs.

Pour un problème binaire avec 950 échantillons négatifs et 50 échantillons positifs:
- Poids pour la classe négative = $n_{\text{samples}} / (2 \cdot n_{\text{negative}}) = 1000 / (2 \cdot 950) = 0.526$
- Poids pour la classe positive = $n_{\text{samples}} / (2 \cdot n_{\text{positive}}) = 1000 / (2 \cdot 50) = 10.0$

La classe positive a 19 fois le poids. Une mauvaise classification d'un échantillon positif coûte autant que 19 échantillons négatifs.

En régression logistique, cela modifie la fonction de perte:

```math
\text{weighted\_loss} = -\sum \bigl( w_i \cdot [y_i \log(p_i) + (1-y_i) \log(1-p_i)] \bigr)
```

où $w_i$dépend de la classe de l'échantillon $i$- Je suis désolé .

Les poids de classe sont mathématiquement équivalents à un suréchantillonnage dans l'attente, mais sans créer de nouveaux points de données.

### Réglage du seuil

La plupart des classifiateurs produisent une probabilité.$P(\text{positive}) \ge 0.5$La valeur de la valeur de la classe est généralement beaucoup plus faible.

Le processus:
1. Formez un modèle
2. Obtenez les probabilités prévues sur l'ensemble de validation
3. Les seuils de balayage de 0,0 à 1,0
4. Comptez F1 (ou votre métrique choisie) à chaque seuil
5. Choisissez le seuil qui maximize votre métrique

```mermaid
flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]
```

Un modèle pourrait produire $P(\text{fraud}) = 0.15$Pour une transaction frauduleuse. Au seuil de 0,5, cela est classé comme non frauduleux. Au seuil de 0,10, il est correctement capturé. La calibration de probabilité compte moins que le classement - tant que la fraude a des probabilités plus élevées que la non-fraude, il existe un seuil qui les sépare.

### Un apprentissage peu coûteux

Généralisation des poids de classe: au lieu de coûts uniformes, attribuer des coûts de classification erronée spécifiques:

| | Predict Positive | Predict Negative |
|--|---|---|
| Actually Positive | 0 (correct) | $C_{\text{FN}} = 100$ |
| Actually Negative | $C_{\text{FP}} = 1$ | 0 (correct) |

Le manque d'une transaction frauduleuse (FN) coûte 100 fois plus cher qu'une fausse alarme (FP).

C'est l'approche la plus fondée sur les principes pour estimer les coûts réels. Un diagnostic de cancer manqué a un coût très différent d'une fausse alarme qui conduit à une biopsie supplémentaire.

### Tableau de débit des décisions

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

## Faites-le

### Étape 1: Générer un ensemble de données déséquilibré

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

### Étape 2: SMOTE à partir de zéro

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

### Étape 3: Prélèvement aléatoire de suréchantillonnage et de sous-échantillonnage

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

### Étape 4: Régression logistique avec poids de classe

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

### Étape 5: réglage du seuil

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

### Étape 6: Fonctions d'évaluation

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

### Étape 7: Comparer toutes les approches

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

Le fichier de code exécute tout cela dans un seul script et imprime les résultats.

## Utilisez-le

Avec le scikit-apprentissage et le déséquilibre-apprentissage, ces techniques sont un-liners:

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

Les implémentations à partir de zéro montrent exactement ce que chaque technique fait. SMOTE est juste une interpolation k-NN sur la classe minoritaire. Les poids de classe multiplient la perte.

## La faire partir

Cette leçon donne:
- `outputs/skill-imbalanced-data.md`-- une liste de contrôle des décisions pour gérer les problèmes de classification déséquilibrés

## Exercices

1. **Borderline-SMOTE**: modifier la mise en œuvre de SMOTE pour ne générer que des échantillons synthétiques pour les points minoritaires proches de la limite de décision (ceux dont les voisins k-proches comprennent des échantillons de classes majoritaires).

2. **Cost matrix optimization**: mettre en œuvre un apprentissage sensible aux coûts où la matrice de coûts est un paramètre. Créer une fonction qui prend une matrice de coûts et renvoie des prédictions optimales qui minimisent les coûts attendus. Testez avec différents rapports de coûts (1:10, 1:100, 1:1000) et tracez comment le compromis de rappel de précision change.

3. **Threshold calibration**: mettre en œuvre l'échelle de plaques (adapter une régression logistique sur les sorties brutes du modèle pour produire des probabilités calibrées). Comparer la courbe de rappel de précision avant et après l'étalonnage.

4. **Ensemble with balanced bagging**: entraîner plusieurs modèles, chacun sur un échantillon de démarrage équilibré (toutes les minorités + sous-ensemble aléatoire de la majorité). Avertir leurs prédictions. Comparer cette approche à un seul modèle avec SMOTE. Mesurer à la fois la performance et la variance entre les courses.

5. **Imbalance ratio experiment**Pour chaque rapport, entraînez avec et sans SMOTE. Le ratio F1 contre le ratio de déséquilibre pour les deux approches.

## Les termes clés

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

## Pour en savoir plus

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813)-- le document original de SMOTE, toujours le travail le plus cité sur l'apprentissage déséquilibré
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907)-- enquête globale couvrant les approches d'échantillonnage, de coûts et d'algorithmes
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)-- bibliothèque Python avec des variantes SMOTE, des stratégies de sous-échantillonnage et une intégration de pipeline
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)-- quand et pourquoi préférer les courbes de relations publiques aux courbes de ROC pour les problèmes de déséquilibre
