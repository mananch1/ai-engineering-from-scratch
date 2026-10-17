# Métodes de mise en œuvre

> Un groupe d'apprenants faibles, combinés correctement, deviennent des apprenants forts.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 2, Lesson 10 (Bias-Variance Tradeoff)
**Time:** ~120 minutes

## Objectifs d'apprentissage

- Implémenter AdaBoost et gradient boosting à partir de zéro et expliquer comment le boosting réduit séquentiellement les biais
- Construire un ensemble de sacs et démontrer comment la moyenne des modèles décorrélés réduit la variance sans augmenter les biais
- Comparer le sachetage, le renforcement et l'empilage en termes de composants d'erreur que chaque méthode vise
- Évaluer la diversité de l'ensemble et expliquer pourquoi la précision des votes majoritaires s'améliore avec des apprenants plus faibles et plus indépendants

## Le problème

Un seul arbre de décision est rapide à former et facile à interpréter, mais il dépasse. Un seul modèle linéaire est insuffisant sur des limites complexes. Vous pouvez passer des jours à concevoir l'architecture modèle parfaite. Ou vous pouvez combiner un tas de modèles imparfaits et obtenir quelque chose de mieux que chacun d'eux individuellement.

Les méthodes d'assemblage font exactement cela. Ce sont les techniques les plus fiables pour gagner des concours Kaggle sur des données tabulaires, elles alimentent la plupart des systèmes ML de production et elles illustrent le compromis de variance de biais en action.

## Le concept

### Pourquoi les groupes travaillent

Supposons que vous ayez$N$des classifiateurs indépendants, chacun avec précision $p > 0.5$- La majorité des voix est exacte:

```math
P(\text{majority correct}) = \sum_{k > N/2} \binom{N}{k} p^k (1-p)^{N-k}
```

Pour 21 classifiateurs, chacun avec une précision de 60%, la précision des voix majoritaires est d'environ 74%. Avec 101 classifiateurs, elle monte à 84%.

L' exigence clé est **diversity**Si tous les modèles commettent les mêmes erreurs, leur combinaison ne sert à rien.

- Différents sous-ensembles de formation (de retard)
- Les sous-ensembles de caractéristiques différentes (forêts aléatoires)
- Correction d'erreur séquentielle (augmentation)
- Familles de modèles différentes (en pile)

### Les produits de la fabrication de boîtes de chargement

Le sacage crée une diversité en formant chaque modèle sur un échantillon de démarrage différent des données de formation.

```mermaid
flowchart TD
    D[Training Data] --> B1[Bootstrap Sample 1]
    D --> B2[Bootstrap Sample 2]
    D --> B3[Bootstrap Sample 3]
    D --> BN[Bootstrap Sample N]

    B1 --> M1[Model 1]
    B2 --> M2[Model 2]
    B3 --> M3[Model 3]
    BN --> MN[Model N]

    M1 --> V[Average or Majority Vote]
    M2 --> V
    M3 --> V
    MN --> V

    V --> P[Final Prediction]
```

Un échantillon de démarrage est tiré avec le remplacement des données originales, de la même taille que l'original. Environ 63,2% des échantillons uniques apparaissent dans chaque démarrage. Les échantillons restants de 36,8% (échantillons hors sac) fournissent un ensemble de validation gratuit.

Le sacage réduit la variance sans augmenter grandement le biais. Chaque arbre individuel surpasse son échantillon de raccourci, mais le surpassage est différent pour chaque arbre, de sorte que la moyenne annule le bruit.

**Random Forests**Les arbres sont en train de se décomposer avec une torsion supplémentaire: à chaque fraction, seuls un sous-ensemble aléatoire de caractéristiques est pris en compte.`sqrt(n_features)`pour la classification et `n_features / 3`pour la régression.

### Résistance (correction d'erreur séquentielle)

Chaque nouveau modèle se concentre sur les exemples que les modèles précédents ont mal interprétés.

```mermaid
flowchart LR
    D[Data with weights] --> M1[Model 1]
    M1 --> E1[Find errors]
    E1 --> W1[Increase weights on errors]
    W1 --> M2[Model 2]
    M2 --> E2[Find errors]
    E2 --> W2[Increase weights on errors]
    W2 --> M3[Model 3]
    M3 --> F[Weighted sum of all models]
```

Le boosting réduit les biais. Chaque nouveau modèle corrige les erreurs systématiques de l'ensemble jusqu'à présent. La prédiction finale est une somme pondérée de tous les modèles, où les meilleurs modèles obtiennent des poids plus élevés.

Le compromis: le booster peut surpasser si vous lancez trop de tirs, car il continue à adapter des exemples plus difficiles, dont certains peuvent être bruyants.

### AdaBoost

AdaBoost (Boosting adaptatif) était le premier algorithme de boosting pratique. Il fonctionne avec n'importe quel apprenant de base, généralement des troncs de décision (arbres de profondeur-1).

L' algorithme:

```
1. Initialize sample weights: w_i = 1/N for all i

2. For t = 1 to T:
   a. Train weak learner h_t on weighted data
   b. Compute weighted error:
      err_t = sum(w_i * I(h_t(x_i) != y_i)) / sum(w_i)
   c. Compute model weight:
      alpha_t = 0.5 * ln((1 - err_t) / err_t)
   d. Update sample weights:
      w_i = w_i * exp(-alpha_t * y_i * h_t(x_i))
   e. Normalize weights to sum to 1

3. Final prediction: H(x) = sign(sum(alpha_t * h_t(x)))
```

Les modèles avec moins d'erreur gagnent plus d'alpha.

### Un accroissement progressif

Le boosting gradient généralise le boosting à des fonctions de perte arbitraire. Au lieu de ré pondérer les échantillons, il correspond à chaque nouveau modèle aux résidus (gradient négatif de la perte) de l'ensemble actuel.

```
1. Initialize: F_0(x) = argmin_c sum(L(y_i, c))

2. For t = 1 to T:
   a. Compute pseudo-residuals:
      r_i = -dL(y_i, F_{t-1}(x_i)) / dF_{t-1}(x_i)
   b. Fit a tree h_t to the residuals r_i
   c. Find optimal step size:
      gamma_t = argmin_gamma sum(L(y_i, F_{t-1}(x_i) + gamma * h_t(x_i)))
   d. Update:
      F_t(x) = F_{t-1}(x) + learning_rate * gamma_t * h_t(x)

3. Final prediction: F_T(x)
```

Pour la perte d'erreur carrée, les pseudo-résidus sont juste les résidus réels: $r_i = y_i - F_{t-1}(x_i)$Chaque arbre correspond littéralement aux erreurs de l'ensemble précédent.

Le taux d'apprentissage (rétrécissement) contrôle la contribution de chaque arbre.

### XGBoost: Pourquoi il domine les données tabulaires

XGBoost (eXtreme Gradient Boosting) est un boosting de gradient avec des optimisations d'ingénierie qui le rendent rapide, précis et résistant au surmatch:

- **Regularized objective:**Les sanctions L1 et L2 sur les poids des feuilles empêchent les arbres individuels d'être trop confiants
- **Second-order approximation:**Utilise à la fois la première et la deuxième dérivées de la perte, donnant de meilleures décisions partagées
- **Sparsity-aware splits:**Traite les valeurs manquantes de manière native en apprenant la meilleure direction pour les données manquantes à chaque fraction
- **Column subsampling:**Comme les forêts aléatoires, les échantillons sont caractéristiques à chaque fraction pour la diversité
- **Weighted quantile sketch:**Trouve efficacement des points de fractionnement pour les caractéristiques continues sur les données distribuées
- **Cache-aware block structure:**L'état de mémoire optimisé pour les lignes de cache du processeur

Pour les données de table, XGBoost (et son successeur LightGBM) dépasse systématiquement les réseaux neuraux. Cela ne changera pas prochainement. Si vos données s'inscrivent dans un tableau avec des lignes et des colonnes, commencez par augmenter le gradient.

### L'accumulation (méta-apprentissage)

L'empilage utilise les prédictions de plusieurs modèles de base comme caractéristiques pour un méta-apprenant.

```mermaid
flowchart TD
    D[Training Data] --> M1[Model 1: Random Forest]
    D --> M2[Model 2: SVM]
    D --> M3[Model 3: Logistic Regression]

    M1 --> P1[Predictions 1]
    M2 --> P2[Predictions 2]
    M3 --> P3[Predictions 3]

    P1 --> META[Meta-Learner]
    P2 --> META
    P3 --> META

    META --> F[Final Prediction]
```

Le méta-apprenant apprend quel modèle de base à faire confiance pour quelles entrées. Si la forêt aléatoire est meilleure dans certaines régions et le SVM dans d'autres, le méta-apprenant apprendra à router en conséquence.

Pour éviter la fuite de données, les prédictions de modèle de base doivent être générées par validation croisée sur l'ensemble de formation.

### Le vote

L'ensemble le plus simple, combiner directement les prédictions.

- **Hard voting:**La majorité vote sur les étiquettes de classe.
- **Soft voting:**Prévues moyennes, choisissez la classe avec la probabilité moyenne la plus élevée, généralement mieux parce qu'elle utilise des informations de confiance.

```figure
f3-ensemble-average
```

## Faites-le

### Étape 1: Décision de base (apprenant de base)

Le code dans `code/ensembles.py`Nous commençons par un tronc de décision: un arbre avec une seule fraction.

```python
class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float("inf")

        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            for thresh in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    pred[polarity * X[:, f] < polarity * thresh] = -1
                    error = np.sum(weights[pred != y])
                    if error < best_error:
                        best_error = error
                        self.feature_idx = f
                        self.threshold = thresh
                        self.polarity = polarity

    def predict(self, X):
        n = X.shape[0]
        pred = np.ones(n)
        idx = self.polarity * X[:, self.feature_idx] < self.polarity * self.threshold
        pred[idx] = -1
        return pred
```

### Étape 2: AdaBoost à partir de zéro

```python
class AdaBoostScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        weights = np.full(n, 1 / n)

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            pred = stump.predict(X)

            err = np.sum(weights[pred != y])
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)
            weights *= np.exp(-alpha * y * pred)
            weights /= weights.sum()

            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        total = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        return np.sign(total)
```

### Étape 3: Renforcement graduel à partir de zéro

```python
class GradientBoostingScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        current_pred = np.full(len(y), self.initial_pred)

        for _ in range(self.n_estimators):
            residuals = y - current_pred
            tree = SimpleRegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            current_pred += self.lr * update
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred
```

### Étape 4: Comparer avec sklearn

Le code vérifie que nos mises en œuvre dès le départ produisent une précision similaire à celle de sklearn `AdaBoostClassifier`et `GradientBoostingClassifier`, et compare toutes les méthodes côte à côte.

## Utilisez-le

### Quand utiliser chaque méthode

| Method | Reduces | Best for | Watch out for |
|--------|---------|----------|---------------|
| Bagging / Random Forest | Variance | Noisy data, many features | Does not help with bias |
| AdaBoost | Bias | Clean data, simple base learners | Sensitive to outliers and noise |
| Gradient Boosting | Bias | Tabular data, competitions | Slow to train, easy to overfit without tuning |
| XGBoost / LightGBM | Both | Production tabular ML | Many hyperparameters |
| Stacking | Both | Getting last 1-2% accuracy | Complex, risk of overfitting meta-learner |
| Voting | Variance | Quick combination of diverse models | Only helps if models are diverse |

### La pile de production des données tabulaires

Pour la plupart des problèmes de prédiction tabulaire, voici l'ordre à essayer:

1. **LightGBM or XGBoost**avec des paramètres par défaut
2. Tune n_estimators, taux d'apprentissage, profondeur maximale, poids min_child
3. Si vous avez besoin de la dernière 0,5%, construisez un ensemble d'empilage avec 3-5 modèles différents
4. Utiliser la validation croisée tout au long de la période

Les réseaux neuraux sur les données tabulaires sont presque toujours pires que l'augmentation du gradient, malgré les tentatives de recherche continues. TabNet, NODE et architectures similaires correspondent parfois mais battent rarement un XGBoost bien ajusté.

## La faire partir

Cette leçon produit `outputs/prompt-ensemble-selector.md`- une requête qui vous aide à choisir la bonne méthode ensemble pour un ensemble de données donné. Décrivez vos données (taille, types de fonctionnalités, niveau de bruit, équilibre de classe) et le problème que vous résolvez. La requête passe par une liste de contrôle de décision, recommande une méthode, suggère de démarrer des hyperparametres, et met en garde contre les erreurs courantes pour cette méthode.`outputs/skill-ensemble-builder.md`avec le guide complet de sélection.

## Exercices

1. Modifier la mise en œuvre AdaBoost pour suivre la précision de l'entraînement après chaque tour.

2. Mettre en œuvre une forêt aléatoire à partir de zéro en ajoutant une fonction de sous-échantillonnage aléatoire à l'arbre de régression.`max_features=sqrt(n_features)`Comparer la réduction de la variance à un seul arbre.

3. Dans la mise en œuvre de la mise en œuvre de la mise en place de la mise en place du gradient, ajoutez l'arrêt précoce: suivre la perte de validation après chaque tour et arrêter lorsqu'elle n'a pas amélioré pendant 10 tours consécutifs.

4. Construisez un ensemble d'empilage avec trois modèles de base (régrésion logistique, arbre de décision, voisines k les plus proches) et un méta-apprenant de régression logistique. Utilisez une validation croisée de 5 fois pour générer des méta-features.

5. Exécutez XGBoost sur le même ensemble de données avec des paramètres par défaut. Comparer sa précision à votre augmentation du gradient à partir de zéro. Temps les deux. Quelle est la différence de vitesse?

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Bagging | "Train on random subsets" | Bootstrap aggregating: train models on bootstrap samples, average predictions to reduce variance |
| Boosting | "Focus on hard examples" | Train models sequentially, each correcting errors of the ensemble so far, to reduce bias |
| AdaBoost | "Reweight the data" | Boosting via sample weight updates; misclassified points get higher weight for the next learner |
| Gradient boosting | "Fit the residuals" | Boosting via fitting each new model to the negative gradient of the loss function |
| XGBoost | "The Kaggle weapon" | Gradient boosting with regularization, second-order optimization, and systems-level speed tricks |
| Stacking | "Models on top of models" | Use predictions of base models as input features for a meta-learner |
| Random forest | "Many randomized trees" | Bagging with decision trees, adding random feature subsampling at each split for diversity |
| Ensemble diversity | "Make different mistakes" | Models must be uncorrelated in their errors for the ensemble to improve over individuals |
| Out-of-bag error | "Free validation" | Samples not in a bootstrap draw (~36.8%) serve as a validation set without needing a holdout |

## Pour en savoir plus

- [Schapire & Freund: Boosting: Foundations and Algorithms](https://mitpress.mit.edu/9780262526036/)-- le livre des créateurs d'AdaBoost
- [Friedman: Greedy Function Approximation: A Gradient Boosting Machine (2001)](https://doi.org/10.1214/aos/1013203451)- le papier d'amélioration du gradient original
- [Chen & Guestrin: XGBoost (2016)](https://arxiv.org/abs/1603.02754)-- le papier XGBoost
- [Wolpert: Stacked Generalization (1992)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)- le papier d'empilage original
- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)-- référence pratique
