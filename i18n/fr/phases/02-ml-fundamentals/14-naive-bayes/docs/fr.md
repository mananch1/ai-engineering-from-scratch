# Bayes naïf

> L'hypothèse naïve est fausse, et elle fonctionne quand même.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Implémenter des Bayes naïfs multinomiels à partir de zéro avec le lissage Laplace pour la classification du texte
- Expliquez pourquoi l'hypothèse naïve d'indépendance est mathématiquement erronée mais produit des classements de classe corrects dans la pratique.
- Comparez les variantes de Bayes naïves multinomial, bernoulli et gaussienne et sélectionnez la bonne pour un type de fonctionnalité donné
- Évaluer les naïfs Bayes contre la régression logistique sur les données rares de haute dimension et expliquer le compromis de biais-variance au travail

## Le problème

Vous devez classer le texte. Les e-mails en spam ou non-spam. Les avis des clients en positifs ou négatifs. Les billets de support en catégories. Vous avez des milliers de fonctionnalités (une par mot) et des données de formation limitées.

La plupart des classifiateurs s'étouffent ici. La régression logistique a besoin de suffisamment d'échantillons pour estimer des milliers de poids de manière fiable. Les arbres de décision se divisent sur un mot à la fois et s'adaptent de manière sauvage. KNN en 10 000 dimensions est sans sens car chaque point est également éloigné de tous les autres points.

Bayes naïf s'occupe de ça. Il fait une hypothèse mathématiquement erronée (que chaque fonctionnalité est indépendante de toutes les autres fonctionnalités données dans la classe), et il surpasse toujours les modèles "smarts" sur la classification des textes, en particulier avec de petits ensembles de formation. Il traîne en une seule fois les données. Il est à l'échelle de millions de caractéristiques. Il produit des estimations de probabilité (bien que souvent mal calibrées en raison de l'hypothèse d'indépendance).

Comprendre pourquoi une supposition erronée conduit à de bonnes prédictions vous apprend quelque chose de fondamental sur l'apprentissage automatique: le meilleur modèle n'est pas le plus correct, c'est celui qui a le meilleur compromis de variance de biais pour vos données.

## Le concept

### Théorème de Bayes (révision rapide)

Le théorème de Bayes renverse les probabilités conditionnelles:

```math
P(\text{class} \mid \text{features}) = \frac{P(\text{features} \mid \text{class}) \cdot P(\text{class})}{P(\text{features})}
```

Nous voulons`P(class | features)`-- la probabilité qu'un document appartient à une classe compte tenu des mots qui y sont.
- `P(features | class)`-- la probabilité de voir ces mots dans des documents de cette classe
- `P(class)`-- la probabilité antérieure de la classe (combien de spam est commun en général?)
- `P(features)`- les preuves, les mêmes pour toutes les classes, afin que nous puissions l'ignorer lors de la comparaison

La classe avec le plus haut .`P(class | features)`Il gagne.

### La supposition naïve de l'indépendance

L' informatique `P(features | class)`Il faut calculer la probabilité commune de toutes les caractéristiques ensemble.$2^{10{,}000}$Des combinaisons possibles.

L'hypothèse naïve: chaque caractéristique est conditionnellement indépendante compte tenu de la classe.

```math
P(w_1, w_2, \ldots, w_n \mid \text{class}) = P(w_1 \mid \text{class}) \cdot P(w_2 \mid \text{class}) \cdots P(w_n \mid \text{class})
```

Au lieu d'une distribution commune impossible, vous estimez $n$Il suffit de compter les chiffres.

Cette hypothèse est évidemment fausse. Les mots "machine" et "apprentissage" ne sont indépendants dans aucun document. Mais le classifiant n'a pas besoin d'estimations de probabilité correctes. Il a besoin de classements corrects - quelle classe a la plus grande probabilité. L'hypothèse d'indépendance introduit des erreurs systématiques, mais ces erreurs affectent toutes les classes de manière similaire, de sorte que le classement reste correct.

### Pourquoi il fonctionne encore

Trois raisons:

1. **Ranking over calibration.**La classification ne nécessite que la classe la plus haut classée pour être correcte.$P(\text{spam}) = 0.99999$Si la probabilité est de 0,7, le classifiateur choisit toujours correctement le spam.

2. **High bias, low variance.**L'hypothèse d'indépendance est un précédent fort. Elle restreint fortement le modèle, ce qui empêche le surpassage. Avec des données de formation limitées, un modèle légèrement erroné mais stable bat un modèle théoriquement correct mais extrêmement instable.

3. **Feature redundancy cancels out.**Les caractéristiques corrélatives fournissent des preuves redondantes. Le classifiant compte ces preuves à double, mais il les compte à double pour la classe correcte aussi. Si "machine" et "apprentissage" apparaissent toujours ensemble, les deux fournissent des preuves pour la classe "tech". NB les compte deux fois, mais il les compte deux fois pour la classe correcte.

Une quatrième raison pratique: Naïf Bayes est extrêmement rapide. La formation est un seul passage à travers les fréquences de comptage des données. La prédiction est une multiplication de matrice. Vous pouvez vous entraîner sur un million de documents en quelques secondes. Cette vitesse signifie que vous pouvez iterer plus rapidement, essayer plus de fonctionnalités et exécuter plus d'expériences que avec des modèles plus lents.

### Les mathématiques étape par étape

Prenons un exemple concret: supposons que nous ayons deux classes: spam et non-spam. Notre vocabulaire a trois mots: "gratuit", "argent", "réunion".

Données de formation:
- Les e-mails de spam mentionnent "gratuit" 80 fois, "argent" 60 fois, "réunion" 10 fois (150 mots au total)
- Les e-mails non-spam mentionnent "gratuit" 5 fois, "argent" 10 fois, "réunion" 100 fois (115 mots au total)
- 40% des e-mails sont des spams, 60% ne sont pas des spams

Avec lissage Laplace ($\alpha=1$):

```math
\begin{aligned}
P(\text{free} \mid \text{spam}) &= (80 + 1) / (150 + 3) = 81/153 = 0.529 \\
P(\text{money} \mid \text{spam}) &= (60 + 1) / (150 + 3) = 61/153 = 0.399 \\
P(\text{meeting} \mid \text{spam}) &= (10 + 1) / (150 + 3) = 11/153 = 0.072 \\[6pt]
P(\text{free} \mid \text{not-spam}) &= (5 + 1) / (115 + 3) = 6/118 = 0.051 \\
P(\text{money} \mid \text{not-spam}) &= (10 + 1) / (115 + 3) = 11/118 = 0.093 \\
P(\text{meeting} \mid \text{not-spam}) &= (100 + 1) / (115 + 3) = 101/118 = 0.856
\end{aligned}
```

Le nouveau courriel contient: "gratuit" (2 fois), "argent" (1 fois), "réunion" (0 fois).

```math
\begin{aligned}
\log P(\text{spam} \mid \text{email}) &= \log(0.4) + 2 \log(0.529) + 1 \log(0.399) + 0 \log(0.072) \\
&= -0.916 + 2(-0.637) + (-0.919) + 0 \\
&= -3.109 \\[6pt]
\log P(\text{not-spam} \mid \text{email}) &= \log(0.6) + 2 \log(0.051) + 1 \log(0.093) + 0 \log(0.856) \\
&= -0.511 + 2(-2.976) + (-2.375) + 0 \\
&= -8.838
\end{aligned}
```

Le mot "gratuit" apparaissant deux fois est une preuve forte du spam.$0 \cdot \log P$En effet, les mots absents n'ont pas d'effet dans le système de calcul de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la

### Trois variantes

Bayes naïf est disponible en trois saveurs.`P(feature | class)`- C'est différent.

#### Bayes naïf à plusieurs noms

Modèles de chaque fonctionnalité comme un compte. Il est préférable pour les données de texte où les caractéristiques sont des fréquences de mots ou des valeurs TF-IDF.

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{count of } \text{word}_i \text{ in class} + \alpha}{\text{total words in class} + \alpha \cdot \text{vocab\_size}}
```

Le `alpha`est le lissage Laplace (expliqué ci-dessous).

#### Bayes naïf gaussien

Modèle de chaque fonctionnalité comme une distribution normale.

```math
P(x_i \mid \text{class}) = \frac{1}{\sqrt{2 \pi \cdot \text{var}}} \cdot \exp\left(-\frac{(x_i - \text{mean})^2}{2 \cdot \text{var}}\right)
```

Chaque classe a sa propre moyenne et sa propre variance par fonctionnalité. Cela fonctionne bien lorsque les fonctionnalités suivent réellement une courbe de cloche au sein de chaque classe.

#### Bernoulli naïf Bayes

Modèles de chaque fonctionnalité comme binaire (présent ou absent).

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{docs in class containing } \text{word}_i + \alpha}{\text{total docs in class} + 2 \alpha}
```

Contrairement à Multinomial, Bernoulli pénalise explicitement l'absence d'un mot. Si "libre" apparaît généralement dans le spam mais est absent de cet e-mail, Bernoulli le compte comme preuve contre le spam.

### Quand utiliser chaque variante

| Variant | Feature Type | Best For | Example |
|---------|-------------|----------|---------|
| Multinomial | Counts or frequencies | Text classification, bag-of-words | Email spam, topic classification |
| Gaussian | Continuous values | Tabular data with normal-ish features | Iris classification, sensor data |
| Bernoulli | Binary (0/1) | Short text, binary feature vectors | SMS spam, presence/absence features |

### Légimentation de la place

Que se passe-t-il lorsqu'un mot apparaît dans les données d'essai mais n'apparaît jamais dans les données de formation pour une classe particulière?

Sans lissage:`P(word | class) = 0/N = 0`Un zéro multiplié par l' ensemble produit fait`P(class | features) = 0`Un seul mot invisible détruit toute la prédiction, peu importe combien d'autres preuves l'appuient.

Le lissage de la laplace ajoute un petit nombre `alpha`(généralement 1) pour chaque nombre de caractéristiques:

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{count}(\text{word}_i, \text{class}) + \alpha}{\text{total\_words\_in\_class} + \alpha \cdot \text{vocab\_size}}
```

Avec $\alpha=1$Le mot "discombobulate" apparaissant dans un e-mail de test ne tue plus la probabilité de spam.

L'alpha supérieur signifie un raffinement plus fort (distributions plus uniformes).

L'effet de l' alpha:

| Alpha | Effect | When to use |
|-------|--------|-------------|
| 0.001 | Almost no smoothing, trust the data | Very large training set, no unseen features expected |
| 0.1 | Light smoothing | Large training set |
| 1.0 | Standard Laplace smoothing | Default starting point |
| 10.0 | Heavy smoothing, flattens distributions | Very small training set, many unseen features expected |

### Compteur log-espace

Multiplication de centaines de probabilités (chacune inférieure à 1) provoque un sous-flux de point flottant. Le produit devient zéro dans le point flottant même si la valeur réelle est un nombre positif très petit.

La solution: travailler dans l'espace log. Au lieu de multiplier les probabilités, ajoutez leurs logarithmes:

```math
\log P(\text{class} \mid x_1, x_2, \ldots, x_n) = \log P(\text{class}) + \sum_i \log P(x_i \mid \text{class})
```

Cela transforme la prédiction en un produit de point:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

La multiplication de matrice. C'est pourquoi la prédiction de Bayes naïve est si rapide -- c'est la même opération qu'un modèle linéaire à couche unique.

### Bayes naïf contre la régression logistique

Les deux sont des classifiants linéaires pour le texte.

| Aspect | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Type | Generative (models $P(X \mid Y){}$) | Discriminative (models $P(Y \mid X){}$) |
| Training | Count frequencies | Optimize loss function |
| Small data | Better (strong prior helps) | Worse (not enough to estimate weights) |
| Large data | Worse (wrong assumption hurts) | Better (flexible boundary) |
| Features | Assumes independence | Handles correlations |
| Speed | Single pass, very fast | Iterative optimization |
| Calibration | Poor probabilities | Better probabilities |

Règle générale: commencez par Naive Bayes. Si vous avez assez de données et des plateaux NB, passez à la régression logistique.

### L'équipement de transport

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

En pratique, nous travaillons dans l'espace log pour éviter le sous-flux des points flottants. Au lieu de multiplier de nombreuses petites probabilités, nous ajoutons leurs logarithmes:

```math
\log P(\text{class} \mid \text{features}) = \log P(\text{class}) + \sum_i \log P(\text{feature}_i \mid \text{class})
```

```figure
naive-bayes
```

## Faites-le

Le code dans `code/naive_bayes.py`Il implémentera à la fois MultinomialNB et GaussianNB à partir de zéro.

### Nombre de noms

La mise en œuvre à partir de zéro:

1. **fit(X, y)**Pour chaque classe, comptez la fréquence de chaque fonctionnalité. Ajoutez le lissage Laplace. Comptez les probabilités de journaux. Réservez les prérogatives de classe (journaux de fréquences de classe).

2. **predict_log_proba(X)**: Pour chaque échantillon, calculer $\log P(\text{class})$+ somme de $\log P(\text{feature}_i \mid \text{class})$Pour toutes les classes. Ceci est une matrice de multiplication: X @ log_probs.T + log_priors.

3. **predict(X)**: Retournez la classe avec la plus grande probabilité de log.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

La prédiction est juste une multiplication de matrice plus un biais.

### Gaussie NB

Pour les caractéristiques continues, on estime la moyenne et la variance par classe par caractéristique:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

La prédiction utilise le PDF gaussien par fonctionnalité, multiplié par fonctionnalité (ajouté dans l'espace de journaux).

### Démo: Classification du texte

Le code génère des données de sacs de mots synthétiques simulant deux classes (articles technologiques contre articles sportifs). Chaque classe a une distribution de fréquence de mots différente.

Les données synthétiques fonctionnent comme ceci: nous créons 200 "words" (columnes de fonctionnalités). Les mots 0-39 ont une fréquence élevée dans les articles techniques et faible dans le sport. Les mots 80-119 ont une fréquence élevée dans le sport et faible dans le domaine technique. Les mots 40-79 sont de fréquence moyenne dans les deux. Cela crée un scénario réaliste où certains mots sont des indicateurs de classe forts et d'autres sont du bruit.

### Démo: fonctionnalités continues

Le code génère des données Iris-like (3 classes, 4 caractéristiques, clusters gaussiens). GaussianNB classifie en utilisant la moyenne et la variance par classe. Chaque classe a un centre différent (vecteur moyen) et une diffusion différente (variance), imitant les données du monde réel où les mesures diffèrent systématiquement entre les catégories.

Le code démontre également:
- **Smoothing comparison:**Formation de MultinomialNB avec différentes valeurs alpha pour montrer l'effet de la force de lissage sur la précision.
- **Training size experiment:**Comment la précision de NB s'améliore à mesure que les données de formation passent de 20 à 1600 échantillons. NB atteint une précision décente même avec très peu d'échantillons - c'est son principal avantage.
- **Confusion matrix:**La précision par classe, le rappel et le score de F1 pour montrer où NB fait des erreurs.

### Vite de prédiction

La prédiction de Bayes naïve est une multiplication de matrice.$n$échantillons avec $d$caractéristiques et $k$classes:
- MultinomialNB: une matrice est multipliée $(n \times d) \mathbin{@} (d \times k) = O(n \cdot d \cdot k)$
- GaussianNB: $n \cdot k$Évaluations Gaussian PDF, chacune à la fin $d$caractéristiques = $O(n \cdot d \cdot k)$

Les deux sont linéaires dans toutes les dimensions. Comparer ceci à KNN (qui nécessite le calcul de la distance à tous les points de formation) ou SVM avec le noyau RBF (qui nécessite l'évaluation du noyau contre tous les vecteurs de support). NB est plus rapide par ordre de magnitude au moment de la prédiction.

## Utilisez-le

Avec sklearn, les deux variantes sont unlinées:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Pour la classification du texte avec sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

Le code dans `naive_bayes.py`Comparer les mises en œuvre à partir de zéro avec les données de sklearn sur les mêmes données pour vérifier leur exactitude.

### TF-IDF avec Naive Bayes

Les nombres de mots bruts donnent à chaque mot le même poids par occurrence. Mais les mots communs comme "le" et "est" apparaissent fréquemment dans chaque classe - ils ne contiennent aucune information. TF-IDF (Term Frequency - Inverse Document Frequency) diminue le poids des mots communs et augmente le poids des mots rares et discriminatoires.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

Les valeurs TF-IDF sont non négatives, elles fonctionnent donc avec MultinomialNB. La combinaison de TF-IDF + MultinomialNB est l'une des lignes de base les plus fortes pour la classification du texte.

### BernoulliNB pour texte court

Pour le texte court (tweets, SMS, messages de chat), BernoulliNB peut surpasser MultinomialNB. Les textes courts ont un faible nombre de mots, de sorte que les informations de fréquence sur lesquelles MultinomialNB s'appuie sont bruyantes.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

Le `binary=True`flag dans CountVectorizer convertit tous les nombres en 0/1. Sans lui, BernoulliNB fonctionne toujours mais voit des nombres pour lesquels il n'a pas été conçu.

### Calibration NB Probabilités

Les probabilités de NB sont mal calibrées.$P(\text{spam}) = 0.95$Si vous avez besoin d'estimations de probabilité fiables (par exemple, pour définir un seuil ou pour le combiner avec d'autres modèles), utilisez le CalibratedClassifierCV de sklearn:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

Cela correspond à une régression logistique en plus des scores bruts de NB en utilisant la validation croisée.

### Les gotches communes

1. **Negative feature values.**MultinomialNB nécessite des caractéristiques non négatives. Si vous avez des valeurs négatives (comme TF-IDF avec certains paramètres ou caractéristiques standardisées), utilisez GaussianNB à la place, ou changez les caractéristiques pour être positives.

2. **Zero variance features.**Le GaussianNB est divisé par variance. Si une fonction a une variance zéro pour une classe (toutes les valeurs sont identiques), le calcul de probabilité est cassé. Le code ajoute un petit terme de lissage (1e-9) à toutes les variances pour éviter cela.

3. **Class imbalance.**Si 99% des e-mails ne sont pas du spam, le précédent $P(\text{not-spam}) = 0.99$Vous pouvez définir manuellement les classes prioritaires ou utiliser le paramètre class_prior dans sklearn.

4. **Feature scaling.**Le multinomialNB n'a pas besoin d'échelle (il fonctionne sur les calculs). Le GaussianNB n'a pas besoin d'échelle non plus (il estime les statistiques par caractéristique).

## La faire partir

Cette leçon donne:
- `outputs/skill-naive-bayes-chooser.md`-- une compétence de décision pour choisir la bonne variante NB
- `code/naive_bayes.py`-- MultinomialNB et GaussianNB à partir de zéro, avec une comparaison sklearn

### Quand Bayes échoue

NB échoue lorsque l'hypothèse d'indépendance provoque des classements incorrects (et non seulement des probabilités incorrectes).

1. **Strong feature interactions.**Si la classe dépend de la combinaison de deux caractéristiques mais pas l'une ou l'autre seule (patterns XOR-like), NB la manquera complètement.

2. **Highly correlated features with opposing evidence.**Si la fonction A dit "spam" et la fonction B dit "non-spam", mais A et B sont parfaitement corrélés (ils sont toujours d'accord dans la réalité), NB verra des preuves contradictoires là où il n'y en a pas.

3. **Very large training sets.**Avec suffisamment de données, des modèles discriminatifs tels que la régression logistique apprennent la véritable limite de décision et dépassent NB. L'hypothèse d'indépendance qui a aidé avec de petites données retient maintenant le modèle.

En pratique, ces modes d'échec sont rares pour la classification du texte. Les caractéristiques du texte sont nombreuses, individuellement faibles, et les erreurs de l'hypothèse d'indépendance ont tendance à annuler. Pour les données de tableau avec peu de caractéristiques fortement corrélatives, considérez d'abord la régression logistique ou les modèles basés sur des arbres.

## Exercices

1. **Smoothing experiment.**Prenez MultinomialNB sur les données de texte avec des valeurs alpha de 0,01, 0,1, 1,0, 10,0 et 100,0.

2. **Feature independence test.**Prenez un ensemble de données de texte réel. Choisissez deux mots qui sont évidemment corrélés ("machine" et "apprentissage"). Compute $P(\text{word}_1 \mid \text{class}) \cdot P(\text{word}_2 \mid \text{class})$et comparer à $P(\text{word}_1 \text{ AND } \text{word}_2 \mid \text{class})$La supposition d'indépendance est-elle fausse ?

3. **Bernoulli implementation.**Extension du code avec une classe BernoulliNB. Convertir les sacs de mots en binaire (présent/absent) et comparer la précision contre MultinomialNB sur les données de texte. Quand Bernoulli gagne-t-il?

4. **NB vs Logistic Regression.**Prenez les deux en fonction des données de texte. Commencez par 100 échantillons de formation et augmentez à 10 000.

5. **Spam filter.**Construire un classifiateur de spam complet: jetonner le texte brut du courrier électronique, créer un vocabulaire, créer des fonctionnalités de sacs de mots, former MultinomialNB, évaluer avec précision et rappeler (pas seulement l'exactitude - pourquoi?).

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Naive Bayes | "Simple probabilistic classifier" | A classifier that applies Bayes' theorem with the assumption that features are conditionally independent given the class |
| Conditional independence | "Features don't affect each other" | $P(A, B \mid C) = P(A \mid C) \cdot P(B \mid C)$ -- knowing $B$ tells you nothing new about $A$ once you know $C$ |
| Laplace smoothing | "Add-one smoothing" | Adding a small count to every feature to prevent zero probabilities from dominating the prediction |
| Prior | "What you believed before seeing data" | $P(\text{class})$ -- the probability of each class before observing any features |
| Likelihood | "How well the data fits" | $P(\text{features} \mid \text{class})$ -- the probability of observing these features if the class is known |
| Posterior | "What you believe after seeing data" | $P(\text{class} \mid \text{features})$ -- the updated probability of the class after observing the features |
| Generative model | "Models how data is generated" | A model that learns $P(X \mid Y)$ and $P(Y)$, then uses Bayes' theorem to get $P(Y \mid X)$ |
| Discriminative model | "Models the decision boundary" | A model that directly learns $P(Y \mid X)$ without modeling how $X$ is generated |
| Log probability | "Avoid underflow" | Working with $\log P$ instead of $P$ to prevent the product of many small numbers from becoming zero in floating point |

## Pour en savoir plus

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)-- les trois variantes avec des détails mathématiques
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)-- la comparaison classique de Multinomial vs Bernoulli pour le texte
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)-- améliorations de la note pour le texte
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)-- prouve que la NB converge plus rapidement que la LR avec moins de données
