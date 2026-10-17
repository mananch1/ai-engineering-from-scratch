# Embedding Word  Word2Vec depuis le début

> Un mot est la compagnie qu'il garde.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## Le problème

Le TF-IDF sait `dog`et `puppy`Il ne sait pas qu'ils signifient presque la même chose.`dog`ne peut pas généraliser à une revue sur `puppy`Vous pouvez le faire en lisant des synonymes, mais cela échoue dans les termes rares, le jargon de domaine et toutes les langues que vous ne vous êtes pas attendues.

Vous voulez une représentation où`dog`et `puppy`- Ils sont proches de l'espace.`king - man + woman`Terres proches`queen`- Un modèle qui a été formé à la`dog`Transfère un signal à `puppy`- Je le fais gratuitement.

Word2Vec nous a donné cet espace. Un réseau neural à deux couches, des milliards de tokens de formation, publié en 2013. L'architecture est presque embarrassantement simple. Les résultats ont remodelé la PNL pendant une décennie.

## Le concept

**Distributional hypothesis**(Première, 1957): " Vous reconnaîtrez un mot par la compagnie qu'il garde. " Si deux mots apparaissent dans des contextes similaires, ils signifient probablement des choses similaires.

Word2Vec est disponible en deux versions, les deux exploitant cette idée.

- **Skip-gram.**Donnez un mot central, prédisez les mots environnants. `cat -> (the, sat, on)`avec la taille de la fenêtre 2.
- **CBOW (continuous bag of words).**Compte tenu des mots environnants, prédire le centre.`(the, sat, on) -> cat`- Je suis désolé .

Le ski-gramme est plus lent à apprendre, mais il gère mieux les mots rares.

Le réseau a une couche cachée sans non-linéarité. L'entrée est un vecteur un-chaud sur le vocabulaire. La sortie est un softmax sur le vocabulaire. Après l'entraînement, vous jetez la couche de sortie. Les poids de couche cachée sont les emblèmes.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Le truc: le softmax de plus de 100 000 mots est trop cher.**negative sampling**Prévoir si ce mot contextuel apparaît près de ce mot central, oui ou non. Prenez une poignée de mots négatifs (non co-occurrents) par paire de formation au lieu de calculer le softmax sur l'ensemble du vocabulaire.

```figure
word-vector-arithmetic
```

## Faites-le

### Étape 1: formation des paires à partir d'un corpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Chaque paire (centre, contexte) dans une fenêtre est un exemple de formation positif.

### Étape 2: intégration des tables

Deux matrices.`W`est la table d'intégration du mot central (la que vous conservez). `W'`est la table des mots de contexte (souvent rejetée, parfois moyennée avec `W`)

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

La taille du vocabulaire 10k et dim 100 est réaliste; pour l'enseignement, 50 vocabulaires x 16 dim suffisent pour voir la géométrie.

### Étape 3: objectif négatif de l'échantillonnage

Pour chaque paire positive `(center, context)`, échantillon `k`Les mots aléatoires du vocabulaire comme négatifs.`W[center] · W'[context]`est élevé pour les positifs et faible pour les négatifs.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

La formule magique: perte logistique sur la paire positive (je veux sigmoïde près de 1) plus perte logistique sur les paires négatives (je veux sigmoïde près de 0). Les gradients coulent vers les deux tables.

### Étape 4: entraînement sur un corps de jouet

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

Après suffisamment d'époques sur un grand corpus, les mots qui partagent des contextes ont des embrasements centraux similaires. Sur un corpus de jouets, vous voyez l'effet faiblement. Sur des milliards de jetons, vous le voyez de manière spectaculaire.

### Étape 5: le truc d' analogie

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

Sur les vecteurs pré-entraînés de 300d Google News:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`Pas parce que le modèle sait ce qu'est la royauté.`(king - man)`capture quelque chose comme "royal", et l'ajouter à `woman`des terres près de la région royale féminine.

## Utilisez-le

L'écriture de Word2Vec à partir de zéro est un enseignement.`gensim`- Je suis désolé .

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Pour le vrai travail, vous ne faites presque jamais de Word2Vec vous-même.

- **GloVe** L'approche de facteurisation de la matrice de co-occurrence de Stanford. 50d, 100d, 200d, 300d points de contrôle. Bonne couverture générale.
- **fastText** L'extension Word2Vec de Facebook qui intègre des caractères n-grammes.
- **Pretrained Word2Vec on Google News** 300d, vocabulaire de 3M mots, publié en 2013. Toujours téléchargé quotidiennement.

### Quand Word2Vec gagnera encore en 2026

- Trainer sur des abstracts médicaux en une heure sur un ordinateur portable, obtenir des vecteurs spécialisés sans captures de modèle général.
- L'ingénierie des caractéristiques de style analogue. `gender_vector = mean(man - woman pairs)`- Soustraire de l'autre mot pour obtenir un axe neutre sur le genre.
- Interprétabilité. 100d est assez petit pour tracer via PCA ou t-SNE et voir en fait des amas de forme.
- N'importe où, les inférences doivent être exécutées sur un appareil sans GPU.

### Lorsque Word2Vec échoue

Le mur polysémique.`bank`a un vecteur. `river bank`et `financial bank`Partagez-le.`table`Un classifiateur en aval ne peut pas distinguer les sens du vecteur.

Les intégrations contextuelles (ELMo, BERT, chaque transformateur depuis) ont résolu cette question en produisant un vecteur différent pour chaque occurrence du mot en fonction du contexte environnant. C'est le saut de Word2Vec à BERT: de statique à contextuel.

Le problème de l'échec du vocabulaire est l'autre.`Zoomer-approved`Si vous avez des informations sur les résultats de formation, vous pouvez les trouver dans les données de formation.

## La faire partir

- Je ne sais pas .`outputs/skill-embedding-probe.md`- Le numéro de la liste:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Exercices

1. **Easy.**Exécutez la boucle d'entraînement sur un petit corpus (20 phrases sur les chats et les chiens).`nearest(vocab, W, W[vocab["cat"]])`Retour `dog`Dans le cas contraire, augmentez les époques ou le vocabulaire.
2. **Medium.**Ajouter un sous-échantillonnage de mots fréquents.`10^-5`Les résultats de l'étude ont été analysés dans des groupes de formation avec une probabilité proportionnelle à leur fréquence.
3. **Hard.**Exercer un modèle sur le corpus des 20 groupes de nouvelles.`he - she`et `doctor - nurse`- les mots de l'occupation de projet sur les deux axes. - Rapporte les occupations ayant le plus grand écart de biais.

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | A dense, low-dim (typically 100-300) representation learned from context. |
| Skip-gram | Word2Vec trick | Predict context words from center word. Slower than CBOW, better for rare words. |
| Negative sampling | Training shortcut | Replace softmax over full vocab with binary classification against `k` random words. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-sensitive vector | Different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | Word not seen in training. Word2Vec cannot produce a vector for these. |

## Pour en savoir plus

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) le papier d'échantillonnage négatif.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) la dérivation la plus claire des gradients, si les mathématiques du papier original semblent denses.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) des réglages de formation de production qui fonctionnent réellement.
