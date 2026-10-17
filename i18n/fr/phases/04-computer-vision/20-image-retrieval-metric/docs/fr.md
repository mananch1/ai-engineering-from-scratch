# Récupération d'image et apprentissage des mesures

> Un système de récupération classe les candidats par distance dans l'espace intégré.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 18 (CLIP)
**Time:** ~45 minutes

## Objectifs d'apprentissage

- Expliquer les pertes d'apprentissage métrique triplé, contrastée et par proxy et choisir la bonne pour un ensemble de données donné
- Mettre en œuvre correctement la normalisation L2 et la similitude cosine et vérifier la différence entre la récupération du "même article" et de la "même classe"
- Construire un index FAISS, le consulter par texte et par image, et signaler recall@K pour un ensemble de requêtes retenues
- Utilisez DINOv2, CLIP et SigLIP comme récipients de rectum et sachez quand chaque gagnant gagne

## Le problème

La récupération est partout dans la vision de production: détection de duplicates, recherche d'image inverse, recherche visuelle ("trouver des produits similaires"), réidentification de visage, réidentification de personne pour la surveillance, correspondance au niveau d'instance pour le commerce électronique. La question du produit est toujours la même: " étant donné cette image de requête, classez mon catalogue".

Deux décisions de conception façonnent l'ensemble du système. L'intégration  quel modèle produit les vecteurs. L'index  comment trouver les voisins les plus proches à l'échelle. Les deux sont des marchandises en 2026 (DINOv2 pour l'intégration, FAISS pour l'index), ce qui relève la barre: la partie difficile est de définir *ce qui compte comme similaire* pour votre application, puis de façonner l'espace d'intégration de sorte que les distances correspondent.

Ce modélisation est un apprentissage métrique, une discipline petite mais à forte élancement.

## Le concept

### Récupération à un coup d'œil

```mermaid
flowchart LR
    Q["Query image<br/>or text"] --> ENC["Encoder"]
    ENC --> EMB["Query embedding"]
    EMB --> IDX["FAISS index"]
    CAT["Catalogue images"] --> ENC2["Encoder (same)"] --> IDX_BUILD["Build index"]
    IDX_BUILD --> IDX
    IDX --> RANK["Top-k nearest<br/>by cosine / L2"]
    RANK --> OUT["Ranked results"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style IDX fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

### Les quatre familles de perdants

| Loss | Requires | Pros | Cons |
|------|----------|------|------|
| **Contrastive** | (anchor, positive) + negatives | Simple, works with any pair label | Slow to converge without many negatives |
| **Triplet** | (anchor, positive, negative) | Intuitive; direct margin control | Hard-triplet mining is expensive |
| **NT-Xent / InfoNCE** | Pairs + batch-mined negatives | Scales to large batches | Needs big batch or momentum queue |
| **Proxy-based (ProxyNCA)** | Class labels only | Fast, stable, no mining | Can overfit to proxies on small datasets |

Pour la plupart des cas d'utilisation de production, commencez par un dos préentraîné et ajoutez une mise à jour de l'apprentissage des mesures si les intégrations off-the-shelf ne fonctionnent pas bien sur votre jeu de test.

### Perte de triplets officiellement

```math
L = \max\left(0, \lVert f(a) - f(p) \rVert^2 - \lVert f(a) - f(n) \rVert^2 + \text{margin}\right)
```

Tirez l' ancre`a`proche de positif `p`, repoussez-le loin du négatif `n`, avec une`margin`La structure en trois images s'allonge à tout ordre de similitude.

Les questions minières: triplets faciles (`n`déjà loin de `a`Les trois types de mines sont les plus utilisés dans les industries de l'exploitation minière.`n`plus de `p`Mais dans la marge) est la recette de FaceNet 2016 et domine toujours.

### Similation de cosine par rapport à L2

Deux mesures, deux conventions:

- **Cosine**: angle entre vecteurs. nécessite des embeddings normalisés L2.
- **L2**: distance euclidienne. Fonctionne sur des embeddings bruts ou normalisés, mais est généralement associée à L2 normalisé + L2 au carré.

Pour la plupart des réseaux modernes, les deux sont équivalents: `||a - b||^2 = 2 - 2 cos(a, b)`quand ?`||a|| = ||b|| = 1`Choisissez la convention qui correspond à votre formation d'embedding; les mélanger en silence change ce que signifie "plus proche".

### Rappel@K

La métrique de récupération standard:

```
recall@K = fraction of queries where at least one correct match is in the top K results
```

Rapporte recall@1, @5, @10 côte à côte. Un recall@10 au-dessus de 0,95 avec recall@1 au-dessous de 0,5 signifie que l'espace d'intégration a la bonne structure mais que le classement est bruyant  essayez des mélodies fines plus longues ou une étape de ré-ranking.

Pour la détection de duplicates, la précision@K est plus importante car chaque faux positif est une erreur visible par l'utilisateur.

### FAISS dans un seul paragraphe

La recherche de similitude d'IA sur Facebook. La bibliothèque de facto pour la recherche du voisin le plus proche.

- `IndexFlatIP`- Je suis là .`IndexFlatL2`- Force brute, exact, pas d'entraînement.
- `IndexIVFFlat` partage en cellules K, recherchez seulement les cellules les plus proches.
- `IndexHNSW` basé sur des graphiques, le plus rapide pour de nombreuses requêtes, grande taille de l'index.

Pour 100 000 vecteurs que vous voulez probablement `IndexFlatIP`Pour 10 millions, vous voulez`IndexIVFFlat`. Pour 100 millions+ combinés à la quantification des produits (`IndexIVFPQ`)

### Récupération au niveau de l'instance par rapport au niveau de la catégorie

Deux problèmes très différents avec le même nom:

- **Category-level** " trouver des chats dans mon catalogue. " similitude de classe conditionnée; les emblèmes CLIP / DINOv2 hors étagère fonctionnent bien.
- **Instance-level** "trouver *ce produit exact* dans mon catalogue". Il faut une discrimination fine entre des objets visuellement similaires de la même classe; les emblèmes off-the-shelf ne fonctionnent pas bien; l'ajustement avec les questions d'apprentissage métrique.

Demandez toujours lequel de ces problèmes vous résolvez avant de choisir un modèle.

```figure
metric-embedding
```

## Faites-le

### Étape 1: Perte de triple

```python
import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.2):
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_ap - d_an + margin).mean()
```

Une ligne, fonctionne sur des embellissements normaux ou bruts.

### Étape 2: Mining semi-difficile

Compte tenu d'un lot d'embellissements et d'étiquettes, trouvez le négatif semi-dur le plus dur pour chaque ancrage.

```python
def semi_hard_negatives(emb, labels, margin=0.2):
    dist = torch.cdist(emb, emb)
    same_class = labels[:, None] == labels[None, :]
    diff_class = ~same_class
    N = emb.size(0)

    positives = dist.clone()
    positives[~same_class] = float("-inf")
    positives.fill_diagonal_(float("-inf"))
    pos_idx = positives.argmax(dim=1)

    semi_hard = dist.clone()
    semi_hard[same_class] = float("inf")
    d_ap = dist[torch.arange(N), pos_idx].unsqueeze(1)
    semi_hard[dist <= d_ap] = float("inf")
    neg_idx = semi_hard.argmin(dim=1)

    fallback_mask = semi_hard[torch.arange(N), neg_idx] == float("inf")
    if fallback_mask.any():
        hardest = dist.clone()
        hardest[same_class] = float("inf")
        neg_idx = torch.where(fallback_mask, hardest.argmin(dim=1), neg_idx)
    return pos_idx, neg_idx
```

Chaque ancre obtient le positif le plus dur de sa catégorie et un négatif semi-dur qui est plus loin que le positif mais à l'intérieur de la marge.

### Étape 3: rappel

```python
def recall_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k=1):
    sim = query_emb @ gallery_emb.T
    _, top_k = sim.topk(k, dim=-1)
    matches = (gallery_labels[top_k] == query_labels[:, None]).any(dim=-1)
    return matches.float().mean().item()
```

Le top-k par produit interne sur les embeddings normalisés L2 est égal au top-k par cosine.

### Étape 4: Rassembler

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class Encoder(nn.Module):
    def __init__(self, in_dim=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

torch.manual_seed(0)
num_classes = 6
protos = F.normalize(torch.randn(num_classes, 128), dim=-1)

def sample_batch(bs=32):
    labels = torch.randint(0, num_classes, (bs,))
    x = protos[labels] + 0.15 * torch.randn(bs, 128)
    return x, labels

enc = Encoder()
opt = Adam(enc.parameters(), lr=3e-3)

for step in range(200):
    x, y = sample_batch(32)
    emb = enc(x)
    pos_idx, neg_idx = semi_hard_negatives(emb, y)
    loss = triplet_loss(emb, emb[pos_idx], emb[neg_idx])
    opt.zero_grad(); loss.backward(); opt.step()
```

Après quelques centaines d'étapes, les grappes d'intégration forment un grappillage par classe.

## Utilisez-le

Stacks de production en 2026:

- **DINOv2 + FAISS**- Retrait visuel à usage général.
- **CLIP + FAISS** lorsque les requêtes sont des messages texte.
- **Fine-tuned DINOv2 + FAISS** Retrouver au niveau de l'instance, réidentifier le visage, la mode, le commerce électronique.
- **Milvus / Weaviate / Qdrant** enveloppes DB vectorielles gérées autour de FAISS ou HNSW.

Pour la récupération d'instance SOTA, la recette est: DINOv2 colonne vertébrale, ajouter une tête d'intégration, régler avec un triple ou la perte InfoNCE sur les paires étiquetées par instance, index dans FAISS.

## La faire partir

Cette leçon donne:

- `outputs/prompt-retrieval-loss-picker.md` une requête qui sélectionne le triple / InfoNCE / ProxyNCA pour un problème de récupération donné.
- `outputs/skill-recall-at-k-runner.md` une compétence qui rédige un harnais d'évaluation propre pour recall@K avec des fractions train/val/galerie et un contrat de données approprié.

## Exercices

1. **(Easy)**Exécutez l'exemple de jouet ci-dessus. Tracez les embeddings avec PCA avant et après l'entraînement pour voir les six grappes se forment.
2. **(Medium)**Ajouter une mise en œuvre de perte de ProxyNCA: un "proxy" appris par classe, entropie croisée standard sur la similitude cosine. Comparer la vitesse de convergence par rapport à la perte triple sur les données de jouets.
3. **(Hard)**Prenez 1000 images de validation d'ImageNet, intégrez-les à DINOv2 via HuggingFace, construisez un index FAISS plat et rapportez le recall@{1, 5, 10} contre les mêmes images que les requêtes (devaient être 1.0) et contre une fraction prolongée avec les étiquettes d'ImageNet comme vérité de fond.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Metric learning | "Shape the space" | Training an encoder so distances in its output space reflect a target similarity |
| Triplet loss | "Pull and push" | $L = \max(0, d(a, p) - d(a, n) + \text{margin})$; the canonical metric-learning loss |
| Semi-hard mining | "Useful negatives" | Negatives further from the anchor than the positive but within margin; empirically the most informative |
| Proxy-based loss | "Class prototypes" | One learned proxy per class; cross-entropy over similarity-to-proxies; no pair mining |
| Recall@K | "Top-K hit rate" | Fraction of queries with at least one correct result in the top K |
| Instance retrieval | "Find this exact thing" | Fine-grained matching; off-the-shelf features usually underperform |
| FAISS | "The NN library" | Facebook's nearest-neighbour library; supports exact and approximate indexes |
| HNSW | "Graph index" | Hierarchical navigable small world; fast approximate NN with small memory overhead |

## Pour en savoir plus

- [FaceNet: A Unified Embedding for Face Recognition (Schroff et al., 2015)](https://arxiv.org/abs/1503.03832) la perte de triplets / papier minier semi-dur
- [In Defense of the Triplet Loss for Person Re-Identification (Hermans et al., 2017)](https://arxiv.org/abs/1703.07737) Guide pratique pour l'ajustement fin des trois
- [FAISS documentation](https://github.com/facebookresearch/faiss/wiki) chaque indice, chaque compromis
- [SMoT: Metric Learning Taxonomy (Kim et al., 2021)](https://arxiv.org/abs/2010.06927) étude des pertes modernes et de leurs liens
