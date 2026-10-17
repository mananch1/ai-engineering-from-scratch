# Une attention à plusieurs têtes

> Un chef d'attention apprend une relation à la fois, huit têtes apprennent huit têtes sont libres, prenez-en plus.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## Le problème

Une seule tête d'attention à soi compute une matrice d'attention. Cette matrice capture un type de relation  généralement celle qui minimise la perte sur quoi que ce soit le signal d'entraînement. Si vos données ont un accord sujet-verbe, une co-référence, un discours à longue portée et un chunking syntaxique tous enracinés ensemble, une seule tête les épluche dans une seule distribution de max douce et perd la moitié du signal.

La correction du papier Vaswani 2017: exécuter plusieurs fonctions d'attention en parallèle, chacune avec ses propres projections Q, K, V, et concatener les sorties. Chaque tête fonctionne dans un sous-espace de dimension plus petit $d_{\text{model}} / n_{\text{heads}}$Les paramètres totaux restent les mêmes.

L'attention multi-tête est la norme par défaut de chaque transformateur dans les navires de 2026. Le seul argument est sur * combien de têtes* et si les touches et les valeurs partagent des projections (attention groupée, attention multi-quéret, attention latente multi-tête).

## Le concept

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.**Prenez .`X`de forme `(N, d_model)`- Projet à Q, K, V de forme`(N, d_model)`- Ressoufflez-vous .`(N, n_heads, d_head)`où $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$- Transposer à`(n_heads, N, d_head)`- Je suis désolé .

**Attend in parallel.**Exécutez des produits à l'échelle de points à l'intérieur de chaque tête.`(N, d_head)`Les têtes fonctionnent sur différents sous-espaces de l'embedding et ne parlent jamais pendant le calcul de l'attention lui-même.

**Concatenate and project.**Les têtes de pile retournent à `(N, d_model)`et multiplier par une matrice de sortie apprise `W_o`de forme `(d_model, d_model)`- Je suis là .`W_o`C'est là que les têtes se mélangent.

**Why it works.**Chaque tête peut se spécialiser sans rivaliser avec les autres pour le budget représentatif. Les études de sondage de 2019 à 2024 montrent des rôles distincts de tête: les têtes positionnelles, la tête qui assiste au jeton précédent, les têtes de copie, les têtes d'entité nommée, les têtes d'induction (qui sous-tendent l'apprentissage dans le contexte).

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA est la norme par défaut moderne car elle réduit la mémoire KV-cache d'un facteur de $N/G$MLA va plus loin en comprimant K/V dans un espace latent, puis en le projetant à l'heure de calcul  coûte FLOPs, économise beaucoup plus de mémoire.

```figure
multihead-split
```

## Faites-le

### Étape 1: séparer les têtes de l'attention à tête unique que nous avons déjà

Prenez le `SelfAttention`Leur résultat est le résultat de la leçon 02 et de l'envelopper avec une paire de fractionnement/concétation.`code/main.py`pour une mise en œuvre numpy; la logique est:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Une remodèle et une transposition, pas de boucle, c'est exactement ce que fait PyTorch sous`nn.MultiheadAttention`- Je suis désolé .

### Étape 2: exécuter des points-produit à l'échelle de l'attention par tête

Chaque tête obtient sa propre tranche de Q, K, V. L'attention devient un matmul en lots:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Sur du matériel réel .`Qh @ Kh.transpose(...)`est un`bmm`La GPU voit une seule forme de matmul .`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`- Ajouter des têtes est gratuit.

### Étape 3: Variante de l'attention de la requête groupée

Seules les projections de clé et de valeur changent.`n_heads`les groupes; K et V obtiennent `n_kv_heads < n_heads`les groupes et sont répétés pour correspondre:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

En conséquence , cela permet d' économiser la mémoire , car seulement`n_kv_heads`Les copies sont en direct dans le cache KV, pas `n_heads`Llama 3 70B utilise 64 têtes de requête avec 8 têtes KV  un rétrécissement de cache 8x.

### Étape 4: enquête sur ce que chaque tête a appris

Exécutez MHA sur une phrase courte avec 4 têtes.`(N, N)`Vous verrez différentes têtes choisir différentes structures même avec l'initialisation aléatoire qui est en partie le signal, en partie la symétrie rotative dans les sous-espaces.

## Utilisez-le

Dans PyTorch, la version à une ligne:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA à partir de PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?**Règles générales des modèles de production en 2026:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`Il est l'unité de la quantité qu'une tête peut "voir".$\sqrt{d_{\text{head}}}$Si vous dépassez 256 points, vous perdez le bénéfice "beaucoup de petits spécialistes".

## La faire partir

Regardez !`outputs/skill-mha-configurator.md`. La compétence recommande le nombre de têtes, le nombre de têtes kv et la stratégie de projection pour un nouveau transformateur, compte tenu du budget des paramètres, de la longueur des séquences et de l'objectif de déploiement.

## Exercices

1. **Easy.**Prenez le MHA de `code/main.py`et le changement `n_heads`de 1 à 16 avec `d_model=64`Une fois que vous avez terminé, vous pouvez faire une copie synthétique.
2. **Medium.**MQA (un seul KV partagé sur tous les têtes de requête). Mesurer combien de paramètres du compte de gouttes par rapport à MHA complet. Comptez combien la taille de la caisse KV se réduit à l'inférence pour N = 2048.
3. **Hard.**Mettez en œuvre une version minuscule de l'attention latente multi-tête: comprimant K, V à un rang-`r`Laissez-le dans le cache KV, décomprimez-le à l'heure de l'attention.`r`la mémoire cache passe-t-elle en dessous de 1/8 de la MHA complète tandis que la qualité reste à l'intérieur de 1 bit de la validation ?

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$ with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## Pour en savoir plus

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) la spécification originale de plusieurs têtes.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) le document MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) comment convertir l'AMM en GQA après la formation.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) MLA et pourquoi il bat MHA/GQA sur la mémoire cache.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) Regardez mécaniquement ce que font réellement les têtes.
