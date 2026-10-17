# Pourquoi les transformateurs  Les problèmes avec les RNN

> Les RNN traitent les jetons un à la fois. Les transformateurs traitent tous les jetons à la fois. Ce pari architectural unique a changé chaque courbe d'échelle dans l'apprentissage en profondeur après 2017.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## Le problème

Avant 2017, chaque modèle de séquence de pointe sur la planète  langue, traduction, parole  était un réseau neuronal récurrent. Les LSTM et GRU ont obtenu des benchmarks de traduction équivalents à ImageNet pendant une demi-décennie. Ils étaient le seul outil que quelqu'un avait.

Le calcul séquentiel ne permettait pas de paralléliser l'axe temporel.`t+1`Il faut le secret de l' état de la marque .`t`Une séquence de 1.024 jetons signifiait 1.024 étapes sérielles sur un GPU qui peut effectuer 1.000.000 opérations de points flottants par cycle.

Les gradients disparus signifiaient que les informations 50 jetons de retour étaient déjà compressées à travers 50 non-linéaires. Les unités récurrentes par voie de fer (LSTM, GRU) ont atténué la dépression mais ne l'ont jamais éliminée.

Les états cachés de largeur fixe signifiaient que l'encodeur comprimait toute la séquence source dans un seul vecteur avant que le décodeur ne voie quoi que ce soit.

L'article de 2017 "Attention est tout ce dont vous avez besoin" a proposé quelque chose de radical: supprimer complètement la récurrence. Laissez chaque position attirer chaque autre position en parallèle.

Le résultat domine toutes les modalités d'ici 2026. Langue (GPT-5, Claude 4, Llama 4), vision (ViT, DINOv2, SAM 3), audio (Whisper), biologie (AlphaFold 3), robotique (RT-2).

## Le concept

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck.**Un RNN compute $h_t = f(h_{t-1}, x_t)$Chaque étape dépend de la précédente.$h_5$avant $h_4$Sur les GPU modernes avec plus de 10 000 cœurs parallèles, cela gaspille 99% du silicium sur une longue séquence.

**Attention as a broadcast.**Les calculs de l' attention personnelle `output_i = sum_j(a_ij * v_j)`Pour chaque paire $(i, j)$Tout le matrix d'attention N×N remplit un matmul en lots.

**The speedup is not a constant.**C' est la différence entre $O(N)$profondeur de série et $O(1)$En pratique, les transformateurs s'entraînent 510x plus vite par époque sur le matériel correspondant à N=512, et l'écart s'élargit avec la longueur de la séquence jusqu'à ce que vous atteignez le$O(N^2)$paroi mémoire de l'attention (qui Flash Attention a plus tard réparé  voir leçon 12).

**What transformers cost.**Les échelles de mémoire d' attention sont:$O(N^2)$Pour le contexte 2K, bien. Pour le contexte 128K, vous avez besoin de fenêtres coulissantes, extrapolation RoPE, carreaux d'attention flash, ou variantes d'attention linéaire.$O(N)$Les transformateurs échangent le temps contre la mémoire et gagnent ensuite le temps par le parallélisme.

**The inductive bias shift.**Les transformateurs ne prennent rien en compte.  chaque paire est un candidat à l'attention. C'est pourquoi les transformateurs ont besoin de plus de données pour bien s'entraîner mais à évoluer une fois qu'ils l'ont. Chinchilla (2022) a formalisé ceci: donné suffisamment de jetons, un transformateur bat toujours un RNN d'un nombre de paramètres égal.

```figure
rnn-vs-parallel
```

## Faites-le

Aucun réseau neural ici  nous simulons le goulet d'étranglement du noyau numériquement afin que vous puissiez sentir l'écart sur votre ordinateur portable.

### Étape 1: mesurer la profondeur de série

Regardez !`code/main.py`Nous construisons deux fonctions. Une encode une séquence comme une chaîne d'additions (série, comme un RNN). Une encode comme une réduction parallèle (diffusion, comme l'attention).

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Nous faisons des séquences jusqu'à 100 000 éléments.$O(N)$Même dans Python pur, la réduction de l'attention est supérieure à 1000 parce que Python est`sum()`est mis en œuvre en C et se répète sans frais d'interprétation par étape.

### Étape 2: compter les opérations théoriques

Les deux algorithmes ajoutent N. La différence est * profondeur de dépendance*: combien d'opérations doivent se produire séquentiellement avant que la prochaine puisse commencer. RNN profondeur = N. profondeur d'attention = log(N) avec une réduction d'arbre, ou 1 avec un scan parallèle.

### Étape 3: Échantillonnage empirique sur de longues séquences

Nous imprimons un tableau de calendrier qui fait le $O(N)$Les séquences de 100 000 montrent une analyse linéaire propre. Étalons cela à un transformateur de 16 384 jetons avec un équivalent LSTM de 12 couches et vous voyez pourquoi l'entraînement de l'horloge murale était un blocage en 2016.

## Utilisez-le

Quand encore choisir un RNN en 2026:

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

Les modèles de l'espace-état (SSM) comme Mamba sont essentiellement des RNN avec une paramétrisation structurée qui leur donne le meilleur des deux: $O(N)$Les tests de réparation de la mémoire de la machine à écrire, de la formation parallèle par le biais d'un scan sélectif. Ils récupèrent 90% de la qualité du transformateur avec une meilleure mise à l'échelle à long contexte.

## La faire partir

Regardez !`outputs/skill-architecture-picker.md`. La compétence choisit une architecture pour un nouveau problème de séquence compte tenu de la longueur, du débit et des contraintes budgétaires de formation.

## Exercices

1. **Easy.**Prenez .`rnn_style`de `code/main.py`et remplacer l'état caché scalaire par un vecteur de longueur-64 d'états cachés.
2. **Medium.**Implémenter une somme parallèle de préfixe (scan Hillis-Steele) en Python pur. Vérifiez qu'il produit la même sortie numérique qu'un scan en série sur la longueur 1024.
3. **Hard.**Porté la réduction de l'attention à PyTorch sur GPU. temps à la fois que vous fouillez la longueur de la séquence de 64 à 65.536.

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
| Attention | "Let tokens look at each other" | Weighted sum $\sum_j a_{ij} v_j$ where $a_{ij}$ comes from a similarity score between positions $i$ and $j$. |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = $O(N^2)$ in sequence length; Flash Attention hides the constants, not the scaling. |

## Pour en savoir plus

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) l'article qui a tué la récurrence dans la PNL traditionnelle.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) où l'attention est née, boulonné sur un RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) le papier LSTM original, pour le compte rendu.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) Réponse récurrente moderne aux transformateurs.
