# Récupération du point de contrôle et de l'activation

> Backprop conserve chaque activation intermédiaire. À 70B paramètres et 128K contexte qui est 3 TB d'activations par rang. Checkpointing négocie FLOPs pour la mémoire: recomputer au lieu de sauvegarder. La question est de savoir quels segments à laisser tomber, et la réponse n'est pas "tous".

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## Le problème

La formation d'un transformateur stocke, pour chaque couche, les entrées de chaque opération différenciée en arrière: les entrées d'attention, les projections Q/K/V, la sortie softmax, les entrées FFN, les sorties de norme et le flux résiduel.`d`, longueur de la séquence `L`, lot `B`, c' est à l' ordre de `12 * B * L * d`flottent par couche.

Pour `d=8192, L=8192, B=1`Un modèle de 64 couches est 51 Go d'activations  et c'est avant que vous multipliez par la taille du microbatch, avant que vous ajoutiez des intermédiaires attention-softmax (`L^2`par tête), et avant de faire des copies partielles parallèles à la tenseur.

Le projet de loi bilatéral: les poids BF16 plus l'état de l'optimisateur peuvent s'adapter à 80 Go, mais les activations vous poussent au-delà. Le contrôle de gradation (alias recomptage d'activation) est la solution standard.

Fait naïvement, le checkpointing coûte environ 33% de plus de FLOPs par étape. Fait bien  checkpointing sélectif par " sélection intelligente " de Korthikanti et al.  vous économisez 5x de mémoire pour moins de 5% de FLOP. Et avec les matmuls FP8, FSDP déchargement, et expert-parallèle MoE cela compte vraiment: vous ne pouvez pas se permettre ni la mémoire ni le calcul gaspillé.

## Le concept

### Ce dont les personnes arriérées ont réellement besoin

`output = layer(input)`- Des désirs rétrograde .`grad_input`et `grad_params`Pour les calculer , il faut:

- `input`(pour calculer `grad_params = input.T @ grad_output`pour les couches linéaires)
- certains intermédiaires des dérivés d'activation (la dérivée de ReLU/GELU/softmax dépend de la valeur d'activation)

Le passe avant stocke automatiquement dans le graphique de l'autograd.`tensor.retain_grad()`et chaque opération qui a besoin de son input conserve une référence.

### Une inspection naïve

Divisez le réseau en deux .`N`Les segments. pendant la phase avant, stocker uniquement l' * entrée * à chaque segment. Lorsque l'arrière a besoin d'intermédiaires, rediriger le passage avant du segment pour les matérialiser, puis différencier.

Exemple: transformateur à 32 couches divisé en 32 segments de 1 couche chacun.

- Mémoire: 32 entrées de couche (petites) contre 32 * (volume d'activation par couche) (énormes).
- Compute supplémentaire: 1 unité supplémentaire à l'avant par segment, c'est-à-dire ~33% de plus de FLOP à l'avant total (puisque l'arrière est 2 fois plus avant, l'étape complète devient 1 + 1 + 2 = 4 unités au lieu de 1 + 2 = 3).

C'est la recette originale de Chen et al. 2016: un point de contrôle par personne `sqrt(L)`Les couches permettent d'équilibrer la mémoire et le calcul.$L=64$, c'est 8 points de contrôle.

### Le contrôle sélectif (Korthikanti 2022)

Toutes les activations ne coûtent pas la même chose.`B*L*L*heads`L'activation cachée du FFN est `B*L*4d`Pour les longues séquences, le softmax domine.

Le point de contrôle sélectif maintient les activations à bas prix (projections linéaires, résidus) et recompte uniquement les plus chères (attention).$O(L^2)$Je ne sais pas.

Megatron-Core implémente cela comme recomputation d'activation " sélective ". Utilisé dans la plupart des séries d'entraînement frontaliers de 2024+.

### Déchargement

Alternative au recomptage: envoyer des activations à la RAM de la CPU entre l'avant et l'arrière. Requiert une largeur de bande PCIe; bénéfique lorsque la bande passante inactive dépasse le coût de la rematérialisation.

FSDP2 décharge comme une option de première classe. déchargement brille lorsque la GPU est bloquée sur la mémoire mais le transfert CPU-GPU a de la place.

### Modèle de coûts de recompte

Des échecs à chaque étape avec des contrôles naïfs à chaque étape.`k`couches de `L`- Le numéro de la liste:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Avec le point de contrôle sélectif, vous ne recommencez à calculer que le noyau d'attention, pas toute la couche:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Modèle d'épargne de mémoire

Volume d'activation par couche: `A`Pour ...`L`couches, mémoire d'activation totale: `L * A`- Je suis désolé .

Point de contrôle complet (taille de segment 1): réservoir uniquement `L * input_volume`(~)`L * 1/10 A`Pour un transformateur standard).`9 * L * A * 1/10`- Je suis désolé .

- Le point de contrôle à chaque fois .`k`couches: stockage `L/k * A`plus `k-1`la valeur des couches au sein du segment actif.

À `k = sqrt(L)`, la mémoire et le coût de recomptage à la fois`sqrt(L)` l'offre optimale pour les couches de coûts uniformes.

### Quand on ne va pas au point de contrôle

- Les couches les plus profondes d'un pipeline sont déjà en vol.
- Les premières et dernières couches si elles dominent le calcul de la scène (rares chez les transformateurs).
- Les noyaux d'attention utilisant déjà FlashAttention  Flash recompte déjà le softmax rapidement, donc le point de contrôle supplémentaire au niveau de couche ajoute peu en haut.

### Modèles de mise en œuvre

1. **Function wrapper:**Envelopper un segment dans `torch.utils.checkpoint.checkpoint(fn, input)`- Les magasins PyTorch seulement .`input`, recompte tout le reste en arrière.

2. **Decorator-based:**les couches d'étiquette comme contrôlables; le formateur décide au moment de la configuration des segments qui seront enveloppés.

3. **Manual explicit recompute:**Écrivez vous-même la passe arrière, appelant une coutume `recompute_forward`qui duplique le forward avec l'entrée stockée.

Les trois donnent le même résultat fonctionnel.

### Interaction avec le TP / PP / FP8

- **Tensor parallel:**Les entrées des points de contrôle doivent être collectées ou réparties sur le recomptage; gérer les coûts de communication.
- **Pipeline parallel:**Le modèle typique est de contrôler l'avancement de chaque étape du pipeline afin que les micro-batches d'ordre inverse puissent réutiliser la mémoire d'activation.
- **FP8 recompute:**Les historiques d'amax mises à jour pendant la recompte doivent correspondre aux dérives de l'échelle FP8 ou à celles de l'avance d'origine.

```figure
activation-recompute
```

## Faites-le

### Étape 1: Un modèle de jouet avec des segments

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Étape 2: Naïf à l'arrière qui a besoin de toutes les activations

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Étape 3: Point de contrôle - Chaque mémoire

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Étape 4: Modèle de coût

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Étape 5: Évaluateur de mémoire

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Étape 6: Taille optimale du segment

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Étape 7: Décision sélective sur le point de contrôle

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Utilisez-le

- **torch.utils.checkpoint**Le numéro de la liste:`from torch.utils.checkpoint import checkpoint` l'emballage canonique dans PyTorch. Enveloppe une fonction; stocke uniquement les entrées, recompte en arrière.
- **Megatron-Core activation recomputation**: appuie `selective`- Je suis là .`full`, et `block`Les modes de formation standard en 2024+
- **FSDP2 offload**Le numéro de la liste:`module.to_empty(device="cpu")`avec `offload_policy`dans FSDP2 les activations partagées vers le CPU au lieu de recomputer.
- **DeepSpeed ZeRO-Offload**: décharge de la CPU pour les états et les activations de l'optimisateur, complétant le point de contrôle.

## La faire partir

Cette leçon produit `outputs/prompt-activation-recompute-policy.md` une requête qui prend votre configuration de modèle (couches, cachées, séq, lots) et la mémoire GPU disponible et émet une politique de recomptage par couche (pas de couche / sélective / plein / déchargement).

## Exercices

1. Vérifiez la précision.`model_forward`+ `model_backward`(activations complètes)`model_forward_checkpointed`+ `model_backward_checkpointed`Les gradients de paramètre doivent être identiques à la précision de la machine.

2. Taille du segment de balayage `k`de 1 à `L`- Trouvez le genou de la courbe.

3. Mettre en œuvre un point de contrôle sélectif: stocker l'entrée du module d'attention mais pas ses intermédiaires. Mesurer le point de contrôle FLOP en fonction de la charge supérieure par rapport au point de contrôle à couche complète pour un modèle de 32 couches à seq=8192.

4. Ajouter le déchargement. Enregistrer les entrées de segment dans un "buffer du processeur" simulé (une liste distincte). Mesurer "largeur de bande PCIe" en octets/temps et trouver le point d'équilibre entre le déchargement et le recomptage.

5. Réservez un véritable transformateur PyTorch avec et sans `torch.utils.checkpoint`. Mesurer la mémoire (via `torch.cuda.max_memory_allocated`) et le temps de marche.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | "Save memory by redoing forward" | Store segment inputs only; recompute intermediates during backward to get gradient-support tensors |
| Activation recomputation | "Same as checkpointing" | The HPC-flavored name for the same technique |
| Segment size (k) | "How many layers per checkpoint" | Number of layers whose intermediates are dropped and rematerialized together |
| Selective checkpointing | "Korthikanti's trick" | Recompute only expensive-to-store activations (attention softmax); keep cheap ones |
| Full checkpointing | "The naive version" | Recompute every layer's intermediates in every segment |
| Block checkpointing | "Coarse-grained" | Checkpoint whole transformer blocks; largest granularity |
| FLOP overhead | "The compute tax" | Extra FLOPs per step = (recompute FLOPs) / (fwd + bwd FLOPs); 33% naive, 5% selective |
| Activation offload | "Ship to CPU" | Move activations to CPU RAM across forward->backward; alternative to recompute |
| sqrt-L rule | "The classical optimum" | For uniform-cost layers, optimal checkpoint spacing is $\sqrt{L}$ layers |
| Attention-softmax volume | "The O(L^2) problem" | $L^2 \cdot \text{heads} \cdot \text{batch}$ floats; dominates activation memory at long contexts |

## Pour en savoir plus

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174)-- le papier original qui formait le point de contrôle des gradients
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198)-- recomputation sélective de l'activation et analyse formelle des coûts
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645)-- une approche alternative de la mémoire constante via la rematérialisation en mode inverse
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840)-- décharge d'activation à l'échelle
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html)-- l'API standard
- [Megatron Bridge activation recomputation documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)-- modes sélectifs, complets et bloqués
