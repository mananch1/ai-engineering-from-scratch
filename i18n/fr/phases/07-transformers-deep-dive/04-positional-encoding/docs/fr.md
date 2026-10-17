# Codification de position  Sinusoïdale, RoPE, ALiBi

> L'attention est invariable en permution. "Le chat s'est assis sur le tapis" et "mat le chat sur le sat" produisent la même sortie sans signal de position. Trois algorithmes le fixent  chacun avec un pari différent sur ce que signifie "position".

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## Le problème

L'attention par produit de point est aveugle à l'ordre.`softmax(Q K^T / √d) V`est calculé à partir de similitudes par paires.`X`La position de l'attention ne compte pas pour l'intérieur.

Ce n'est pas un bug dans un modèle de sacs de mots. Pour le langage, le code, l'audio, la vidéo  tout ce qui a un sens dans l'ordre  c'est fatal.

La solution est d'injecter la position dans les embeddings.

1. **Absolute sinusoidal**(Vaswani 2017). Ajouter `sin/cos`Il est simple, sans apprentissage, et il est peu extrapolé au-delà des longues longues.
2. **RoPE — Rotary Position Embeddings**(Su 2021). Rotate les vecteurs Q et K par un angle proportionnel à la position. Encode la position *relative* directement dans le produit de point. Dominant en 2026.
3. **ALiBi — Attention with Linear Biases**(Prés 2022). Sautez entièrement les emblèmes; ajoutez une pénalité linéaire par tête aux scores d'attention en fonction de la distance. Excellente extrapolation de longueur.

En 2026, pratiquement tous les modèles frontaliers ouverts utilisent le RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi.

## Le concept

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Une forme sinusoïdale absolue

Précompte une matrice fixe `PE`de forme `(max_len, d_model)`- Le numéro de la liste:

```math
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(pos / 10000^{2i / d_{\text{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(pos / 10000^{2i / d_{\text{model}}}\right)
\end{aligned}
```

Alors ...`X' = X + PE[:N]`Chaque dimension est un sinus à une fréquence différente.`max_len`: rien n'a dit au modèle ce qui se passe à la position 2048 quand il ne voyait que les positions 02047.

### REPE

Rotation des vecteurs Q et K (pas intégrés). Pour une paire de dimensions $(2i, 2i+1)$- Le numéro de la liste:

```math
\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\ \sin(pos \cdot \theta_i) & \cos(pos \cdot \theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
\qquad \theta_i = \text{base}^{-2i / d_{\text{head}}}, \quad \text{base} = 10000 \text{ by default}
```

Appliquer la même rotation aux touches avec position `pos_k`Le produit de la dot$q'_m \cdot k'_n$devient une fonction de $(m - n)$Je suis seul.**the attention score depends only on the relative distance**Même si la rotation était bloquée.

Extension de l' ROPE: `base`Llama 3 est étendu de 8K à 128K de cette façon.

### Le groupe

Laissez tomber le truc de l'intégration.

```math
\text{attn\_score}[i, j] = \frac{q_i \cdot k_j}{\sqrt{d}} - m_h \cdot |i - j|
```

Où ?$m_h$est une pente spécifique à la tête (par exemple $1 / 2^{8h/H}$Les jetons plus proches sont renforcés; les jetons plus éloignés sont pénalisés. Aucun coût de temps d'entraînement.

### Que choisir en 2026

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

Le RoPE a gagné parce qu'il attire l'attention sans changer l'architecture, encode la position relative et ses`base`l'hyperparamètre donne un bouton propre pour l'ajustement fin de long contexte.

```figure
rope-explorer
```

## Faites-le

### Étape 1: codage sinusoïdale

Regardez !`code/main.py`Un calcul à quatre lignes:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Ajoutez cela à la matrice d'intégration avant la première couche d'attention.

### Étape 2: RoPE appliqué à Q, K

Le RoPE fonctionne sur place sur Q et K. Pour chaque paire de décolores:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Crucial: appliquer la même fonction à Q à la position $m$et K à la position $n$Leur produit dot prend un$\cos((m-n) \cdot \theta_i)$L'attention apprend gratuitement la position relative.

### Étape 3: Pistes et biais ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Ajouter `bias[h]`à la `(seq_len, seq_len)`Matrice de score d' attention de la tête `h`, puis softmax.

### Étape 4: vérifier la propriété relative à la distance de la RoPE

Choisissez deux vecteurs aléatoires `a, b`- Retourne par`(pos_a, pos_b)`- Alors , par`(pos_a + k, pos_b + k)`- Les deux produits de point doivent correspondre à l'erreur de point flottant. Cette propriété est l'ensemble du point de RoPE  il est invariant à l'opposition absolue, seule l'écart relatif compte.

## Utilisez-le

PyTorch 2.5+ expédie des équipements RoPE en `torch.nn.functional`La plupart des codes de production utilisent`flash_attn`ou `xformers`où le RoPE est appliqué à l'intérieur du noyau d'attention.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.**Réscale `base`à `base * (scale_factor)^(d/(d-2))`lorsque vous passez de 4K à 16K+.
- **YaRN.**Une interpolation plus intelligente qui préserve l'entropie de l'attention dans de longs contextes.
- **LongRoPE.**La méthode 2024 de Microsoft qui utilise la recherche évolutionniste pour choisir des facteurs d'échelle par dimension.
- **Position interpolation + fine-tuning.**Réduisez vos positions par le facteur d'extension et réglez les jetons de 15B.

## La faire partir

Regardez !`outputs/skill-positional-encoding-picker.md`. La compétence choisit une stratégie de codage pour un nouveau modèle compte tenu de la longueur du contexte cible, des besoins d'extrapolation et du budget de formation.

## Exercices

1. **Easy.**Tracer le sinus.`PE`matrice comme carte thermique pour `max_len=512, d=128`Confirmer le modèle "les bandes s'élargissent à mesure que l'indice de dimension augmente".
2. **Medium.**Mettre en œuvre une mise à l'échelle de RoPE à la connaissance du NTK. Exercer un petit LM sur des séquences de longueur 256, puis tester sur longueur 1024 avec et sans mise à l'échelle. Mesurer la perplexité.
3. **Hard.**Mettre en œuvre ALiBi et RoPE dans le même module d'attention. entraîner un transformateur à 4 couches sur une tâche de copie avec des séquences de longueur 512. Extrapoler à 2048 au moment de l'essai. Comparer la dégradation.

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position. |
| Sinusoidal | "The original one" | `sin/cos` at geometric frequencies added to embeddings; doesn't extrapolate. |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance. |
| ALiBi | "Linear bias trick" | Add $-m \cdot \lvert i-j \rvert$ to attention scores; no embedding needed, great extrapolation. |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context at inference. |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed when context expands. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation that preserves attention entropy. |
| Extrapolation | "Works beyond trained length" | Can the position scheme serve correct output past `max_len` seen in training? |

## Pour en savoir plus

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) sinusoïdale originale.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) papier RoPE.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)- Il est bien.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) l'état de l'art de l'échelle RoPE.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) Llama 2 du Meta, un document dans un long contexte.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) la méthode Microsoft utilisée par Phi-3-Long et citée dans la section Utiliser.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) Implémentations de chaque système d'échelle RoPE en classe de production (par défaut, linéaire, dynamique, YaRN, LongRoPE, Llama-3).
