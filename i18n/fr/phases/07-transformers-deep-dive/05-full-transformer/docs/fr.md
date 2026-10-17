# Le transformateur complet  Encoder + Décoder

> L'attention est l'étoile. Tout le reste, résidus, normalisation, flux, attention croisée, est l'échafaudage qui vous permet de l'empiler profondément.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## Le problème

Une seule couche d'attention est un extracteur de caractéristiques, pas un modèle. Un matmul par couche n'est pas suffisant pour la capacité de langue. Vous avez besoin de profondeur  et de profondeurs de ruptures sans la bonne plomberie.

Le papier Vaswani 2017 a compilé six décisions de conception qui ont transformé une couche d'attention en un bloc empilable. Chaque transformateur depuis  encodeur-seul (BERT), décodeur-seul (GPT), encodeur-decodeur (T5)  hérite du même squelette.

Cette leçon est le squelette. Les leçons suivantes le spécialisent  06 pour les encoders, 07 pour les décoders, 08 pour l'encodeur-décodeur.

## Le concept

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### Les six pièces

1. **Embedding + positional signal.**Tokens → vecteurs. Position injectée par RoPE (moderne) ou sinusoïdale (classique).
2. **Self-attention.**Chaque position est à l'aise avec l'autre.
3. **Feed-forward network (FFN).**PAM à deux couches en fonction de la position: `W_2 · activation(W_1 · x)`- Le ratio d'expansion est 4x par défaut.
4. **Residual connection.** `x + sublayer(x)`Sans cela, les gradients disparaissent après environ 6 couches.
5. **Layer normalization.** `LayerNorm`ou `RMSNorm`(moderne) stabilise le flux résiduel.
6. **Cross-attention (decoder only).**Les requêtes proviennent du décodeur, des clés et des valeurs de la sortie du codeur.

Observez le flux d'un vecteur à travers un bloc: l'attention se mélange entre les positions, le résidu le transporte vers l'avant, le FFN le transforme et la norme maintient le flux stable.

```figure
transformer-block
```

### Bloc d'encodeur (utilisé par BERT, encodeur T5)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Le codeur est bidirectionnel, pas de masquage, toutes les positions voient toutes les positions.

### Bloc de décodeur (utilisé par GPT, décodeur T5)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Le décodeur a trois sous-couches par bloc. Le centre  attention croisée  est le seul endroit où les informations circulent du décodeur au décodeur. Dans une architecture pure de décodeur uniquement (GPT), l'attention croisée est omise et vous avez simplement masqué l'attention personnelle + FFN.

### Pré-norme contre post-norme

Papel original: `x + sublayer(LN(x))`contre`LN(x + sublayer(x))`. Après la normalisation, il est plus difficile de s'entraîner profondément sans un réchauffement minutieux.`LN`* avant * sous-couche) est la version par défaut 2026: Llama, Qwen, GPT-3+, Mistral l'utilisent tous.

### Le bloc modernisé de 2026

Vaswani 2017 a été livré LayerNorm + ReLU. Les piles modernes ont remplacé les deux.

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm réduit la moyenne de centre de LayerNorm (une soustraction inférieure), ce qui permet de réduire les calculs et est empiriquement au moins aussi stable.`Swish(W1 x) ⊙ W3 x`) dépasse systématiquement le RELU/GELU FFN de ~0,5 points par rapport aux documents Llama, PaLM et Qwen.

### Compte de paramètres

Pour un bloc avec `d_model = d`et l'expansion du FFN `r`- Le numéro de la liste:

- MHA: `4 · d²`(Projections Q, K, V, O)
- FFN (SwiGLU): `3 · d · (r · d)`- Je suis là.`3rd²`
- Normes: négligeables

À `d = 4096, r = 2.6, layers = 32`(environ Llama 3 8B), total: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`Les correspondances publiées sont comptées.

## Faites-le

### Étape 1: les blocs de construction

En utilisant le minuscule`Matrix`classe de la leçon 03 (copié dans ce dossier pour l'indépendance):

- `layer_norm(x, eps=1e-5)` Soustraire la moyenne, diviser par std.
- `rms_norm(x, eps=1e-6)` Divise par RMS. Aucune soustraction moyenne.
- `gelu(x)`et `silu(x) * W3 x`Je suis désolé.
- `ffn_swiglu(x, W1, W2, W3)`- Je suis désolé .
- `encoder_block(x, params)`et `decoder_block(x, enc_out, params)`- Je suis désolé .

Regardez !`code/main.py`pour le câblage complet.

### Étape 2: brancher un encodeur à 2 couches et un décodeur à 2 couches

Passez l'enregistrement du codeur à chaque décodeur, ajoutez un LN final avant la projection de sortie.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Étape 3: aller de l'avant sur un exemple de jouet

Passez une source de 6 jetons et une cible de 5 jetons.`(5, vocab)`Cette leçon est sur l'architecture, pas sur la perte.

### Étape 4: échange en RMSNorm + SwiGLU

Remplacez LayerNorm et ReLU-FFN par RMSNorm et SwiGLU. Confirmez que les formes correspondent toujours.

## Utilisez-le

Les mises en œuvre de référence PyTorch/TF: `nn.TransformerEncoderLayer`- Je suis là .`nn.TransformerDecoderLayer`Mais la plupart des codes de production 2026 ont leur propre bloc car:

- L'attention flash est appelée à l'intérieur de l'attention, pas par l'intermédiaire de `nn.MultiheadAttention`- Je suis désolé .
- Les GQA/MLA ne sont pas dans la référence stdlib.
- RoPE, RMSNorm, SwiGLU ne sont pas les défauts PyTorch.

HF `transformers`a des blocs de référence propres que vous devriez lire: `modeling_llama.py`C'est le bloc canonique 2026 avec un décodeur uniquement.

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

Le décodeur-seul a gagné le langage parce qu'il est le plus propre et gère à la fois la compréhension et la génération.

## La faire partir

Regardez !`outputs/skill-transformer-block-reviewer.md`. La compétence examine une nouvelle mise en œuvre de bloc de transformateur contre les défauts de 2026 et détecte les pièces manquantes (pre-norme, RoPE, RMSNorm, GQA, FFN ratio d'expansion).

## Exercices

1. **Easy.**Comptez les paramètres dans votre encoder_block à `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Valider en mettant en œuvre le bloc et en utilisant `sum(p.numel() for p in block.parameters())`- Je suis désolé .
2. **Medium.**Passez de post-norme à pré-norme. Initializer les deux et mesurer la norme d'activation après 12 couches empilées sur entrée aléatoire.
3. **Hard.**Implémenter un encodeur-décodeur à 4 couches sur une tâche de copie de jouet (copie `x`Retour à la ligne de départ.

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks. |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same empirical stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model, usually 4 (LayerNorm) or 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers; slight ppl improvement, smaller model. |

## Pour en savoir plus

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) spécifications originales du bloc.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)Pourquoi la pré-norme bat profondément la post-norme.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) le papier SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) bloc canonique 2026 décoteur seulement.
