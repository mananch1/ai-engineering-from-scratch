# Modélisation autorégressive visuelle (VAR): prédiction à l'échelle suivante

> Les modèles de diffusion échantillonnent de manière itérative dans le temps (désignant les étapes). Les échantillons VAR échantillonnent de manière itérative dans l'échelle  il prédit un jeton 1x1, puis 2x2, puis 4x4, jusqu'à la résolution finale, chaque échelle conditionnant sur la précédente.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## Le problème

La génération autorégressive domine la modélisation du langage parce qu'elle évolue de manière prévisible: plus de calcul, plus de paramètres, moins de perplexité, de meilleures sorties. La génération d'images a eu deux tentatives principales de RA avant 2024: PixelRNN/PixelCNN (pixel-par-pixel) et DALL-E 1 / Parti / MuseGAN (token-par-token sur les codes VQ-VAE).

Les pixels et les jetons sont disposés dans une grille 2D, mais le modèle AR doit les visiter dans un ordre raster 1D. Un pixel de coin précoce n'a aucune idée de ce que l'image devient finalement.

VAR résout le problème de l'ordre de génération en modifiant ce qui est généré. Au lieu de prédire les jetons d'image un par un dans l'espace, VAR prédit une image entière à des résolutions croissantes. Étape 1: prédire un jeton 1x1 (l'image globale " résumé "). Étape 2: prédire une grille de jetons 2x2 (caractéries plus grossières). Étape 3: prédire une grille 4x4. Étape K: prédire la grille finale (H/8) x ((W/8)).

Chaque échelle suit toutes les échelles précédentes (causalement dans l'ordre de l'échelle) et parallèle dans sa propre échelle.

## Le concept

### Le marqueur à échelle multiple VQ-VAE

Le VAR a besoin d' une**multi-scale discrete tokenizer**Pour une image .$x$, il produit une séquence de grilles de jetons de résolution progressivement plus élevée:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Chacun d' eux .$z_k$Il utilise le même codebook (taille typique 4096-16384). La tokenization à chaque échelle n'est pas indépendante  elle est formée de sorte que la somme des résidus à chaque échelle se reconstruit $f$- Le numéro de la liste:

```math
f \approx \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_1), \text{target\_size}) + \cdots + \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_K), \text{target\_size})
```

C' est une**residual VQ**Variante.$k$capture quelles écailles $1 \ldots k-1$Le décodeur prend la somme de toutes les emblèmes de l'échelle et produit l'image.

Le tokenizer VQ à grande échelle est entraîné une fois (comme VQGAN) puis congelé.

### Prédition à l'échelle suivante

Le modèle génératif est un transformateur qui voit les jetons de toutes les échelles précédentes et prédit les jetons à l'échelle suivante.

Structure de séquence d'entrée:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Les emplacements de position codent à la fois l'indice d'échelle et la position spatiale dans l'échelle.$k$, position $(i, j)$peut assister à tous les jetons à l' échelle $1 \ldots k$et aux jetons à l' échelle $k$La VAR utilise une attention positionnelle fixe sans causalité intra-échelle  toutes les positions dans une échelle sont prédites en parallèle.

Perte de formation: à chaque échelle $k$, prédire les jetons $z_k$La même structure que la GPT, sauf la "sequence", est maintenant structurée à l'échelle.

### Génération

Pour la conclusion:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Pour $K = 10$Les échelles, génération est 10 transformateurs passes en avant. Chaque pass produit son échelle entière en parallèle  pas autorégression par jeton dans une échelle. Pour une image 256x256 ceci est environ 10 passes par rapport à 28-50 de DiT.

### Pourquoi la prochaine édition gagne la prochaine édition

Trois victoires structurelles:
1. **Coarse-to-fine aligns with natural image statistics.**La perception visuelle humaine et les ensembles de données d'images présentent à la fois des régularités dépendantes de l'échelle: la structure à basse fréquence est stable et prévisible; les détails à haute fréquence sont conditionnés par le contenu à basse fréquence.
2. **Parallel generation within scale.**Contrairement au GPT-style token AR, VAR produit tous les tokens à l'échelle en une seule étape.
3. **No generation order bias.**Les jetons à l'échelle k voient tous l'échelle k-1; il n'y a pas de biais "de gauche" ou "de dessus" qui oblige les jetons précoces à s'engager avant que le contexte tardif ne soit disponible.

### Loi de l'échelle

Tian et al. Il a démontré que le VAR suit une courbe d'échelle de la loi de pouvoir pour la FID sur ImageNet  tout comme le GPT le fait pour la perplexité. Le doublage des paramètres ou du calcul réduit de manière fiable l'erreur de moitié. C'était le premier modèle générateur d'images à exprimer ce type de comportement d'échelle aussi nettement que les modèles de langage. Le résultat est que les prédictions à l'échelle VAR deviennent prévisibles à partir de calcul, et non des conjectures empiriques par architecture.

### La relation avec la diffusion

La VAR et la diffusion partagent la même histoire de compression de données: les deux décomposent le problème de génération en une séquence de sous-problèmes plus faciles.

- Diffusion: ajoutez progressivement du bruit, apprenez à annuler une étape.
- VAR: ajoutez progressivement la résolution, apprenez à prédire la prochaine échelle.

Les deux génèrent des distributions conditionnelles traitables. En empirie, le VAR est plus rapide à l'inférence (moins de passes, toutes parallèles dans une échelle) et correspond ou bat le DiT sur l'ImageNet classé.

```figure
gx-var-next-scale
```

## Faites-le

Dans `code/main.py`vous allez:
1. Faites une petite .**multi-scale VQ tokenizer**sur des données "image" synthétiques (2 anneaux gaussiens en D).
2. - Le train a**VAR-style transformer**pour prédire les jetons à l'échelle suivante.
3. Prenez l'échantillon en appelant le transformateur 4 fois (4 échelles) et décodant.
4. Vérifiez que la formation organisée à l'échelle rend la génération parallèle à l'échelle.

Il s'agit d'une mise en œuvre de jouets. Le but est de voir le masque d'attention structuré à l'échelle et la génération parallèle à l'échelle fonctionner réellement.

## La faire partir

Cette leçon produit `outputs/skill-var-tokenizer-designer.md` une compétence pour concevoir un tokenizer à plusieurs échelles: nombre d'échelles, ratios d'échelle, taille de codebook, partage résiduel, architecture de décodeur.

## Exercices

1. **Scale count ablation.**Traînez VAR avec des échelles 4, 6, 8, 10. Mesurez la qualité de la reconstruction par rapport au nombre de passes autorégressives. Plus de échelles = plus de résidus fins = meilleure qualité mais plus de passes.

2. **Codebook size.**Les codes de train avec des codes de taille 512, 4096, 16384 offrent une meilleure reconstruction mais une prédiction plus difficile.

3. **Parallel-within-scale check.**Pour un VAR formé, mesurez explicitement le modèle d'attention. Dans l'échelle k, le modèle prend-il en charge des positions à l'échelle transversale mais pas à l'échelle interne?

4. **VAR vs DiT scaling.**Pour la même tâche de classe ImageNet, entraînez VAR et DiT à des budgets paramétriques correspondants (par exemple, 33M, 130M, 458M).

5. **Text conditioning.**Extension de VAR pour prendre une intégration de texte (CLIP pooled) comme entrée de conditionnement supplémentaire via adaLN. C'est la recette HART.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | Image generation by next-scale prediction over a pyramid of VQ token grids |
| Next-scale prediction | "Predict coarser, then finer" | The model predicts tokens at increasing resolution scales, conditioning on all previous scales |
| Multi-scale VQ tokenizer | "Residual VQ" | VQ-VAE that produces K token grids of increasing resolution, with decoder summing all scales |
| Scale k | "Pyramid level k" | One of K resolution levels, from 1x1 at k=1 up to (H/p)x(W/p) at k=K |
| Parallel-within-scale | "One forward per scale" | All tokens at scale k are predicted in one transformer pass, not autoregressively |
| Causal-across-scales | "Scale-ordered attention" | Token at scale $k$ can attend to all of scales $1 \ldots k$ but not scales $k+1 \ldots K$ |
| Residual VQ | "Additive tokenization" | Each scale's tokens encode the residual left by lower scales; decoder sums all scale embeddings |
| VAR scaling law | "Image GPT scaling" | FID follows a predictable power law in compute, like language models' perplexity |
| HART | "Hybrid VAR + text" | Text-conditional VAR variant combining MaskGIT-style iterative decoding with VAR's scale structure |
| Scale position embedding | "(scale, row, col) triple" | Positional encoding carries both the scale index and spatial coordinates within the scale |

## Pour en savoir plus

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) le document VAR, référence canonique
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) DiT, la ligne de base de comparaison de diffusion
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) VQGAN, le tokenizer de la famille VAR
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) VQ-VAE, fondement de la symbolisation discrète d'images
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) VAR en termes de texte
