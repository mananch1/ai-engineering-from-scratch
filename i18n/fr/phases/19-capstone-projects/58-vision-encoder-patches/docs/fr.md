# Les patchs de l'encodeur de vision

> Un modèle de vision qui lit des pixels a besoin d'un tokenizer pour les pixels. L'embedding de patch est ce tokenizer. Coupez l'image en une grille de carrés, aplaniez chaque carré, projettez-le à travers une couche linéaire, puis ajoutez un signal de position 2D afin que le transformateur sache où chaque carré était assis dans l'image originale.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 lessons 30-37 (Track B foundations)
**Time:** ~90 minutes

## Objectifs d'apprentissage

- Symbolez une image dans une séquence de longueur fixe d'embeddements de patch.
- La mise en œuvre d'une `Conv2d`- une projection de patch qui correspond aux mathématiques de déploiement-et-linéaire.
- Construire une position sinusoïdale 2D déterministe intégrant donc l'ordre symbolique encode la position spatiale.
- Vérifiez le nombre de patchs, la forme de l'intégration et `Conv2d`/déployer l'équivalence sur un appareil synthétique.

## Le problème

Un transformateur mange une séquence de vecteurs. Une image est une grille à trois canaux. Lire chaque pixel comme un jeton explose la longueur de la séquence: une image RGB de 224x224 est de 150.528 jetons, ce qu'un transformateur de 12 couches ne peut pas se permettre d'attirer l'attention. Lire l'image comme un vecteur plat géant jette la localisation, dont la couche d'attention ne peut pas se remettre. Le travail de l'emballage avant est de compresser la grille de pixels en quelques centaines de jetons qui résument chacun une région carrée.

L'intégration de patch résoud cela avec une projection linéaire. Une image de 224x224 coupée en 16x16 patches produit une grille de 14x14 de 196 patches. Chaque patch est aplatie à partir de`(3, 16, 16) = 768`Les valeurs de pixel dans un vecteur, puis une couche linéaire le cartographient à la dimension cachée du modèle.`hidden`C'est une séquence que le reste du réseau peut suivre.

## Le concept

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### Pourquoi des patchs, pas des pixels

L'attention est quadratique en longueur de séquence.`196 * 196 = 38,416`points d'attention par tête par couche; une séquence de 150.528 jetons coûte `150,528 * 150,528 = 22.6 billion`. Les patches achètent une réduction de 590.000x du calcul de l'attention, et une seule région 16x16 transporte suffisamment de signal pour des tâches de vision de haut niveau. Le coût est une perte de détails spatiaux fins à l'intérieur d'un patch, c'est pourquoi les piles multimodelles en aval utilisent souvent une deuxième branche haute résolution lorsque la localisation fine est importante.

### Pourquoi une projection linéaire est suffisante

Chaque patch est traité comme un vecteur indépendant. La projection apprend une base: détecteurs de bord, filtres de couleur, textures simples.`768 * 768 = 589,824`Les étiquettes convolutives plus profondes existent (la ViT "hybride"), mais une projection linéaire plate est la norme, et la plupart des encoders à poids ouvert modernes ont cette forme exacte.

### Le `Conv2d`un truc

Une .`Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)`sans rembourrage donne le même résultat numérique que déployer-et-linéaire, parce que chaque position de sortie produit les pixels de patch contre un filtre. La convolutions est la projection de patch, et la plupart des bases de code de production l'expédition de cette façon parce qu'il est plus rapide sur GPU et utilise un moins de remodelage.

### Embeddings de position

Les jetons ne sont pas en ordre dans la projection.`(row, col)`position. La moitié de la dimension d'embedding encode la position de la rangée avec sin/cos à plusieurs fréquences; l'autre moitié encode la position de la colonne.

| Component | Shape | Parameters |
|-----------|-------|------------|
| Patch projection (`Conv2d`) | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| Position embedding (fixed) | `(num_patches, hidden)` | 0 (computed, not learned) |
| CLS token (learned) | `(1, hidden)` | `hidden` |

Pour ViT-Base/16 à 224 résolution: 590.592 paramètres dans la projection, 768 dans le jeton CLS, et zéro pour la position sinusoïdale.

### L'équivalence en tant que contrôle de la santé mentale

La étape de patch a deux orthographes: a `Conv2d`Les tests de cette leçon exercent cette équivalence.

```figure
ch-patch-tokenizer
```

## Faites-le

`code/main.py`les implémentations:

- `PatchEmbed`, une`nn.Module`enveloppement `Conv2d`pour la projection de patch.
- `sinusoidal_2d(grid_h, grid_w, dim)`, une fonction sans état qui construit la table de position 2D.
- `VisionFrontEnd`, qui compose l'intégration de patch, le prépendicule CLS et l'ajout de position dans un passage vers l'avant.
- Une .`synthesize_image(seed)`aide qui construit un fichier déterministe 224x224x3 à partir de `numpy.random`- Je suis désolé .
- Une démo qui exécute une image de fichier à travers la partie avant et imprime la forme de sortie, la norme du jeton CLS et une rangée de l'intégration de position.

- Je vais le faire.

```bash
python3 code/main.py
```

Sortie: le fichier 224x224 est symbolisé par une séquence de formes `(1, 197, 768)`. Le premier jeton est le CLS; les 196 suivants sont des jetons de patch.

## Utilisez-le

Le même patch avant apparaît dans tous les modèles modernes de langage de vision: CLIP ViT-L/14, SigLIP, DINOv2, la famille Qwen-VL et la pile InternVL commencent tous à partir d'une`Conv2d`Les différences entre les familles vivent en aval (CLS vs no-CLS pooling, jetons de registre, tailles de patch variées 14 vs 16, résolution dynamique via des positions interpolées).

## Tests

`code/test_main.py`couvertures:

- le nombre de patches correspondant `(image_size / patch_size) ** 2`
- correspondant à la forme de sortie `(batch, num_patches + 1, hidden)`
- le `Conv2d`projection égale à déployer manuellement-et-linéaire sur un petit appareil
- la table de position sinusoïdale est déterministe sur les appels
- Transmissions de jetons CLS à travers des lots déformés sans fuite

- Je vais les faire.

```bash
python3 -m unittest code/test_main.py
```

## Exercices

1. Remplacez la position sinusoïdale par une position de l' élève.`nn.Parameter`Les positions apprises gagnent à résolution fixe, les positions sinusoïdes gagnent quand on change de résolution après l'entraînement.

2. Échangez le `Conv2d`pour une explicite`nn.Unfold`plus `nn.Linear`et affirmer que les sorties correspondent à la tolérance à flot.

3. Ajouter un support pour les tailles de patch non carrés (par exemple 32x16 pour les entrées à large aspect) et vérifier que la table de position gère les grilles non carrés.

4. Profiliser le pas de patch aux tailles de lot 1, 8, 64. La projection du patch est rarement le goulot d'étranglement; les couches d'attention en aval dominent.

5. Exercer la partie avant comme extracteur de fonctionnalités gelées sur un ensemble de données de forme synthétique de 4 classes (cercles, carrés, triangles, étoiles).

## Les termes clés

| Term | What it means |
|------|---------------|
| Patch | A square sub-region of the image, typically 14x14 or 16x16 |
| Patch embedding | Linear projection of one flattened patch to the hidden dim |
| Sequence length | Number of tokens after patch tokenization, usually plus CLS |
| Sinusoidal position | Fixed sin/cos signal that encodes 2D grid coordinates |
| CLS token | Learned vector prepended to the sequence as the pooling head |

## Pour en savoir plus

- Une image vaut 16x16 mots (ViT, 2021) pour le cadre original intégré à patch.
- Attention est tout ce dont vous avez besoin (2017) pour la formule de position sinusoïdale adaptée ici à la 2D.
- Le papier DINOv2 pour les jetons de registre, une extension que vous pouvez ajouter comme exercice 6.
