# Fondements de l'image  Pixels, canaux, espaces de couleur

> Une image est un tensor d'échantillons de lumière.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 Lesson 12 (Tensor Operations), Phase 3 Lesson 11 (Intro to PyTorch)
**Time:** ~45 minutes

## Objectifs d'apprentissage

- Expliquez comment une scène continue est discrétisée en pixels et pourquoi les décisions d'échantillonnage/quantification fixent le plafond sur chaque modèle en aval
- Lire, découper et inspecter les images en tant qu'encadrements NumPy et passer couramment entre les lignes HWC et CHW
- Convertir entre RGB, échelle de gris, HSV et YCbCr et justifier l'existence de chaque espace de couleur
- Appliquer le prétraitement au niveau des pixels (normalité, normalisation, redimensionnement, premier canal) exactement comme le s'y attendent les modèles de vision PyTorch prétraînés

## Le problème

Chaque article que vous lirez, chaque poids prétrainé que vous téléchargerez, chaque API de vision que vous appelez suppose un codage spécifique de l'entrée.`uint8`image où le modèle le souhaite `float32`et il continuera à fonctionner  et produira silencieusement des ordures. Donner BGR à un réseau formé sur RGB et la précision s'effondre de dix points. Donner un modèle de canaux - dernière entrée quand il attend des canaux - premier et la première couche de conv traite la hauteur comme un canal fonctionnel. Rien de tout cela ne jette une erreur. Il gâche simplement vos métriques et vous passez une semaine à chasser un bug qui vit dans la façon dont vous avez chargé le fichier.

Une convolutions ne sont pas compliquées une fois que vous savez ce qu'elle fait glisser. La partie difficile est qu'" une image " signifie différentes choses à une caméra, un décodeur JPEG, PIL, OpenCV, torchvision et un noyau CUDA. Chaque pile a son propre ordre d'axe, une gamme de octets et une convention de canaux. Un ingénieur de vision qui ne peut pas garder ces bateaux droits en rupture.

Cette leçon fixe les bases pour que le reste de la phase puisse s'y appuyer. À la fin, vous saurez ce qu'est un pixel, pourquoi il y a trois nombres par pixel au lieu d'un, ce que fait réellement "normalizer avec les statistiques d'ImageNet" et comment se déplacer entre les deux ou trois layouts que chaque autre leçon de cette phase prendra.

## Le concept

### Le pipeline de pré-traitement complet à un coup d'œil

Chaque système de vision de production est la même séquence de transformations réversibles.

```mermaid
flowchart LR
    A["Image file<br/>(JPEG/PNG)"] --> B["Decode<br/>uint8 HWC"]
    B --> C["Convert<br/>colorspace<br/>(RGB/BGR/YCbCr)"]
    C --> D["Resize<br/>shorter side"]
    D --> E["Center crop<br/>model size"]
    E --> F["Divide by 255<br/>float32 [0,1]"]
    F --> G["Subtract mean<br/>Divide by std"]
    G --> H["Transpose<br/>HWC → CHW"]
    H --> I["Batch<br/>CHW → NCHW"]
    I --> J["Model"]

    style A fill:#fef3c7,stroke:#d97706
    style J fill:#ddd6fe,stroke:#7c3aed
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#bfdbfe,stroke:#2563eb
```

Les deux boîtes rouges et bleues sont le lieu où 80% des défaillances silencieuses vivent: manque de normalisation et mauvaise mise en page.

### Un pixel est un échantillon, pas un carré

Un capteur de caméra compte les photons qui atterrissent sur une grille de détecteurs minuscules. Chaque détecteur intègre la lumière pendant une fraction de seconde et émet une tension proportionnelle au nombre de photons qui l'atteignent. Le capteur discrète ensuite cette tension en un nombre entier. Un détecteur devient un pixel.

```
Continuous scene                 Sensor grid                     Digital image
(infinite detail)                (H x W detectors)               (H x W integers)

    ~~~~~                        +--+--+--+--+--+                 210 198 180 155 120
   Je suis un homme qui a une grande expérience.
  - L'équipe de la police de la ville de New York
   ~~~~~                         |  |  |  |  |  |                 195 185 170 148 112
                                 +--+--+--+--+--+                 188 180 165 145 108
```

Deux choix se font à cette étape et ils fixent le plafond sur tout en aval:

- **Spatial sampling**Il y a des détecteurs qui décident combien de détecteurs par degré de scène. trop peu, et les bords deviennent déchirés (aliasing). trop, et le stockage et le calcul explosent.
- **Intensity quantization**Les 8 bits donnent 256 niveaux et sont standard pour l'affichage. 10, 12, 16 bits donnent des gradients et des matières plus lisses pour l'imagerie médicale, HDR et les pipelines de capteurs bruts.

Un pixel n'est pas un carré coloré avec une surface. C'est une seule mesure. Lorsque vous redimensionnez ou tournez, vous reprenez le modèle de la grille de mesure.

### Pourquoi trois canaux ?

Un détecteur compte les photons sur tout le spectre visible  qui est à l'échelle de gris. Pour obtenir la couleur, le capteur recouvre la grille avec une mosaïque de filtres rouges, verts et bleus. Après démoisaïcation, chaque emplacement spatial a trois entiers: la réponse du détecteur filtré rouge, vert-filtré et bleu à proximité. Ces trois entiers sont le triple RGB d'un pixel.

```
One pixel in memory:

    (R, G, B) = (210, 140, 30)   <- reddish-orange

An H x W RGB image:

    shape (H, W, 3)     stored as   H rows of W pixels of 3 values
                                    each in [0, 255] for uint8
```

Les caméras de profondeur ajoutent un canal Z. Les satellites ajoutent des bandes infrarouges et ultraviolets. Les scanners médicaux ont souvent un canal (rayons X, TC) ou plusieurs (hyperspectraux). Le nombre de canaux est le dernier axe; les couches de convection apprennent à se mélanger à travers elle.

### Deux conventions de mise en page: HWC et CHW

Le même tensor, deux ordres, chaque bibliothèque en choisit un.

```
HWC (height, width, channels)           CHW (channels, height, width)

   W ->                                    H ->
  +-----+-----+-----+                     +-----+-----+
H |R G B|R G B|R G B|                   C |R R R R R R|
| +-----+-----+-----+                   | +-----+-----+
v |R G B|R G B|R G B|                   v |G G G G G G|
  +-----+-----+-----+                     +-----+-----+
                                          |B B B B B B|
                                          +-----+-----+

   PIL, OpenCV, matplotlib,              PyTorch, most deep learning
   almost every image file on disk       frameworks, cuDNN kernels
```

CHW existe parce que les noyaux de convolutions glissent à travers H et W. Garder l'axe du canal en premier signifie que chaque noyau voit un plan 2D contigu par canal, qui vectorize de manière propre.

La conversion en une ligne, vous taperiez mille fois:

```
img_chw = img_hwc.transpose(2, 0, 1)      # NumPy
img_chw = img_hwc.permute(2, 0, 1)        # PyTorch tensor
```

L'affichage de la mémoire, visualisé:

```mermaid
flowchart TB
    subgraph HWC["HWC — pixels stored interleaved (PIL, OpenCV, JPEG)"]
        H1["row 0: R G B | R G B | R G B ..."]
        H2["row 1: R G B | R G B | R G B ..."]
        H3["row 2: R G B | R G B | R G B ..."]
    end
    subgraph CHW["CHW — channels stored as stacked planes (PyTorch, cuDNN)"]
        C1["plane R: entire H x W of red values"]
        C2["plane G: entire H x W of green values"]
        C3["plane B: entire H x W of blue values"]
    end
    HWC -->|"transpose(2, 0, 1)"| CHW
    CHW -->|"transpose(1, 2, 0)"| HWC
```

### Département d'accès

Trois assemblées dominent:

| Convention | dtype | Range | Where you see it |
|------------|-------|-------|------------------|
| Raw | `uint8` | [0, 255] | Files on disk, PIL, OpenCV output |
| Normalized | `float32` | [0.0, 1.0] | After `img.astype('float32') / 255` |
| Standardized | `float32` | roughly [-2, +2] | After subtracting mean and dividing by std |

Les réseaux convolutifs ont été formés sur des entrées standardisées.`mean=[0.485, 0.456, 0.406]`- Je suis là .`std=[0.229, 0.224, 0.225]`sont la moyenne arithmétique et l'écart standard des trois canaux sur l'ensemble complet de formation ImageNet, calculé sur [0, 1] pixels normalisés.`uint8`Dans un modèle qui s'attend à une flotte standardisée, la défaillance silencieuse la plus commune dans la vision appliquée est la seule.

### Les espaces de couleur et leur existence

RGB est le format de capture mais ce n'est pas toujours la représentation la plus utile pour un modèle.

```
 RGB               HSV                       YCbCr / YUV

 R red             H hue (angle 0-360)       Y luminance (brightness)
 G green           S saturation (0-1)        Cb chroma blue-yellow
 B blue            V value/brightness (0-1)  Cr chroma red-green

 Linear to         Separates color from      Separates brightness from
 sensor output     brightness. Useful for    color. JPEG and most video
                   color thresholding, UI    codecs compress the chroma
                   sliders, simple filters   channels harder because the
                                             human eye is less sensitive
                                             to chroma detail than to Y.
```

Pour la plupart des CNN modernes, vous nourrissez RGB.

- **HSV** code de CV classique, segmentation basée sur la couleur, équilibrage des blancs.
- **YCbCr** lecture des internes JPEG, des pipelines vidéo, des modèles de super résolution qui fonctionnent uniquement sur Y.
- **Grayscale** OCR, modèles de documents, tout cas où la couleur est variable de nuisance plutôt que de signal.

L'échelle de gris de la RGB est une somme pondérée, pas une moyenne, car l'œil humain est plus sensible au vert que au rouge ou au bleu:

```math
Y = 0.299 R + 0.587 G + 0.114 B \qquad \text{(ITU-R BT.601, the classic weights)}
```

### Ratio d'aspect, redimensionnement et interpolation

Chaque modèle a une taille d'entrée fixe (224x224 pour la plupart des classifiateurs ImageNet, 384x384 ou 512x512 pour les détecteurs modernes). Vos images correspondent rarement.

- **Resize shorter side, then center crop** la recette standard d'ImageNet. préserve le rapport d'aspect, jette une bande de pixels de bord.
- **Resize and pad** préserve le rapport d'aspect et chaque pixel, ajoute des barres noires.
- **Resize directly to target**Il est bon pour de nombreuses tâches de classification.

La méthode d'interpolation détermine comment les pixels intermédiaires sont calculés lorsque la nouvelle grille ne s'aligne pas avec l'ancienne:

```
Nearest neighbour     fastest, blocky, only choice for masks/labels
Bilinear              fast, smooth, default for most image resizing
Bicubic               slower, sharper on upscaling
Lanczos               slowest, best quality, used for final display
```

Règle générale: bilinéaire pour l'entraînement, bicubique ou lanczos pour les actifs que vous regarderez, le plus proche pour tout ce qui contient des identifiants de classe entière.

```figure
conv-output-size
```

## Faites-le

### Étape 1: Construisez un tensor d'image et inspectez sa forme

Commencez par une image synthétique déterministe afin que le premier laboratoire se déroule hors ligne avec uniquement NumPy. Le décodeur de fichier est une limite distincte: une fois qu'un décodeur JPEG ou PNG renvoie des octets RGB, chaque opération tensorielle ci-dessous est la même.

```python
import numpy as np

def synthetic_rgb(h=128, w=192, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    r = (np.sin(xx * 6) * 0.5 + 0.5) * 255
    g = yy * 255
    b = (1 - yy) * xx * 255
    rgb = np.stack([r, g, b], axis=-1) + rng.normal(0, 6, (h, w, 3))
    return np.clip(rgb, 0, 255).astype(np.uint8)

arr = synthetic_rgb()

print(f"type:   {type(arr).__name__}")
print(f"dtype:  {arr.dtype}")
print(f"shape:  {arr.shape}     # (H, W, C)")
print(f"min:    {arr.min()}")
print(f"max:    {arr.max()}")
print(f"pixel at (0, 0): {arr[0, 0]}")
```

Résultats attendus: `shape: (H, W, 3)`- Je suis là .`dtype: uint8`, la portée `[0, 255]`C'est la représentation décodée canonique que les octets provenaient d'une caméra, d'un décodeur d'image ou d'un générateur synthétique.

### Étape 2: Divisez les canaux et réorganisez

Retirez R, G, B séparément, puis convertissez de HWC en CHW pour PyTorch.

```python
R = arr[:, :, 0]
G = arr[:, :, 1]
B = arr[:, :, 2]
print(f"R shape: {R.shape}, mean: {R.mean():.1f}")
print(f"G shape: {G.shape}, mean: {G.mean():.1f}")
print(f"B shape: {B.shape}, mean: {B.mean():.1f}")

arr_chw = arr.transpose(2, 0, 1)
print(f"\nHWC shape: {arr.shape}")
print(f"CHW shape: {arr_chw.shape}")
```

Trois plans à l'échelle grise, un par canal. CHW réordonne simplement les axes; aucune copie de données n'est strictement requise lorsque la mise en page de la mémoire le permet.

### Étape 3: Conversion en grise et HSV

L'échelle de grises pondérée, puis un manuel RGB-HSV.

```python
def rgb_to_grayscale(rgb):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb.astype(np.float32) @ weights).astype(np.uint8)

def rgb_to_hsv(rgb):
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.max(rgb_f, axis=-1)
    cmin = np.min(rgb_f, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 0
    argmax = np.argmax(rgb_f, axis=-1)
    rmax = mask & (argmax == 0)
    gmax = mask & (argmax == 1)
    bmax = mask & (argmax == 2)
    h[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6
    h[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2
    h[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4
    h = h * 60.0

    s = np.divide(delta, cmax, out=np.zeros_like(delta), where=cmax > 0)
    v = cmax
    return np.stack([h, s, v], axis=-1)

gray = rgb_to_grayscale(arr)
hsv = rgb_to_hsv(arr)
print(f"gray shape: {gray.shape}, range: [{gray.min()}, {gray.max()}]")
print(f"hsv   shape: {hsv.shape}")
print(f"hue range: [{hsv[..., 0].min():.1f}, {hsv[..., 0].max():.1f}] degrees")
print(f"sat range: [{hsv[..., 1].min():.2f}, {hsv[..., 1].max():.2f}]")
print(f"val range: [{hsv[..., 2].min():.2f}, {hsv[..., 2].max():.2f}]")
```

Hue est exprimé en degrés, saturation et valeur en [0, 1].`hsv_full`La convention.

### Étape 4: normaliser, normaliser et inverser

Passez des octets bruts au tensor exact qu'un modèle ImageNet prétrainé attend, puis retournez.

```python
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_imagenet(rgb_uint8):
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)
    return x

def deprocess_imagenet(chw_float32):
    x = chw_float32.transpose(1, 2, 0)
    x = x * std + mean
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x

x = preprocess_imagenet(arr)
print(f"preprocessed shape: {x.shape}     # (C, H, W)")
print(f"preprocessed dtype: {x.dtype}")
print(f"preprocessed mean per channel:  {x.mean(axis=(1, 2)).round(3)}")
print(f"preprocessed std  per channel:  {x.std(axis=(1, 2)).round(3)}")

roundtrip = deprocess_imagenet(x)
max_diff = np.abs(roundtrip.astype(int) - arr.astype(int)).max()
print(f"roundtrip max pixel diff: {max_diff}    # should be 0 or 1")
```

La moyenne par canal doit être proche de zéro, std proche d'un.`transforms.Normalize`L'appel est sous le capot.

### Étape 5: Redimensionner à partir de zéro

Les coordonnées de sortie des voisins les plus proches sont à un pixel source. L'interpolation bilinéaire trouve les quatre pixels environnants et les mélange par distance.

```python
def resize_coordinates(source_length, target_length):
    if target_length == 1:
        return np.zeros(1, dtype=np.float32)
    return np.linspace(0, source_length - 1, target_length, dtype=np.float32)

def nearest_resize(image, target_height, target_width):
    y = np.rint(resize_coordinates(image.shape[0], target_height)).astype(int)
    x = np.rint(resize_coordinates(image.shape[1], target_width)).astype(int)
    return image[y[:, None], x[None, :]]

def bilinear_resize(image, target_height, target_width):
    y = resize_coordinates(image.shape[0], target_height)
    x = resize_coordinates(image.shape[1], target_width)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    wy = (y - y0)[:, None, None]
    wx = (x - x0)[None, :, None]

    source = image.astype(np.float32)
    top = source[y0[:, None], x0[None, :]] * (1 - wx)
    top += source[y0[:, None], x1[None, :]] * wx
    bottom = source[y1[:, None], x0[None, :]] * (1 - wx)
    bottom += source[y1[:, None], x1[None, :]] * wx
    result = top * (1 - wy) + bottom * wy
    return np.clip(np.rint(result), 0, 255).astype(image.dtype)

target_height = arr.shape[0] * 3
target_width = arr.shape[1] * 3
nearest = nearest_resize(arr, target_height, target_width)
bilinear = bilinear_resize(arr, target_height, target_width)

def local_roughness(x):
    gy = np.diff(x.astype(float), axis=0)
    gx = np.diff(x.astype(float), axis=1)
    return float(np.abs(gy).mean() + np.abs(gx).mean())

for name, out in [("nearest", nearest), ("bilinear", bilinear)]:
    print(f"{name:>8}  shape={out.shape}  roughness={local_roughness(out):6.2f}")
```

Le plus proche marque le plus haut sur la rugosité parce qu'il garde les bords durs. Bilinéaire est plus lisse parce que chaque nouveau pixel mélange deux positions sur chaque axe. Le compagnon exécutable étend la même idée séparable à quatre voisins par axe avec un noyau cube Catmull-Rom, puis imprime les trois résultats sans bibliothèque d'images.

## Utilisez-le

PyTorch effectue les mêmes opérations sur des tensors en lots, sensibles aux appareils. Le code ci-dessous redimensionne le côté plus court, prend une culture centrale, normalise chaque canal et produit le tensor NCHW qu'un modèle prétrainé attend.

```python
import torch
import torch.nn.functional as F

image_hwc = torch.from_numpy(synthetic_rgb(256, 320))
batch = image_hwc.permute(2, 0, 1).unsqueeze(0).float() / 255.0

height, width = batch.shape[-2:]
scale = 256 / min(height, width)
resized_height = round(height * scale)
resized_width = round(width * scale)
batch = F.interpolate(
    batch,
    size=(resized_height, resized_width),
    mode="bilinear",
    align_corners=False,
    antialias=True,
)

top = (resized_height - 224) // 2
left = (resized_width - 224) // 2
batch = batch[:, :, top:top + 224, left:left + 224]

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
batch = (batch - mean) / std

print(f"tensor dtype: {batch.dtype}")
print(f"batched shape: {tuple(batch.shape)}")
print(f"per-channel mean: {batch.mean(dim=(0, 2, 3)).tolist()}")
print(f"per-channel std:  {batch.std(dim=(0, 2, 3)).tolist()}")
```

Quatre étapes, dans cet ordre exact: convertir des octets en flottant et échanger HWC en NCHW, redimensionner le côté plus court à 256, prendre une culture centrale de 224x224, puis soustraire la moyenne d'ImageNet et diviser par son écart standard.

## La faire partir

Cette leçon donne:

- `outputs/prompt-vision-preprocessing-audit.md` une demande qui transforme toute carte modèle ou carte de jeu de données en une liste de contrôle des invariants pré-traitement exacts qu'une équipe doit respecter.
- `outputs/skill-image-tensor-inspector.md` une compétence qui, compte tenu de tout tensor ou tableau en forme d'image, rapporte dtype, disposition, plage, et si elle semble brune, normalisée ou standardisée.

## Exercices

1. **(Easy)**Créer un RGB 2x2 `uint8`Convertir HWC en CHW et retour, imprimer les deux formes, et prouver que le retour conserve toutes les valeurs.
2. **(Medium)**Écrivez`standardize(img, mean, std)`et son inverse qui ensemble passent un `roundtrip_max_diff <= 1`Vos fonctions doivent fonctionner sur une seule image dans HWC et sur un lot dans NCHW avec le même appel.
3. **(Hard)**Prenez un tensor standardisé à 3 canaux ImageNet et courez-le à travers un convex 1x1 qui apprend un mélange pondéré de RGB dans un seul canal à échelle de gris.`[0.299, 0.587, 0.114]`, les congeler, et vérifier la sortie correspond à votre manuel `rgb_to_grayscale`Quelles autres transformations classiques de l'espace couleur peuvent être écrites comme des convolutions 1x1 ?

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Pixel | "A coloured square" | One sample of light intensity at one grid location — three numbers for colour, one for grayscale |
| Channel | "The colour" | One of the parallel spatial grids stacked into an image tensor; last axis in HWC, first in CHW |
| HWC / CHW | "The shape" | Axis orderings for an image tensor; disk and PIL use HWC, PyTorch and cuDNN use CHW |
| Normalize | "Scale the image" | Divide by 255 so pixels live in [0, 1] — necessary but not sufficient |
| Standardize | "Zero-center" | Subtract mean and divide by std per channel so the input distribution matches what the model was trained on |
| Grayscale conversion | "Average the channels" | A weighted sum with coefficients 0.299/0.587/0.114 that matches human luminance perception |
| Interpolation | "How resize picks pixels" | The rule that decides output values when the new grid does not align with the old one — nearest for labels, bilinear for training, bicubic for display |
| Aspect ratio | "Width over height" | The ratio that distinguishes "resize and pad" from "resize and stretch" |

## Pour en savoir plus

- [Charles Poynton — A Guided Tour of Color Space](https://web.archive.org/web/20251220000525/https://poynton.ca/PDFs/Guided_tour.pdf) le traitement technique le plus clair de la raison pour laquelle il y a tant d'espaces de couleur et quand chacun d'eux compte
- [PyTorch Vision Transforms Docs](https://pytorch.org/vision/stable/transforms.html) l'ensemble des transformations que vous allez réellement composer en production
- [How JPEG Works (Colt McAnlis)](https://www.youtube.com/watch?v=F1kYBnY6mwg) une visite visuelle nette du sous-échantillonnage de chrome, DCT, et pourquoi JPEG encode YCbCr plutôt que RGB
- [ImageNet Preprocessing Conventions (torchvision models)](https://pytorch.org/vision/stable/models.html) la source de la vérité pour `mean=[0.485, 0.456, 0.406]`et pourquoi chaque modèle au zoo s' y attend
