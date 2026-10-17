# Génération d'images  GAN

> Un GAN est deux réseaux neuronaux dans un jeu fixe, un tirage au sort, un critique, ils s'améliorent ensemble jusqu'à ce que les dessins trompent le critique.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 3 Lesson 06 (Optimizers), Phase 3 Lesson 07 (Regularization)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Expliquez le jeu minimax entre générateur et discriminateur et pourquoi l'équilibre correspond à $p_{\text{model}} = p_{\text{data}}$
- Implémenter un DCGAN en PyTorch et le faire générer des images synthétiques cohérentes 32x32 en moins de 60 lignes
- Stabiliser la formation GAN avec les trois astuces standard: perte non saturante, norme spectrale, TTUR (règle de mise à jour à deux échelles)
- Lire les courbes d'entraînement qui distinguent la convergence saine de mode effondrement, oscillation, et discriminateur-gains-completement

## Le problème

La classification enseigne à un réseau à cartographier des images sur des étiquettes. La génération inverse le problème: échantillonnez de nouvelles images qui semblent provenir de la même distribution. Il n'y a pas de sortie "correcte" à laquelle vous pouvez vous opposer; il n'y a qu'une distribution que vous voulez imiter.

Les fonctions de perte standard (MSE, entropie croisée) ne peuvent pas mesurer " ce modèle provient-il de la réelle distribution. " La réduction de l'erreur par pixel produit des moyennes floues, pas des échantillons réalistes.

Les GAN (Goodfellow et coll., 2014) ont défini ce cadre. En 2018, StyleGAN produisait 1024x1024 visages indistinguibles des photographies.

## Le concept

### Les deux réseaux

```mermaid
flowchart LR
    Z["z ~ N(0, I)<br/>noise"] --> G["Generator<br/>transposed convs"]
    G --> FAKE["Fake image"]
    REAL["Real image"] --> D["Discriminator<br/>conv classifier"]
    FAKE --> D
    D --> OUT["P(real)"]

    style G fill:#dbeafe,stroke:#2563eb
    style D fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

Le **generator**G prend un vecteur de bruit `z`et produit une image.**discriminator**D prend une image et produit un seul échelle: la probabilité que l'image soit réelle.

### Le jeu

G veut que D ait tort, D veut avoir raison.

```math
\min_G \max_D \; \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))]
```

Lire de droite à gauche: D maximiser la précision sur le réel (`log D(real)`) et faux (`log (1 - D(fake))`G réduit au minimum l'exactitude de D sur les faux  il veut `D(G(z))`Pour être high.

Goodfellow a prouvé que ce minimum a un équilibre global où `p_G = p_data`, D donne des sorties de 0,5 partout, et la divergence Jensen-Shannon entre les distributions générées et réelles est de zéro.

### Perte non saturante

La forme ci-dessus est numériquement instable.`D(G(z))`est proche de zéro pour chaque faux, donc `log(1 - D(G(z)))`Il y a des gradients qui disparaissent par rapport à G. La solution: la perte de G.

```math
\begin{aligned}
L_D &= -\mathbb{E}_x[\log D(x)] - \mathbb{E}_z[\log(1 - D(G(z)))] \\
L_G &= -\mathbb{E}_z[\log D(G(z))] \qquad \text{(non-saturating)}
\end{aligned}
```

Maintenant quand ?`D(G(z))`C'est un train de GAN moderne avec cette variante.

### Règles d'architecture DCGAN

Radford, Metz, Chintala (2015) ont distillé des années d'expériences ratées en cinq règles qui rendent l'entraînement GAN stable:

1. Remplacez le regroupement par des convois à pas (les deux filets).
2. Utiliser la norme de lot dans le générateur et le discriminateur, à l'exception de la sortie de G et de l'entrée de D.
3. Retirez les couches entièrement connectées sur des architectures plus profondes.
4. G utilise la LIR sur toutes les couches à l'exception de la sortie (tanh pour la sortie dans [-1, 1]).
5. D utilise LeakyReLU (negative_slope=0.2) sur toutes les couches.

Chaque GAN moderne basé sur des conve (StyleGAN, BigGAN, GigaGAN) commence toujours à partir de ces règles et remplace les pièces une à la fois.

### Les modes d'échec et leurs signatures

```mermaid
flowchart LR
    M1["Mode collapse<br/>G produces a narrow<br/>set of outputs"] --> S1["D loss low,<br/>G loss oscillating,<br/>sample variety drops"]
    M2["Vanishing gradients<br/>D wins completely"] --> S2["D accuracy ~100%,<br/>G loss huge and static"]
    M3["Oscillation<br/>G and D keep trading<br/>wins forever"] --> S3["Both losses swing<br/>wildly with no downward trend"]

    style M1 fill:#fecaca,stroke:#dc2626
    style M2 fill:#fecaca,stroke:#dc2626
    style M3 fill:#fecaca,stroke:#dc2626
```

- **Mode collapse**: G trouve une image qui trompe D et produit seulement cela.
- **Discriminator wins**Le D devient trop fort, les gradients de G disparaissent.
- **Oscillation**Le résultat de l'opération est le résultat de l'opération de résolution de la situation de l'économie de marché.

### Évaluation

Les GAN n'ont pas de vérité, alors comment savez-vous qu'ils fonctionnent ?

- **Sample inspection** Regardez seulement 64 échantillons à la fin de chaque époque.
- **FID (Fréchet Inception Distance)** distance entre les répartitions de fonctionnalités d'Inception-v3 des ensembles réels et générés.
- **Inception Score** plus âgée, plus fragile; préférer la FID.
- **Precision/Recall for generative models** mesure séparément la qualité (precision) et la couverture (reprise).

Pour une petite analyse de données synthétiques, l'inspection par échantillon suffit.

```figure
cv-gan-image
```

## Faites-le

### Étape 1: Générateur

Un petit générateur DCGAN qui prend le bruit de 64 dimensions et produit une image 32x32.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))
```

Quatre convois transposés, chacun avec `kernel_size=4, stride=2, padding=1`Ils doublent la taille de l'espace.

### Étape 2: Discriminateur

Le LeakyReLU, les convois à pas, se termine par une logique scalaire.

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 2, feat * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x).view(-1)
```

La dernière conve réduit un `4x4`carte de caractéristiques à `1x1`. La sortie est un seul scalaire par image; appliquer sigmoid uniquement lors du calcul des pertes.

### Étape 3: étape de formation

Alternative: mettre à jour D une fois, puis G une fois, chaque lot.

```python
import torch.nn.functional as F

def train_step(G, D, real, z, opt_g, opt_d, device):
    real = real.to(device)
    bs = real.size(0)

    # D step
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
              + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
    loss_d.backward()
    opt_d.step()

    # G step
    opt_g.zero_grad()
    d_fake = D(G(z))
    loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()
```

`G(z).detach()`Dans la phase D, il est essentiel de ne pas laisser les gradients s'écouler dans la phase G pendant la mise à jour.

### Étape 4: cycle d'entraînement complet sur les formes synthétiques

```python
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def synthetic_images(num=2000, size=32, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((num, 3, size, size), dtype=np.float32) - 1.0
    for i in range(num):
        r = rng.uniform(6, 12)
        cx, cy = rng.uniform(r, size - r, size=2)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.5, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = synthetic_images()
loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

G = Generator(z_dim=64, img_channels=3, feat=32).to(device)
D = Discriminator(img_channels=3, feat=32).to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(10):
    for (batch,) in loader:
        z = torch.randn(batch.size(0), 64, device=device)
        ld, lg = train_step(G, D, batch, z, opt_g, opt_d, device)
    print(f"epoch {epoch}  D {ld:.3f}  G {lg:.3f}")
```

`Adam(lr=2e-4, betas=(0.5, 0.999))`est la défaillance DCGAN  la faible bêta1 empêche le temps d'élan de stabiliser trop le jeu adversaire.

### Étape 5: Prise d'échantillons

```python
@torch.no_grad()
def sample(G, n=16, z_dim=64, device="cpu"):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1)
```

Pour DCGAN, cela compte parce que les statistiques d'exécution de la norme de lot sont utilisées au lieu des statistiques du lot.

### Étape 6: Normalisation spectrale

Un remplacement de BN dans le discriminateur qui garantit le réseau est 1-Lipschitz.

```python
from torch.nn.utils import spectral_norm

def build_sn_discriminator(img_channels=3, feat=64):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(img_channels, feat, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat, feat * 2, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 2, feat * 4, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 4, 1, 4, 1, 0)),
    )
```

Échange `Discriminator`pour `build_sn_discriminator()`La norme spectrale est la mise à niveau de robustesse la plus simple que vous puissiez appliquer.

## Utilisez-le

Pour la génération sérieuse, utilisez des poids prétraînés ou passez à la diffusion.

- `torch_fidelity`computes le FID / IS sur votre générateur sans écrire un code d'évaluation personnalisé.
- `pytorch-gan-zoo`(héritage) et `StudioGAN`les mises en œuvre testées par le navire de DCGAN, WGAN-GP, SN-GAN, StyleGAN et BigGAN.

En 2026, les GAN sont toujours le meilleur choix pour: génération d'images en temps réel (la latence <10 ms), transfert de style, traduction image-à-image avec contrôle précis (Pix2Pix, CycleGAN).

## La faire partir

Cette leçon donne:

- `outputs/prompt-gan-training-triage.md` une requête qui lit une description de la courbe d'entraînement et choisit le mode défaillance (effondrement du mode, D-wins, oscillation) plus la seule correction recommandée.
- `outputs/skill-dcgan-scaffold.md` une compétence qui écrit un échafaudage DCGAN à partir `z_dim`, cible `image_size`, et `num_channels`, y compris la boucle d'entraînement et le conservateur d'échantillons.

## Exercices

1. **(Easy)**En effet, les données de la série de cercles synthétiques sont généralement utilisées pour la sélection de données de cercles synthétiques.
2. **(Medium)**Remplacez la norme de lot du discriminateur par la norme spectrale. Exercez les deux versions côte à côte.
3. **(Hard)**Implémenter un DCGAN conditionnel: alimenter l'étiquette de classe dans G et D (concentrer un-chaud au bruit dans G, concasser un canal d'intégration de classe dans D).

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Generator (G) | "The draws-stuff net" | Maps noise to images; trained to fool the discriminator |
| Discriminator (D) | "The critic" | Binary classifier; trained to distinguish real from generated images |
| Minimax | "The game" | min over G, max over D of an adversarial loss; equilibrium is $p_G = p_{\text{data}}$ |
| Non-saturating loss | "The numerically sane version" | G's loss is $-\log(D(G(z)))$ instead of $\log(1 - D(G(z)))$ to avoid vanishing gradients early in training |
| Mode collapse | "Generator makes one thing" | G produces only a small subset of the data distribution; fix with SN, minibatch discrimination, or larger batch |
| TTUR | "Two learning rates" | D learns faster than G, typically by a factor of 2-4; stabilises training |
| Spectral norm | "1-Lipschitz layer" | A weight-normalisation that bounds each layer's Lipschitz constant; stops D from becoming arbitrarily steep |
| FID | "Fréchet Inception Distance" | Distance between Inception-v3 feature distributions of real and generated sets; the standard evaluation metric |

## Pour en savoir plus

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)Le journal qui a tout commencé
- [DCGAN (Radford, Metz, Chintala, 2015)](https://arxiv.org/abs/1511.06434) les règles d'architecture qui ont permis de former les GAN
- [Spectral Normalization for GANs (Miyato et al., 2018)](https://arxiv.org/abs/1802.05957) le plus utile des trucs de stabilisation
- [StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423)Le SOTA GAN, c'est un album de tous les tricks de la dernière décennie.
