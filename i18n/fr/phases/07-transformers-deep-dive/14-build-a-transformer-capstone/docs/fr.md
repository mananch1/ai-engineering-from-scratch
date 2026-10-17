# Construire un transformateur à partir de zéro  La pierre de taille

> 13 leçons, un modèle, pas de raccourcis.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## Le problème

Vous avez lu tous les articles, vous avez mis en place des points d'attention, des divisions de plusieurs têtes, des codes positionnels, des blocs d'encodeur et de décodeur, des pertes de BERT et de GPT, des MoE, du cache KV.

Le point culminant: entraîner un petit transformateur de décodeur uniquement de bout en bout sur une tâche de modélisation du langage au niveau des personnages. Il lit Shakespeare. Il génère un nouveau Shakespeare. Il est assez petit pour s'entraîner sur un ordinateur portable en moins de 10 minutes. Il est assez correct que l'échange d'un ensemble de données plus grand et une formation plus longue vous donne un véritable LM.

Le tutoriel de Karpathy pour 2023 est la mise en œuvre de référence que chaque élève écrit au moins une fois. Nous le levons et le réorganisons autour de ce que nous avons couvert.

## Le concept

![Transformer-from-scratch block diagram](../assets/capstone.svg)

L'architecture, annotée:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Ce que nous expédions

- `GPTConfig` un seul endroit pour configurer tous les hyperparametres.
- `MultiHeadAttention` cause, en lots, avec une voie optionnelle de style Flash (PyTorch's `scaled_dot_product_attention`)
- `SwiGLUFFN` FFN moderne.
- `Block` pré-norme, attention enveloppée résiduelle + FFN.
- `GPT` intégrations, blocs empilés, tête LM, générer().
- Boucle d'entraînement avec AdamW, cosine LR, coupage de gradient.
- Un jeton de niveau Char sur le texte de Shakespeare.

### Ce que nous ne livrons pas

- RoPE  est mis en œuvre conceptuellement dans la leçon 04. Nous utilisons ici des embellissements positionnels appris pour la simplicité.
- Le cache KV pendant la génération  chaque étape de génération recompte l'attention sur le préfixe complet. Plus lent mais plus simple.
- Attention Flash  PyTorch 2.0+ envois automatiques si les entrées correspondent; nous utilisons `F.scaled_dot_product_attention`- Je suis désolé .
- Vous avez vu le MoE dans la leçon 11.

### Mesures cibles

Sur un ordinateur portable Mac M2, un four-couche, quatre têtes, d_model=128 GPT entraîné pour 2000 pas sur `tinyshakespeare.txt`- Le numéro de la liste:

- La perte d'entraînement converge de ~4,2 (random) à ~1,5 en environ 6 minutes.
- L'échantillon de production semble en forme de Shakespeare: des mots archaïques, des interruptions de lignes, des noms propres comme "ROMEO:" émergent.
- La perte de valeur (détenue à 10% du texte final) suit de près la perte de formation; aucune surcoche à cette taille/budget.

```figure
n5-block-stack
```

## Faites-le

Cette leçon utilise PyTorch.`torch`(La construction du processeur est bonne).`code/main.py`Le scénario est:

- Téléchargement `tinyshakespeare.txt`si elle manque (ou si elle lit une copie locale).
- Le jeton char au niveau octet.
- Le train/val est divisé à 90/10.
- Boucle d'entraînement avec bf16 autocast sur le matériel pris en charge.
- L'échantillonnage après l'entraînement est terminé.

### Étape 1: données

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 caractères uniques, un petit vocabulaire, une taille de 4 bytes, pas de BPE, pas de drame de tokenizer.

### Étape 2: modèle

Regardez !`code/main.py`Le bloc est un manuel de cours de la leçon 05  pré-norme, RMSNorm, SwiGLU, MHA causal.

### Étape 3: boucle d'entraînement

Il y a un lot aléatoire de vitres de 256 longues, en avant, en entropie croisée, en arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en marche arrière, en arrière, en arrière, en arrière, en arrière, en arrière, en arrière, en arrière, en arrière, en arrière, en arrière, en arrière, en arrière, en arrière, en arrière.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Étape 4: échantillon

Si vous recevez une demande, faites-la à plusieurs reprises, prenez un échantillon des logits supérieurs, ajoutez-le et continuez. Arrêtez après 500 jetons.

### Étape 5: lire la sortie

Après 2 000 pas:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Pas Shakespeare, mais en forme de Shakespeare, une victoire claire pour 800 000 paramètres et 6 minutes sur un ordinateur portable.

## Utilisez-le

Cette pierre d'achèvement est une architecture de référence.

1. **Swap the tokenizer.**Utilisez le BPE (p. ex. `tiktoken.get_encoding("cl100k_base")`La taille du vocabulaire passe de 65 à 50 000 exemplaires.
2. **Train on a bigger corpus.**Utilisation `OpenWebText`ou `fineweb-edu`Les jetons 10B sur un seul A100 prennent environ 24 heures pour un GPT de 125M param.
3. **Add RoPE + KV cache + Flash Attention.**Les exercices ci-dessous vous guideront dans chacun d'eux.

Ce dernier se termine par un GPT de 125 M qui génère un anglais fluide. Pas un modèle frontalier. Mais le même chemin de code  juste plus grand  est ce que Karpathy, EleutherAI et l'Institut Allen utilisent pour former les points de contrôle de recherche en 2026.

## La faire partir

Regardez !`outputs/skill-transformer-review.md`. La compétence examine une mise en œuvre transformatrice à partir de zéro pour vérifier la précision des 13 leçons précédentes.

## Exercices

1. **Easy.**On court .`code/main.py`Vérifiez que la perte de validation de votre modèle formé est inférieure à 2.0.`max_steps`La perte de val est-elle toujours en amélioration ?
2. **Medium.**Remplacez les embellissements positionnels appris par RoPE.`MultiHeadAttention`La perte de valve est au moins aussi faible.
3. **Medium.**Implémenter un cache KV dans la boucle d'échantillonnage. Générer 500 jetons avec et sans cache. Le mur-horloge devrait s'améliorer de 520x sur un ordinateur portable.
4. **Hard.**Ajoutez une deuxième tête au modèle qui prédit le prochain jeton plus un (MTP  Multi-Token Prediction de DeepSeek-V3).
5. **Hard.**Remplacez le FFN unique par bloc par un MoE de 4 experts. Routeur + routage top-2. Voir comment la perte de val change aux paramètres actifs correspondants.

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; the canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial since 2015 uses it. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, keep optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; prevents training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw; overfit detector. |

## Pour en savoir plus

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) la mise en œuvre classique annotée.
