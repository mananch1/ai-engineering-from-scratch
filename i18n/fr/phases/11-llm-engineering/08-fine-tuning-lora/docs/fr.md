# Ajustement avec LoRA et QLoRA

> La mise à jour complète d'un modèle 7B nécessite 56 Go de RAM. Vous n'avez pas cela. La plupart des entreprises ne le font pas non plus. LoRA vous permet de mettre à jour le même modèle en 6 Go en entraînant moins de 1% des paramètres. Ce n'est pas un compromis - il correspond à la qualité de mise à jour complète sur la plupart des tâches. L'ensemble de l'écosystème de mise à jour open source fonctionne sur cette astuce.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 06 (Instruction Tuning / SFT)
**Time:** ~75 minutes
**Related:**La phase 10 couvre les boucles SFT/DPO à partir de zéro. Cette leçon les branche dans les kits d'outils PEFT 2026 (PEFT, TRL, Unsloth, Axolotl, LLaMA-Factory).

## Objectifs d'apprentissage

- Implémentation de l'AER par injection de matrices adaptatrices de basse qualité (A et B) dans les couches d'attention d'un modèle prétrainé
- Calculer les économies de paramètres de LoRA par rapport à l' ajustement complet: rang $r$avec $d_{\text{model}}$dimensions des trains $2 \cdot r \cdot d$Paramètres au lieu de $d^2$
- Télégraphie d'un modèle à l'aide de QLoRA (4 bits de base quantifiée + adaptateurs LoRA) pour s'intégrer dans la mémoire GPU du consommateur
- Réunir les poids LoRA dans le modèle de base pour le déploiement et comparer la vitesse d'inférence avec et sans adaptateurs

## Le problème

Vous avez un modèle de base. Llama 3 8B. Vous voulez qu'il réponde aux billets de support client dans la voix de votre entreprise. SFT est la réponse. Mais SFT a un problème de coût.

Llama 3 8B a 8 milliards de paramètres. En fp16, chaque paramètre prend 2 octets. C'est 16 Go juste pour charger les poids. Pendant l'entraînement, vous avez également besoin de gradients (16 Go), d'états d'optimisation pour Adam (32 Go pour la dynamique + variance) et d'activations. Total: environ 56 Go de VRAM pour un seul modèle 8 B.

Un A100 de 80 Go peut à peine s'adapter à ça.$3-4/hour on cloud providers. Training for 3 epochs on 50,000 examples takes 6-10 hours. That's $30-40 par expérience, 10 expériences pour obtenir les hyperparametres correctement et vous avez dépensé 400 $ avant de déployer quoi que ce soit.

Si vous faites passer le nombre à Llama 3 70B, les chiffres deviennent absurdes. 140 Go pour les poids seuls.

Il y a aussi un problème plus profond. L'ajustement complet modifie chaque poids du modèle. Si vous ajustez les données de support client, vous pouvez dégrader les capacités générales du modèle. On appelle cela l'oubli catastrophique. Le modèle s'améliore dans votre tâche et pire dans tout le reste.

Vous avez besoin d'une méthode qui entraîne moins de paramètres, utilise moins de mémoire et ne détruit pas les connaissances existantes du modèle.

## Le concept

### L'adaptation à un niveau inférieur

Edward Hu et ses collègues de Microsoft ont publié LoRA en juin 2021. L'idée du papier: les mises à jour de poids lors de la mise à jour fine ont un rang intrinsèque faible. Vous n'avez pas besoin de mettre à jour tous les 16,7 millions de paramètres dans une matrice de poids 4096x4096. Les informations utiles dans la mise à jour peuvent être capturées par une matrice de rang 16 ou 32.

Voici les mathématiques. Une couche linéaire standard compute:

```math
y = Wx
```

Où ?$W$est une $d_{\text{out}} \times d_{\text{in}}$Pour une projection d'attention de 4096x4096, c'est 16 777 216 paramètres.

La LORA gèle .$W$et ajoute une décomposition de rang inférieur:

```math
y = Wx + BAx
```

Où ?$B$est $(d_{\text{out}} \times r)$et $A$est $(r \times d_{\text{in}})$- Le rang .$r$est beaucoup plus petit que $d$-- généralement 8, 16 ou 32.

Pour $r=16$sur une couche de 4096x4096:
- Paramètres originaux: 4096 x 4096 = 16 777 216
- Paramètres de l'AEM: (4096 x 16) + (16 x 4096) = 65 536 + 65 536 = 131 072
- Réduction: 131.072 / 16.777.216 = 0,78%

Vous entraînez 0,78% des paramètres et obtenez 95-100% de la qualité.

```mermaid
graph LR
    X["Input x"] --> W["Frozen W (d x d)"]
    X --> A["A (r x d)"]
    A --> B["B (d x r)"]
    W --> Plus["+ (merge)"]
    B --> Plus
    Plus --> Y["Output y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff
```

A est initialisé avec un gaussien aléatoire. B est initialisé à zéro. Cela signifie que la contribution LoRA commence à zéro - le modèle commence à s'entraîner à partir de son comportement initial et apprend progressivement l'adaptation.

### Le facteur d'échelle: Alpha

L' ARL introduit un facteur d' évolutions $\alpha$qui contrôle la quantité d'effet de la mise à jour de bas rang sur la sortie:

```math
y = Wx + \frac{\alpha}{r} \cdot BAx
```

Quand ?$\alpha = r$, l' évolue est 1x. Quand $\alpha = 2r$(le standard commun), l'échelle est de 2x. Cet hyperparamètre contrôle le taux d'apprentissage du parcours LoRA indépendamment du taux d'apprentissage de base.

Conseils pratiques:
- $\alpha = 2 \cdot \text{rank}$est une convention commune communautaire (le papier original utilisé $\alpha = \text{rank}$dans la plupart des expériences)
- $\alpha = \text{rank}$donne une échelle de 1x, conservatrice mais stable
- Plus haut $\alpha$signifie des mises à jour plus importantes par étape, ce qui peut accélérer la convergence ou provoquer une instabilité

### Où appliquer le LoRA

Un transformateur a de nombreuses couches linéaires. Vous n'avez pas besoin d'ajouter LoRA à toutes.

| Target Layers | Trainable Params (7B) | Quality |
|--------------|----------------------|---------|
| q_proj only | 4.7M | Good |
| q_proj + v_proj | 9.4M | Better |
| q_proj + k_proj + v_proj + o_proj | 18.9M | Best for attention |
| All linear (attention + MLP) | 37.7M | Marginal gain, 2x params |

Le point de départ pour la plupart des tâches: q_proj + v_proj. Cela cible la requête et les projections de valeur en auto-attention, qui contrôlent ce que le modèle attend et quelles informations il extrait.

### Sélection de rang

Le rang$r$contrôle l'expression de l'adaptation:

| Rank | Trainable Params (per layer) | Best For |
|------|---------------------------|----------|
| 4 | 32,768 | Simple classification, sentiment |
| 8 | 65,536 | Single-domain Q&A, summarization |
| 16 | 131,072 | Multi-domain tasks, instruction following |
| 32 | 262,144 | Complex reasoning, code generation |
| 64 | 524,288 | Diminishing returns for most tasks |
| 128 | 1,048,576 | Rarely justified |

Hu et al. ont montré que $r=4$Il est déjà possible de réaliser une adaptation à des tâches simples. $r=8$et $r=16$Les choix les plus courants sont en pratique.$r=64$Il améliore rarement la qualité et commence à perdre l'avantage de la mémoire de LoRA.

### QLoRA: Quantification à 4 bits + LoRA

Tim Dettmers et ses collègues de l'Université de Washington ont publié QLoRA en mai 2023.

Cela change considérablement l'équation de mémoire:

| Method | Weight Memory (7B) | Training Memory (7B) | GPU Required |
|--------|-------------------|---------------------|-------------|
| Full fine-tune (fp16) | 14GB | ~56GB | 1x A100 80GB |
| LoRA (fp16 base) | 14GB | ~18GB | 1x A100 40GB |
| QLoRA (4-bit base) | 3.5GB | ~6GB | 1x RTX 3090 24GB |

QLoRA apporte trois contributions techniques:

**NF4 (Normal Float 4-bit)**Le NF4 place ses 16 niveaux de quantification aux quantiles d'une distribution normale standard. C'est une information théoriquement optimale pour les données normalement distribuées. Il perd moins d'informations que la quantification uniforme à 4 bits (INT4) ou la float4 standard.

**Double quantization**Les constantes de quantification elles-mêmes prennent la mémoire. Chaque bloc de 64 poids a besoin d'un facteur d'échelle fp32 (4 octets). Pour un modèle 7B, c'est un supplément de 0,4 GB. La double quantification quantifie ces constantes à fp8, réduisant le coût général à 0,1 GB. Petit mais il s'additionne.

**Paged optimizers**Les optimistes de pages utilisent la mémoire unifiée de NVIDIA pour automatiquement rediriger les états d'optimisation vers la RAM du processeur lorsque la mémoire du GPU est épuisée et les rediriger au besoin. Cela empêche les pannes OOM au prix d'un certain débit.

### La question de la qualité

La réduction des paramètres ou la quantification de la base nuisent-elles à la qualité?

| Method | MMLU (5-shot) | MT-Bench | HumanEval |
|--------|--------------|----------|-----------|
| Full fine-tune (Llama 2 7B) | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16 (NF4) | 47.5 | 6.61 | 13.4 |
| QLoRA r=64 (NF4) | 48.1 | 6.70 | 14.2 |

- Je suis là .$r=16$Le taux de référence est inférieur à 1% de l'ajustement complet sur la plupart des indicateurs de référence.$r=16$Le taux de décomposition des données est de 0,5%$r=64$Il correspond essentiellement à un réglage complet tout en utilisant 90% de mémoire en moins.

### Coûts réels

Llama 3 8B à régler sur 50 000 exemplaires (3 époques):

| Method | GPU | Time | Cost |
|--------|-----|------|------|
| Full fine-tune | 2x A100 80GB | 8 hours | ~$32 |
| LoRA r=16 | 1x A100 40GB | 4 hours | ~$8 |
| QLoRA r=16 | 1x RTX 4090 24GB | 6 hours | ~$5 |
| QLoRA r=16 (Unsloth) | 1x RTX 4090 24GB | 2.5 hours | ~$2 |
| QLoRA r=16 | 1x T4 16GB | 12 hours | ~$4 |

C'est pourquoi la communauté de réglage des poids ouverts a explosé en 2023 et pourquoi chaque cadre de formation inférieur à celui-ci expédie QLoRA par défaut en 2026.

### La pile PEFT 2026

| Framework | What it is | Pick when |
|-----------|-----------|-----------|
| **Hugging Face PEFT** | The canonical LoRA/QLoRA/DoRA/IA3 library | You want raw control and your training loop is already on `transformers.Trainer` |
| **TRL** | HF's reinforcement-from-feedback trainers (SFT, DPO, GRPO, PPO, ORPO) | You need DPO/GRPO after SFT; built on top of PEFT |
| **Unsloth** | Triton-kernel rewrite of the forward/backward pass | You want 2-5x speedup + half the VRAM with no accuracy loss; Llama/Mistral/Qwen family |
| **Axolotl** | YAML-config wrapper over PEFT + TRL + DeepSpeed + Unsloth | You want reproducible, version-controlled training runs |
| **LLaMA-Factory** | GUI/CLI/API over PEFT + TRL | You want zero-code fine-tuning; 100+ model families supported |
| **torchtune** | Native PyTorch recipes, no `transformers` dep | You want minimal deps and your org already standardizes on PyTorch |

Règle générale: utilisation de la recherche ou expérimentation ponctuelle → PEFT. Pipeline de production répétée → Axolotl avec noyaux Unsloth activés. Prototypage jetable → LLaMA-Factory.

### Les adaptateurs de fusion

Après l'entraînement, vous avez deux choses: le modèle de base gelé et un petit adaptateur LoRA (généralement 10 à 100 Mo).

1. **Keep them separate**: Charger le modèle de base, charger l'adaptateur en haut. Swap adaptateurs pour différentes tâches. C'est ainsi que vous servez plusieurs variantes finement réglées d'un modèle de base.

2. **Merge them permanently**: calcul $W' = W + (\alpha/r) \cdot BA$Le modèle fusionné est de la même taille que l'original, sans déduction, sans adaptateur à gérer.

Pour effectuer plusieurs tâches (adaptateur de support client, adaptateur de code, adaptateur de traduction), gardez-les séparées.

Techniques de fusion avancées pour combiner plusieurs adaptateurs:

- **TIES-Merging**(Yadav et coll. 2023): Trims paramètres de petite taille, résolve les conflits de signalisation, puis fusionne. Réduit l'interférence entre les adaptateurs.
- **DARE**(Yu et coll. 2023): Il dépose aléatoirement les paramètres de l'adaptateur avant de fusionner et rééchelle le reste.
- **Task arithmetic**L'ajout d'un adaptateur "code" et d'un adaptateur "mathématique" produit souvent un modèle bon pour les deux.

### Quand ne pas régler

Le réglage est la troisième option, pas la première.

**First: prompt engineering.**Écrivez une meilleure mise à jour du système. Ajoutez quelques exemples. Utilisez la chaîne de pensée. Cela ne coûte rien et prend des minutes. Si la mise à jour vous obtient 80% du chemin, vous n'avez probablement pas besoin de régler.

**Second: RAG.**Si le modèle a besoin de connaître vos données spécifiques (documents, base de connaissances, catalogue de produits), la récupération est moins chère et plus maîtrisable que la mise en poids.

**Third: fine-tuning.**Utilisez ceci lorsque vous avez besoin du modèle pour adopter un style, un format ou un motif de raisonnement spécifique qui ne peuvent pas être atteints par la demande. Lorsque vous avez besoin d'une sortie structurée cohérente. Lorsque vous devez distiller un modèle plus grand en un modèle plus petit. Lorsque la latence est importante et que vous ne pouvez pas vous permettre les jetons supplémentaires à partir de quelques coups de demande.

```mermaid
graph TD
    Start["Need better model behavior?"] --> PE["Try prompt engineering"]
    PE -->|"Works"| Done["Ship it"]
    PE -->|"Not enough"| RAG["Need external knowledge?"]
    RAG -->|"Yes"| RAGBuild["Build RAG pipeline"]
    RAG -->|"No, need style/format change"| FT["Fine-tune with LoRA/QLoRA"]
    RAGBuild -->|"Works"| Done
    RAGBuild -->|"Also need style change"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff
```

```figure
lora-params
```

## Faites-le

Nous mettons en œuvre LoRA à partir de zéro dans PyTorch pur. Pas de bibliothèques. Pas de magie. Vous construirez la couche LoRA, l'injectez dans un modèle, l'entraînez, et fusionnez les poids.

### Étape 1: L'épaisseur de la LORA

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

A est initialisé avec des valeurs aléatoires à l'échelle. B est initialisé à zéro. Le produit BA commence à zéro, donc le modèle commence avec son comportement original.

### Étape 2: Layer linéaire enveloppé en LoRA

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

La couche linéaire originale est gelée. Seuls les paramètres LoRA (A et B) sont entraînables.

### Étape 3: Injecter du LoRA dans un modèle

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

Tout d'abord, congélez tous les paramètres du modèle, puis marchez sur l'arbre du modèle, trouvez des couches linéaires correspondant à vos noms cibles et remplacez-les par des versions enveloppées de LoRA. Les matrices LoRA A et B sont les seuls paramètres entraînables dans l'ensemble du modèle.

### Étape 4: Compte des paramètres

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### Étape 5: Réunir les poids

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

Après la fusion, les couches LoRA sont perdues. le modèle est de la même taille que l'original avec l'adaptation cuite dans les poids.

### Étape 6: Quantification QLoRA simulée

```python
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

Il simule la quantification à 4 bits en cartographiant les poids en 16 niveaux distincts dans des blocs de 64.

### Étape 7: La formation

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### Étape 8: Démo complète

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

La démo crée un petit modèle, injecte le LoRA en deux couches, l'entraîne et regroupe les poids. Le nombre de paramètres passe de plein entraînement à ~1% entraînement pendant la formation LoRA, puis revient à l'architecture originale après la fusion.

## Utilisez-le

Avec l'écosystème Hugging Face, LoRA sur un modèle réel prend environ 20 lignes:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Pour QLoRA, ajoutez la quantification des bits et des bytes:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

C'est tout. La même boucle d'entraînement, le même pipeline de données. Le modèle de base vit maintenant en 4 bits, les adaptateurs LoRA s'entraînent en fp16, et tout se passe en 6 Go.

Pour l' entraînement avec le coach de visage en étreignant:

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

L'adaptateur enregistré est de 10 à 100 Mo. Le modèle de base reste intact. Vous pouvez partager les adaptateurs sur le Hub Face Hugging sans redistribuer le modèle complet.

## La faire partir

Cette leçon donne:
- `outputs/prompt-lora-advisor.md`-- une requête qui vous aide à décider du classement LoRA, des modules cibles et des hyperparametres pour votre tâche spécifique
- `outputs/skill-fine-tuning-guide.md`- une compétence qui enseigne aux agents l'arbre de décision pour quand et comment affiner

## Exercices

1. **Rank ablation study.**Exécutez la démo avec les rangs 2, 4, 8, 16, 32 et 64. Plot final perte vs rang. Trouver le point de rendement diminuant où doubler le rang ne réduit plus la perte de moitié. Pour une simple tâche de classification sur les caractéristiques 256-dim, cela devrait être autour de r = 8-16.

2. **Target module comparison.**Modifiez inject_lora pour cibler uniquement la couche "0", uniquement la couche "2", uniquement la couche "4", et les trois. Exercez chaque variante pendant 20 époques. Comparer la vitesse de convergence et la perte finale. Cela reflète la décision réelle de cibler q_proj vs v_proj vs toutes les couches linéaires.

3. **Quantization error analysis.**Prenez les matrices de poids du modèle formé avant et après quantize_to_nf4 / dequantize_from_nf4. Computez l'erreur carrée moyenne, l'erreur maximale absolue et la corrélation entre les poids originaux et reconstitués.

4. **Multi-adapter serving.**Prenez deux adaptateurs LoRA sur différents sous-ensembles de données (même indices contre indices imparts). Salvez les deux adaptateurs. Chargez le modèle de base une fois, puis changez les adaptateurs et vérifiez que chacun produit des sorties différentes sur la même entrée. C'est ainsi que les systèmes de production servent plusieurs modèles finement réglés à partir d'une base.

5. **Merge vs. unmerged inference.**Comparez la sortie du modèle LoRA avant et après les poids de merge_lora sur les mêmes entrées 100. Vérifiez que les sorties sont identiques (dans la tolérance à point flottant de 1e-5).

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| LoRA | "Efficient fine-tuning" | Low-Rank Adaptation: freeze base weights, train two small matrices A and B whose product approximates the full weight update |
| QLoRA | "Fine-tune on a laptop" | Quantized LoRA: load the base model in 4-bit NF4, train LoRA adapters in fp16 on top, enabling 7B fine-tuning in 6GB VRAM |
| Rank (r) | "How much the model can learn" | The inner dimension of the A and B matrices; controls expressiveness vs. parameter count |
| Alpha | "LoRA learning rate" | Scaling factor applied to the LoRA output; $\alpha/r$ scales the adaptation's contribution to the final output |
| NF4 | "4-bit quantization" | Normal Float 4: a 4-bit data type with quantization levels at normal distribution quantiles, optimal for neural network weights |
| Adapter | "The small trained part" | The LoRA A and B matrices saved as a separate file (10-100MB), loadable on top of any copy of the base model |
| Target modules | "Which layers to LoRA" | The specific linear layers (q_proj, v_proj, etc.) where LoRA adapters are injected |
| Merging | "Bake it in" | Computing $W + (\alpha/r) \cdot BA$ and replacing the original weight, eliminating the adapter overhead at inference |
| Paged optimizers | "Don't OOM during training" | Offloading optimizer states (Adam momentum, variance) to CPU when GPU memory is exhausted |
| Catastrophic forgetting | "Fine-tuning broke everything else" | When updating all weights causes the model to lose previously learned capabilities |

## Pour en savoir plus

- Hu et coll., "LoRA: Adaptation à bas rang de modèles de langage de grande taille" (2021) -- le document original introduisant la méthode de décomposition à bas rang, testé sur GPT-3 175B avec un rang aussi bas que 4
- Dettmers et coll., "QLoRA: Finetuning efficace des modèles de langage quantifié" (2023) -- introduit NF4, double quantification et optimisateurs page, permettant 65B de finetuning sur un seul GPU de 48 Go
- Documentation de la bibliothèque PEFT (huggingface.co/docs/peft) - la bibliothèque standard pour les méthodes LoRA, QLoRA et autres méthodes efficaces en termes de paramètres dans l'écosystème Hugging Face
- Yadav et coll., "TIES-Merging: résoudre les interférences lors de la fusion de modèles" (2023) -- techniques pour combiner plusieurs adaptateurs LoRA sans dégradation de qualité
- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023)](https://arxiv.org/abs/2305.18290)-- dérivation DPO; l'étape de réglage des préférences qui vient après la FTS, pas besoin de modèle de récompense.
- [TRL documentation](https://huggingface.co/docs/trl/)-- référence officielle pour `SFTTrainer`- Je suis là .`DPOTrainer`- Je suis là .`KTOTrainer`, et la surface d'intégration avec PEFT/bitsandbytes/Unsloth.
- [Unsloth documentation](https://docs.unsloth.ai/)-- les noyaux fusionnés qui doublent le débit de réglage fin et réduisent de moitié la mémoire; la couche de performance sous TRL.
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/)-- Formateur multi-GPU SFT/DPO/QLoRA configuré par YAML; l'alternative de configuration en code aux scripts écrits à la main.
