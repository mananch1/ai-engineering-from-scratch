# Déboguer les réseaux neuronaux

> Votre réseau a été compilé. Il a fonctionné. Il a produit un numéro. Le numéro est erroné et rien n'a écrasé. Bienvenue dans le type de débogage le plus difficile - le type où il n'y a pas de message d'erreur.

**Type:** Build
**Languages:** Python, PyTorch
**Prerequisites:** Phase 03 Lessons 01-10 (especially backpropagation, loss functions, optimizers)
**Time:** ~90 minutes

## Objectifs d'apprentissage

- Diagnostication des défaillances courantes du réseau neuronal (perte de NaN, courbe de perte plate, suradaptation, oscillation) en utilisant des stratégies de débogage systématiques
- Appliquez la technique "sur-match" pour vérifier que votre architecture de modèle et votre boucle de formation sont corrects
- Inspecter les magnitudes des gradients, les répartitions d'activation et les normes de poids pour identifier les problèmes de gradients qui disparaissent ou explodent
- Construire une liste de contrôle de débogage qui couvre les problèmes de pipeline de données, d'architecture de modèle, de fonction de perte, d'optimisateur et de taux d'apprentissage

## Le problème

Un logiciel traditionnel s'écrase lorsqu'il est cassé. Un pointeur nul lance une exception. Un type de déséquilibre échoue au moment de la compilation. Une erreur de type unique produit une sortie clairement erronée.

Les réseaux neuraux ne vous donnent pas ce luxe.

Un réseau neural cassé est terminé, imprime une valeur de perte et fait des prédictions. La perte pourrait diminuer. Les prédictions peuvent sembler plausibles. Mais le modèle est silencieusement faux: apprendre des raccourcis, mémoriser le bruit, ou converger à un minimum local inutile. Les chercheurs de Google ont estimé que 60-70% du temps de débogage ML est consacré à des bugs "silencieux" qui ne produisent aucune erreur mais dégradent la qualité du modèle.

La différence entre un modèle de travail et un modèle cassé réside souvent dans une seule ligne manquante: une ligne manquante `zero_grad()`, une dimension transposée, un taux d'apprentissage de 10 fois supérieur. La recette canonique "Recipe pour la formation des réseaux neuronaux" (2019) commence par: "Les erreurs de réseau neuronal les plus courantes sont les bugs qui ne se crashent pas".

Cette leçon vous apprend à trouver ces insectes.

## Le concept

### L'attitude mentale défectueuse

Oubliez le débogage d'impression et de prélèvement. Le débogage du réseau neuronal nécessite une approche systématique car la boucle de rétroaction est lente (de minutes à heures par séance d'entraînement) et les symptômes sont ambiguës (une mauvaise perte pourrait signifier 20 choses différentes).

La règle d' or:**start simple, add complexity one piece at a time, and verify each piece independently.**

```mermaid
flowchart TD
    A["Loss not decreasing"] --> B{"Check learning rate"}
    B -->|"Too high"| C["Loss oscillates or explodes"]
    B -->|"Too low"| D["Loss barely moves"]
    B -->|"Reasonable"| E{"Check gradients"}
    E -->|"All zeros"| F["Dead ReLUs or vanishing gradients"]
    E -->|"NaN/Inf"| G["Exploding gradients"]
    E -->|"Normal"| H{"Check data pipeline"}
    H -->|"Labels shuffled"| I["Random-chance accuracy"]
    H -->|"Preprocessing bug"| J["Model learns noise"]
    H -->|"Data is fine"| K{"Check architecture"}
    K -->|"Too small"| L["Underfitting"]
    K -->|"Too deep"| M["Optimization difficulty"]
```

### Symptome 1: la perte ne diminue pas

C'est la plainte la plus courante: la boucle d'entraînement se poursuit, les époques passent, et la perte reste plate ou oscille de façon effrénée.

**Wrong learning rate.**Trop élevé: la perte oscille ou saute à NaN. Trop bas: la perte diminue si lentement qu'elle semble plate. Pour Adam, commencez à 1e-3. Pour SGD, commencez à 1e-1 ou 1e-2. Essayez toujours 3 taux d'apprentissage couvrant 10 fois chacun (par exemple, 1e-2, 1e-3, 1e-4) avant de conclure que quelque chose d'autre est mal.

**Dead ReLUs.**Si un neurone ReLU reçoit une entrée négative importante, il donne 0 et son gradient est 0. Il ne s'active jamais à nouveau. Si suffisamment de neurones meurent, le réseau ne peut pas apprendre. Vérifiez: imprimez la fraction d'activations qui sont exactement 0 après chaque couche ReLU. Si > 50% sont morts, passez à LeakyReLU ou réduisez le taux d'apprentissage.

**Vanishing gradients.**Dans les réseaux profonds avec des activations sigmoïdes ou tanh, les gradients se rétrécissent de façon exponentielle à mesure qu'ils se propagent vers l'arrière.

**Exploding gradients.**Le problème inverse - les gradients augmentent de façon exponentielle. communs dans les RNN et les réseaux très profonds.`torch.nn.utils.clip_grad_norm_`), réduire le taux d'apprentissage ou ajouter une normalisation.

### Symptome 2: Les pertes diminuent mais le modèle est mauvais

La perte diminue, la précision de l'entraînement atteint 99%, mais la précision des tests est de 55%, ou le modèle produit des résultats absurdes sur des données réelles.

**Overfitting.**Le modèle mémorise les données d'entraînement au lieu des schémas d'apprentissage. L'écart entre l'entraînement et la perte de validation augmente avec le temps.

**Data leakage.**Les données des tests ont été divulguées dans l'entraînement. La précision est suspectément élevée. Causes courantes: mélange avant la fractionnement, pré-traitement avec des statistiques provenant du jeu complet de données, échantillons dupliqués sur des fractions. Réparation: fractionner en premier, pré-traiter en second, vérifier les duplicates.

**Label errors.**5 à 10% des étiquettes dans la plupart des ensembles de données réels sont erronées (Northcutt et coll., 2021 -- "Erres de l'étiquette pervasives dans les ensembles de test"). Le modèle apprend le bruit.

### Symptome 3: NaN ou Inf en perte

La valeur de perte devient `nan`ou `inf`- L'entraînement est mort.

**Learning rate too high.**Les mises à jour de la phase sont trop longues pour que les poids explosent.

**log(0) or log(negative).**Compute la perte de l' entropie croisée `log(p)`Si votre modèle donne exactement 0 ou une probabilité négative, le journal explose.`[eps, 1-eps]`où `eps=1e-7`- Je suis désolé .

**Division by zero.**La normalisation de lot se divise par déviation standard. Un lot avec des valeurs constantes a std=0. Fix: ajouter l'epsilon au dénominateur (PyTorch le fait par défaut, mais les implémentations personnalisées peuvent ne pas).

**Numerical overflow.**De grandes activations sont introduites `exp()`La solution est de soustraire le maximum avant d'exposer (le truc de log-sum-exp).

### Technique 1: vérification progressive

Comparer vos gradients analytiques (de l'arrière-prop) aux gradients numériques (de la différence finie).

Gradient numérique pour paramètre `w`- Le numéro de la liste:

```math
\text{grad\_numerical} = \frac{\text{loss}(w + \text{eps}) - \text{loss}(w - \text{eps})}{2 \cdot \text{eps}}
```

Métrique de l'accord (différence relative):

```math
\text{rel\_diff} = \frac{\lvert \text{grad\_analytical} - \text{grad\_numerical} \rvert}{\max(\lvert \text{grad\_analytical} \rvert, \lvert \text{grad\_numerical} \rvert, 10^{-8})}
```

Si vous`rel_diff < 1e-5`: correct. si`rel_diff > 1e-3`C'est presque certainement un insecte.

```mermaid
flowchart LR
    A["Parameter w"] --> B["w + eps"]
    A --> C["w - eps"]
    B --> D["Forward pass"]
    C --> E["Forward pass"]
    D --> F["loss+"]
    E --> G["loss-"]
    F --> H["(loss+ - loss-) / 2eps"]
    G --> H
    H --> I["Compare to backprop gradient"]
```

### Technique 2: Statistiques d'activation

Surveiller la moyenne et l'écart standard des activations après chaque couche pendant l'entraînement.

| Health indicator | Mean | Std | Diagnosis |
|-----------------|------|-----|-----------|
| Healthy | ~0 | ~1 | Network is learning normally |
| Saturated | >>0 or <<0 | ~0 | Activations stuck at extreme values |
| Dead | 0 | 0 | Neurons are dead (all zeros) |
| Exploding | >>10 | >>10 | Activations growing without bound |

### Técnique 3: Visualisation du flux de débit

Dans un réseau sain, les magnitudes de gradients doivent être approximativement similaires entre les couches. Si les premières couches ont des gradients 1000 fois plus petits que les couches ultérieures, vous avez des gradients qui disparaissent.

```mermaid
graph LR
    subgraph "Healthy Gradient Flow"
        L1["Layer 1<br/>grad: 0.05"] --- L2["Layer 2<br/>grad: 0.04"] --- L3["Layer 3<br/>grad: 0.06"] --- L4["Layer 4<br/>grad: 0.05"]
    end
```

```mermaid
graph LR
    subgraph "Vanishing Gradient Flow"
        V1["Layer 1<br/>grad: 0.0001"] --- V2["Layer 2<br/>grad: 0.003"] --- V3["Layer 3<br/>grad: 0.02"] --- V4["Layer 4<br/>grad: 0.08"]
    end
```

### Technique 4: Test sur mesure en un seul lot

La technique de débogage la plus importante dans l'apprentissage profond.

Prenez un petit lot (8-32 échantillons). Exercez-le pendant plus de 100 itérations. La perte devrait aller à presque zéro et la précision de formation devrait atteindre 100%. Si ce n'est pas le cas, votre modèle ou boucle de formation a un bug fondamental - ne pas procéder à la formation complète.

Ce test détecte:
- Fonctions de perte brisées
- Passe-pièces en arrière cassées
- L'architecture est trop petite pour représenter les données
- Optimisateur non connecté aux paramètres du modèle
- Les données et les étiquettes sont mal alignées

Cela prend 30 secondes pour exécuter et économise des heures de débogage des exercices d'entraînement complets.

### Técnique 5: Rate Finder d'apprentissage

Leslie Smith (2017) a proposé de balayer le taux d'apprentissage de très petit (1e-7) à très grand (10) sur une époque tout en enregistrant la perte.

```mermaid
graph TD
    subgraph "LR Finder Plot"
        direction LR
        A["1e-7: loss=2.3"] --> B["1e-5: loss=2.3"]
        B --> C["1e-3: loss=1.8"]
        C --> D["1e-2: loss=0.9 -- steepest"]
        D --> E["1e-1: loss=0.5"]
        E --> F["1.0: loss=NaN -- too high"]
    end
```

Le meilleur LR dans cet exemple: ~1e-3 (un ordre de grandeur avant le point le plus raide).

### Les insectes pyTorch communs

Ce sont les insectes qui perdent le plus d'heures collectives dans la communauté PyTorch:

| Bug | Symptom | Fix |
|-----|---------|-----|
| Forgetting `optimizer.zero_grad()` | Gradients accumulate across batches, loss oscillates | Add `optimizer.zero_grad()` before `loss.backward()` |
| Forgetting `model.eval()` at test time | Dropout and batch norm behave differently, test accuracy varies between runs | Add `model.eval()` and `torch.no_grad()` |
| Wrong tensor shapes | Silent broadcasting produces wrong results, no error | Print shapes after every operation during debugging |
| CPU/GPU mismatch | `RuntimeError: expected CUDA tensor` | Use `.to(device)` on model AND data |
| Not detaching tensors | Computation graph grows forever, OOM | Use `.detach()` or `with torch.no_grad()` |
| In-place operations breaking autograd | `RuntimeError: modified by in-place operation` | Replace `x += 1` with `x = x + 1` |
| Data not normalized | Loss stuck at random-chance level | Normalize inputs to mean=0, std=1 |
| Labels as wrong dtype | Cross-entropy expects `Long`, got `Float` | Cast labels: `labels.long()` |

### La table de débogage principale

| Symptom | Likely cause | First thing to try |
|---------|-------------|-------------------|
| Loss stuck at -log(1/num_classes) | Model predicting uniform distribution | Check data pipeline, verify labels match inputs |
| Loss NaN after a few steps | Learning rate too high | Reduce LR by 10x |
| Loss NaN immediately | log(0) or division by zero | Add epsilon to log/division operations |
| Loss oscillating wildly | LR too high or batch size too small | Reduce LR, increase batch size |
| Loss decreasing then plateaus | LR too high for fine-tuning phase | Add LR schedule (cosine or step decay) |
| Training acc high, test acc low | Overfitting | Add dropout, weight decay, more data |
| Training acc = test acc = chance | Model not learning anything | Run overfit-one-batch test |
| Training acc = test acc but both low | Underfitting | Bigger model, more layers, more features |
| Gradients all zero | Dead ReLUs or detached computation graph | Switch to LeakyReLU, check `.requires_grad` |
| Out of memory during training | Batch too large or graph not freed | Reduce batch size, use `torch.no_grad()` for eval |

```figure
learning-curves
```

## Faites-le

Un ensemble d'outils de diagnostic qui surveille les activations, les gradients et les courbes de perte.

### Étape 1: La classe de débogage réseau

Connecte avec un modèle PyTorch pour enregistrer des statistiques d'activation et de gradient par couche.

```python
import torch
import torch.nn as nn
import math


class NetworkDebugger:
    def __init__(self, model):
        self.model = model
        self.activation_stats = {}
        self.gradient_stats = {}
        self.loss_history = []
        self.lr_losses = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.LeakyReLU)):
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self.hooks.append(hook)
                hook = module.register_full_backward_hook(self._make_gradient_hook(name))
                self.hooks.append(hook)

    def _make_activation_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                out = output.detach().float()
                self.activation_stats[name] = {
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "fraction_zero": (out == 0).float().mean().item(),
                    "min": out.min().item(),
                    "max": out.max().item(),
                }
        return hook

    def _make_gradient_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                with torch.no_grad():
                    grad = grad_output[0].detach().float()
                    self.gradient_stats[name] = {
                        "mean": grad.mean().item(),
                        "std": grad.std().item(),
                        "abs_mean": grad.abs().mean().item(),
                        "max": grad.abs().max().item(),
                    }
        return hook

    def record_loss(self, loss_value):
        self.loss_history.append(loss_value)

    def check_loss_health(self):
        if len(self.loss_history) < 2:
            return "NOT_ENOUGH_DATA"
        recent = self.loss_history[-10:]
        if any(math.isnan(v) or math.isinf(v) for v in recent):
            return "NAN_OR_INF"
        if len(self.loss_history) >= 20:
            first_half = sum(self.loss_history[:10]) / 10
            second_half = sum(self.loss_history[-10:]) / 10
            if second_half >= first_half * 0.99:
                return "NOT_DECREASING"
        if len(recent) >= 5:
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            if max(diffs) - min(diffs) > 2 * abs(sum(diffs) / len(diffs)):
                return "OSCILLATING"
        return "HEALTHY"

    def check_activations(self):
        issues = []
        for name, stats in self.activation_stats.items():
            if stats["fraction_zero"] > 0.5:
                issues.append(f"DEAD_NEURONS: {name} has {stats['fraction_zero']:.0%} zero activations")
            if abs(stats["mean"]) > 10:
                issues.append(f"EXPLODING_ACTIVATIONS: {name} mean={stats['mean']:.2f}")
            if stats["std"] < 1e-6:
                issues.append(f"COLLAPSED_ACTIVATIONS: {name} std={stats['std']:.2e}")
        return issues if issues else ["HEALTHY"]

    def check_gradients(self):
        issues = []
        grad_magnitudes = []
        for name, stats in self.gradient_stats.items():
            grad_magnitudes.append((name, stats["abs_mean"]))
            if stats["abs_mean"] < 1e-7:
                issues.append(f"VANISHING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
            if stats["abs_mean"] > 100:
                issues.append(f"EXPLODING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
        if len(grad_magnitudes) >= 2:
            first_mag = grad_magnitudes[0][1]
            last_mag = grad_magnitudes[-1][1]
            if last_mag > 0 and first_mag / last_mag > 100:
                issues.append(f"GRADIENT_RATIO: first/last = {first_mag/last_mag:.0f}x (vanishing)")
        return issues if issues else ["HEALTHY"]

    def print_report(self):
        print("\n=== NETWORK DEBUGGER REPORT ===")
        print(f"\nLoss health: {self.check_loss_health()}")
        if self.loss_history:
            print(f"  Last 5 losses: {[f'{v:.4f}' for v in self.loss_history[-5:]]}")
        print("\nActivation diagnostics:")
        for item in self.check_activations():
            print(f"  {item}")
        print("\nGradient diagnostics:")
        for item in self.check_gradients():
            print(f"  {item}")
        print("\nPer-layer activation stats:")
        for name, stats in self.activation_stats.items():
            print(f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f} zero={stats['fraction_zero']:.1%}")
        print("\nPer-layer gradient stats:")
        for name, stats in self.gradient_stats.items():
            print(f"  {name}: abs_mean={stats['abs_mean']:.2e} max={stats['max']:.2e}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
```

### Étape 2: Test sur mesure en une seule série

```python
def overfit_one_batch(model, x_batch, y_batch, criterion, lr=0.01, steps=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("\n=== OVERFIT ONE BATCH TEST ===")
    print(f"Batch size: {x_batch.shape[0]}, Steps: {steps}")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                preds = (output > 0).float() if output.shape[-1] == 1 else output.argmax(dim=1)
                targets = y_batch if y_batch.dim() == 1 else y_batch.squeeze()
                acc = (preds.squeeze() == targets).float().mean().item()
            print(f"  Step {step:3d} | Loss: {loss.item():.6f} | Accuracy: {acc:.1%}")

    final_loss = loss.item()
    if final_loss > 0.1:
        print(f"\n  FAIL: Loss did not converge ({final_loss:.4f}). Model or training loop is broken.")
        return False
    print(f"\n  PASS: Loss converged to {final_loss:.6f}")
    return True
```

### Étape 3: Rate Finder d'apprentissage

```python
def find_learning_rate(model, x_data, y_data, criterion, start_lr=1e-7, end_lr=10, steps=100):
    import copy
    original_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / steps)

    model.train()
    results = []
    best_loss = float("inf")
    current_lr = start_lr

    print("\n=== LEARNING RATE FINDER ===")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)

        if math.isnan(loss.item()) or loss.item() > best_loss * 10:
            break

        best_loss = min(best_loss, loss.item())
        results.append((current_lr, loss.item()))

        loss.backward()
        optimizer.step()

        current_lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    model.load_state_dict(original_state)

    if len(results) < 10:
        print("  Could not complete LR sweep -- loss diverged too quickly")
        return results

    min_loss_idx = min(range(len(results)), key=lambda i: results[i][1])
    suggested_lr = results[max(0, min_loss_idx - 10)][0]

    print(f"  Swept {len(results)} steps from {start_lr:.0e} to {results[-1][0]:.0e}")
    print(f"  Minimum loss {results[min_loss_idx][1]:.4f} at lr={results[min_loss_idx][0]:.2e}")
    print(f"  Suggested learning rate: {suggested_lr:.2e}")

    return results
```

### Étape 4: vérification des degrés

```python
def _flat_to_multi_index(flat_idx, shape):
    multi_idx = []
    remaining = flat_idx
    for dim in reversed(shape):
        multi_idx.insert(0, remaining % dim)
        remaining //= dim
    return tuple(multi_idx)


def gradient_check(model, x, y, criterion, eps=1e-4):
    model.train()
    x_double = x.double()
    y_double = y.double()
    model_double = model.double()

    print("\n=== GRADIENT CHECK ===")
    overall_max_diff = 0
    checked = 0

    for name, param in model_double.named_parameters():
        if not param.requires_grad:
            continue

        layer_max_diff = 0

        model_double.zero_grad()
        output = model_double(x_double)
        loss = criterion(output, y_double)
        loss.backward()
        analytical_grad = param.grad.clone()

        num_checks = min(5, param.numel())
        for i in range(num_checks):
            idx = _flat_to_multi_index(i, param.shape)
            original = param.data[idx].item()

            param.data[idx] = original + eps
            with torch.no_grad():
                loss_plus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original - eps
            with torch.no_grad():
                loss_minus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = analytical_grad[idx].item()

            denom = max(abs(numerical), abs(analytical), 1e-8)
            rel_diff = abs(numerical - analytical) / denom

            layer_max_diff = max(layer_max_diff, rel_diff)
            checked += 1

        overall_max_diff = max(overall_max_diff, layer_max_diff)
        status = "OK" if layer_max_diff < 1e-5 else "MISMATCH"
        print(f"  {name}: max_rel_diff={layer_max_diff:.2e} [{status}]")

    model.float()

    print(f"\n  Checked {checked} parameters")
    if overall_max_diff < 1e-5:
        print("  PASS: Gradients match (rel_diff < 1e-5)")
    elif overall_max_diff < 1e-3:
        print("  WARN: Small differences (1e-5 < rel_diff < 1e-3)")
    else:
        print("  FAIL: Gradient mismatch detected (rel_diff > 1e-3)")
    return overall_max_diff
```

### Étape 5: Les réseaux délibérément cassés

Appliquez le kit d'outils aux réseaux cassés et diagnostiquez chacun d'eux.

```python
def demo_broken_networks():
    torch.manual_seed(42)
    x = torch.randn(64, 10)
    y = (x[:, 0] > 0).long()

    print("\n" + "=" * 60)
    print("BUG 1: Learning rate too high (lr=10)")
    print("=" * 60)
    model1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger1 = NetworkDebugger(model1)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=10.0)
    criterion = nn.CrossEntropyLoss()
    for step in range(20):
        optimizer1.zero_grad()
        out = model1(x)
        loss = criterion(out, y)
        debugger1.record_loss(loss.item())
        loss.backward()
        optimizer1.step()
    debugger1.print_report()
    debugger1.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 2: Dead ReLUs from bad initialization")
    print("=" * 60)
    model2 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    with torch.no_grad():
        for m in model2.modules():
            if isinstance(m, nn.Linear):
                m.weight.fill_(-1.0)
                m.bias.fill_(-5.0)
    debugger2 = NetworkDebugger(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for step in range(50):
        optimizer2.zero_grad()
        out = model2(x)
        loss = criterion(out, y)
        debugger2.record_loss(loss.item())
        loss.backward()
        optimizer2.step()
    debugger2.print_report()
    debugger2.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 3: Missing zero_grad (gradients accumulate)")
    print("=" * 60)
    model3 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger3 = NetworkDebugger(model3)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    for step in range(50):
        out = model3(x)
        loss = criterion(out, y)
        debugger3.record_loss(loss.item())
        loss.backward()
        optimizer3.step()
    debugger3.print_report()
    debugger3.remove_hooks()

    print("\n" + "=" * 60)
    print("HEALTHY NETWORK: Correct setup for comparison")
    print("=" * 60)
    model_good = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger_good = NetworkDebugger(model_good)
    optimizer_good = torch.optim.Adam(model_good.parameters(), lr=1e-3)
    for step in range(50):
        optimizer_good.zero_grad()
        out = model_good(x)
        loss = criterion(out, y)
        debugger_good.record_loss(loss.item())
        loss.backward()
        optimizer_good.step()
    debugger_good.print_report()
    debugger_good.remove_hooks()

    print("\n" + "=" * 60)
    print("OVERFIT-ONE-BATCH TEST (healthy model)")
    print("=" * 60)
    model_test = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    overfit_one_batch(model_test, x[:8], y[:8], criterion)

    print("\n" + "=" * 60)
    print("LEARNING RATE FINDER")
    print("=" * 60)
    model_lr = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    find_learning_rate(model_lr, x, y, criterion)

    print("\n" + "=" * 60)
    print("GRADIENT CHECK")
    print("=" * 60)
    model_grad = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 2))
    gradient_check(model_grad, x[:4], y[:4], criterion)
```

## Utilisez-le

### Outils intégrés à PyTorch

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

with torch.autograd.detect_anomaly():
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.abs().mean():.2e}")
```

### Intégration des poids et des biais

```python
import wandb

wandb.init(project="debug-training")

for epoch in range(100):
    loss = train_one_epoch()
    wandb.log({
        "loss": loss,
        "lr": optimizer.param_groups[0]["lr"],
        "grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")),
    })

    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu().numpy())})
```

### Tableau de tension

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/debug_experiment")

for epoch in range(100):
    loss = train_one_epoch()
    writer.add_scalar("Loss/train", loss, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)
```

### Liste de débogage (avant la formation complète)

1. Faites un test de sur-fit à un lot.
2. Récapitulatif du modèle imprimé - vérifier le nombre de paramètres est raisonnable.
3. Exécutez un seul passe avant avec des données aléatoires - vérifiez la forme de sortie.
4. Traînez pendant 5 époques - vérifier les pertes diminue.
5. Vérifiez les statistiques d'activation. Pas de couches mortes, pas d'explosions.
6. Vérifiez le débit de gradient. Pas de disparition, pas d'explosion.
7. Vérifiez le pipeline de données -- imprimez 5 échantillons aléatoires avec des étiquettes.

## La faire partir

Cette leçon donne:
- `outputs/prompt-nn-debugger.md`-- une demande pour diagnostiquer les défaillances de formation des réseaux neuronaux
- `outputs/skill-debug-checklist.md`-- une liste de contrôle des arbres de décision pour les problèmes de débogage de formation

Modèles de déploiement clés pour le débogage:
- Ajouter des crochets de surveillance aux scripts de formation de production
- Activation de journaux et statistiques de gradients à W&B ou TensorBoard à chaque N étapes
- Implementer des alertes automatiques pour la perte de NaN, les neurones morts (> 80% zéro) ou l'explosion de gradient
- Exécuter toujours le test sur-ensemble en un lot lors du changement d'architectures ou de pipelines de données

## Exercices

1. **Add an exploding gradient detector.**Modifier le `NetworkDebugger`Pour détecter lorsque les gradients dépassent un seuil et suggérer automatiquement une valeur de coupure de gradient.

2. **Build a dead neuron resurrector.**Écrivez une fonction qui identifie les neurones morts de ReLU (souvent en sortie 0) et réinitialise leurs poids entrants avec l'initialisation de Kaiming. Montrez que cela récupère un réseau où > 70% des neurones sont morts.

3. **Implement the learning rate finder with plotting.**Extension `find_learning_rate`pour enregistrer les résultats en tant que CSV et écrire un script séparé qui lit le CSV et affiche la courbe LR vs perte en utilisant matplotlib. Identifier la LR optimale pour ResNet-18 sur CIFAR-10.

4. **Create a data pipeline validator.**Écrivez une fonction qui vérifie: les échantillons dupliqués sur les fractions train/test, le déséquilibre de distribution des étiquettes (> rapport 10:1), la normalisation des entrées (média proche de 0, std proche de 1), et les valeurs NaN/Inf dans les données.

5. **Debug a real failure.**Prenez le mini-cadre de la leçon 10, introduisez un bug subtil (par exemple, transposez la matrice de poids en arrière), et utilisez la vérification des gradients pour localiser exactement quel paramètre a des gradients incorrects.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Silent bug | "It runs but gives bad results" | A bug that produces no error but degrades model quality -- the dominant failure mode in ML |
| Dead ReLU | "The neurons died" | A ReLU neuron whose input is always negative, so it outputs 0 and receives 0 gradient permanently |
| Vanishing gradients | "Early layers stop learning" | Gradients shrink exponentially through layers, making weights in early layers effectively frozen |
| Exploding gradients | "Loss went to NaN" | Gradients grow exponentially through layers, causing weight updates so large they overflow |
| Gradient checking | "Verify backprop is correct" | Comparing analytical gradients from backprop to numerical gradients from finite differences |
| Overfit-one-batch | "The most important debug test" | Training on a single small batch to verify the model CAN learn -- if it cannot, something is fundamentally broken |
| LR finder | "Sweep to find the right learning rate" | Exponentially increasing the learning rate over one epoch and picking the rate just before loss diverges |
| Data leakage | "Test data leaked into training" | When information from the test set contaminates training, producing artificially high accuracy |
| Activation statistics | "Monitor layer health" | Tracking mean, std, and zero-fraction of each layer's output to detect dead, saturated, or exploding neurons |
| Gradient clipping | "Cap the gradient magnitude" | Scaling gradients down when their norm exceeds a threshold, preventing exploding gradient updates |

## Pour en savoir plus

- Smith, "Taux d'apprentissage cycliques pour la formation des réseaux neuronaux" (2017) - le document introduisant le test de la plage de fréquence d'apprentissage (LR finder)
- Northcutt et coll., "Erres de marque généralisées dans les ensembles de test déstabilisent les critères de référence de l'apprentissage automatique" (2021) -- démontre que 3 à 6% des étiquettes de ImageNet, CIFAR-10 et d'autres critères de référence majeurs sont erronés
- Zhang et coll., "Comprendre l'apprentissage profond nécessite une rééducation générale" (2017) -- le document montrant que les réseaux neuraux peuvent mémoriser des étiquettes aléatoires, c'est pourquoi le test sur-ensemble fonctionne
- Documents de PyTorch sur `torch.autograd.detect_anomaly`et `torch.autograd.set_detect_anomaly`pour la détection intégrée de NaN/Inf
