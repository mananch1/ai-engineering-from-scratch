# Configuration du GPU et du Cloud

> L'entraînement sur le processeur est bon pour apprendre.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Objectifs d'apprentissage

- Vérifiez la disponibilité du GPU local en utilisant `nvidia-smi`et l'API CUDA de PyTorch
- Configurer Google Colab avec un GPU T4 pour des expériences basées sur le cloud gratuites
- Indiquez la multiplication de la matrice de référence sur le processeur par rapport au processeur graphique et mesurez la vitesse
- Évaluer le modèle le plus grand qui s'adapte à votre VRAM en utilisant la règle de pouce fp16

## Le problème

La plupart des cours des phases 1-3 fonctionnent bien sur le processeur. Mais une fois que vous commencez à former des CNN, des transformateurs ou des LLM (phases 4+), vous avez besoin d'accélération de la GPU.

Vous avez trois options: GPU local, GPU cloud ou Google Colab (gratuit).

## Le concept

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## Faites-le

### Option 1: GPU NVIDIA local

Vérifiez si vous en avez un:

```bash
nvidia-smi
```

Installez PyTorch avec CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Option 2: Google Colab

1. Allez à la[colab.research.google.com](https://colab.research.google.com)
2. Temps d'exécution > Modifier le type d'exécution > T4 GPU
3. On court .`!nvidia-smi`pour vérifier

Téléchargez les carnets de ce cours directement à Colab.

### Option 3: GPU dans le cloud

Pour Lambda Labs, RunPod ou Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Pas de GPU ?

La plupart des leçons fonctionnent sur le processeur. Ceux qui ont besoin de GPU diront ainsi et incluront des liens Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Construire: GPU contre CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Exercices

1. Exécutez le point de référence ci-dessus et comparer les temps de CPU vs GPU
2. Si vous n'avez pas de GPU, exécutez-le sur Google Colab et comparez
3. Vérifiez combien de mémoire GPU vous avez et estimez le plus grand modèle que vous pouvez adapter (règle générale: 2 octets par paramètre pour fp16)

### Des indices

**Exercise 1.**Exécutez le code de la section Construire comme il est, puis écrivez le temps de processeur, le temps de GPU et la ligne de vitesse. Exécutez-le une deuxième fois; la première exécution de GPU inclut le coût de démarrage, de sorte que les deuxièmes chiffres sont ceux à comparer.

**Exercise 2.**Dans Colab, sélectionnez Runtime > Change de type de runtime > GPU, puis collez le point de référence dans une cellule.`!nvidia-smi`dans une autre cellule d'abord; il doit montrer un nom de GPU avant que le résultat de référence signifie quelque chose.

**Exercise 3.**Lisez la mémoire totale avec `torch.cuda.get_device_properties(0).total_memory / 1e9`, comme dans `code/gpu_check.py`. Divisez la mémoire en octets par 2 pour obtenir le nombre de paramètres pour fp16, puis gardez une marge, car les activations et le contexte CUDA utilisent également la mémoire.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
