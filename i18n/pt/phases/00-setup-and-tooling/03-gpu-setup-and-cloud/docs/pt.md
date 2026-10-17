# Configuração de GPU e nuvem

> O treinamento com CPU é bom para aprender.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Objetivos de aprendizagem

- Verifique a disponibilidade local da GPU usando `nvidia-smi`e a API CUDA da PyTorch
- Configure o Google Colab com uma GPU T4 para experimentos gratuitos baseados em nuvem
- Marque a multiplicação de matriz de referência na CPU vs GPU e mede a aceleração
- Estima o modelo maior que se encaixa no seu VRAM usando a regra fp16 do polegar

## O problema

A maioria das aulas nas fases 1-3 funciona bem na CPU. Mas uma vez que você começa a treinar CNNs, transformadores ou LLMs (fasas 4+), você precisa de aceleração da GPU. Uma corrida de treinamento que leva 8 horas na CPU leva 10 minutos na GPU.

Você tem três opções: GPU local, GPU em nuvem ou Google Colab (gratuito).

## O conceito

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

## Construí-lo

### Opção 1: GPU local NVIDIA

Verifique se tem um .

```bash
nvidia-smi
```

Instalar PyTorch com CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Opção 2: Google Colab

1. Vai para o[colab.research.google.com](https://colab.research.google.com)
2. Tempo de execução > Mudança de tipo de tempo de execução > GPU T4
3. Corra .`!nvidia-smi`para verificar

Faça o upload dos cadernos deste curso diretamente para o Colab.

### Opção 3: GPU em nuvem

Para Lambda Labs, RunPod ou Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### - Sem GPU?

A maioria das aulas funciona com CPU. Aqueles que precisam de GPU dirão isso e incluirão links Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Construir: GPU vs CPU benchmark

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

## Exercícios

1. Execute o benchmark acima e compare CPU vs GPU vezes
2. Se você não tem uma GPU, execute no Google Colab e compare
3. Verifique a quantidade de memória de GPU que você tem e estimar o maior modelo que você pode caber (regra geral: 2 bytes por parâmetro para fp16)

### Indicios

**Exercise 1.**Execute o código da seção Construir como está, depois anote o tempo da CPU, o tempo da GPU e a linha de aceleração.

**Exercise 2.**No Colab, selecione Runtime > Alterar o tipo de runtime > GPU, e depois colar o índice de referência em uma célula.`!nvidia-smi`em outra célula primeiro; deve mostrar um nome de GPU antes que o resultado de referência signifique qualquer coisa.

**Exercise 3.**Leia a memória total com `torch.cuda.get_device_properties(0).total_memory / 1e9`, como em `code/gpu_check.py`Divida a memória em bytes por 2 para obter a contagem de parâmetros para fp16, e mantenha uma margem, porque as ativações e o contexto CUDA também usam memória.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
