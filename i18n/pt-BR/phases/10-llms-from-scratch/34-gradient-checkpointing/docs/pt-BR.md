# Controlos de graduação e recomputação de ativação

> O Backprop mantém todas as atividades intermediárias. Em parâmetros 70B e contexto 128K que é 3 TB de atividades por rank.

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## O problema

O treinamento de um transformador armazena, para cada camada, as entradas para cada operação que é diferenciada para trás: as entradas de atenção, as projeções Q/K/V, a saída softmax, as entradas FFN, as saídas normais e o fluxo residual.`d`, comprimento da sequência `L`, lote `B`, isto é por ordem de`12 * B * L * d`flutuam por camada.

Para o`d=8192, L=8192, B=1`, que é 800 MB / camada em BF16. Um modelo de 64 camadas é 51 GB de ativações  e isso é antes de você multiplicar pelo tamanho do microbatch, antes de adicionar atenção-softmax intermediários (`L^2`por cabeça), e antes de fazerem cópias parciais tensor-paralelas.

A conta bilateral: BF16 pesos mais estado de otimização pode caber em 80GB, mas as ativações empurram você para além. Gradiente de controle (aka recomputação de ativação) é a solução padrão. Largar a maioria das ativações; refazer o avanço durante o retorno para recuperá-los. Custo: FLOPs extras. Benefício: diminuição da memória pela proporção de segmentos de pontos de controle para as camadas totais.

Feito ingenuamente, o checkpointing custa cerca de 33% mais FLOPs de passagem avançada por passo. Bem feito  checkpointing seletivo por "seleção inteligente" de Korthikanti et al.  você economiza 5x de memória para menos de 5% FLOP gastos. E com matmuls FP8, FSDP offload, e especialista paralelo MoE isso realmente importa: você não pode pagar nem a memória nem o desperdício de computação.

## O conceito

### O que o retrógrado realmente precisa

`output = layer(input)`- Quer para trás .`grad_input`E ...`grad_params`Para os calcular , é preciso:

- `input`(para calcular `grad_params = input.T @ grad_output`para camadas lineares)
- Alguns intermediários derivados de ativação (a derivada de ReLU/GELU/softmax depende do valor de ativação)

O passe avançado armazena-os automaticamente no gráfico de autogrado.`tensor.retain_grad()`E cada operação que precisa de sua entrada mantém uma referência.

### Naívo em todo o ponto de controlo

Divide a rede em `N`Segmentos. Durante o avanço, armazenar apenas a * entrada * para cada segmento. Quando o retrocessor precisa de intermediários, reexibir o passagem para a frente do segmento para materializá-los, em seguida, diferenciar.

Exemplo: Transformador de 32 camadas dividido em 32 segmentos de 1 camada cada.

- Memória: 32 entradas de camadas (pequenas) vs 32 * (volume de ativação por camada) (grande).
- Computação extra: 1 unidade adicional para a frente por segmento, ou seja, ~ 33% mais FLOPs para a frente no total (uma vez que o retroceder é 2x para a frente, o passo completo torna-se 1 + 1 + 2 = 4 unidades em vez de 1 + 2 = 3).

Esta é a receita original de Chen et al. 2016: um ponto de controlo por semana `sqrt(L)`Elas são capazes de equilibrar memória e computação.$L=64$, que são 8 pontos de controlo.

### Controlos seletivos (Korthikanti 2022)

Não todas as ativas custam o mesmo.`B*L*L*heads`A ativação oculta do FFN é `B*L*4d`Para sequências longas, o softmax domina.

O ponto de controlo seletivo mantém as ativas de armazenamento baratas (projeções lineares, resíduos) e recalcula apenas as caras (atenção).$O(L^2)$Memória.

Megatron-Core implementa isso como recomputada de ativação "seletiva".

### Descarga

Alternativa à recomputada: ativações de navegação para a RAM da CPU entre o avanço e o retorno. Requer largura de banda PCIe; beneficial quando a largura de banda ociosa excede o custo da rematerialização. Estratégias mistas são comuns: ponto de verificação de algumas camadas, descarregar outras.

O FSDP2 desliga como uma opção de primeira classe. O desliga brilha quando a GPU está em engarrafamento na memória, mas a transferência CPU-GPU tem espaço de cabeça.

### Modelo de custos de recomposição

FLOPs por passo com um controlo ingênuo a cada `k`- Equipamentos de`L`- Não .

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Com o checkpoint seletivo, recomputamos apenas o núcleo de atenção, não toda a camada:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Modelo de poupança de memória

Volume de ativação por camada: `A`- Para o ...`L`camadas, memória de ativação total: `L * A`- Não .

Ponto de controlo completo ( tamanho do segmento 1): só armazenamento `L * input_volume`(~)`L * 1/10 A`Para um transformador padrão).`9 * L * A * 1/10`- Não .

Ponto de controlo a cada`k`camadas: armazenamento `L/k * A`- E mais .`k-1`Valor das camadas no segmento ativo.

- Não .`k = sqrt(L)`, memória e recomputador custo ambos a escala com `sqrt(L)` a compensação ideal para as camadas de custos uniformes.

### Quando não chegar ao ponto de controlo

- As camadas mais íntimas de um estágio de oleoduto já estão em voo.
- As primeiras e últimas camadas se dominarem a computação do estágio (raras em transformadores).
- Os kernels de atenção já usando FlashAttention  Flash já recalcula o softmax rápido, então o controle adicional de nível de camada adiciona pouco no topo.

### Padrões de implementação

1. **Function wrapper:**Envolver um segmento em `torch.utils.checkpoint.checkpoint(fn, input)`- Só em lojas PyTorch .`input`, recalcula tudo o resto para trás.

2. **Decorator-based:**As camadas de rotulagem são verificáveis; o treinador decide, no momento da configuração, quais os segmentos que serão enrolados.

3. **Manual explicit recompute:**Escreve a passagem para trás, chamando-a de costume.`recompute_forward`que duplica o forward com a entrada armazenada.

Os três dão o mesmo resultado funcional.

### Interação com TP / PP / FP8

- **Tensor parallel:**As entradas dos pontos de controlo devem ser recolhidas ou redistribuídas no recomputo; suportar os custos de comunicação.
- **Pipeline parallel:**padrão típico é para controlar a direcção de cada fase do pipeline para a frente para que micro batches de ordem inversa possam reutilizar a memória de ativação.
- **FP8 recompute:**As histórias da amax atualizadas durante a recomputada devem corresponder às da escala inicial ou às derivações da escala FP8.

```figure
activation-recompute
```

## Construí-lo

### Passo 1: Um modelo de brinquedo com segmentos

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Passo 2: Naívo para trás, precisa de todas as atividades

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Passo 3: Memória de cada ponto de verificação

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Passo 4: Modelo de custo

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Passo 5: Estimador de memória

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Passo 6: Tamanho óptimo do segmento

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Passo 7: Decisão seletiva sobre o ponto de controlo

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Usá-lo

- **torch.utils.checkpoint**- Não .`from torch.utils.checkpoint import checkpoint`- o envolvente canônico no PyTorch. Enrola uma função; armazenar apenas entradas, recomputar para trás.
- **Megatron-Core activation recomputation**: apoios `selective`- Não .`full`, e `block`Modos de formação de fronteira.
- **FSDP2 offload**- Não .`module.to_empty(device="cpu")`com`offload_policy`em FSDP2 fragmenta a ativação para a CPU em vez de recomputar.
- **DeepSpeed ZeRO-Offload**: Descarga da CPU para estados e ativações do optimizador, complementando o ponto de verificação.

## Envia-o

Esta lição produz`outputs/prompt-activation-recompute-policy.md` um prompt que leva a configuração do modelo (camadas, escondidas, sequentes, lotes) e a memória disponível da GPU e emite uma política de recomputação por camada (não / seletiva / completa / descarregada).

## Exercícios

1. Verifique a correcção.`model_forward`+ `model_backward`(ativações completas) vs `model_forward_checkpointed`+ `model_backward_checkpointed`Os gradientes dos parâmetros devem ser idênticos à precisão da máquina.

2. Dimensão do segmento de varredura `k`de 1 a `L`- Traçar o plano de cabeça e memória.

3. Implementar um ponto de verificação seletivo: armazenar a entrada do módulo de atenção, mas não os seus intermediários.

4. Adicionar descarga. Salvar entradas de segmento para um "buffer de CPU" simulado (uma lista separada). Medir "PCIe largura de banda" como bytes/tempo e encontrar o ponto de equilíbrio entre descarga e recomputada.

5. Marque de referência um verdadeiro transformador PyTorch com e sem `torch.utils.checkpoint`. Medir a memória (via `torch.cuda.max_memory_allocated`) e tempo de passagem.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | "Save memory by redoing forward" | Store segment inputs only; recompute intermediates during backward to get gradient-support tensors |
| Activation recomputation | "Same as checkpointing" | The HPC-flavored name for the same technique |
| Segment size (k) | "How many layers per checkpoint" | Number of layers whose intermediates are dropped and rematerialized together |
| Selective checkpointing | "Korthikanti's trick" | Recompute only expensive-to-store activations (attention softmax); keep cheap ones |
| Full checkpointing | "The naive version" | Recompute every layer's intermediates in every segment |
| Block checkpointing | "Coarse-grained" | Checkpoint whole transformer blocks; largest granularity |
| FLOP overhead | "The compute tax" | Extra FLOPs per step = (recompute FLOPs) / (fwd + bwd FLOPs); 33% naive, 5% selective |
| Activation offload | "Ship to CPU" | Move activations to CPU RAM across forward->backward; alternative to recompute |
| sqrt-L rule | "The classical optimum" | For uniform-cost layers, optimal checkpoint spacing is $\sqrt{L}$ layers |
| Attention-softmax volume | "The O(L^2) problem" | $L^2 \cdot \text{heads} \cdot \text{batch}$ floats; dominates activation memory at long contexts |

## Mais leitura

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174)- O papel original que formalizou o ponto de checagem de gradientes
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198)-- recomputamento seletivo de ativação e análise formal de custos
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645)-- abordagem alternativa de memória constante através da rematerialização de modo inverso
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840)-- descarga de ativação em escala
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html)-- a API padrão
- [Megatron Bridge activation recomputation documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)-- modos seletivos, completos e bloqueados
