# Visão auto-supervisionada  SimCLR, DINO, MAE

> Os rótulos são o gargalo da visão supervisionada. O auto-supervisão pré-treinamento remove-os: aprenda características visuais de 100 milhões de imagens não rotuladas, sintonize-as em 10 mil.

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 04 (Image Classification), Phase 4 Lesson 14 (ViT)
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Rastrear as três principais famílias auto-supervisionadas  contrastiva (SimCLR), professor-estudante (DINO), reconstrução enmascarada (MAE)  e indicar o que cada uma otimiza
- Implementar uma perda do InfoNCE a partir do zero e explicar por que um lote de 512 funciona mas um lote de 32 falha
- Explicar por que a proporção de mascaramento de 75% do MAE não é arbitrária e como é diferente da proporção de 15% do BERT para o texto
- Utilize os pontos de controlo DINOv2 ou MAE ImageNet para a investigação linear e a recuperação de tiros zero

## O problema

A Supervised ImageNet tem 1,3 milhões de imagens rotuladas, que custam cerca de 10 milhões de dólares para anotar. Os conjuntos de dados médicos e industriais são menores e ainda mais caros de rotulado. Cada equipe de visão pergunta: podemos pre-treinar em dados baratos sem rotulagem  quadros do YouTube, rastreamentos na web, imagens de webcam, varreduras por satélite  e depois ajustar em um pequeno conjunto rotulado?

A aprendizagem auto-supervisionada é a resposta. Uma ViT moderna auto-supervisionada treinada em LAION ou JFT atinge ou supera a precisão da ImageNet supervisionada quando ajustada. Também transfere melhor para tarefas ao fundo (detecção, segmentação, profundidade) do que a pré-treinamento supervisionado. DINOv2 (Meta, 2023) e MAE (Meta, 2022) são as padrões atuais de produção para recursos de visão transferíveis.

A mudança conceitual é que a tarefa de pretexto  a coisa para a qual o modelo é treinado  não tem de ser a tarefa de baixo nível. O que importa é que força o modelo a aprender características úteis. Previr a cor das imagens em escala de cinza, girar as imagens e pedir ao modelo que classifique a rotação, mascarar os parches e reconstruí-los  tudo funcionou. As três abordagens que fazem essa escala são a aprendizagem contrastiva, a destilação professor-estudante e a reconstrução enmascarada.

## O conceito

### Três famílias

```mermaid
flowchart LR
    A["Contrastive<br/>SimCLR, MoCo, CLIP"] --> AT["positive pairs<br/>(same image, 2 augs)<br/>pulled together,<br/>negatives pushed apart"]
    B["Teacher-student<br/>DINO, BYOL, iBOT"] --> BT["student predicts<br/>teacher's output;<br/>teacher is EMA of student"]
    C["Masked reconstruction<br/>MAE, BEiT, SimMIM"] --> CT["mask 75% of patches;<br/>reconstruct pixel or<br/>token targets"]

    style A fill:#dbeafe,stroke:#2563eb
    style B fill:#fef3c7,stroke:#d97706
    style C fill:#dcfce7,stroke:#16a34a
```

### Aprendizagem contrasta (SimCLR)

Tome uma imagem, aplique duas ampliações aleatórias, obtenha duas visualizações. Alimenta as duas através do mesmo codificador e uma cabeça de projeção. Minimize uma perda que diz "estes dois incorporados devem ser próximos" e "este incorporado deve estar longe dos outros embutidos de imagem no lote".

```
Loss for positive pair (z_i, z_j) among 2N views per batch:

   L_ij = -log( exp(sim(z_i, z_j) / tau) / sum_k in batch \ {i} exp(sim(z_i, z_k) / tau) )

sim = cosine similarity
tau = temperature (0.1 standard)
```

Esta é a perda do InfoNCE. Requer muitos negativos por positivo, então o tamanho do lote importa. SimCLR precisa de 512-8192.

### Professor-aluno (DINO)

Duas redes com a mesma arquitetura: aluno e professor. O professor é uma média móvel exponencial (EMA) dos pesos do aluno. Ambos veem visões aumentadas da imagem. A saída do aluno é treinada para corresponder aos negativos explícitos do professor.

```
loss = CE( student_output(view_1),  teacher_output(view_2) )
     + CE( student_output(view_2),  teacher_output(view_1) )

teacher_weights = m * teacher_weights + (1 - m) * student_weights   (m ≈ 0.996)
```

Por que não se desmorona para "previr uma constante": a produção do professor é centrada (subtrair a média por dimensão) e afiada (dividida por pequena temperatura).

DINO é o que DINOv2 escala, em 142M de imagens curadas. As características resultantes são a atual SOTA para recuperação visual de tiros zero e previsão densa.

### Reconstrução mascarada (MAE)

Mascarar 75% dos patches de uma entrada ViT. Passar apenas os 25% visíveis através do codificador. Um pequeno decodificador recebe a saída do codificador mais tokens de máscara em posições mascaradas, e é treinado para reconstruir os pixels dos patches mascarados.

```
Encoder:  visible 25% of patches -> features
Decoder:  features + mask tokens at masked positions -> reconstructed pixels
Loss:     MSE between reconstructed and original pixels on masked patches only
```

As principais opções de design que fazem funcionar a MAE:

- **75% mask ratio**Força o codificador a aprender características semânticas; reconstruir 25% seria quase trivial (pixéis vizinhos são tão correlacionados que uma CNN poderia pregá-lo).
- **Asymmetric encoder/decoder**O grande codificador ViT só vê manchas visíveis; um pequeno decodificador (8 camadas, 512-dim) lida com a reconstrução. 3x mais rápido pré-treino do que o ingênuo BEiT.
- **Pixel-space reconstruction target** mais simples do que o alvo tokenizado da BEiT e funciona melhor no ViT.

Depois do treinamento, descartem o decodificador.

### Por que 75% e não 15%

O BERT mascara 15% dos tokens, o MAE mascara 75%, a diferença é a densidade de informação.

- A linguagem natural tem alta entropia por token. Previr 15% de tokens ainda é difícil porque cada posição mascarada tem muitas conclusões plausíveis.
- Os parches de imagem têm baixa entropia  um bairro desmascarado muitas vezes determina os pixels do parche mascarado quase exatamente. Para fazer previsão é necessário entender semântico, você tem que mascar agressivamente.

75% é suficientemente alto para que a simples extrapolação espacial não possa resolver a tarefa; o codificador deve representar o conteúdo da imagem.

### Avaliação por sonda linear

Após um pré-treino auto-supervisado, a avaliação padrão é uma avaliação de**linear probe**A informação é feita através de um sistema de classificação linear, que é um sistema de classificação linear.

- SimCLR ResNet-50: ~71% (2020)
- DINO ViT-S/16: ~77% (2021)
- MAE ViT-L/16: ~76% (2022)
- DINOv2 ViT-g/14: ~86% (2023)

A sonda linear é uma medida pura da qualidade das características; o ajuste fino geralmente adiciona 2-5 pontos, mas também mistura o efeito de reestruturação da cabeça.

```figure
data-augmentation
```

## Construí-lo

### Passo 1: Pipeline de aumento de duas visões

```python
import torch
import torchvision.transforms as T

two_view_train = lambda: T.Compose([
    T.RandomResizedCrop(96, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
])


class TwoViewDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self.aug = two_view_train()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        v1 = self.aug(img)
        v2 = self.aug(img)
        return v1, v2
```

Cada um .__getitem__Retorna duas visualizações aumentadas da mesma imagem; não são necessárias etiquetas.

### Passo 2: Perda de InfoNCE

```python
import torch.nn.functional as F

def info_nce(z1, z2, tau=0.1):
    """
    z1, z2: (N, D) L2-normalised embeddings of paired views
    """
    N, D = z1.shape
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = z @ z.T / tau              # (2N, 2N)

    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)
```

L2 - Normalize as inserções antes de ligar. `tau=0.1`é o padrão SimCLR; menor torna a perda mais nítida e requer mais negativos.

### Passo 3: Verificação da sanidade

```python
z1 = F.normalize(torch.randn(16, 32), dim=-1)
z2 = z1.clone()
loss_same = info_nce(z1, z2, tau=0.1).item()
z2_random = F.normalize(torch.randn(16, 32), dim=-1)
loss_random = info_nce(z1, z2_random, tau=0.1).item()
print(f"InfoNCE with identical pairs:  {loss_same:.3f}")
print(f"InfoNCE with random pairs:     {loss_random:.3f}")
```

Os pares idênticos devem dar uma baixa perda (cerca de 0 para um lote grande e temperatura fria).$\log(2N-1) \approx \log(31) \approx 3.4$com um lote de 16 pares.

### Passo 4: Mascaramento no estilo MAE

```python
def random_mask_indices(num_patches, mask_ratio=0.75, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_keep = int(num_patches * (1 - mask_ratio))
    perm = torch.randperm(num_patches, generator=g)
    visible = perm[:n_keep]
    masked = perm[n_keep:]
    return visible.sort().values, masked.sort().values


num_patches = 196
visible, masked = random_mask_indices(num_patches, mask_ratio=0.75)
print(f"visible: {len(visible)} / {num_patches}")
print(f"masked:  {len(masked)} / {num_patches}")
```

Simples, rápidos e deterministas para uma determinada semente.

## Usá-lo

DINOv2 é o padrão de produção em 2026:

```python
import torch
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()

# Per-image embeddings for zero-shot retrieval
with torch.no_grad():
    inputs = processor(images=[pil_image], return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0]  # CLS token
```

A incorporação resultante de 768-dim é a espinha dorsal da recuperação de imagens moderna, correspondência densa e canalizações de transferência de tiros zero.

Para as incorporações de imagem-texto, o SigLIP ou o OpenCLIP é o equivalente; para a sintonização fina de estilo MAE, o `timm`O repo vai a todos os pontos de controlo da MAE.

## Envia-o

Esta lição produz:

- `outputs/prompt-ssl-pretraining-picker.md` um prompt que seleciona SimCLR / MAE / DINOv2 dado o tamanho do conjunto de dados, computação e tarefa a jusante.
- `outputs/skill-linear-probe-runner.md` uma habilidade que escreve a avaliação de sonda linear para qualquer codificador congelado + conjunto de dados rotulado.

## Exercícios

1. **(Easy)**Verifique se a perda de InfoNCE cai quando você diminui a temperatura para as incorporações bem alinhadas e aumenta quando você diminui a temperatura para as incorporações aleatórias.`tau in [0.05, 0.1, 0.2, 0.5]`- Contra a perda.
2. **(Medium)**Implementar um amortecimento do centro de estilo DINO. Mostre que sem o centramento, o aluno desmorona para um vetor constante dentro de algumas épocas.
3. **(Hard)**Treinar MAE em CIFAR-100 usando a TinyUNet a partir da lição 10 como a espinha dorsal. Relatar a precisão da sonda linear em 10, 50 e 200 épocas. Mostrar que uma sonda linear treinada pela MAE bate uma sonda linear supervisionada desde zero no mesmo subconjunto de 1.000 imagens.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Self-supervised | "Label-free" | A pretext task that produces useful representations from unlabelled data |
| Pretext task | "The fake task" | The objective used during SSL (reconstruct patches, match views); discarded after pretraining |
| Linear probe | "Frozen encoder + linear head" | Standard SSL evaluation: train only a linear classifier on top of frozen features |
| InfoNCE | "Contrastive loss" | softmax over cosine similarities; positive pair is the target class, all others are negatives |
| EMA teacher | "Moving-average teacher" | Teacher whose weights are an exponential moving average of the student's; used by BYOL, MoCo, DINO |
| Mask ratio | "% of patches hidden" | Fraction of patches masked during MAE; 75% for vision, 15% for text |
| Representation collapse | "Constant output" | SSL failure where the encoder outputs a constant vector for all inputs; prevented by centring, sharpening, or negatives |
| DINOv2 | "Production SSL backbone" | Meta's 2023 self-supervised ViT; strongest general-purpose image features in 2026 |

## Mais leitura

- [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709) Referência de aprendizagem contrastada
- [DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294) professor-aluno com impulso, centramento, afiamento
- [MAE (He et al., 2022)](https://arxiv.org/abs/2111.06377) Autoencoder mascarado Pre- treino para ViT
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) A escala da ViT auto-supervisionada para as características de produção
