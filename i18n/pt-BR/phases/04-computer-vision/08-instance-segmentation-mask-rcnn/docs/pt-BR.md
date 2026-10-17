# Segmentação de instâncias  Máscara R-CNN

> Adicione um pequeno ramo de máscara a um detector R-CNN mais rápido e você tem segmentação de instância.

**Type:** Build + Learn
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (YOLO), Phase 4 Lesson 07 (U-Net)
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Rastrear a arquitetura R-CNN de ponta a ponta: espinha dorsal, FPN, RPN, RoIAlign, cabeça de caixa, cabeça de máscara
- Implementar RoIAlign a partir do zero e explicar por que o RoIPool não é mais utilizado
- Use a visão de tocha .`maskrcnn_resnet50_fpn_v2`Modelo pré-treinado para máscaras de instância de qualidade de produção e leitura correta do formato de saída
- Tune-finos Mascar R-CNN em um pequeno conjunto de dados personalizados substituindo a caixa e as cabeças de máscaras e mantendo a coluna vertebral congelada

## O problema

Segmentação semântica dá-lhe uma máscara por classe. Segmentação de instância dá-lhe uma máscara por objeto, mesmo quando dois objetos compartilham uma classe. Contando indivíduos, rastreando através de quadros, e medindo coisas (a caixa de contorno de cada tijolo em uma parede, cada célula em uma imagem do microscópio) todos exigem segmentação de instância.

Mask R-CNN (He et al., 2017) resolveu isso reformulaindo a segmentação de instância como detecção-mais-uma-masca. O design foi tão limpo que durante os próximos cinco anos quase todos os documentos de segmentação de instância foram uma variante de Mask R-CNN, e a implementação de torchvision ainda é a padrão de produção para conjuntos de dados pequenos a médios.

O problema difícil da engenharia é a amostragem: como cortar uma região de características de tamanho fixo de uma caixa de propostas cujos cantos não se alinham com os limites dos pixels?

## O conceito

### A arquitetura

```mermaid
flowchart LR
    IMG["Input"] --> BB["ResNet<br/>backbone"]
    BB --> FPN["Feature<br/>Pyramid Network"]
    FPN --> RPN["Region<br/>Proposal<br/>Network"]
    FPN --> RA["RoIAlign"]
    RPN -->|"top-K proposals"| RA
    RA --> BH["Box head<br/>(class + refine)"]
    RA --> MH["Mask head<br/>(14x14 conv)"]
    BH --> NMS["NMS"]
    MH --> NMS
    NMS --> OUT["boxes +<br/>classes + masks"]

    style BB fill:#dbeafe,stroke:#2563eb
    style FPN fill:#fef3c7,stroke:#d97706
    style RPN fill:#fecaca,stroke:#dc2626
    style OUT fill:#dcfce7,stroke:#16a34a
```

Cinco peças para entender:

1. **Backbone** ResNet-50 ou ResNet-101 treinado em ImageNet. Produz uma hierarquia de mapas de recursos em passos 4, 8, 16, 32.
2. **FPN (Feature Pyramid Network)** ligações laterais de cima para baixo que dão a cada nível C canais de recursos ricos em semântica.
3. **RPN (Region Proposal Network)** uma pequena cabeça de convecção que, em cada posição de âncora, prevê "há um objeto aqui?" e "como refinar a caixa?". Produz ~ 1000 propostas por imagem.
4. **RoIAlign** amostras de um parche de tamanho fixo (por exemplo, 7x7) de qualquer caixa em qualquer nível de FPN.
5. **Heads** cabeça de caixa de duas camadas que refina a caixa e escolhe uma classe, mais uma pequena cabeça de convecção que produz uma `28x28`Mascaras binárias para cada proposta.

### Por que RoIAlign, e não RoIPool

O Fast R-CNN original usou RoIPool, que divide uma caixa de proposta em uma grade, leva o recurso máximo em cada célula e redonda todas as coordenadas em números inteiros.

```
RoIPool:
  box (34.7, 51.3, 98.2, 142.9)
  round -> (34, 51, 98, 142)
  split grid -> round each cell boundary
  misalignment accumulates at every step

RoIAlign:
  box (34.7, 51.3, 98.2, 142.9)
  sample at exact float coordinates using bilinear interpolation
  no rounding anywhere
```

O RoIAlign aumenta a massa AP em 3-4 pontos em COCO gratuitamente.

### A RPN num parágrafo

Em cada posição de um mapa de características, coloque caixas de âncora K de diferentes tamanhos e formas. Previr uma pontuação de objetividade para cada âncora e um deslocamento de regressão para transformar a âncora em uma caixa melhor adequada. Mantenha as caixas superiores por pontuação, aplique NMS em 0,7 IU e entregue os sobreviventes às cabeças. O RPN é treinado com sua própria mini-perda  a mesma estrutura que a perda YOLO da lição 6, apenas com duas classes (objeto / nenhum objeto).

### A cabeça da máscara

Para cada proposta (depois do RoIAlign) a cabeça da máscara é uma pequena FCN: quatro convas 3x3, uma deconva 2x, uma conva final 1x1 que produz `num_classes`canais de saída em `28x28`Resolução. Apenas o canal correspondente à classe prevista é mantido; os outros são ignorados.

Apresenta a máscara de 28x28 para o tamanho original da proposta para produzir a máscara binária final.

### Perdas

Mas R-CNN tem quatro perdas adicionadas:

```math
L = L_{\text{rpn\_cls}} + L_{\text{rpn\_box}} + L_{\text{box\_cls}} + L_{\text{box\_reg}} + L_{\text{mask}}
```

- `L_rpn_cls`- Não .`L_rpn_box` objectividade + regressão de caixa para as propostas de RPN.
- `L_box_cls` entropia cruzada sobre as classes (C+1) (incluindo o fundo) no classificador da cabeça.
- `L_box_reg` L1 liso no refinamento da caixa da cabeça.
- `L_mask` Entropia binária cruzada por pixel na saída de máscara 28x28.

Cada perda tem o seu próprio peso padrão; a implementação da visão de tocha expõe-as como argumentos de construção.

### Formatos de saída

`torchvision.models.detection.maskrcnn_resnet50_fpn_v2`Retorna uma lista de ditos, um por imagem:

```
{
    "boxes":  (N, 4) in (x1, y1, x2, y2) pixel coordinates,
    "labels": (N,) class IDs, 0 = background so indices are 1-based,
    "scores": (N,) confidence scores,
    "masks":  (N, 1, H, W) float masks in [0, 1] — threshold at 0.5 for binary,
}
```

A máscara já tem resolução de imagem completa.

```figure
cv3-roialign-sampling
```

## Construí-lo

### Passo 1: Alignar a RoIA a partir do zero

Este é o único componente da Mask R-CNN que é mais fácil de entender como código do que como prosa.

```python
import torch
import torch.nn.functional as F

def roi_align_single(feature, box, output_size=7, spatial_scale=1 / 16.0):
    """
    feature: (C, H, W) single-image feature map
    box: (x1, y1, x2, y2) in original image pixel coordinates
    output_size: side of the output grid (7 for box head, 14 for mask head)
    spatial_scale: reciprocal of the feature map stride
    """
    C, H, W = feature.shape
    x1, y1, x2, y2 = [c * spatial_scale - 0.5 for c in box]
    bin_w = (x2 - x1) / output_size
    bin_h = (y2 - y1) / output_size

    grid_y = torch.linspace(y1 + bin_h / 2, y2 - bin_h / 2, output_size)
    grid_x = torch.linspace(x1 + bin_w / 2, x2 - bin_w / 2, output_size)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    gx = 2 * (xx + 0.5) / W - 1
    gy = 2 * (yy + 0.5) / H - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(feature.unsqueeze(0), grid, mode="bilinear",
                            align_corners=False)
    return sampled.squeeze(0)
```

Cada número está numa posição bilinearmente amostrada, sem arredondamento, quantização, gradientes baixados.

### Passo 2: Comparar com o RoIAlign da torchvision

```python
from torchvision.ops import roi_align

feature = torch.randn(1, 16, 50, 50)
boxes = torch.tensor([[0, 10, 20, 100, 90]], dtype=torch.float32)  # (batch_idx, x1, y1, x2, y2)

ours = roi_align_single(feature[0], boxes[0, 1:].tolist(), output_size=7, spatial_scale=1/4)
theirs = roi_align(feature, boxes, output_size=(7, 7), spatial_scale=1/4, sampling_ratio=1, aligned=True)[0]

print(f"shape ours:   {tuple(ours.shape)}")
print(f"shape theirs: {tuple(theirs.shape)}")
print(f"max|diff|:    {(ours - theirs).abs().max().item():.3e}")
```

Com o`sampling_ratio=1`E ...`aligned=True`, os dois coincidem com dentro .`1e-5`- Não .

### Passo 3: Carregar uma máscara R-CNN pré-treinada

```python
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
print(f"classes (including background): {len(model.roi_heads.box_predictor.cls_score.out_features * [0])}")
```

Para a primeira classe (id 0), o modelo detecta tudo o que realmente começa com id 1.

### Passo 4: Execução de inferências

```python
with torch.no_grad():
    x = torch.randn(3, 400, 600)
    predictions = model([x])
p = predictions[0]
print(f"boxes:  {tuple(p['boxes'].shape)}")
print(f"labels: {tuple(p['labels'].shape)}")
print(f"scores: {tuple(p['scores'].shape)}")
print(f"masks:  {tuple(p['masks'].shape)}")
```

O tensor da máscara é forma .`(N, 1, H, W)`Limite de 0,5 para obter uma máscara binária por objeto:

```python
binary_masks = (p['masks'] > 0.5).squeeze(1)  # (N, H, W) boolean
```

### Passo 5: Troca as cabeças para uma contagem de classes personalizada

A receita comum de ajuste fino: reutilizar a espinha dorsal, FPN e RPN; substituir as duas cabeças do classificador.

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_custom_maskrcnn(num_classes):
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

custom = build_custom_maskrcnn(num_classes=5)
print(f"custom cls_score.out_features: {custom.roi_heads.box_predictor.cls_score.out_features}")
```

`num_classes`deve incluir a classe de fundo, para que um conjunto de dados com 4 classes de objetos use `num_classes=5`- Não .

### Passo 6: Congele o que não precisa de treinamento

Em pequenos conjuntos de dados, congelar a espinha dorsal e o FPN. Somente a objetividade RPN + regressão e as duas cabeças aprendem.

```python
def freeze_backbone_and_fpn(model):
    # torchvision Mask R-CNN packs the FPN inside `model.backbone` (as
    # `model.backbone.fpn`), so iterating `model.backbone.parameters()` covers
    # both the ResNet feature layers and the FPN lateral/output convs.
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model

custom = freeze_backbone_and_fpn(custom)
trainable = sum(p.numel() for p in custom.parameters() if p.requires_grad)
print(f"trainable after freeze: {trainable:,}")
```

Em conjuntos de dados de 500 imagens, esta é a diferença entre convergência e sobreajuste.

## Usá-lo

O ciclo de treinamento completo para a Mask R-CNN na torchvision é de 40 linhas e não muda significativamente entre tarefas  swap datasets e go.

```python
def train_step(model, images, targets, optimizer):
    model.train()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    return {k: v.item() for k, v in loss_dict.items()}
```

O `targets`A lista deve ter dicts por imagem com `boxes`- Não .`labels`, e `masks`(como `(num_instances, H, W)`O modelo retorna um ditado de quatro perdas durante o treinamento e uma lista de previsões durante a avaliação, com teclado em `model.training`- Não .

O `pycocotools`O evaluador produz mAP@IoU=0.5:0.95 tanto para caixas como para máscaras; é necessário ambos os números para saber se a cabeça de caixa ou a cabeça de máscara é o gargalo de engarrafamento.

## Envia-o

Esta lição produz:

- `outputs/prompt-instance-vs-semantic-router.md` um prompt que faz três perguntas e escolhe instância vs semântica vs panóptica, mais o modelo exato para começar.
- `outputs/skill-mask-rcnn-head-swapper.md` uma habilidade que gera as 10 linhas de código para trocar cabeças em qualquer modelo de detecção de torchvision, dada a nova `num_classes`- Não .

## Exercícios

1. **(Easy)**Verifique o seu RoIAlign contra `torchvision.ops.roi_align`Em 100 caixas aleatórias. Relate a diferença absoluta máxima. Também execute RoIPool (comportamento anterior a 2017) e mostre que diverge por ~1-2 pixels de mapa de características em caixas perto da fronteira.
2. **(Medium)**- A música é perfeita .`maskrcnn_resnet50_fpn_v2`Em um conjunto de dados personalizados de 50 imagens (qualquer duas classes: balões, peixes, poços, logotipos).
3. **(Hard)**Substitua a cabeça de máscara da Mask R-CNN por uma que prevê a 56x56 em vez de 28x28. Messa mAP@IoU = 0,75 antes e depois. Explique por que o ganho (ou falta de um) corresponde ao limite esperado de precisão / câmbio de memória.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Mask R-CNN | "Detection plus masks" | Faster R-CNN + a small FCN head that predicts a 28x28 mask per proposal per class |
| FPN | "Feature pyramid" | Top-down + lateral connections that give every stride level C channels of semantic-rich features |
| RPN | "Region proposer" | A small conv head that produces ~1000 object/no-object proposals per image |
| RoIAlign | "No-rounding crop" | Bilinearly samples a fixed-size feature grid from any float-coordinate box |
| RoIPool | "Pre-2017 crop" | Same purpose as RoIAlign but rounds box coordinates; obsolete |
| Mask AP | "Instance mAP" | Average precision computed with mask IoU instead of box IoU; the COCO instance segmentation metric |
| Binary mask head | "Per-class mask" | Predicts one binary mask per class for each proposal; only the predicted class's channel is kept |
| Background class | "Class 0" | The catch-all "no object" class; indices for real classes start at 1 |

## Mais leitura

- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870) o artigo; a secção 3 sobre RoIAlign é a leitura crítica
- [FPN: Feature Pyramid Networks (Lin et al., 2017)](https://arxiv.org/abs/1612.03144)O papel FPN; todos os detectores modernos o usam
- [torchvision Mask R-CNN tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) a referência para o ciclo de ajuste fino
- [Detectron2 model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) Implementações de produção com pesos treinados para quase todas as variantes de detecção e segmentação
