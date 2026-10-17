# Classificação de imagens

> Um classificador é uma função de pixels para uma distribuição de probabilidade em classes.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 Lesson 09 (Model Evaluation), Phase 3 Lesson 10 (Mini Framework), Phase 4 Lesson 03 (CNNs)
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Construir um conjunto de classificação de imagens de ponta a ponta no CIFAR-10: conjunto de dados, aumento, modelo, ciclo de formação, avaliação
- Explique o papel de cada componente (datloader, perda, optimizador, agendador, aumento) e prevê como a quebra de qualquer um deles se manifesta na curva de perda
- Implementar mistura, corte e suavização de rótulos a partir do zero e justificar quando cada um vale a pena ser adicionado
- Leia uma matriz de confusão e uma tabela de precisão/recolha por classe para diagnosticar falhas de conjuntos de dados e modelos além da precisão agregada

## O problema

Cada tarefa de visão que é enviada reduz-se à classificação de imagem em algum nível. Detecção classifica regiões. Segmentação classifica pixels. Retorno classifica por semelhança com classe centroides. Obter classificação correta  o ciclo do conjunto de dados, a política de aumento, a perda, a avaliação  é a habilidade que transfere para todas as outras tarefas na fase.

A maioria dos bugs de classificação não estão no modelo. Eles vivem no pipeline: uma normalização quebrada, um conjunto de treinamento não alterado, um aumento que distorce as etiquetas, uma divisão de validação contaminada por dados de treinamento, uma taxa de aprendizagem que silenciosamente diverge após a época 30. Uma CNN que atingiria 93% no CIFAR-10 com uma configuração correta normalmente marca 70-75% com uma quebrada, e a curva de perda parece plausível o tempo todo.

Esta lição conecta todo o gasoduto à mão para que cada parte seja inspecionável.`torchvision.datasets`Isso pode esconder um inseto.

## O conceito

### O canal de classificação

```mermaid
flowchart LR
    A["Dataset<br/>(images + labels)"] --> B["Augment<br/>(random transforms)"]
    B --> C["Normalise<br/>(mean/std)"]
    C --> D["DataLoader<br/>(batch + shuffle)"]
    D --> E["Model<br/>(CNN)"]
    E --> F["Logits<br/>(N, C)"]
    F --> G["Cross-entropy loss"]
    F --> H["Argmax<br/>at eval"]
    G --> I["Backward"]
    I --> J["Optimizer step"]
    J --> K["Scheduler step"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

Cada linha neste loop é onde um bug pode viver.`model(x).softmax()`As aumentas aplicam-se apenas às entradas, não às etiquetas  exceto para mistura, que mistura as duas. `optimizer.zero_grad()`O que acontece é que o erro de aprendizagem é muito maior que o que acontece com os erros de aprendizagem.

### Entropia cruzada, logitas e softmax

Um classificador produz `C`números por imagem chamados logits. Aplicando softmax, eles são convertidos em uma distribuição de probabilidade:

```math
\mathop{\mathrm{softmax}}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
```

A entropia cruzada mede a probabilidade de registro negativo da classe correta:

```math
\begin{aligned}
\mathrm{CE}(z, y) &= -\log\left( \mathop{\mathrm{softmax}}(z)_y \right) \\
&= -z_y + \log\left( \sum_j \exp(z_j) \right)
\end{aligned}
```

A forma à direita é a numéricamente estável (log-sum-exp).`nn.CrossEntropyLoss`A aplicação de softmax em primeiro lugar é quase sempre um erro  você calcula $\log(\mathop{\mathrm{softmax}}(\mathop{\mathrm{softmax}}(z)))$, uma quantidade sem sentido.

### Por que a ampliação funciona

Uma CNN tem preconceito indutivo para a tradução (de compartilhamento de peso), mas não há invariância embutida para culturas, viradas, nervosismo de cores ou oclusão. A única maneira de ensiná-lo essas invariâncias é mostrá-lo pixels que os exercitam.

```
Original crop:  "dog facing left"
Flip:           "dog facing right"       <- same label, different pixels
Rotate(+15):    "dog, slight tilt"
Colour jitter:  "dog in warmer light"
RandomErasing:  "dog with patch missing"
```

A regra: o aumento deve preservar o rótulo. O corte e a rotação em um dígito podem virar "6" para "9"; para esse conjunto de dados você usa intervalos de rotação menores e escolhe aumentos que respeitem as invariâncias específicas de dígitos.

### Mistura e corte

A ampliação normal transforma pixels mas mantém os rótulos um só. **Mixup**E ...**cutmix**quebrar isso interpolar os dois.

```
Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  paste a random rectangle of x_j into x_i
  y = area-weighted mix of y_i and y_j
```

Por que ajuda: o modelo deixa de memorizar alvos picantes e aprende a interpolar entre as aulas. A perda de treinamento aumenta, a precisão dos testes aumenta. É a atualização de robustez única mais barata para qualquer classificador.

### Limeamento de rótulos

Um primo de confusão, em vez de treinar contra o`[0, 0, 1, 0, 0]`, treno contra`[eps/C, eps/C, 1-eps, eps/C, eps/C]`Para um pequeno .`eps`O modelo não produz logites arbitrariamente afiados e melhora a calibração quase sem custo.`nn.CrossEntropyLoss(label_smoothing=0.1)`desde a PyTorch 1.10.

### Avaliação além da precisão

A precisão agregada esconde desequilíbrio. Um classificador binário de 90 a 10 que sempre prevê a classe majoritária pontua 90%.

- **Per-class accuracy** um número por classe; imediatamente aparece categorias com baixo desempenho.
- **Confusion matrix**- Não .$C \times C$Grade com linha $i$Col.$j$= contagem de classe verdadeira $i$Previsto como classe .$j$A diagonal é correta, as fora-diagonais são onde o modelo vive.
- **Top-1 / Top-5** se a classe correta está nas previsões de 1 ou 5 melhores; Top-5 importa para a ImageNet porque classes como "Norwich Terrier" vs "Norfolk Terrier" são genuinamente ambíguas.
- **Calibration (ECE)** uma previsão de confiança de 0,8 consegue ser correta 80% do tempo? redes modernas são sistematicamente excessivamente confiantes; corrigir com escala de temperatura ou suavização de rótulos.

```figure
receptive-field
```

## Construí-lo

### Passo 1: Um conjunto de dados sintéticos deterministas

CIFAR-10 vive no disco. Para tornar esta lição reprodutivel e rápida, construímos um conjunto de dados sintéticos que se parece com CIFAR  32x32 imagens RGB com estrutura específica de classe que o modelo deve aprender.

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

Cada classe recebe sua própria paleta de cores e padrão de frequência, além de ruído gaussiano para forçar o modelo a aprender o sinal em vez de memorizar pixels. Dez classes, mil imagens cada, permutadas.

### Passo 2: Normalização e aumento

As duas transformações que todos os canais de visão têm.

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

Refletir pad antes da colheita, não pad zero, porque as fronteiras pretas são um sinal que o modelo aprenderia a ignorar de forma inútil.

### Passo 3: Mistura

Mistura duas imagens e duas etiquetas dentro da etapa de treinamento. Implementado como um lote de transformação para que viva ao lado da passagem dianteira em vez de dentro do conjunto de dados.

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy`É a entropia cruzada contra uma distribuição de etiqueta macia.

### Passo 4: O ciclo de treinamento

A receita completa: uma passagem dos dados, gradientes uma vez por lote, cronógrafo passo uma vez por época.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # Training accuracy vs the un-mixed labels `y` is only an approximation
        # when mixup is on (the model saw soft targets, not y). Treat it as a
        # rough progress signal; rely on val accuracy for real performance.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

Cinco invariantes que você verifica sempre que escrever um ciclo de treinamento:

1. `model.train()`antes da formação, `model.eval()`antes da avaliação  revertem o comportamento de abandono e de batchnorma.
2. `.zero_grad()`Antes de`.backward()`- Não .
3. `.item()`Quando estamos a acumular métricas, então nada mantém o gráfico de cálculo vivo.
4. `@torch.no_grad()`Durante a avaliação, a utilização de um sistema de avaliação de dados permite a economia de memória e de tempo, previne acidentes sutis.
5. Argmax contra logits brutos, não softmax  o mesmo resultado, uma operação menor.

### Passo 5: Coloque-o juntos

Use o `TinyResNet`A partir da lição anterior, treinar por algumas épocas, avaliar.

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet comes from the previous lesson (03-cnns-lenet-to-resnet).
# Adjust the import path to wherever you stored the previous lesson's code.
from cnns_lenet_to_resnet import TinyResNet  # example placeholder

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

No conjunto de dados sintético, isso chega a uma precisão de validação quase perfeita dentro de cinco épocas, o que é o ponto: o pipeline é correto, o modelo pode aprender o que é apropriado.

### Passo 6: Leia a matriz de confusão

A precisão sozinha nunca diz onde o modelo está a falhar.

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

As linhas são classes verdadeiras, as colunas são previsões. Um conjunto de contagens fora de diagonais entre as classes 3 e 5 significa que o modelo confunde essas duas e dá-lhe um ponto de partida para a coleta de dados direcionada ou um aumento específico para a classe.

## Usá-lo

`torchvision`Para o CIFAR-10 real, o conjunto completo é de quatro linhas e um ciclo de treinamento.

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

Duas coisas a observar: a média/std são **dataset-specific** computado no conjunto de treinamento CIFAR-10, não ImageNet  e o pad de reflexão é a política de colheita padrão da comunidade.

## Envia-o

Esta lição produz:

- `outputs/prompt-classifier-pipeline-auditor.md` um aviso que verifica um roteiro de treinamento para as cinco invariantes acima e revela a primeira violação.
- `outputs/skill-classification-diagnostics.md` uma habilidade que, dada uma matriz de confusão e uma lista de nomes de classes, resume falhas por classe e propõe a solução única mais impactante.

## Exercícios

1. **(Easy)**Explique por que a perda de trens com mistura é maior, mas a precisão de val é similar ou melhor.
2. **(Medium)**Implementar Cutout  zero um quadrado aleatório de 8x8 em cada imagem de treinamento  e executar uma ablação vs nenhum aumento, hflip+crop, hflip+crop+cutout, hflip+crop+mixup.
3. **(Hard)**Construir um pipeline CIFAR-100 (100 classes, mesmo tamanho de entrada) e reproduzir um treinamento ResNet-34 executado com uma precisão de 1% da publicada.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Logits | "Raw outputs" | The pre-softmax vector of C numbers per image; cross-entropy expects these, not softmaxed values |
| Cross-entropy | "The loss" | Negative log-probability of the correct class; combines log-softmax and NLL in one stable op |
| DataLoader | "The batcher" | Wraps a dataset with shuffling, batching, and (optional) multi-worker loading; gets blamed for half of training bugs |
| Augmentation | "Random transforms" | Any pixel-level transform at training time that preserves the label; teaches invariances the CNN does not have natively |
| Mixup / Cutmix | "Mix two images" | Blend both inputs and labels so the classifier learns smooth interpolations instead of hard boundaries |
| Label smoothing | "Softer targets" | Replace one-hot with (1-eps, eps/(C-1), ...); improves calibration and slightly boosts accuracy |
| Top-k accuracy | "Top-5" | The correct class is in the k highest-probability predictions; used on datasets with genuinely ambiguous classes |
| Confusion matrix | "Where errors live" | C x C table where entry (i, j) counts images of true class i predicted as j; diagonal is right, off-diagonal tells you what to fix |

## Mais leitura

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/) ainda a viagem mais clara do pipeline de formação em uma única página
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187) cada pequeno truque que juntos adiciona 3-4% à precisão da ResNet na ImageNet
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412) o papel original de mistura; três páginas de teoria e experimentos convincentes
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599)O papel que provou que as redes modernas são erroneamente calibradas e fixadas com um parâmetro escalar
