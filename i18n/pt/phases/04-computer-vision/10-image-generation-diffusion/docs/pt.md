# Geração de imagens  Modelos de difusão

> Um modelo de difusão aprende a denotar, treina-o a remover um pequeno pouco de ruído de uma imagem barulhenta, repita isso para trás mil vezes, e você tem um gerador de imagem.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 07 (U-Net), Phase 1 Lesson 06 (Probability), Phase 3 Lesson 06 (Optimizers)
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Derivar o processo de som para a frente `x_0 -> x_1 -> ... -> x_T`e explicar por que a forma fechada`q(x_t | x_0)`aplica-se a qualquer t
- Implementar um objetivo de formação de estilo DDPM que regredisse o ruído adicionado em cada etapa e um amostreador que volte do ruído puro para uma imagem
- Construir uma U-Net condicionada ao tempo (pouco pequena para treinar na CPU) que prevê o ruído para qualquer passo do tempo
- Explique a diferença entre a amostragem DDPM e a amostragem DDIM e quando cada uma é adequada (a lição 23 abrange a correspondência de fluxo e o fluxo rectificado em profundidade)

## O problema

Os GANs geram um tiro: ruído, imagem, passagem para frente. São rápidos e difíceis de treinar. Os modelos de difusão geram iterativamente: começam a partir de ruído puro, denotam em pequenos passos, surge imagem. São lentos e fáceis de treinar. Nos últimos cinco anos, esta última propriedade tem dominado: qualquer pequena equipa pode treinar um modelo de difusão e obter amostras razoáveis; o treinamento GAN é um artefacto que se aprende durante anos de corridas falhadas.

Além da estabilidade de treinamento, a estrutura iterativa da difusão é o que desbloqueia tudo o que a geração moderna de imagens faz: condicionamento de texto, pintura, edição de imagem, super-resolução, estilo controlável. Cada etapa do ciclo de amostragem é um lugar para injetar uma nova restrição. Esse gancho é por isso que a Stable Diffusion, Imagen, DALL-E 3, Midjourney, e todos os modelos de imagem controlables que você usará são baseados em difusão.

Esta lição constrói o DDPM mínimo: ruído para frente, denoção para trás, ciclo de treinamento.

## O conceito

### O processo avançado

Tome uma imagem .`x_0`Adicione uma pequena quantidade de ruído gaussiano para obter .`x_1`Adicione mais uma pequena quantidade para obter .`x_2`- Continua .$T$Passo até `x_T`É quase indistinguível do ruído gaussiano puro.

```math
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \; \beta_t I\right)
```

`beta_t`é um cronograma de variância pequeno, tipicamente linear de 0,0001 a 0,02 sobre $T=1000$Cada passo reduz ligeiramente o sinal e injeta um novo ruído.

### O salto fechado

Adicionar ruído um passo a cada vez é uma cadeia de Markov, mas a matemática se dobra: você pode amostrar `x_t`directamente de`x_0`Em um passo.

```math
\begin{aligned}
&\text{Define } \alpha_t = 1 - \beta_t \\
&\text{Define } \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s \\[6pt]
&\text{Then: } q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \; (1 - \bar{\alpha}_t) I\right) \\[6pt]
&\text{Equivalently: } x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, I)
\end{aligned}
```

Esta equação única é a razão pela qual a difusão é prática.`t`, amostra `x_t`directamente de`x_0`, e treinar em um passo  sem simulação da cadeia completa de Markov necessária.

### O processo inverso

O processo avançado é fixo.`p(x_{t-1} | x_t)`Os modelos de difusão não prevêem a`x_{t-1}`direto; eles prevêem o ruído `epsilon`adicionado em passo $t$, e a matemática deriva`x_{t-1}`- Não.

```mermaid
flowchart LR
    X0["x_0<br/>(clean image)"] --> Q1["q(x_t|x_0)<br/>add noise"]
    Q1 --> XT["x_t<br/>(noisy)"]
    XT --> MODEL["model(x_t, t)"]
    MODEL --> EPS["predicted epsilon"]
    EPS --> LOSS["MSE against<br/>true epsilon"]

    XT -.->|sampling| STEP["p(x_{t-1}|x_t)"]
    STEP -.-> XT1["x_{t-1}"]
    XT1 -.->|repeat 1000x| X0S["x_0 (sampled)"]

    style X0 fill:#dcfce7,stroke:#16a34a
    style MODEL fill:#fef3c7,stroke:#d97706
    style LOSS fill:#fecaca,stroke:#dc2626
    style X0S fill:#dbeafe,stroke:#2563eb
```

### A perda de treinamento

Para cada etapa de formação:

1. Amostra de uma imagem real .`x_0`- Não .
2. Escolha um passo no tempo `t`uniformemente a partir de $[1, T]$- Não .
3. Resumo de amostra`epsilon ~ N(0, I)`- Não .
4. Computação`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`- Não .
5. Previsão`epsilon_theta(x_t, t)`com a rede.
6. Minimizar `|| epsilon - epsilon_theta(x_t, t) ||^2`- Não .

A rede neural aprende a prever o ruído em qualquer etapa do tempo. A perda é MSE. Não há jogo adversário, não há colapso, não há oscilação.

### O amostragem (DDPM)

Para gerar: começar a partir de `x_T ~ N(0, I)`e caminhar para trás, um passo a cada vez.

```
for t = T, T-1, ..., 1:
    eps = model(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sqrt(beta_t) * z
    where z ~ N(0, I) if t > 1, else 0
return x_0
```

A chave é que, embora a condicional inversa não seja conhecida em forma fechada em geral, para este processo avançado Gaussian específico é. Os coeficientes feio-parecem ser o que a regra de Bayes dá.

### Por que mil passos

O cronograma de ruído para a frente é escolhido para que cada passo adicione apenas ruído suficiente para que o passo inverso seja quase gaussiano.$T=1000$com um cronograma linear é o padrão DDPM.

### DDIM: amostragem 20 vezes mais rápida

O treinamento é o mesmo. As mudanças de amostragem. DDIM (Song et al., 2020) define um processo determinista inverso que salta etapas de tempo sem re treinamento. A amostragem em 50 etapas com DDIM dá uma qualidade de DDPM de quase 1000 etapas. Todo sistema de produção usa DDIM ou uma variante ainda mais rápida (DPM-Solver, ancestral de Euler).

### Condicionamento de tempo

A rede .`epsilon_theta(x_t, t)`Os modelos de difusão modernos injectam`t`através de embalagens de tempo sinusoidais (a mesma ideia que codificação posicional em transformadores) que são adicionadas a mapas de recursos em todos os níveis da U-Net.

```
t_embedding = sinusoidal(t)
feature_map += MLP(t_embedding)
```

Sem condicionar o tempo, a rede tem que adivinhar o nível de ruído da própria imagem, que funciona mas é muito menos eficiente na amostra.

```figure
cv-diffusion-image
```

## Construí-lo

### Passo 1: Programa de ruído

```python
import torch

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }

schedule = precompute_schedule(linear_beta_schedule(T=1000))
```

Precomputa uma vez, recolha por índice durante o treinamento e a amostragem.

### Passo 2: Difusão avançada (qu_sampla)

```python
def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise
```

Formulário fechado de uma linha. `t`é um lote de etapas temporais, uma por imagem no lote.

### Passo 3: Uma pequena rede U-Net com condição de tempo

```python
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=32, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)
```

U-Net de dois níveis com condicionamento de tempo injetado no gargalo de engarrafamento.

### Passo 4: Localização

```python
def train_step(model, x0, schedule, optimizer, device, T=1000):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.randint(0, T, (bs,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

Não há jogo GAN, nenhuma perda especializada, uma chamada MSE.

### Passo 5: Prótese de amostragem (DDPM)

```python
@torch.no_grad()
def sample(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

1000 passes para a frente para produzir um lote de amostras.

### Passo 6: amostragem DDIM (determinista, ~ 20 vezes mais rápida)

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = ts[i]
        t_prev = ts[i + 1]
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x
```

`eta=0`é totalmente determinista (a mesma entrada de ruído produz sempre a mesma saída). `eta=1`recupera a DDPM.

## Usá-lo

Para o trabalho de produção, utiliza-se `diffusers`- Não .

```python
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2)
scheduler = DDPMScheduler(num_train_timesteps=1000)
```

A biblioteca envia agendadores prontos (DDPM, DDIM, DPM-Solver, Euler, Heun), U-Nets configuráveis, canais para texto-a-imagem e imagem-a-imagem e auxiliares de ajuste fino LoRA.

Para pesquisas,`k-diffusion`(Katherine Crowson) tem as implementações de referência mais fiéis e as melhores variantes de amostragem.

## Envia-o

Esta lição produz:

- `outputs/prompt-diffusion-sampler-picker.md` um prompt que seleciona DDPM / DDIM / DPM-Solver / Euler com base no objetivo de qualidade, orçamento de latência e tipo de condicionamento.
- `outputs/skill-noise-schedule-designer.md` uma habilidade que produz um cronograma beta linear, cosino ou sigmoide dado T e nível de corrupção alvo, mais gráficos diagnósticos de relação sinal-ruído ao longo do tempo.

## Exercícios

1. **(Easy)**Visualize o processo avançado: tome uma imagem e trace `x_t`- Não .`t in [0, 100, 250, 500, 750, 1000]`Verifica isso .`x_1000`Parece um ruído gaussiano puro.
2. **(Medium)**Treinar o TinyUNet no conjunto de dados de círculos sintéticos durante 20 épocas e amostrar 16 círculos. Comparar a amostragem DDPM (1000 passos) e DDIM (50 passos)  produzem imagens semelhantes a partir da mesma semente de ruído?
3. **(Hard)**Implementar um cronograma de ruído cosinário (Nichol & Dhariwal, 2021): `alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`- Treinar o mesmo modelo com cronogramas lineares e cosinosos e mostrar que o cosino dá melhores amostras em baixos números de passos.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Forward process | "Add noise over time" | Fixed Markov chain that corrupts an image into Gaussian noise over $T$ steps |
| Reverse process | "Denoise step by step" | Learned distribution that walks back from noise to image |
| Epsilon prediction | "Predict the noise" | The training target: `epsilon_theta(x_t, t)` predicts the noise added at step $t$ |
| Beta schedule | "Noise amounts" | Sequence of $T$ small variances that define how much noise enters per step |
| $\bar{\alpha}_t$ | "Cumulative retain factor" | Product of $(1 - \beta_s)$ up to time $t$; bigger $t$ means less signal left |
| DDPM sampler | "Ancestral, stochastic" | Samples each $x_{t-1}$ from its conditional Gaussian; 1000 steps |
| DDIM sampler | "Deterministic, fast" | Rewrites sampling as a deterministic ODE; 20-100 steps with similar quality |
| Time conditioning | "Tell the model which t" | Sinusoidal embedding of $t$ injected into the U-Net so it knows the noise level |

## Mais leitura

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) o papel que fez a difusão prática e superou os GANs na FID
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) calendário cosínico e parâmetria v
- [DDIM (Song, Meng, Ermon, 2020)](https://arxiv.org/abs/2010.02502) o amostragem determinista que possibilitou a inferência em tempo real
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) uma visão unificada de cada escolha de projeto de difusão; melhor referência atual
