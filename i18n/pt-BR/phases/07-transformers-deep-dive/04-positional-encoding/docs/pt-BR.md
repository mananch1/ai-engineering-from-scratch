# Encodificação de posição  Sinusoidal, RoPE, ALiBi

> A atenção é invariavel em permutação. "O gato sentou no tapete" e "mat o gato sat no tapete" produzem a mesma saída sem sinal posicional. Três algoritmos o corrigem  cada um com uma aposta diferente sobre o que significa "posição".

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## O problema

A atenção do produto de ponto é cega à ordem.`softmax(Q K^T / √d) V`É calculado a partir de semelhanças em pares.`X`Nada dentro da atenção se importa com a posição.

Não é um erro num modelo de saco de palavras. Para linguagem, código, áudio, vídeo  qualquer coisa onde a ordem carrega significado  é fatal.

A solução é injetar posição nos embeddings de alguma forma.

1. **Absolute sinusoidal**(Vaswani 2017). Adicionar `sin/cos`Simples, livres de aprendizagem, extrapolam mal além dos comprimentos treinados.
2. **RoPE — Rotary Position Embeddings**(Su 2021). Rotar os vetores Q e K por um ângulo proporcional à posição. Encode * posição relativa * diretamente no produto de pontos. Dominant em 2026.
3. **ALiBi — Attention with Linear Biases**(Presião 2022). Salte integrados inteiramente; adicione uma penalidade linear por cabeça às pontuações de atenção com base na distância. Excelente extrapolação de comprimento.

A partir de 2026, praticamente todos os modelos abertos de fronteira usam RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi.

## O conceito

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### A posição de posição é igual a:

Pre-computação de uma matriz fixa `PE`de forma`(max_len, d_model)`- Não .

```math
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(pos / 10000^{2i / d_{\text{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(pos / 10000^{2i / d_{\text{model}}}\right)
\end{aligned}
```

Então ...`X' = X + PE[:N]`Cada dimensão é um sinusoide em uma frequência diferente.`max_len`O modelo não tinha conhecimento do que acontece na posição 2048 quando só viu posições 02047.

### RoPE

Rotar os vetores Q e K (não incorporados). Para um par de dimensões $(2i, 2i+1)$- Não .

```math
\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\ \sin(pos \cdot \theta_i) & \cos(pos \cdot \theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
\qquad \theta_i = \text{base}^{-2i / d_{\text{head}}}, \quad \text{base} = 10000 \text{ by default}
```

Aplicar a mesma rotação às teclas com posição `pos_k`O produto de pontos$q'_m \cdot k'_n$torna-se uma função de $(m - n)$- Só a mim.**the attention score depends only on the relative distance**- É um truque bonito.

Extensão do RoPE: `base`O Llama 3 foi ampliado de 8K para 128K de tal forma.

### A.L.B.

Esqueça o truque de inserção.

```math
\text{attn\_score}[i, j] = \frac{q_i \cdot k_j}{\sqrt{d}} - m_h \cdot |i - j|
```

Onde ?$m_h$é uma inclinação específica da cabeça (por exemplo $1 / 2^{8h/H}$O documento mostra que a extrapolação de comprimento supera o sinusoidal e corresponde ao RoPE em seu comprimento treinado original.

### O que escolher em 2026

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

O RoPE ganhou porque chama a atenção sem alterar a arquitetura, codifica a posição relativa e a sua`base`O hiperparâmetro dá um botão limpo para ajuste fino de longo contexto.

```figure
rope-explorer
```

## Construí-lo

### Passo 1: codificação sinusoidal

Veja .`code/main.py`Um cálculo de 4 linhas:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Adicione isto à matriz de incorporação antes da primeira camada de atenção.

### Passo 2: RoPE aplicado a Q, K

O RoPE opera no local em Q e K. Para cada par de dimmers:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Crucial: aplicar a mesma função a Q na posição $m$E K na posição $n$O produto deles pega um .$\cos((m-n) \cdot \theta_i)$A atenção aprende a posição relativa de graça.

### Passo 3: Pistas e preconceitos do ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Adicionar`bias[h]`- O que é ?`(seq_len, seq_len)`Matriz de pontuação de atenção da cabeça `h`, depois softmax.

### Passo 4: verificar a propriedade relativa de distância do RoPE

Escolha dois vetores aleatórios .`a, b`- Vira por .`(pos_a, pos_b)`- Então , por ...`(pos_a + k, pos_b + k)`. Ambos os produtos de pontos devem corresponder no erro de ponto flutuante. Essa propriedade é o ponto inteiro de RoPE  é invariante ao deslocamento absoluto, só a diferença relativa é importante.

## Usá-lo

PyTorch 2.5+ embarca serviços de RoPE em `torch.nn.functional`A maioria dos códigos de produção usa`flash_attn`ou `xformers`onde o RoPE é aplicado dentro do núcleo de atenção.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.**Redescálculos`base`- Não .`base * (scale_factor)^(d/(d-2))`quando se estende de 4K a 16K+.
- **YaRN.**Interpolação mais inteligente que preserva a entropia da atenção em contextos longos.
- **LongRoPE.**O método 2024 da Microsoft que usa a pesquisa evolutiva para escolher fatores de escala por dimensão.
- **Position interpolation + fine-tuning.**Apenas reduzir as posições pelo fator de extensão e ajustar para tokens de 15B. Surpreendentemente eficaz.

## Envia-o

Veja .`outputs/skill-positional-encoding-picker.md`. A habilidade escolhe uma estratégia de codificação para um novo modelo, dada a extensão do contexto-alvo, as necessidades de extrapolação e o orçamento de formação.

## Exercícios

1. **Easy.**Traçar o sinusoidal `PE`Matrix como mapa de calor para `max_len=512, d=128`Confirmar o padrão "as tiras ficam mais largas à medida que o índice de dimensões cresce".
2. **Medium.**Implementar a escalação do RoPE com NTK. Treinar um pequeno LM em sequências de comprimento 256, depois testar no comprimento 1024 com e sem escalação. Medir a perplexidade.
3. **Hard.**Implementar ALiBi e RoPE no mesmo módulo de atenção. Treinar um transformador de 4 camadas em uma tarefa de cópia com sequências de comprimento 512. Extrapolar para 2048 no momento do teste. Comparar degradação.

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position. |
| Sinusoidal | "The original one" | `sin/cos` at geometric frequencies added to embeddings; doesn't extrapolate. |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance. |
| ALiBi | "Linear bias trick" | Add $-m \cdot \lvert i-j \rvert$ to attention scores; no embedding needed, great extrapolation. |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context at inference. |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed when context expands. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation that preserves attention entropy. |
| Extrapolation | "Works beyond trained length" | Can the position scheme serve correct output past `max_len` seen in training? |

## Mais leitura

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762)- Sinusoidal original.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)Papel RoPE.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)- Alibi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) Estado de ponta de escalagem RoPE.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) O artigo de longo contexto Llama 2 do Meta.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) o método Microsoft utilizado pela Phi-3-Long e citado na secção Use It.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) Implementações em nível de produção de cada esquema de escalagem de RoPE (default, linear, dinâmico, YaRN, LongRoPE, Llama-3).
