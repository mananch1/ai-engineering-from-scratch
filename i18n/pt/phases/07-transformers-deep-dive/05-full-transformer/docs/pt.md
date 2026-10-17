# O Transformador Completo  Encoder + Decodificador

> A atenção é a estrela. Tudo o resto, resíduos, normalização, alimentação, atenção cruzada, é o andaime que permite que você a apile profundamente.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## O problema

Uma camada de atenção única é um extractor de características, não um modelo. Um matmul por camada não é capacidade suficiente para a linguagem. Você precisa de profundidade  e rupturas de profundidade sem a canalização certa.

O documento Vaswani de 2017 contou com seis decisões de design que transformaram uma camada de atenção em um bloco empilhável. Cada transformador desde  encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5)  herda o mesmo esqueleto. Em 2026 os blocos foram refinados (RMSNorm, SwiGLU, pre-norma, RoPE), mas o esqueleto é idêntico.

Esta lição é o esqueleto. As seguintes lições especializam-no  06 para codificadores, 07 para decodificadores, 08 para codificador-decodificador.

## O conceito

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### As seis peças

1. **Embedding + positional signal.**Tokens → vectores. posição injetada através de RoPE (moderno) ou sinusoidal (clássico).
2. **Self-attention.**Cada posição atende a cada outra, mascarada em decodificadores.
3. **Feed-forward network (FFN).**MLP de duas camadas em termos de posição: `W_2 · activation(W_1 · x)`- Proporção de expansão 4x por padrão.
4. **Residual connection.** `x + sublayer(x)`Sem isto, os gradientes desaparecem depois de 6 camadas.
5. **Layer normalization.** `LayerNorm`ou `RMSNorm`Estabiliza o fluxo residual.
6. **Cross-attention (decoder only).**As consultas vêm do decodificador, chaves e valores da saída do encodificador.

Observe um fluxo de vetor através de um bloco: a atenção mistura-se entre as posições, o residual leva-o para a frente, o FFN transforma-o, e a norma mantém o fluxo estável.

```figure
transformer-block
```

### Bloco de codificação (utilizado pelo codificador BERT, T5)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

O codificador é bidirecional, não há mascaragem, todas as posições veem todas as posições.

### Bloco de decodificador (usado pelo decodificador GPT, T5)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

O decodificador tem três subcamadas por bloco. O centro  atenção cruzada  é o único lugar onde as informações fluem de um codificador para um decodificador. Em uma arquitetura pura de apenas decodificador (GPT), a atenção cruzada é omitida e você apenas tem a auto-atenção mascarada + FFN.

### Pre-norma versus pós-norma

Papel original: `x + sublayer(LN(x))`- Não .`LN(x + sublayer(x))`O que é mais difícil de treinar profundamente sem um aquecimento cuidadoso.`LN`* antes de * subcamada) é o padrão 2026: Llama, Qwen, GPT-3+, Mistral todos usam.

### O bloco modernizado de 2026

Vaswani 2017 enviou LayerNorm + ReLU. As pilhas modernas substituíram ambas.

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

O RMSNorm diminui a centriação média do LayerNorm (uma subtracção menor), o que economiza a computação e é empíricamente pelo menos tão estável.`Swish(W1 x) ⊙ W3 x`) supera consistentemente o FFN ReLU/GELU em ~0,5 pontos na publicação Llama, PaLM e Qwen.

### Contagem de parâmetros

Por um quarteirão com `d_model = d`e expansão do FFN `r`- Não .

- MHA: `4 · d²`(Projeções Q, K, V, O)
- FFN (SwiGLU): `3 · d · (r · d)`- Não .`3rd²`
- Normas: insignificantes

- Não .`d = 4096, r = 2.6, layers = 32`(aproximadamente Llama 3 8B), total: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`(mais inserções e cabeçalho).

## Construí-lo

### Passo 1: os blocos de construção

Usando o pequeno`Matrix`classe da lição 03 (copiada para este arquivo para independência):

- `layer_norm(x, eps=1e-5)`Subtrair a média, dividir por std.
- `rms_norm(x, eps=1e-6)` Divida por RMS. Sem subtracção média.
- `gelu(x)`E ...`silu(x) * W3 x`- Não.
- `ffn_swiglu(x, W1, W2, W3)`- Não .
- `encoder_block(x, params)`E ...`decoder_block(x, enc_out, params)`- Não .

Veja .`code/main.py`Para o cabo completo.

### Passo 2: conectar um codificador de 2 camadas e um decodificador de 2 camadas

Aponte-os, passe a saída do codificador em cada atenção transversal do decodificador, adicione um LN final antes da projeção de saída.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Passo 3: avançar em um exemplo de brinquedo

Passe uma fonte de 6 tokens e um alvo de 5 tokens. Verifique a forma de saída é `(5, vocab)`Esta lição é sobre arquitetura, não sobre perda.

### Passo 4: troca em RMSNorm + SwiGLU

Substitua LayerNorm e ReLU-FFN por RMSNorm e SwiGLU. Confirme que as formas ainda coincidem. Esta é a modernização de 2026 com uma substituição de função.

## Usá-lo

As implementações de referência PyTorch/TF: `nn.TransformerEncoderLayer`- Não .`nn.TransformerDecoderLayer`Mas a maioria dos códigos de produção 2026 faz o seu próprio bloco porque:

- A atenção flash é chamada para dentro da atenção, não através de`nn.MultiheadAttention`- Não .
- GQA / MLA não estão na referência stdlib.
- RoPE, RMSNorm, SwiGLU não são as configurações padrão do PyTorch.

HF `transformers`tem blocos de referência limpos que deve ler: `modeling_llama.py`É o bloco canônico 2026 apenas para decodificadores.

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

O código-descodificador é ainda melhor quando a entrada tem uma identidade clara de "seqüência de fonte" (tradução, reconhecimento de voz, tarefas estruturadas).

## Envia-o

Veja .`outputs/skill-transformer-block-reviewer.md`. A competência revisa a implementação de um novo bloco de transformador contra as anomalias de 2026 e indica as peças faltantes (pre-norma, RoPE, RMSNorm, GQA, FFN ratio de expansão).

## Exercícios

1. **Easy.**Conte os parâmetros no seu bloco de codificação em `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Valida através da implementação do bloco e usando `sum(p.numel() for p in block.parameters())`- Não .
2. **Medium.**Passe de pós-norma para pré-norma. Iniciar ambos e medir a norma de ativação após 12 camadas empilhadas em entrada aleatória.
3. **Hard.**Implementar um codificador-decodificador de 4 camadas em uma tarefa de cópia de brinquedo (cópia `x`A taxa de perda de RMSNorm + SwiGLU + RoPE  diminui?

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks. |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same empirical stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model, usually 4 (LayerNorm) or 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers; slight ppl improvement, smaller model. |

## Mais leitura

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) especificação original do bloco.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) por que a pré-norma bate profundamente a pós-norma.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)O papel SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) bloco canônico 2026 apenas para decodificadores.
