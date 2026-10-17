# Atenção de várias cabeças

> Uma cabeça de atenção aprende uma relação por vez, oito cabeças aprendem oito cabeças são livres, pegue mais delas.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## O problema

Uma única cabeça de auto-atenção calcula uma matriz de atenção. Essa matriz capta um tipo de relação  geralmente aquela que minimiza a perda em qualquer sinal de treinamento. Se os seus dados têm acordo sujeito-verbo, co-referência, discurso de longo alcance e fragmentos sintáticos todos emaranhados juntos, uma única cabeça os esmagia em uma única distribuição de max soft e perde metade do sinal.

A correção do artigo Vaswani de 2017: executar várias funções de atenção em paralelo, cada uma com suas próprias projeções Q, K, V, e concatenar as saídas.$d_{\text{model}} / n_{\text{heads}}$Os parâmetros totais permanecem iguais.

A atenção multi-cabeça é o padrão de cada transformador em 2026 navios com. O único argumento é sobre * quantas cabeças * e se as chaves e valores compartilham projeções (Attenção de Queria Grupada, Attenção de Queria Multi, Attenção Latente de Multi-cabeça).

## O conceito

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.**- Toma .`X`de forma`(N, d_model)`Projeto para Q, K, V de cada forma`(N, d_model)`- Refazer para`(N, n_heads, d_head)`onde$d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$Transpor para`(n_heads, N, d_head)`- Não .

**Attend in parallel.**Execute uma escala de atenção de ponto-produto dentro de cada cabeça.`(N, d_head)`As cabeças operam em diferentes subespaços da incorporação e nunca falam durante o próprio cálculo da atenção.

**Concatenate and project.**- A cabeça de pila volta para o`(N, d_model)`e multiplicar por uma matriz de saída aprendida `W_o`de forma`(d_model, d_model)`- Não .`W_o`É onde as cabeças se misturam.

**Why it works.**Cada cabeça pode se especializar sem competir com os outros para o orçamento representativo. Estudos de pesquisa de 20192024 mostram papéis distintos de cabeça: cabeças posicionais, cabeça que atende ao token anterior, cabeças de cópia, cabeças de entidade nomeada, cabeças de indução (que são a base da aprendizagem no contexto).

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA é o padrão moderno porque reduz a memória de cache KV por um fator de $N/G$MLA vai mais longe comprimindo K/V em um espaço latente, e depois projetando de volta no tempo de computação  custa FLOPs, economiza muito mais memória.

```figure
multihead-split
```

## Construí-lo

### Passo 1: separar cabeças da atenção de cabeça única que já temos

Leva o .`SelfAttention`A partir da lição 02 e envolver com um par de divisão/conca.`code/main.py`para uma implementação numpica; a lógica é:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Um reformula e outro transponha, sem ciclo, é exatamente o que o PyTorch faz sob`nn.MultiheadAttention`- Não .

### Passo 2: executar escala-pontos-produto atenção por cabeça

Cada cabeça recebe sua própria fatia de Q, K, V. A atenção torna-se um matmul em lote:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Em hardware real .`Qh @ Kh.transpose(...)`É um .`bmm`A GPU vê um único batch de forma .`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`Adicionar cabeças é livre.

### Passo 3: Variante de atenção de perguntas agrupadas

Só as projeções de chave e valor mudam.`n_heads`grupos; K e V obtêm`n_kv_heads < n_heads`grupos e são repetidas para corresponderem:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Em inferência , isso economiza memória porque só`n_kv_heads`cópias vivem no cache KV, não `n_heads`Llama 3 70B usa 64 cabeças de consulta com 8 cabeças de KV  um encolher de cache 8x.

### Passo 4: Analise o que cada cabeça aprendeu

Exerça a MHA numa frase curta com 4 cabeças.`(N, N)`Você verá diferentes cabeças escolher diferentes estruturas mesmo com inicialização aleatória que é parcialmente sinal, parcialmente simetria de rotação nos subespaços.

## Usá-lo

Na PyTorch, a versão de uma linha:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA em relação ao PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?**Regras práticas dos modelos de produção em 2026:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`Quase sempre cai em 64 ou 128. É a unidade de quanto uma cabeça pode "ver".$\sqrt{d_{\text{head}}}$• se ultrapassar o 256, perderá o benefício de "muitos especialistas pequenos".

## Envia-o

Veja .`outputs/skill-mha-configurator.md`. A competência recomenda o número de cabeças, o número de cabeças kv e a estratégia de projecção para um novo transformador, dado o orçamento de parâmetros, o comprimento da sequência e o objetivo de implantação.

## Exercícios

1. **Easy.**Tome o MHA de `code/main.py`e mudança .`n_heads`de 1 a 16 com `d_model=64`A perda de um modelo de uma camada pequena em uma tarefa de cópia sintética.
2. **Medium.**Implementar MQA (um cabeçalho KV compartilhado em todas as cabeçalhas de consulta). Medir a quantidade de parâmetros que caem em conta versus MHA completa.
3. **Hard.**Implementar uma versão pequena de Atenção Latente Multi-Cé: comprimir K, V para um rank-`r`O que é que é que é o que é que é o que é que é o que é que é o que é que é o que é que é o que é que é o que é que é o que é que é o que é que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é o que é?`r`A memória cache passa abaixo de 1/8 da MHA completa enquanto a qualidade permanece dentro de 1 bit da validação?

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$ with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## Mais leitura

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) a especificação original de cabeça múltipla.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) o documento da MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) como converter o MHA em GQA após o treino.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) MLA e por que é melhor do que MHA/GQA na memória cache.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) olhar mecanicista para o que as cabeças realmente fazem.
