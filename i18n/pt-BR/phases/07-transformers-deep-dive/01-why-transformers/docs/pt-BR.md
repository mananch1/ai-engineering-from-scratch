# Por que os transformadores  Os problemas com RNNs

> As RNNs processam tokens uma a uma. Os transformadores processam todos os tokens de uma só vez. Essa única aposta arquitetônica mudou cada curva de escalagem no deep learning após 2017.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## O problema

Antes de 2017, cada modelo de sequência de última geração no planeta  língua, tradução, fala  era uma rede neural recorrente. LSTMs e GRUs ganharam benchmarks de tradução equivalentes à ImageNet por meio década. Eles eram a única ferramenta que alguém tinha.

Os cálculos sequenciais significavam que não se podia paralelalizar ao longo do eixo do tempo:`t+1`Precisa do estado oculto do token .`t`Uma sequência de 1.024 tokens significava 1.024 passos em série em uma GPU que pode fazer 1.000.000 de operações de pontos flutuantes por ciclo.

Os gradientes desaparecidos significaram que a informação 50 tokens atrás já estava comprimida através de 50 não-lineares. Unidades recorrentes de gate (LSTM, GRU) suavizaram a esmagamento, mas nunca o eliminaram. Dependências de longo alcance  "o livro que li no verão passado em um avião para Quioto foi..."  rotineiramente falhou.

Os estados ocultos de largura fixa significavam que o codificador apertou toda a sequência de fonte em um único vetor antes que o decodificador visse qualquer coisa. Não importa se a fonte é de 5 tokens ou 500; o gargalo de engarrafamento é a mesma forma.

O artigo de 2017 "Attenção é tudo que você precisa" propôs algo radical: deixar cair a recorrência inteiramente. Deixe cada posição atender a todas as outras posições em paralelo. Treine em uma grande multiplicação de matriz em vez de 1.024 sequenciais.

O resultado domina todas as modalidades até 2026. Língua (GPT-5, Claude 4, Llama 4), visão (ViT, DINOv2, SAM 3), áudio (Whisper), biologia (AlphaFold 3), robótica (RT-2).

## O conceito

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck.**Um RNN computa $h_t = f(h_{t-1}, x_t)$Cada passo depende do anterior.$h_5$Antes de$h_4$Em GPUs modernas com mais de 10.000 núcleos paralelos, isso desperdiça 99% do silício numa longa sequência.

**Attention as a broadcast.**Computações de auto-atenção .`output_i = sum_j(a_ij * v_j)`Para cada par .$(i, j)$A matriz de atenção N×N completa um matmul em lote.

**The speedup is not a constant.**É a diferença entre $O(N)$profundidade serial e $O(1)$Na prática, os transformadores treinam 510x mais rápido por época em hardware correspondente em N=512, e a lacuna se amplia com o comprimento da sequência até atingir o $O(N^2)$parede de memória da atenção (que a Flash Attention mais tarde fixou  ver lição 12).

**What transformers cost.**Escalas de memória da atenção como $O(N^2)$Para o contexto de 2K, bem. Para o contexto de 128K, você precisa de janelas deslizantes, extrapolação de RoPE, azulejos de atenção flash, ou variantes de atenção linear.$O(N)$transformadores trocam tempo por memória e depois ganham o tempo de volta através do paralelismo.

**The inductive bias shift.**Os transformadores assumem localidade e recência. Os transformadores não assumem nada. Cada par é um candidato à atenção. É por isso que os transformadores precisam de mais dados para treinar bem, mas escalar mais quando tiverem. Chinchilla (2022) formalizou isso: dado tokens suficientes, um transformador sempre bate um RNN de igual número de parâmetros.

```figure
rnn-vs-parallel
```

## Construí-lo

Não há rede neural aqui Simula-se o gargalo do núcleo numéricamente para que você sinta a lacuna no seu laptop.

### Passo 1: medir a profundidade serial

Veja .`code/main.py`- Construímos duas funções. Uma codifica uma sequência como uma cadeia de adições (serial, como um RNN).

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Temos tempo em sequências até 100.000 elementos.$O(N)$Mesmo em Python puro, a redução de estilo de atenção bate-o em comprimento ≥ 1.000 porque Python `sum()`é implementado em C e reitera sem custos superiores de intérprete por passo.

### Passo 2: contar as operações teóricas

Os dois algoritmos adicionam N. A diferença é * profundidade de dependência *: quantas operações devem acontecer sequencialmente antes que a próxima possa começar. RNN profundidade = N. profundidade de atenção = log(N) com uma redução de árvore, ou 1 com uma varredura paralela. profundidade, não op contagem, decide o tempo da GPU.

### Passo 3: Escalação empírica em sequências longas

Imprimimos uma tabela de cronometragem que faz o $O(N)$Em um laptop Mac de 2026, as sequências abaixo de 1.000 elementos são muito rápidas para medir. Sequências de 100.000 mostram uma varredura linear limpa. Escale isso para um transformador de 16.384 tokens com um equivalente LSTM de 12 camadas e você vê por que o treinamento de relógio de parede foi um bloqueador em 2016.

## Usá-lo

Quando ainda escolher um RNN em 2026:

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

Modelos de espaço-estado (SSM) como Mamba são essencialmente RNNs com parametrização estruturada que lhes dá o melhor de ambos: $O(N)$A maioria dos laboratórios de fronteira treina modelos híbridos de transformadores SSM+ (por exemplo, Jamba, Samba)  recorrência não está morta, é um componente.

## Envia-o

Veja .`outputs/skill-architecture-picker.md`A competência escolhe uma arquitetura para um novo problema de sequência dada a comprimento, o rendimento e as restrições orçamentais de treinamento.

## Exercícios

1. **Easy.**- Toma .`rnn_style`de`code/main.py`E substituir o estado oculto escalar por um vector de comprimento-64 de estados ocultos.
2. **Medium.**Implementar um prefixo paralelo-suma (Hillis-Steele scan) em Python puro. Verifique que produz a mesma saída numérica que uma varredura em série em comprimento 1024.
3. **Hard.**Portar a redução de estilo de atenção para PyTorch na GPU. Tempo tanto como você varrer o comprimento da sequência de 64 para 65.536.

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
| Attention | "Let tokens look at each other" | Weighted sum $\sum_j a_{ij} v_j$ where $a_{ij}$ comes from a similarity score between positions $i$ and $j$. |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = $O(N^2)$ in sequence length; Flash Attention hides the constants, not the scaling. |

## Mais leitura

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762)O artigo que matou a recorrência na PNL convencional.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)- onde nasceu a atenção, ligada a um RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) o papel original do LSTM, para o registro.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) Resposta recorrente moderna aos transformadores.
