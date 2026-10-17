# Embedings de Word  Word2Vec a partir do zero

> Uma palavra é a companhia que mantém, e se treinar uma rede superficial sobre essa ideia, a geometria cai.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## O problema

O TF-IDF sabe .`dog`E ...`puppy`O que é que é preciso para que o sistema de classificação seja mais eficaz?`dog`Não pode generalizar-se para uma revisão sobre `puppy`Pode-se documentar isto listando sinônimos, mas isso falha em termos raros, jargão de domínio e em todas as línguas que não se anteciparam.

Queres uma representação onde`dog`E ...`puppy`Terras próximas no espaço.`king - man + woman`terras próximas`queen`Onde um modelo treinou em`dog`Transfere algum sinal para `puppy`- Não.

O Word2Vec deu-nos esse espaço. Rede neural de duas camadas, trilhões de tokens de treinamento, publicado em 2013. A arquitetura é quase embaraçosamente simples. Os resultados remodelaram a PNL durante uma década.

## O conceito

**Distributional hypothesis**(Primeiro, 1957): "Você saberá uma palavra pela companhia que ela mantém". Se duas palavras aparecem em contextos semelhantes, elas provavelmente significam coisas semelhantes.

Word2Vec vem em dois sabores, ambos explorando essa ideia.

- **Skip-gram.**Dado uma palavra central, prevê as palavras circundantes. `cat -> (the, sat, on)`com tamanho de janela 2.
- **CBOW (continuous bag of words).**Dadas as palavras circundantes, prevê o centro.`(the, sat, on) -> cat`- Não .

O Skip-gram é mais lento para treinar, mas lida melhor com palavras raras.

A rede tem uma camada oculta sem não linearidade. A entrada é um vetor de um só calor sobre o vocabulário. A saída é um softmax sobre o vocabulário. Depois do treinamento, você joga fora a camada de saída. Os pesos das camadas ocultas são os embebimentos.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

O truque: a softmax superior a 100 mil palavras é proibitivamente cara.**negative sampling**Previnir "se essa palavra contextual aparece perto desta palavra central, sim ou não". Amostra um punhado de palavras negativas (não co-ocorrentes) por par de treinamento em vez de calcular softmax sobre todo o vocabulário.

```figure
word-vector-arithmetic
```

## Construí-lo

### Passo 1: par de treinamento a partir de um corpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Cada par (centro, contexto) numa janela é um exemplo positivo de treinamento.

### Passo 2: inserção de tabelas

Duas matrizes.`W`é a tabela de inserção de palavras centrais (a que você mantém). `W'`É a tabela de palavras contextuais (muitas vezes descartada, às vezes mediada com `W`)).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

O tamanho do vocabulário 10k e dim 100 é realista; para ensino, 50 vocabulários x 16 dim são suficientes para ver a geometria.

### Passo 3: objetivo negativo de amostragem

Para cada par positivo `(center, context)`, amostra `k`Exercite o modelo para que o produto ponto`W[center] · W'[context]`É alto para os positivos e baixo para os negativos.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

A fórmula mágica: perda logística em pares positivos (quer sigmoide perto de 1) mais perda logística em pares negativos (quer sigmoide perto de 0). Gradientes fluem para ambas as tabelas.

### Passo 4: treinar em um corpo de brinquedo

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

Depois de épocas suficientes em um grande corpus, palavras que compartilham contextos têm embutições centrais semelhantes. Em um corpus de brinquedos, você vê o efeito de forma fraca. Em bilhões de tokens, você vê dramaticamente.

### Passo 5: o truque de analogia

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

Em vetores pré-treinados de 300d de Google News:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`Não porque o modelo saiba o que é a realeza, porque o vetor...`(king - man)`Captura algo como "royal", e adicionando-o a `woman`terras perto da região real-fêmea.

## Usá-lo

Escrever Word2Vec a partir do zero é ensinar.`gensim`- Não .

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Para o trabalho real, quase nunca treinas Word2Vec sozinho.

- **GloVe** A abordagem de factorizamento de matriz de co-ocorrência de Stanford. 50d, 100d, 200d, 300d pontos de controlo. Boa cobertura geral. A lição 04 abrange especificamente o GloVe.
- **fastText** A extensão Word2Vec do Facebook que incorpora caracteres n-gramas.
- **Pretrained Word2Vec on Google News** 300d, vocabulário de palavras 3M, publicado em 2013. Ainda sendo baixado diariamente.

### Quando o Word2Vec ainda vencer em 2026

- Treinar resumos médicos em uma hora num laptop, obter vetores especializados sem captura de modelos gerais.
- Engenharia de recursos em estilo analógico. `gender_vector = mean(man - woman pairs)`Subtrai-o de outras palavras para obter um eixo neutro de gênero.
- Interpretabilidade. 100d é pequeno o suficiente para traçar através de PCA ou t-SNE e realmente ver aglomerados forma.
- Qualquer lugar que seja, a inferência tem que ser executada no dispositivo sem GPU.

### Onde o Word2Vec falha

A parede policémica.`bank`tem um vetor.`river bank`E ...`financial bank`Partilha.`table`Um classificador para baixo não pode distinguir os sentidos do vetor.

Embedings contextuais (ELMo, BERT, todos os transformadores desde então) resolveram isso produzindo um vetor diferente para cada ocorrência da palavra com base no contexto circundante.

O problema da falta de vocabulário é o outro fracasso.`Zoomer-approved`O fastText corrige isto com a composição de subpalavras (leção 04).

## Envia-o

Salva como`outputs/skill-embedding-probe.md`- Não .

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Exercícios

1. **Easy.**Exerce o ciclo de treinamento num pequeno corpo (20 frases sobre gatos e cães).`nearest(vocab, W, W[vocab["cat"]])`Retorno `dog`Se não, aumenta os tempos ou o vocabulário.
2. **Medium.**Adicionar sub-esampulação de palavras frequentes.`10^-5`Os resultados da análise de dados são apresentados em pares de formação com probabilidade proporcional à sua frequência.
3. **Hard.**Treinar um modelo no corpus de 20 Newsgroups.`he - she`E ...`doctor - nurse`- Projeto de palavras de ocupação em ambos os eixos. Relata quais ocupações têm a maior lacuna de viés. Este é o tipo de investigação de equidade da sonda usada pelos pesquisadores.

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | A dense, low-dim (typically 100-300) representation learned from context. |
| Skip-gram | Word2Vec trick | Predict context words from center word. Slower than CBOW, better for rare words. |
| Negative sampling | Training shortcut | Replace softmax over full vocab with binary classification against `k` random words. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-sensitive vector | Different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | Word not seen in training. Word2Vec cannot produce a vector for these. |

## Mais leitura

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)O papel de amostragem negativa é curto e legível.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) a derivação mais clara dos gradientes, se a matemática do papel original se sentir densa.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) configurações de formação de produção que realmente funcionam.
