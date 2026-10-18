# Word Embeddings  Word2Vec từ đầu

> Một từ là công ty mà nó giữ. Hãy tập trung vào ý tưởng đó và hình học sẽ rơi ra.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## Vấn đề

TF-IDF biết `dog`và `puppy`là những từ khác nhau. nó không biết chúng có nghĩa gần như cùng một điều.`dog`không thể nói chung đến một đánh giá về `puppy`Bạn có thể viết qua nó bằng cách liệt kê các từ đồng nghĩa, nhưng điều đó không có trong các thuật ngữ hiếm, thuật ngữ miền, và mọi ngôn ngữ bạn không dự đoán.

Anh muốn một đại diện ở đâu?`dog`và `puppy`- Không gian gần nhau.`king - man + woman`Đất gần đó`queen`Một người mẫu được đào tạo`dog`chuyển một số tín hiệu đến `puppy`miễn phí.

Word2Vec đã cho chúng ta không gian đó. Mạng lưới thần kinh hai lớp, hàng nghìn tỷ token được phát hành vào năm 2013. Kiến trúc này gần như đơn giản đáng xấu hổ. Kết quả đã định hình lại NLP trong một thập kỷ.

## Khái niệm

**Distributional hypothesis**(Tất cả hai từ này được viết trong năm 1957): "Bạn sẽ biết một từ từ từ những người mà nó giữ".

Word2Vec có hai hương vị, cả hai đều khai thác ý tưởng đó.

- **Skip-gram.**Với một từ trung tâm, hãy dự đoán những từ xung quanh. `cat -> (the, sat, on)`với kích thước cửa sổ 2.
- **CBOW (continuous bag of words).**Với những từ xung quanh, hãy dự đoán trung tâm.`(the, sat, on) -> cat`- Tôi không biết.

Skip-gram là chậm hơn để đào tạo nhưng xử lý từ hiếm hơn.

Mạng lưới có một lớp ẩn mà không có tính không tuyến tính. Input là một vector nóng trên từ vựng. Output là một softmax trên từ vựng. Sau khi đào tạo, bạn ném ra lớp xuất.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Trù: Softmax trên 100k từ là quá đắt tiền. Word2Vec sử dụng **negative sampling**để biến nó thành một nhiệm vụ phân loại nhị phân. Dự đoán "có bao nhiêu từ ngữ trong ngữ cảnh xuất hiện gần từ trung tâm này, có hay không".

```figure
word-vector-arithmetic
```

## Hãy xây dựng nó

### Bước 1: các cặp đào tạo từ một corpus

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

Mỗi cặp (trung tâm, ngữ cảnh) trong cửa sổ là một ví dụ đào tạo tích cực.

### Bước 2: nhúng bảng

Hai cái.`W`là bảng nhúng từ trung tâm (một bạn giữ). `W'`là bảng từ ngữ ngữ cảnh (thường bị loại bỏ, đôi khi trung bình với `W`().

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Từ vựng kích thước 10k và dim 100 là thực tế; cho việc giảng dạy, 50 từ vựng x 16 dim là đủ để xem hình học.

### Bước 3: mục tiêu lấy mẫu tiêu cực

Đối với mỗi cặp tích cực `(center, context)`, mẫu `k`Từ ngẫu nhiên từ từ vựng như âm.`W[center] · W'[context]`là cao cho tích cực và thấp cho tiêu cực.

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

Công thức ma thuật: mất hậu cần trên cặp dương (nhiều cần sigmoid gần 1) cộng với mất hậu cần trên cặp âm (nhiều cần sigmoid gần 0).

### Bước 4: tập luyện trên một bộ đồ chơi

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

Sau đủ thời gian trên một tập hợp lớn, những từ chia sẻ bối cảnh có những nhúng cốt tương tự. trên một tập hợp đồ chơi, bạn thấy hiệu ứng yếu. trên hàng tỷ token, bạn thấy nó đáng kể.

### Bước 5: thủ thuật tương tự

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

Trên các vector Google News 300d được đào tạo trước:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`Không phải vì người mẫu biết hoàng gia là gì, vì người dẫn đường là người có quyền.`(king - man)`bắt được thứ gì đó như " hoàng gia", và thêm nó vào `woman`đất gần vùng đất nữ hoàng.

## Sử dụng nó

Viết Word2Vec từ đầu là dạy.`gensim`- Tôi không biết.

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

Để làm việc thực sự, bạn hầu như không bao giờ tự đào tạo Word2Vec. Bạn tải xuống các vector được đào tạo trước.

- **GloVe** Phương pháp phân tích các yếu tố tử liệu đồng xuất hiện của Stanford. 50d, 100d, 200d, 300d điểm kiểm soát.
- **fastText** Word2Vec mở rộng của Facebook nhúng n-gram ký tự. xử lý từ ngoài từ vựng bằng cách soạn các từ phụ. Bài học 04.
- **Pretrained Word2Vec on Google News** 300d, từ vựng 3M, được xuất bản năm 2013.

### Khi Word2Vec vẫn thắng vào năm 2026

- Đọc về các bản tóm tắt y tế trong một giờ trên máy tính xách tay, có được các vector chuyên dụng không có mô hình tổng quát.
- Kỹ thuật tính năng theo kiểu tương tự. `gender_vector = mean(man - woman pairs)`- Trả nó ra từ những từ khác để có được một trục trung lập về giới tính.
- 100d đủ nhỏ để vẽ qua PCA hoặc t-SNE và thực sự thấy các cụm hình thành.
- Bất cứ nơi nào, suy luận phải chạy trên thiết bị mà không có GPU. Word2Vec tìm kiếm là một hàng lấy.

### Khi Word2Vec thất bại

Bức tường đa sắc.`bank`có một vector. `river bank`và `financial bank`Hãy chia sẻ nó.`table`Một bộ phân loại dòng chảy xuống không thể phân biệt các giác quan từ vector.

Các bản nhúng ngữ cảnh (ELMo, BERT, mọi biến thể kể từ đó) giải quyết vấn đề này bằng cách tạo ra một vector khác nhau cho mỗi sự xuất hiện của từ dựa trên bối cảnh xung quanh.

Vấn đề không có từ vựng là sự thất bại khác.`Zoomer-approved`nếu nó không có trong dữ liệu đào tạo. Không có sự lùi. fastText sửa chữa điều này bằng cách tạo thành từ phụ (đọc 04).

## Chuyển nó

Cứ như `outputs/skill-embedding-probe.md`- Có thể là:

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

## Các bài tập

1. **Easy.**Thực hiện vòng tròn huấn luyện trên một bộ phận nhỏ (20 câu về mèo và chó).`nearest(vocab, W, W[vocab["cat"]])`trả lại `dog`Nếu không, tăng thời đại hoặc từ vựng.
2. **Medium.**Thêm mẫu phụ của các từ thường xuyên.`10^-5`được loại bỏ từ các cặp đào tạo với xác suất tương xứng với tần suất của chúng.
3. **Hard.**Tập một mô hình trên 20 Newsgroups corpus.`he - she`và `doctor - nurse`. Dự án các từ nghề trên cả hai trục. báo cáo các nghề có khoảng cách thiên vị lớn nhất. Đây là loại hình của các nhà nghiên cứu công bằng thăm dò sử dụng.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | A dense, low-dim (typically 100-300) representation learned from context. |
| Skip-gram | Word2Vec trick | Predict context words from center word. Slower than CBOW, better for rare words. |
| Negative sampling | Training shortcut | Replace softmax over full vocab with binary classification against `k` random words. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-sensitive vector | Different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | Word not seen in training. Word2Vec cannot produce a vector for these. |

## Đọc thêm

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) giấy lấy mẫu âm.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) dẫn xuất rõ ràng nhất của các gradient, nếu toán học của giấy gốc cảm thấy dày đặc.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) cài đặt đào tạo sản xuất thực sự hoạt động.
