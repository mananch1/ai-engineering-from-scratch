# ورڈ ادغام  Word2Vec از ابتدا

> یک کلمه شرکت است که نگه می دارد. یک شبکه سطحی را روی آن ایده تمرین کنید و هندسه سقوط می کند.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## مشکل

TF-IDF ميدونه`dog`و`puppy`این کلمات متفاوت هستند. نمی داند که تقریباً یک چیز را معنی دارند.`dog`نمیتونم به یک بررسی درباره`puppy`شما می توانید با فهرست مترادفات این موضوع را بررسی کنید، اما این در اصطلاحات نادر، اصطلاحات دامنه و هر زبانی که پیش بینی نکرده اید، شکست می خورد.

تو ميخواي يه نمايشگاهي داشته باشي که`dog`و`puppy`در فضا به هم نزديک زمين مي ريزم`king - man + woman`زمین های نزدیک`queen`. جایی که یک مدل آموزش دیده`dog`يه سيگنال رو به `puppy`. به صورت رایگان

Word2Vec به ما این فضای را داد. دو لایه شبکه عصبی، تریلیون توکن تمرین، منتشر شده در سال 2013. معماری تقریباً شرم آور ساده است. نتایج NLP را برای یک دهه تغییر شکل داد.

## مفهوم

**Distributional hypothesis**(اولین، 1957): "یک کلمه را از طریق صحبت هایی که می کند می شناسید".

Word2Vec به دو نوع عرضه می شود، هر دو از این ایده استفاده می کنند.

- **Skip-gram.**با توجه به کلمه ی مرکزی، کلمات اطراف را پیش بینی کنید.`cat -> (the, sat, on)`با اندازه پنجره 2
- **CBOW (continuous bag of words).**با توجه به کلمات اطراف، مرکز را پیش بینی کنید.`(the, sat, on) -> cat`. .

اسکیپ گرام در آموزش آهسته تر است اما با کلمات نادر بهتر کار می کند.

شبکه دارای یک لایه پنهان بدون عدم خطی است. ورودی یک ویکتور یک گرم بر روی لغت است. خروجی یک نرم حداکثر بر روی لغت است. پس از آموزش، شما لایه خروجی را دور می اندازید. وزنه های لایه پنهان گنجانده شده هستند.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

ترفند: نرمترین مقدار بیش از 100 هزار کلمه بسیار گران است.**negative sampling**برای تبدیل آن به یک کار طبقه بندی دوگانه. پیش بینی "آیا این کلمه زمینه در نزدیکی این کلمه مرکزی ظاهر می شود، بله یا نه". نمونه چند کلمه منفی (غیر همزمان) در هر جفت آموزش به جای محاسبه نرم حداکثر در کل لغات.

```figure
word-vector-arithmetic
```

## آن را بسازید

### مرحله ی اول: زوج های آموزشی از یک کورپوس

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

هر جفت (مرکز، زمینه) در پنجره یک مثال آموزشی مثبت است.

### مرحله دوم: قرار دادن جدول ها

دو ماتریس`W`جدول ادغام کلمه مرکزی (آن که شما نگه دارید) است.`W'`جدول متن کلمه ( اغلب رد می شود، گاهی اوقات با `W`)

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

اندازه لغت 10k و dim 100 واقع بینانه است؛ برای آموزش، 50 لغت x 16 dim برای دیدن هندسه کافی است.

### مرحله سوم: هدف منفی نمونه گیری

برای هر جفت مثبت`(center, context)`نمونه`k`کلمات تصادفی از لغات به عنوان منفی. مدل را آموزش دهید تا محصول نقطه ای`W[center] · W'[context]`برای مثبت ها بالا و برای منفی ها پایین است.

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

فرمول جادویی: از دست دادن لجستیک در جفت مثبت (خواهید سیگمائید نزدیک به 1) به علاوه از دست دادن لجستیک در جفت منفی (خواهید سیگمائید نزدیک به 0). درجه بندی ها به هر دو جدول جریان دارند. مشتق کامل در کاغذ اصلی است؛ اگر می خواهید چسبید یک بار با قلم و کاغذ از آن عبور کنید.

### مرحله 4: تمرین روی یک جسم اسباب بازی

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

پس از دوره های کافی در یک کورپوس بزرگ، کلمات که زمینه های مشترک را دارند، دارای ورق های مرکزی مشابهی هستند. در یک کورپوس اسباب بازی، شما اثر را کم کم می بینید. در میلیارد ها توکن، شما آن را به طور چشمگیری می بینید.

### مرحله پنجم: ترفند مشابه

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

در 300d ویکتورهای پیش آموزش دیده گوگل نیوز:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`نه چون مدل ميدونه شاهزاده چيست چون متری`(king - man)`چیزی شبیه به "رویال" رو میگیرم و به اون میگم`woman`زمين ها نزديک منطقه بانوان سلطنتي

## ازش استفاده کن

نوشتن Word2Vec از ابتدا آموزش است. تولید از NLP استفاده می کند `gensim`. .

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

برای کار واقعی، تقریباً هرگز خودت Word2Vec را آموزش نمی دهی.

- **GloVe** رویکرد عامل سازی ماتریس هم وقوع استنفورد. نقاط بازرسی 50d، 100d، 200d، 300d. پوشش کلی خوب. درس 04 به طور خاص به GloVe مربوط می شود.
- **fastText** افزونه Word2Vec فیس بوک که شامل n-gram حرف می شود. کلمات خارج از ذخایر لغات را با ترکیب زیرکلمات اداره می کند. درس 04.
- **Pretrained Word2Vec on Google News** 300d، 3M لغت واژه، منتشر شده 2013. هنوز هم روزانه دانلود می شود.

### وقتی Word2Vec هنوز در سال 2026 برنده میشه

- بازيافتي مخصوص دامنه ي سبک، آموزش در مورد خلاصه هاي پزشكي در يک ساعت با لپ تاپ، وکتور هاي تخصصي بدون گرفتن مدل هاي عمومي
- مهندسی ویژگی های سبک مشابه.`gender_vector = mean(man - woman pairs)`از کلمات دیگر آن را حذف کنید تا محور خنثی جنسیتی پیدا کنید. هنوز هم در تحقیقات عدالت استفاده می شود.
- تفسیر. 100d به اندازه کافی کوچک است تا از طریق PCA یا t-SNE نقشه برداری کند و در واقع به شکل خوشه ها نگاه کند.
- هر جا که استناد بشه بايد بدون GPU اجرا بشه Word2Vec یک ردیف است

### جایی که Word2Vec شکست خورده است

ديوار پليسمي`bank`یک ویکتور دارد.`river bank`و`financial bank`بهشون بده`table`(برنامه بر اساس مبلمان) آن را به اشتراک می گذارد. یک طبقه بندی کننده به سمت پایین نمی تواند حواس را از ویکتور تشخیص دهد.

گنجانده شدن های زمینه ای (ELMo، BERT، هر ترانسفورماتور از آن زمان) این مسئله را با تولید یک ویکتور مختلف برای هر بروز کلمه بر اساس زمینه اطراف حل کردند. این قفسه از Word2Vec به BERT: از ثابت به زمینه ای است. مرحله 7 نصف ترانسفورم را پوشش می دهد.

مشکل خارج از ذخایر لغات شکست دیگری است. Word2Vec هرگز ندیده است`Zoomer-approved`اگر در داده های آموزش نبود. هیچ بازنگری. fastText این را با ترکیب زیرکلمه (درسی 04) حل می کند.

## -باده

پس از`outputs/skill-embedding-probe.md`:

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

## تمرینات

1. **Easy.**چرخه آموزش را روی یک کورپوس کوچک (20 جمله در مورد گربه ها و سگ ها) اجرا کنید. پس از 200 دوره، تایید کنید`nearest(vocab, W, W[vocab["cat"]])`بازپرداخت`dog`در صورت عدم آن، دوره ها یا لغت ها را افزایش دهید.
2. **Medium.**اضافه کردن نمونه فرعی از کلمات مکرر. کلمات با فرکانس بالاتر`10^-5`در این مطالعه، نتایج نتایج نتایج نتایج را در مورد دو کلمه آموزش داده شده با احتمال متناسب با فرکانس آنها، ارزیابی می شود.
3. **Hard.**مدل را بر روی 20 Newsgroups corpus آموزش دهید. دو محور تعصب را محاسبه کنید:`he - she`و`doctor - nurse`.برنامه ی کلمات شغلی را روی هر دو محور تهیه کنید. گزارش دهید که کدام مشاغل بیشترین شکاف تعصب را دارند. این نوع از تحقیقات تحقیقات است.

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | A dense, low-dim (typically 100-300) representation learned from context. |
| Skip-gram | Word2Vec trick | Predict context words from center word. Slower than CBOW, better for rare words. |
| Negative sampling | Training shortcut | Replace softmax over full vocab with binary classification against `k` random words. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-sensitive vector | Different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | Word not seen in training. Word2Vec cannot produce a vector for these. |

## خواندن بیشتر

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) کاغذ نمونه گیری منفی کوتاه و قابل خواندن
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) واضح ترین مشتق گرادینت ها، اگر ریاضیات کاغذ اصلی احساس کثافت.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html)تنظیمات آموزش تولید که واقعا کار می کنند.
