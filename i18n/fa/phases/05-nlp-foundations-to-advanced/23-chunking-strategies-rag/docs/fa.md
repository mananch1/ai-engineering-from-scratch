# استراتژی های شکستن برای RAG

> پیکربندی های شکسته شده بر کیفیت بازیافت به اندازه انتخاب مدل گنجانده شدن (Vectara NAACL 2025) تأثیر می گذارد.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## مشکل

شما یک قرارداد 50 صفحه را به یک سیستم RAG می گذارید. کاربر می پرسد: "حجره پایان نامه چیست؟" بازیافت کننده صفحه اصلی را باز می گرداند. چرا؟ چون مدل در 512 تکه آموزش دیده است و حجره پایان نامه 20 صفحه است، تقسیم شده در یک صفحه، بدون کلمات کلیدی محلی که آن را به جستجو متصل کند.

مشکل این نیست که "یک مدل بهتر از این مدل بخر" بلکه مشکل اینه که "چقدر بزرگ، چه شکلی، چه شکلی، کجا تقسیم بشه، چه حاشیه ای؟

فبروری 2026 شاخص های مرجعی نتایج شگفت انگیزی را نشان می دهند:

- مطالعه 2026 ویکتارا: شکستن 512 توکن تکراری از شکستن معنوی 69٪ → 54٪ دقت را شکست می دهد.
- SPLADE + Mistral-8B در مسائل طبیعی: تعادل سود قابل اندازه گیری صفر را فراهم می کند.
- صخره زمینه: کیفیت پاسخ به شدت در حدود 2500 توکن زمینه کاهش می یابد.

پاسخ "آشکاری" (تکثیر معنوی، 20٪ تعادل، 1000 توکن) اغلب اشتباه است. این درس برای شش استراتژی هوشیاری ایجاد می کند و به شما می گوید که چه زمانی به کدام یک برسید.

## مفهوم

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.**هر N حرف و نماد رو تقسیم کن ساده ترین خط اصلی وسط جمله شکسته فشرده سازی خوب و منسجمیت بد

**Recursive.**لانگچين`RecursiveCharacterTextSplitter`. سعی کن بربخشش`\n\n`اول، بعدش`\n`، پس`.`بعد فضا، به صورت پیش فرض به سال 2026 برمی گردد

**Semantic.**هر جمله را دربرگیرید. شباهت کوسین بین جمله های مجاور را محاسبه کنید. تقسیم کنید که شباهت زیر یک حد کاهش می یابد. پیوستگی موضوع را حفظ می کند. آهسته تر؛ گاهی اوقات تکه های کوچک 40 توکن را تولید می کند که به بازیافت آسیب می رساند.

**Sentence.**به مرز جمله تقسیم کنید. یک جمله در هر قطعه یا پنجره ای از جمله های N. مطابقت با قطعه معنوی تا ~ 5k توکن در یک بخش از هزینه.

**Parent-document.**قطعات کوچک کودک را برای بازیافت ذخیره کنید * و * قطعات بزرگتر والدین را برای زمینه. به صورت کودک بازیافت کنید؛ والدین را برگردانید. به صورت زیبا تخریب می شود: قطعات کودک بد هنوز والدین معقول را بازیافت می کنند.

**Late chunking (2024).**در ابتدا کل سند را در سطح توکن ها قرار دهید، سپس توکن های کوبیده را به کوبیدهای کوبیده جمع کنید. زمینه های کوبیده را حفظ می کند. با کوبیدهای طولانی (BGE-M3 ، Jina v3) کار می کند. محاسبه بالاتر.

**Contextual retrieval (Anthropic, 2024).**هر بخش را با خلاصه ای از موقعیت آن در سند تولید شده توسط LLM آماده کنید ("این بخش بخش بخش 3.2 از بند های پایان نامه است ..."). 35-50٪ بهبود بازیافت در معیار خود Anthropic.

### قانون که هر شکست را شکست می دهد

اندازه قطعه را با نوع سوال مطابقت دهید:

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

مقادیر زیادی باید برای پاسخ به همراه زمینه محلی، به اندازه کافی کوچک باشد تا بازیافت کننده بالا K به جای صداهای زمینه روی پاسخ تمرکز کند.

```figure
n5-chunk-cuts
```

## آن را بسازید

### مرحله ی اول: قطع بندی ثابت و تکراری

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### مرحله دوم: شکستن معنوی

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

صداي`threshold`خیلی بالا به سمت قطعات خیلی پایین به سمت یک قطعه بزرگ

### مرحله سوم: سند والدین

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

نکته کلیدی: والدین متاهل. چندین کودک می توانند به یک پدر و مادر نقشه بزنند؛ بازگردانیدن همه آنها، ضایع کنونیست.

### مرحله 4: بازیافت زمینه ای (نمونه انسان)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

در زمان جستجو، بازیافت از سیگنال های اطراف اضافی بهره می برد.

### مرحله 5: ارزیابی

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

همیشه معیار. بهترین استراتژی برای کارپوس شما ممکن است با هیچ پست وبلاگ مطابقت نداشته باشد.

## دام ها

- **Chunking evaluated only on factoid queries.**سوالات چند تا برندهای بسیار متفاوتی را نشان می دهند. از مجموعه ارزیابی طبقه بندی شده نوع سوال استفاده کنید.
- **Semantic chunking without a minimum size.**40 تا شکه از شکه ها رو تولید ميکنه که به بازيابي آسیب مي رسونه`min_tokens`. .
- **Overlap as cargo cult.**مطالعات 2026 نشان می دهد که تعادل اغلب سود صفر و هزینه شاخص دو برابر را فراهم می کند.
- **No min/max enforcement.**قطعات 5 تا و یا 5000 تا هر دو بازيابي رو متوقف ميکنن
- **Cross-doc chunking.**هرگز اجازه ندهيد يه قطعه دو سند رو از هم جدا کنه هميشه قطعه اي به هر دو دوک، پس همگامش کنيد

## ازش استفاده کن

دسته 2026:

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

با 512 تکراری شروع کنید. یادآوری را در یک مجموعه 50 سوال ارزیابی کنید. از آنجا تنظیم کنید.

## -باده

پس از`outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## تمرینات

1. **Easy.**یک سند ۲۰ صفحه را با ثابت ((512, 0) ، مکرر ((512, 0) و مکرر ((512, 100) مقایسه کنید.
2. **Medium.**یک مجموعه ارزیابی 30 سوال را در 5 سند بسازید. یادآوری@5 را برای مستند تکراری، معنوی و والدین اندازه گیری کنید. کدام یک برنده می شود؟ آیا با پست های وبلاگ مطابقت دارد؟
3. **Hard.**پیاده سازی بازیافت زمینه ای. اندازه گیری بهبود MRR نسبت به خط اصلی بازگشت. گزارش هزینه شاخص (دعوات LLM) در مقابل افزایش دقت.

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | A piece of a doc | Sub-document unit that gets embedded, indexed, and retrieved. |
| Overlap | Safety margin | N tokens shared between adjacent chunks; often useless in 2026 benchmarks. |
| Semantic chunking | Smart chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Two-level retrieval | Retrieve small children, return larger parents. |
| Late chunking | Chunk after embedding | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | Anthropic's trick | LLM-generated summary prepended to each chunk before indexing. |
| Context cliff | 2500-token wall | Quality drop observed around 2.5k context tokens in RAG (Jan 2026). |

## خواندن بیشتر

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) عدم انجام تولید
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) تکه تکه کردن مهمه مثل انتخاب کردن
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) کاغذي که دير ميکنه
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) 35-50٪ بهبود بازیافت با پیشگوهای زمینه تولید شده توسط LLM.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) اندازه قطعه بر اساس نوع سوال
