# Chiến lược làm mảnh vỡ cho RAG

> Các cấu hình phân chia ảnh hưởng đến chất lượng thu hồi cũng như sự lựa chọn của mô hình nhúng (Vectara NAACL 2025).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## Vấn đề

Bạn đặt một hợp đồng 50 trang vào một hệ thống RAG. Người dùng hỏi: "Điều khoản chấm dứt là gì?" Người tìm kiếm trả lại trang bìa. Tại sao? Bởi vì mô hình được đào tạo trên 512 token và điều khoản chấm dứt nằm trong 20 trang, chia rẽ qua một đoạn trang, mà không có từ khóa địa phương liên kết nó với truy vấn.

Việc sửa chữa không phải là "mua một mô hình nhúng tốt hơn". Việc sửa chữa là chunking.

Các điểm chuẩn tháng 2 năm 2026 cho thấy kết quả đáng ngạc nhiên:

- Nghiên cứu 2026 của Vectara: việc chia 512 token tái tạo vượt qua việc chia semantic 69% → 54% chính xác.
- SPLADE + Mistral-8B về các vấn đề tự nhiên: sự chồng chéo đã cung cấp không lợi ích có thể đo lường.
- Vẻng ngữ cảnh: chất lượng phản ứng giảm mạnh xung quanh 2.500 token ngữ cảnh.

Câu trả lời "đáng rõ ràng" (chunking ngữ nghĩa, 20% chồng chéo, 1000 token) thường sai. Bài học này xây dựng trực giác cho sáu chiến lược và cho bạn biết khi nào để đạt được.

## Khái niệm

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.**Chia từng chữ cái hoặc mã thông báo N đơn giản nhất, phá vỡ giữa câu, nén tốt, liên kết xấu.

**Recursive.**LangChain `RecursiveCharacterTextSplitter`- Hãy thử chia tay đi.`\n\n`Đầu tiên, sau đó `\n`, sau đó `.`- Rồi không gian, rơi lại sạch sẽ, mặc định là năm 2026.

**Semantic.**Đưa vào mỗi câu. Xét tương đồng cosine giữa các câu lân cận. Chia khi tương đồng giảm xuống dưới ngưỡng. Giữ sự liên kết chủ đề. chậm hơn; đôi khi tạo ra các mảnh nhỏ 40 token làm tổn thương việc lấy lại.

**Sentence.**Chia thành giới hạn câu. Một câu mỗi phần hoặc một cửa sổ của N câu. Tương ứng với phần ngữ nghĩa lên đến ~ 5k token với một phần nhỏ chi phí.

**Parent-document.**Cung cấp nhỏ trẻ em để lấy lại * và * phần lớn cha mẹ cho ngữ cảnh. lấy lại theo trẻ em; trả lại cha mẹ. Thảm xuống lịch sự: phần trẻ em xấu vẫn trả lại cha mẹ hợp lý.

**Late chunking (2024).**Nhúng toàn bộ tài liệu ở cấp độ token trước, sau đó tích hợp các nhúng token vào các nhúng chip. Bảo tồn ngữ cảnh chéo-chunk. Làm việc với các nhúng ngữ cảnh dài (BGE-M3, Jina v3). tính toán cao hơn.

**Contextual retrieval (Anthropic, 2024).**Chuẩn bị từng phần với một bản tóm tắt được tạo ra bởi LLM về vị trí của nó trong tài liệu ("Bản phần này là phần 3.2 của các điều khoản chấm dứt..."). 35-50% cải thiện tìm kiếm trong chỉ số chuẩn của Anthropic.

### Quy tắc đánh bại mọi người mặc định

Tích kích thước phần với kiểu truy vấn:

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

Điểm chuẩn của NVIDIA năm 2026. Phần này nên đủ lớn để chứa câu trả lời cộng với bối cảnh địa phương, đủ nhỏ để các phần trả lời top-K của máy thu hồi tập trung vào câu trả lời thay vì tiếng ồn bối cảnh.

```figure
n5-chunk-cuts
```

## Hãy xây dựng nó

### Bước 1: Phân tích cố định và tái tạo

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

### Bước 2: Phân tích ngữ nghĩa

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

Định âm`threshold`quá cao → mảnh vỡ. quá thấp → một mảnh khổng lồ.

### Bước 3: Tài liệu cha mẹ

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

Một ý kiến quan trọng: cha mẹ không có con cái. Nhiều đứa trẻ có thể tìm đến cùng một cha mẹ; trả lại tất cả sẽ lãng phí ngữ cảnh.

### Bước 4: lấy lại ngữ cảnh (phát hình nhân tạo)

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

Chỉ số các đoạn kết hợp. Vào thời điểm truy vấn, lấy lại lợi ích từ tín hiệu xung quanh thêm.

### Bước 5: đánh giá

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

Luôn luôn là điểm chuẩn. Chiến lược "tốt nhất" cho cơ thể của bạn có thể không phù hợp với bất kỳ bài đăng trên blog nào.

## Những bẫy

- **Chunking evaluated only on factoid queries.**Các truy vấn đa hop cho thấy những người chiến thắng rất khác nhau. Sử dụng một tập hợp đánh giá phân cấp kiểu truy vấn.
- **Semantic chunking without a minimum size.**Tạo ra 40 đoạn mã thông báo làm tổn thương việc thu hồi.`min_tokens`- Tôi không biết.
- **Overlap as cargo cult.**Các nghiên cứu năm 2026 cho thấy sự chồng chéo thường cung cấp lợi ích không và chi phí chỉ số gấp đôi.
- **No min/max enforcement.**5 token hoặc 5000 token đều phá vỡ việc lấy lại.
- **Cross-doc chunking.**Đừng bao giờ để một mảnh vỡ trải dài hai tài liệu.

## Sử dụng nó

Số 2026:

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

Bắt đầu với 512 tái phát. đo recall@5 trên một tập hợp đánh giá 50 truy vấn. Tune từ đó.

## Chuyển nó

Cứ như `outputs/skill-chunker.md`- Có thể là:

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

## Các bài tập

1. **Easy.**Chia một tài liệu 20 trang với cố định ((512, 0), tái tạo ((512, 0), và tái tạo ((512, 100). So sánh số lượng phần và chất lượng biên giới.
2. **Medium.**Xây dựng một tập hợp đánh giá 30 câu hỏi trên 5 tài liệu. đo recall@5 cho tài liệu thu hồi, ngữ nghĩa và chính.
3. **Hard.**Thực hiện tìm kiếm ngữ cảnh. đo lường sự cải thiện MRR so với dòng tiền đề tái phát. báo cáo chi phí chỉ số (call LLM) so với tăng độ chính xác.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | A piece of a doc | Sub-document unit that gets embedded, indexed, and retrieved. |
| Overlap | Safety margin | N tokens shared between adjacent chunks; often useless in 2026 benchmarks. |
| Semantic chunking | Smart chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Two-level retrieval | Retrieve small children, return larger parents. |
| Late chunking | Chunk after embedding | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | Anthropic's trick | LLM-generated summary prepended to each chunk before indexing. |
| Context cliff | 2500-token wall | Quality drop observed around 2.5k context tokens in RAG (Jan 2026). |

## Đọc thêm

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) thất bại trong sản xuất.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) chia nhỏ vấn đề như việc nhúng lựa chọn.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) tờ giấy trộn trộn.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) 35-50% cải thiện việc thu hồi với các tiền đề ngữ cảnh được tạo ra bởi LLM.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) kích thước phần theo loại truy vấn.
