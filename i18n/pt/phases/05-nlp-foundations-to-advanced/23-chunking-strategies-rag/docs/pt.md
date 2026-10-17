# Estratégias de desmantelamento de RAG

> A configuração de fragmentação influencia a qualidade da recuperação tanto quanto a escolha do modelo de incorporação (Vectara NAACL 2025).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## O problema

Você coloca um contrato de 50 páginas em um sistema RAG. O usuário pergunta: "Qual é a cláusula de rescisão?" O retriever retorna a página de capa. Por quê? Porque o modelo foi treinado em 512 tokens e a cláusula de rescisão fica em 20 páginas, dividida em uma pausa de página, sem palavras-chave locais ligando-a à consulta.

A solução não é "comprar um modelo de incorporação melhor". A solução é a de massa.

Os índices de referência de Fevereiro de 2026 mostram resultados surpreendentes:

- O estudo de 2026 de Vectara: o chunking recorrente de 512 tokens venceu o chunking semântico com precisão de 69% → 54%.
- SPLADE + Mistral-8B sobre questões naturais: a sobreposição proporcionou benefício medível zero.
- Cliff de contexto: a qualidade da resposta cai drasticamente em torno de 2.500 tokens de contexto.

A resposta "obvia" (combinação semântica, sobreposição de 20% e 1000 tokens) é muitas vezes errada.

## O conceito

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.**Divide todos os caracteres N ou tokens. linha de base mais simples. quebra no meio da frase. boa compressão, má coerência.

**Recursive.**A LangChain `RecursiveCharacterTextSplitter`Tenta dividir-te .`\n\n`Primeiro, depois.`\n`, então`.`O 2026 é o padrão.

**Semantic.**Embed cada frase. Computa a semelhança cosínea entre frases adjacentes. Divida onde a semelhança cai abaixo de um limiar. Preserva a coerência do tópico. Mais lento; às vezes produz pequenos fragmentos de 40 tokens que prejudicam a recuperação.

**Sentence.**Divida em limites de frases. Uma frase por peça ou uma janela de frases N. Correspondem a um pedaço semântico de até ~ 5k tokens em uma fração do custo.

**Parent-document.**Armazenar pequenos pedaços de criança para recuperação * e * o maior pedaço pai para contexto. Retirada por criança; retornar pai. Degrados graciosamente: pedaços de criança ruim ainda retornar pais razoáveis.

**Late chunking (2024).**Embed o documento inteiro no nível de token primeiro, em seguida, pool embeddings de token em embutidos de pedaços. Preserva contexto de pedaços cruzados. Funciona com embutidos de longo contexto (BGE-M3, Jina v3). Computação superior.

**Contextual retrieval (Anthropic, 2024).**Prepare cada peça com um resumo gerado pelo LLM da sua posição no documento ("Esta peça é a seção 3.2 das cláusulas de rescisão...").

### A regra que vence todos os padrões

Compare o tamanho do pedaço com o tipo de consulta:

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

O comparativo da NVIDIA para 2026. O pedaço deve ser grande o suficiente para conter a resposta mais contexto local, pequeno o suficiente para que os retrievers top-K retornem focando na resposta em vez de ruído de contexto.

```figure
n5-chunk-cuts
```

## Construí-lo

### Passo 1: despejo fixo e recorrente

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

### Passo 2: fragmentação semântica

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

- Tune .`threshold`Muito alto → fragmentos.

### Passo 3: documento dos pais

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

Uma ideia importante: pais dedupidos. Multiples filhos podem mapear para o mesmo pai; devolver tudo desperdiçaria contexto.

### Passo 4: recuperação contextual (patrão antropológico)

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

Indique os blocos contextualizados. No momento da consulta, a recuperação beneficia do sinal adicional circundante.

### Passo 5: Avaliação

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

A melhor estratégia para o seu corpo pode não corresponder a qualquer post no blog.

## Encurralagens

- **Chunking evaluated only on factoid queries.**As consultas multi-hop revelam vencedores muito diferentes. Use um conjunto de avaliação estratificado de tipo de consulta.
- **Semantic chunking without a minimum size.**Produz 40 fragmentos de tokens que prejudicam a recuperação.`min_tokens`- Não .
- **Overlap as cargo cult.**Os estudos de 2026 encontram que a sobreposição geralmente proporciona benefícios zero e duplica o custo do índice.
- **No min/max enforcement.**Peças de 5 tokens ou 5000 tokens quebram a recuperação.
- **Cross-doc chunking.**Nunca deixe um pedaço de dois documentos, sempre pedaço por documento, e depois merge.

## Usá-lo

A pilha de 2026:

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

Comece com o recorrente 512. Medir recall@5 num conjunto de avaliação de 50 perguntas.

## Envia-o

Salva como`outputs/skill-chunker.md`- Não .

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

## Exercícios

1. **Easy.**Caso o documento seja de 20 páginas, comparar o número de partes e a qualidade da fronteira.
2. **Medium.**Construir um conjunto de avaliação de 30 perguntas em 5 documentos. Medir recall@5 para documento recursivo, semântico e parental. Qual ganha?
3. **Hard.**Implementar a recuperação contextual. Medir a melhoria do MRR em relação ao recorrente de linha de base.

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | A piece of a doc | Sub-document unit that gets embedded, indexed, and retrieved. |
| Overlap | Safety margin | N tokens shared between adjacent chunks; often useless in 2026 benchmarks. |
| Semantic chunking | Smart chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Two-level retrieval | Retrieve small children, return larger parents. |
| Late chunking | Chunk after embedding | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | Anthropic's trick | LLM-generated summary prepended to each chunk before indexing. |
| Context cliff | 2500-token wall | Quality drop observed around 2.5k context tokens in RAG (Jan 2026). |

## Mais leitura

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) o incumprimento da produção.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) A questão da parcela é tão importante quanto a de inserir a escolha.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)O papel de pedaços atrasados.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) Melhoria de recuperação de 35-50% com prefixos de contexto gerados pelo MLL.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) tamanho do pedaço por tipo de consulta.
