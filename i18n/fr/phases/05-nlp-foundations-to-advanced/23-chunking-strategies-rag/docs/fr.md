# Stratégies de déchiquetage pour les RAG

> La configuration de l'emballage influence la qualité de la récupération autant que le choix du modèle d'emballage (Vectara NAACL 2025).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## Le problème

Vous mettez un contrat de 50 pages dans un système RAG. L'utilisateur demande: "Quelle est la clause de résiliation?" Le récupérateur renvoie la page d'accueil. Pourquoi? Parce que le modèle a été formé sur 512 jetons et la clause de résiliation est de 20 pages, divisée sur une page, sans mots clés locaux liant à la requête.

Le problème n'est pas "acheter un meilleur modèle d'intégration". Le problème est de se déchiffrer.

Les critères de référence de février 2026 montrent des résultats surprenants:

- L'étude de Vectara de 2026: le déchiquetage récursif de 512 jetons a dépassé le déchiquetage sémantique avec une précision de 69% → 54%.
- SPLADE + Mistral-8B sur les questions naturelles: la chevauchement a fourni un bénéfice mesurable zéro.
- Cliff de contexte: la qualité de la réponse diminue fortement autour de 2500 jetons de contexte.

La réponse " évidente " (chunking sémantique, 20% de chevauchement, 1000 jetons) est souvent fausse.

## Le concept

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.**Divisez tous les caractères N ou les jetons, la ligne de base la plus simple, rompt le milieu de la phrase, bonne compression, mauvaise cohérence.

**Recursive.**La chaîne de langage `RecursiveCharacterTextSplitter`- Essayez de vous séparer .`\n\n`D'abord, puis`\n`Alors ...`.`Il est en train de tomber, le 2026 est en défaut.

**Semantic.**Embed chaque phrase. Compute la similitude cosine entre les phrases adjacentes. Divisez où la similitude tombe en dessous d'un seuil. Préserve la cohérence du sujet. Plus lent; produit parfois de minuscules fragments de 40 jetons qui nuisent à la récupération.

**Sentence.**Partagez-les en limites de phrases. Une phrase par morceau ou une fenêtre de N phrases.

**Parent-document.**Remplissez les petits morceaux d'enfants pour les récupérer * et * la plus grande partie parent pour le contexte. Remplissez par enfant; retournez parent. Dégrade gracieusement: les mauvais morceaux d'enfants retournent toujours des parents raisonnables.

**Late chunking (2024).**Embed l'ensemble du document au niveau des jetons d'abord, puis pool jetons intégrations dans des emblèmes de pièces. préserve le contexte de pièces croisées. Fonctionne avec les emblèmes de long-context (BGE-M3, Jina v3).

**Contextual retrieval (Anthropic, 2024).**Préparez chaque pièce avec un résumé de sa position dans le document généré par le LLM ("Cette pièce est la section 3.2 des clauses de résiliation...").

### La règle qui vaille tous les défauts

Correspondre la taille de la pièce au type de requête:

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

Le résultat de la recherche devrait être suffisamment grand pour contenir la réponse plus le contexte local, suffisamment petit pour que le retour de K du retriever se concentre sur la réponse plutôt que sur le bruit de contexte.

```figure
n5-chunk-cuts
```

## Faites-le

### Étape 1: déchiquetage fixe et récursif

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

### Étape 2: décomposition sémantique

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

La musique`threshold`trop haut → fragments trop bas → une pièce géante

### Étape 3: document parental

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

Les parents dédupeurs: plusieurs enfants peuvent se rendre au même parent; tout retourner serait une perte de contexte.

### Étape 4: récupération contextuelle (motifs anthropiques)

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

L'indexation des éléments contextualisés. Au moment de la requête, la récupération bénéficie du signal supplémentaire environnant.

### Étape 5: évaluer

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

La meilleure stratégie pour votre corps peut ne pas correspondre à un article de blog.

## Les pièges

- **Chunking evaluated only on factoid queries.**Les requêtes multi-hop révèlent des gagnants très différents.
- **Semantic chunking without a minimum size.**Produit 40 fragments de jetons qui nuisent à la récupération.`min_tokens`- Je suis désolé .
- **Overlap as cargo cult.**Les études de 2026 révèlent que le chevauchement offre souvent un bénéfice nul et un coût double.
- **No min/max enforcement.**Les morceaux de 5 jetons ou 5000 jetons sont tous les deux en panne.
- **Cross-doc chunking.**Ne laissez jamais une pièce couvrir deux documents, toujours par pièce, puis fusionnez.

## Utilisez-le

La pile de 2026:

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

Commencez par le récursif 512. Mesurer le rappel@5 sur un ensemble d'évaluation de 50 requêtes.

## La faire partir

- Je ne sais pas .`outputs/skill-chunker.md`- Le numéro de la liste:

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

## Exercices

1. **Easy.**Comparer le nombre de pièces et la qualité des limites.
2. **Medium.**Construire un ensemble d'évaluation de 30 requêtes sur 5 documents. Mesurer le rappel@5 pour le document récursif, sémantique et parent. Lequel gagne?
3. **Hard.**Mettre en œuvre la récupération contextuelle. Mesurer l'amélioration du MRR par rapport à la référence récursive. Rapporter le coût de l'index (appels LLM) par rapport à l'augmentation de la précision.

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | A piece of a doc | Sub-document unit that gets embedded, indexed, and retrieved. |
| Overlap | Safety margin | N tokens shared between adjacent chunks; often useless in 2026 benchmarks. |
| Semantic chunking | Smart chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Two-level retrieval | Retrieve small children, return larger parents. |
| Late chunking | Chunk after embedding | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | Anthropic's trick | LLM-generated summary prepended to each chunk before indexing. |
| Context cliff | 2500-token wall | Quality drop observed around 2.5k context tokens in RAG (Jan 2026). |

## Pour en savoir plus

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) le défaut de production.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) Le déchiquetage est aussi important que le choix de l'intégration.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)- Le papier à déchiquetage tardif.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) Amélioration de la récupération de 35 à 50% avec les préfixes de contexte générés par le LLM.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) taille de la pièce par type de requête.
