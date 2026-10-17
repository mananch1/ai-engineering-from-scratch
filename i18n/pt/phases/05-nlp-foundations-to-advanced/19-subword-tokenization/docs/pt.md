# Tokenization de subpalavras  BPE, WordPiece, Unigram, SentencePiece

> Os tokenizadores de palavras sufocam-se em palavras invisíveis, os tokenizadores de caracteres aumentam o comprimento da sequência, os tokenizadores de subpalavras dividem a diferença, cada Mestrado moderno é um.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## O problema

O seu vocabulário tem 50.000 palavras. Um usuário digita "inconizable". O seu tokenizer retorna.`[UNK]`O modelo agora não tem sinal sobre a palavra. O pior: o documento de 90 percentual no seu corpus tem 40 palavras raras, o que significa 40 bits de informação perdida por documento.

A tokenização de subpalavras resolve isso. Palavras comuns permanecem tokens únicos. Palavras raras se descomponem em peças significativas:`untokenizable`→ `un`- Não .`token`- Não .`izable`Os dados de treinamento cobrem tudo porque qualquer cadeia é, em última análise, uma sequência de bytes.

Cada LLM de fronteira em 2026 é enviado em um dos três algoritmos (BPE, Unigram, WordPiece), envolto em uma das três bibliotecas (tiktoken, SentencePiece, HF Tokenizers).

## O conceito

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).**Comece com um vocabulário de nível de caracteres. Conte cada par adjacente. Combine o par mais frequente em um novo token. Repita até atingir o tamanho do vocabulário alvo. Algoritmo dominante: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.**O mesmo algoritmo, mas com bytes brutos (256 tokens base) em vez de caracteres Unicode.`[UNK]`Tokens  qualquer código de sequência de byte. GPT-2 usa 50.257 tokens (256 bytes + 50.000 fusões + 1 especial).

**Unigram.**Comece com um vocabulário enorme. atribuir a cada token uma probabilidade de unigrama. Iterativamente poda tokens cuja remoção aumenta o mínimo a probabilidade de registro do corpus. Provavelmente na inferência: pode amostrar tokenizations (útil para aumento de dados através da regularização de subpalavras).

**WordPiece.**Combinar pares que maximizam a probabilidade do corpo de treinamento em vez de freqüência bruta.

**SentencePiece vs tiktoken.**SentencePiece é a biblioteca que *treina* vocabulários (BPE ou Unigram) diretamente em texto Unicode bruto, codificando o espaço branco como `▁`. tiktoken é o codificador rápido da OpenAI contra vocabulários pré-construídos; não treina.

Regra geral:

- **Training a new vocabulary:**SentencePiece (multilíngue, sem pre-tokenization) ou HF Tokenizers.
- **Fast inference against GPT vocab:**Tiktoken (cl100k_base, o200k_base).
- **Both:**HF Tokenizers  uma biblioteca, formação + serviço.

```figure
bpe-merge
```

## Construí-lo

### Passo 1: BPE a partir do zero

Veja .`code/main.py`O ciclo:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Três fatos que o algoritmo codifica.`</w>`Marcas de termo final para que "baixo" (sufixo) e "baixo" (prefixo) permanecem distintos. ponderação de frequência faz os pares de alta frequência ganhar cedo.

### Passo 2: codificar com as fusões aprendidas

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Ingênuo .$O(n \cdot \lvert \text{merges} \rvert)$. As implementações de produção (tiktoken, HF Tokenizers) utilizam uma pesquisa de posições de fusão com filas de prioridade e executam-se em tempo quase linear.

### Passo 3: SentençaPiece na prática

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Nota: não é necessária a pré-tokenization, espaço codificado como `▁`- Não .`character_coverage`Controles como caracteres agressivamente raros são preservados versus mapeados para `<unk>`- Não .

### Passo 4: tiktoken para vocabulários compatíveis com o OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Apenas codificação. Rapido (backend Rust). Exactamente correspondente com a tokenization GPT-4/5 para contagem de byte, estimativa de custos, orçamento de janela de contexto.

## Encurralagens que ainda se lançam em 2026

- **Tokenizer drift.**Treinamento em vocabulário A, implantação contra vocabulário B. Identificadores de tokens diferem; modelo de saída lixo.`tokenizer.json`- O hash em CI.
- **Whitespace ambiguity.**BPE "olá" vs "olá" produz diferentes tokens.`add_special_tokens`E ...`add_prefix_space`- Explicitamente.
- **Multilingual undertraining.**Corporações pesadas em inglês produzem vocabulários que dividem escrituras não latinas em 5-10x mais tokens.
- **Emoji splits.**Um único emoji pode levar 5 tokens.

## Usá-lo

A pilha de 2026:

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | Train custom BPE on domain corpus, merge with base vocab |
| Edge inference, small model | Unigram (smaller vocabularies work better) |

O tamanho do vocabulário é uma decisão de escala, não uma constante. Heurística aproximada: 32k para <1B parâmetros, 50-100k para 1-10B, 200k+ para multilíngue / fronteira.

## Envia-o

Salva como`outputs/skill-bpe-vs-wordpiece.md`- Não .

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Exercícios

1. **Easy.**Treinar um BPE de 500 fusões em`code/main.py`O que é que é um "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" de "toko" " " " "toko" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "
2. **Medium.**Compare contagens de tokens em 100 frases da Wikipédia em inglês entre `cl100k_base`- Não .`o200k_base`, e um SentencePiece BPE que você treina com vocabulário = 32k. Relata a relação de compressão de cada.
3. **Hard.**Treinar o mesmo corpo com BPE, Unigram e WordPiece. Medir a precisão do fluxo de fundo ao usar cada um em um pequeno classificador de sentimentos.

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Greedy merge of most-frequent character pairs until target vocab size hit. |
| Byte-level BPE | No unknown tokens ever | BPE over raw 256 bytes; GPT-2 / Llama use this. |
| Unigram | Probabilistic tokenizer | Prunes from a large candidate set using log-likelihood; used by T5, Gemma. |
| SentencePiece | The whitespace one | Library that trains BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | The fast one | OpenAI's Rust-backed BPE encoder for pre-built vocabs. No training. |
| Merge list | The magic numbers | Ordered list of `(a, b) → ab` merges; inference applies in order. |
| Character coverage | How rare is too rare? | Fraction of characters in training corpus the tokenizer must cover; ~0.9995 typical. |

## Mais leitura

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) o papel BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959)O papel Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)- A biblioteca.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) Referência concisa.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) livro de cozinha + lista de codificação.
