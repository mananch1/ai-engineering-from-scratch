# Tokenization de sous-parts  BPE, WordPiece, Unigramme, SentencePiece

> Les jetons de mots s'étouffent sur des mots invisibles, les jetons de caractères augmentent la longueur de la séquence, les jetons de sous-vérités divisent la différence, chaque LLM moderne en fait un.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## Le problème

Votre vocabulaire a 50 000 mots. Un utilisateur tape "non-tokenizable". Votre tokenizer revient.`[UNK]`Le modèle n'a plus de signal sur le mot. Pire encore: le document de 90 pour cent dans votre corpus a 40 mots rares, ce qui signifie 40 bits d'informations perdues par document.

La symbolisation des sous-parts résolve cela. Les mots communs restent des jetons uniques. Les mots rares se décomposent en morceaux significatifs:`untokenizable`- Je suis là.`un`- Je suis là .`token`- Je suis là .`izable`Les données de formation couvrent tout parce que n'importe quelle chaîne est en fin de compte une séquence de bytes.

Chaque LLM frontalier en 2026 est livré sur l'un des trois algorithmes (BPE, Unigram, WordPiece), enveloppé dans une des trois bibliothèques (tiktoken, SentencePiece, HF Tokenizers).

## Le concept

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).**Commencez par un vocabulaire au niveau des caractères. Comptez chaque paire adjacente. Fusez la paire la plus fréquente dans un nouveau jeton. Répétez jusqu'à ce que vous atteigniez la taille du vocabulaire cible. Algorithme dominant: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.**Le même algorithme mais avec des octets bruts (256 jetons de base) au lieu de caractères Unicode.`[UNK]`Les tokens  sont des codes de séquences de octets. GPT-2 utilise 50 257 tokens (256 octets + 50 000 fusions + 1 spécial).

**Unigram.**Commencez par un énorme vocabulaire. Assignez à chaque jeton une probabilité de singramme. Prenez à plusieurs reprises des jetons dont la suppression augmente le moins la probabilité de log de corpus.

**WordPiece.**Les paires de fusion qui maximisent la probabilité du corpus d'entraînement plutôt que la fréquence brute.

**SentencePiece vs tiktoken.**SentencePiece est la bibliothèque qui entraîne les vocabulaires (BPE ou Unigram) directement sur le texte brut Unicode, en encodant l'espace blanc comme `▁`. tiktoken est le codeur rapide d'OpenAI contre les vocabulaires prédéfinis; il ne s'entraîne pas.

Règle générale:

- **Training a new vocabulary:**SentencePiece (multilingue, sans pré-tokenization) ou HF Tokenizers.
- **Fast inference against GPT vocab:**Il est également possible de modifier le code de la marque.
- **Both:**HF Tokenizers  une bibliothèque, formation + service.

```figure
bpe-merge
```

## Faites-le

### Étape 1: BPE à partir de zéro

Regardez !`code/main.py`- La boucle:

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

Trois faits que l'algorithme encode.`</w>`Les deux types de combinaisons sont les mêmes: les "faites" et les "faites" sont les mêmes.

### Étape 2: encoder avec les fusions apprises

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

Naïf .$O(n \cdot \lvert \text{merges} \rvert)$. Les mises en œuvre de production (tiktoken, HF Tokenizers) utilisent une recherche de classement fusionné avec des files d'attente prioritaires et fonctionnent en temps quasi linéaire.

### Étape 3: SentencePiece en pratique

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

Remarque: aucune pré-tokenization n'est requise, espace codé comme `▁`- Je suis là .`character_coverage`contrôle la conservation des caractères rares par rapport à la mappage à `<unk>`- Je suis désolé .

### Étape 4: Tiktoken pour les vocaux compatibles avec OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Rapide (backend de Rust). correspondant exactement à la jetonisation GPT-4/5 pour le décompte des octets, l'estimation des coûts, le budget de la fenêtre contextuelle.

## Des pièges qui vont encore arriver en 2026

- **Tokenizer drift.**Formation sur le vocabulaire A, déploiement contre le vocabulaire B. Les identifiants des jetons diffèrent; le modèle produit des déchets.`tokenizer.json`le hash dans l'IC.
- **Whitespace ambiguity.**BPE "hello" contre "hello" produisent des jetons différents.`add_special_tokens`et `add_prefix_space`explicitement.
- **Multilingual undertraining.**Les corpus anglais lourds produisent des vocabulaires qui divisent les scripts non latins en 5 à 10 fois plus de jetons.
- **Emoji splits.**Un seul emoji peut prendre 5 jetons.

## Utilisez-le

La pile de 2026:

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | Train custom BPE on domain corpus, merge with base vocab |
| Edge inference, small model | Unigram (smaller vocabularies work better) |

La taille du vocabulaire est une décision d'échelle, pas une constante. Heuristique approximative: 32k pour < 1B paramètres, 50-100k pour 1-10B, 200k+ pour multilingue / frontière.

## La faire partir

- Je ne sais pas .`outputs/skill-bpe-vs-wordpiece.md`- Le numéro de la liste:

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

## Exercices

1. **Easy.**On met en place un BPE à 500 fusions`code/main.py`Encode trois mots détenus. Combien ont produit exactement 1 jeton contre > 1 jeton?
2. **Medium.**Comparer le nombre de jetons sur 100 phrases de Wikipédia en anglais entre `cl100k_base`- Je suis là .`o200k_base`, et un BPE SentencePiece que vous entraînez avec le vocabulaire = 32k.
3. **Hard.**Exercez le même corpus avec BPE, Unigram et WordPiece. Mesurez la précision en aval lorsque vous utilisez chacun sur un petit classificateur de sentiment. Le choix déplace-t-il l'aiguille de plus d'un point F1?

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Greedy merge of most-frequent character pairs until target vocab size hit. |
| Byte-level BPE | No unknown tokens ever | BPE over raw 256 bytes; GPT-2 / Llama use this. |
| Unigram | Probabilistic tokenizer | Prunes from a large candidate set using log-likelihood; used by T5, Gemma. |
| SentencePiece | The whitespace one | Library that trains BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | The fast one | OpenAI's Rust-backed BPE encoder for pre-built vocabs. No training. |
| Merge list | The magic numbers | Ordered list of `(a, b) → ab` merges; inference applies in order. |
| Character coverage | How rare is too rare? | Fraction of characters in training corpus the tokenizer must cover; ~0.9995 typical. |

## Pour en savoir plus

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) le papier BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) le journal Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)La bibliothèque.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) référence concise.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) livre de cuisine + liste de codage.
