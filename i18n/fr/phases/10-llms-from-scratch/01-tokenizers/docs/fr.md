# Les symboles: BPE, WordPiece, SentencePiece

> Votre LLM ne lit pas l'anglais, il lit les nombres entiers. Le tokenizer décide si ces nombres entiers ont une signification ou le gaspillent.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 05 (NLP Foundations)
**Time:** ~90 minutes

## Objectifs d'apprentissage

- Implémenter des algorithmes de jetonage BPE, WordPiece et Unigram à partir de zéro et comparer leurs stratégies de fusion
- Expliquez comment la taille du vocabulaire affecte l'efficacité du modèle: trop petit crée de longues séquences, trop grand déchet incrustant des paramètres
- Analyse des objets de tokenization dans les langues et les codes, en identifiant les endroits où des tokenizers spécifiques se décomposent
- Utilisez les bibliothèques de jetons et de pièces de phrase pour symboliser le texte et inspecter les identifiants de jetons résultants

## Le problème

Votre LLM ne lit pas l'anglais, il ne lit aucune langue, il lit les chiffres.

Le vide entre "Hello, world!" et [15496, 11, 995, 0] est le jeton. Chaque mot, chaque espace, chaque marque de ponctuation doit être converti en un entier avant qu'un modèle puisse le traiter. Cette conversion n'est pas neutre.

Si vous vous trompez, votre modèle gaspille la capacité de codage des mots communs avec plusieurs jetons. "malheureusement" devient quatre jetons au lieu d'un. Votre fenêtre de contexte 128K a rétréci de 75% pour le texte lourd en mots multiesyllabiques. Faites-le bien et la même fenêtre contextuelle contient deux fois plus de sens. La différence entre "ce modèle gère bien le code" et "ce modèle étouffe Python" se résume souvent à la façon dont le tokeniseur a été formé.

Chaque appel d'API que vous faites à GPT-4 ou Claude est évalué par jeton. Chaque jeton que votre modèle génère coûte de calcul. Plus le nombre de jetons requis pour représenter une sortie est faible, plus l'inférence de bout en bout est rapide.

## Le concept

### Trois méthodes qui ont échoué (et une qui a gagné)

Il y a trois façons évidentes de convertir le texte en chiffres.

**Word-level tokenization**Le mot "cat sat" devient simple. Mais qu'en est-il de la "tokenization"? ou "GPT-4o"? ou d'un mot composé allemand comme "Geschwindigkeitsbegrenzung"? Le niveau des mots nécessite un vocabulaire massif pour couvrir chaque mot dans chaque langue.`[UNK]`token -- la façon dont le modèle dit "je n'ai aucune idée de ce que c'est". L'anglais seul a plus d'un million de formes de mots. Ajoutez du code, des URL, des notations scientifiques et 100 autres langues et vous avez besoin d'un vocabulaire infini.

**Character-level tokenization**Le vocabulaire est minuscule (quelques centaines de caractères). Aucun jeton inconnu jamais. Mais les séquences deviennent extrêmement longues. Une phrase qui serait de 10 jetons au niveau de mots devient de 50 jetons au niveau de caractères. Le modèle doit apprendre que "t", "h", "e" ensemble signifient "le" - la capacité d'attention brûlante sur quelque chose que l'homme apprend à l'âge de trois ans.

**Subword tokenization**Les mots communs restent entiers: "le" est un symbole. Les mots rares se décomposent en morceaux significatifs: "malheur" devient ["un", "happy", "ness"]. Le vocabulaire reste gérable (30K à 128K de jetons). Les séquences restent courtes. Les jetons inconnus disparaissent essentiellement parce que n'importe quel mot peut être construit à partir de morceaux de sous-parts.

Chaque LLM moderne utilise la symbolisation de sous-parts. GPT-2, GPT-4, BERT, Llama 3, Claude - tous. La question est de savoir quel algorithme.

```mermaid
graph TD
    A["Text: 'unhappiness'"] --> B{"Tokenization Strategy"}
    B -->|Word-level| C["['unhappiness']\n1 token if in vocab\n[UNK] if not"]
    B -->|Character-level| D["['u','n','h','a','p','p','i','n','e','s','s']\n11 tokens"]
    B -->|Subword BPE| E["['un','happi','ness']\n3 tokens"]

    style C fill:#ff6b6b,color:#fff
    style D fill:#ffa500,color:#fff
    style E fill:#51cf66,color:#fff
```

### BPE: Codage par paire de octets

BPE est un algorithme de compression avide réutilisé pour la tokenization.

Commencez par des caractères individuels. Comptez chaque paire adjacente dans le corpus d'entraînement. Fusez la paire la plus fréquente dans un nouveau jeton. Répétez jusqu'à ce que vous atteigniez la taille de votre vocabulaire cible.

```figure
tokenizer-bpe
```

Voici le BPE sur un petit corpus avec les mots "inférieur", "inférieur" et "nouveau":

```
Corpus (with word frequencies):
  "lower"  x5
  "lowest" x2
  "newest" x6

Step 0 -- Start with characters:
  l o w e r       (x5)
  l o w e s t     (x2)
  n e w e s t     (x6)

Step 1 -- Count adjacent pairs:
  (e,s): 8    (s,t): 8    (l,o): 7    (o,w): 7
  (w,e): 13   (e,r): 5    (n,e): 6    ...

Step 2 -- Merge most frequent pair (w,e) -> "we":
  l o we r        (x5)
  l o we s t      (x2)
  n e we s t      (x6)

Step 3 -- Recount and merge (e,s) -> "es":
  l o we r        (x5)
  l o we s t      (x2)    <- 'es' only forms from 'e'+'s', not 'we'+'s'
  n e we s t      (x6)    <- wait, the 'e' before 'we' and 's' after 'we'

Actually tracking this precisely:
  After "we" merge, remaining pairs:
  (l,o): 7   (o,we): 7   (we,r): 5   (we,s): 8
  (s,t): 8   (n,e): 6    (e,we): 6

Step 3 -- Merge (we,s) -> "wes" or (s,t) -> "st" (tied at 8, pick first):
  Merge (we,s) -> "wes":
  l o we r        (x5)
  l o wes t       (x2)
  n e wes t       (x6)

Step 4 -- Merge (wes,t) -> "west":
  l o we r        (x5)
  l o west        (x2)
  n e west        (x6)

...continue until target vocab size reached.
```

La table de fusion est le tokenizer. Pour coder un nouveau texte, appliquez des fusions dans l'ordre dans lequel elles ont été apprises. Le corpus de formation détermine quelles fusions existent, et ce choix façonne définitivement ce que le modèle voit.

```mermaid
graph LR
    subgraph Training["BPE Training Loop"]
        direction TB
        T1["Start: character vocabulary"] --> T2["Count all adjacent pairs"]
        T2 --> T3["Merge most frequent pair"]
        T3 --> T4["Add merged token to vocab"]
        T4 --> T5{"Reached target\nvocab size?"}
        T5 -->|No| T2
        T5 -->|Yes| T6["Done: save merge table"]
    end
```

### BPE de niveau octal (GPT-2, GPT-3, GPT-4)

Le BPE standard fonctionne sur des caractères Unicode. Le BPE de niveau octet fonctionne sur des octets bruts (0-255). Cela vous donne un vocabulaire de base de exactement 256, gère n'importe quel langage ou codage et ne produit jamais un jeton inconnu.

GPT-2 a introduit cette approche. Le vocabulaire de base couvre tous les octets possibles. BPE fusionne construit en plus de cela.

- GPT-2: 50 257 jetons
- GPT-3.5/GPT-4: ~100,256 jetons (encoding cl100k_base)
- GPT-4o: 200 019 jetons (encoding de base de 200 k)

### Le code de référence

WordPiece ressemble à BPE mais choisit de fusionner différemment. Au lieu de fréquence brute, il maximize la probabilité des données de formation:

```
BPE merge criterion:      count(A, B)
WordPiece merge criterion: count(AB) / (count(A) * count(B))
```

Le BPE demande: " Quel couple apparaît le plus souvent ? " WordPiece demande: " Quel couple apparaît plus souvent ensemble que vous ne le pensez par hasard ? " Cette différence subtile produit différents vocabulaires.

WordPiece utilise également un préfixe "##" pour les sous-parts de la continuation:

```
"unhappiness" -> ["un", "##happi", "##ness"]
"embedding"   -> ["em", "##bed", "##ding"]
```

Le préfixe "##" vous indique que cette pièce continue un jeton précédent. BERT utilise WordPiece avec un vocabulaire de 30 522 jetons.

### La phrase "Piece" (Llama, T5)

SentencePiece traite les entrées comme un flux brut de caractères Unicode, y compris l'espace blanc. Aucune étape de pré-tokenization. Aucune règle spécifique pour la langue sur les limites des mots. Cela le rend véritablement linguistique - il fonctionne sur le chinois, le japonais, le thaïlandais et d'autres langues où les espaces ne séparent pas les mots.

SentencePiece prend en charge deux algorithmes:
- **BPE mode**: la même logique de fusion que la BPE standard, appliquée aux séquences de caractères bruts
- **Unigram mode**Le contraire de BPE - prune au lieu de fusionner.

Llama 2 utilise SentencePiece BPE avec un vocabulaire de 32 000 jetons. T5 utilise SentencePiece Unigram avec 32 000 jetons.

### Comptes de taille du vocabulaire

C'est une décision d'ingénierie avec des conséquences mesurables.

```mermaid
graph LR
    subgraph Small["Small Vocab (32K)\ne.g., BERT, T5"]
        S1["More tokens per text"]
        S2["Longer sequences"]
        S3["Smaller embedding matrix"]
        S4["Better rare-word handling"]
    end
    subgraph Large["Large Vocab (128K+)\ne.g., Llama 3, GPT-4o"]
        L1["Fewer tokens per text"]
        L2["Shorter sequences"]
        L3["Larger embedding matrix"]
        L4["Faster inference"]
    end
```

Pour un vocabulaire de 128K avec des embeds de 4 096 dimensions, la matrice d'embedding seule est de 128 000 x 4 096 = 524 millions de paramètres. Pour un vocabulaire de 32K, il est de 131 millions de paramètres. C'est une différence de paramètres de 400M par rapport au choix du tokenizer seul.

Mais les vocabulaires plus grands comprennent le texte plus agressivement. Le même paragraphe anglais qui prend 100 jetons avec un vocabulaire de 32K pourrait prendre 70 jetons avec un vocabulaire de 128K. Cela signifie 30% moins de passes avant pendant la génération. Pour un modèle qui répond à des millions de demandes, c'est une réduction directe du coût de calcul.

La tendance est claire: les tailles du vocabulaire augmentent. GPT-2 utilise 50 257. GPT-4 utilise environ 100 K. Llama 3 utilise 128 K. GPT-4o utilise 200 K.

| Model | Vocab Size | Tokenizer Type | Avg Tokens per English Word |
|-------|-----------|----------------|---------------------------|
| BERT | 30,522 | WordPiece | ~1.4 |
| GPT-2 | 50,257 | Byte-level BPE | ~1.3 |
| Llama 2 | 32,000 | SentencePiece BPE | ~1.4 |
| GPT-4 | ~100,256 | Byte-level BPE | ~1.2 |
| Llama 3 | 128,256 | Byte-level BPE (tiktoken) | ~1.1 |
| GPT-4o | 200,019 | Byte-level BPE | ~1.0 |

### La taxe multilingue

Les tokenizers formés principalement en anglais sont brutaux envers les autres langues. Le texte coréen dans le tokenizer de GPT-2 a en moyenne 2-3 tokens par mot. Le chinois peut être pire. Cela signifie qu'un utilisateur coréen a effectivement une fenêtre de contexte qui est la moitié de la taille d'un utilisateur anglais - payant le même prix pour une densité d'information moindre.

C'est pourquoi Llama 3 a quadruplé son vocabulaire de 32 000 à 128 000 Tokens plus dédiés aux scripts non anglais signifie une compression plus équitable entre les langues.

```figure
tokenizer-tradeoff
```

## Faites-le

### Étape 1: Tokenizer au niveau des caractères

Un jeton au niveau des caractères cartographiera chaque caractère à son point de code Unicode. Pas besoin de formation. Pas de jetons inconnus.

```python
class CharTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)
```

"bonjour" devient [104, 101, 108, 108, 111]. Chaque personnage est son propre jeton.

### Étape 2: Tokenizer BPE à partir de zéro

La vraie mise en œuvre. Nous nous entraînons sur les octets bruts (comme GPT-2), compter les paires, fusionner les plus fréquents, et enregistrer chaque fusion dans l'ordre.

```python
from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def _get_pairs(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, text, num_merges):
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = 256 + i
            tokens = self._merge_pair(tokens, best_pair, new_token)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        return self

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for pair, new_token in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_token)
        return tokens

    def decode(self, tokens):
        byte_sequence = b"".join(self.vocab[t] for t in tokens)
        return byte_sequence.decode("utf-8", errors="replace")
```

La boucle d'entraînement est le noyau de BPE: compter les paires, fusionner le gagnant, répéter.`num_merges`Les résultats de la recherche ont été obtenus en fonction des résultats obtenus.

Le codage applique les fusions dans l'ordre exact dans lequel elles ont été apprises. Cela compte. Si la fusion 1 crée "th" et la fusion 5 crée "the", le codage doit d'abord appliquer la fusion 1 afin que "the" puisse se former à partir de "th" + "e" dans la fusion 5.

Le décodeur est l'inverse: recherchez chaque identifiant de jeton dans le vocabulaire, concateniez les octets, décodez en UTF-8.

### Étape 3: Encode et décodeur de la route

```python
corpus = (
    "The cat sat on the mat. The cat ate the rat. "
    "The dog sat on the log. The dog ate the frog. "
    "Natural language processing is the study of how computers "
    "understand and generate human language. "
    "Tokenization is the first step in any NLP pipeline."
)

tokenizer = BPETokenizer()
tokenizer.train(corpus, num_merges=40)

test_sentences = [
    "The cat sat on the mat.",
    "Natural language processing",
    "tokenization pipeline",
    "unhappiness",
]

for sentence in test_sentences:
    encoded = tokenizer.encode(sentence)
    decoded = tokenizer.decode(encoded)
    raw_bytes = len(sentence.encode("utf-8"))
    ratio = len(encoded) / raw_bytes
    print(f"'{sentence}'")
    print(f"  Tokens: {len(encoded)} (from {raw_bytes} bytes) -- ratio: {ratio:.2f}")
    print(f"  Roundtrip: {'PASS' if decoded == sentence else 'FAIL'}")
```

Le ratio de compression vous indique l'efficacité du tokenizer. Un ratio de 0,50 signifie que le tokenizer a comprimé le texte à la moitié du nombre de jetons que les octets bruts. Plus bas, c'est mieux. Sur le corps d'entraînement, le rapport sera bon. Sur le texte hors distribution comme "malheur" (qui ne figure pas dans le corpus), le rapport sera pire - le tokenizer revient à l'encodage au niveau des caractères pour les modèles invisibles.

### Étape 4: Comparer avec tiktoken

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

texts = [
    "The cat sat on the mat.",
    "unhappiness",
    "Hello, world!",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Geschwindigkeitsbegrenzung",
]

for text in texts:
    our_tokens = tokenizer.encode(text)
    tiktoken_tokens = enc.encode(text)
    tiktoken_pieces = [enc.decode([t]) for t in tiktoken_tokens]
    print(f"'{text}'")
    print(f"  Our BPE:   {len(our_tokens)} tokens")
    print(f"  tiktoken:  {len(tiktoken_tokens)} tokens -> {tiktoken_pieces}")
```

TikTok utilise exactement le même algorithme mais entraîné sur des centaines de gigaoctets de texte avec 100 000 fusions. L'algorithme est identique. La différence est les données de formation et le nombre de fusions. Votre tokeniseur entraîné sur un paragraphe avec 40 fusions ne peut pas rivaliser avec les 100K de tiktoken fusion sur un corpus massif. Mais le mécanisme est le même.

### Étape 5: Analyse du vocabulaire

```python
def analyze_vocabulary(tokenizer, test_texts):
    total_tokens = 0
    total_chars = 0
    token_usage = Counter()

    for text in test_texts:
        encoded = tokenizer.encode(text)
        total_tokens += len(encoded)
        total_chars += len(text)
        for t in encoded:
            token_usage[t] += 1

    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Total tokens across all texts: {total_tokens}")
    print(f"Total characters: {total_chars}")
    print(f"Avg tokens per character: {total_tokens / total_chars:.2f}")

    print(f"\nMost used tokens:")
    for token_id, count in token_usage.most_common(10):
        token_bytes = tokenizer.vocab[token_id]
        display = token_bytes.decode("utf-8", errors="replace")
        print(f"  Token {token_id:4d}: '{display}' (used {count} times)")

    unused = [t for t in tokenizer.vocab if t not in token_usage]
    print(f"\nUnused tokens: {len(unused)} out of {len(tokenizer.vocab)}")
```

Cela révèle la distribution Zipf dans votre vocabulaire. Quelques jetons dominent (espaces, "le", "e"). La plupart des jetons sont rarement utilisés. Les jetons de production optimisent cette distribution - les modèles communs obtiennent des identifiants de jetons courts, les modèles rares obtiennent des représentations plus longues.

## Utilisez-le

Votre BPE fonctionne, voyez à quoi ressemblent les outils de production.

### Tickets (OpenAI)

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "Tokenizers convert text to integers"
tokens = enc.encode(text)
print(f"Tokens: {tokens}")
print(f"Pieces: {[enc.decode([t]) for t in tokens]}")
print(f"Roundtrip: {enc.decode(tokens)}")
```

tiktoken est écrit en Rust avec des liaisons Python. Il encode des millions de jetons par seconde. Le même algorithme BPE, une mise en œuvre de force industrielle.

### Les symboles de visage

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(vocab_size=1000, special_tokens=["<pad>", "<eos>", "<unk>"])
tokenizer.train(["corpus.txt"], trainer)

output = tokenizer.encode("The cat sat on the mat.")
print(f"Tokens: {output.tokens}")
print(f"IDs: {output.ids}")
```

La bibliothèque de jetons Hugging Face est également Rust sous le capot. Elle entraîne BPE sur des corpora à l'échelle de gigabyte en quelques secondes. C'est ce que vous utilisez pour entraîner votre propre modèle.

### Chargement du jeton de Llama

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

text = "Tokenizers are the unsung heroes of LLMs"
tokens = tokenizer.encode(text)
print(f"Token IDs: {tokens}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
print(f"Vocab size: {tokenizer.vocab_size}")

multilingual = ["Hello world", "Hola mundo", "Bonjour le monde"]
for text in multilingual:
    ids = tokenizer.encode(text)
    print(f"'{text}' -> {len(ids)} tokens")
```

Le vocabulaire 128K de Llama 3 comprime le texte non anglais nettement mieux que le vocabulaire 50K de GPT-2. Vous pouvez vérifier cela vous-même - encoder la même phrase dans plusieurs langues et compter les jetons.

## La faire partir

Cette leçon produit `outputs/prompt-tokenizer-analyzer.md`-- une requête réutilisable qui analyse l'efficacité de la jetonisation pour toute combinaison de texte et de modèle.

## Exercices

1. Modifiez le tokenizer BPE pour imprimer le vocabulaire à chaque étape de fusion. Observez comment "t" + "h" devient "th", puis "th" + "e" devient "the". Suivez comment les mots anglais communs se rassemblent pièce par pièce.

2. Ajouter des jetons spéciaux (`<pad>`- Je suis là .`<eos>`- Je suis là .`<unk>`) au tokenizer BPE. attribuer les identifiants 0, 1, 2 et déplacer tous les autres tokens en conséquence.

3. Appliquez le critère de fusion WordPiece (ratio de probabilité au lieu de fréquence). Exercez à la fois BPE et WordPiece sur le même corpus avec le même nombre de fusions. Comparer les vocabulaires résultants - lequel produit des sous-words plus significatifs linguistiquement?

4. Construisez un benchmark d'efficacité du tokenizer multilingue. Prenez 10 phrases en anglais, espagnol, chinois, coréen et arabe. Tokenize chacune avec un tiktoken (cl100k_base) et mesurez les tokens moyens par caractère. Quantifiez la "taxe multilingue" pour chaque langue.

5. Prenez votre jeton BPE sur un corpus plus grand (téléchargez un article de Wikipédia). Ajoutez le nombre de fusions pour obtenir un ratio de compression de 10% de tiktoken sur le même texte. Cela vous oblige à comprendre la relation entre la taille du corpus, le nombre de fusions et la qualité de compression.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Token | "A word" | A unit in the model's vocabulary -- could be a character, subword, word, or multi-word chunk |
| BPE | "Some compression thing" | Byte Pair Encoding -- iteratively merge the most frequent adjacent pair of tokens until the target vocabulary size is reached |
| WordPiece | "BERT's tokenizer" | Like BPE but merges maximize the likelihood ratio count(AB)/(count(A)*count(B)) instead of raw frequency |
| SentencePiece | "A tokenizer library" | A language-agnostic tokenizer that operates on raw Unicode without pre-tokenization, supporting BPE and Unigram algorithms |
| Vocabulary size | "How many words it knows" | The total number of unique tokens: GPT-2 has 50,257, BERT has 30,522, Llama 3 has 128,256 |
| Fertility | "Not a tokenizer term" | Average number of tokens per word -- measures tokenizer efficiency across languages (1.0 is perfect, 3.0 means the model works three times harder) |
| Byte-level BPE | "GPT's tokenizer" | BPE operating on raw bytes (0-255) instead of Unicode characters, guaranteeing no unknown tokens for any input |
| Merge table | "The tokenizer file" | Ordered list of pair merges learned during training -- this IS the tokenizer, and order matters |
| Pre-tokenization | "Splitting on spaces" | Rules applied before subword tokenization: whitespace splitting, digit separation, punctuation handling |
| Compression ratio | "How efficient the tokenizer is" | Tokens produced divided by input bytes -- lower means better compression and faster inference |

## Pour en savoir plus

- [Sennrich et al., 2016 -- "Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909)-- le document qui a introduit le BPE pour la PNL, transformant un algorithme de compression de 1994 en la base de la tokenization moderne
- [Kudo & Richardson, 2018 -- "SentencePiece: A simple and language independent subword tokenizer"](https://arxiv.org/abs/1808.06226)-- la symbolisation linguistique-agnostique qui a rendu les modèles multilingues pratiques
- [OpenAI tiktoken repository](https://github.com/openai/tiktoken)-- mise en œuvre de la production BPE dans Rust avec liaisons Python, utilisée par GPT-3.5/4/4o
- [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers)-- formation en jeton de qualité de production avec performances Rust
