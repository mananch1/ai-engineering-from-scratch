# نشانه گذاری زیر کلمه  BPE, WordPiece, Unigram, SentencePiece

> توکنيزر کلمات به کلمات ناشناخته خفه می شوند توکنيزر شخصیت طول دنباله را بالا می برد توکنيزر زیر کلمه تفاوت را تقسیم می کند هر مدرک مدرن بر روی یک می فرستد

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## مشکل

ذخيره لغت شما 50 هزار کلمه داره. يک کاربر "غیر قابل توکنيزيشن" ميگيره. توکنيزر شما باز مياد`[UNK]`حالا مدل هیچ سیگنال در مورد کلمه ای ندارد. بدتر از این، سند 90 درصد در کورپوس شما 40 کلمه نادر دارد، که به معنای 40 بیت اطلاعات از دست رفته در هر سند است.

نشانه گذاری زیر کلمه این مسئله را حل می کند. کلمات مشترک تنها نشانه هایی باقی می مانند. کلمات نادر به قطعات معنی دار تجزیه می شوند:`untokenizable`→ `un`،`token`،`izable`داده های آموزش همه چیز را پوشش می دهد چون هر رشته ای در نهایت یک ردیف بایت است.

هر رشته ی LLM مرزی در سال 2026 بر روی یکی از سه الگوریتم (BPE، Unigram، WordPiece) ارسال می شود، در یکی از سه کتابخانه (tiktoken، SentencePiece، HF Tokenizers) بسته شده است. شما نمی توانید یک مدل زبان را بدون انتخاب یکی ارسال کنید.

## مفهوم

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).**با یک لغت سطح شخصیت شروع کنید. هر جفت همسایه را بشمارید. رایج ترین جفت را به یک توکن جدید ادغام کنید. تا اندازه لغت هدف را بدست آورید تکرار کنید. الگوریتم غالب: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.**همان الگوریتم اما با بیت خام (256 توکن پایه) به جای کاراکترهای یونیکوید.`[UNK]`توکن ها  هر کد ردیابی بائیت. GPT-2 از 50،257 توکن استفاده می کند (256 بائیت + 50،000 ادغام + 1 ویژه).

**Unigram.**با یک لغت بزرگ شروع کنید. به هر توکن یک احتمال unigram اختصاص دهید. توکن هایی را که حذف آنها حداقل احتمال ثبت corpus را افزایش می دهد، به طور مکرر برش دهید. احتمال در نتیجه گیری: می تواند توکن ها را نمونه کند (فایدمند برای افزایش داده ها از طریق تنظیم فرعی کلمات). توسط T5 ، mBART ، ALBERT ، XLNet ، Gemma استفاده می شود.

**WordPiece.**جفت های ادغام که احتمالات کارپوس آموزش را به جای فرکانس خام افزایش می دهند.

**SentencePiece vs tiktoken.**SentencePiece کتابخانه ای است که * آموزش * لغات (BPE یا Unigram) را مستقیماً بر روی متن یونیکود خام، کد فضای سفید به عنوان `▁`. tiktoken یک کدرس سریع OpenAI برای استفاده از لغات های پیش ساخته است؛ این آموزش نیست.

قانون عمومي:

- **Training a new vocabulary:**SentencePiece (متعدد زبانه، بدون قبل از توکن سازی) یا HF Tokenizers.
- **Fast inference against GPT vocab:**تیتوکن (cl100k_base, o200k_base).
- **Both:**HF Tokenizers  یک کتابخانه، آموزش + خدمت.

```figure
bpe-merge
```

## آن را بسازید

### مرحله ی اول: BPE از ابتدا

ببین`code/main.py`. حلقه:

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

سه تا از حقيقتهايي که الگوریتم رمزنگاري ميکنه`</w>`علامت های پایان کلمه به طوری که "کم" (تکلیف) و "کم" (پرکس) باقی می ماند متمایز. وزن فرکانس باعث می شود جفت های فرکانس بالا زودتر برنده شوند. لیست ترکیب ترتیب داده شده است  نتیجه گیری در ترتیب آموزش استفاده می شود.

### مرحله دوم: با ادغام های آموخته رمزگذاری کنید

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

ساده .$O(n \cdot \lvert \text{merges} \rvert)$. پیاده سازی های تولید (تککن ها، توکن های HF) از جستجوی رتبه های ادغام با صف های اولویت استفاده می کنند و در زمان تقریبا خطی اجرا می شوند.

### مرحله سوم: جملاتقطعه در عمل

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

توجه: نیازی به پیش از توکن شدن نیست، فضای رمزگذاری شده به عنوان `▁`،`character_coverage`کنترل می کند که چگونه شخصیت های نادر به شدت حفظ می شوند و یا به `<unk>`. .

### مرحله 4: تکتون برای لغات های سازگار با OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

فقط کدگذاری. سریع (Rust backend). دقیقا با توکن GPT-4/5 برای بایت شمارش، تخمین هزینه، بودجه بندی پنجره های زمینه مطابقت دارد.

## خطرهایی که هنوز در سال 2026 وجود دارند

- **Tokenizer drift.**آموزش در لغت A، استفاده در مقابل لغت B. شناسه های توکن متفاوت هستند، مدل تولید زباله می کند. چک`tokenizer.json`هاشي در IC
- **Whitespace ambiguity.**BPE "سلام" در مقابل "سلام" توکن های مختلف تولید می کند. همیشه مشخص کنید `add_special_tokens`و`add_prefix_space`به طور صریح
- **Multilingual undertraining.**در زبان انگلیسی، corpus های سنگین لغات تولید می کنند که اسکریپت های غیر لاتین را به ۵ تا ۱۰ برابر بیشتر توکن تقسیم می کنند. همان پرامپت ۵ تا ۱۰ برابر بیشتر در زبان ژاپنی / عربی در GPT-3.5 هزینه می کند. o200k_base تا حدی این را حل کرد.
- **Emoji splits.**يک ايموجي ميتونه 5 تا توکن داشته باشه.

## ازش استفاده کن

دسته 2026:

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | Train custom BPE on domain corpus, merge with base vocab |
| Edge inference, small model | Unigram (smaller vocabularies work better) |

اندازه لغت یک تصمیم مقیاس بندی است، نه ثابت. هوریستیک خشن: 32k برای <1B پارامتر، 50-100k برای 1-10B، 200k+ برای چندزبان / مرز.

## -باده

پس از`outputs/skill-bpe-vs-wordpiece.md`:

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

## تمرینات

1. **Easy.**يه BPE 500 ميليک رو راه بنداز`code/main.py`این یک کارپوس کوچک است. سه کلمه ای را رمزگذاری کنید. چند تا دقیقا 1 توکن در مقابل 1 توکن تولید کردند؟
2. **Medium.**تعداد توکن ها را در 100 جمله ویکی پدیا انگلیسی با `cl100k_base`،`o200k_base`و یک BPE SentencePiece که با لغت آموزش میدی 32k گزارش نسبت فشرده سازی هر یک
3. **Hard.**با استفاده از BPE، Unigram و WordPiece، یک کارپوس مشابه را تمرین کنید. دقت پایین تر را هنگام استفاده از هر یک از آنها در یک طبقه بندی کننده کوچک احساس اندازه گیری کنید. آیا انتخاب سوزن را بیش از 1 نقطه F1 حرکت می دهد؟

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Greedy merge of most-frequent character pairs until target vocab size hit. |
| Byte-level BPE | No unknown tokens ever | BPE over raw 256 bytes; GPT-2 / Llama use this. |
| Unigram | Probabilistic tokenizer | Prunes from a large candidate set using log-likelihood; used by T5, Gemma. |
| SentencePiece | The whitespace one | Library that trains BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | The fast one | OpenAI's Rust-backed BPE encoder for pre-built vocabs. No training. |
| Merge list | The magic numbers | Ordered list of `(a, b) → ab` merges; inference applies in order. |
| Character coverage | How rare is too rare? | Fraction of characters in training corpus the tokenizer must cover; ~0.9995 typical. |

## خواندن بیشتر

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) کاغذ BPE
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) روزنامه ی یونیگرام
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) کتابخانه
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) اشاره خلاصه ای
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) کتاب آشپزی + لیست کدگذاری
