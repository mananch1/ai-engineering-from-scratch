# ترانسفورماتور کامل  کدگر + کدگر

> توجه ستاره است. همه چیز دیگه  بقای، عادی سازی، ارسال، توجه متقابل  است که به شما اجازه می دهد تا آن را عمیق.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## مشکل

یک لایه توجه واحد یک استخراج کننده ویژگی است، نه یک مدل. یک ماتمل در هر لایه ظرفیت کافی برای زبان نیست. شما نیاز به عمق  و شکاف عمق بدون لوله کشی مناسب دارید.

مقاله Vaswani 2017 شش تصمیم طراحی را بسته بندی کرد که یک لایه توجه را به یک بلوک قابل جمع بندی تبدیل کرد. هر ترانسفورماتور از زمان  کدگر فقط (BERT) ، کدگر فقط (GPT) ، کدگر-دکودر (T5)  به میراث همان اسکلت می رود. در سال 2026 بلوک ها (RMSNorm ، SwiGLU ، pre-norm ، RoPE) اصلاح شده اند اما اسکلت یکسان است.

این درس اسکلت است. درس های بعدی تخصصی آن را  06 برای کدرها، 07 برای کدرها، 08 برای کدرها-دکودرها.

## مفهوم

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### شش قطعه

1. **Embedding + positional signal.**توکن ها → ویکتورها. موقعیت تزریق شده از طریق RoPE (مودرن) یا سینوساید (کلاسیک).
2. **Self-attention.**هر موقعیت به هر موقعیت دیگه ای کمک میکنه
3. **Feed-forward network (FFN).**دو لایه MLP در جهت موقعیت: `W_2 · activation(W_1 · x)`. نسبت گسترش 4× به طور پیش فرض
4. **Residual connection.** `x + sublayer(x)`بدون اين، گرادينت ها بعد از 6 لايه از بين ميرن
5. **Layer normalization.** `LayerNorm`یا`RMSNorm`. (مودرن) ثابت ميکنه که جریان باقي مانده
6. **Cross-attention (decoder only).**سوالات از کدگر، کلید ها و ارزش های از محصول کدگر می آیند.

یک ویکتور را از طریق یک بلوک مشاهده کنید: توجه در میان موقعیت ها مخلوط می شود، باقی مانده آن را به جلو حمل می کند، FFN آن را تبدیل می کند و نورم جریان را پایدار نگه می دارد.

```figure
transformer-block
```

### بلوک کدرها (که توسط کدرها BERT، T5 استفاده می شود)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

کدر دو طرفه، هیچ نقاب پوشی نیست، همه موقعیت ها همه موقعیت ها را می بینند

### بلاک دیکودر (که توسط GPT، دیکودر T5 استفاده می شود)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

یک دیکودر سه لایه زیر در هر بلوک دارد. در میان  توجه متقابل  تنها جایی است که اطلاعات از کدر به دیکودر جریان می یابد. در یک معماری خالص فقط از طریق دیکودر (GPT) ، توجه متقابل حذف می شود و شما فقط خود توجه را پوشانده + FFN دارید.

### پیش از استاندارد در مقابل پس از استاندارد

کاغذ اصلی: `x + sublayer(LN(x))`vs `LN(x + sublayer(x))`پس از نورم در حدود سال 2019 محبوبیت را از دست داد. بدون گرمایش دقیق سخت تر است.`LN`*قبل* ذیلی لایه) پیش فرض 2026 است: Llama، Qwen، GPT-3+، Mistral همه از آن استفاده می کنند.

### بلوک مدرن شده 2026

واسوانی 2017 LayerNorm + ReLU را عرضه کرد. استیک های مدرن هر دو را جایگزین کردند.

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm متوسط مرکز LayerNorm را کاهش می دهد (یک تخفیف کمتر) که محاسبه را ذخیره می کند و حداقل از نظر تجربی ثابت است.`Swish(W1 x) ⊙ W3 x`) به طور مداوم نسبت به ReLU/GELU FFN با ~0.5 امتیاز در مقالات Llama، PaLM و Qwen بالاتر است.

### تعداد پارامتر ها

براي يک بلوک با`d_model = d`و گسترش FFN`r`:

- مسا:`4 · d²`(Q، K، V، O پیش بینی)
- FFN (SwiGLU): `3 · d · (r · d)`≈ ≈`3rd²`
- استاندارد: قابل توجهی

در`d = 4096, r = 2.6, layers = 32`(تقریباً Llama 3 8B) ، کل: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`(به علاوه ورق و سر) تعداد مطابقت منتشر شده

## آن را بسازید

### مرحله ی اول: بلوک های ساختمانی

با استفاده از کوچک`Matrix`کلاس از درس 03 (برای استقلال به این پرونده کپی شده است):

- `layer_norm(x, eps=1e-5)` از میانگین تخفیف، تقسیم با std
- `rms_norm(x, eps=1e-6)` تقسیم با RMS. هیچ معادل تخفیف.
- `gelu(x)`و`silu(x) * W3 x`(سویگل)
- `ffn_swiglu(x, W1, W2, W3)`. .
- `encoder_block(x, params)`و`decoder_block(x, enc_out, params)`. .

ببین`code/main.py`براي تمام سيم هاي برق

### مرحله دوم: یک کدگر دو لایه و یک کدگر دو لایه را به سیم متصل کنید

آنها را جمع کنید. خروجی کدرها را به هر خروجی کراس توجه منتقل کنید. قبل از نمایش خروجی یک LN نهایی اضافه کنید.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### مرحله سوم: روی یک مثال اسباب بازی حرکت کنید

منبع 6 توکن و هدف 5 توکن را از طریق آن ارسال کنید. شکل خروجی را بررسی کنید.`(5, vocab)`. هيچ آموزشي نيست . اين درس درباره معماريه نه از دست دادن

### مرحله 4: تبادل در RMSNorm + SwiGLU

جایگزین LayerNorm و ReLU-FFN با RMSNorm و SwiGLU کنید. شکل های تایید هنوز هم مطابقت دارند. این مدرن سازی 2026 با یک جایگزین عملکرد است.

## ازش استفاده کن

پیاده سازی های مرجع PyTorch/TF: `nn.TransformerEncoderLayer`،`nn.TransformerDecoderLayer`اما بیشتر کد تولید 2026 بلوک خودش رو می ریزد چون:

- توجه فلاش در داخل توجه، نه از طریق `nn.MultiheadAttention`. .
- GQA / MLA در مرجع stdlib نیست.
- RoPE، RMSNorm، SwiGLU، پیش فرض PyTorch نیستند.

HF `transformers`دارای بلوک های مرجع تمیز است که باید بخوانید: `modeling_llama.py`این بلاک فقط برای کد کد 2026 است. حدود 500 خط است و ارزش یک بار از طریق رفتن دارد.

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

زبان تنها توسط دیکودر به دست آمده است زیرا تمیز ترین مقیاس و مدیریت درک و تولید را دارد. کدگر-دیکودر هنوز هم بهترین زمانی است که ورودی دارای هویت "سلسل منبع" واضح (ترجمات، تشخیص گفتار، وظایف ساختاری) است.

## -باده

ببین`outputs/skill-transformer-block-reviewer.md`مهارت بررسی یک پیاده سازی جدید بلوک ترانسفورماتور در برابر معیارهای 2026 و نشان دادن قطعات گمشده (پیش از استاندارد، RoPE، RMSNorm، GQA، FFN تناسب گسترش).

## تمرینات

1. **Easy.**پارامترهای این کدگر_ بلاک را در  شمارش کنید`d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. با استفاده از بلوک و استفاده از`sum(p.numel() for p in block.parameters())`. .
2. **Medium.**از پس نورم به قبل نورم تغییر دهید. هر دو را شروع کنید و بعد از 12 لایه ی پراکنده شده با ورودی تصادفی، نورم فعال سازی را اندازه گیری کنید. فعال سازی پس نورم باید منفجر شود؛ نورم های قبل از نورم باید محدود بمانند.
3. **Hard.**پیاده سازی یک کدگر- دیکوتر چهار لایه در یک کار کپی بازی (کپی `x`. 100 قدم راه اندازی . گزارش خسارت . تبادل در RMSNorm + SwiGLU + RoPE  خسارت کاهش می یابد؟

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks. |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same empirical stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model, usually 4 (LayerNorm) or 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers; slight ppl improvement, smaller model. |

## خواندن بیشتر

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) مشخصات اصلی بلوک
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)چرا قبل از نورم بعد از نورم خیلی بهتره
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) RMSNorm
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) کاغذ SwiGLU
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) بلاک کاینونیک 2026 فقط برای کلاهبردار
