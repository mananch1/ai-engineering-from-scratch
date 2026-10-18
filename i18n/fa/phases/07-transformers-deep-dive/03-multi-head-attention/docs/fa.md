# توجه چند سر

> يه سر توجه يه رابطه در يک زمان ياد مي گيرد هشت سر هشت سر ياد مي گيرد سر ها آزاد هستند چند تا از آنها رو برداريد

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## مشکل

یک سر توجه به خود یک ماتریس توجه را محاسبه می کند. این ماتریس یک نوع رابطه را ضبط می کند. معمولاً آن رابطه را که از دست دادن هر سیگنال آموزشی به حداقل می رساند. اگر داده های شما دارای توافق موضوع و فعل، مرجع، گفتار طولانی مدت و ترکیب بندی ترکیب شده است، یک سر آنها را به یک توزیع نرم حداکثر واحد می کند و نیمی از سیگنال را از دست می دهد.

اصلاح از مقاله Vaswani 2017: چندین عملکرد توجه را به طور موازی اجرا کنید، هر کدام با طرح های Q، K، V خود، و تولیدات را یک زنجیره ای کنید. هر سر در یک فرعی کوچک تر از ابعاد عمل می کند $d_{\text{model}} / n_{\text{heads}}$. پارامترهای کل یکسان باقی می مانند . قدرت بیانگر بالا می رود

توجه چند سر پیش فرض هر ترانسفورماتور در 2026 کشتی است. تنها استدلال در مورد *چقدر* سر و اینکه آیا کلید ها و ارزش ها طرح های مشترک (گرگوده شده توجه سوال، توجه چند سوال، توجه چند سر پنهان) است.

## مفهوم

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.**اینو بگیر`X`شکل`(N, d_model)`. پروژه به Q، K، V هرکدوم شکل`(N, d_model)`. دوباره به`(N, n_heads, d_head)`کجا$d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$. به`(n_heads, N, d_head)`. .

**Attend in parallel.**در هر سر توجه نقطه ای را تولید کنید.`(N, d_head)`سر ها در فرعي مکان هاي مختلف از گنجينه ها عمل ميکنن و هرگز در طول حساب توجه صحبت نميکنن

**Concatenate and project.**سر هاي دسته به سمت`(N, d_model)`و با ماتریکس خروجی آموخته ضرب کنید `W_o`شکل`(d_model, d_model)`.`W_o`اين همون جاييه که سر ها ميگيرند

**Why it works.**هر سر می تواند بدون رقابت با دیگران برای بودجه نمایندگی تخصصی داشته باشد. مطالعات بررسی از سال 20192024 نقش های سر متفاوتی را نشان می دهد: سر موقعیت، سر که به توکن قبلی توجه می کند، سر کپی، سر نهاد های نامگذاری، سر های ادغام (که در زمینه یادگیری پایه ای هستند).

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA استاندارد پیش فرض مدرن است زیرا حافظه KV-cache را به یک عامل از $N/G$در حالی که کیفیت تقریبا کامل را حفظ می کند. MLA با فشرده سازی K / V به یک فضای پنهان، سپس پیش بینی مجدد در زمان محاسبه  هزینه FLOPs، ذخیره حافظه بسیار بیشتر می کند.

```figure
multihead-split
```

## آن را بسازید

### مرحله اول: سر های جداگانه از توجه تک سر که قبلاً داریم

.`SelfAttention`از درس 02 و با یک جفت تقسیم/کونتک بسته بندی کنید.`code/main.py`برای اجرای یک برنامه ی نوکنده؛ منطق این است که:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

يه شکل باز و يه تغيير بدون حلقه اين همون کارييه که PyTorch تحتش انجام ميده`nn.MultiheadAttention`. .

### مرحله دوم: توجه به هر محصول در مقیاس نقطه ای را اجرا کنید

هر سر قطعه ی خودش از Q، K، V می گیرد. توجه به یک ممل دسته ای می شود:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

در سخت افزار واقعي`Qh @ Kh.transpose(...)`یک است`bmm`گپيو يک دسته ي شکل رو مي بينه`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`اضافه کردن سر ها آزاده

### مرحله 3: گروه بندی- سوال توجه

فقط طرح های کلیدی و ارزش تغییر می کنند.`n_heads`گروه های K و V`n_kv_heads < n_heads`گروه ها و برای مطابقت با آنها تکرار می شود:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

در نتیجه این حافظه را ذخیره می کند چون فقط`n_kv_heads`کپي ها در حافظه KV زنده هستند، نه `n_heads`.لاما 3 70B 64 سر سوال با 8 سر KV استفاده می کند

### مرحله چهارم: بررسی آنچه هر سر آموخته است

با 4 سر به یک جمله کوتاه از MHA اجرا کنید.`(N, N)`شما می بینید که سر های مختلف ساختار متفاوتی را انتخاب می کنند حتی با ابتدایی تصادفی که بخشی از اشعار و بخشی از تعادل چرخش در فرعی است.

## ازش استفاده کن

در PyTorch، نسخه ی یک خط:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA از PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?**قوانین انگشت از مدل های تولید در سال 2026:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`تقریبا همیشه در 64 یا 128 فرود می آید. این واحد از مقدار یک سر می تواند "بینند". زیر 32 سقوط و سر شروع به مبارزه با عامل مقیاس بندی.$\sqrt{d_{\text{head}}}$؛ بالاتر از 256 می رسی و مزایای "بسیاری از متخصصین کوچک" را از دست می دهی.

## -باده

ببین`outputs/skill-mha-configurator.md`مهارت توصیه می کند تعداد سر، تعداد kv-head و استراتژی پروژکتور برای یک ترانسفورماتور جدید با توجه به بودجه پارامتر، طول دنباله و هدف انتشار.

## تمرینات

1. **Easy.**از " MHA " بگير`code/main.py`و تغییر`n_heads`از 1 تا 16 با `d_model=64`نقشه از دست دادن یک مدل کوچک یک لایه در یک کار کپی مصنوعی. آیا سر های بیشتر کمک می کند، سطح بالا یا آسیب می رساند؟
2. **Medium.**پیاده سازی MQA (یک سر KV به اشتراک گذاشته شده در تمام سر های جستجو). اندازه گیری اینکه چقدر تعداد پارامتر ها به برابر MHA کامل کاهش می یابد. محاسبه کنید که اندازه KV-کاش در نتیجه برای N = 2048 چقدر کاهش می یابد.
3. **Hard.**یک نسخه کوچک از توجه چند سر را اجرا کنید: K،V را به یک درجه فشرده کنید`r`.خاموش شده، خفه شده رو در حافظه KV نگه داريد، در زمان توجه کم کم کنید`r`آیا حافظه کش زیر 1/8 MHA کامل عبور می کند در حالی که کیفیت در عرض 1 بیت اعتبارسنجی باقی می ماند؟

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$ with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## خواندن بیشتر

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) مشخصات اصلی چند سر
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) مقاله MQA
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) چگونه MHA را پس از آموزش به GQA تبدیل کنیم.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) MLA و چرا از MHA/GQA در حافظه کش میترسد.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) نگاه مکانیستی به آنچه که سر ها واقعا انجام می دهند.
