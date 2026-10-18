# کد بندی موقعیت  سینوساید، RoPE، ALiBi

> توجه به تغییر تغییر است. "خیر گربه روی فرش نشسته" و "خیر گربه روی فرش نشسته" تولید output یکسان بدون سیگنال موقعیت. سه الگوریتم آن را حل می کند هر یک با شرط متفاوت در مورد "موقع" معنی است.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## مشکل

توجه نقطه محصول در مقیاس، از نظر سفارش نابیناست.`softmax(Q K^T / √d) V`از شباهت های جفتی محاسبه می شود.`X`هيچ چيز در داخل توجه به جايگاه اهميتي نميده

این یک اشکال در یک مدل کیسه کلمات نیست. برای زبان، کد، صوتی، ویدیو  هر چیزی که نظم معنی دارد  آن را کشنده است.

راه حل اينه که به طوري موقعيت رو به داخل داخل هاي گنجينه ها تزريق کنيم

1. **Absolute sinusoidal**(واسواني 2017) اضافه کنید `sin/cos`ساده، بدون یادگیری، به شدت خارج از طول آموزش دیده است.
2. **RoPE — Rotary Position Embeddings**(Su 2021). ویکتورهای Q و K را با زاویه متناسب با موقعیت چرخش کنید. موقعیت * رشتہ ای * را مستقیماً در محصول نقطه ای رمزگذاری می کند. در سال 2026 غالب است.
3. **ALiBi — Attention with Linear Biases**(Press 2022). کاملاً از گنجانده شدن ها اجتناب کنید؛ به امتیاز توجه بر اساس فاصله یک مجازات خطی در هر سر اضافه کنید. استخراج طول عالی.

از سال 2026 به طور اساسی هر مدل باز مرزی از RoPE استفاده می کند: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. چند مدل با زمینه طولانی از ALiBi یا انواع مدرن آن استفاده می کنند.

## مفهوم

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### بدون شک

پیش از حساب کردن ماتریکس ثابت`PE`شکل`(max_len, d_model)`:

```math
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(pos / 10000^{2i / d_{\text{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(pos / 10000^{2i / d_{\text{model}}}\right)
\end{aligned}
```

پس`X' = X + PE[:N]`هر ابعاد یک سینوساید در فرکانس متفاوت است. مدل یاد می گیرد موقعیت را از الگوی فاز بخواند.`max_len`: هیچ چیزی به مدل نگفت که در موقعیت 2048 چه اتفاقی می افتد وقتی فقط موقعیت 02047 را می بیند.

### رپی

قطعات Q و K را (نه گنجانده ها) برای یک جفت ابعاد چرخید $(2i, 2i+1)$:

```math
\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\ \sin(pos \cdot \theta_i) & \cos(pos \cdot \theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
\qquad \theta_i = \text{base}^{-2i / d_{\text{head}}}, \quad \text{base} = 10000 \text{ by default}
```

به کلید های با موقعیت مشابهی روی آن ها روی کنید `pos_k`. محصول نقطه$q'_m \cdot k'_n$تبدیل به یک تابع از$(m - n)$تنها، یعنی:**the attention score depends only on the relative distance**، با وجود اينکه چرخش رو از موقعيت هاي مطلق باز کرد

گسترش RoPE: `base`Llama 3 از 8K به 128K زمینه گسترش یافته است.

### علیبی

. از ترفند "بند" رد شو

```math
\text{attn\_score}[i, j] = \frac{q_i \cdot k_j}{\sqrt{d}} - m_h \cdot |i - j|
```

کجا$m_h$یک منحنی خاص سر است (به عنوان مثال $1 / 2^{8h/H}$) توکن های نزدیک تر افزایش می یابند؛ توکن های دور مجازات می شوند. هیچ هزینه زمانی آموزش نیست. مقاله نشان می دهد که استخراج طول از سینوسوائید و مطابقت با RoPE در طول آموزش اولیه اش است.

### چه چیزی را در سال 2026 انتخاب کنیم

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

RoPE برنده شد چون بدون تغییر معماری به توجه می رسد، موقعیت نسبی را رمزگذاری می کند و موقعیت آن را تغییر می دهد.`base`هائپر پارامتر یک دکمه پاک برای تنظیم دقیق در زمینه طولانی می دهد.

```figure
rope-explorer
```

## آن را بسازید

### مرحله اول: کدگذاری سینوسایدی

ببین`code/main.py`. 4 خط محاسبه:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

این را قبل از اولین لایه توجه به ماتریس گنجانده اضافه کنید.

### مرحله دوم: RoPE برای Q، K اعمال می شود

RoPE در محل کار در Q و K. برای هر جفت کم:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

مهم: همان تابع را در موقعیت Q اعمال کنید $m$و K در موقعیت$n$. محصولشون يه نقطه رو ميگيره$\cos((m-n) \cdot \theta_i)$توجه به موقعیت نسبي به صورت رایگان یاد می گیرد.

### مرحله سوم: منحنیات و تعصب ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

اضافه کردن`bias[h]`به`(seq_len, seq_len)`نمره توجه ماتریکس سر`h`، بعدش نرم ماکس

### مرحله 4: بررسی ویژگی فاصله نسبی RoPE

دو متری تصادفی را انتخاب کنید`a, b`. به طرف چرخید`(pos_a, pos_b)`پس از اون`(pos_a + k, pos_b + k)`. هر دو محصول نقطه باید در خطا نقطه شناور مطابقت داشته باشد. این ویژگی کل نقطه RoPE  است. این غیر متغیر به تعویض مطلق است، تنها شکاف نسبی مهم است.

## ازش استفاده کن

PyTorch 2.5+ کشتی های RoPE را در `torch.nn.functional`. بیشتر کد تولید استفاده می کنه`flash_attn`یا`xformers`در آن جا RoPE در داخل هسته توجه اعمال می شود.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.**دوباره اندازه گیری`base`به`base * (scale_factor)^(d/(d-2))`وقتی از 4K تا 16K+ گسترش می یابد.
- **YaRN.**اينترپلاسيون باهوش تر که اينترپيا توجه رو در مواقيع بلند حفظ ميکنه.
- **LongRoPE.**روش مایکروسافت در سال 2024 که با استفاده از جستجوی تکاملی برای انتخاب عوامل در مقیاس هر ابعاد استفاده می کند.
- **Position interpolation + fine-tuning.**فقط موقعیت ها رو با فاکتور تمدید کوچک کن و برای توکن های 15B خوب تنظیم کن

## -باده

ببین`outputs/skill-positional-encoding-picker.md`مهارت استراتژی کدگذاری برای یک مدل جدید را با توجه به طول زمینه هدف، نیازهای استخراج و بودجه آموزش انتخاب می کند.

## تمرینات

1. **Easy.**نقشه ي سينوسويدا رو بزن`PE`ماتریکس به عنوان نقشه گرما برای `max_len=512, d=128`.تصديق به الگوي "با رشد شاخص ابعاد، شريط ها بزرگتر مي شوند"
2. **Medium.**استفاده از مقیاس RoPE آگاه از NTK. یک LM کوچک را روی دنباله های طول 256 تمرین کنید، سپس در طول 1024 با و بدون مقیاس آزمایش کنید. پیچیدگی را اندازه گیری کنید.
3. **Hard.**ALiBi و RoPE را در یک ماژول توجه اجرا کنید. یک ترانسفورماتور چهار لایه را در یک کار کپی با دنباله های طول 512 تمرین کنید. در زمان آزمایش به 2048 اضافه کنید. تخریب را مقایسه کنید.

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position. |
| Sinusoidal | "The original one" | `sin/cos` at geometric frequencies added to embeddings; doesn't extrapolate. |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance. |
| ALiBi | "Linear bias trick" | Add $-m \cdot \lvert i-j \rvert$ to attention scores; no embedding needed, great extrapolation. |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context at inference. |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed when context expands. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation that preserves attention entropy. |
| Extrapolation | "Works beyond trained length" | Can the position scheme serve correct output past `max_len` seen in training? |

## خواندن بیشتر

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) اصل سینوسوائید
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) کاغذ RoPE
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) علیبی
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) پیشرفته ترین مقیاس بندی RoPE
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) مقاله طولانی زمینه ای Llama 2 Meta
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) روش مایکروسافت که توسط Phi-3-Long استفاده می شود و در بخش Use It ذکر شده است.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) پیاده سازی در سطح تولید هر طرح مقیاس RoPE (پیش فرض، خطی، پویا، YaRN، LongRoPE، Llama-3).
