# از ابتدا ترانسفورمر بسازید

> 13 تا درس، يک مدل، هيچ راه کوتاهي نيست

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## مشکل

شما هر مقاله ای را خوانده اید. شما توجه، تقسیم های چند سر، کدگذاری موقعیت، بلاک های کدگذاری و دیکودر، BERT و GPT از دست دادن، MoE، KV کش را اجرا کرده اید. حالا آنها را در یک کار واقعی با هم کار کنید.

سنگ اصلی: یک ترانسفورماتور کوچک فقط برای کپیکتر را در یک کار مدل سازی زبان سطح شخصیت آموزش دهید. این کار شکسپیر را می خواند. این شکسپیر جدید را تولید می کند. این کار به اندازه کافی کوچک است تا در کمتر از 10 دقیقه در یک لپ تاپ آموزش داده شود. این به اندازه کافی درست است که تغییر در مجموعه داده های بزرگتر و آموزش طولانی تر به شما یک LM واقعی می دهد.

این "نانو جی پی تی" دوره است. این اصلی نیست. آموزش 2023 نانو جی پی تی کارپتی است که هر دانش آموز حداقل یک بار می نویسد. ما شکل را بالا می بریم و آن را در اطراف آنچه که پوشش داده ایم تغییر می دهیم.

## مفهوم

![Transformer-from-scratch block diagram](../assets/capstone.svg)

معماری، به شرح زیر:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### آنچه که می فرستیم

- `GPTConfig` یک مکان برای تنظیم تمام پارامترهای هیپر
- `MultiHeadAttention` علت، دسته بندی شده، با راه فلاش اختیاری (PyTorch's `scaled_dot_product_attention`)
- `SwiGLUFFN` FFN مدرن
- `Block` توجه پیش از استاندارد، بسته بندی باقیمانده + FFN.
- `GPT` گنجانده شدن، بلوک های جمع شده، سر LM، تولید (().
- حلقه آموزش با AdamW، cosine LR، تراز gradient.
- توکنيزر سطح چار در متن شکسپير

### چيزي که ما نمي فرستيم

- RoPE  در درس 04 به طور مفهومی اجرا شده است. در اینجا ما از گنجانده های موضعی آموخته شده برای سادگی استفاده می کنیم. تمرین ها از شما می خواهند در RoPE تبادل کنید.
- KV cache در طول نسل  هر مرحله نسل توجه را بر روی پیشگویی کامل محاسبه می کند. آهسته اما ساده تر. تمرین ها از شما می خواهند یک KV cache اضافه کنید.
- توجه فلاش  PyTorch 2.0+ ارسال خودکار اگر ورودی ها مطابقت داشته باشد؛ ما استفاده می کنیم `F.scaled_dot_product_attention`. .
- MoE  یک FFN در هر بلوک. شما MoE را در درس 11 دیدید.

### متریک هدف

در لپ تاپ مک م 2، یک لپ تاپ 4 لایه، 4 سر، d_model=128 GPT آموزش داده شده برای 2000 قدم در`tinyshakespeare.txt`:

- خسارت تمرین از ~4.2 (به طور تصادفی) به ~1.5 در حدود 6 دقیقه نزدیک می شود.
- نمونه گیری محصول به شکل شکسپیر به نظر می رسد: کلمات باستانی، شکاف خط، نام های خاص مانند "ROMEO:" ظاهر می شوند.
- از دست دادن Val (۱۰ درصد نهایی متن) از دست دادن آموزش به طور دقیق پیگیری می شود؛ هیچ اضافه ای در این اندازه / بودجه وجود ندارد.

```figure
n5-block-stack
```

## آن را بسازید

اين درس از PyTorch استفاده ميکنه`torch`(سازش پردازنده خوبه)`code/main.py`. اسکریپت کار می کنه:

- دانلود`tinyshakespeare.txt`اگر گم شده باشد (یا در حال خواندن یک نسخه محلی).
- توکنيزر کار بايت سطح
- قطب/مربع در 90/10
- حلقه آموزش با bf16 اتوماتیک پخش در سخت افزار پشتیبانی شده
- نمونه برداری بعد از آموزش تمام شده

### مرحله اول: داده ها

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 تا حرف منحصر به فرد، ذخيره کوچکي لغت، اندازه 4 بايت لغت، بدون BPE، بدون تکه سازي

### مرحله دوم: مدل

ببین`code/main.py`. بلوک کتاب درسی از درس 05  پیش از استاندارد، RMSNorm، SwiGLU، MHA علت. شمارش پارامتر برای 4/4/128: ~800K.

### مرحله سوم: حلقه آموزش

يه دسته تصادفي از پنجره هاي رمزي طول-256 رو پيدا کنيد جلو، تعديل به طرف، تعديل به عقب، قدم آدم وو، ثبت، تکرار

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### مرحله 4: نمونه

به صورت پرسپورت، بارها و بارها به جلو، نمونه از logits top-p، اضافه کنید و ادامه دهید. پس از 500 توکن متوقف کنید.

### مرحله 5: خروجی را بخونید

بعد از دو هزار قدم:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

شکسپير نيست ولي شکسپير شکل داره و براي 800 هزار پارامتر و 6 دقيقه روي لپ تاپ برنده ميشه

## ازش استفاده کن

اين سنگ پايين يک معمار مرجعيه سه تمديد براي ارسالش به چيزي واقعي:

1. **Swap the tokenizer.**استفاده از BPE (به عنوان مثال `tiktoken.get_encoding("cl100k_base")`اندازه تلفظ از 65 تا 50 هزار افزایش می یابد. ظرفیت مدل باید برای تعویض افزایش یابد.
2. **Train on a bigger corpus.**استفاده کنید`OpenWebText`یا`fineweb-edu`توکن 10B روی یک A100 فقط 24 ساعت طول می کشد تا یک GPT 125M-param داشته باشد.
3. **Add RoPE + KV cache + Flash Attention.**تمرینات زیر شما را در هر یک از آنها راهنمایی می کند.

این به عنوان یک GPT 125M-پارامتر که تولید انگلیسی روان است. نه یک مدل مرزی. اما همان مسیر کد  فقط بزرگتر  است که کارپاتی، EleutherAI و موسسه آلن برای آموزش پوائنت های بازرسی تحقیقاتی در سال 2026 استفاده می کنند.

## -باده

ببین`outputs/skill-transformer-review.md`مهارت بررسی یک پیاده سازی ترانسفارمر از نو برای دقت در تمام ۱۳ درس قبلی.

## تمرینات

1. **Easy.**فرار کن`code/main.py`.آزمایش کنید که از دست دادن اعتبار در مرحله آخر مدل آموزش دیده شما کمتر از 2.0 باشد.`max_steps`از 2000 تا 5000  آیا از دست دادن ویال بهبود می یابد؟
2. **Medium.**جای جای جای گیری های موضعی آموخته شده را با RoPE جایگزین کنید. گردش را به Q و K در داخل اعمال کنید `MultiHeadAttention`.در حال حاضر، از دست دادن وول کم تر از اونه
3. **Medium.**یک حافظه کش KV را در حلقه نمونه گیری پیاده سازی کنید. 500 توکن با و بدون حافظه کش تولید کنید. ساعت دیواری باید در لپ تاپ 520x بهبود یابد.
4. **Hard.**یک سر دوم را به مدل اضافه کنید که به توکن بعدی اضافه شده (MTP  Multi-Token Prediction from DeepSeek-V3) پیش بینی می کند.
5. **Hard.**جایگزین FFN واحد در هر بلوک با یک MoE 4 متخصص روتر + top-2 روتر. ببینید که چگونه از دست دادن val در پارامترهای فعال مطابقت تغییر می کند.

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; the canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial since 2015 uses it. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, keep optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; prevents training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw; overfit detector. |

## خواندن بیشتر

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) اجرای کلاسیک با اشاره
