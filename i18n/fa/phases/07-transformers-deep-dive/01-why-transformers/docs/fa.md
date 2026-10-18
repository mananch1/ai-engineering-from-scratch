# چرا ترانسفورماتورها  مشکلات RNN

> RNN ها یک به یک توکن ها را پردازش می کنند. ترانسفورمرها تمام توکن ها را به یکباره پردازش می کنند. این شرط معماری واحد هر منحنی مقیاس در یادگیری عمیق را پس از سال 2017 تغییر داد.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## مشکل

قبل از سال 2017، هر مدل جدیدتر در این سیاره  زبان، ترجمه، گفتار  یک شبکه عصبی تکراری بود. LSTMs و GRUs برای نیم دهه معیارهای ترجمه معادل ImageNet را کسب کردند. آنها تنها ابزار کسی بودند.

سه ضعف مرگبار داشتند. حسابات دنباله دار به این معنی بود که نمی تونستی در طول محور زمان موازی کنی:`t+1`به حالت پنهان از توکن نیاز داره`t`یک سری از 1,024 توکن به معنای 1,024 مرحله سریال در یک GPU است که می تواند 1,000,000 نقطه شناور در هر چرخه انجام دهد. زمان آموزش دیوار ساعت به صورت خطی با طول سری در سخت افزار طراحی شده برای موازی است.

گرادینت های ناپدید شده به این معنی است که اطلاعات 50 توکن به عقب در حال حاضر از طریق 50 غیر خطی فشرده شده است. واحد های بازپسین (LSTM، GRU) شکنجه را نرم کرده اند اما هرگز آن را از بین نمی برند. وابستگی های طولانی مدت  "کتابی که تابستان گذشته در یک هواپیما به کیوتو خوانده بودم..."  به طور معمول شکست خورده است.

حالت پنهان عرض ثابت به این معنی است که کدگر کل دنباله منبع را در یک ویکتور واحد فشرده کرد قبل از اینکه کدگر چیزی را ببیند. مهم نیست که منبع 5 توکن یا 500 است؛ گلو بطری شکل یکسان است.

مقاله 2017 "اهتمام تنها چیزی است که شما نیاز دارید" چیزی رادیکال را پیشنهاد کرد: تکرار را به طور کامل رها کنید. اجازه دهید هر موقعیت به هر موقعیت دیگر به طور موازی توجه کند. تمرین در یک ضرب ماتریک بزرگ به جای 1,024 ضرب متسلسل.

نتیجه تا سال 2026 بر هر نوع زبان (GPT-5، کلاود 4، لاما 4) ، بینایی (ViT، DINOv2، SAM 3) ، صوتی (سسس) ، زیست شناسی (AlphaFold 3) ، رباتیک (RT-2) تسلط دارد.

## مفهوم

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck.**یک RNN حساب می کنه$h_t = f(h_{t-1}, x_t)$هر مرحله به مرحله ي قبل بستگی داره$h_5$قبل از$h_4$در GPU های مدرن با 10000+ هسته موازی، این 99 درصد از سیلیکون را در یک سری طولانی از بین می برد.

**Attention as a broadcast.**حسابات توجه به خود`output_i = sum_j(a_ij * v_j)`برای هر جفت$(i, j)$تمام ماتریکس توجه N×N در یک دسته می پر شود هیچ مرحله ای به دیگری بستگی ندارد GPU ها عاشقش هستند

**The speedup is not a constant.**فرق بين$O(N)$عمق سریال و$O(1)$در عمل، ترانسفورماتورها در هر دوره با سخت افزار مشابه در N=512 510x سریع تر تمرین می کنند و شکاف با طول ردیف تا زمانی که شما به $O(N^2)$دیوار حافظه توجه (که Flash Attention بعداً اصلاح کرد)

**What transformers cost.**توجه حافظه به شکل$O(N^2)$برای زمینه 2K، خوب. برای زمینه 128K، شما نیاز به پنجره های شیفت، استثمار RoPE، فلش توجه تایلینگ، یا متغیر توجه خطی دارید. تکرار بود.$O(N)$در زمان و حافظه هم، ترانسفورماتورها زمان را با حافظه عوض می کنند و سپس زمان را از طریق موازی باز می کنند.

**The inductive bias shift.**RNN ها محل و تازه بودن را فرض می کنند. ترانسفورماتورها هیچ چیز را فرض نمی کنند. هر جفت برای توجه است. به همین دلیل ترانسفورماتورها برای آموزش خوب نیاز به داده های بیشتری دارند اما پس از آن مقیاس بیشتری دارند. چینچیلا (2022) این را رسمی کرد: با توجه به توکن های کافی، یک ترانسفورماتور همیشه RNN با تعداد پارامتر برابر را شکست می دهد.

```figure
rnn-vs-parallel
```

## آن را بسازید

هیچ شبکه عصبی اینجا نیست ما شکاف هسته ای را به صورت عددی شبیه سازی می کنیم تا شکاف را در لپ تاپ خود احساس کنید.

### مرحله ی اول: اندازه گیری عمق سریال

ببین`code/main.py`. ما دو تابع را ایجاد می کنیم. یکی یک دنباله را به عنوان یک زنجیره اضافه (سلسل، مانند RNN) کدگذاری می کند. یکی آن را به عنوان یک کاهش موازی (بخش، مانند توجه) کدگذاری می کند. همان ریاضی، نمودار وابستگی های مختلف.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

ما هر دو رو در دنباله ها تا 100 هزار عنصر زمان می گیریم$O(N)$حتی در پایتون خالص، کاهش توجه به سبک آن در طول ≥ 1000 به دلیل پایتون`sum()`در C اجرا می شود و بدون هزینه مترجم در هر مرحله تکرار می شود.

### مرحله دوم: حساب عملیات نظری

هر دو الگوریتم N را اضافه می کنند. تفاوت در * عمق وابستگی * است: چند عملیات باید به ترتیب قبل از شروع بعدی انجام شوند. عمق RNN = N. عمق توجه = log(N) با کاهش درخت، یا 1 با اسکن موازی. عمق، نه شمارش آپ، زمان GPU را تعیین می کند.

### مرحله 3: مقیاس بندی تجربی در دنباله های طولانی

ما يه جدول زماني چاپ مي کنيم که باعث ميشه$O(N)$در یک لپ تاپ مک ۲۰۲۶، دنباله های زیر ۱۰۰۰ عنصر برای اندازه گیری خیلی سریع است. دنباله های ۱۰۰۰۰۰ یک اسکن خطی تمیز را نشان می دهد. این را به یک تراکنشگر ۱۶۳۸۴ توکن با معادل ۱۲ لایه LSTM مقیاس کنید و می بینید که چرا ساعت دیواری آموزش در سال ۲۰۱۶ یک مانع بود.

## ازش استفاده کن

چه وقت بايد هنوز در سال 2026 RNN رو انتخاب کني:

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

مدل های فضایی دولتی (SSM) مانند Mamba اساسا RNN با پارامترسازی ساختاری هستند که به آنها بهترین از هر دو را می دهد: $O(N)$حافظه اسکن، آموزش موازی از طریق اسکن انتخابی. آنها 90% از کیفیت ترانسفورم را با مقیاس بندی طولانی بهتر بهبود می بخشند. در سال 2026 بیشتر آزمایشگاه های مرزی مدل های ترانسفورم های هیبریدی SSM + را آموزش می دهند (به عنوان مثال Jamba، Samba)

## -باده

ببین`outputs/skill-architecture-picker.md`مهارت برای یک مشکل جدید در ترتیب با توجه به طول، تولید و محدودیت های بودجه آموزش، معماری را انتخاب می کند. باید همیشه بدون ذکر معامله، توصیه یک RNN خالص برای تمرینات بالای 1B را رد کند.

## تمرینات

1. **Easy.**اینو بگیر`rnn_style`از`code/main.py`و حالت پنهان اسکالر را با ویکتور طول-64 از حالت پنهان جایگزین کنیم. دوباره اندازه گیری کنید.
2. **Medium.**یک جمع پیشگویی موازی (سکن هیلیس-ستیل) را در پایتون خالص اجرا کنید. بررسی کنید که همان محصول عددی را با یک اسکن سریال در طول 1024 تولید می کند. عمق را محاسبه کنید.
3. **Hard.**به طور توجه به PyTorch روی GPU انتقال دهید. زمان هر دو را در حالی که طول دنباله را از 64 تا 65،536 می پوشانید. نقشه و شکل منحنی را توضیح دهید.

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
| Attention | "Let tokens look at each other" | Weighted sum $\sum_j a_{ij} v_j$ where $a_{ij}$ comes from a similarity score between positions $i$ and $j$. |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = $O(N^2)$ in sequence length; Flash Attention hides the constants, not the scaling. |

## خواندن بیشتر

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) مقاله ای که تکرار در NLP اصلی را کشت.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) جایی که توجه متولد شد، به یک RNN متصل شد.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) کاغذ LSTM اصلی، برای ثبت.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) پاسخ متكرر مدرن به ترانسفورماتورها
