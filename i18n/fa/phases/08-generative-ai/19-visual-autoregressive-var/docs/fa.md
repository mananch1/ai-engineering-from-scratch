# مدل سازی خودکار بازخورد بصری (VAR): پیش بینی در مقیاس بعدی

> مدل های انتشار نمونه گیری تکراری در زمان (نمونه گیری مراحل) نمونه های VAR به طور تکراری در مقیاس  آن پیش بینی یک توکن 1x1 ، سپس 2x2 ، سپس 4x4 ، تا رزولوشن نهایی ، هر مقیاس شرایط در قبیل قبلی. مقاله 2024 نشان داد که VAR مطابقت دارد قوانین مقیاس بندی سبک GPT برای تولید تصویر و شکست DiT در همان بودجه محاسبه. این درس مکانیسم اصلی را ایجاد می کند.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## مشکل

نسل خودکشی بر مدل سازی زبان تسلط داشت زیرا مقیاس قابل پیش بینی است: محاسبه بیشتر، پارامترهای بیشتر، کم پیچیدگی، خروجی بهتر. نسل تصویر قبل از سال 2024 دو تلاش اصلی AR داشت: PixelRNN/PixelCNN (پیکسل به پیکسل) و DALL-E 1 / Parti / MuseGAN (توکن به توکن در کد VQ-VAE).

هر دو از یک مشکل نظم نسل رنج می بردند. پیکسل ها و توکن ها در یک شبکه 2D مرتب شده اند، اما مدل AR باید آنها را در یک ترتیب 1D راستر بازدید کند. پیکسل گوشه اولیه هیچ ایده ای ندارد که تصویر در نهایت چه چیزی می شود. کیفیت نسل بدتر از GPT-در متن است و هرگز به کیفیت مدل انتشار در محاسبه مطابقت نرسیده است.

VAR مشکل نظم تولید را با تغییر آنچه تولید می شود حل می کند. به جای پیش بینی یک به یک توکن تصویر در فضا، VAR یک تصویر را با افزایش وضوح پیش بینی می کند. مرحله 1: یک توکن 1x1 (تصاویر کلی "همگوشی") پیش بینی کنید. مرحله 2: یک شبکه 2x2 از توکن ها (تخصیصات خشن تر) پیش بینی کنید. مرحله 3: یک شبکه 4x4 پیش بینی کنید. مرحله K: آخرین شبکه (H/8) x ((W/8) را پیش بینی کنید.

هر مقیاس به تمام مقیاس های قبلی (به طور علنی در "ترتیب مقیاس") و موازی در مقیاس خود توجه می کند. مشکل ترتیب ناپدید می شود: کل تصویر در مقیاس k در یک گذرگاه ترانسفورم تولید می شود.

## مفهوم

### VQ-VAE Tokenizer چند مقیاس

VAR به یک**multi-scale discrete tokenizer**. براي يه تصويري$x$، این یک سری از شبکه های توکن با وضوح بالاتر را تولید می کند:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

هرکدوم$z_k$استفاده از همان کد بوک (حجم معمول 4096-16384) نشان دادن در هر مقیاس مستقل نیست  آن را آموزش داده شده است تا جمع باقیمانده در هر مقیاس بازسازی می شود $f$:

```math
f \approx \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_1), \text{target\_size}) + \cdots + \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_K), \text{target\_size})
```

این یه**residual VQ**متغیر.$k$چه ترازو را ضبط می کند$1 \ldots k-1$کدگر مجموع تمام گنجانده های مقیاس را می گیرد و تصویر را تولید می کند.

توکنایزر VQ چند مقیاس یک بار آموزش داده می شود (مانند VQGAN) و سپس منجمد می شود. تمام کار تولیدی توسط مدل autoregressive در بالا انجام می شود.

### پیش بینی در مقیاس بعدی

مدل تولید کننده یک ترانسفورماتور است که توکن ها را از تمام مقیاس های قبلی می بیند و توکن ها را در مقیاس بعدی پیش بینی می کند.

ساختار تسلسل ورودی:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

گنجانده شدن موقعیت هر دو شاخص مقیاس و موقعیت فضایی را در مقیاس رمزگذاری می کند. توجه به ترتیب مقیاس علت است: نماد در مقیاس $k$، موقعیت$(i, j)$ميتونم به تمام توکن ها در مقیاس ها توجه کنم$1 \ldots k$و به توکن ها در مقیاس$k$خود که در هر ترتیب درون مقیاس مورد استفاده قرار می گیرد (VAR از توجه موقعیت ثابت بدون علل در مقیاس استفاده می کند  همه موقعیت ها در یک مقیاس به طور موازی پیش بینی می شوند).

از دست دادن آموزش: در هر مقیاس$k$، توکن ها رو پیش بینی کن$z_k$با توجه به تمام توکن های مقیاس قبلی، از دست دادن انتروپی متقابل در کد های VQ متمایز. ساختار مشابه GPT به جز "سلسل" اکنون ساختار مقیاس است.

### نسل

در نتیجه:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

برای$K = 10$در مقیاس ها، تولید 10 ترانسفورمتر به جلو عبور می کند. هر گذر تمام مقیاس خود را در موازی تولید می کند  هیچ خود پاشنه هر توکن در یک مقیاس. برای یک تصویر 256x256 این تقریبا 10 گذر نسبت به 28-50 DiT است.

### چرا مقیاس بعدی بر مقیاس بعدی برنده می شود

سه پیروزی ساختاری:
1. **Coarse-to-fine aligns with natural image statistics.**درک بصری انسان و مجموعه داده های تصویر هر دو دارای منظمات وابسته به مقیاس هستند: ساختار فرکانس پایین پایدار و قابل پیش بینی است؛ جزئیات فرکانس بالا وابسته به محتوای فرکانس پایین است. پیش بینی مقیاس بعدی از این بهره می برد.
2. **Parallel generation within scale.**بر خلاف GPT، VAR تمام توکن ها را در یک مرحله در مقیاس تولید می کند. طول تولید موثر به جای خطی مقیاس ثبت است.
3. **No generation order bias.**توکن های مقیاس k تمام مقیاس k-1 را می بینند؛ هیچ تعصب "باید" یا "بالا" وجود ندارد که توکن های اولیه را مجبور به تعهد کند قبل از اینکه زمینه دیر در دسترس باشد.

### قانون مقیاس بندی

تیان و همکارانش نشان داد که VAR یک منحنی مقیاس قانون قدرت را برای FID در ImageNet  دنبال می کند درست مانند GPT برای گیج شدن. دو برابر کردن پارامترها یا محاسبه به طور قابل اعتماد خطای را نصف می کند. این اولین مدل تولید کننده تصویر بود که چنین رفتار مقیاس بندی را به همان اندازه مدل های زبان نشان می دهد. نتیجه این است که پیش بینی های مقیاس VAR از طریق محاسبه قابل پیش بینی می شوند، نه حدس های تجربی در هر معماری.

### رابطه با انتشار

VAR و انتشار داستان فشرده سازی داده های مشابه را به اشتراک می گذارند: هر دو مشکل تولید را به یک سری از زیرمشکل های آسان تر تقسیم می کنند.

- پخش: به تدریج اضافه کردن صدا، یاد بگیرید که یک قدم را رد کنید.
- VAR: تدریجی رزولوشن را اضافه کنید، یاد بگیرید که مقیاس بعدی را پیش بینی کنید.

این دو محور مختلف در طول مشکل هستند. هر دو توزیع مشروط قابل کنترل را ارائه می دهند. از نظر تجربی VAR در نتیجه گیری سریعتر است (کم تر گذر، همه موازی در یک مقیاس) و با DiT در کلاس مشروط ImageNet مطابقت دارد یا غلبه می کند. VAR مشروط متن (VARclip، HART) یک جهت تحقیق فعال است.

```figure
gx-var-next-scale
```

## آن را بسازید

در`code/main.py`شما:
1. يه کوچيک بساز**multi-scale VQ tokenizer**در داده های "تصاویر" مصنوعی (2D حلقه های گاوسی)
2. قطار یک**VAR-style transformer**تا از تکه های بعدی پیش بینی کنیم.
3. نمونه با تماس با ترانسفورمتر 4 بار (4 مقیاس) و رمزگذاری.
4. بررسی کنید که آموزش در مقیاس ترتیب شده تولید را در مقیاس موازی می کند.

این یک پیاده سازی اسباب بازی است. نکته این است که ماسک توجه ساختار مقیاس و نسل موازی در مقیاس واقعا کار کند.

## -باده

این درس به ما کمک می کند`outputs/skill-var-tokenizer-designer.md` مهارت برای طراحی یک توکنایزر چند مقیاس: تعداد مقیاس ها، نسبت مقیاس، اندازه کتاب کد، اشتراک گذاری باقیمانده، معماری کدسر.

## تمرینات

1. **Scale count ablation.**VAR را با مقیاس های 4, 6, 8, 10 تمرین کنید. کیفیت بازسازی را در مقابل تعداد گذرگاه های خودکشی اندازه گیری کنید. مقیاس های بیشتر = بقایای دقیق تر = کیفیت بهتر اما گذرگاه های بیشتر.

2. **Codebook size.**توکن هاي ترن با اندازه هاي کد بوک 512, 4096, 16384

3. **Parallel-within-scale check.**برای یک VAR آموزش دیده، الگوی توجه را به طور صریح اندازه گیری کنید. در مقیاس k، مدل به موقعیت های مقیاس مختلف اما نه در مقیاس توجه می کند؟ اجرای ماسک را بررسی کنید.

4. **VAR vs DiT scaling.**برای همان کار کلاس-شرطی ImageNet، VAR و DiT را با بودجه پارامترهای مشابه (به عنوان مثال، 33M، 130M، 458M) آموزش دهید. پلاٹ FID در مقابل محاسبه. VAR باید در هر اندازه از DiT جلوتر باشد  نتایج کاغذ را در مقیاس کوچک بازیافت کنید.

5. **Text conditioning.**گسترش VAR برای گرفتن یک ورق ورق متن (CLIP جمع آوری شده) به عنوان ورق ضمیمه اضافی از طریق adaLN. این نسخه HART است. FID چقدر در نمونه گیری متن محور بهبود می بخشد؟

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | Image generation by next-scale prediction over a pyramid of VQ token grids |
| Next-scale prediction | "Predict coarser, then finer" | The model predicts tokens at increasing resolution scales, conditioning on all previous scales |
| Multi-scale VQ tokenizer | "Residual VQ" | VQ-VAE that produces K token grids of increasing resolution, with decoder summing all scales |
| Scale k | "Pyramid level k" | One of K resolution levels, from 1x1 at k=1 up to (H/p)x(W/p) at k=K |
| Parallel-within-scale | "One forward per scale" | All tokens at scale k are predicted in one transformer pass, not autoregressively |
| Causal-across-scales | "Scale-ordered attention" | Token at scale $k$ can attend to all of scales $1 \ldots k$ but not scales $k+1 \ldots K$ |
| Residual VQ | "Additive tokenization" | Each scale's tokens encode the residual left by lower scales; decoder sums all scale embeddings |
| VAR scaling law | "Image GPT scaling" | FID follows a predictable power law in compute, like language models' perplexity |
| HART | "Hybrid VAR + text" | Text-conditional VAR variant combining MaskGIT-style iterative decoding with VAR's scale structure |
| Scale position embedding | "(scale, row, col) triple" | Positional encoding carries both the scale index and spatial coordinates within the scale |

## خواندن بیشتر

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) ورق VAR، مرجع قانونی
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) DiT، خط پایه مقایسه انتشار
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) VQGAN، توکنایزر خانواده VAR توکنایزر چند مقیاس گسترش می یابد
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) VQ-VAE، پایه ی توکن سازی تصویر متمایز
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) VAR متن شرط بندی
