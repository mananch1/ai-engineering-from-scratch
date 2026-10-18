# دروازه های AI  LiteLLM, Portkey, Kong AI Gateway, Bifrost

> دروازه ای بین برنامه های شما و ارائه دهندگان مدل قرار دارد. ویژگی های اصلی روتینگ ارائه دهنده، بازگشت، بازجویی، محدودیت نرخ، مرجع مخفی، مشاهده، محافظ هستند. تقسیم بازار در 2026: **LiteLLM**MIT OSS با 100+ ارائه دهنده، سازگار با OpenAI است، اما در حدود 2000 RPS (8 GB حافظه، شکست های کاسکادی در معیار های منتشر شده) تجزیه می شود؛ بهترین برای پایتون، <500 RPS، توسعه / نمونه سازی. **Portkey**در سطح کنترل قرار گرفته است (گاردریل، ویرایش PII، تشخیص jailbreak، مسیرهای حسابرسی) ، Apache 2.0 در ماه مارس 2026، 20-40 ms تاخیر بیش از حد،$49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $قیمت 100/مودل/ماه (ماکس 5 در سطح Plus) ؛ مناسب برای کسب و کار اگر شما قبلا در Kong هستید. **Bifrost**(ماکسيم AI)  بازتجربه اتوماتيك با بازپرداخت قابل تنظیم، بازپرداخت به آنترپيك در OpenAI 429 **Cloudflare / Vercel AI Gateways** مدیریت شده، صفر عملیات، بازتجربه اولیه. اقامت داده تصمیم خود میزبان را هدایت می کند؛ پورتکی و کانگ در وسط با مدیریت OSS + اختیاری قرار دارند.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## اهداف یادگیری

- شش ویژگی اصلی دروازه را ذکر کنید (روتینگ، عقب نشینی، تلاش مجدد، محدودیت سرعت، راز، قابل مشاهده، محافظ).
- نقشه چهار دروازه 2026 (LiteLLM، Portkey، Kong AI، Bifrost) برای مقیاس سقف ها و موارد استفاده.
- شاخص کنگ را ذکر کنید (228٪ در مقابل Portkey، 859٪ در مقابل LiteLLM) و توضیح دهید که چرا برای > 500 RPS مهم است.
- انتخاب خود میزبان در مقابل مدیریت داده ها اقامت و بودجه عملیات.

## مشکل

محصول شما OpenAI، Anthropic و Llama خود میزبان است. هر ارائه دهنده دارای SDK، مدل خطا، محدودیت نرخ و طرح auth متفاوت است. شما می خواهید شکست (اگر OpenAI 429s، سعی کنید Anthropic) ، یک فروشگاه اعتبار واحد، مشاهده پذیری یکپارچه و محدودیت نرخ هر مستاجر.

ایجاد مجدد این در لایه اپلیکیشن هر سرویس را به هر ارائه دهنده متصل می کند. یک لایه دروازه آن را به یک فرآیند با یک API (معمولا سازگار با OpenAI) که به ارائه دهندگان پخش می شود، تحکیم می کند.

## مفهوم

### شش ویژگی اصلی

1. **Provider routing** OpenAI، Anthropic، Gemini، خود میزبان، و غیره پشت یک API.
2. **Fallback** در 429، 5xx، یا شکست کیفیت، دوباره در جای دیگه ای امتحان کنید.
3. **Retries** بازپرداخت نمایی، تلاش های محدود
4. **Rate limits** هر مستاجر، هر کليد، هر مدل
5. **Secret references** در زمان اجرا (هیچ وقت در اپلیکیشن) اعتبارات را از خزانه خارج کنید.
6. **Observability** ویژگی های OTel + GenAI (فاز 17 · 13) + نسبت هزینه.
7. **Guardrails** حذف اطلاعات اطلاعات، شناسایی jailbreak، فیلترهای موضوع مجاز

### LiteLLM  MIT OSS، پایتون

- 100+ ارائه دهنده، سازگار با OpenAI، تنظیم روتر، عقب نشینی، قابل مشاهده بودن اساسی.
- در حدود 2000 RPS در مقياس کنگ خراب می شود. 8 گيبايت حافظه، شکست های کاسک در تحت بار مداوم.
- بهترین تناسب: برنامه پایتون، <500 RPS، دروازه های توسعه/ مرحله بندی، مسیر آزمایشی.
- هزینه: 0 دلار برای OSS؛ سطح رایگان ابر وجود دارد.

### موقعیت خط کنترل Portkey 

- اپاچی 2.0 از مارس 2026، رایل های نگهبان، ویرایش اطلاعات شخصی، شناسایی jailbreak، ردیابی های حسابرسی.
- 20-40 ms در هر درخواست تاخیر هزینه های بالا
- 49 دلار در ماه براي سطح توليد با نگهداري + SLA
- بهترین مناسب: صنایع تنظیم شده که نیاز به محافظ + مشاهده پذیری بسته شده دارند.

### دروازه هوش مصنوعی کنگ  بازی مقیاس

- ساخته شده در Kong Gateway (نتایج گیتوی API بالغ، lua+OpenResty).
- شاخص کنگ در برابر 12 CPU: 228% سریعتر از Portkey، 859% سریعتر از LiteLLM.
- قیمت: 100 دلار/ماه مدل، حداکثر 5 دلار در سطح پلاس
- بهترین تناسب: قبلاً روی Kong؛ > 1000 RPS؛ آماده مجوز.

### بیفروز (ماکسیم AI)

- دوباره امتحانات خودکار با تنظیم کننده backkoff
- بازگشت به آنترپک در OpenAI 429 یک دستور کار قانونی است.
- تازه وارد، تجاري

### دروازه هوش مصنوعی Cloudflare / دروازه هوش مصنوعی ورسل

- موفق شد، عملیات صفر، دوباره امتحان و قابل مشاهده بودن
- بهترین مناسب: برنامه های جاوا اسکریپت سرویس دهنده ای در Cloudflare / Vercel.
- در مقایسه با Kong/Portkey در رایل های حفاظت و محدودیت های سرعت محدود است.

### خود میزبان در مقابل مدیریت

اقامت داده ها عملکرد اجباری است. مراقبت های بهداشتی و مالی پیش فرض خود میزبان (LiteLLM یا Portkey OSS یا Kong). محصولات مصرفی پیش فرض مدیریت (Cloudflare AI Gateway) یا سطح متوسط (Portkey مدیریت). ترکیبی: خود میزبان برای مستاجر تنظیم شده، مدیریت برای دیگران.

### بودجه تاخیر

- لایت للم: 5-15 ms عامه معمولی
- 20 تا 40 ثانیه بالا
- 3-8 ms در بالا
- Cloudflare/Vercel: 1-3 ms overhead (فائده کناری).

تاخیر دروازه مستقیما به TTFT اضافه می شود. برای TTFT P99 < 100 ms SLA، Kong یا Cloudflare. برای P99 < 500 ms، هر.

### موضوع معنوی محدودیت نرخ

تکه توکن ساده تا مقیاس متوسط کار می کند. چندین مستاجر نیاز به پنجره سلئیڈنگ + تخفیف انفجار + تکه هر مستاجر دارد. LiteLLM تکه توکن را می فرستد؛ Kong تکه پنجره سلئیڈنگ را می فرستد؛ Portkey تکه ها را طبقه بندی می کند.

### Gateway + قابل مشاهده + رویتینگ ترکیب

مرحله 17 · 13 (ببیننده بودن) + 16 (مودل مسیر) + 19 (گیته ها) یک لایه تولید هستند. یک ابزار را انتخاب کنید که سه مورد را پوشش دهد یا آنها را با دقت به کار ببرید: اکثر انتشار 2026 هلیکون (ببیننده بودن) یا پورتکی (گاردریل) را با کانگ (سکیل) برای نقش های تقسیم ترکیب می کند.

### شماره هایی که باید به یاد داشته باشی

- لایت للم: در حدود 2000 RPS، 8 جی بی حافظه شکسته می شود.
- پورتکی: 20-40 ms overhead؛ Apache 2.0 از مارس 2026
- کانگ: 228 درصد سریعتر از پورتکی، 859 درصد سریعتر از لایت للم
- قیمت کانگ: 100 دلار/ماه/ماه، 5تا در سطح پلاس
- Cloudflare/Vercel: 1-3 ms در کنار.

```figure
mx-gateway-fallback
```

## ازش استفاده کن

`code/main.py`شبیه سازی مسیر دروازه با بازگشت در سه ارائه دهنده تحت تزریق 429/5xx. گزارش تاخیر، نرخ تلاش مجدد و نرخ ضربه برگشت.

## -باده

این درس به ما کمک می کند`outputs/skill-gateway-picker.md`با توجه به مقیاس، حالت عملیات، رعایت، بودجه تاخیر، یک دروازه انتخاب می کند.

## تمرینات

1. فرار کن`code/main.py`. تنظیم کردن بازگشت از OpenAI→Anthropic→خود میزبان. نرخ ضربه انتظار می رود در نرخ خطا 5٪ ارائه دهنده چیست؟
2. SLA شما TTFT P99 < 200 ms در خط پایه 300 ms است. کدام دروازه ها در بودجه باقی می مانند؟
3. براي يه مشتری مراقبت هاي بهداشتی بايد خود ميزباني بشه + اصلاح اطلاعات شخصي + حسابرسي
4. مقایسه LiteLLM vs Kong: یک تیم باید در چه سقف RPS مهاجرت کند؟
5. طراحی سیاست محدودیت نرخ برای یک سرویس SaaS چند مستاجر: سطح رایگان، سطح آزمایشی، سطح پرداخت شده. سطحی توکن یا پنجره سلایدی؟

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | Process sitting between apps and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated log" | Immutable record of every LLM call |
| Token-bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding-window | "precise rate limit" | Time-windowed rate limiter; better fairness |

## خواندن بیشتر

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
