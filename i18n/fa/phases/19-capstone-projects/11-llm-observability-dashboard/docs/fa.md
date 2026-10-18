# Capstone 11  LLM مشاهده و Eval Dashboard

> لانگفوز به طور کاملاً باز شد آریز فینکس نقشه های 2026 GenAI semconv را منتشر کرد. هلیکون و برینترست هر دو در نسبت هزینه های هر کاربر دو برابر شده اند. OpenLLMetry Traceloop به عنوان ابزار SDK واقعی تبدیل شد. شکل تولید ClickHouse برای ردیف ها، Postgres برای متادتا، Next.js برای UI، و یک ارتش کوچک از کارهای ارزیابی (DeepEval، RAGAS، LLM-judge) که بر روی ردیف نمونه ها اجرا می شود. یک میزبان خود را بسازید، از حداقل چهار خانواده SDK مصرف کنید و نشان دهید که در کمتر از پنج دقیقه بازپسین تزریق شده را می گیرید.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:**P11 · P13 · P17 · P18
**Time:** 25 hours

## مشکل

هر تیم هوش مصنوعی که در سال 2026 ترافیک تولید را اجرا می کند یک سطح مشاهده ای را در کنار مدل نگه می دارد. هزینه های مربوط به هزینه تشخیص توهم نظارت بر حرکت . سیگنال "جیل بریک" داشبورد های SLO اطلاعات اطلاعاتي از تخلیه اطلاعات مرجع های منبع باز  Langfuse، Phoenix، OpenLLMetry  به عنوان طرح مصرفی بر روی کنوانسیون های معنوی OpenTelemetry GenAI به هم پیوستند. شما اکنون می توانید ابزار OpenAI، Anthropic، Google، LangChain، LlamaIndex و vLLM را با یک SDK و ارسال دامنه های سازگار انجام دهید.

شما یک داشبورد خود میزبان را ایجاد خواهید کرد که از حداقل چهار خانواده SDK استفاده می کند، مجموعه ای کوچک از کارهای ارزیابی را بر روی ردیاب نمونه ها اجرا می کند، حرکت و هشدارها را تشخیص می دهد. ردیف اندازه گیری: با توجه به بازگشت به طور عمدی تزریق شده (یک پیامک که تولید PII را آغاز می کند) ، داشبورد آن را دریافت می کند و هشدار را در کمتر از پنج دقیقه اجرا می کند.

## مفهوم

Ingest OTLP HTTP است. SDK دامنه GenAI-semconv را تولید می کند: `gen_ai.system`،`gen_ai.request.model`،`gen_ai.usage.input_tokens`،`gen_ai.response.id`،`llm.prompts`،`llm.completions`. در ClickHouse برای تجزیه و تحلیل ستون ها زمین می گیرد؛ متادتا (کاربری، جلسات، برنامه ها) در Postgres زمین می گیرد.

Evals به عنوان کار دسته ای بر روی ردیف های نمونه ای اجرا می شود. DeepEval وفاداری، سمی و ارتباط پاسخ را نشان می دهد. RAGAS متریک های بازیافت را زمانی که ردیف زمینه بازیافت را دارد، نشان می دهد. قاضیان LLM سفارشی چک های خاص دامنه را اجرا می کنند (فشای PII، پاسخ خارج از سیاست). Eval runs به همان ClickHouse به عنوان دامنه های ارزیابی مرتبط با ردیف اصلی بازیافت می کند.

تشخیص جریان، توزیع فضای گنجانده در طول زمان (با توجه به انحراف PSI یا KL در گنجانده های فوری) و همچنین روند ارزیابی را مشاهده می کند. هشدارها به Prometeus Alertmanager و سپس Slack / PagerDuty تغذیه می کنند. UI Next.js 15 با Recharts است.

## معماری

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## دسته

- مصرف: SDK های OpenTelemetry + کنوانسیون های معنوی GenAI؛ حمل و نقل HTTP OTLP
- جمع کننده: جمع کننده OpenTelemetry با پردازنده نمونه گیری دم (برای کنترل هزینه)
- ذخیره سازی: ClickHouse برای مدت زمان، Postgres برای متاداتا، S3 برای آرشیو رویداد خام
- Evals: DeepEval، RAGAS 0.2، آریز فینیکس بسته ارزیابی کننده، قاضی LLM سفارشی
- انحراف: PSI / KL در هر هفته در پیوند های فوری جمع آوری شده (ترانسفارمرهای جمله)
- هشدار: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server اعمال
- SDK های پشتیبانی شده از جعبه: OpenAI، Anthropic، Google GenAI، LangChain، LlamaIndex، vLLM

```figure
ce-otel-drift
```

## آن را بسازید

1. **Collector config.**OpenTelemetry Collector با گیرنده HTTP OTLP، یک نمونه گیری دم که ۱۰۰٪ از ردیابی های اشتباه و ۱۰٪ از موفقیت ها را حفظ می کند و صادرات به ClickHouse و S3 است.

2. **ClickHouse schema.**جدول`spans`با ستون هایی که نشان دهنده GenAI semconv هستند: `gen_ai_system`،`gen_ai_request_model`،`input_tokens`،`output_tokens`،`latency_ms`،`prompt_hash`،`trace_id`،`parent_span_id`، به علاوه کیف JSON برای بارهای مفید طولانی. شاخص های ثانویه را با user_id و app_id اضافه کنید.

3. **SDK coverage test.**یک برنامه مشتری کوچک با استفاده از هر SDK (OpenAI، Anthropic، Google، LangChain، LlamaIndex، vLLM) با ابزار خودکار OpenLLMetry بنویسید. هر یک از آنها دامنه های GenAI کانونیک را تولید می کند که در ClickHouse فرود می آید.

4. **Eval jobs.**یک کار برنامه ریزی شده، ردیابی نمونه های 15 دقیقه اخیر را می خواند و وفاداری، سمی بودن و ارتباط پاسخ را در DeepEval اجرا می کند. نتایج در طول زمان ارزیابی مرتبط با ردیابی اصلی است.

5. **Custom LLM-judge.**یک قاضی در مورد دزدی اطلاعات شخصی: اگر پاسخ داده شود، یک نگهبان LLM را برای امتیاز احتمال دزدی اطلاعات شخصی تماس بگیرید. پاسخ های بالا در صف انتخاب قرار می گیرند.

6. **Drift detection.**شغل هفتگی حساب PSI بین جمع آوری برنامه های فوری این هفته و خط پایه 4 هفته بعد می کند.

7. **Dashboard.**Next.js 15 با صفحات: مرور کلی (مدد / ثانیه، هزینه / کاربر، تاخیر p95) ، ردیابی (تلاش + آبشار) ، ارزیابی (مزه وفاداری، سمی) ، حرکت (PSI در طول زمان) ، هشدارها.

8. **Alerting chain.**صادر کننده Prometheus مجموعه های امتیاز ارزیابی و درصد های تاخیر را می خواند؛ مسیرهای Alertmanager به Slack برای هشدارها و PagerDuty برای نقض های حیاتی.

9. **Regression probe.**یک خطا تزریق کنید: چت روت ارزیابی شده شروع به لیک کردن SSN های جعلی می کند 1٪ از زمان. MTTR را اندازه گیری کنید: از خطا به هشدار Slack.

## ازش استفاده کن

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## -باده

`outputs/skill-llm-observability.md`در یک برنامه LLM، داشبورد ردیابی خود را جذب می کند، ارزیابی ها، هشدارهای در مورد حرکت را اجرا می کند و در Next.js تقسیم هزینه / کاربر را نشان می دهد.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## تمرینات

1. ابزار سفارشی برای چارچوب Haystack اضافه کنید. دامنه های قانونی را در ClickHouse با faithful تایید کنید `gen_ai.*`ویژگی ها

2. با DeepEval براي ارزیابی کننده هاي فنيکس در همان مسیرها عوض کنين

3. تیز کردن آشکارساز حرکت: محاسبه PSI به هر app-ID به جای جهانی نشان دهید مسیرهای حرکت در هر app.

4. صفحه "تأثر کاربر" را اضافه کنید: هزینه به هر کاربر و نرخ شکست به هر کاربر با خطوط پرتاب.

5. ایجاد یک سیاست نمونه گیری دم که 100٪ از آثار با سمی بودن > 0.5 و یک نمونه طبقه بندی شده 10٪ از بقیه را حفظ کند.

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## خواندن بیشتر

- [Langfuse](https://github.com/langfuse/langfuse) پلت فرم مشاهده ای باز هسته ای مرجع
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) مرجع متناوب با حمایت قوی از حرکت
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) خانواده SDK های خودآداپذیری
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) طرح مصرف
- [Helicone](https://www.helicone.ai) مشاهده ای جایگزین میزبان
- [Braintrust](https://www.braintrust.dev) پلتفرم اول ارزیابی جایگزین
- [ClickHouse documentation](https://clickhouse.com/docs) فروشگاه زمان عمودی
- [DeepEval](https://github.com/confident-ai/deepeval) کتابخانه ارزیابی کننده
