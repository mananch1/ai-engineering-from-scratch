# چرا چند عامل؟

> یک مامور به دیوار می خورد حرکت هوشمندانه یک مامور بزرگتر نیست بلکه بیشتر مامورین است

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## اهداف یادگیری

- مشخص کردن سقف یک عامل (تجاوز زمینه، تخصص مخلوط، گوشه بطری سلسله ای) و توضیح دهید که تقسیم به چندین عامل حرکت درست است
- الگوهای آرکیستراسیون را مقایسه کنید (پایپ لاین، فان آئو، نظارت، سلسله مراتب) و یکی مناسب را برای یک ساختار کار داده انتخاب کنید
- طراحی یک سیستم چند عامل با مرزهای نقش واضح، وضعیت مشترک و قرارداد ارتباطی
- تجزیه و تحلیل تعادل پیچیدگی چند عامل (خاموشی، هزینه، دشواری های دیبگ) در مقابل سادگی یک عامل

## مشکل

شما یک عامل واحد را در مرحله 14 ساخته اید. این کار می کند. می تواند فایل ها را بخواند، دستورات را اجرا کند، API ها را فراخشد و در مورد نتایج استدلال کند. سپس شما آن را به یک پایگاه کد واقعی نشان می دهید: 200 فایل، سه زبان، تست هایی که به زیرساخت بستگی دارند، و نیاز به تحقیق در API های خارجی قبل از نوشتن کد.

این عامل خفه می شود. نه به این دلیل که LLM احمق است، بلکه به این دلیل که وظیفه فراتر از آنچه یک حلقه عامل می تواند انجام دهد. پنجره زمینه ای با محتوای فایل پر می شود. عامل فراموش می کند که چهل تماس ابزار قبل چهل خوانده است. سعی می کند یک محقق، یک کدر و یک بازرس در یک زمان باشد، و هر سه را به خوبی انجام نمی دهد.

اين سقف واحد عامله. هر بار که يه کار نياز داره:

- **More context than fits in one window**- خواندن 50 پرونده از 200 هزار توکن عبور ميکنه
- **Different expertise at different stages**- تحقیق نیاز به انگیزه ای متفاوت از تولید کد دارد
- **Work that can happen in parallel**چرا سه پرونده رو به دنباله دار بخونيم وقتي ميتونيم آنها رو همزمان بخونيم؟

## مفهوم

### سقف یک عامل

یک عامل واحد یک حلقه، یک پنجره زمینه، یک پیام سیستم است. تصور کنید:

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

سه تا چيز شکسته شده:

1. **Context saturation**تا نوبت 30، مامور 150 هزار توکن از محتوای فایل، محصولات دستور و استدلال قبلی مصرف کرده است. جزئیات مهم از نوبت 5 از دست می روند.

2. **Role confusion**- یک پیام سیستم که می گوید "شما یک محقق، کدگر، بازرس و آزمایش کننده هستید" یک عامل تولید می کند که نیمه تحقیق می کند، نیمه کد می کند و هرگز بازبینی را تمام نمی کند.

3. **Sequential bottleneck**-عميل پرونده اي رو ميخواد، بعد پرونده ب و بعد پرونده سي سه تماس سريال براي مدرک تحصيلي، سه عمليات سريال براي ابزار

### راه حل چند عامل

کار رو تقسیم کن. به هر مامور يه کار بده، يه پنجره ي زمینه و يه پيغام سيستمي براي اون کار تنظیم شده

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

هر مامور داره:
- یک پیام رسان سیستم متمرکز ("شما یک بازرس کد هستید. تنها کاری که شما انجام می دهید یافتن اشکال است. ")
- پنجره ی زمینه ی خود (که توسط کارهای دیگر عوامل آلوده نشده است)
- قرارداد روشن درآمدی/خروجی (ملاحظات تحقیقاتی دریافت می کند، کد خروجی)

### سیستم های واقعی که این کار را انجام می دهند

**Claude Code subagents**- وقتي کلاود کود يه ذوبگنت رو به وجود مياره`Task`، این یک عامل کودک با یک وظیفه محدودی ایجاد می کند. پدر و مادر زمینه اش را تمیز نگه می دارد. کودک کار متمرکز را انجام می دهد و خلاصه ای را باز می گرداند.

**Devin**برنامه ريزي کار را به مراحل مي شکند، برنامه ريزي كود كد مي نويسد، مرورگر اسناد را بررسي مي كند. هر يك از آنها زمینه ي جداگانه اي دارد.

**Multi-agent coding teams (SWE-bench)**- سیستم های با عملکرد بالا در SWE-bench از یک محقق استفاده می کنند که پایگاه کد را می خواند، یک برنامه نویس که تصحیح را طراحی می کند و یک کدگر که آن را اجرا می کند.

**ChatGPT Deep Research**- عوامل جستجو چندگانه را به طور موازی تولید می کند، هر کدام زاویه متفاوتی را کشف می کنند، سپس نتایج را ترکیب می کند.

### طیف

چند عامل دوگانه نیست، بلکه طیف:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent**-یک حلقه، یک پرامپرت .

**Subagents**- يه پدر و مادر بچه ها رو براي انجام فرعي کار هاي متمرکز ميخوره پدر و مادر برنامه رو حفظ ميکنه بچه ها گزارش ميده اينکار رو کلاود کود ميکنه

**Pipeline**- اجنتی ها به ترتیب اجرا می شوند. محصول اجنتی A به ورودی اجنتی B تبدیل می شود. برای جریان های کاری مرحله ای مناسب است: تحقیق -> کد -> بررسی -> آزمایش.

**Team**- ماموران در موازی با یک اتوبوس پیام مشترک کار می کنند. هر کدام نقش دارند. یک هماهنگی سازنده. خوب است وقتی مهارت های مختلف همزمان مورد نیاز است.

**Swarm**-آژانتهاي مشابه يا نزديک به هم با حالت مشترک . بدون آرکستر ثابت .آژانتهاي کار را از صف ميبرند . براي وظایف موازی با تولید بالا خوب است

### چهار الگوی چند عامل

#### مدل ۱: خط لوله

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

هر عامل اطلاعات رو تبديل ميکنه و منتقل ميکنه. ساده براي استدلال. شکست در يک مرحله باقيات رو مسدود ميکنه.

#### مدل 2: فان Out / فان In

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

کار رو در عوامل موازی تقسیم کن، سپس نتایج رو ترکیب کن. برای کارهایی که به زیرکار مستقل تجزیه می شوند خوب است.

#### مدل سوم: کارکن آرکستر

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

یک گروه سازنده هوشمند تصمیم می گیرد چه کاری انجام دهد، به کارگران می دهد و نتایج را ترکیب می کند. گروه سازنده خودش یک عامل با ابزار برای تولید مثل کارگران است.

#### مدل ۴: جمع همسالان

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

هيچ آرکستراتيک مرکزی وجود نداره، ماموران با همديگه ارتباط برقرار ميکنن، تصميمها از تعامل مياد، تر از همديگه مشکل تر است، اما به اندازه چند مامور مياد

### چه زمانی نباید از چند عامل استفاده کنید

چند عامل به پیچیدگی اضافه می کند. هر پیام بین عوامل یک نقطه شکست بالقوه است. اشکال زدایی از "یک مکالمه را بخوانید" به "پیام های ردیابی در پنج عامل" می رود.

**Stay single-agent when:**
- این کار در یک پنجره زمینه (در زیر ~ 100k توکن داده های کاری) مناسب است
- شما به دستورات مختلف سیستم برای مراحل مختلف نیاز ندارید
- اعدام متسلسل به اندازه کافی سریع است
- این کار به اندازه کافی ساده است که تقسیم آن اضافه کردن هزینه های بیشتر از ارزش

**The complexity cost:**
- هر مرز عامل یک مرحله فشرده سازی ضایع کننده است: زمینه کامل عامل A به یک پیام برای عامل B خلاصه می شود
- منطق هماهنگی (چه کسی چه کاری را انجام می دهد، چه زمانی، در چه ترتیب) منبع خود از اشکال است
- افزایش تاخیر: N عوامل به معنای N سری LLM تماس حداقل، بیشتر اگر آنها نیاز به صحبت کردن به عقب و جلو
- هزینه چند برابر می شود: هر عامل به طور مستقل توکن ها را می سوزاند

قانون عمومي: اگر یک کار کمتر از 20 تماس ابزار را می گیرد و در 100k توکن قرار می گیرد، آن را یک عامل نگه دارید.

```figure
swarm-messages
```

## آن را بسازید

### مرحله ی اول: عامل تک تک

این یک عامل واحد است که سعی می کند همه چیز را انجام دهد. این یک پیام سیستم بزرگ و یک پنجره زمینه ای دارد که تحقیقات، کد و بررسی ها را نگه می دارد:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

مشکلات این رویکرد:
- پنجره زمینه با هر مرحله رشد می کند. از مرحله بررسی، شامل یادداشت های تحقیقاتی و کد و استدلال قبلی است.
- دستور سیستم عمومی است. نمیتونه برای هر مرحله تنظیم بشه.
- هيچ چيز موازی نميگيره

### مرحله دوم: ماموران تخصصی

حالا بهشون بديد، هر مامور يه شغل داره

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

هر متخصص یک پیام متمرکز دارد. هر یک یک یک پنجره زمینه تمیز با فقط ورودی که نیاز دارد.

### مرحله سوم: با ارسال پیام ها هماهنگی برقرار کنید

به متخصصين خبر بده با پيام صريح:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

هر مامور فقط پيام هاي ارجاعي را دریافت ميکنه. هيچ تلوثي در زمینه اي وجود نداره. 50 هزار توکن اسناد مطالعه شده توسط محقق هرگز وارد سياق بازرس نمي شه.

### مرحله چهارم: مقایسه

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

نسخه چند عامل از توکن های کل بیشتری استفاده می کند (سه عامل، سه تماس LLM جداگانه) اما زمینه هر عامل تمیز باقی می ماند. کیفیت هر مرحله بهبود می یابد زیرا پرامپت سیستم تخصصی است.

## ازش استفاده کن

این درس یک دستور کار قابل استفاده برای تصمیم گیری در مورد اینکه چه زمانی باید به چند عامل بپردازید، را تولید می کند.`outputs/prompt-multi-agent-decision.md`. .

## تمرینات

1. یک متخصص چهارم را اضافه کنید: یک عامل "تستر" که کد را از کدگر دریافت می کند و بازبینی بازخورد را از بازبینی کننده می کند، سپس آزمایشات را می نویسد
2. تغییر خط لوله به طوری که بازرس می تواند بازخورد را به کدگر برای یک حلقه بازخورد ارسال کند (ماکس 2 دور)
3. تبدیل خط لوله دنباله دار به یک فان-آउट: محقق و یک "تحلیلی نیاز" را به طور موازی اجرا کنید، سپس تولیدات آنها را قبل از انتقال به کدگر ترکیب کنید

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | A set of peer agents with shared state and no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and delegates but may not do the actual work. |
| Coordinator | "The traffic cop" | A non-agent component (often just code, not an LLM) that routes messages between agents based on rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns that arise from agent interactions but were not explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting a task across parallel agents (fan-out), then combining their results (fan-in). |
| Message passing | "Agents talk to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## خواندن بیشتر

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)- بررسی الگوهای چند عامل
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)- چارچوب مکالمه چند آژانس مایکروسافت
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code)- چطور کلاود کود با کار نوتیفیکیشن می کند
- [CrewAI documentation](https://docs.crewai.com/)- چارچوب چند عامل مبتنی بر نقش
