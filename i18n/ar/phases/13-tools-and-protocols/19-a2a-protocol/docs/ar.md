# A2A  بروتوكول وكيل إلى وكيل

> المكتب العمومي هو العميل إلى الأداة. A2A (Agent2Agent) هو وكيل إلى وكيل بروتوكول مفتوح للسماح وكلاء غير مرئية بنيت على إطار مختلف التعاون. أصدرتها جوجل في أبريل 2025 ، وتبرعت إلى مؤسسة لينكس في يونيو 2025 ، ووصلت إلى v1.0 في أبريل 2026 مع 150 + مؤيدين بما في ذلك AWS و Cisco و Microsoft و Salesforce و SAP و ServiceNow. لقد استوعب نظام إيك بي ام ACP و أضاف تمديد المدفوعات AP2. هذه الدروس تتبع بطاقة العميل، دورة حياة المهمة، والثلاثة إرتباطات بروتوكول، باستخدام أسماء الأسلاك A2A 1.0.1.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## أهداف التعلم

- تمييز بين حالة استخدام وكيل إلى أداة (MCP) من حالة استخدام وكيل إلى وكيل (A2A).
- نشر بطاقة العميل في `/.well-known/agent-card.json`مع مهارات و`supportedInterfaces`البيانات المعدنية
- إمتحان دورة حياة المهمة: `TASK_STATE_SUBMITTED`،`TASK_STATE_WORKING`،`TASK_STATE_INPUT_REQUIRED`، و الحالات النهائية`TASK_STATE_COMPLETED`،`TASK_STATE_FAILED`،`TASK_STATE_CANCELED`،`TASK_STATE_REJECTED`. . .
- استخدم رسائل تحتوي كل جزء على واحد من`text`،`raw`،`url`أو`data`و الأثاث كمخرجات

## المشكلة

وكيل خدمة العملاء يحتاج إلى تفويض كتابة التقرير إلى وكيل كاتب متخصص. الخيارات قبل A2A:

- إطار إطار إعادة التأهيل المخصص يعمل لكن كل إزواج هو مرة واحدة
- قاعدة شفرة مشتركة تتطلب من العملاء أن يستخدموا نفس الإطار
- المكسبين: لا يناسب: المكسبين هو للدعوة الأدوات، وليس لعاملين يتعاونونون مع الحفاظ على كل عامل من التفكير الداخلي غير الشفاف.

A2A تملأ الفجوة. فإنه ينمثل التفاعل عندما يرسل وكيل واحد مهمة إلى آخر، مع دورة حياة، والرسائل، والقطع الأثرية. يبقى حالة الداخلية للوكيل المطلق غير واضحة  يرى المدعو فقط عمليات انتقال حالة المهمة والمخرجات النهائية.

A2A هو بروتوكول "دع العملاء عبر الإطار يتحدثون مع بعضهم البعض". لا يحل محل MCP؛ والثنين يكملون.

## المفهوم

### وكيل بطاقة

كل وكيل متوافق مع A2A ينشر بطاقة في `/.well-known/agent-card.json`:

```json
{
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "version": "1.2.0",
  "supportedInterfaces": [
    {
      "url": "https://research.example.com/a2a",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true},
  "securitySchemes": {
    "bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}
  },
  "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/markdown"],
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "tags": ["research", "summarization"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["text/markdown"]
    }
  ]
}
```

الاكتشاف يستند إلى عنوانات: احضر البطاقة، اختر الأول `supportedInterfaces`إدخال`protocolBinding`العميل يتحدث ويعد مهاراتك أطر الدخول والخروج هي أنواع الوسائط

### بطاقات العميل الموقعة

بطاقة يمكن أن تحمل`signatures`كل مدخل هو JWS (RFC 7515) حاسوب على بطاقة RFC 8785 JSON القنوني، مع `signatures`المستهلكون يطرحون البطاقة بنفس الطريقة ويتحققون. يمنعون التمثيل.

### دورة حياة المهمة

```text
TASK_STATE_SUBMITTED
  -> TASK_STATE_WORKING
  -> TASK_STATE_COMPLETED | TASK_STATE_FAILED | TASK_STATE_CANCELED | TASK_STATE_REJECTED

TASK_STATE_WORKING
  -> TASK_STATE_INPUT_REQUIRED
  -> TASK_STATE_WORKING (the client sends a message with the same taskId)
```

العملاء يبدأون مع `SendMessage`و الخادم يخلق المهمة. العميل المطلوب ينتقل عبر الولايات. العملاء يستطلعون مع`GetTask`أو تدفق على SSE مع `SendStreamingMessage`و`SubscribeToTask`.التي تحمل النهر`statusUpdate`و`artifactUpdate`الحالات وتغلق عندما تصل المهمة إلى حالة نهائية.`final`العلم

### الرسائل والأجزاء

رسالة لديها`messageId`، أ`role`(`ROLE_USER`أو`ROLE_AGENT`), و جزء واحد أو أكثر. كل جزء يحمل بالضبط حقل محتوي واحد، وهذا الاسم من الحقل هو النوع. لا يوجد `kind`المجال

- `text`: محتوى بسيط.
- `raw`: البايتات الملفة، قاعدة64 في JSON، عادة مع `filename`و`mediaType`. . .
- `url`: رابط لمحتويات الملف
- `data`: تحميل JSON المهيكلي (المدخول المهيكلي للوكيل المطلوب).

مثال:

```json
{
  "messageId": "msg-001",
  "role": "ROLE_USER",
  "parts": [
    {"text": "Summarize this paper."},
    {"raw": "...", "filename": "paper.pdf", "mediaType": "application/pdf"},
    {"data": {"targetLength": "3 paragraphs"}, "mediaType": "application/json"}
  ]
}
```

### الأثاث

المخرجات هي الأثاث، وليس السلاسل الخام. الأثاث هو المخرج المسمى، المخطط:

```json
{
  "artifactId": "art-001",
  "name": "summary",
  "parts": [{"text": "...", "mediaType": "text/markdown"}]
}
```

يمكن نشر الأثاث كقطع`artifactUpdate`الحدث يحمل القطع الأثرية بالإضافة`append`و`lastChunk`المتصل يتراكم

### ثلاثة ملزمات بروتوكول

1. **JSON-RPC 2.0 over HTTP**(`JSONRPC`(). POST للطلبات، SSE للاتصال.`SendMessage`،`SendStreamingMessage`،`GetTask`،`ListTasks`،`CancelTask`،`SubscribeToTask`،`CreateTaskPushNotificationConfig`،`GetTaskPushNotificationConfig`،`ListTaskPushNotificationConfigs`،`DeleteTaskPushNotificationConfig`و`GetExtendedAgentCard`. . .
2. **gRPC**(`GRPC`(). لبيئات المؤسسات حيث gRPC هو الأصلي. أسماء الطرق نفسها.
3. **HTTP+JSON/REST**(`HTTP+JSON`) عنوانات المورد مثل `POST /message:send`و`GET /tasks/{id}`. . .

كل ثلاثة التزامات تحمل نفس نموذج البيانات.`supportedInterfaces`اسم الدخول واحد ملزم و `protocolVersion`العملاء يرسلون الرأس`A2A-Version: 1.0`في كل طلب، لأن خادم يقرأ طلب بدونها كإصدار 0.3.

```http
POST /a2a HTTP/1.1
Host: research.example.com
Content-Type: application/json
A2A-Version: 1.0

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{"text": "Summarize this paper."}]
    }
  }
}
```

### الحفاظ على الكفاءة

مبدأ تصميم رئيسي: حالة العميل الداخلي المطلق غير شفافة. يرى المدعو حالة المهمة والقطع الأثرية. سلسلة الأفكار للعميل المطلق، ودعوات أدائه، وفوضيته الفرعية للعميل غير مرئية. هذا يختلف عن MCP، حيث الدعوات الأداة شفافة.

المنطق: A2A تمكن المنافسين من التعاون دون الكشف عن الداخلية. A2A يمكن أن يكون "تصل هذه وكيل خدمة العملاء" دون أن يتعلم المتصل كيفية تنفيذ هذا وكيل الخدمة.

### خط زمني

- **2025-04-09.**جوجل تعلن عن A2A
- **2025-06-23.**تبرع لمؤسسة لينكس
- **2025-08.**يمتص جهاز "إيك بي إم"
- **2025-09.**سفن AP2 تمديد (دفع العملاء).
- **2026-04.**إصدار v1.0 مع 150 + منظمة دعم.

### العلاقة مع المؤسسة

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | Agent Card `securitySchemes` + `securityRequirements` |
| Transport | Stdio / Streamable HTTP | JSON-RPC / gRPC / HTTP+JSON |

استخدم MCP عندما تريد استدعاء أداة محددة. استخدم A2A عندما تريد تفويض مهمة كاملة إلى وكيل آخر. العديد من أنظمة الإنتاج تستخدم كليهما: وكيل يستخدم MCP لتطبيق أداة و A2A لتطبيق تعاونه.

```figure
a2a-task-lifecycle
```

## استخدمها

`code/main.py`يطبق قاعدة A2A الحد الأدنى: وكيل الكتاب ينشر بطاقته، وكيل البحث يرسلها `SendMessage`الطلب مع جزء PDF و تعليمات نصية، والمهام تتحرك من خلال `TASK_STATE_WORKING``TASK_STATE_INPUT_REQUIRED``TASK_STATE_WORKING``TASK_STATE_COMPLETED`قبل إرجاع كائنات نصية. جميع stdlib؛ يستخدم نقل في الذاكرة للتركيز على أشكال الرسالة.

ما الذي يجب أن ننظر إليه:

- شكل بطاقة العميل JSON.
- تعيين صفحة المهام من جانب الخادم والانتقالات الحالة.
- الأجزاء التي يتم كتابتها بحيث يكون حقل المحتوى موجوداً.
- `TASK_STATE_INPUT_REQUIRED`فرع منتصف المهمة.
- العودة الفنية عند الانتهاء.

## أرسله

هذا الدرس يُنتج`outputs/skill-a2a-agent-spec.md`. بالنظر إلى وكيل جديد يجب أن يكون قابلاً للدعوة من قبل وكلاء آخرين، فإن المهارة تنتج بطاقة وكيل JSON، مخطط مهارات، ورسم خط النهاية.

## التمارين

1. أركض`code/main.py`تتبع دورة حياة المهمة بأكملها، بما في ذلك`TASK_STATE_INPUT_REQUIRED`توقف عندما يطلب العميل المكالم توضيح

2. أضف بطاقة وكيل موقعة ضع إدخال واحد في "جيه دبليو إس"`signatures`مع`alg`المحددة إلى`HS256`, توقيع بطاقة JSON القنوني دون `signatures`إكتب مؤكداً و تأكد أنه يفشل على بطاقة متحولة

3. تنفيذ المهام المتدفقة مع `SendStreamingMessage`: وكيل الكتاب يُبعث`task`، ثلاثة `artifactUpdate`قطع، و `statusUpdate`مع`TASK_STATE_COMPLETED`ثم يغلق التيار، ويتراكم المُتصل بالقطع

4. تصميم وكيل A2A الذي يلف خادم MCP. رسم كل أداة MCP إلى مهارة A2A. لاحظ التنازلات  ما هو الضموضة التي فقدت؟

5. اقرأ إعلان A2A v1.0 وتحدد الميزة الوحيدة التي لم يتم تنفيذها بعد من قبل أي إطار اعتبارا من أبريل 2026. (لمحة: يتعلق الأمر بتفويض المهام متعددة المكالمات).

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Open protocol for opaque agent collaboration |
| Agent Card | "`/.well-known/agent-card.json`" | Published metadata describing an agent's skills and `supportedInterfaces` |
| Skill | "A callable unit" | A named operation the agent supports (analog to MCP tool) |
| Task | "Unit of delegation" | A work item with a lifecycle and final artifact |
| Message | "Task input" | Carries Parts (`text`, `raw`, `url`, `data`) |
| Part | "Typed chunk" | Exactly one of `text` / `raw` / `url` / `data`, plus optional `mediaType`; no `kind` field |
| Artifact | "Task output" | Named, typed output returned on completion |
| AP2 | "Agent Payments Protocol" | Payments extension built on A2A; card signing is core A2A (`signatures`) |
| Opacity | "Black-box collaboration" | Called agent's internals are hidden from caller |
| `TASK_STATE_INPUT_REQUIRED` | "Task pause" | Interrupted state when the agent needs more info |

## المزيد من القراءة

- [a2a-protocol.org](https://a2a-protocol.org/latest/) مواصفات A2A القنونية
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) تنفيذات مرجعية و SDKs
- [A2A v1.0.1 release](https://github.com/a2aproject/A2A/tree/v1.0.1): المسموح بها`docs/specification.md`والنظام التنظيمي`specification/a2a.proto`هذا الدرس يتبع
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) يونيو 2025 تحويل الحكم
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade)خريطة الطريق وسرعة الشركاء
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) إصدارات إصدارات v1.0 وتوجيهات تراجعية
