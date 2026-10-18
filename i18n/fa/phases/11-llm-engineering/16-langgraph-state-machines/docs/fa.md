# دستگاه های دولتی  نمودارها، گره ها، نقاط بازرسی

> یک حلقه ReAct نوشته شده به دست یک `while True`همان حلقه ای که به عنوان یک نمودار صریح نوشته شده چیزی است که می توانید از طریق آن کنترل کنید، قطع کنید، شاخه بزنید و در زمان سفر کنید. عامل تغییر نکرده است. حاشیه اطرافش تغییر کرده است.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## مشکل

شما یک عامل تماس عملکرد را ارسال می کنید. این سه نوبت کار می کند، سپس چیزی اشتباه می شود: مدل یک ابزار را امتحان می کند که 500 را باز می آورد، کاربر در وسط کار ذهن خود را تغییر می دهد، یا عامل تصمیم می گیرد که بدون امضای انسانی سفارش را بازپرداخت کند.`while True:`شما نمی توانید آن را متوقف کنید، نمی توانید آن را برگردانید، و شما نمی توانید شاخه به "چه اگر مدل ابزار دیگر را انتخاب کرد". در لحظه شما ارسال این گذشته یک دمو، آژانس تبدیل به یک جعبه سیاه که یا کار کرد یا نه.

قدم بعدش وقتی که می بینی واضحه عامل قبلاً یک ماشین دولتی است  سیستم فوری پلاس سابقه پیام پلاس تماس های ابزار منتظر پلاس عمل بعدی. دستگاه دولت را واضح کنید: گره ها برای "نموذج فکر می کند"، "یک ابزار اجرا می شود"، "یک انسان تایید می کند"، و حاشیه ها برای انتقال مشروط بین آنها. هنگامی که نمودار صریح است، هنیس چهار چیز را به صورت رایگان دریافت می کند: چکپوائنتینگ (حالت ذخیره بین مراحل) ، قطع (مقطوع برای یک انسان) ، جریان (توکین های جریان و رویدادهای میانگین) و سفر زمان (بازگردان به حالت قبلی و شاخه دیگری را امتحان کنید).

اجرای مرجع این انتزاع LangGraph است. این یک چارچوب عامل در مفهوم LangChain نیست ("این یک عامل اجرا کننده است، خوش شانس"). این یک زمان اجرا گراف با حالت درجه اول، استقامت درجه اول و قطعات درجه اول است. حلقه عامل چیزی است که شما می رسمید، نه چیزی که شما دست نوشته اید.

## مفهوم

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

A`StateGraph`سه تا چيز داره

1. **State.**یک dict تایپ شده (TypedDict یا مدل Pydantic) که از طریق نمودار جریان می یابد. هر گره حالت کامل را دریافت می کند و یک به روز رسانی جزئی را باز می گرداند، که LangGraph با استفاده از *reducer* در هر زمینه `operator.add`برای لیست هایی که باید جمع شوند، بطور پیش فرض اضافه کنید.
2. **Nodes.**عملکردهای پایتون`state -> partial_state`هر یک از این مراحل جداگانه است: "مودل را صدا کنید، ابزارها را اجرا کنید، خلاصه کنید".
3. **Edges.**انتقال بین گره ها. حواشی جامد به یک جا می روند. حواشی مشروط به یک تابع روتر می پردازند`state -> next_node_name`بنابراین نمودار می تواند شاخه در تولید مدل.

شما نمودار را مرتب می کنید. مرتب کردن توپولوژی را متصل می کند، یک چک پوائنتر (اختیاری اما ضروری برای تولید) را متصل می کند و یک قابل اجرا را باز می گرداند. شما آن را با یک حالت اولیه و یک `thread_id`هر مرحله از اعدام يه نقطه بازي براي تفتيش ادامه داره`(thread_id, checkpoint_id)`. .

### چهار ابرقدرت

**Checkpointing.**هر انتقال گره ای حالت جدید را به یک فروشگاه می نویسد (در حافظه برای تست ها، Postgres/Redis/SQLite برای prod).`thread_id`.گراف از جایی که توقف کرده است برمی گردد

**Interrupts.**یک گره را با  مشخص کنید`interrupt_before=["human_review"]`و اجرای قبل از اجرای این گره متوقف می شود. حالت ادامه دارد. API شما به کاربر با "انتظار تایید" پاسخ می دهد. یک درخواست بعد به همان `thread_id`با`Command(resume=...)`اعدام دوباره شروع ميشه

**Streaming.** `graph.stream(state, mode="updates")`به صورت صورت واقعي به دلتاس ايالت مياد`mode="messages"`توکن های LLM را در داخل گره های مدل پخش می کند. `mode="values"`شما انتخاب می کنید که چه چیزی در UI شما ظاهر شود.

**Time-travel.** `graph.get_state_history(thread_id)`تمام دفترچه بازرسی رو برگردونه.`checkpoint_id`به`graph.invoke`برای بازیافت (چه می شد اگر مدل ابزار B را انتخاب کرده بود؟) و برای تست های بازپسین که ردیابی تولید را تکرار می کنند.

### کاهش دهنده ها نکته ی مهم هستند

هر میدان حالت دارای یک کم کننده است. اکثر پیش فرض ها خوب هستند  یک مقدار جدید قدیمی را بیش از حد می نویسد. اما لیست پیام نیاز دارد `operator.add`بنابراین پیام های جدید به جای جایگزینی اضافه می شوند. حاشیه های موازی به روز رسانی خود را از طریق کاهش دهنده ترکیب می کنند. اگر دو گره هر دو به روز شوند`messages`و تو فراموش کردي`Annotated[list, add_messages]`،دومی به طور خاموش برنده می شود و شما نیمی از نوبت را از دست می دهید. کم کننده تنها چیز ظریف در کتابخانه است. درست کنید و بقیه ترکیب می شود.

### نمودار ReAct در چهار گره

یک عامل تولید ReAct چهار گره و دو لبه دارد:

1. `agent` به LLM با تاریخچه پیام فعلی تماس می گیرد. پیام دستیار را باز می گرداند (که ممکن است شامل tool_calls باشد).
2. `tools` هر تماس tool_call را در آخرین پیام دستیار اجرا می کند، نتایج ابزار را به عنوان پیام ابزار اضافه می کند.
3. يه لبه مشروط از`agent`که مسیرها به`tools`اگر آخرین پیام دارای tool_calls باشد، در غیر این صورت به `END`. .
4. يه لبه اي از`tools`برگردیم به`agent`. .

این تمام است. شما کامل حلقه ReAct (فکر → عمل → مشاهده → فکر → ...) را با چک پوائنٹنگ، قطع و پخش، در حدود 40 خط کد دریافت می کنید.

### StateGraph vs ارسال (فانو)

`Send(node_name, state)`اجازه می دهد یک گره فرعی موازی را ارسال کند. مثال: عامل تصمیم می گیرد سه بازیافت کننده را به یکباره جستجو کند. هر یک `Send`این روش نشان دهنده الگوی کارگران-ورستراتور بدون رشته سازی ابتدایی است.

### زیرنویس

یک نمودار مرتب شده می تواند یک گره در نمودار دیگر باشد. نمودار بیرونی یک گره را می بیند؛ نمودار داخلی دارای وضعیت و نقاط کنترل خود است. این گونه تیم ها عوامل کارکن نظارت را ایجاد می کنند: نمودار نظارت قصد کاربر را به زیرگراف کارکن در هر دامنه هدایت می کند.

```figure
l5-state-graph-ledger
```

## آن را بسازید

### مرحله اول: حالت و گره ها

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages`کاهش دهنده ای است که باعث جمع آوری لیست پیام ها به جای اضافه کردن آن می شود. فراموش کردن آن رایج ترین خطای LangGraph است.

### مرحله دوم: با یک رشته اجرا کنید

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

هر تازه کاری یک فرمان است`{node_name: state_delta}`. فرونتند شما می تواند این ها را به UI پخش کند تا کاربران ببینند "عميل فکر می کند ... به جستجو_ویب زنگ می زند ... نتیجه پیدا می کند ... پاسخ می دهد. "

### مرحله سوم: اضافه کردن یک اختلال انسانی در حلقه

یک گره را نشان دهید تا قبل از اجرا توقف کند.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

حالت، نقطه بازرسی و موضوع همه در طول وقفه باقی می مانند.

### مرحله 4: سفر زمان برای دیبگینگ

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

گذر کردن`None`وقتی ورودی از نقطه بازرسی داده شده باز می گردد، گذراندن یک مقدار آن را به عنوان یک بروزرسانی به حالت نقطه بازرسی قبل از شروع به کار اضافه می کند. این نحوه بازخورد یک عامل بد بدون بازخوردگی کل مکالمه است.

### مرحله 5: چک پوائنتر را برای تولید عوض کنید

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite، Redis و Postgres فرستاده شده`MemorySaver`هر چيزي که در طول باز کردن ادامه داره يه فروشگاه واقعي ميخواد

## مهارت

> شما به عنوان گراف ها، نه به عنوان`while True`حلقه ها

قبل از اینکه به لانگ گراف برسید، یک طراحی ۶۰ ثانیه ای انجام دهید:

1. **Name the nodes.**هر تصمیمی و عمل متمایز یا تاثیر جانبی یک گره است. "آژانت فکر می کند،" "آزهار اجرا می شود،" "مراجع تأیید می کند،" "سرید پاسخ". اگر نمی توانید آنها را لیست کنید، کار هنوز به شکل عامل نیست.
2. **Declare the state.**حداقل تایپ شده با یک کم کننده برای هر میدان لیست. همه چیز را در آن قرار ندهید`messages`; زمینه های خاص وظیفه را بالا ببرید (یک کار)`plan`، یک`budget`ديدار، یک`retrieved_docs`لیست) به سطح بالا.
3. **Draw the edges.**ثابت است مگر اینکه مرحله بعدی به محصول مدل بستگی داشته باشد. هر لبه مشروط به یک تابع روتر با شاخه های نامگذاری شده نیاز دارد.
4. **Choose a checkpointer up front.** `MemorySaver`برای تست ها، Postgres/Redis/SQLite برای هر چیز دیگری. بدون یک  هیچ چک پوائنتر به معنای هیچ رزومه، هیچ وقفه، هیچ سفر زمان نیست.
5. **Decide interrupts before tools run, not after.**تاییدات به یک گره تاثیر جانبی می رود تا بتوانید قبل از آسیب را لغو کنید؛ اعتبارگذاری به طرف مدل می رود تا بتوانید تماس های بد را ارزان تر رد کنید.
6. **Stream by default.** `mode="updates"`برای UI`mode="messages"`برای جریان سطح توکن در داخل گره های مدل، `mode="values"`برای عکس های کامل در طول ارزیابی.

رد کردن ارسال یک عامل LangGraph که هیچ چک پوائنتر ندارد، رد کردن ارسال یک که قطع * پس از * اثر جانبی. رد کردن ارسال یک`messages`فیلدی بدون `add_messages`به عنوان کاهش دهنده آن

## تمرینات

1. **Easy.**نمودار چهار گره ReAct را با یک ابزار محاسب و یک ابزار جستجو وب پیاده سازی کنید.`list(app.get_state_history(config))`حداقل چهار نقطه بازرسی برای یک مکالمه دو نوبت به دست می آورد.
2. **Medium.**اضافه کنید`planner`گره ای که قبل از آن اجرا می شود`agent`و نوشته است که ساختار یافته است`plan: list[str]`به ایالت بريم`agent`خط خط را به عنوان انجام شده نشان دهید.`plan`در یک رزومه نقطه بازرسی گم شده است (کم کننده اشتباه).
3. **Hard.**یک نمودار نظارت را بسازید که بین سه زیر نمودار (`researcher`،`writer`،`reviewer`) استفاده می کند`Send`هر ذرهگراف دارای حالت و نقطه بازرسی خودش هست.`interrupt_before=["writer"]`بر روی نمودار بیرونی تا انسان بتواند گزارش تحقیق را تایید کند. تایید کنید که سفر زمان از یک نقطه بازرسی قبلی فقط شاخه شکاف را دوباره اجرا می کند.

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| StateGraph | "The LangGraph graph" | The builder object you add nodes and edges to before compile. |
| Reducer | "How the field merges" | A function `(old, new) -> merged` applied when a node returns an update for that field; default is overwrite, `add_messages` appends. |
| Thread | "A conversation ID" | A `thread_id` string that scopes all checkpoints for one session. |
| Checkpoint | "A paused state" | A persisted snapshot of the full graph state after a node transition, keyed on `(thread_id, checkpoint_id)`. |
| Interrupt | "Pause for a human" | `interrupt_before` / `interrupt_after` stop execution at a node boundary; resume with `Command(resume=...)`. |
| Time-travel | "Fork from a prior step" | `graph.invoke(None, config_with_old_checkpoint_id)` replays from that checkpoint forward. |
| Send | "Parallel subgraph dispatch" | A constructor a node can return to spawn N parallel executions of a target node. |
| Subgraph | "A compiled graph as a node" | A compiled StateGraph used as a node in another graph; preserves its own state scope. |

## خواندن بیشتر

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) مرجع قانونی برای StateGraph، کاهش دهنده ها، چک پوائنترها و قطعات.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) مدل ذهنی که این درس از آن استفاده می کند، مستقیما از منبع.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) جزئیات در فروشگاه های Postgres/SQLite/Redis، مکان های نام پوائنٹ و شناسه های موضوع.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) `interrupt_before`،`interrupt_after`،`Command(resume=...)`، و الگوی حالت ویرایش
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) الگوی هر عامل LangGraph اجرا می شود؛ برای استدلال و استدلال دنباله ای آن را بخوانید.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) چه شکل های گرافیکی (سلسل، روتر، کارگران ورقساز، ارزیابی کننده- بهینه ساز) را ترجیح دهید و چه زمانی.
- مرحله 11 · 09 (دعوت عملکرد)  اولین صدا در هر گره عامل LangGraph دوباره استفاده می شود.
- مرحله 11 · 14 (پروتوکول مدل زمینه)  کشف ابزار خارجی که به یک LangGraph وصل می شود `ToolNode`از طریق آداپتور MCP
- مرحله 11 · 17 (توسط توافقات چارچوب عامل)  زمانی که LangGraph را بر روی CrewAI، AutoGen یا Agno انتخاب کنید.
