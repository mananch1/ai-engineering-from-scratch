# Máy cơ nhà nước đại lý  Hình đồ, nút, Bàn kiểm soát

> Một vòng ReAct được viết bằng tay là một `while True`. cùng vòng lặp được viết như một biểu đồ rõ ràng là một cái gì đó bạn có thể kiểm soát, gián đoạn, nhánh, và đi qua thời gian.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## Vấn đề

Bạn gửi một đại lý gọi chức năng. Nó hoạt động trong ba lượt, sau đó có một cái gì đó sai: mô hình thử một công cụ trả lại 500, người dùng thay đổi ý kiến của họ giữa nhiệm vụ, hoặc đại lý quyết định hoàn lại một đơn đặt hàng mà không có một người ký.`while True:`bạn không thể tạm dừng nó, bạn không thể xoay lại nó, và bạn không thể phân nhánh ra "vậy nếu mô hình đã chọn công cụ khác".

Bước tiếp theo là rõ ràng khi bạn thấy nó. Trưởng đã là một máy trạng thái  hệ thống nhanh chóng cộng với lịch sử tin nhắn cộng với chờ đợi công cụ gọi cộng với hành động tiếp theo. Làm cho máy trạng thái rõ ràng: nút cho "chương trình nghĩ", "một công cụ chạy", "một con người chấp thuận", và cạnh cho các chuyển đổi điều kiện giữa chúng. Một khi biểu đồ rõ ràng, vòng xoáy nhận được bốn thứ miễn phí: kiểm tra (giữ trạng thái giữa các bước), gián đoạn (phơi lỏng cho con người), phát trực tuyến (điều kiện dòng chảy và các sự kiện trung gian), và du lịch thời gian (lúc lại trạng thái trước và thử một chi nhánh khác).

Việc thực hiện tham chiếu của trừu tượng này là LangGraph. Nó không phải là một khung đại lý theo nghĩa LangChain ("đây là một AgentExecutor, may mắn"). Nó là một thời gian chạy đồ thị với trạng thái hạng nhất, sự kiên trì hạng nhất và sự gián đoạn hạng nhất.

## Khái niệm

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

A `StateGraph`có ba điều.

1. **State.**Một dict được gõ (TypedDict hoặc mô hình Pydantic) chảy qua biểu đồ. Mỗi nút nhận được trạng thái đầy đủ và trả lại một bản cập nhật một phần, mà LangGraph hợp nhất bằng cách sử dụng một * giảm * cho mỗi trường `operator.add`cho các danh sách nên tích lũy, ghi lại theo mặc định.
2. **Nodes.**Phụng chức năng Python `state -> partial_state`Mỗi bước là một bước riêng biệt: " gọi mô hình, " " chạy công cụ, " " tóm tắt. "
3. **Edges.**Chuyển đổi giữa các nút. Biên tĩnh đi một chỗ. Biên điều kiện có chức năng router`state -> next_node_name`Vì vậy, biểu đồ có thể phân ngành trên đầu ra mô hình.

Bạn biên soạn biểu đồ. biên soạn liên kết topology, gắn một điểm kiểm tra (tự chọn nhưng cần thiết cho sản xuất), và trả lại một runnable. Bạn gọi nó với một trạng thái ban đầu và một `thread_id`Mỗi bước hành quyết đều là một điểm kiểm soát được khóa vào`(thread_id, checkpoint_id)`- Tôi không biết.

### Bốn siêu cường

**Checkpointing.**Mỗi chuyển đổi nút viết trạng thái mới vào một cửa hàng (trong bộ nhớ cho các thử nghiệm, Postgres / Redis / SQLite cho prod).`thread_id`Chữ đồ họa bắt đầu từ nơi nó dừng lại.

**Interrupts.**Đánh dấu một nút với `interrupt_before=["human_review"]`và thực thi dừng lại trước khi nút đó chạy. trạng thái vẫn tồn tại. API của bạn trả lời người dùng với "đợi phê duyệt". Một yêu cầu sau đó cho cùng `thread_id`với `Command(resume=...)`bắt đầu hành quyết.

**Streaming.** `graph.stream(state, mode="updates")`đưa ra các vùng Delta như chúng xảy ra. `mode="messages"`truyền các token LLM bên trong các nút mô hình. `mode="values"`bạn chọn những gì sẽ xuất hiện trong UI của bạn.

**Time-travel.** `graph.get_state_history(thread_id)`trả lại toàn bộ nhật ký kiểm soát.`checkpoint_id`đến`graph.invoke`Và bạn đi từ đó. Rất tốt cho debugging ("như thế nào nếu mô hình đã chọn công cụ B thay vào đó?") và cho các thử nghiệm hồi quy để chơi lại các dấu vết sản xuất.

### Các giảm là điểm

Mỗi trường trạng thái có một reducer. Hầu hết các mặc định đều ổn  một giá trị mới ghi lại giá trị cũ. Nhưng danh sách tin nhắn cần `operator.add`để các tin nhắn mới được thêm vào thay vì thay thế. Biên song song kết kết hợp cập nhật của họ thông qua giảm. Nếu hai nút cả hai cập nhật `messages`Và anh quên `Annotated[list, add_messages]`, thứ hai thắng im lặng và bạn mất một nửa lượt.

### Chữ đồ thị ReAct trong bốn nút

Một đại lý ReAct sản xuất là bốn nút và hai cạnh:

1. `agent` gọi LLM với lịch sử tin nhắn hiện tại. Trả lại tin nhắn trợ lý (có thể chứa tool_calls).
2. `tools` thực hiện bất kỳ tool_call nào trong thông điệp trợ lý cuối cùng, thêm kết quả công cụ như thông điệp công cụ.
3. Một cạnh điều kiện từ `agent`đường này đến `tools`nếu tin nhắn cuối cùng có tool_calls, nếu không thì `END`- Tôi không biết.
4. Một cạnh tĩnh từ `tools`quay lại `agent`- Tôi không biết.

Bạn có được vòng lặp ReAct đầy đủ (Think → Action → Observation → Thought → ...) với kiểm soát, gián đoạn và phát trực tuyến, trong khoảng 40 dòng mã.

### StateGraph vs Send (phân tích)

`Send(node_name, state)`cho phép một nút gửi các tiểu hình song song. ví dụ: đại lý quyết định truy vấn ba máy tìm kiếm cùng một lúc. Mỗi `Send`tạo ra một thực hiện song song của nút mục tiêu; các sản phẩm của chúng hợp nhất thông qua nhà giảm trạng thái. Đây là cách LangGraph thể hiện mô hình nhạc công-người làm việc mà không cần threading nguyên thủy.

### Các phụ đề

Một biểu đồ được biên soạn có thể là một nút trong biểu đồ khác. biểu đồ bên ngoài nhìn thấy một nút duy nhất; biểu đồ bên trong có trạng thái riêng của nó và các điểm kiểm soát riêng của nó. Đây là cách các nhóm xây dựng các đại lý người giám sát: biểu đồ giám sát định hướng ý định của người dùng đến một tiểu biểu đồ người lao động trên mỗi miền.

```figure
l5-state-graph-ledger
```

## Hãy xây dựng nó

### Bước 1: trạng thái và nút

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

`add_messages`là bộ giảm tính làm cho danh sách tin nhắn tích lũy thay vì ghi lại.

### Bước 2: chạy với một sợi dây

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Mỗi bản cập nhật đều là một lời khuyên.`{node_name: state_delta}`Frontend của bạn có thể phát trực tuyến này đến UI để người dùng thấy "hợp tác viên đang nghĩ... gọi search_web... có kết quả... trả lời".

### Bước 3: thêm một người trong vòng cắt đứt

Đánh dấu một nút để thực thi dừng lại trước khi nó chạy.

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

Tình trạng, điểm kiểm soát và dây liên tục tồn tại trong suốt thời gian gián đoạn.

### Bước 4: thời gian đi du lịch để debugging

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Đi qua `None`khi đầu vào lặp lại từ điểm kiểm soát được cung cấp; vượt qua một giá trị thêm nó như một cập nhật cho trạng thái của điểm kiểm soát đó trước khi tiếp tục. Đây là cách bạn tái tạo một hành động xấu chạy mà không chạy lại toàn bộ cuộc trò chuyện.

### Bước 5: Thay đổi điểm kiểm soát cho sản xuất

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis, và Postgres đã được vận chuyển. `MemorySaver`Bất cứ thứ gì tồn tại sau khi khởi động lại đều cần một cửa hàng thực sự.

## Khả năng

> Bạn xây dựng các đại lý như đồ thị, không như `while True`- Các vòng lặp.

Trước khi bạn tìm thấy LangGraph, hãy thiết kế 60 giây:

1. **Name the nodes.**Mỗi quyết định riêng biệt hoặc hành động tác động phụ là một nút. "Điều đại lý nghĩ", "công cụ chạy", "đánh giá chấp thuận", "thường xuyên phản hồi". Nếu bạn không thể liệt kê chúng, nhiệm vụ vẫn chưa có hình dạng đại lý.
2. **Declare the state.**Tối thiểu TypedDict với một giảm cho mỗi trường danh sách. Đừng nhồi tất cả mọi thứ vào `messages`; nâng các lĩnh vực cụ thể về nhiệm vụ (một công việc)`plan`, một `budget`đếm, một `retrieved_docs`(Dân trí) đến cấp cao nhất.
3. **Draw the edges.**Static trừ khi bước tiếp theo phụ thuộc vào đầu ra mô hình. Mỗi cạnh điều kiện cần một chức năng router với các nhánh được đặt tên.
4. **Choose a checkpointer up front.** `MemorySaver`cho các thử nghiệm, Postgres/Redis/SQLite cho bất cứ điều gì khác. Không vận chuyển mà không có một  không kiểm tra chỉ có nghĩa là không có hồ sơ, không gián đoạn, không có thời gian đi du lịch.
5. **Decide interrupts before tools run, not after.**Sự chấp thuận đi vào cạnh vào một nút tác dụng phụ để bạn có thể hủy trước khi gây hại; xác thực đi vào cạnh ra khỏi mô hình để bạn có thể từ chối cuộc gọi xấu rẻ.
6. **Stream by default.** `mode="updates"`cho UI, `mode="messages"`cho các dòng chảy cấp token bên trong các nút mô hình, `mode="values"`cho các bức ảnh chụp toàn bộ trong quá trình đánh giá.

Không gửi một đại lý LangGraph mà không có điểm kiểm soát. Không gửi một đại lý mà gián đoạn sau tác dụng phụ. Không gửi một`messages`trường không có `add_messages`như là chất giảm.

## Các bài tập

1. **Easy.**Thực hiện biểu đồ ReAct bốn nút trên với một công cụ máy tính và một công cụ tìm kiếm web.`list(app.get_state_history(config))`trả lại ít nhất bốn điểm kiểm soát cho một cuộc trò chuyện hai vòng.
2. **Medium.**Thêm một `planner`nút chạy trước `agent`và viết một cấu trúc`plan: list[str]`- Tôi đã đi vào tiểu bang.`agent`Đánh dấu các bước kế hoạch như đã thực hiện.`plan`bị mất qua một hồ sơ kiểm soát (trầm giảm).
3. **Hard.**Xây dựng một biểu đồ giám sát mà đường dẫn giữa ba biểu đồ phụ (`researcher`- `writer`- `reviewer`) sử dụng `Send`Mỗi phụ lục có trạng thái và điểm kiểm soát riêng của nó.`interrupt_before=["writer"]`trên biểu đồ bên ngoài để một con người có thể chấp thuận nghiên cứu ngắn. xác nhận rằng thời gian đi từ một điểm kiểm soát trước đó chạy lại chỉ là chi nhánh.

## Các điều khoản chính

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

## Đọc thêm

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) tham chiếu kinh điển cho StateGraph, giảm, kiểm tra và gián đoạn.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) mô hình tâm lý bài học này sử dụng, trực tiếp từ nguồn.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) chi tiết về các cửa hàng Postgres/SQLite/Redis, không gian tên điểm kiểm soát và ID chuỗi.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) `interrupt_before`- `interrupt_after`- `Command(resume=...)`, và các edit-state pattern.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) mô hình mà mỗi đại lý LangGraph thực hiện; đọc nó để lý luận lý luận.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) hình dạng biểu đồ nào ( chuỗi, bộ định tuyến, nhạc công, đánh giá- tối ưu hóa) để thích và khi nào.
- Giai đoạn 11 · 09 (Tạm dịch gọi)  các công cụ gọi nguyên thủy mỗi nút đại lý LangGraph sử dụng lại.
- Giai đoạn 11 · 14 (Mô hình Công thức ngữ cảnh)  phát hiện công cụ bên ngoài kết nối vào một LangGraph `ToolNode`qua bộ chuyển đổi MCP.
- Giai đoạn 11 · 17 (Tương đương với cơ sở quản lý của các đại lý)  khi nào chọn LangGraph thay vì CrewAI, AutoGen hoặc Agno.
