# Stateful Graph Orchestration  Thực hiện lâu dài và các điểm kiểm soát

> Agent là một máy trạng thái; nút là chức năng; cạnh là chuyển đổi; trạng thái được đặt điểm kiểm soát sau mỗi nút. Bắt đầu từ bất kỳ thất bại nào tại điểm kiểm soát thành công cuối cùng. LangGraph là tham chiếu 2026 cho mô hình này của dàn xếp trạng thái cấp thấp.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Mục tiêu học tập

- Mô tả mô hình cốt lõi của LangGraph: máy trạng thái với trạng thái đánh dấu, nút chức năng, cạnh điều kiện và các điểm kiểm tra sau nút.
- Hãy nêu tên bốn khả năng mà các tài liệu nhấn mạnh: thực hiện bền, phát trực tuyến, người trong vòng lặp, bộ nhớ toàn diện.
- Giải thích ba topology dàn xếp LangGraph hỗ trợ: giám sát, peer-to-peer (swarm), bậc phân cấp (năm subgraph).
- Thực hiện biểu đồ trạng thái stdlib với trạng thái nhập, cạnh có điều kiện và chu kỳ kiểm soát/thường hồi.

## Vấn đề

Các đại lý và dòng công việc chia sẻ một vấn đề: khi một lần chạy 40 bước thất bại ở bước 38, bạn muốn tiếp tục từ bước 38, chứ không phải bắt đầu lại. Các mô hình trạng thái hạng hai để cho các nhà điều hành hack lại các thử nghiệm xung quanh thư viện giả định chạy mới.

Câu trả lời thiết kế của LangGraph: trạng thái là một đối tượng đánh chữ hạng nhất, đột biến là rõ ràng, và các điểm kiểm soát tồn tại sau mỗi nút.`load_state(session_id)`gọi.

## Khái niệm

### Hình đồ

Một biểu đồ được định nghĩa bởi:

- **State type.**Một lệnh đánh dấu (hoặc mô hình Pydantic) mà mỗi nút đọc và đột biến.
- **Nodes.**Các chức năng tinh khiết`(state) -> state_update`Các bản cập nhật được hợp nhất thành trạng thái sau khi trở về.
- **Edges.**Chuyển đổi điều kiện hoặc trực tiếp giữa các nút.
- **Entry and exit.** `START`và `END`Các nút Sentinel đánh dấu ranh giới.

Ví dụ: một đại lý với `classify`- `refund`- `bug`- `sales`- `done`nút  một dòng công việc định tuyến như một biểu đồ.

### Thực hiện lâu dài

Sau khi mỗi nút trở lại, runtime sẽ xoay quanh trạng thái và viết nó vào một điểm kiểm tra (SQLite, Postgres, Redis, tùy chỉnh).`resume(session_id)`và bắt đầu từ bước N + 1 với trạng thái chính xác.

Các tài liệu LangGraph rõ ràng nhấn mạnh người dùng sản xuất nơi đây quan trọng: Klarna, Uber, JP Morgan.

### Chuyển phát

Mỗi nút có thể tạo ra một phần đầu ra. biểu đồ truyền các sự kiện delta mỗi nút đến người gọi để UI cập nhật khi biểu đồ chạy.

### Người trong vòng lặp

Kiểm tra và sửa đổi trạng thái giữa các nút. Thực hiện: tạm dừng trước một nút quan trọng, trạng thái bề mặt cho một con người, chấp nhận sửa đổi, tiếp tục. Điểm kiểm tra làm cho điều này dễ dàng bởi vì trạng thái đã được phân phối theo chuỗi.

### Tưởng thức

Giao diện LangGraph tích hợp với các hệ thống bộ nhớ bên ngoài (Mem0, tùy chỉnh) thông qua các công cụ.

### Ba topology

1. **Supervisor.**Các bộ định tuyến trung tâm LLM gửi cho các chuyên gia phụ. `create_supervisor()`trong `langgraph-supervisor`(mặc dù nhóm LangChain vào năm 2026 khuyên nên làm điều này thông qua các công cụ gọi trực tiếp cho sự kiểm soát bối cảnh nhiều hơn).
2. **Swarm / peer-to-peer.**Các nhân viên giao thông trực tiếp qua một công cụ chung không có bộ định tuyến trung tâm.
3. **Hierarchical.**Giám sát viên quản lý các giám sát viên phụ, được thực hiện như các hình phụ tổ.

### Khi mô hình này đi sai

- **Checkpoints too small.**Chỉ cần kiểm tra các vòng quay cuộc trò chuyện để lại trạng thái công cụ và bộ nhớ viết không thể phục hồi.
- **Non-deterministic nodes.**Thử nghiệm lại giả định đầu vào nút tạo ra cùng một cập nhật trạng thái. hạt giống ngẫu nhiên, đồng hồ tường, API bên ngoài phải được chụp.
- **Over-use of conditional edges.**Một biểu đồ với mọi cạnh điều kiện là một máy trạng thái mà không thể lý luận về.

```figure
langgraph-state
```

## Hãy xây dựng nó

`code/main.py`thực hiện biểu đồ trạng thái stdlib:

- `State` một lệnh đánh dấu với `messages`- `step`- `route`- `output`- `human_approval`- Tôi không biết.
- `Node` được gọi là nhận trạng thái và trả lại một bản cập nhật lệnh.
- `StateGraph` nút + cạnh + cạnh điều kiện + chạy + tiếp tục.
- `SQLiteCheckpointer`(trong bộ nhớ giả)  tựa trạng thái sau mỗi nút; `load(session_id)`phục hồi.
- Một biểu đồ demo: phân loại -> chi nhánh(chuyển / lỗi / bán hàng) -> cổng con người -> gửi.

Đi đi.

```
python3 code/main.py
```

Hình ảnh cho thấy lần chạy đầu tiên thất bại ở cổng của con người, kiên trì, sau đó tiếp tục sản xuất sản xuất cuối cùng.

## Sử dụng nó

- **LangGraph** tài liệu tham chiếu, sẵn sàng sản xuất.`create_react_agent`- `create_supervisor`, hoặc xây dựng biểu đồ của riêng bạn.
- **AutoGen v0.4**(Dạy 14)  mô hình diễn viên thay thế cho các kịch bản cạnh tranh cao.
- **Claude Agent SDK**(Dạy 17)  quản lý vòng xoáy với cửa hàng phiên tích hợp.
- **Custom** khi bạn cần kiểm soát chính xác hình dạng trạng thái hoặc backend điểm kiểm tra.

## Chuyển nó

`outputs/skill-state-graph.md`tạo ra một biểu đồ trạng thái hình LangGraph trong bất kỳ thời gian chạy mục tiêu nào với kiểm tra và tiếp tục được kết nối.

## Các bài tập

1. Thêm một cạnh điều kiện từ `classify`đến`end`khi sự tin tưởng phân loại dưới ngưỡng.`route`tay.
2. Thay đổi các SQLite giống giả với một SQLite thực sự kiểm tra điểm.
3. Thực hiện cạnh song song: hai nút chạy đồng thời, hợp nhất bằng một bộ giảm tùy chỉnh.
4. Đọc `langgraph-supervisor`Đưa đồ chơi vào `create_supervisor`So sánh hình dạng dấu vết.
5. Thêm streaming: mỗi nút sẽ có trạng thái một phần trong khi nó chạy. Bác in các delta khi chúng đến.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | Serializes state after every node; enables resume |
| Reducer | "State merger" | Function that combines current state with a node's update |
| Conditional edge | "Branch" | Edge chosen by a function of state |
| Subgraph | "Nested graph" | A graph used as a node inside another graph |
| Durable execution | "Resume from failure" | Restart at the last successful node with exact state |
| Supervisor | "Router LLM" | Central dispatcher for specialist subagents |
| Swarm | "P2P agents" | Agents hand off via shared tools; no central router |

## Đọc thêm

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) các tài liệu tham chiếu
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) API mô hình giám sát
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) Đổi thay cho người diễn viên-chương mẫu
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) cửa hàng phiên và các bộ phận phụ
