# Các nhóm đại lý dựa trên vai trò  Vai trò, nhiệm vụ, quy trình

> Bốn nguyên thủy: Agent, Task, Crew, Process. Hai hình dạng cấp cao: Crews (tự trị, hợp tác dựa trên vai trò) và Flow (tự động, xác định). CrewAI là thực hiện tham chiếu năm 2026, và tài liệu của nó là thẳng thắn: "cho bất kỳ ứng dụng sẵn sàng sản xuất nào, bắt đầu với Flow".

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## Mục tiêu học tập

- Tên bốn nguyên thủy của CrewAI (Đội ngũ, nhiệm vụ, phi hành đoàn, quy trình) và những gì mỗi người sở hữu.
- Sự phân biệt giữa trình tự, cấp bậc và quy trình đồng thuận được lên kế hoạch; chọn một trong mỗi khối lượng công việc.
- Sự khác biệt giữa các nhóm (tự trị dựa trên vai trò) và các dòng chảy (chỉ định dựa trên sự kiện) và giải thích khuyến nghị sản xuất của các docs.
- Các công cụ dây với `@tool`trang trí và `BaseTool`Subclass; lý luận về các kết quả có cấu trúc so với văn bản tự do.
- Hãy cho tên bốn loại bộ nhớ CrewAI và khi nào mỗi loại sẽ được trả giá.
- Thực hiện một đội ngũ 3 nhân viên (phác giả, nhà văn, biên tập viên) tạo ra một bản ngắn.
- Nhận ra ba chế độ thất bại của CrewAI: Thổi bốc, thuế quản lý-LLM, giao hàng dễ vỡ.

## Vấn đề

Các nhóm áp dụng các khung đa đại lý chạm vào cùng một bức tường. "Công tác tự động" nghe có vẻ tuyệt vời trong một bản demo. Sau đó một khách hàng gửi lỗi và bạn cần lặp lại xác định. hoặc tài chính hỏi một đội ngũ LLM-routed chi phí mỗi chạy. hoặc trên cuộc gọi cần biết đại lý nào đã dừng lại vào 3 giờ sáng.

Các nhóm chuyên môn tự do không trả lời một câu hỏi nào một cách sạch sẽ, nhưng những DAG hoàn toàn không có câu trả lời nào, nhưng mất đi hình dạng khám phá mà một nhân viên đầu óc cần.

Sự chia rẽ của CrewAI là trung thực về thương mại. Đội ngũ cho công việc hợp tác, dựa trên vai trò, khám phá. Xu hướng cho sự kiện, sở hữu mã, kiểm toán sản xuất. cùng một khung, hai hình dạng, chọn cho mỗi bề mặt.

## Khái niệm

### Bốn nguyên thủy

Màn hình của CrewAI nhỏ, hãy ghi nhớ lại và phần còn lại sẽ được cấu hình.

- **Agent.** `role + goal + backstory + tools + (optional) llm`- Hình ảnh hậu trường là chịu tải. Nó hình thành giọng nói, phán đoán, khi đại lý dừng lại.
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`Một đơn vị công việc có thể được sử dụng nhiều lần.`expected_output`là hợp đồng.`context`danh sách các nhiệm vụ trước dòng mà các kết quả được chuyển vào. `output_pydantic`tạo ra một hình dạng có cấu trúc.
- **Crew.**Container, sở hữu danh sách các`agents`, danh sách của `tasks`, `process`, và tùy chọn `memory`+ `verbose`+ `manager_llm`cài đặt.
- **Process.**Chiến lược thực hiện: theo trình tự, Hiérarchical, Consensus (được lên kế hoạch).

Các nhân viên không nhìn thấy nhau trực tiếp, nhiệm vụ của các nhân viên tham khảo, các nhân viên theo dõi nhiệm vụ, quá trình quyết định ai chọn nhiệm vụ tiếp theo.

> **Validated against**CrewAI 0.86 (2026-05). Các phiên bản mới hơn có thể đổi tên hoặc hợp nhất các loại quy trình; kiểm tra các [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)trước khi dựa vào một hình dạng cụ thể.

### Theo trình tự vs Đường bậc vs Thỏa thuận

- **Sequential.**Các nhiệm vụ chạy theo thứ tự tuyên bố.`context`N + 1 là một nhiệm vụ. chi phí thấp nhất.
- **Hierarchical.**Một quản lý đại lý (câu lạc LLM riêng biệt) các tuyến đường giữa các chuyên gia.`manager_llm`config hoặc mặc định. người quản lý chọn nhiệm vụ tiếp theo mỗi vòng và có thể từ chối hoặc định tuyến lại. sử dụng khi bạn có bốn chuyên gia hoặc nhiều hơn và đặt hàng thực sự phụ thuộc vào đầu ra trước đó.
- **Consensus.**Được lên kế hoạch, hiện chưa được triển khai trong API công cộng. Các tài liệu lưu tên cho một quy trình dựa trên bỏ phiếu trong tương lai. Đừng dựa vào nó hôm nay.

Các hệ thống phân cấp thêm một cuộc gọi LLM mỗi vòng (người quản lý) trên đầu mỗi cuộc gọi chuyên gia. Chi phí token có thể tăng gấp ba lần trong một lần chạy năm bước. Chỉ trả cho nó khi bạn cần đường dẫn.

### Đội ngũ vs dòng chảy

Đây là khung hình mà các bác sĩ dẫn đầu vào năm 2026.

- **Crew.**Phương pháp tự trị dựa trên LLM. Khung tâm chọn hình dạng khi chạy. Khả năng cho: nghiên cứu, suy nghĩ, bản thảo đầu tiên, bất cứ nơi nào con đường là một phần của câu trả lời. Khó để chơi lại. Khó để thử nghiệm. Murah để nguyên mẫu.
- **Flow.**Hình đồ dựa trên sự kiện mà bạn sở hữu.`@start`đánh dấu lối vào. `@listen(topic)`là một bước phát nổ khi một bước khác phát ra chủ đề đó. mỗi bước là Python đơn giản (có thể gọi một Crew nội bộ).

Các bác sĩ đề nghị sản xuất năm 2026: bắt đầu với một dòng chảy.`Crew.kickoff()`Flow cho bạn những con đường kiểm tra, Crew cho bạn những cuộc khám phá.

### Kết hợp công cụ

Ba cách để cho một nhân viên một công cụ.

1. **`@tool` decorator.**Các chức năng tinh khiết trở thành công cụ. Biên bản là sơ đồ; docstring là mô tả mà LLM thấy.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.**Công cụ dựa trên lớp với schema args rõ ràng, hỗ trợ async, thử lại. Sử dụng khi công cụ có trạng thái (một client, một cache) hoặc cần các args cấu trúc.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits.**CrewAI cung cấp các bộ chuyển đổi bên đầu tiên: `SerperDevTool`- `FileReadTool`- `DirectoryReadTool`- `CodeInterpreterTool`- `RagTool`- `WebsiteSearchTool`- Được cáp bằng một khẩu.

Các kết quả cấu trúc sử dụng Pydantic.`output_pydantic=MyModel`CrewAI xác nhận phản ứng LLM đối với mô hình và hoặc ép buộc hoặc thử lại.`expected_output`string. Các kết quả văn bản tự do là tốt cho bản thảo; kết quả cấu trúc là những gì dòng chảy hạ lưu có thể tiêu thụ.

### Cây móc nhớ

CrewAI đưa ra bốn loại bộ nhớ ra khỏi hộp.

> **Validated against**CrewAI 0.86 (2026-05). Các bản phát hành gần đây sẽ hướng mọi thứ qua một hệ thống thống nhất `Memory`mô hình khái niệm dưới đây vẫn còn tồn tại, nhưng bề mặt lớp công cộng có thể sụp đổ thành một`Memory`điểm nhập vào trong các phiên bản mới hơn; kiểm tra [CrewAI memory docs](https://docs.crewai.com/concepts/memory)cho API hiện tại.

- **Short-term.**Buffer trò chuyện trong một lần chạy.
- **Long-term.**Cung cấp trong các lần chạy. Cung cấp trong một vector DB (Chroma mặc định, có thể thay đổi).
- **Entity.**"Điều kiện của khách hàng X là trong kế hoạch doanh nghiệp". Định nghĩa bởi tổ chức, không phải bởi sự tương đồng.
- **Contextual.**Lấy lại thời gian tập hợp, lấy bộ nhớ liên quan vào thời điểm mà đại lý cần nó, không phải tải trước.

Khả năng trên phi hành đoàn với `memory=True`Memory là một trong những nơi mà CrewAI kiếm được sự giữ gìn của mình đối với các khung mỏng hơn; LangGraph tinh khiết đòi hỏi bạn phải tự dây mỗi trong số này.

### Khi các nhóm dựa trên vai trò phù hợp

- 3 đến 6 nhân viên với vai trò được đặt tên và một quy trình làm việc hợp tác.
- Đường dẫn nơi phán quyết của LLM về bước tiếp theo là một phần của giá trị (Tầm bậc).
- Bất cứ nơi nào mà đội ngũ đều vui hơn khi đọc sách.`role + goal + backstory`hơn là đọc một định nghĩa biểu đồ.

### Khi họ không

- Các DAG xác định với thứ tự nghiêm ngặt. Sử dụng LangGraph (Dạy học 13).
- Các ngân sách chậm thứ hai. Hiérarchical thêm đi lại và đi lại. Ngay cả Sequential cũng liên tục liên tục các yêu cầu bao gồm các câu chuyện sau và các kết quả trước đó.
- Các vòng lặp đơn đại lý. Trượt khung; một vòng lặp đại lý (Dạy 1) cộng với một sổ đăng ký công cụ ngắn hơn.

Bài học 17 (Tradeoffs của cơ quan cơ quan) đưa ra điều này trong một matrix.

### Hình dạng phụ thuộc

Không phụ thuộc vào LangChain. Python 3.10 đến 3.13. sử dụng `uv`Nhìn xem.[crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)(Snapshot từ năm 2026-05). Sự tích hợp AWS Bedrock được ghi nhận; các điểm chuẩn của nhà cung cấp báo cáo về tốc độ tăng đáng kể so với LangGraph trên khối lượng công việc QA, nhưng phương pháp (dữ liệu, phần cứng, métrics đánh giá) không được công bố, vì vậy hãy coi số nhà cung cấp khung như chỉ hướng.

### Khi mô hình này đi sai

- **Prompt-bloat from backstories.**Một câu chuyện hậu cảnh 2000 từ cho mỗi đại lý và một đội ngũ 5 đại lý đốt cháy ngân sách ngữ cảnh trước khi gọi công cụ đầu tiên. Giữ câu chuyện hậu thuẫn dưới 200 từ. Sử dụng lại các cụm từ trên các đại lý; không lặp lại phong cách nhà năm lần.
- **Manager-LLM token tax.**Quá trình phân cấp thêm một cuộc gọi LLM quản lý trước mỗi cuộc gọi chuyên gia. Trong một đội ngũ năm nhiệm vụ là sáu cuộc gọi LLM thay vì năm, và cuộc gọi quản lý mang theo danh sách nhiệm vụ đầy đủ cộng với các đầu ra trước đó.
- **Brittle handoffs.**Nhiệm vụ N.`expected_output`là "một phác thảo". Nhiệm vụ N+1 đọc nó như `context`Và cố gắng phân tích ba phần. LLM đã tạo ra bốn.`output_pydantic`trên Task N do đó Task N+1 đọc một đối tượng được gõ, không phải văn bản tự do.
- **Crew-as-prod.**Đơn vị tự do Crew được vận chuyển đến sản xuất mà không có một vòng vòng vòng dòng.

```figure
ae-crew-vs-flow
```

## Hãy xây dựng nó

`code/main.py`thực hiện các phiên bản STDlib của cả hai hình dạng cộng với một phi hành đoàn 3 nhân viên.

Hình dạng:

- `Agent`- `Task`Các lớp dữ liệu phù hợp với bề mặt của CrewAI.
- `SequentialCrew.kickoff(inputs)`chạy các nhiệm vụ theo thứ tự tuyên bố, threading các kết quả như `context`- Tôi không biết.
- `HierarchicalCrew.kickoff(topic)`thêm một quản lý đại lý chọn chuyên gia tiếp theo mỗi vòng, dừng lại ở "đã làm".
- `Flow`với `@start`và `@listen(topic)`những người trang trí, một vòng lặp nhỏ, và một dấu vết.
- `tool(name)`Bộ trang trí phản chiếu CrewAI `@tool`hình dạng.
- `Memory`với `short_term`- `long_term`- `entity`cửa hàng; sự tương đồng đùa sử dụng numpy.
- Phản ứng LLM giả là chuỗi mã hóa cứng được khóa khỏi vai trò cộng với tiền tố nhập. Không mạng.

Demo cụ thể: nhà nghiên cứu, nhà văn, nhóm biên tập viên sản xuất một bản tóm tắt về "kỹ thuật đại lý 2026". Nhà nghiên cứu kéo (cười) các nguồn. Nhà văn bản thảo. biên tập viên chặt chẽ. cùng một nhóm chạy qua một dòng chảy để hiển thị hình dạng xác định.

Đi đi.

```bash
python3 code/main.py
```

Các dấu vết: các dòng băng thông của thủy thủ đoàn`context`, nhóm cấp bậc với các lựa chọn quản lý (phác giả, nhà văn, biên tập viên, sau đó "được thực hiện"), dòng chảy chạy cùng ba bước với các chủ đề rõ ràng (`researched`- `drafted`- `edited`), các cuộc gọi công cụ được chuyển qua `@tool`, và trí nhớ lâu dài sống sót sau hai cú đá.

Hình ảnh của phi hành đoàn là chất lỏng, người quản lý có thể đặt lại nguyên tắc.

## Sử dụng nó

- **CrewAI Flow**cho sản xuất. ngay cả khi dòng chảy là một bước đòi hỏi`Crew.kickoff()`- Tầm chảy cho phép kiểm toán.
- **CrewAI Crew (Sequential)**cho việc hợp tác sắp xếp rõ ràng, đặc biệt là các bản thảo đầu tiên và vòng xem xét.
- **CrewAI Crew (Hierarchical)**khi định tuyến phụ thuộc vào đầu ra và bạn có bốn chuyên gia hoặc nhiều hơn.
- **LangGraph**(Dân học 13) cho máy trạng thái rõ ràng, hồ sơ lâu dài, sắp xếp nghiêm ngặt.
- **AutoGen v0.4**(Dân học 14) cho đồng thời diễn viên-chương model và sự cô lập lỗi.
- **OpenAI Agents SDK**(Dân học 16) cho các sản phẩm đầu tiên của OpenAI với bàn tay và hàng rào.
- **Claude Agent SDK**(Dân học 17) cho các sản phẩm Claude-first với các bộ phận phụ và cửa hàng phiên.

## Chuyển nó

`outputs/skill-crew-or-flow.md`chọn Crew vs Flow cho một nhiệm vụ và chuẩn bị cho việc thực hiện tối thiểu. Hard từ chối về Crew-without-backstory, Flow-without-explicit-topics, Hierarchical với dưới ba chuyên gia.

## Những bẫy

- **Backstory as flavor.**Nó tạo ra các kết quả, thử 3 biến thể cho mỗi đại lý, biến thể là thực.
- **Skipping `expected_output`.**Không có hợp đồng cho mỗi nhiệm vụ, các nhiệm vụ tiếp theo sẽ thu được bất cứ điều gì mà LLM đã sản xuất.
- **Memory always-on.**Long term viết mỗi run vector DB tăng lên lấy lại trở nên ồn ào phạm vi viết cho các nhiệm vụ nơi thực tế là duy trì
- **Manager prompt drift.**Nếu đường dẫn trở nên kỳ lạ, hãy bỏ nó vào chế độ từ ngữ và đọc.
- **Tool side effects in Crews.**Một phi hành đoàn có thể gọi công cụ nhiều lần hơn dự kiến.

## Các bài tập

1. Chuyển đổi bộ phận của bộ máy theo dõi thành dòng chảy, đếm các điểm liên lạc khi biến động giảm, ghi lại khi độ đọc giảm.
2. Thêm bộ nhớ thực thể vào đội ngũ: các thông tin về khách hàng tồn tại qua các kickoff.
3. Thực hiện một quy trình Hiérarchical trong đó người quản lý từ chối định tuyến đến biên tập viên cho đến khi xuất của nhà văn có ít nhất ba đoạn.
4. Đưa dây a `BaseTool`Subclass cho một tìm kiếm web (phản chế). So sánh hình dạng dấu vết so với `@tool`phiên bản trang trí.
5. Thêm `output_pydantic=Brief`cho nhiệm vụ biên tập viên, nơi `Brief`- Có `title`- `summary`- `sections`. Làm cho tác vụ nhà văn xuất phát JSON bị trục trặc một lần; xác minh hành vi thử lại của CrewAI trong theo dõi.
6. Đọc phần giới thiệu về tài liệu của CrewAI.`crewai`API, phiên bản STDlib bỏ qua những đảm bảo nào?
7. Đưa AgentOps hoặc Langfuse (Dạy 24) vào một cuộc chạy thực sự.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "Unit of work" | Description + expected output + assignee + optional structured output |
| Crew | "Agent team" | Container for Agents + Tasks + Process |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensus (planned) |
| Flow | "Deterministic workflow" | Event-driven, code-owned, testable |
| Backstory | "Persona prompt" | Tone and judgment shaper for the Agent |
| `@tool` | "Function tool" | Decorator that turns a function into a tool the Agent can call |
| `BaseTool` | "Class tool" | Class-based tool with args schema, retries, async support |
| Entity memory | "Per-entity facts" | Memory scoped to a customer / account / issue |
| Long-term memory | "Cross-run memory" | Vector-backed memory that survives between kickoffs |
| Contextual memory | "Just-in-time retrieval" | Memory pulled at the moment the Agent needs it |
| Manager LLM | "Router agent" | Extra LLM in Hierarchical process that picks the next task |
| `expected_output` | "Task contract" | String that tells the Agent (and audit) what shape to return |

## Đọc thêm

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): khái niệm và con đường sản xuất được khuyến cáo
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): hình dạng dựa trên sự kiện,`@start`- `@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)`@tool`- `BaseTool`, bộ dụng cụ tích hợp
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): ngắn hạn, dài hạn, đơn vị, ngữ cảnh
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): khi multi-agent giúp đỡ và khi nó không
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): sự thay thế của máy nhà nước
