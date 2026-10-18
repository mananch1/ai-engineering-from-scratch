# Chương diện công cụ  Tại sao các đại lý cần có cấu trúc I/O

> Một mô hình ngôn ngữ tạo ra các token. Một chương trình thực hiện các hành động. Khoảng cách giữa hai đó là giao diện công cụ: một hợp đồng cho phép mô hình yêu cầu một hành động và chủ nhà thực hiện nó. Mỗi 2026 xếp hàng  hàm gọi vào OpenAI, Anthropic và Gemini; MCP `tools/call`; Các phần nhiệm vụ của A2A là một mã hóa khác của cùng một vòng lặp bốn bước. Bài học này đặt tên vòng lặp và cho thấy cơ chế tối thiểu để chạy nó.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## Mục tiêu học tập

- Giải thích tại sao một LLM chỉ có thể tạo ra văn bản không thể, một mình, thực hiện hành động chống lại thế giới thực.
- Hình vẽ vòng gọi công cụ bốn bước (xác định → quyết định → thực hiện → quan sát) và đặt tên ai là chủ sở hữu của mỗi bước.
- Viết mô tả công cụ như ba phần: tên, đầu vào JSON Schema và hàm thực thi xác định.
- Sự khác biệt giữa các công cụ tinh khiết và tác dụng phụ và nêu rõ tại sao việc chia cắt quan trọng đối với an toàn.

## Vấn đề

Một LLM phát ra phân phối xác suất trên token tiếp theo. Đó là toàn bộ bề mặt đầu ra. Nếu bạn hỏi một mô hình trò chuyện "giờ này thời tiết ở Bengaluru là gì", nó có thể viết một câu có thể tin được, nhưng nó không thể quay vào API thời tiết. Câu đó có thể đúng bởi tình cờ hoặc ba ngày bị lỗi.

Đó là mục đích của giao diện công cụ. Chương trình chủ  thời gian chạy của đại lý của bạn, Claude Desktop, ChatGPT, Cursor, hoặc kịch bản tùy chỉnh  quảng cáo danh sách các công cụ có thể gọi cho mô hình. Mô hình, khi nó quyết định một hành động là cần thiết, phát ra một tải trọng hữu ích được cấu trúc đặt tên cho một công cụ và các lập luận của nó. Người chủ phân tích tải trọng, chạy công cụ thực sự, và cung cấp kết quả trở lại. Loop tiếp tục cho đến khi mô hình quyết định không cần thêm cuộc gọi nữa.

Phiên bản đầu tiên của hợp đồng này được xuất bản vào tháng 6 năm 2023 như là tham số "các chức năng" của OpenAI.`tool_use`Các khối trong Claude 2.1.`functionDeclarations`Một vài tháng sau. Mỗi nhà cung cấp hiện đang tiết lộ cùng một hình dạng: một danh sách công cụ kiểu JSON-Schema, một công cụ tải trọng trả phí JSON. Mô hình giao thức ngữ cảnh (Triều 11 năm 2024) đã tổng quát hợp đồng vì vậy một danh sách công cụ phục vụ mỗi mô hình. A2A (Triều 4, 2026, v1.0) đã layered nguyên thủy tương tự cho đại lý-to-agent đại diện.

Các vòng lặp bốn bước là sự bất biến bên dưới tất cả những điều này.

## Khái niệm

### Bước 1: mô tả

Người chủ tuyên bố mỗi công cụ với ba trường.

- **Name.**Một thẻ nhận dạng ổn định, có thể đọc được bằng máy.`get_weather`Không phải "cái thời tiết".
- **Description.**Một đoạn ngắn ngôn ngữ tự nhiên. "Điều sử dụng khi người dùng hỏi về điều kiện hiện tại cho một thành phố cụ thể.
- **Input schema.**Một đối tượng JSON Schema (mở 2020-12) mô tả các lập luận của công cụ.

Mô hình nhận được danh sách. Các nhà cung cấp hiện đại liên tục các tuyên bố này vào hệ thống nhắc sử dụng một mẫu cụ thể cho nhà cung cấp, vì vậy bạn như người gọi chỉ xử lý các hình thức có cấu trúc.

### Bước 2: quyết định

Với thông điệp của người dùng và các công cụ có sẵn, mô hình chọn một trong ba hành vi.

1. **Answer directly**Không có lời gọi công cụ.
2. **Call one or more tools.**Phát ra các đối tượng gọi có cấu trúc.`parallel_tool_calls: true`(tầm định trên OpenAI và Gemini, chọn vào Anthropic) mô hình có thể phát ra nhiều cuộc gọi trong một lượt.
3. **Refuse.**Các kết quả kết cấu trong chế độ nghiêm ngặt có thể tạo ra một kiểu `refusal`chặn thay vì gọi điện.

Một công cụ gọi tải hữu ích có ba trường ổn định: một cuộc gọi `id`, một công cụ `name`, và một JSON `arguments`ID tồn tại để máy chủ có thể liên quan kết quả sau đó với cuộc gọi cụ thể, điều quan trọng khi các cuộc gọi song song trở lại không phù hợp.

### Bước 3: Thực hiện

Người chủ nhận được cuộc gọi, xác nhận các lập luận chống lại sơ đồ được tuyên bố và chạy trình thực hiện. Các lập luận không hợp lệ có nghĩa là mô hình ảo giác một trường hoặc sử dụng loại sai  một chế độ thất bại rất phổ biến trên các mô hình yếu. Các máy chủ sản xuất làm một trong ba điều trên các lập luận không hợp lệ: thất bại nhanh chóng và làm bề mặt lỗi cho mô hình, sửa chữa JSON bằng một trình phân tích bị hạn chế hoặc thử lại mô hình với lỗi xác thực được bao gồm trong lời nhắc.

Bản thân trình thực trình là mã thông thường. Python, TypeScript, lệnh shell, truy vấn cơ sở dữ liệu. Nó tạo ra kết quả, thường là một chuỗi nhưng có thể là bất kỳ giá trị JSON hoặc khối nội dung được cấu trúc (tin nhắn văn bản, hình ảnh hoặc tham chiếu tài nguyên trong MCP). Kết quả phải được tựa.

### Bước 4: quan sát

Người chủ kết nối kết quả công cụ vào cuộc trò chuyện (như một `tool`thông điệp vai trò với sự phù hợp `id`(văn khoái) và triệu tập lại mô hình. mô hình bây giờ có công cụ đầu ra trong bối cảnh và có thể tạo ra một câu trả lời cuối cùng hoặc yêu cầu nhiều cuộc gọi hơn.

### Sự tin tưởng chia rẽ

Các công cụ có hai hương vị quan trọng cho an toàn.

- **Pure.**Chỉ đọc, quyết định, không có tác dụng phụ. `get_weather`- `search_docs`- `get_current_time`- Được gọi theo cách suy đoán.
- **Consequential.**Nhập trạng thái, chi tiền, chạm vào dữ liệu người dùng. `send_email`- `delete_file`- `execute_trade`- Phải có cửa.

Meta's 2026 "Rule of Two" cho bảo mật đại lý nói một lượt có thể kết hợp tối đa hai: đầu vào không đáng tin cậy, dữ liệu nhạy cảm, hành động hậu quả. giao diện công cụ là nơi bạn thực thi quy tắc đó bằng cách từ chối cuộc gọi, yêu cầu xác nhận người dùng hoặc leo thang phạm vi. Xem giai đoạn 13 · 15 cho toàn bộ chương bảo mật và giai đoạn 14 · 09 cho chính sách quyền cấp đại lý.

### Ở đâu vòng lặp sống

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

Ở khắp mọi nơi, bốn bước tương tự.

### Tại sao không chỉ yêu cầu mô hình phát ra JSON?

"Hãy yêu cầu mô hình trả lời trong JSON" là mô hình gọi trước chức năng. Nó thất bại từ 5 đến 15% thời gian trên các mô hình biên giới và nhiều hơn nữa trên các mô hình nhỏ hơn. Các chế độ thất bại bao gồm các dây đeo bị thiếu, dấu ngoặc sau, các trường ảo giác và các loại sai. Sau đó bạn cần một thẻ sửa chữa JSON, thử lại hoặc một bộ giải mã bị hạn chế.

Đổi tiếng gọi chức năng bản địa là tốt hơn vì ba lý do. Đầu tiên, nhà cung cấp đào tạo mô hình từ đầu đến cuối theo hình dạng gọi chính xác, do đó tỷ lệ JSON hợp lệ tăng lên 98 đến 99% trên chế độ nghiêm ngặt. Thứ hai, tải trọng hữu ích của cuộc gọi nằm trong khe giao thức riêng của nó, không nằm trong văn bản tự do  do đó một cuộc gọi công cụ không bao giờ rò rỉ vào câu trả lời hiển thị của người dùng. Thứ ba, các nhà cung cấp thực thi tuân thủ các kế hoạch với việc giải mã hạn chế (chế độ nghiêm ngặt của OpenAI, Anthropic `tool_use`, của Gemini `responseSchema`(c) Tạo ra được đảm bảo để xác nhận.

Giai đoạn 13 · 02 đi cùng ba API nhà cung cấp. Giai đoạn 13 · 04 đi sâu vào các kết quả có cấu trúc.

### Máy cắt mạch

Loop kết thúc khi mô hình ngừng phát ra cuộc gọi hoặc máy chủ đạt con số lượt tối đa. Các máy chủ sản xuất đặt điều này ở khoảng 5 đến 20 lượt. Ngoài ra, bạn gần như chắc chắn đang trong một vòng lặp mà mô hình không thể thoát ra. Claude Code mặc định là 20; OpenAI Assistants là 10; chế độ đại lý của Cursor là 25.

Các vòng lặp không giới hạn thay thế xuất hiện mỗi sáu tháng như "nhà nhân chi 400 đô la cho các cuộc gọi API qua đêm" sau khi chết.

Giai đoạn 14 · 12 bao gồm việc phục hồi lỗi và tự chữa trị sâu sắc; Giai đoạn 17 bao gồm giới hạn tốc độ sản xuất.

### Từ đây, giai đoạn 13 sẽ đi đến

- Bài học 02 đến 05 làm sáng tỏ bề mặt gọi công cụ cấp nhà cung cấp.
- Bài học 06 đến 14 tổng hợp vòng lặp thành MCP.
- Bài học 15 đến 18 bảo vệ vòng lặp chống lại các máy chủ thù địch, người dùng đối kháng và các bề mặt auth từ xa không xác nhận.
- Bài học 19 đến 22 mở rộng mô hình đến sự hợp tác giữa các đại lý, khả năng quan sát, định tuyến và đóng gói.
- Bài học 23 tạo ra một hệ sinh thái hoàn chỉnh sử dụng mọi thứ nguyên thủy.

Mỗi bài học còn lại là một sự phức tạp của vòng lặp bốn bước này. Hãy nhớ rằng như là không biến đổi.

```figure
tp-tool-loop
```

## Sử dụng nó

`code/main.py`chạy vòng lặp bốn bước mà không có LLM. Một chức năng "những người quyết định" giả mô phỏng mô hình bằng cách so sánh mô hình trên thông điệp người dùng; người thực thi, xác nhận sơ đồ và vòng kiểm tra bước thực.

Những gì cần xem:

- Các tool registry có ba trường cho mỗi tool: tên, mô tả, schema và một trình tham khảo thực thi.
- Các xác thực viên là một bộ phụ quy trình JSON tối thiểu (loại, yêu cầu, enum, min/max) được viết bằng stdlib.
- Các đại lý sản xuất cần một bộ cắt mạch chính xác như thế này.

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/skill-tool-interface-reviewer.md`. Với một bản thảo định nghĩa công cụ (tên + mô tả + sơ đồ + phác thảo trình thực hiện), kỹ năng kiểm tra nó cho phù hợp vòng lặp: tên là máy ổn định, mô tả là một sử dụng ngắn gọn hoàn chỉnh, liệu sơ đồ sử dụng JSON Schema 2020-12 đúng không, và là phân loại tinh khiết đối với hậu quả rõ ràng.

## Các bài tập

1. Thêm một công cụ thứ tư vào `code/main.py`gọi `get_stock_price(ticker)`. Tác lại mô tả của nó như " Sử dụng khi người dùng yêu cầu giá cổ phiếu hiện tại bằng ticker. Không sử dụng cho giá lịch sử hoặc bản tóm tắt thị trường. " Đánh vào vòng và xác nhận các truy vấn đường lối quyết định giả sử đề cập ticker cho công cụ mới.

2. Tháo xác minh schema.`arguments`object bị thiếu một trường yêu cầu, và xác nhận host từ chối nó trước khi thực hiện. Sau đó thực hiện một cuộc gọi với một trường không rõ ràng thêm. quyết định: host phải từ chối hoặc bỏ qua? biện minh cho sự lựa chọn của bạn với một lập luận an toàn.

3. Đánh phân loại mỗi công cụ trong vòng đeo là thuần túy hoặc hậu quả.`consequential: true`cờ vào các mục đăng ký cần nó, và thay đổi vòng lặp để in một dòng "sẽ xác nhận với người dùng" bất cứ khi nào một công cụ liên quan được chọn. Đây là hình dạng của cổng xác nhận mà mọi máy chủ sản xuất cần.

4. Hình vẽ vòng lặp bốn bước trên giấy với bảng cột nhà cung cấp ở trên được điền vào cho khách hàng yêu thích của bạn (Claude Desktop, Cursor, ChatGPT hoặc một ngăn xếp tùy chỉnh).

5. Đọc hướng dẫn gọi hàm của OpenAI từ đầu xuống dưới. Xác định một trường nằm trong yêu cầu nhưng không nằm trong vòng vòng bốn bước như được trình bày ở đây. Giải thích điều gì nó thêm và tại sao nó thuận tiện hơn là cần thiết.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tool | "A thing the model can call" | A triple of name + JSON-Schema-typed input + executor function |
| Function calling | "Native tool use" | Provider-level API support for emitting structured tool calls instead of prose |
| Tool call | "The model's request to act" | A JSON payload with `id`, `name`, `arguments` emitted by the model |
| Tool result | "What the tool returned" | The executor's output, wrapped in a `tool` role message with matching id |
| Parallel tool calls | "Many calls at once" | Multiple call objects in one model turn, independent and orderable by id |
| Strict mode | "Guaranteed JSON" | Constrained decoding that forces the model's output to validate against the declared schema |
| Pure tool | "Read-only tool" | No side effects; safe to re-run |
| Consequential tool | "Action tool" | Mutates external state; requires gate, audit, or user confirmation |
| Four-step loop | "The tool-call cycle" | describe → decide → execute → observe |
| Host | "Agent runtime" | The program that holds the tool registry, calls the model, and runs the executor |

## Đọc thêm

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) Khán giả tham khảo về các thông báo công cụ kiểu OpenAI và hình thức gọi
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) Claude `tool_use`- `tool_result`định dạng khối
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) `functionDeclarations`và ngữ nghĩa gọi song song trong Gemini
- [Model Context Protocol — Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) sự phổ biến hiện tại của giao diện công cụ không có nhà nước, cung cấp-những người không biết về nhà cung cấp
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) phương ngữ schema mọi công cụ hiện đại API nói
