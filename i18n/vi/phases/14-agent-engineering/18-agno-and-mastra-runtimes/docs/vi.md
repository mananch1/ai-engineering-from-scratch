# Thời gian vận hành của đại lý sản xuất  Lần thực hiện nhanh và dòng công việc được đánh dấu

> Một trình sản xuất vận hành thời gian tối ưu hóa những gì các khung nguyên mẫu bỏ qua: chi phí thực hiện, bề mặt luồng công việc được gõ, và một phần mềm sẵn sàng phục vụ. Sự kết hợp năm 2026: Agno (Python) nhằm mục đích thực hiện thực hiện đại lý microsecond và các phần mềm FastAPI không có trạng thái. Mastra gửi các đại lý, công cụ, luồng công việc, định tuyến mô hình thống nhất và lưu trữ tổng hợp trên nền SDK Vercel AI.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Mục tiêu học tập

- Hãy xác định mục tiêu hiệu suất của Agno và khi nào chúng quan trọng.
- Tên gọi ba nguyên thủy của Mastra  Đại lý, Công cụ, Luồng làm việc  và các bộ điều chỉnh máy chủ được hỗ trợ.
- Giải thích tại sao một backend FastAPI không có nội dung được phân tích theo phiên bản là con đường sản xuất Agno được khuyến cáo.
- Chọn Agno vs Mastra cho một đống nhất định (Python-đầu tiên vs TypeScript-đầu tiên).

## Vấn đề

LangGraph, AutoGen, CrewAI là các khung nặng. Các nhóm muốn "chỉ cần vòng tròn đại lý, nhanh chóng, trong thời gian chạy của tôi" tiếp cận với Agno (Python) hoặc Mastra (TypeScript). Cả hai đều trao đổi một số nguyên thủy thuộc sở hữu khung cho tốc độ thô và phù hợp chặt chẽ hơn với hàng đống xung quanh.

## Khái niệm

### Agno

- Thời gian chạy Python, trước đây là Phi-data.
- "Không có đồ thị, chuỗi, hoặc các mẫu phức tạp  chỉ là Python thuần túy".
- Mục tiêu hiệu suất từ tài liệu của họ: ~ 2μs đại lý thực hiện, ~ 3,75 KiB bộ nhớ mỗi đại lý, ~ 23 nhà cung cấp mô hình.
- Chế độ sản xuất: FastAPI backend không có trạng thái phiên bản. Mỗi yêu cầu bắt đầu một đại lý mới; trạng thái phiên bản sống trong DB.
- Native multimodal (text, image, audio, video, file) và RAG cơ quan.

Các mục tiêu tốc độ quan trọng khi bạn có hàng ngàn nhân viên ngắn ngủi mỗi giây (bàn máy trò chuyện, đường ống đánh giá), nhưng ít quan trọng hơn khi một nhân viên chạy trong 10 phút.

### Mastra

- TypeScript, được xây dựng trên Vercel AI SDK.
- Ba nguyên thủy:**Agents**- **Tools**(Tý dạng vùng),**Workflows**- Tôi không biết.
- Mô hình thống nhất Router  3,300+ mô hình trên 94 nhà cung cấp (tháng 3 năm 2026).
- lưu trữ kết hợp: bộ nhớ, luồng công việc, khả năng quan sát cho các nền khác nhau; ClickHouse được khuyến cáo để có thể quan sát được ở quy mô.
- Apache 2.0 với `ee/`Các thư mục dưới giấy phép doanh nghiệp có sẵn từ nguồn.
- Các bộ điều chỉnh máy chủ cho Express, Hono, Fastify, Koa; kết hợp Next.js và Astro hạng nhất.
- Tàu Mastra Studio (localhost:4111) để debugging.
- 22k+ GitHub sao, 300k+ tải về npm hàng tuần ở mức 1.0 (Từ tháng 1 năm 2026).

### Định vị

Họ cũng không muốn trở thành LangGraph.

- **Language fit.**Agno cho các nhóm đầu tiên của Python; Mastra cho TypeScript đầu tiên.
- **Runtime ergonomics.**Agno = gần không chi phí hàng đầu; Mastra = tích hợp với hệ sinh thái Vercel.
- **Observability.**Cả hai tích hợp với Langfuse / Phoenix / Opik (Dạy 24) nhưng Mastra Studio là bên đầu tiên.

### Khi nào để chọn mỗi

- **Agno** Python backend, nhiều đại lý ngắn ngủi, yêu cầu perf mạnh mẽ, FastAPI shop.
- **Mastra** TypeScript backend, Next.js / Vercel triển khai, định tuyến mô hình đa nhà cung cấp thống nhất, công cụ kiểu Zod.
- **LangGraph**(Dân học 13)  khi trạng thái bền và lý luận biểu đồ rõ ràng quan trọng hơn tốc độ thô.
- **OpenAI / Claude Agent SDK** khi bạn muốn hình dạng sản xuất của nhà cung cấp (Dạy học 1617).

### Khi mô hình này đi sai

- **Perf-for-perf's-sake.**Chọn Agno vì "2μs" nghe có vẻ tốt khi tải trọng công việc là một cuộc gọi chậm của đại lý theo yêu cầu.
- **Ecosystem lock-in.**Sự tích hợp hương vị Vercel của Mastra là một cộng với Vercel, một trừ ở nơi khác.
- **Enterprise license confusion.**Mastra của `ee/`thư mục có nguồn, không phải Apache 2.0. đọc giấy phép nếu bạn đang lên kế hoạch để fork.

```figure
wb-runtime-spawn
```

## Hãy xây dựng nó

Bài học này chủ yếu là so sánh  không có một tạo vật mã duy nhất sẽ làm cho cả hai khung công bằng. Xem `code/main.py`cho một đồ chơi bên cạnh: một dòng chảy tối thiểu "đưa ra một chất, truyền ra đầu ra, tiếp tục phiên" được thực hiện hai lần (một lần hình Agno, một lần hình Mastra).

Đi đi.

```
python3 code/main.py
```

Hai dấu vết khác nhau về cấu trúc nhưng tương đương về chức năng.

## Sử dụng nó

- **Agno** Python backend cần tốc độ và hình dạng FastAPI.
- **Mastra** TypeScript backend với nhiều nhà cung cấp và dòng công việc nguyên thủy.
- Cả hai tàu đều có các cái nén quan sát được sử dụng.

## Chuyển nó

`outputs/skill-runtime-picker.md`chọn Agno, Mastra, LangGraph, hoặc một SDK nhà cung cấp dựa trên xếp chồng, ngân sách độ trễ và hình dạng hoạt động.

## Các bài tập

1. Đọc tài liệu của Agno, chuyển mạch ReAct (Dạy 01) đến Agno.
2. Đọc các tài liệu của Mastra. Đưa cùng một vòng lặp đến Mastra. Điều gì thay đổi trong công cụ gõ (Zod vs không gì)?
3. Định nghĩa: đo thời gian trễ của các đại lý trên đống của bạn.
4. Thiết kế một di chuyển: nếu bạn đã chạy CrewAI trong Python, những gì bị phá vỡ nếu bạn chuyển đến Agno?
5. Đọc bài của Mastra `ee/`Những hạn chế nào sẽ ảnh hưởng đến một fork mã nguồn mở?

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability each to a different store |
| Mastra Studio | "Local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "Not OSS" | License permits source reading but restricts commercial use |

## Đọc thêm

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) Mục tiêu hiệu suất, tích hợp FastAPI
- [Mastra docs](https://mastra.ai/docs) nguyên thủy, máy chủ chuyển đổi, Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) thay thế biểu đồ nhà nước
- [Comet Opik](https://www.comet.com/site/products/opik/) so sánh khả năng quan sát được trích dẫn bởi các tích hợp Mastra
