# Sử dụng công cụ và gọi chức năng

> Toolformer (Schick et al., 2023) bắt đầu tự giám sát ghi chú công cụ. Berkeley Function Calling Leaderboard V4 (Patil et al., 2025) đặt thanh 2026: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% ảo giác.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## Mục tiêu học tập

- Giải thích tín hiệu đào tạo tự giám sát của Toolformer: chỉ giữ ghi chú công cụ khi thực thi giảm mất token tiếp theo.
- Hãy nêu tên năm loại đánh giá của BFCL V4 và mỗi loại đánh giá là gì.
- Thực hiện một danh sách công cụ stdlib với xác thực schema, buộc lập luận và sandboxing thực thi.
- Chẩn đoán ba vấn đề mở năm 2026: chuỗi công cụ đường dài, ra quyết định năng động và bộ nhớ.

## Vấn đề

Sử dụng công cụ sớm được hỏi: mô hình có thể dự đoán một cuộc gọi chức năng chính xác không? Sử dụng công cụ hiện đại hỏi: có thể các công cụ chuỗi mô hình trên 40 bước, với bộ nhớ, với khả năng quan sát một phần, với sự phục hồi từ thất bại công cụ, mà không ảo giác các công cụ không tồn tại?

Toolformer đã thiết lập đường cơ sở: các mô hình có thể học được khi nào gọi công cụ với sự tự giám sát. BFCL V4 xác định mục tiêu đánh giá năm 2026 .

## Khái niệm

### Toolformer (Schick et al., NeurIPS 2023)

Ý tưởng: để cho mô hình ghi chú bản thân trước khi tập với các cuộc gọi API ứng cử viên. Đối với mỗi ứng cử viên, thực hiện nó. Giữ ghi chú chỉ khi bao gồm kết quả công cụ làm giảm tổn thất trên token tiếp theo.

Các công cụ được bao gồm: máy tính, hệ thống kiểm tra chất lượng, công cụ tìm kiếm, dịch giả, lịch.

Kết quả quy mô: việc sử dụng công cụ xuất hiện trên quy mô. Các mô hình nhỏ hơn bị tổn thương bởi các chú thích công cụ; các mô hình lớn hơn tăng. Đây là lý do tại sao các mô hình biên giới 2026 có sử dụng công cụ mạnh mẽ được nấu trong khi hầu hết các mô hình 7B cần thiết thiết lập cụ thể sử dụng công cụ để được đáng tin cậy.

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025)

BFCL là đánh giá thực tế năm 2026.

- **Agentic (40%)** Các quỹ đạo đầy đủ: trí nhớ, nhiều vòng quay, quyết định năng động.
- **Multi-Turn (30%)** cuộc trò chuyện tương tác với chuỗi công cụ.
- **Live (10%)** Các yêu cầu thực được người dùng gửi (cải phân phối khó khăn hơn).
- **Non-Live (10%)** Các trường hợp thử nghiệm tổng hợp.
- **Hallucination (10%)** phát hiện khi không cần gọi công cụ.

V3 giới thiệu đánh giá dựa trên trạng thái: sau một chuỗi công cụ, kiểm tra trạng thái thực tế của API (ví dụ: "tệp đã được tạo ra không?") thay vì phù hợp với AST của các cuộc gọi công cụ. V4 thêm các loại tìm kiếm web, bộ nhớ và độ nhạy định dạng.

Phát hiện quan trọng của năm 2026: gọi hàm một lượt gần như đã được giải quyết. Các lỗi tập trung trong bộ nhớ (tiếp tục mang bối cảnh qua các lượt), ra quyết định năng động (chọn các công cụ dựa trên kết quả trước đó), chuỗi đường chân trời dài (trở đi sau 20+ bước), và phát hiện ảo giác (đà từ chối gọi khi không có công cụ phù hợp).

### Chế hoạch công cụ

Mỗi nhà cung cấp đều có một kế hoạch.

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Sử dụng nhân loại `input_schema`trực tiếp. OpenAI sử dụng `function.parameters`. Cả hai đều chấp nhận JSON Schema. mô tả chịu tải  mô hình đọc chúng để chọn công cụ đúng. mô tả công cụ xấu là nguyên nhân gốc số 1 của sự thất bại trong việc chọn công cụ sai.

### Định giá lý lẽ

Không tin vào công cụ gọi.

1. **Type coercion.**Mô hình có thể trả lại chuỗi "5" nơi schema nói int. Cố gắng nếu không rõ ràng; từ chối nếu không.
2. **Enum validation.**Nếu kế hoạch nói `status in {"open", "closed"}`và các loại phát thải mô hình `"in_progress"`, từ chối với một lỗi mô tả.
3. **Required fields.**Thiếu trường yêu cầu -> quan sát lỗi ngay lập tức trở lại mô hình, không phải là một vụ tai nạn.
4. **Format validation.**Ngày, email, URL  xác nhận bằng các parsers bê tông, không phải regex.

Mỗi thất bại xác nhận nên trả lại một quan sát có cấu trúc để mô hình có thể thử lại với hình dạng chính xác.

### Các cuộc gọi công cụ song song

Các nhà cung cấp hiện đại hỗ trợ các cuộc gọi công cụ song song trong một lượt trợ lý.

1. Mô hình phát ra 3 tool call với các điểm khác nhau `tool_use_id`S.
2. Runtime thực hiện chúng (trong song nếu độc lập).
3. Mỗi kết quả đều trở lại như một`tool_result`khối liên quan đến `tool_use_id`- Tôi không biết.

Quy tắc kỹ thuật: coi ID tương quan như là chịu tải, thay đổi chúng và bạn có được đường dẫn từ công cụ sai đến kết quả sai.

### Sandboxing

Việc thực hiện công cụ là ranh giới hộp cát. Xem Bài học 09 để biết chi tiết. Phiên bản ngắn: mỗi công cụ nên chỉ định bề mặt đọc/tập, truy cập mạng, thời gian, memory cap.`run_shell(cmd)`là một lá cờ đỏ; cụ thể `git_status()`an toàn hơn.

```figure
tool-routing
```

## Hãy xây dựng nó

`code/main.py`thực hiện một sổ đăng ký công cụ hình dạng sản xuất:

- JSON Schema subset validator (stdlib chỉ).
- Đăng ký công cụ với mô tả, sơ đồ nhập, thời gian nghỉ và trình thực hiện.
- Sự buộc buộc và xác nhận bằng chứng.
- Chuyển dụng đồng bộ với các thẻ xác định tương quan.
- Các quan sát lỗi như chuỗi cấu trúc.

Đi đi.

```
python3 code/main.py
```

Hình ảnh này cho thấy một đại lý nhỏ gọi ba công cụ cùng một lượt, với một cuộc gọi cố ý bị sai lệch mà bị từ chối với một lỗi mô hình mô tả có thể hành động.

## Sử dụng nó

Mỗi nhà cung cấp có kế hoạch công cụ riêng của mình  Anthropic, OpenAI, Gemini, Bedrock. Sử dụng một lớp dịch (OpenAI Agents SDK, Vercel AI SDK, LangChain tool adapter) nếu bạn cần nhiều nhà cung cấp. BFCL là điểm tham khảo  chạy nó chống lại đại lý của bạn trước khi vận chuyển nếu việc sử dụng công cụ là trung tâm của sản phẩm.

## Chuyển nó

`outputs/skill-tool-registry.md`tạo ra danh mục công cụ, sơ đồ và đăng ký cho một tên miền nhiệm vụ nhất định. Bao gồm kiểm tra chất lượng mô tả (có mô tả của mỗi công cụ cho mô hình biết khi nào sử dụng nó?).

## Các bài tập

1. Thêm một công cụ "không có" cho phép mô hình rõ ràng từ chối sử dụng bất kỳ công cụ nào khác.
2. Thực hiện buộc buộc tranh luận cho int-as-string và float-as-string.
3. Thêm một thời gian nghỉ mỗi công cụ và một bộ cắt mạch (đánh bỏ công cụ trong 60s sau 3 thất bại liên tiếp). Điều này thay đổi gì về cách mô hình phục hồi?
4. Đọc mô tả BFCL V4. Chọn một danh mục (ví dụ: "multi-turn") và chạy 10 ví dụ yêu cầu thông qua đại lý của bạn.
5. Đưa xác nhận STDlib vào Pydantic hoặc Zod.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Structured-output tool invocation with validated schema |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — keep tool calls whose results reduce next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Function signature for the model" | name, description, JSON Schema of arguments |
| tool_use_id | "Correlation ID" | Ties a tool call to its result; essential for parallel dispatch |
| Hallucination detection | "Know when not to call" | V4 category: refuse to call when no tool fits |
| Argument coercion | "String-to-int repair" | Narrow fixes for predictable schema-mismatch; reject if ambiguous |
| Sandboxing | "Tool execution boundary" | Per-tool read/write surface, network, timeout, memory cap |

## Đọc thêm

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) Nhận xét công cụ tự giám sát
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) Định nghĩa chuẩn đánh giá năm 2026
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) Sơ đồ công cụ sản xuất trong SDK Claude Agent
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) loại công cụ chức năng và Guardrails
