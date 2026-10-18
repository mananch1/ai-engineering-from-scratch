# A2A  Phương pháp giao dịch giữa các đại lý

> MCP là đại lý-đồ dùng. A2A (Agent2Agent) là một giao thức mở để cho phép các đại lý không minh bạch được xây dựng trên các khung khác nhau hợp tác. Được phát hành bởi Google vào tháng 4 năm 2025, được quyên góp cho Quỹ Linux vào tháng 6 năm 2025, đạt v1.0 vào tháng 4 năm 2026 với 150 + người ủng hộ bao gồm AWS, Cisco, Microsoft, Salesforce, SAP và ServiceNow. Nó hấp thụ ACP của IBM và thêm mở rộng thanh toán AP2. Bài học này sẽ hướng dẫn thẻ đại lý, vòng đời nhiệm vụ và ba giao thức liên kết, sử dụng tên dây A2A 1.0.1.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Mục tiêu học tập

- Sự khác biệt giữa các trường hợp sử dụng từ đại lý đến công cụ (MCP) và các trường hợp sử dụng từ đại lý đến đại lý (A2A).
- Giới thiệu thẻ đại lý tại `/.well-known/agent-card.json`có kỹ năng và`supportedInterfaces`Metadata.
- Đi theo vòng đời nhiệm vụ: `TASK_STATE_SUBMITTED`- `TASK_STATE_WORKING`- `TASK_STATE_INPUT_REQUIRED`, và các trạng thái cuối cùng `TASK_STATE_COMPLETED`- `TASK_STATE_FAILED`- `TASK_STATE_CANCELED`- `TASK_STATE_REJECTED`- Tôi không biết.
- Sử dụng Thông điệp có từng phần chứa một phần của `text`- `raw`- `url`, hoặc`data`, và các tác phẩm tạo ra như các sản phẩm.

## Vấn đề

Một đại lý dịch vụ khách hàng cần ủy quyền viết báo cáo cho một đại lý viết chuyên ngành.

- REST API tùy chỉnh, hoạt động nhưng mỗi cặp đều là một lần.
- Hình thức chia sẻ mã, yêu cầu hai đại lý chạy cùng một khung.
- MCP không phù hợp: MCP là để gọi công cụ, không phải cho hai đại lý hợp tác trong khi vẫn giữ cho lý luận nội bộ không rõ ràng của mỗi đại lý.

A2A lấp đầy khoảng trống. Nó mô hình hóa sự tương tác khi một đại lý gửi một Task đến một đại lý khác, với chu kỳ sống, tin nhắn và đồ tạo. trạng thái nội bộ của đại lý được gọi vẫn không minh bạch  người gọi chỉ thấy chuyển đổi trạng thái nhiệm vụ và đầu ra cuối cùng.

A2A là giao thức "cho phép các đại lý trên các khung nói chuyện với nhau".

## Khái niệm

### Cảnh sát Thư

Mỗi đại lý tuân thủ A2A đều xuất bản một thẻ tại `/.well-known/agent-card.json`- Có thể là:

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

Khám phá dựa trên URL: lấy thẻ, chọn thẻ đầu tiên `supportedInterfaces`mục có `protocolBinding`khách hàng của bạn nói, và liệt kê kỹ năng.

### Thẻ đại lý đã ký

Một thẻ có thể mang theo một `signatures`mỗi mục là một JWS (RFC 7515) được tính trên thẻ RFC 8785 JSON, với `signatures`người tiêu dùng canonicalize thẻ theo cùng một cách và xác minh. ngăn chặn giả mạo.

### Chuyển đời của nhiệm vụ

```text
TASK_STATE_SUBMITTED
  -> TASK_STATE_WORKING
  -> TASK_STATE_COMPLETED | TASK_STATE_FAILED | TASK_STATE_CANCELED | TASK_STATE_REJECTED

TASK_STATE_WORKING
  -> TASK_STATE_INPUT_REQUIRED
  -> TASK_STATE_WORKING (the client sends a message with the same taskId)
```

Khách hàng bắt đầu với `SendMessage`, và máy chủ tạo Task. người gọi là đại lý chuyển qua các trạng thái; khách hàng thăm dò với `GetTask`hoặc chảy qua SSE với `SendStreamingMessage`và `SubscribeToTask`Một dòng chảy mang theo`statusUpdate`và `artifactUpdate`các sự kiện và kết thúc khi nhiệm vụ đạt đến trạng thái cuối cùng.`final`cờ.

### Thông điệp và phần

Một tin nhắn có một `messageId`, một `role`(`ROLE_USER`hoặc `ROLE_AGENT`), và một hoặc nhiều phần. Mỗi phần chứa chính xác một trường nội dung, và tên trường đó là loại. Không có `kind`trường.

- `text`: nội dung đơn giản.
- `raw`: file bytes, base64 trong JSON, thường với `filename`và `mediaType`- Tôi không biết.
- `url`: một liên kết đến nội dung tập tin.
- `data`: tải trọng hữu ích JSON có cấu trúc (truyền nhập có cấu trúc cho đại lý được gọi).

Ví dụ:

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

### Các đồ tạo tác

Các sản phẩm là các sản phẩm, không phải chuỗi nguyên liệu. Một sản phẩm là một sản phẩm có tên, được gõ:

```json
{
  "artifactId": "art-001",
  "name": "summary",
  "parts": [{"text": "...", "mediaType": "text/markdown"}]
}
```

Các đồ tạo vật có thể được chuyển qua các mảnh.`artifactUpdate`Sự kiện mang theo đồ tạo vật cộng với `append`và `lastChunk`Người gọi sẽ tích lũy.

### Ba giao thức liên kết

1. **JSON-RPC 2.0 over HTTP**(`JSONRPC`). POST cho các yêu cầu, SSE cho phát trực tuyến.`SendMessage`- `SendStreamingMessage`- `GetTask`- `ListTasks`- `CancelTask`- `SubscribeToTask`- `CreateTaskPushNotificationConfig`- `GetTaskPushNotificationConfig`- `ListTaskPushNotificationConfigs`- `DeleteTaskPushNotificationConfig`, và`GetExtendedAgentCard`- Tôi không biết.
2. **gRPC**(`GRPC`(c) Đối với môi trường doanh nghiệp nơi gRPC là bản địa.
3. **HTTP+JSON/REST**(`HTTP+JSON`). Các URL tài nguyên như `POST /message:send`và `GET /tasks/{id}`- Tôi không biết.

Cả ba liên kết đều có cùng mô hình dữ liệu.`supportedInterfaces`tên mục một liên kết và của nó `protocolVersion`Khách hàng gửi tiêu đề`A2A-Version: 1.0`trên mỗi yêu cầu, bởi vì một máy chủ đọc một yêu cầu mà không có nó như phiên bản 0.3.

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

### Bảo tồn độ trống

Một nguyên tắc thiết kế chính: trạng thái nội bộ của đại lý được gọi là không minh bạch. Người gọi thấy trạng thái nhiệm vụ và các vật liệu. Dòng tư tưởng của đại lý được gọi, các cuộc gọi công cụ của nó, đại diện phụ của nó  tất cả đều vô hình. Điều này khác với MCP, nơi các cuộc gọi công cụ là minh bạch.

Lý luận: A2A cho phép các đối thủ cạnh tranh hợp tác mà không tiết lộ nội bộ. A2A có thể là "hãy gọi cho đại lý dịch vụ khách hàng này" mà không cần người gọi học cách đại lý đó thực hiện dịch vụ.

### Thời gian

- **2025-04-09.**Google công bố A2A.
- **2025-06-23.**Được tặng cho Linux Foundation.
- **2025-08.**Thuốc ACP của IBM.
- **2025-09.**Tàu mở rộng AP2 (Giá nhân viên).
- **2026-04.**v1.0 được phát hành với 150+ tổ chức hỗ trợ.

### Mối quan hệ với MCP

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | Agent Card `securitySchemes` + `securityRequirements` |
| Transport | Stdio / Streamable HTTP | JSON-RPC / gRPC / HTTP+JSON |

Sử dụng MCP khi bạn muốn gọi một công cụ cụ thể. Sử dụng A2A khi bạn muốn ủy thác toàn bộ nhiệm vụ cho một đại lý khác. Nhiều hệ thống sản xuất sử dụng cả hai: một đại lý sử dụng MCP cho lớp công cụ của mình và A2A cho lớp hợp tác của mình.

```figure
a2a-task-lifecycle
```

## Sử dụng nó

`code/main.py`thực hiện một vòng A2A tối thiểu: đại lý viết xuất bản thẻ của mình, đại lý nghiên cứu gửi nó một `SendMessage`yêu cầu với một phần PDF và một hướng dẫn văn bản, và nhiệm vụ di chuyển qua `TASK_STATE_WORKING`→ `TASK_STATE_INPUT_REQUIRED`→ `TASK_STATE_WORKING`→ `TASK_STATE_COMPLETED`trước khi trả lại một vật liệu văn bản. Tất cả stdlib; sử dụng một di chuyển trong bộ nhớ để tập trung vào hình dạng tin nhắn.

Những gì cần xem:

- Hình dạng JSON của thẻ đại lý.
- Đề xuất ID nhiệm vụ bên máy chủ và chuyển đổi trạng thái.
- Các phần được đánh dấu bởi các trường nội dung hiện diện.
- `TASK_STATE_INPUT_REQUIRED`Lớp giữa nhiệm vụ.
- Các vật liệu sẽ được trả về khi hoàn thành.

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/skill-a2a-agent-spec.md`Với một đại lý mới mà nên được gọi bởi các đại lý khác, kỹ năng tạo ra thẻ đại lý JSON, kế hoạch kỹ năng và bản phác thảo điểm cuối.

## Các bài tập

1. Đi chạy`code/main.py`. Theo dõi toàn bộ chu kỳ đời của nhiệm vụ, bao gồm cả `TASK_STATE_INPUT_REQUIRED`dừng lại khi đại lý gọi yêu cầu giải thích.

2. Thêm thẻ đại lý ký tên, thêm một mục JWS vào.`signatures`với `alg`được thiết lập`HS256`, ký kết JSON của thẻ không có `signatures`viết một xác minh và xác nhận nó thất bại trên một thẻ đột biến.

3. Thực hiện các nhiệm vụ streaming với `SendStreamingMessage`: đại lý viết phát `task`, ba `artifactUpdate`các mảnh, và một `statusUpdate`với `TASK_STATE_COMPLETED`Người gọi sẽ tích lũy các mảnh.

4. Thiết kế một đại lý A2A bao gồm một máy chủ MCP. Bản đồ mỗi công cụ MCP để một kỹ năng A2A. Lưu ý các sự thỏa hiệp  bất độ sáng bị mất?

5. Đọc thông báo A2A v1.0 và xác định một tính năng chưa được thực hiện bởi bất kỳ khung nào vào tháng 4 năm 2026. (Thông dụ: nó liên quan đến ủy quyền nhiệm vụ đa hop).

## Các điều khoản chính

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

## Đọc thêm

- [a2a-protocol.org](https://a2a-protocol.org/latest/) Cấu chỉ A2A
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) Các thực hiện và SDK tham chiếu
- [A2A v1.0.1 release](https://github.com/a2aproject/A2A/tree/v1.0.1): các thẻ `docs/specification.md`và quy định `specification/a2a.proto`Bài học này tiếp theo
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) Tháng 6 năm 2025 chuyển giao quản trị
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) Bản đồ đường và động lực của đối tác
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) v1.0 thông báo phát hành và hướng dẫn ngược
