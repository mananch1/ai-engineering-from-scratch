# Xây dựng máy chủ MCP: Python và TypeScript không có quốc tịch

> Một máy chủ MCP hiện đại không nhớ một cú tay. Nó xác nhận metadata trên mỗi yêu cầu, chạy một bộ xử lý và trả lại một kết quả được gõ.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## Mục tiêu học tập

- Thực hiện bắt buộc `server/discover`cho MCP `2026-07-28`- Tôi không biết.
- Thiết lập phiên bản giao thức và khả năng của khách hàng trên mỗi yêu cầu.
- Khám phá các công cụ, tài nguyên và lời nhắc với thứ tự danh sách xác định.
- Trở lại`resultType`, danh tính máy chủ, và cache gợi ý về kết quả chính xác.
- Dịch vụ hợp đồng không có quốc tịch tương tự trên newline-delimited studio trong Python và TypeScript.

## Vấn đề

Một máy chủ lưu trữ khả năng của khách hàng sau khi tin nhắn đầu tiên là dễ dàng để xây dựng và khó hoạt động.

MCP `2026-07-28`ứng dụng của bạn vẫn có thể giữ các ghi chú lâu dài, công việc, hoặc xử lý trạng thái rõ ràng. Điều mà nó không thể giữ là trạng thái giao thức ẩn thay đổi cách giải mã yêu cầu sau đó.

Bài học này xây dựng một máy chủ ghi chú hai lần. Phiên bản Python và TypeScript chỉ sử dụng thư viện tiêu chuẩn của họ cho lõi giao thức. Cả hai phơi bày các phương pháp tương tự và thực thi hợp đồng dây tương tự.

## Khái niệm

### Loop chuyển giao hiện đại

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

Ba quy tắc của studio vẫn còn quan trọng:

- Chỉ viết tin nhắn JSON-RPC cho stdout.
- Định nghĩa các tin nhắn bằng một dòng mới và đánh dấu mỗi câu trả lời.
- Hãy ra khỏi ngay khi Stdin đến EOF.

Thời gian vận chuyển là thời gian vận chuyển, không phải là một phiên MCP hiện đại.

### Đơn xin xác thực

Mỗi yêu cầu phải có:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Hai trường đầu tiên là cần thiết. `clientInfo`được khuyến cáo. xác nhận hình dạng danh tính hiện tại, nhưng không coi nó như xác thực.

Nếu phiên bản không được hỗ trợ, trả lại mã `-32022`với `requested`và `supported`. Mất dữ liệu liên quan đến yêu cầu là param không hợp lệ, mã `-32602`Đừng bao giờ điền vào các trường bị mất từ cuộc gọi trước đó.

### Việc khám phá bắt buộc

Các máy chủ hiện đại phải triển khai `server/discover`. Kết quả phát hiện đầy đủ bao gồm các phiên bản hiện đại được hỗ trợ, khả năng, hướng dẫn tùy chọn, gợi ý cache và nhận dạng máy chủ kết quả `_meta`- Có thể là:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

Discovery không mở khóa máy chủ.`tools/list`Không gọi là khám phá bởi vì`tools/list`đã mang cùng một yêu cầu metadata.

### Công cụ

`tools/list`trả lại một danh sách xác định của mô tả công cụ. Stable ordering cải thiện lưu trữ dự trữ phản ứng và giữ cho bối cảnh mô hình ổn định. Kết quả cũng yêu cầu`ttlMs`và `cacheScope`- Tôi không biết.

`tools/call`trả lại các khối nội dung và `isError`Sử dụng lỗi JSON-RPC khi gói giao thức hoặc các tham số phương pháp không hợp lệ. Sử dụng `isError: true`khi một công cụ invocation hợp lệ chạy nhưng công cụ tự thất bại.

Các chú thích về công cụ vẫn là gợi ý, không phải là thực thi:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

Người chủ nhà nên sử dụng chúng để xác nhận và trình bày.

### Tài nguyên

`resources/list`trả lại các mô tả URI ổn định. `resources/read`trả lại nội dung được gõ. Cả hai đều có thể lưu trữ trong `2026-07-28`, nên cả hai đều bao gồm`ttlMs`và `cacheScope`- Tôi không biết.

Sử dụng `cacheScope: "private"`cho dữ liệu ghi chú cụ thể cho người dùng. Một bộ nhớ cache được chia sẻ không được sử dụng lại một phản ứng riêng tư trên các bối cảnh ủy quyền.

Việc chuyển đổi hiện đại không sử dụng `resources/subscribe`Một khách hàng mở cửa`subscriptions/listen`và yêu cầu `resourceSubscriptions`Bài học 10 xây dựng dòng chảy đó.

### Các lời nhắc nhở

`prompts/list`là cacheable và xác định. `prompts/get`kết quả prompt được render hoàn chỉnh, nhưng nó không phải là một trong danh sách cache hoặc kết quả đọc cần các gợi ý cache.

### Mỗi kết quả thành công đều được đánh dấu

Các ví dụ sử dụng một gói cho mỗi thành công:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

Danh sách, đọc và phát hiện người xử lý thêm `ttlMs`+`cacheScope`Việc tập trung bao bì này ngăn cản một người xử lý từ chối lặng lẽ bỏ qua các trường kết quả hiện đại.

### Không có yêu cầu được khởi động bởi máy chủ

Một máy chủ hiện đại có thể gửi thông báo liên quan đến yêu cầu của khách hàng hoặc thông báo trên một máy chủ mở `subscriptions/listen`Nó không được gửi yêu cầu JSON-RPC của riêng nó.

Khi một người xử lý cần lấy mẫu, tạo ra hoặc nhập gốc, nó trả lại một `input_required`kết quả. Client đáp ứng các yêu cầu nhập được nhúng và thử lại phương pháp ban đầu với một ID yêu cầu mới. Bài học 11 bao gồm mô hình yêu cầu nhiều chuyến đi vòng.

### Sự tương thích rõ ràng của sản phẩm

Một máy chủ hai thời đại cũng có thể thực hiện các`2025-11-25`Nhấn tay vào một nhánh di sản rõ ràng tách biệt. Nó chọn hành vi hiện đại khi cần hiện đại`_meta`các trường hiện diện và di sản hành vi khi nó nhận `initialize`- Tôi không biết.

Đừng đặt một `2026-07-28`xin thông qua con đường bắt tay cũ. Đừng đóng dấu hiện đại `resultType`mã trong bài học này là cố ý hiện đại chỉ để các biến số của nó vẫn còn hiển thị.

```figure
t3-dispatch-loop
```

## Sử dụng nó

Thực hiện demo và thử nghiệm hữu hạn của máy chủ Python:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

Tiến TypeScript với một trình chạy TypeScript:

```bash
npx tsx main.ts --demo
```

Demo gửi đi`server/discover`, liệt kê từng nguyên thủy, gọi các công cụ, và hiển thị lỗi phiên bản không được hỗ trợ.

## Chuyển nó

Bài học này sẽ đi theo `outputs/skill-mcp-server-scaffolder.md`Nó tạo ra một kế hoạch máy chủ hiện đại với hợp đồng phát hiện, xác nhận theo yêu cầu, danh sách cache xác định và một bộ chuyển đổi di sản riêng biệt tùy chọn.

## Các bài tập

1. Xóa các khả năng từ một yêu cầu và chứng minh máy chủ không sử dụng lại tuyên bố yêu cầu trước đó.
2. Chuyển lại `TOOLS`- `PROMPTS`, và ghi chú thứ tự để đưa vào.
3. Thêm một cái phá hủy `notes_delete`công cụ và yêu cầu kiểm tra ủy quyền bên trong trình thực.`destructiveHint`chỉ là một gợi ý UX.
4. Thêm `resources/templates/list`với `ttlMs`- `cacheScope`, và định nghĩa thứ tự.
5. Xây dựng một bộ chuyển đổi cũ riêng cho `2025-11-25`Thêm thêm các xét nghiệm chứng minh rằng một yêu cầu hiện đại không bao giờ được đưa vào.

## Các điều khoản chính

| Term | Meaning |
|------|---------|
| Stateless server | Handles each request from its own metadata without protocol-session memory |
| `server/discover` | Mandatory modern method that advertises versions and capabilities |
| Complete result | Successful modern result with `resultType: "complete"` |
| Cacheable result | Discovery, list, or resource-read result with `ttlMs` and `cacheScope` |
| Deterministic list | Same logical registry produces the same item order |
| Server identity | Recommended `io.modelcontextprotocol/serverInfo` in result `_meta` |
| Tool error | Valid tool call that returns content with `isError: true` |
| Protocol error | Invalid JSON-RPC or MCP request returned through `error` |

## Đọc thêm

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
