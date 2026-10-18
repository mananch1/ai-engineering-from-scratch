# Capstone 82  Thống kê jailbreak

> Một vòng dây an toàn không có phân loại là một cái đúc tiền xu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## Vấn đề

Một mô hình được triển khai mà không có mô hình tấn công là mô hình được bảo vệ chống lại bất cứ điều gì cụ thể. Các nhà điều hành đọc một liên kết Twitter, nhận ra thủ thuật, viết một regex, gửi nó, và tiếp tục. Điều tiếp theo là một đoạn phần. Regex bị bỏ lỡ. Một tuần sau, ai đó cho thấy cùng một trò chơi được gói trong base64 và người vận hành viết một regex thứ hai. Đến tháng 3, hệ thống có 40 quy tắc được sửa chữa, không có từ vựng chung, không có cách nói về cuộc tấn công thực sự là gì, và một số lượng vỏ bọc phát triển nhanh hơn các bản sửa chữa.

Trước khi bất kỳ bộ phát hiện, phân loại hoặc động cơ quy tắc nào trong đường đua này làm bất cứ điều gì hữu ích, nhóm cần một cách chia sẻ để gắn nhãn tấn công. Không phải vì nhãn ngăn chặn các cuộc tấn công, mà vì nhãn biến một dòng tấn công thành một histogram. Một histogram trở thành một biểu đồ bao phủ. Một biểu đồ bảo hiểm sẽ dẫn đến sprint tiếp theo. Các bài học trong bài học 83-87 dành thời gian của mình để quyết định liệu một lời nhắc có phải là, ví dụ, một cuộc tấn công đóng vai trò chống lại một chính sách từ chối so với một cuộc tấn công buôn lậu bối cảnh chống lại một công cụ. Quyết định đó là không thể nếu không có phân loại.

Bạch đá này xác định một phân loại sáu loại đủ rộng để bao gồm hầu hết các cuộc tấn công được nhìn thấy trong tự nhiên, đủ hẹp để hai nhà đánh giá thường đồng ý về danh mục, và đủ cụ thể để mỗi loại có ít nhất bảy thiết bị được xây dựng bằng tay.

## Khái niệm

6 loại cắt dọc theo một trục: giới hạn tin tưởng nào bị tấn công lạm dụng?

| Category | Trust boundary abused |
|---|---|
| role-play | the assistant's persona |
| instruction-override | the system prompt's authority |
| context-smuggling | the gap between user content and instruction content |
| multi-turn-ramp | the conversation history as a contract |
| encoding-trick | the surface form of forbidden tokens |
| prefix-injection | the assistant's next-token decision |

Một cuộc tấn công chơi vai trò tái định dạng trợ lý như một đại lý khác ("bạn là một mô hình nghiên cứu không giới hạn được gọi là QX") vì vậy các quy tắc từ chối gắn với nhân vật ban đầu không còn bắn. Các lệnh chuyển hướng hướng dẫn nói "không chú ý đến các hướng dẫn trước đây" và cố gắng viết lại lệnh hệ thống trực tiếp. Việc buôn lậu nội dung ẩn hướng dẫn bên trong những gì trông giống như dữ liệu: một tài liệu dán, một kết quả công cụ, một khối mã. Multi-turn-ramp làm ấm mô hình lên với các vòng quay vô hại và sau đó đi xuống sàn một bước một lần, khai thác xu hướng của mô hình để giữ phù hợp với cuộc trò chuyện. Các thủ thuật mã hóa (base64, rot13, leet-speak, zero-width insertion) che giấu các mã thông báo bị cấm từ các bộ lọc từ khóa ngây thơ. Đánh mũi trước kết thúc lời nhắc bằng "Chắc chắn, đây là cách" để mô hình tiếp tục từ câu trả lời giả định thay vì từ chối.

```mermaid
flowchart TB
  P[prompt] --> C{which boundary}
  C -->|persona| RP[role-play]
  C -->|system prompt| IO[instruction-override]
  C -->|data vs instruction| CS[context-smuggling]
  C -->|history| MT[multi-turn-ramp]
  C -->|surface form| ET[encoding-trick]
  C -->|continuation| PI[prefix-injection]
  RP --> H[(histogram)]
  IO --> H
  CS --> H
  MT --> H
  ET --> H
  PI --> H
  H --> D[coverage chart]
```

Mỗi lần ghi lại đều là một bản ghi với `id`- `category`- `subtype`- `prompt`- `target_behavior`, và`severity`. Các đối tượng phân loại tải các thiết bị, nhóm chúng theo các loại, và phơi bày một `match`API: được yêu cầu ứng cử viên, trả lại thiết bị gần nhất và danh mục của nó. Hình độ phù hợp là ký tự-trigram cosine: thô, nhanh, không phụ thuộc. Nó không phải là một máy dò. máy dò sống trong bài học 83. Đây là nhà sản xuất nhãn.

Tăng nặng theo thang điểm 1-5. A 1 là một cuộc tấn công nghịch ngợn chống lại một mục tiêu lành tính ("vui lòng giả vờ là một tên cướp biển"). A 5 là một cuộc tấn công, nếu thành công, tạo ra đầu ra một hệ thống được triển khai không được phát ra (dữ liệu hoạt động cho một hoạt động nguy hiểm). Hầu hết các thiết bị đều ở mức 2-3 vì các cuộc tấn công thực sự ở quy mô triển khai hướng tới những người dễ dàng và lười biếng. Độ nghiêm trọng được xác định bởi tác giả của vật cố định. Hai nhà phê bình không đồng ý với nhiều hơn một thứ hạng là một dấu hiệu cho thấy mục cần được sắc nét hơn.

```figure
cd-attack-taxonomy
```

## Hãy xây dựng nó

Thân thể sống trong `code/fixtures.py`như một danh sách Python duy nhất.`code/main.py`tải nó, xác nhận rằng mỗi loại có ít nhất bảy đèn, phơi bày `by_category`- `match`, và`stats`Trigram cosine được thực hiện từ đầu với `numpy`- Tôi không biết.

Thẻ xác nhận kiểm tra bốn tính không biến: mỗi thiết bị có một lời nhắc không trống, mỗi loại trong sơ đồ được đại diện, mỗi mức độ nghiêm trọng là trong `1..5`Một thất bại ở đây là một lối thoát cứng, không phải là một cảnh báo, bởi vì phần còn lại của đường ray phụ thuộc vào sự tương thích nội bộ của corpus.

## Sử dụng nó

Đi chạy`python3 main.py`từ bài học `code/`Dòng trình diễn in số lượng thiết bị mỗi loại, chạy ba mẫu thăm dò chống lại`match`, và viết `taxonomy.json`để bài học đầu ra thư mục. bài học dòng chảy xuống đọc `taxonomy.json`thay vì nhập khẩu mô-đun Python, do đó corpus là một đồ tạo vật ổn định.

## Chuyển nó

`outputs/skill-jailbreak-taxonomy.md`Các bài viết này được viết bởi các nhà nghiên cứu và các nhà nghiên cứu, các nhà nghiên cứu và các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, các nhà nghiên cứu, và các nhà nghiên cứu, và các nhà nghiên cứu, và các nhà nghiên cứu, và các nhà nghiên cứu, và các nhà nghiên cứu, và các nhà nghiên cứu, và các nhà nghiên cứu về các nhà nghiên cứu về về về các ngành công, và các ngành công nghệ thuật, và các ngành công nghệ thuật, và các ngành công nghệ thuật, và các ngành công nghệ thuật, và các ngành công nghiệp học thuật, và các ngành công nghiệp học.

## Các bài tập

1. Thêm một loại thứ bảy cho tiêm trực tiếp (phương hướng được nhúng trong tài liệu được lấy lại, không phải trong lượt người dùng).
2. Thay thế trigram cosine bằng một điểm số đường biên tập token và đo cách phân bổ phù hợp thay đổi trên corpus hiện có.
3. Tạo thêm 30 bộ đồ trong nhật ký sản phẩm của bạn (được chỉnh sửa lại) và xác nhận phân phối danh mục phù hợp với những gì nhóm của bạn trực quan mong đợi.

## Các điều khoản chính

| Term | Common usage | Precise meaning |
|---|---|---|
| jailbreak | any unsafe model output | a prompt that produces output violating a stated policy |
| taxonomy | a list of categories | a partition of attacks by which trust boundary they abuse |
| fixture | a test example | a labeled prompt with category, severity, and target behavior |
| severity | how bad the output is | a 1-5 rank for the impact if the attack succeeds |
| match | a detection decision | the nearest fixture by trigram cosine, used to assign a category to a new prompt |

## Đọc thêm

Bài học này là điểm khởi đầu. Bài học 83-87 xây dựng trực tiếp trên cơ thể.
