# Tiền lưu trữ nhanh và kinh tế lưu trữ ngữ nghĩa

> **Pricing snapshot dated 2026-04.**Các yêu cầu số dưới đây phản ánh thẻ giá bán hàng được thu thập tại ấn phẩm bài học này; kiểm tra với các tài liệu liên kết trước khi trích dẫn chúng theo dòng chảy.

> L2 ( cấp độ nhà cung cấp) prompt/prefix caching sử dụng lại sự chú ý KV cho các prefix lặp đi lặp lại  Các tài liệu cache prompt của Anthropic quảng cáo giảm chi phí lên đến 90% và giảm độ trễ 85% trên các yêu cầu dài; cho Claude 3.5 Sonnet đọc cache là $0.30/M vs $3,00/M tươi với TTL 5 phút và tiền thưởng viết 2 lần cho tùy chọn TTL 1 giờ (docs.anthropic.com, 2026-04). OpenAI prompt caching áp dụng tự động cho các prompt ≥1024 token và giá nhập cache với giá giảm giá khoảng 90% so với mới (platform.openai.com, 2026-04); tỷ lệ cache chính xác cho mỗi mô hình phụ thuộc vào thẻ tỷ lệ sống. L1 (tầng ứng dụng) lưu trữ ngữ nghĩa bỏ qua LLM hoàn toàn trên nhúng hit tương tự. Nhà cung cấp "95% chính xác" đề cập đến sự phù hợp, không đạt tỷ lệ  tỷ lệ đạt được sản xuất được báo cáo dao động từ 10% (tác thảo mở) đến 70% (FAQ có cấu trúc); không có nhà cung cấp nào xuất bản một đường cơ sở chính thức, vì vậy hãy coi chúng như viễn thông cộng đồng chứ không phải là đảm bảo. Các bẫy sản xuất: song song giết cache (N yêu cầu song song được phát hành trước khi ghi cache đầu tiên có thể làm tăng chi tiêu nhiều lần), và nội dung động bên trong tiền tố ngăn chặn cache tấn công hoàn toàn. ProjectDiscovery báo cáo chuyển từ 7% lên tỷ lệ hit 74% (2025-11) bằng cách di chuyển văn bản động ra khỏi tiền tố có thể lưu trữ.

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Mục tiêu học tập

- Sự phân biệt giữa L2 prompt/prefix caching (kV tái sử dụng tại nhà cung cấp) và L1 semantic caching (LLM bypass trên các prompt tương tự).
- Giải thích Anthropic `cache_control`đánh dấu rõ ràng và hai tùy chọn TTL (5 phút so với 1 giờ) với nhân giá của chúng.
- Xét dự kiến tiết kiệm hàng tháng với tỷ lệ hit, hỗn hợp phản hồi/quick, và giá token.
- Hãy cho tên mẫu chống đối tương đồng làm tăng tỷ lệ tiền bằng 5-10x và mẫu chống đối nội dung động làm giảm tỷ lệ hit.

## Vấn đề

Bạn thêm bộ nhớ cache nhanh vào dịch vụ RAG của bạn. Tài khoản vẫn ổn định. Bạn đo tỷ lệ hit; nó là 7%. Các yêu cầu của bạn trông tĩnh nhưng họ không phải là  yêu cầu hệ thống bao gồm ngày hiện tại định dạng đến phút, một ID yêu cầu, và một ví dụ ngẫu nhiên sắp xếp lại cho sự đa dạng. Mỗi yêu cầu viết một mục bộ nhớ cache mới, đọc bằng không.

Một cách riêng biệt, đại lý của bạn chạy mười cuộc gọi song song với công cụ mỗi câu hỏi của người dùng. Tất cả mười đều đến với nhà cung cấp trước khi ghi nhớ cache đầu tiên hoàn thành. mười viết, không đọc. hóa đơn của bạn là 5-10 lần so với "với ghi nhớ cache" được cho là chi phí.

Caching là một giao thức, không phải là một cờ.

## Khái niệm

### L2  Caching prompt/prefix của nhà cung cấp

Nhà cung cấp lưu trữ sự chú ý KV cho một prefix có thể lưu trữ và sử dụng lại nó trên yêu cầu tiếp theo phù hợp với prefix. Bạn trả chi phí viết một lần, đọc gần như miễn phí.

**Anthropic (Claude 3.5 / 3.7 / 4 series)**: rõ ràng `cache_control`TTL: 5 phút (tài liệu chi phí 1.25x cơ sở) hoặc 1 giờ (tài liệu chi phí 2x cơ sở).$0.30/M on Claude 3.5 Sonnet vs $3,00/M tươi  10 lần rẻ hơn (docs.anthropic.com, từ năm 2026-04). Giá khác nhau cho mỗi mẫu (Opus/Haiku được xuất bản riêng); luôn luôn kiểm tra chéo trang giá trực tiếp.

**OpenAI**: tự động lưu trữ trước khi gửi các thông báo ≥1024 token (platform.openai.com, 2026-04). Không có cờ rõ ràng. Cài nhập trước khi gửi là khoảng 10 lần rẻ hơn so với mới trên thẻ tốc độ gpt-4o / gpt-5 hiện tại. Cả các tài liệu và các bản ghi bản phát hành đều không công bố một đường cơ sở chính thức về tỷ lệ hit; báo cáo cộng đồng tập hợp khoảng 3060% với thiết kế nhanh chóng cẩn thận.`usage.cached_tokens`để đo lường của riêng bạn.

**Google (Gemini)**: cache ngữ cảnh thông qua API rõ ràng; 1M-token context có nghĩa là cache trả tiền nhiều hơn nữa.

**Self-hosted (vLLM, SGLang)**: Giai đoạn 17 · 06 bao gồm RadixAttention  mô hình tương tự trong tính toán của bạn.

### L1  Caching ngữ nghĩa cấp ứng dụng

Trước khi gọi LLM, hãy chọn hash prompt, nhúng nó và tìm kiếm yêu cầu được lưu trữ trong cache tương tự (sự tương đồng đồng đồng tính trên ngưỡng, thường là 0,95+).

Mã nguồn mở: Redis Vector Similarity, GPTCache, Qdrant. Tiếp thị: Portkey Cache, Helicone Cache.

Các yêu cầu chính xác của nhà cung cấp đề cập đến việc trả lời được lưu trữ trong cache trở lại thường xuyên như thế nào là phù hợp theo nghĩa ngữ  chứ không phải là bạn đánh số thường xuyên.

- - Tự động trò chuyện: 10-15%.
- Các câu hỏi thường gặp / hỗ trợ có cấu trúc: 40-70%.
- Các câu hỏi mã: 20-30% (những biến thể nhỏ giết chết các lượt truy cập).
- Các đại lý giọng nói lặp lại các lời nhắc: 50-80% (định dạng bình thường giọng nói cố định).

### Phương pháp chống đồng đều hóa

Trưởng lý của bạn thực hiện 10 cuộc gọi công cụ song song. Tất cả 10 đều có cùng một lệnh báo 4K-token hệ thống. Anthropic cache viết là theo yêu cầu; đầu tiên cache-tập hoàn thành khoảng 300 ms sau khi nhà cung cấp thấy lệnh báo. Các yêu cầu 2-10 đến trong cùng một cửa sổ millisecond và mỗi thấy cache bị bỏ lỡ. Bạn trả 10 tiền thưởng viết, 0 giảm giá đọc.

Xác định: hàng với thứ tự đầu tiên  thực hiện yêu cầu 1 một mình, sau đó phát 2-10 sau khi bộ nhớ nhớ bộ nhớ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ nhớ bộ

### Các nội dung động chống mô hình

Đơn vị hệ thống của bạn trông như:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Mỗi yêu cầu đều độc đáo, mỗi yêu cầu đều có điểm số không truy cập.

Xác định: di chuyển mọi thứ thực sự tĩnh vào tiền tố có thể lưu trữ; thêm nội dung động sau ranh giới lưu trữ:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery chuyển từ 7% lên 74% tỷ lệ hit cache theo cách này và xuất bản giải phẫu.

### Lưu trữ hàng loạt + cache cho tải trọng làm việc qua đêm

Các API hàng loạt (Phase 17 · 15) cung cấp giảm 50% khi quay 24 giờ. Tiền nhập được lưu trữ trên cùng cung cấp cho bạn ~ 10 lần trên cùng. Nhiệm vụ phân loại, dán nhãn và sản xuất báo cáo qua đêm có thể giảm xuống ~ 10% chi phí không được xếp chồng đồng bộ bằng cách xếp chồng.

### Những con số mà bạn nên nhớ

Điểm giá được ghi lại 2026-04 từ các tài liệu nhà cung cấp liên kết và trôi qua mỗi vài tháng  kiểm tra lại trước khi dựa vào chúng.

- Anthropic được lưu trữ đọc: $0.30/M trên Claude 3.5 Sonnet, khoảng 10 lần rẻ hơn so với đầu vào mới (docs.anthropic.com).
- Antropic cache write premium: 1.25x (5-min TTL) hoặc 2x (1-hour TTL).
- OpenAI tự động lưu trữ: áp dụng cho các mã thông báo ≥1024 mã thông báo; nhập khẩu được lưu trữ trong cache với giá khoảng 10% của nhập khẩu mới trên thẻ giá hiện tại (platform.openai.com).
- Tỷ lệ hit cache ngữ nghĩa (được báo cáo bởi cộng đồng): ~ 10% mở trò chuyện; lên đến ~ 70% FAQ có cấu trúc. Không có đường cơ sở được cung cấp bằng văn bản.
- ProjectDiscovery: 7% → 74% tỷ lệ hit bằng cách di chuyển động từ tiền tố (blog dự án, 2025-11).
- Phương pháp chống đồng bộ hóa: báo cáo điển hình về lạm phát hóa đơn 510x khi các yêu cầu song song N bỏ lỡ ghi nhớ cache đầu tiên.

```figure
semantic-cache-hit
```

## Sử dụng nó

`code/main.py`mô phỏng L1 + L2 lưu trữ trên tải trọng công việc hỗn hợp. báo cáo đánh giá, hóa đơn, và cho thấy hình phạt song song.

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/skill-cache-auditor.md`Với mô hình và lưu lượng truy cập nhanh chóng, kiểm toán khả năng lưu trữ và khuyến cáo tái cấu trúc.

## Các bài tập

1. Đi chạy`code/main.py`- Đổi cờ tương đồng.
2. Đặt ngày cho hệ thống của bạn, hãy di chuyển nó ra.
3. Xét điểm hòa cho 1 giờ TTL (2x viết) so với 5 phút TTL (1.25x viết) với tốc độ đến yêu cầu của bạn.
4. Cache ngữ nghĩa ở ngưỡng 0,95 đạt 20%. ở mức 0,85 đạt 50% nhưng bạn thấy các phản ứng được lưu trữ trong cache không chính xác.
5. Bạn xếp hàng 10 câu hỏi phụ song song cho mỗi câu hỏi của người dùng.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | Provider stores KV for repeated prefix |
| `cache_control` | "Anthropic cache marker" | Explicit attribute marking cacheable blocks |
| Cache write premium | "write tax" | Extra cost for first miss-to-cache (1.25x or 2x) |
| L1 semantic cache | "embedding cache" | App-level hash-and-embed before calling LLM |
| GPTCache | "LLM caching lib" | Popular OSS L1 cache library |
| Cache hit rate | "hits / total" | Fraction of requests served from cache |
| Parallelization anti-pattern | "the N-write trap" | N parallel requests miss cache N times |
| Dynamic content trap | "the time-in-prompt trap" | Dynamic bytes in prefix kill hit rate |
| RadixAttention | "intra-replica cache" | SGLang's prefix-cache implementation |

## Đọc thêm

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) chính thức `cache_control`ngữ nghĩa và TTL.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) Hành vi lưu trữ tự động và đủ điều kiện.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
