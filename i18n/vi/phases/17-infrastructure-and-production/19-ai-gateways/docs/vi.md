# AI Gateways  LiteLLM, Portkey, Kong AI Gateway, Bifrost

> Một cửa ngõ nằm giữa các ứng dụng của bạn và các nhà cung cấp mô hình. Các tính năng cốt lõi là định tuyến nhà cung cấp, trở lại, thử lại, giới hạn tốc độ, tham chiếu bí mật, khả năng quan sát, đường dây bảo vệ. Thị trường chia rẽ vào năm 2026: **LiteLLM**là MIT OSS với hơn 100 nhà cung cấp, tương thích với OpenAI, nhưng phá vỡ khoảng ~ 2000 RPS (8 GB bộ nhớ, thất bại hàng loạt trong các tiêu chuẩn được xuất bản); tốt nhất cho Python, <500 RPS, phát triển / tạo mẫu. **Portkey**được đặt trên máy điều khiển (các màn hình bảo vệ, biên soạn PII, phát hiện jailbreak, đường viếng kiểm tra), đã Apache 2.0 mã nguồn mở vào tháng 3 năm 2026, 20-40 ms latency overhead,$49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $Giá 100/mô hình/tháng (tối đa 5 trên cấp Plus); phù hợp với doanh nghiệp nếu bạn đã sử dụng Kong. **Bifrost**(Maxim AI)  tự động thử lại với cấu hình backoff, trở lại Anthropic trên OpenAI 429. **Cloudflare / Vercel AI Gateways** quản lý, không hoạt động, thử lại cơ bản. Data Residency thúc đẩy quyết định tự chủ; Portkey và Kong ngồi giữa với OSS + tùy chọn quản lý.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## Mục tiêu học tập

- Đăng danh sáu tính năng cổng cổng cổng cổng cốt lõi (các định tuyến, quay lại, thử lại, giới hạn tốc độ, bí mật, khả năng quan sát, hàng rào).
- Bản đồ bốn cửa ngõ 2026 (LiteLLM, Portkey, Kong AI, Bifrost) để mở rộng các trần và trường hợp sử dụng.
- Hãy trích dẫn chỉ số chuẩn Kong (228% so với Portkey, 859% so với LiteLLM) và giải thích tại sao nó quan trọng đối với > 500 RPS.
- Chọn tự lưu trữ vs quản lý với số liệu cư trú và ngân sách hoạt động.

## Vấn đề

Sản phẩm của bạn gọi là OpenAI, Anthropic và một Llama tự lưu trữ. Mỗi nhà cung cấp có SDK, mô hình lỗi, giới hạn tỷ lệ và chương trình auth khác nhau. Bạn muốn lỗi (nếu OpenAI 429, thử Anthropic), một cửa hàng tín chỉ duy nhất, khả năng quan sát thống nhất và giới hạn tỷ lệ cho mỗi thuê nhà.

Việc tái tạo điều này tại lớp ứng dụng kết hợp mọi dịch vụ với mỗi nhà cung cấp. Một lớp cổng kết hợp nó thành một quy trình với một API (thường tương thích với OpenAI) mà truyền tải đến các nhà cung cấp.

## Khái niệm

### 6 tính năng cốt lõi

1. **Provider routing** OpenAI, Anthropic, Gemini, tự lưu trữ, vv. đằng sau một API.
2. **Fallback** trên 429, 5xx, hoặc thất bại chất lượng, thử lại ở nơi khác.
3. **Retries** Vị trục trặc, cố gắng hạn chế.
4. **Rate limits** mỗi người thuê, mỗi người dùng, mỗi người mẫu.
5. **Secret references** rút thông tin tín dụng từ kho trong thời gian chạy (không bao giờ trong ứng dụng).
6. **Observability** Các thuộc tính OTel + GenAI (Phase 17 · 13) + tính phí.
7. **Guardrails** Phóng thông tin thông tin thông tin, phát hiện jailbreak, bộ lọc các chủ đề được phép.

### LiteLLM  MIT OSS, Python

- 100+ nhà cung cấp, tương thích với OpenAI, cấu hình router, sự suy giảm, khả năng quan sát cơ bản.
- Hạn nứt khoảng 2000 RPS trong chuẩn mực của Kong; 8 GB bộ nhớ, thất bại hàng loạt dưới tải liên tục.
- Ứng dụng Python, < 500 RPS, dev/staging gateway, định tuyến thử nghiệm.
- Chi phí: $ 0 cho OSS; lớp miễn phí đám mây tồn tại.

### Portkey  định vị máy bay điều khiển

- Apache 2.0 OSS tính từ tháng 3 năm 2026. Guardrails, PII redaction, jailbreak detection, audit trails.
- 20-40 ms mỗi yêu cầu thời gian trễ.
- $49/mo cho cấp sản xuất với lưu giữ + SLA.
- Khả năng phù hợp nhất: các ngành công nghiệp được quy định cần bao phủ + khả năng quan sát được kết hợp.

### Kong AI Gateway  chơi quy mô

- Được xây dựng trên Kong Gateway (thế phẩm Gateway API trưởng thành, lua+OpenResty).
- Chỉ số chuẩn của Kong trên tương đương 12 CPU: 228% nhanh hơn Portkey, 859% nhanh hơn LiteLLM.
- Giá: 100 đô la/mô hình/tháng, tối đa 5 trên Plus tier.
- Tích hợp tốt nhất: đã có Kong; > 1000 RPS; sẵn sàng cấp phép.

### Bifrost (Maximum AI)

- Lần thử lại tự động với backoff có thể cấu hình.
- Fallback to Anthropic on OpenAI 429 là một công thức truyền thống.
- Người mới đến, thương mại.

### Cloudflare AI Gateway / Vercel AI Gateway

- Được quản lý, không hoạt động, thử lại và có thể quan sát được.
- Tích hợp tốt nhất: Các ứng dụng JavaScript phục vụ Edge trên Cloudflare / Vercel.
- Giới hạn so với Kong/Portkey trên đường dây và giới hạn tốc độ.

### Tự lưu trữ vs quản lý

Data residency là chức năng buộc. Chăm sóc sức khỏe và tài chính tự chủ mặc định (LiteLLM hoặc Portkey OSS hoặc Kong).

### Ngân sách thời gian trễ

- LiteLLM: 5-15 ms tiêu thụ chung điển hình.
- Portkey: 20-40 ms trên cao.
- - 3-8 ms trên cao.
- Cloudflare/Vercel: 1-3 ms overhead (lợi thế cạnh).

Thời gian trễ Gateway trực tiếp thêm vào TTFT. Đối với TTFT P99 < 100 ms SLA, Kong hoặc Cloudflare. Đối với P99 < 500 ms, bất kỳ.

### Thuật ngữ giới hạn tỷ lệ

Simple token-bucket hoạt động ở quy mô vừa phải. Multi-tenant đòi hỏi cửa sổ trượt + allowance bùng nổ + tiering cho mỗi người thuê. LiteLLM vận chuyển token-bucket; Kong vận chuyển cửa sổ trượt; Portkey vận chuyển cấp.

### Gateway + khả năng quan sát + định tuyến kết hợp

Giai đoạn 17 · 13 (sự quan sát) + 16 (sự định tuyến mô hình) + 19 (cổng) là cùng một lớp trong sản xuất. Chọn một công cụ bao gồm cả ba hoặc dây chúng cẩn thận: hầu hết các triển khai 2026 kết hợp Helicone (sự quan sát) hoặc Portkey (các cửa sổ) với Kong (scale) cho vai trò chia rẽ.

### Những con số mà bạn nên nhớ

- LiteLLM: phá vỡ ở khoảng 2000 RPS, bộ nhớ 8 GB.
- Portkey: 20-40 ms overhead; Apache 2.0 từ tháng 3 năm 2026.
- Kong: 228% nhanh hơn Portkey, 859% nhanh hơn LiteLLM.
- Giá Kong: 100 đô la/mô hình/tháng, tối đa 5 đô la trên Plus tier.
- Cloudflare/Vercel: 1-3 ms trên đầu ở cạnh.

```figure
mx-gateway-fallback
```

## Sử dụng nó

`code/main.py`mô phỏng định tuyến đường cổng với sự lùi ngược trên 3 nhà cung cấp dưới sự tiêm 429/5xx. báo cáo độ trễ, tỷ lệ thử lại và tỷ lệ hit sự lùi ngược.

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/skill-gateway-picker.md`Với quy mô, tư thế hoạt động, tuân thủ, ngân sách thời gian trễ, chọn một cửa ngõ.

## Các bài tập

1. Đi chạy`code/main.py`. Định cấu hình fallback từ OpenAI→Anthropic→ tự lưu trữ.
2. SLA của bạn là TTFT P99 < 200 ms trên đường cơ sở 300 ms.
3. Một khách hàng chăm sóc sức khỏe cần tự lưu trữ + biên soạn PII + kiểm toán.
4. So sánh LiteLLM vs Kong: ở mức tối đa RPS nào một đội nên di chuyển?
5. Thiết kế một chính sách giới hạn lãi suất cho một SaaS đa thuê: cấp độ miễn phí, cấp độ thử nghiệm, cấp độ trả tiền.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | Process sitting between apps and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated log" | Immutable record of every LLM call |
| Token-bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding-window | "precise rate limit" | Time-windowed rate limiter; better fairness |

## Đọc thêm

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
