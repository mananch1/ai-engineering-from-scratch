# Việc chọn các nhóm quan sát LLM

> Thị trường khả năng quan sát năm 2026 chia thành hai loại. Các nền tảng phát triển (LangSmith, Langfuse, Comet Opik) kết hợp giám sát với các đánh giá, quản lý nhanh chóng, lặp lại phiên. Các công cụ Gateway/instrumentation (Helicone, SigNoz, OpenLLMetry, Phoenix) tập trung vào viễn thông. Langfuse là lõi được cấp phép MIT với cân bằng OSS mạnh (50K sự kiện / tháng đám mây miễn phí). Phoenix là OpenTelemetry-native dưới Elastic License 2.0  tuyệt vời cho việc hình dung drift / RAG, không phải là một backend sản xuất bền vững. Arize AX sử dụng tích hợp Iceberg / Parquet không sao chép bằng cách tuyên bố giá rẻ hơn 100 lần so với khả năng quan sát monolithic. LangSmith dẫn đầu cho LangChain / LangGraph, $ 39 / người dùng / tháng, tự lưu trữ trong Enterprise chỉ. Helicone dựa trên proxy với 15-30 phút thiết lập, 100K req / mo miễn phí, nhưng ít sâu hơn trên dấu vết của đại lý. Mô hình sản xuất chung: Gateway (Helicone/Portkey) + nền tảng eval (Phoenix/TruLens) dán bằng OpenTelemetry.

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Mục tiêu học tập

- Sự khác biệt giữa các nền tảng phát triển (được tập hợp: đánh giá + yêu cầu + phiên) và các công cụ gateway/telemetry (chỉ theo dõi + métrics).
- Bản đồ sáu công cụ chính (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) cho các trường hợp cấp phép, giá cả và sử dụng điểm ngọt của họ.
- Giải thích mô hình dán OpenTelemetry cho phép bạn kết hợp công cụ cửa ngõ với nền tảng đánh giá riêng biệt.
- Hãy nêu tên phân biệt chi phí năm 2026 (chương pháp sao chép bằng không của Arize AX so với tiêu thụ đơn phương) và nêu số nhân khoảng 100x.

## Vấn đề

Bạn đã gửi một tính năng LLM. Nó hoạt động. Bạn không có khả năng nhìn thấy các lỗi nhanh, vòng lặp công cụ, hồi quy độ trễ, tăng chi phí hoặc tỷ lệ hit của cache nhanh. Bạn Google "Làm chứng của LLM" và nhận được tám công cụ tất cả tuyên bố họ giải quyết cùng một vấn đề với ba điểm giá khác nhau.

Họ không giải quyết cùng một vấn đề. LangSmith trả lời "tại sao chạy LangGraph này thất bại?" Phoenix trả lời "hợp ống RAG của tôi đang dẫn dắt?" Helicone trả lời "tên nào đang đốt token?" Langfuse trả lời "Tôi có thể tự lưu trữ toàn bộ nó không?"

Việc chọn bao gồm bốn trục: hàng đống (LangChain? SDK nguyên liệu? đa nhà cung cấp?), dung nạp giấy phép (chỉ MIT? Elastic OK? phạt thương mại?), ngân sách (tầng miễn phí? $100/mo? $1000/mo?), và tự chủ (đáng lẽ phải? tốt để có? bao giờ?).

## Khái niệm

### Hai loại

**Development platforms**bạn chạy thí nghiệm, xem prompt nào hoạt động, set dữ liệu trở lại một prompt mới chống lại những người chiến thắng cũ. LangSmith, Langfuse, Comet Opik.

**Gateway/telemetry tools**Các công cụ kết luận gọi  prompt, phản ứng, token, độ trễ, mô hình, chi phí. Helicone, SigNoz, OpenLLMetry, Phoenix. Minimalist. Có thể được kết hợp với một công cụ đánh giá riêng biệt thông qua OpenTelemetry.

### Longfuse  cân bằng OSS

- Core Apache / MIT cấp phép; tự lưu trữ thông qua Docker.
- Tầng mây miễn phí: 50 nghìn sự kiện/tháng.
- Evals, quản lý nhanh chóng, dấu vết, tập hợp dữ liệu, bảo hiểm hợp lý của tất cả bốn tính năng của nền tảng phát triển.
- Điểm ngọt ngào: bạn muốn các tính năng lớp LangSmith nhưng phải tự lưu trữ hoặc ở lại giấy phép OSS.

### Phoenix (Arize)  Telemetry-first, OpenTelemetry-native

- Giấy phép 2.0; tự chủ, tầm thường.
- Tốt nhất trong RAG và hình ảnh drift.
- Không được thiết kế như là hậu trường sản xuất bền vững  chủ yếu là khả năng quan sát trong thời gian phát triển.
- Điểm thích hợp: Phát triển đường ống RAG, điều chỉnh drift, cặp với một cửa cổng riêng cho sản xuất.

### Arize AX  chơi quy mô

- Tiếp thị, tích hợp dữ liệu hồ nước bằng Iceberg/Parquet.
- Thuyết toán: bạn lưu trữ dấu vết trong Parquet của riêng bạn trên S3; Arize đọc trực tiếp.
- Điểm mấu chốt: > 10M dấu vết/ngày, hồ dữ liệu hiện có, muốn bảng điều khiển cụ thể LLM mà không có giá Datadog.

### LangSmith  LangChain/LangGraph đầu tiên

- Tiếp thị, 39 đô la/tháng, tự lưu trữ chỉ trên Enterprise.
- Tốt nhất trong lớp cho các đống LangChain và LangGraph. Nếu bạn không tham gia vào cả hai, nó ít hấp dẫn hơn.
- Địa điểm ngọt ngào: đội ngũ cam kết với LangChain, sẵn sàng trả tiền.

### Helicone  dựa trên proxy tối thiểu khả thi

- 15-30 phút thiết lập bằng cách đổi `OPENAI_API_BASE`cho nhân viên của Helicone.
- MIT cấp phép; 100K req / tháng miễn phí, trả $20 / tháng +.
- Bao gồm failover, cache, giới hạn lãi suất  cũng hoạt động như một cổng thông tin.
- Độ sâu thấp hơn trên các dấu vết đại lý / nhiều bước.
- Sweet spot: khởi động nhanh, ứng dụng đơn, cần gateway + khả năng quan sát trong một.

### Opik (Comet)  nền tảng phát triển OSS

- Apache 2.0, hoàn toàn OSS.
- Một tính năng tương tự như Langfuse với di sản sao chổi.
- Địa điểm ngọt ngào: các nhóm ML đã ở trên Comet, muốn LLM có thể quan sát được trong cùng một bảng.

### SigNoz  OpenTelemetry- đầu tiên APM đầy đủ

- Apache 2.0. xử lý APM chung cộng với LLM thông qua OpenTelemetry.
- Điểm ngọt ngào: khả năng quan sát thống nhất giữa các dịch vụ và các cuộc gọi LLM.

### Lớp dán: OpenTelemetry + GenAI các quy ước ngữ nghĩa

OpenTelemetry đã công bố các công ước ngữ nghĩa GenAI vào cuối năm 2025 (`gen_ai.system`- `gen_ai.request.model`- `gen_ai.usage.input_tokens`Các công cụ tiêu thụ OTel có thể tương tác.

1. Cho phép OTel với các hội nghị GenAI từ mỗi cuộc gọi LLM.
2. Hành trình đến cổng (Helicone / Portkey) cho ngày càng ngày.
3. Dual-ship to eval platform (Phoenix / Langfuse) cho các regressions.
4. Tái lưu trữ trong hồ dữ liệu (Iceberg) để phân tích lâu dài thông qua Arize AX hoặc DuckDB.

### Trạm: dùng dụng cụ ở lớp sai

Các công cụ bên trong khung đại lý của bạn (ví dụ, thêm dấu vết LangSmith) kết nối bạn với khung đó.

### Chọn mẫu, không thể giữ được tất cả.

Với > 1M yêu cầu/ngày, việc giữ lại toàn bộ chi phí cao hơn các cuộc gọi LLM. Dấu mẫu theo quy tắc: 100% lỗi, 100% chi phí cao, 5% thành công.

### Những con số mà bạn nên nhớ

- Lớp đám mây miễn phí Langfuse: 50K sự kiện/tháng.
- LangSmith: 39 đô la/tháng.
- Không sử dụng trực thăng: 100K req/tháng.
- Arize AX tuyên bố: ~ 100 lần rẻ hơn so với monolithic trên quy mô.
- Công ước OpenTelemetry GenAI: 2025 vận chuyển, 2026 được chấp nhận rộng rãi.

```figure
i4-otel-glue
```

## Sử dụng nó

`code/main.py`mô phỏng một ngày theo dõi 1M trên các chiến lược giữ lại (100% tiêu thụ, lấy mẫu, lấy mẫu + lỗi). báo cáo chi phí lưu trữ và những gì bị mất dưới mỗi.

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/skill-observability-stack.md`. Với hàng, quy mô, ngân sách, tư thế giấy phép, chọn công cụ (s).

## Các bài tập

1. Nhóm của anh ở LangChain muốn OSS tự lưu trữ khả năng quan sát.
2. Với 5M tracks/day với Datadog trích dẫn 150K USD/tháng, tính toán break-even cho Arize AX.
3. Thiết kế một thuộc tính OpenTelemetry GenAI đặt hướng dẫn của tổ chức của bạn nên yêu cầu trong mỗi cuộc gọi LLM.
4. Thảo luận liệu Phoenix một mình có đủ cho sản xuất.
5. Helicone là 20ms đại diện trênheadhead. tại P99 TTFT 300ms, đó là chấp nhận được?

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OpenLLMetry | "OTel for LLMs" | Open-source OpenTelemetry instrumentation for LLMs |
| GenAI conventions | "OTel attributes" | Standard OTel attribute names for LLM calls |
| LangSmith | "LangChain observability" | Commercial platform bundled with LangChain ecosystem |
| Langfuse | "OSS LangSmith" | MIT OSS with similar feature set |
| Phoenix | "Arize dev tool" | OpenTelemetry-native dev/eval platform |
| Arize AX | "scale observability" | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | "proxy observability" | HTTP proxy collecting LLM telemetry + gateway features |
| Opik | "Comet LLM" | Apache 2.0 OSS dev platform from Comet |
| Session replay | "trace rerun" | Replay a full agent session with tool calls |
| Eval | "offline test" | Running candidate model/prompt over labeled dataset |

## Đọc thêm

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
