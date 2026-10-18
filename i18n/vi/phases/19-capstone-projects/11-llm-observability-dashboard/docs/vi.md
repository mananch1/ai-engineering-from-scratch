# Capstone 11  LLM Observability & Eval Dashboard

> Langfuse đã mở lõi. Arize Phoenix đã công bố bản đồ semconv GenAI năm 2026. Helicone và Braintrust đều tăng gấp đôi về chi phí cho mỗi người dùng. OpenLLMetry của Traceloop trở thành công cụ SDK thực tế. Hình thức sản xuất là ClickHouse cho các dấu vết, Postgres cho metadata, Next.js cho UI, và một đội ngũ nhỏ các công việc đánh giá (DeepEval, RAGAS, LLM-đánh án) chạy qua các dấu vết lấy mẫu. Xây dựng một máy chủ tự lưu trữ, tiêu thụ từ ít nhất bốn gia đình SDK, và chứng minh bắt được sự lùi lại tiêm trong chưa đầy năm phút.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:**P11 · P13 · P17 · P18
**Time:** 25 hours

## Vấn đề

Mỗi nhóm AI điều hành giao thông sản xuất vào năm 2026 giữ một trình độ quan sát bên cạnh mô hình. Thu nhập chi phí. Khám phá ảo giác. Chống dõi lưu động. Đáp ra báo hiệu jailbreak. Các bảng điều khiển SLO. Thông tin tin rò rỉ. Các tham chiếu nguồn mở  Langfuse, Phoenix, OpenLLMetry  hội tụ vào các quy ước ngữ nghĩa OpenTelemetry GenAI như là sơ đồ hấp thụ. Bây giờ bạn có thể công cụ OpenAI, Anthropic, Google, LangChain, LlamaIndex, và vLLM với một SDK và vận chuyển các khoảng thời gian tương thích.

Bạn sẽ xây dựng một bảng điều khiển tự lưu trữ mà hấp thụ từ ít nhất bốn gia đình SDK, chạy một bộ nhỏ các công việc đánh giá trên các dấu vết lấy mẫu, phát hiện drift và cảnh báo.

## Khái niệm

Ingest là OTLP HTTP. SDK tạo ra các khoảng GenAI-semconv: `gen_ai.system`- `gen_ai.request.model`- `gen_ai.usage.input_tokens`- `gen_ai.response.id`- `llm.prompts`- `llm.completions`. Thâm nhập vào ClickHouse để phân tích cột; metadata (người dùng, phiên, ứng dụng) rơi vào Postgres.

Evals chạy như các công việc hàng loạt trên các dấu vết được lấy mẫu. DeepEval ghi điểm độ trung thành, độc tính và liên quan đến câu trả lời. RAGAS ghi điểm các métrics lấy lại khi dấu vết mang theo bối cảnh lấy lại. Các thẩm phán LLM tùy chỉnh chạy kiểm tra cụ thể về miền (poll leak, phản ứng ngoài chính sách).

Drift detection đồng hồ phân phối không gian nhúng theo thời gian (PSI hoặc KL khác biệt trên nhúng nhanh) cộng với xu hướng đánh giá điểm. Các cảnh báo cung cấp Prometheus Alertmanager và sau đó Slack / PagerDuty.

## Kiến trúc

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Thống

- Ingest: OpenTelemetry SDK + GenAI các quy ước ngữ nghĩa; OTLP HTTP vận chuyển
- Bộ sưu tập: Bộ sưu tập điện tử mở với bộ xử lý lấy mẫu đuôi (để kiểm soát chi phí)
- lưu trữ: ClickHouse cho các khoảng thời gian, Postgres cho các metadata, S3 cho lưu trữ sự kiện nguyên thô
- Evals: DeepEval, RAGAS 0.2, Arize Phoenix đánh giá gói, tùy chỉnh LLM- thẩm phán
- Drift: PSI / KL trên các tích hợp nhanh chóng (phản dịch-giới chuyển) hàng tuần
- Alert: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- SDK được hỗ trợ ngoài hộp: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## Hãy xây dựng nó

1. **Collector config.**OpenTelemetry Collector với máy thu nhận HTTP OTLP, một mẫu đuôi giữ 100% dấu vết sai và 10% thành công, và xuất khẩu đến ClickHouse và S3.

2. **ClickHouse schema.**Bảng `spans`với cột phản ánh GenAI semconv: `gen_ai_system`- `gen_ai_request_model`- `input_tokens`- `output_tokens`- `latency_ms`- `prompt_hash`- `trace_id`- `parent_span_id`, cộng với túi JSON cho tải trọng hữu ích dài. Thêm chỉ số thứ cấp theo user_id và app_id.

3. **SDK coverage test.**Viết một ứng dụng khách hàng nhỏ sử dụng mỗi SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) với công cụ tự động OpenLLMetry.

4. **Eval jobs.**Một công việc được lên lịch đọc các dấu vết lấy mẫu trong 15 phút cuối cùng và chạy độ trung thành, độc tính và liên quan đến câu trả lời của DeepEval.

5. **Custom LLM-judge.**Một thẩm phán rò rỉ PII: khi nhận được câu trả lời, gọi cho một giám sát LLM để đánh giá khả năng rò rỉ PII.

6. **Drift detection.**Công việc hàng tuần tính toán PSI giữa các tích hợp nhanh chóng của tuần này và đường cơ sở 4 tuần sau đó.

7. **Dashboard.**Next.js 15 với các trang: tổng quan (span/sec, cost/user, p95 latency), dấu vết (hướng tìm + thác nước), đánh giá (khối cảnh trung thành, độc tính), lưu động (PSI theo thời gian), cảnh báo.

8. **Alerting chain.**Prometheus xuất khẩu đọc tổng số điểm đánh giá và phần trăm độ trễ; Alertmanager hướng đến Slack để cảnh báo và PagerDuty cho vi phạm quan trọng.

9. **Regression probe.**Nhổ một lỗi: chatbot được đánh giá bắt đầu rò rỉ SSN giả 1% thời gian. đo MTTR: từ lỗi được triển khai đến cảnh báo Slack.

## Sử dụng nó

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Chuyển nó

`outputs/skill-llm-observability.md`Với ứng dụng LLM, bảng điều khiển thu thập dấu vết của nó, chạy đánh giá, cảnh báo về drift và hiển thị phân bố chi phí / người dùng trong Next.js.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## Các bài tập

1. Thêm công cụ tùy chỉnh cho khung Haystack. Kiểm tra phạm vi cânonic hạ cánh trong ClickHouse với faithful `gen_ai.*`thuộc tính.

2. Thay đổi DeepEval cho các nhà đánh giá Phoenix trên cùng một dấu vết.

3. Nhạc máy dò dẫn dốc: tính toán PSI theo ID ứng dụng thay vì toàn cầu.

4. Thêm một trang " tác động người dùng": chi phí cho người dùng và tỷ lệ thất bại cho người dùng với các dòng phát sáng.

5. Xây dựng một chính sách lấy mẫu đuôi giữ 100% các dấu vết với độc tính > 0,5 cộng với một mẫu phân cấp 10% của phần còn lại.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## Đọc thêm

- [Langfuse](https://github.com/langfuse/langfuse) nền tảng quan sát mở lõi tham chiếu
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) tham chiếu thay thế với hỗ trợ xoay mạnh
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) gia đình SDK tự động dụng cụ
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) quy trình tiêu thụ
- [Helicone](https://www.helicone.ai) khả năng quan sát được lưu trữ thay thế
- [Braintrust](https://www.braintrust.dev) nền tảng đánh giá đầu tiên thay thế
- [ClickHouse documentation](https://clickhouse.com/docs) cửa hàng span cột
- [DeepEval](https://github.com/confident-ai/deepeval) thư viện đánh giá
