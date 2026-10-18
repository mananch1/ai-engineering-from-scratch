# Các API hàng loạt  giảm giá 50% như tiêu chuẩn ngành

> Mỗi nhà cung cấp lớn đều gửi một API lô async với giảm 50% và quay vòng 24 giờ. OpenAI, Anthropic, Google và hầu hết các nền tảng suy luận (Tầng hàng loạt Fireworks, cùng nhau) thực hiện cùng một mô hình. Các lô hàng với lưu trữ dự trữ nhanh chóng và đường ống qua đêm giảm xuống còn ~ 10% chi phí đồng bộ không lưu trữ. Quy tắc rất đơn giản: nếu nó không tương tác, nó thuộc về hàng loạt. Các đường ống tạo nội dung, phân loại tài liệu, khai thác dữ liệu, tạo báo cáo, gắn nhãn hàng loạt, gắn thẻ danh mục  bất cứ thứ gì dung nạp thời gian trễ 24 giờ là tiền còn lại trên bàn cho đến khi nó chuyển sang hàng loạt. Mô hình sản xuất năm 2026 là phân loại mỗi khối lượng công việc LLM mới thành ba đường dây: tương tác (tự đồng bộ với lưu trữ cache), bán tương tác (trung hàng không đồng bộ với fallback), hàng (trong đêm, đầu vào lưu trữ cache xếp chồng). Những khối lượng công việc giả vờ là tương tác nhưng dung nạp phút trễ lãng phí nhiều nhất.

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## Mục tiêu học tập

- Tên gọi ba API hàng cung cấp (OpenAI, Anthropic, Google) và giảm giá 50% + bảo đảm hoàn trả 24h chung.
- Xét chi phí xếp chồng lên hàng + đầu vào cache trên khối lượng công việc phân loại qua đêm và so sánh với đường cơ sở không được lưu trữ đồng bộ.
- Triage một khối lượng làm việc thành loạt tương tác / bán tương tác / và biện minh đường.
- Tên hai cái bẫy: tương tác một phần (người dùng mong đợi nhanh hơn 24h) và trôi động sơ đồ đầu ra (tình thức tập tin lô khác nhau cho mỗi nhà cung cấp).

## Vấn đề

Nhóm của bạn gửi một hệ thống sản xuất báo cáo hàng đêm. 50.000 tài liệu, tóm tắt mỗi tài liệu, tập hợp các bản tóm tắt, biên soạn một bản báo cáo điều hành. Tiếp tục đồng bộ mất 4 giờ với giá 2.000 đô la/ngày. Bạn nghe về các API hàng loạt.

Các gói này cung cấp 50% giảm giá. Bạn cũng có thể bật cache nhanh trên hệ thống nhắc (cùng với tất cả 50k cuộc gọi).

Batch là đòn bẩy rẻ nhất trong bộ công cụ chi phí LLM mà không ai kéo ra. Lý do chủ yếu là tổ chức: các nhóm nghĩ "quá thời gian thực" khi SLA thực sự là "tối sáng". Bài học này là về việc không để lại 90% hóa đơn trên bàn.

## Khái niệm

### 3 bộ API

**OpenAI Batch API**: JSONL file upload với một danh sách các yêu cầu. hứa hẹn 24 giờ quay (thường là ~ 2-8 giờ trong thực tế). 50% giảm giá trên các token nhập và ra. `/v1/batches`Endpoint. Các đầu vào có thể lưu trữ được cũng được đặt giá vào bộ nhớ nhớ trước.

**Anthropic Message Batches**: JSONL tải lên, 24 giờ quay lại, giảm 50%.`cache_control` cache viết là rõ ràng, đọc xảy ra tự động trong lô.

**Google Vertex AI Batch Prediction**: BigQuery hoặc đầu vào GCS. Giảm giá 50% tương tự cho Gemini.

### Hình ngữ: không đồng bộ, không chậm

Bộ hàng là "Tôi hứa sẽ quay lại trong 24 giờ"  không phải "thầy sẽ mất 24 giờ". P50 điển hình là 2-6 giờ. Nhà cung cấp lập lịch hàng của bạn trong các cửa sổ ngoài đỉnh khi tồn kho GPU là thiếu sử dụng.

### Đứng đống với cache

Một bản tóm tắt tài liệu 50k với cùng một hệ thống mã thông báo 4K:

- Không lưu trữ đồng bộ: 50000 × ($input × 4000 + $sản lượng x 200) với tốc độ đầy đủ.
- Simchronous cached: system prompt cached sau khi viết đầu tiên; còn lại 49999 nhận được đầu vào 10x rẻ hơn.
- Nhóm lưu trữ: tất cả các điều trên cộng với giảm 50% cho cả đọc và viết.

Các hàng: batch + cache = ~ 10% của đồng bộ hóa hóa đơn không lưu trữ. Bất kỳ tải trọng công việc nào chạy qua đêm và có một hệ thống chia sẻ yêu cầu nên sử dụng điều này.

### Phân loại tải trọng công việc

**Interactive** người dùng chờ phản hồi. TTFT quan trọng. cuộc gọi đồng bộ với cache nhanh chóng. Không thể hàng.

**Semi-interactive** người dùng gửi một nhiệm vụ, kiểm tra lại trong vài phút. Thống kê đồng bộ với fallback để đồng bộ hóa nếu batch không có sẵn. Hãy nghĩ về chỉ mục RAG khối lượng vừa phải.

**Batch** người dùng mong đợi kết quả "tối sáng" hoặc "giờ tới". Các đường ống nội dung, phân loại trên quy mô, phân tích ngoại tuyến.

Sai lầm phổ biến: phân loại mọi thứ như tương tác vì đường ống là sản xuất. sản xuất không phải là một đặc điểm độ trễ  SLA là.

### Trầm lẫy tương tác một phần

Một số tính năng trông tương tác nhưng dung nạp 5-10 phút. Ví dụ: báo cáo sức khỏe của khách hàng hàng hàng đêm với nút "phải mới". Người dùng nhấp vào refresh; chờ 10 phút là ổn. Nhóm gửi nó như đồng bộ. 50 refresh đồng thời chi phí 10 lần so với hàng và giao hàng qua email sẽ chi phí.

Câu hỏi đặt ra: "24 giờ có nghĩa gì với người dùng này?" Nếu câu trả lời là "họ sẽ không nhận ra", hãy đính kèm.

### Trạm đường dẫn

Các định dạng tập tin hàng loạt khác nhau cho mỗi nhà cung cấp:

- OpenAI: JSONL, một yêu cầu cho mỗi dòng.
- Anthropic: JSONL, một tin nhắn mỗi dòng; định dạng phản hồi được nhúng.
- Vertex: bảng BigQuery hoặc tiền đề GCS với TFRecord.

Viết "một khách hàng hàng hàng loạt" trên các nhà cung cấp có nghĩa là mã bộ chuyển đổi cho mỗi nhà cung cấp. Các cửa khẩu quảng cáo hàng loạt nhiều nhà cung cấp (Portkey, LiteLLM một số cấp) vẫn gói mỏng định dạng nguyên thô.

### Những con số mà bạn nên nhớ

- Giảm giá hàng loạt giữa các nhà cung cấp: 50% cố định trên đầu vào + đầu ra.
- SLA quay trở lại: 24 giờ đảm bảo, 2-6 giờ điển hình P50.
- Các lô hàng được xếp chồng + đầu vào được lưu trữ trong cache: ~ 10% chi phí đồng bộ hóa không lưu trữ.
- Quy tắc phân loại tải trọng làm việc: nếu thời gian trễ 24h được chấp nhận, luôn luôn đợt.

```figure
batch-lane-triage
```

## Sử dụng nó

`code/main.py`tính toán chi phí trên đồng bộ, đồng bộ sáp + cache, lô, và lô + cache cho khối lượng công việc tài liệu 50k. báo cáo tiết kiệm bằng $ và phần trăm.

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/skill-batch-triager.md`Với các đặc điểm khối lượng công việc, phân loại thành tương tác/ bán/chọi và ước tính tiết kiệm.

## Các bài tập

1. Đi chạy`code/main.py`. Đối với một đường ống 100k-doc với hệ thống 3K-token prompt và 500-token đầu ra, tính toán tiết kiệm của đầy đủ hàng (batch + cache) so với đồng bộ hóa cơ sở.
2. Chọn ba tính năng trong một sản phẩm thực sự bạn biết. Tria mỗi trong tương tác / bán / lô.
3. Một người dùng phàn nàn báo cáo của họ mất 3 giờ. Đó là một sai lệch phân tích hàng hoặc một tương tác hợp pháp?
4. SLA trả lại API của lô của bạn là 24h nhưng P99 là 20h. Làm thế nào để truyền đạt điều này cho người dùng  hành vi hệ thống dòng chảy xuống trên trường hợp cạnh là gì?
5. Lập toán: ở độ dài nào của tiền đề chia sẻ batch + cache trở nên rẻ hơn so với chạy qua đêm trên GPU riêng của bạn?

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Batch API | "async discount" | 50% off with 24h turnaround |
| JSONL | "batch format" | One JSON request per line; OpenAI/Anthropic standard |
| Message Batches | "Anthropic batch" | Anthropic's batch API product name |
| Batch prediction | "Vertex batch" | Vertex AI's batch API product |
| Turnaround SLA | "24h promise" | Guarantee, not typical; typical is 2-6h |
| Workload triage | "interactivity decision" | Interactive / semi / batch routing decision |
| Output schema | "response format" | Per-provider JSONL layout; not portable |
| Stacked discount | "batch + cache" | ~10% of uncached sync bill when both apply |

## Đọc thêm

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) định dạng JSONL và `/v1/batches`ngữ nghĩa.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) định dạng lô và `cache_control`tương tác.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) Biểu ngữ của nhóm Gemini.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
