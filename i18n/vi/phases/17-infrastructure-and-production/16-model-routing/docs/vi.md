# Mô hình định tuyến như một nguyên thủy giảm chi phí

> Một nhà môi giới năng động đánh giá mọi yêu cầu (loại nhiệm vụ, chiều dài token, nhúng tương tự, sự tự tin) và gửi các yêu cầu đơn giản đến một mô hình rẻ tiền, leo thang những yêu cầu phức tạp đến một mô hình biên giới. Cũng gọi là mẫu vỏ. Các nghiên cứu trường hợp sản xuất cho thấy giảm chi phí từ 20-60% ở chất lượng iso trên các triển khai của Mỹ / Anh / EU; cải thiện hiệu quả định tuyến 30% trên SaaS khối lượng cao chuyển thành tiết kiệm hàng năm sáu chữ số. Tầm quan điểm của năm 2026 là giá suy luận LLM giảm ~ 10 lần mỗi năm  một token lớp GPT-4 đã đi từ $20/M to ~$0,40/M từ cuối năm 2022 đến năm 2026. Phần lớn sự giảm là phục vụ tốt hơn các đống (Phase 17 · 04-09), chứ không phải phần cứng. Đường dẫn là cách bạn chuyển đổi giá giảm thành biên mà không có sự lùi sản phẩm. Phương thức thất bại là biến động mô hình rẻ tiền: tuyến đường đẩy 40% lên mô hình yếu hơn, chất lượng giảm 3-5% đối với các nhiệm vụ lý luận, không ai nhận ra cho một phần tư. Các tuyến đường Gate theo các métrics chất lượng trực tuyến, không chỉ là các thiết lập đánh giá ngoại tuyến.

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 minutes

## Mục tiêu học tập

- Giải thích mô hình hàng loạt: rẻ tiền đầu tiên với kiểm tra sự tin tưởng, leo thang trên sự tin tưởng thấp.
- Đếm theo bốn tín hiệu định tuyến (tỷ lệ phân loại nhiệm vụ, độ dài nhanh, nhúng tương tự với bộ cứng được biết, tự tin từ lần qua đầu tiên).
- Xét chi phí hỗn hợp dự kiến tại mục tiêu định tuyến chia và dung nạp mất chất lượng.
- Hãy cho biết số liệu giám sát drift (cổng chất lượng trực tuyến) bắt được những người lăn lăn rẻ tiền.

## Vấn đề

Dịch vụ của bạn chi phí 80k USD/tháng trên GPT-5. phân tích của bạn cho thấy 70% các câu hỏi là đơn giản: "Thời gian là bao nhiêu giờ ở Paris?" "được định nghĩa lại câu này". Một mô hình lớp Haiku xử lý hoàn hảo với 3% chi phí. 30% cần lý luận của GPT-5  mã hóa, toán học, lập kế hoạch đa bước.

Nếu bạn chuyển 70% sang rẻ và 30% sang đắt tiền, hóa đơn của bạn giảm khoảng 65% với chất lượng sản phẩm tương tự. Đây là chuyển hướng. Trù là xây dựng nhà môi giới mà không làm giảm chất lượng.

## Khái niệm

### Bốn tín hiệu định tuyến

1. **Task classification**: đơn giản/ phức tạp/ codegen/ toán học/ trò chuyện. Có thể là một phân loại dựa trên quy tắc, một LLM nhỏ (Haiku-class ở $0.25/M), hoặc nhúng sự tương tự với các bình có nhãn.

2. **Prompt length**Các mã thông báo <500 mã thông báo thường không.

3. **Embedding similarity to known-hard set**: nếu truy vấn gần (cosine > 0,88) với một thùng cứng được biết đến, leo thang trực tiếp đến biên giới.

4. **Self-confidence from first-pass**: gửi cho rẻ; nếu các bản kiểm tra nhật ký của mô hình cho thấy sự tin cậy thấp OR nó từ chối OR xuất khẩu ngôn ngữ bảo hiểm, thử lại ở biên giới.

### Ba mô hình

**Pre-route**(classifier trước): ~ 5-10ms latency thêm; nhanh nhất tổng thể.

**Cascade**(tô-lệ nhất, tăng lên trên độ tin cậy thấp): ~ 1.2x độ trễ trung bình (điều chạy rẻ cộng với xác minh), ~ 2x trên tăng lên. chất lượng tốt nhất sàn.

**Ensemble route**(điều kiện rẻ và biên giới song song cho một mẫu, chọn mô hình thưởng): chất lượng cao nhất, chi phí cao nhất; chỉ sử dụng cho A/B quan trọng.

### Thực hiện

Các cổng thông tin AI (Phase 17 · 19) phơi bày định tuyến. LiteLLM đã `router`config với fallback và cost-routing. Portkey có guard + routing. Kong AI Gateway có plugin-based routing.

Mã nguồn mở: RouteLLM (LMSYS), Không Diamond (thị thương mại), Prompt Mule.

### Lập giá năm 2026

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

Phần lớn cải tiến là phục vụ hiệu quả  các bài học cốt lõi trong giai đoạn 17 · 04-09 biến thành giảm chi phí bên nhà cung cấp. Routing cho phép bạn nắm bắt những lợi nhuận đó ở lớp ứng dụng thay vì chờ đợi tất cả người dùng của bạn di chuyển sang cấp độ rẻ.

### Sự trôi chảy là rủi ro thực sự

Tuy nhiên, bạn có thể không nhận thấy được các thông tin về các công cụ phân loại của mình, vì nó được đào tạo dựa trên dữ liệu Q1.

Các tuyến đường cổng theo các chỉ số chất lượng trực tuyến:

- Người dùng thumbs-up / thumbs-down trên mỗi tuyến đường.
- Thẩm phán LLM tự động trên một mẫu (5%) cho mỗi tuyến đường.
- Tốc độ leo thang: nếu dòng băng đang tăng lên > 30%, mô hình rẻ tiền đang bị chuyển quá mức.
- Tỷ lệ từ chối trên mỗi tuyến đường.

### Những con số mà bạn nên nhớ

- 2026 tiết kiệm đường dẫn ở chất lượng iso: 20-60% nghiên cứu trường hợp.
- Thảm giá LLM 2022-2026: tổng cộng 10 lần mỗi năm.
- GPT-4 cấp 2022 vs 2026: ~$20/M → ~$0,40/M.
- Tác động độ trễ ngập: ~ 1.2x trung bình, ~ 2x leo thang (~ 10% lưu lượng truy cập).

```figure
model-cascade-router
```

## Sử dụng nó

`code/main.py`mô phỏng trước đường, hàng loạt và tập hợp trên một khối lượng công việc hỗn hợp.

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/skill-router-plan.md`Với khối lượng công việc và ngân sách chất lượng, chọn một mô hình định tuyến và tín hiệu.

## Các bài tập

1. Đi chạy`code/main.py`- Ở tầng độ chính xác nào, thủy thủ đợt vượt qua đường dẫn trước?
2. Cơ sở người dùng của bạn là 30% doanh nghiệp (phân tích phức tạp), 70% cấp độ miễn phí ( đơn giản). Thiết kế phân chia định tuyến.
3. Một tuyến đường giảm chất lượng 2% nhưng tiết kiệm 40%. đó là một con tàu?
4. Thực hiện kiểm tra sự tin cậy bằng cách sử dụng logprobs từ OpenAI / Anthropic API.
5. Trong vòng 6 tháng, tỷ lệ leo thang tăng từ 8% lên 22%. Chẩn đoán 3 nguyên nhân và khắc phục cho mỗi nguyên nhân.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## Đọc thêm

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://web.archive.org/web/20260413143335/https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) Gateway đa mô hình với các nguyên thủy định tuyến.
