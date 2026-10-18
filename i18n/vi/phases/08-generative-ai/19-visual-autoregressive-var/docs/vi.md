# Mô hình hóa tự động thị giác (VAR): Dự đoán quy mô tiếp theo

> Các mô hình phân tán lấy mẫu lặp đi lặp lại theo thời gian (chước đánh dấu). Các mẫu VAR lặp đi lặp lại theo quy mô  nó dự đoán một token 1x1, sau đó 2x2, sau đó 4x4, cho đến độ phân giải cuối cùng, mỗi quy mô điều chỉnh trên quy mô trước đó. Bài báo 2024 cho thấy VAR phù hợp với các quy luật quy mô kiểu GPT cho việc tạo hình ảnh và đánh bại DiT với ngân sách tính toán tương tự. Bài học này xây dựng cơ chế cốt lõi.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## Vấn đề

Thế hệ tự do thống trị mô hình hóa ngôn ngữ vì nó có quy mô dự đoán: tính toán nhiều hơn, nhiều tham số hơn, độ bối rối thấp hơn, kết quả tốt hơn.

Cả hai đều bị một vấn đề thứ tự thế hệ. Pixel và token được sắp xếp trong một lưới 2D, nhưng mô hình AR phải truy cập chúng trong một thứ tự raster 1D. Một pixel góc sớm không biết hình ảnh cuối cùng trở thành gì. Chất lượng thế hệ leo thang tồi tệ hơn GPT-on-text và không bao giờ đạt đến chất lượng mô hình phân tán với tính toán phù hợp.

VAR khắc phục vấn đề thứ tự thế hệ bằng cách thay đổi những gì đang được tạo ra. Thay vì dự đoán các mã thông báo hình ảnh một một trong không gian, VAR dự đoán một bức ảnh toàn bộ với độ phân giải tăng lên. Bước 1: dự đoán một mã thông báo 1x1 (tổng hình "sự tóm tắt"). Bước 2: dự đoán một lưới mã thông báo 2x2 (chức năng rỗng hơn). Bước 3: dự đoán một lưới 4x4. Bước K: dự đoán lưới cuối cùng (H/8) x ((W/8)).

Mỗi thang đo sẽ theo dõi tất cả các thang đo trước đó (từ nguyên nhân đến "sự sắp xếp quy mô") và song song song trong thang đo riêng của nó. Vấn đề thứ tự biến mất: toàn bộ hình ảnh ở thang k được tạo ra trong một chuyển đổi.

## Khái niệm

### VQ-VAE Multi-Scale Tokenizer

VAR cần một **multi-scale discrete tokenizer**Để có hình ảnh$x$, nó tạo ra một chuỗi các lưới token độ phân giải cao dần dần:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Mỗi người$z_k$sử dụng cùng một cuốn sách mã (kích thước điển hình 4096-16384). Việc mã hóa tại mỗi thang không độc lập  nó được đào tạo để tổng hợp các dư ở mỗi thang tái tạo $f$- Có thể là:

```math
f \approx \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_1), \text{target\_size}) + \cdots + \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_K), \text{target\_size})
```

Đây là một **residual VQ**biến thể.$k$bắt được những cân bằng nào $1 \ldots k-1$Decoder lấy tổng số tất cả các tích hợp quy mô và tạo ra hình ảnh.

VQ Tokenizer đa quy mô được đào tạo một lần (như VQGAN) và sau đó đóng băng.

### Dự đoán về quy mô tiếp theo

Mô hình tạo là một bộ biến đổi nhìn thấy các token từ tất cả các thang trước đó và dự đoán các token ở thang kế tiếp.

Cấu trúc chuỗi đầu vào:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Các vị trí nhúng mã hóa cả chỉ số quy mô và vị trí không gian trong quy mô.$k$, vị trí $(i, j)$có thể tham gia tất cả các token ở quy mô $1 \ldots k$và các token ở quy mô $k$chính nó xuất hiện sớm hơn trong bất kỳ thứ tự nội quy nào được sử dụng (VAR sử dụng sự chú ý vị trí cố định mà không có tính nhân quả nội quy  tất cả các vị trí trong một quy mô được dự đoán song song).

Thiệt hại đào tạo: trên mỗi thang$k$, dự đoán các token $z_k$GPT chỉ có "đường" mà bây giờ có cấu trúc quy mô.

### Thế hệ

Khi kết luận:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Vì $K = 10$Scales, thế hệ là 10 chuyển đổi chuyển tiếp. Mỗi lần đi lại tạo ra toàn bộ quy mô của nó song song  không có tự rút theo mã thông báo trong một quy mô. Đối với một hình ảnh 256x256 đây là khoảng 10 lần đi qua so với DiT's 28-50.

### Tại sao quy mô tiếp theo thắng hơn quy mô tiếp theo

Ba chiến thắng cơ cấu:
1. **Coarse-to-fine aligns with natural image statistics.**Nhận thức trực quan của con người và các bộ dữ liệu hình ảnh đều có tính thường lệ phụ thuộc vào quy mô: cấu trúc tần số thấp ổn định và có thể dự đoán được; chi tiết tần số cao phụ thuộc vào nội dung tần số thấp. Dự đoán quy mô tiếp theo khai thác điều này.
2. **Parallel generation within scale.**Không giống như token AR kiểu GPT, VAR tạo ra tất cả các token ở một quy mô trong một bước.
3. **No generation order bias.**Các token ở thang k nhìn thấy tất cả các thang k-1; không có "vẫn" hoặc "nâng" thiên vị buộc các token sớm phải cam kết trước khi bối cảnh muộn có sẵn.

### Luật quy mô

Tian và các đồng nghiệp. chứng minh rằng VAR theo đường cong quy mô pháp luật cho FID trên ImageNet  giống như GPT làm cho sự bối rối. Việc tăng gấp đôi các tham số hoặc tính toán một cách đáng tin cậy làm giảm một nửa lỗi. Đây là mô hình tạo hình ảnh đầu tiên thể hiện kiểu hành vi quy mô này một cách sạch sẽ như mô hình ngôn ngữ. Kết quả là dự đoán quy mô VAR trở nên dự đoán từ tính toán, không phải phỏng đoán thực nghiệm cho mỗi kiến trúc.

### Mối quan hệ với sự pha trộn

VAR và diffusion chia sẻ cùng một câu chuyện nén dữ liệu: cả hai phá vỡ vấn đề sản xuất thành một chuỗi các phụ vấn dễ dàng hơn.

- Phân phối: dần dần thêm tiếng ồn, học cách hủy bỏ một bước.
- VAR: dần dần thêm độ phân giải, học cách dự đoán quy mô tiếp theo.

Cả hai đều tạo ra phân phối điều kiện có thể xử lý. Về cơ bản, VAR nhanh hơn khi suy luận ( ít hơn đi qua, tất cả đều song song trong một thang) và phù hợp hoặc vượt qua DiT trên ImageNet theo điều kiện lớp học. VAR theo điều kiện văn bản (VARclip, HART) là một hướng nghiên cứu tích cực.

```figure
gx-var-next-scale
```

## Hãy xây dựng nó

Trong `code/main.py`bạn sẽ:
1. Hãy xây một cái nhỏ.**multi-scale VQ tokenizer**trên dữ liệu "hình ảnh" tổng hợp (2D vòng Gaussian).
2. Đào một **VAR-style transformer**để dự đoán các token quy mô tiếp theo.
3. Mô hình bằng cách gọi bộ biến đổi 4 lần (4 thang) và giải mã.
4. Kiểm tra rằng đào tạo theo quy mô sắp xếp tạo ra tương đồng trong quy mô.

Đây là một sự triển khai đồ chơi. Điểm là để thấy mặt nạ tập trung quy mô cấu trúc và thế hệ song song trong quy mô thực sự hoạt động.

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/skill-var-tokenizer-designer.md` một kỹ năng thiết kế một tokenizer đa quy mô: số lượng quy mô, tỷ lệ quy mô, kích thước sổ mã, chia sẻ dư thừa, kiến trúc decoder.

## Các bài tập

1. **Scale count ablation.**Đào tạo VAR với 4, 6, 8, 10 thang. đo chất lượng tái thiết so với số lượng các đường đi tự rút.

2. **Codebook size.**Đường bộ mã hóa với kích thước codebook 512, 4096, 16384.

3. **Parallel-within-scale check.**Đối với một VAR được đào tạo, đo mô hình chú ý một cách rõ ràng. Trong thang k, mô hình có quan tâm đến các vị trí qua thang nhưng không phải trong thang?

4. **VAR vs DiT scaling.**Đối với cùng một nhiệm vụ tùy thuộc vào lớp ImageNet, đào tạo VAR và DiT với ngân sách param phù hợp (ví dụ: 33M, 130M, 458M).

5. **Text conditioning.**Lũ rộng VAR để lấy một bản ghi văn bản (CLIP hợp nhất) như một đầu vào điều kiện thêm thông qua adaLN. Đây là công thức HART. FID cải thiện bao nhiêu trên lấy mẫu phù hợp với văn bản?

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | Image generation by next-scale prediction over a pyramid of VQ token grids |
| Next-scale prediction | "Predict coarser, then finer" | The model predicts tokens at increasing resolution scales, conditioning on all previous scales |
| Multi-scale VQ tokenizer | "Residual VQ" | VQ-VAE that produces K token grids of increasing resolution, with decoder summing all scales |
| Scale k | "Pyramid level k" | One of K resolution levels, from 1x1 at k=1 up to (H/p)x(W/p) at k=K |
| Parallel-within-scale | "One forward per scale" | All tokens at scale k are predicted in one transformer pass, not autoregressively |
| Causal-across-scales | "Scale-ordered attention" | Token at scale $k$ can attend to all of scales $1 \ldots k$ but not scales $k+1 \ldots K$ |
| Residual VQ | "Additive tokenization" | Each scale's tokens encode the residual left by lower scales; decoder sums all scale embeddings |
| VAR scaling law | "Image GPT scaling" | FID follows a predictable power law in compute, like language models' perplexity |
| HART | "Hybrid VAR + text" | Text-conditional VAR variant combining MaskGIT-style iterative decoding with VAR's scale structure |
| Scale position embedding | "(scale, row, col) triple" | Positional encoding carries both the scale index and spatial coordinates within the scale |

## Đọc thêm

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) giấy VAR, tài liệu tham chiếu theo luật
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) DiT, đường cơ sở so sánh phân tán
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) VQGAN, gia đình tokenizer VAR của tokenizer đa quy mô mở rộng
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) VQ-VAE, nền tảng của việc mã hóa hình ảnh riêng biệt
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) VAR theo văn bản
