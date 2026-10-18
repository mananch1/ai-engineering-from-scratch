# Vision Encoder Patches

> Một mô hình hình ảnh đọc pixel cần một tokenizer cho pixel. Patch embedding là tokenizer đó. cắt hình ảnh thành một lưới vuông, phẳng mỗi vuông, chiếu nó qua một lớp tuyến tính, sau đó thêm một tín hiệu vị trí 2D để biến thể biết mỗi vuông ngồi ở đâu trong hình ảnh ban đầu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 lessons 30-37 (Track B foundations)
**Time:** ~90 minutes

## Mục tiêu học tập

- Đánh dấu một hình ảnh thành một chuỗi dài cố định của các bản che.
- Thực hiện một`Conv2d`- dựa trên dự đoán đệm phù hợp với toán học của mở ra-thì-lín.
- Xây dựng một vị trí xác định 2D sinusoidal nhúng để thứ tự token mã hóa vị trí không gian.
- Kiểm tra số lượng các vá, hình dạng nhúng, và `Conv2d`/tạo tương đương trên một vật cố định tổng hợp.

## Vấn đề

Một biến thể ăn một chuỗi các vector. Một hình ảnh là một lưới 3 kênh. Đọc từng pixel như một token làm nổ chiều dài chuỗi: một hình ảnh RGB 224x224 là 150.528 token, mà một biến thể 12 lớp không thể đủ khả năng để chú ý. Đọc hình ảnh như một vector phẳng khổng lồ ném ra địa điểm, mà lớp chú ý không thể phục hồi. Công việc của đầu đầu mã hóa là nén lưới pixel thành vài trăm token mà mỗi token tóm tắt một vùng vuông.

Patch embedding giải quyết vấn đề này bằng một dự đoán tuyến tính. Một hình ảnh 224x224 được cắt thành 16x16 patch tạo ra một lưới 14x14 gồm 196 patch.`(3, 16, 16) = 768`Các giá trị pixel thành một vector, sau đó một lớp tuyến tính lập bản đồ nó đến chiều kích ẩn của mô hình.`hidden`(thường là 768) cộng với một token CLS. Đó là một chuỗi mà phần còn lại của mạng có thể nhai.

## Khái niệm

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### Tại sao phải dán, không phải là pixel

Sự chú ý là hình vuông trong chiều dài chuỗi.`196 * 196 = 38,416`điểm chú ý mỗi đầu mỗi lớp; chi phí chuỗi 150.528 token `150,528 * 150,528 = 22.6 billion`Các bản vá mua một sự giảm 590,000 lần tính toán chú ý, và một khu vực 16x16 duy nhất mang đủ tín hiệu cho các nhiệm vụ tầm nhìn cấp cao. Chi phí là mất chi tiết không gian tinh tế bên trong một bản vá, đó là lý do tại sao các đống đa phương tiện dòng chảy thường chạy một chi nhánh độ phân giải cao thứ hai khi định vị tinh tế quan trọng.

### Tại sao một dự đoán tuyến tính là đủ

Mỗi vá được xử lý như một vector độc lập.`768 * 768 = 589,824`Các bộ máy mã hóa có thể được sử dụng để tạo ra các thông số về các thông số của ViT-Base và các bộ máy mã hóa nhanh.

### - `Conv2d`Tránh

A `Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)`không có đệm cho kết quả số lượng tương tự như mở-sau-lín, bởi vì mỗi vị trí đầu ra tạo ra các pixel vá chống lại một bộ lọc.

### Các vị trí được nhúng

Các token không mang theo thứ tự nào ra khỏi dự đoán.`(row, col)`Position. Một nửa chiều kích nhúng mã hóa vị trí hàng với sin/cos ở nhiều tần số; nửa kia mã hóa vị trí cột.

| Component | Shape | Parameters |
|-----------|-------|------------|
| Patch projection (`Conv2d`) | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| Position embedding (fixed) | `(num_patches, hidden)` | 0 (computed, not learned) |
| CLS token (learned) | `(1, hidden)` | `hidden` |

Đối với ViT-Base/16 ở độ phân giải 224: 590.592 tham số trong dự đoán, 768 trong token CLS, và không cho vị trí sinus. Bài học tiếp theo (59) xếp chồng một biến thể 12 lớp trên đầu đầu này.

### Tương đương với kiểm tra sức khoẻ

Bước đệm có hai cách đánh vần: a `Conv2d`Nếu không, toán học mở rộng là sai, và phần còn lại của bộ mã hóa được xây dựng trên cát. Các bài kiểm tra trong bài học này thực hiện tương đương.

```figure
ch-patch-tokenizer
```

## Hãy xây dựng nó

`code/main.py`thực hiện:

- `PatchEmbed`, một `nn.Module` bao bì`Conv2d`cho việc chiếu các vá.
- `sinusoidal_2d(grid_h, grid_w, dim)`, một hàm không có quốc gia tạo ra bảng vị trí 2D.
- `VisionFrontEnd`, bao gồm nhúng đệm, CLS prepend, và vị trí bổ sung vào một đi trước.
- A `synthesize_image(seed)`trợ lý xây dựng một thiết bị xác định 224x224x3 từ `numpy.random`- Tôi không biết.
- Một bản demo chạy một hình ảnh cố định qua đầu và in hình thức đầu ra, chuẩn token CLS và một hàng của việc nhúng vị trí.

Đi đi.

```bash
python3 code/main.py
```

Kết quả: thiết bị 224x224 được biểu tượng hóa thành một chuỗi hình dạng `(1, 197, 768)`. Điểm đầu tiên là CLS; 196 tiếp theo là các Điểm Patch. Các chuẩn mực nhúng vị trí đều đồng nhất trong một hàng, đó là chữ ký sinusoidal.

## Sử dụng nó

Cùng một bản vá đầu xuất hiện trong mọi mô hình ngôn ngữ thị giác hiện đại: CLIP ViT-L/14, SigLIP, DINOv2, gia đình Qwen-VL, và InternVL stack tất cả bắt đầu từ một `Conv2d`Các khác biệt giữa các gia đình sống theo dòng chảy (CLS vs không có CLS hợp nhất, mã đăng ký, kích thước các bản vá khác nhau 14 vs 16, độ phân giải động thông qua các vị trí liên kết).

## Các thử nghiệm

`code/test_main.py`bao gồm:

- số lượng vá phù hợp `(image_size / patch_size) ** 2`
- kết hợp hình dạng đầu ra `(batch, num_patches + 1, hidden)`
- `Conv2d`chiếu bằng cách mở rộng bằng tay-vào-sự trên một vật cố định nhỏ
- bảng vị trí sinusoidal là xác định trên các cuộc gọi
- Truyền thông mã thông CLS qua các lô bị mờ mà không có rò rỉ

Đi xem chúng:

```bash
python3 -m unittest code/test_main.py
```

## Các bài tập

1. Thay thế vị trí chân lưng bằng vị trí học `nn.Parameter`và so sánh sự mất mát thời kỳ đầu tiên trên một nhiệm vụ phân loại tổng hợp nhỏ.

2. Thay đổi `Conv2d`cho một sự rõ ràng `nn.Unfold`+`nn.Linear`và khẳng định các kết quả tương ứng với trong dung lượng nổi.

3. Thêm hỗ trợ cho kích thước các vá không bình phương (ví dụ: 32x16 cho các đầu vào rộng) và xác minh bảng vị trí xử lý lưới không bình phương.

4. Tạo hồ sơ bước đệm ở kích thước đệm 1, 8, 64. Phân chiếu đệm hiếm khi là nút thắt; các lớp chú ý phía dưới dòng thống trị.

5. Đào tạo đầu tiên như một bộ thu thập tính năng đóng băng trên một bộ dữ liệu hình dạng tổng hợp 4 lớp (thây, vuông, tam giác, sao).

## Các điều khoản chính

| Term | What it means |
|------|---------------|
| Patch | A square sub-region of the image, typically 14x14 or 16x16 |
| Patch embedding | Linear projection of one flattened patch to the hidden dim |
| Sequence length | Number of tokens after patch tokenization, usually plus CLS |
| Sinusoidal position | Fixed sin/cos signal that encodes 2D grid coordinates |
| CLS token | Learned vector prepended to the sequence as the pooling head |

## Đọc thêm

- Một hình ảnh có giá trị 16x16 từ (ViT, 2021) cho khung nhám nhám gốc.
- Cảnh sát là tất cả những gì bạn cần (2017) cho công thức vị trí sinus được điều chỉnh ở đây cho 2D.
- Bảng giấy DINOv2 cho mã đăng ký, một phần mở rộng bạn có thể thêm vào như bài tập 6.
