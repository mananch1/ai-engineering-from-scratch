# Xây dựng một Transformer từ đầu  Capstone

> 13 bài học, 1 mô hình, không có lối tắt.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## Vấn đề

Bạn đã đọc mọi bài báo, bạn đã thực hiện sự chú ý, chia chia nhiều đầu, mã hóa vị trí, mã hóa và mã hóa khối, BERT và GPT mất mát, MoE, KV cache. Bây giờ làm cho chúng làm việc cùng nhau trên một nhiệm vụ thực sự.

Chất cốt: đào tạo một bộ biến đổi nhỏ chỉ có trình giải mã từ đầu đến cuối trong một nhiệm vụ mô hình hóa ngôn ngữ cấp độ nhân vật. Nó đọc Shakespeare. Nó tạo ra Shakespeare mới. Nó đủ nhỏ để đào tạo trên máy tính xách tay trong vòng chưa đầy 10 phút. Nó đủ chính xác rằng việc trao đổi trong một tập dữ liệu lớn hơn và đào tạo lâu hơn sẽ giúp bạn có một LM thực sự.

Đây là "nanoGPT" của khóa học. Nó không phải là gốc  Karpathy's 2023 nanoGPT hướng dẫn là thực hiện tham chiếu mỗi học sinh viết ít nhất một lần. Chúng tôi nâng hình dạng và tái cấu trúc nó xung quanh những gì chúng tôi đã bao gồm.

## Khái niệm

![Transformer-from-scratch block diagram](../assets/capstone.svg)

Kiến trúc, ghi chú:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Những gì chúng ta vận chuyển

- `GPTConfig` một nơi để cấu hình tất cả các siêu tham số.
- `MultiHeadAttention` nguyên nhân, đợt, với hướng theo kiểu Flash tùy chọn (PyTorch's `scaled_dot_product_attention`().
- `SwiGLUFFN` FFN hiện đại.
- `Block` trước chuẩn, lưu tâm bao bì dư thừa + FFN.
- `GPT` nhúng, khối xếp chồng, đầu LM, tạo().
- Chuyện tập với AdamW, cosine LR, cắt gradient.
- Đồ ký cấp Char về văn bản Shakespeare.

### Những gì chúng ta không vận chuyển

- RoPE  được thực hiện theo khái niệm trong Bài học 04. Ở đây chúng tôi sử dụng các tích hợp vị trí được học để đơn giản hóa.
- KV cache trong quá trình tạo  mỗi bước tạo tính lại sự chú ý trên tiền tố đầy đủ. chậm hơn nhưng đơn giản hơn. Các bài tập yêu cầu bạn thêm một cache KV.
- Flash Attention  PyTorch 2.0+ tự động gửi nếu đầu vào phù hợp; chúng tôi sử dụng `F.scaled_dot_product_attention`- Tôi không biết.
- MoE  một FFN mỗi khối. Bạn đã xem MoE trong Bài học 11.

### Các số liệu mục tiêu

Trên máy tính xách tay Mac M2, một máy tính xách tay 4 tầng, 4 đầu, d_model=128 GPT được đào tạo cho 2.000 bước trên `tinyshakespeare.txt`- Có thể là:

- Khối mất tập luyện hội tụ từ ~ 4,2 (thuộc tình cờ) đến ~ 1,5 trong khoảng 6 phút.
- Kết quả lấy mẫu trông hình Shakespeare: từ cổ điển, đoạn đường, tên thích hợp như "ROMEO:" xuất hiện.
- Sự mất mát của Val (từ 10% cuối cùng của văn bản) theo dõi chặt chẽ sự mất mát của đào tạo; không quá phù hợp với quy mô/ ngân sách này.

```figure
n5-block-stack
```

## Hãy xây dựng nó

Bài học này sử dụng PyTorch.`torch`(Phát triển CPU là tốt).`code/main.py`- Lịch bản xử lý:

- Tải xuống `tinyshakespeare.txt`nếu thiếu (hoặc đọc bản sao địa phương).
- Tầm biểu tượng cấp độ byte.
- Đường lửa/vùng chia ở 90/10.
- Chuyển tập vòng với bf16 tự độngcast trên phần cứng hỗ trợ.
- Việc lấy mẫu sau khi huấn luyện hoàn thành.

### Bước 1: dữ liệu

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 ký tự độc đáo, từ vựng nhỏ, có kích thước từ vựng 4byte, không có BPE, không có kịch bản token.

### Bước 2: mô hình

Nhìn xem`code/main.py`. Bảng là sách giáo khoa từ Bài học 05  chuẩn trước, RMSNorm, SwiGLU, MHA nguyên nhân.

### Bước 3: vòng đào tạo

Nhận một loạt các cửa sổ biểu tượng dài 256 trước, chuyển đổi qua một, ngược, bước AdamW, ghi lại, lặp lại.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Bước 4: mẫu

Khi được yêu cầu, liên tục chuyển tiếp, lấy mẫu từ các log top-p, thêm vào, và tiếp tục.

### Bước 5: đọc đầu ra

Sau 2.000 bước:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Không phải Shakespeare, nhưng hình Shakespeare, một chiến thắng rõ ràng với khoảng 800K tham số và 6 phút trên máy tính xách tay.

## Sử dụng nó

Ngọc đá này là một kiến trúc tham chiếu.

1. **Swap the tokenizer.**Sử dụng BPE (ví dụ:`tiktoken.get_encoding("cl100k_base")`(Vocal size jumps from 65 to ~ 50,000.
2. **Train on a bigger corpus.**Sử dụng `OpenWebText`hoặc `fineweb-edu`(HuggingFace). 10B token trên một A100 chỉ mất khoảng 24 giờ để có được một GPT 125M-param.
3. **Add RoPE + KV cache + Flash Attention.**Các bài tập dưới đây sẽ hướng dẫn bạn thông qua từng bài tập.

Điều này kết thúc như một GPT 125M tham số mà tạo ra tiếng Anh lưu động. Không phải là mô hình biên giới. Nhưng cùng một đường mã  chỉ lớn hơn  là những gì Karpathy, EleutherAI, và Viện Allen sử dụng để đào tạo các điểm kiểm soát nghiên cứu vào năm 2026.

## Chuyển nó

Nhìn xem`outputs/skill-transformer-review.md`. Kỹ năng xem xét việc thực hiện biến đổi từ đầu để xác định tính chính xác trong tất cả 13 bài học trước đó.

## Các bài tập

1. **Easy.**Đi chạy`code/main.py`Hãy xác minh rằng mất tích xác thực bước cuối cùng của mô hình được đào tạo của bạn là dưới 2.0. Thay đổi `max_steps`từ 2.000 đến 5.000  liệu sự mất val tiếp tục cải thiện?
2. **Medium.**Thay thế các embedment vị trí học được bằng RoPE.`MultiHeadAttention`- Đường và xác minh mất val ít nhất là thấp.
3. **Medium.**Thực hiện một bộ nhớ cache KV trong vòng lấy mẫu. Tạo 500 token với và không có bộ nhớ cache. Clock tường nên được cải thiện 520x trên máy tính xách tay.
4. **Hard.**Thêm một đầu thứ hai vào mô hình dự đoán mã thông báo cộng thêm một tiếp theo (MTP  Multi-Token Prediction từ DeepSeek-V3).
5. **Hard.**Thay thế FFN đơn lẻ trên mỗi khối bằng một MoE 4 chuyên gia. Router + top-2 định tuyến. Xem cách mất val thay đổi ở các tham số hoạt động phù hợp.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; the canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial since 2015 uses it. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, keep optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; prevents training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw; overfit detector. |

## Đọc thêm

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) thực hiện thông tin chú thích cổ điển.
