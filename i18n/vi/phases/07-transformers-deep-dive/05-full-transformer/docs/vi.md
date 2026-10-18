# Bộ biến đổi đầy đủ  Encoder + Decoder

> Sự chú ý là ngôi sao. Mọi thứ khác, những dư thừa, bình thường hóa, chuyển tiếp, sự chú ý qua nhau, là những cái bàn phế để bạn xếp nó sâu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## Vấn đề

Một lớp chú ý đơn là một bộ thu hoạch tính năng, không phải là mô hình. Một matmul mỗi lớp không đủ dung lượng cho ngôn ngữ. Bạn cần độ sâu  và độ sâu phá vỡ mà không cần ống nước đúng.

Báo Vaswani năm 2017 đã đóng gói sáu quyết định thiết kế biến một lớp chú ý thành một khối xếp chồng. Mỗi biến thể kể từ khi  chỉ có mã hóa (BERT), chỉ có mã hóa (GPT), chỉ có mã hóa (T5)  thừa kế cùng một bộ xương. Năm 2026 các khối đã được tinh chỉnh (RMSNorm, SwiGLU, pre-norm, RoPE) nhưng bộ xương là giống nhau.

Bài học này là bộ xương. Bài học tiếp theo chuyên về nó  06 cho các mã hóa, 07 cho các mã hóa, 08 cho mã hóa-các mã hóa.

## Khái niệm

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### 6 mảnh

1. **Embedding + positional signal.**Địa chỉ → vector. Vị trí được tiêm qua RoPE (công nghệ hiện đại) hoặc sinusoidal (classic).
2. **Self-attention.**Mỗi vị trí đều được bảo vệ bởi các máy giải mã.
3. **Feed-forward network (FFN).**MLP hai lớp theo vị trí: `W_2 · activation(W_1 · x)`Tỷ lệ mở rộng 4x theo mặc định.
4. **Residual connection.** `x + sublayer(x)`Không có nó, gradient biến mất sau khoảng 6 lớp.
5. **Layer normalization.** `LayerNorm`hoặc `RMSNorm`(công nghệ hiện đại) ổn định dòng lưu lượng dư thừa.
6. **Cross-attention (decoder only).**Các truy vấn đến từ bộ giải mã, các khóa và giá trị từ đầu ra bộ giải mã.

Xem một vector chảy qua một khối: sự chú ý trộn lẫn qua các vị trí, dư sẽ mang nó về phía trước, FFN biến đổi nó, và chuẩn giữ cho dòng chảy ổn định.

```figure
transformer-block
```

### Bloc mã hóa (được sử dụng bởi BERT, mã hóa T5)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Mã hóa là hai chiều, không che giấu, mọi vị trí đều thấy mọi vị trí.

### Bloc decoder (được sử dụng bởi GPT, decoder T5)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Các bộ giải mã có ba lớp phụ mỗi khối. trung tâm  sự chú ý chéo  là nơi duy nhất thông tin chảy từ bộ giải mã đến bộ giải mã. Trong một kiến trúc chỉ có trình giải mã (GPT), sự chú ý chéo bị bỏ qua và bạn chỉ có sự chú ý tự ẩn + FFN.

### Pre-norm vs post-norm

Bức giấy gốc: `x + sublayer(LN(x))`vs `LN(x + sublayer(x))`. Sau chuẩn bị mất đi sự ủng hộ vào khoảng năm 2019  việc đào tạo sâu hơn mà không cần được ấm áp cẩn thận.`LN`* trước * lớp phụ) là mặc định 2026: Llama, Qwen, GPT-3+, Mistral tất cả sử dụng nó.

### Phòng hiện đại hóa năm 2026

Vaswani 2017 đã xuất khẩu LayerNorm + ReLU.

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm giảm trung tâm trung bình của LayerNorm (một lần trừ ít hơn), giúp tiết kiệm tính toán và là thực nghiệm ít nhất ổn định.`Swish(W1 x) ⊙ W3 x`) luôn vượt trội hơn ReLU/GELU FFN bằng ~ 0,5 điểm trong các bài báo Llama, PaLM và Qwen.

### Số lượng tham số

Một khối với `d_model = d`và mở rộng FFN `r`- Có thể là:

- MHA: `4 · d²`(Q, K, V, O dự đoán)
- FFN (SwiGLU): `3 · d · (r · d)`≈ ≈`3rd²`
- Các tiêu chuẩn: không đáng kể

Tại `d = 4096, r = 2.6, layers = 32`(khoảng Llama 3 8B), tổng số: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`(cộng với các bản nhúng và đầu)

## Hãy xây dựng nó

### Bước 1: các khối xây dựng

Sử dụng cái nhỏ `Matrix`lớp từ Bài học 03 (được sao chép vào tệp này để độc lập):

- `layer_norm(x, eps=1e-5)` trừ trung bình, chia bằng std.
- `rms_norm(x, eps=1e-6)` chia bằng RMS. Không trừ trung bình.
- `gelu(x)`và `silu(x) * W3 x`(SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`- Tôi không biết.
- `encoder_block(x, params)`và `decoder_block(x, enc_out, params)`- Tôi không biết.

Nhìn xem`code/main.py`cho toàn bộ dây.

### Bước 2: cáp một bộ mã hóa 2 lớp và một bộ mã hóa 2 lớp

Đặt chúng lên, chuyển đầu ra mã hóa vào mỗi bộ giải mã, thêm LN cuối cùng trước khi dự đoán đầu ra.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Bước 3: chạy về phía trước trên một ví dụ đồ chơi

Đưa nguồn 6 token và mục tiêu 5 token qua.`(5, vocab)`Không có đào tạo, bài học này là về kiến trúc, không phải về sự mất mát.

### Bước 4: trao đổi trong RMSNorm + SwiGLU

Thay thế LayerNorm và ReLU-FFN bằng RMSNorm và SwiGLU. xác nhận hình dạng vẫn phù hợp. Đây là hiện đại hóa 2026 với một thay thế chức năng.

## Sử dụng nó

Các thực hiện tham chiếu PyTorch/TF: `nn.TransformerEncoderLayer`- `nn.TransformerDecoderLayer`Nhưng hầu hết mã sản xuất 2026 đều có khối riêng vì:

- Flash Attention được gọi vào trong sự chú ý, không phải qua `nn.MultiheadAttention`- Tôi không biết.
- GQA / MLA không có trong tham chiếu stdlib.
- RoPE, RMSNorm, SwiGLU không phải là các mặc định PyTorch.

HF `transformers`có các khối tham chiếu sạch bạn nên đọc: `modeling_llama.py`là khối chỉ có decoder 2026 của Canon. Nó khoảng 500 dòng và đáng để đi qua một lần.

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

Chỉ có trình giải mã đã giành chiến thắng ngôn ngữ vì nó có quy mô sạch nhất và xử lý cả sự hiểu biết và phát triển.

## Chuyển nó

Nhìn xem`outputs/skill-transformer-block-reviewer.md`. Khả năng xem xét việc triển khai khối biến thể mới đối với các mặc định 2026 và đánh dấu các phần thiếu (pre-norm, RoPE, RMSNorm, GQA, FFN tỷ lệ mở rộng).

## Các bài tập

1. **Easy.**Đếm các tham số trong block encoder của bạn ở `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Thiết lập bằng cách thực hiện khối và sử dụng `sum(p.numel() for p in block.parameters())`- Tôi không biết.
2. **Medium.**Chuyển từ post-norm sang pre-norm. khởi động cả hai và đo chuẩn kích hoạt sau 12 lớp xếp chồng vào vào ngẫu nhiên.
3. **Hard.**Thực hiện một bộ mã hóa-chế lập 4 lớp trên một nhiệm vụ sao chép đồ chơi (cói `x`Trở 100 bước. báo cáo mất mát. Thay đổi trong RMSNorm + SwiGLU + RoPE  mất mát giảm?

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks. |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same empirical stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model, usually 4 (LayerNorm) or 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers; slight ppl improvement, smaller model. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) đặc điểm khối ban đầu.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) tại sao tiền chuẩn đánh bại hậu chuẩn sâu sắc.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) tờ SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) khối chỉ có decoder canonical 2026
