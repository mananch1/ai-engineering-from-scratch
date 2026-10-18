# Mã hóa vị trí  Sinusoidal, RoPE, ALiBi

> Sự chú ý là không thay đổi. "Căn nuôi ngồi trên thảm" và "mát trên mèo trên tàu" tạo ra cùng một đầu ra mà không có tín hiệu vị trí. Ba thuật toán sửa chữa nó  mỗi một cược khác nhau về điều gì " vị trí " có nghĩa là.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## Vấn đề

Sự chú ý của sản phẩm điểm quy mô là mù thứ tự.`softmax(Q K^T / √d) V`được tính từ sự tương đồng cặp.`X`Không có gì trong sự chú ý quan tâm về vị trí.

Đó không phải là một lỗi trong một mô hình túi từ. Đối với ngôn ngữ, mã, âm thanh, video  bất cứ điều gì mà thứ tự mang ý nghĩa  nó là chết người.

Trình sửa là để đưa vị trí vào các nội dung bằng cách nào đó.

1. **Absolute sinusoidal**(Vaswani 2017). Thêm `sin/cos`Đơn giản, không thể học được, không thể phân tích tốt hơn các chiều dài được đào tạo.
2. **RoPE — Rotary Position Embeddings**(Su 2021). Chuyển các vector Q và K theo một góc tương xứng với vị trí. Mã hóa vị trí * tương đối * trực tiếp trong sản phẩm chấm.
3. **ALiBi — Attention with Linear Biases**(Bấm vào năm 2022). Trượt các nhúng hoàn toàn; thêm một hình phạt tuyến tính mỗi đầu cho điểm chú ý dựa trên khoảng cách.

Tính đến năm 2026, hầu hết các mô hình mở biên giới đều sử dụng RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. Một số mô hình ngữ cảnh dài sử dụng ALiBi hoặc các biến thể hiện đại của nó.

## Khái niệm

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Tự nhiên

Lập trước một matrix cố định `PE`hình dạng`(max_len, d_model)`- Có thể là:

```math
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(pos / 10000^{2i / d_{\text{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(pos / 10000^{2i / d_{\text{model}}}\right)
\end{aligned}
```

Vậy thì`X' = X + PE[:N]`Mỗi chiều là một hình âm ở tần số khác nhau. mô hình học cách đọc vị trí từ mô hình pha.`max_len`: không có gì cho mô hình biết điều gì xảy ra ở vị trí 2048 khi nó chỉ thấy vị trí 02047.

### RoPE

Chuyển các vector Q và K (không phải nhúng).$(2i, 2i+1)$- Có thể là:

```math
\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\ \sin(pos \cdot \theta_i) & \cos(pos \cdot \theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
\qquad \theta_i = \text{base}^{-2i / d_{\text{head}}}, \quad \text{base} = 10000 \text{ by default}
```

Lấy cùng một xoay vào các phím với vị trí `pos_k`. sản phẩm điểm $q'_m \cdot k'_n$trở thành một chức năng của $(m - n)$Chỉ riêng mình.**the attention score depends only on the relative distance**Mặc dù quay được khóa khỏi vị trí tuyệt đối.

Lợi ích của RoPE: `base`Llama 3 mở rộng từ 8K đến 128K theo cách này.

### ALiBi

Trượt qua thủ thuật nhúng vào.

```math
\text{attn\_score}[i, j] = \frac{q_i \cdot k_j}{\sqrt{d}} - m_h \cdot |i - j|
```

Ở đâu $m_h$là một đường ngốc cụ thể cho đầu (ví dụ $1 / 2^{8h/H}$Các token gần hơn được tăng cường; các token xa bị phạt. Không có chi phí thời gian đào tạo.

### Những gì để chọn vào năm 2026

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

RoPE thắng vì nó thu hút sự chú ý mà không thay đổi kiến trúc, mã hóa vị trí tương đối, và nó `base`siêu tham số cung cấp một nút sạch cho điều chỉnh tinh tế trong bối cảnh dài.

```figure
rope-explorer
```

## Hãy xây dựng nó

### Bước 1: mã hóa hình âm

Nhìn xem`code/main.py`Một tính toán 4 dòng:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Thêm vào các matrix nhúng trước lớp chú ý đầu tiên.

### Bước 2: RoPE được áp dụng cho Q, K

RoPE hoạt động tại chỗ trên Q và K. Đối với mỗi cặp bóng tối:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Quan trọng: áp dụng cùng một hàm cho Q tại vị trí $m$và K ở vị trí $n$Sản phẩm của họ có thể nhận được một$\cos((m-n) \cdot \theta_i)$chú ý học được vị trí tương đối miễn phí.

### Bước 3: Cột độ và thiên vị của ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Thêm `bias[h]`đến `(seq_len, seq_len)`điểm chú ý của đầu `h`, sau đó là Softmax.

### Bước 4: xác minh tính chất tương đối khoảng cách của RoPE

Chọn hai vector ngẫu nhiên `a, b`- Chuyển đi`(pos_a, pos_b)`Rồi rồi.`(pos_a + k, pos_b + k)`. Cả hai sản phẩm điểm phải phù hợp trong sai lầm điểm nổi. Cất lượng đó là toàn bộ điểm của RoPE  nó không thay đổi đối với sự bù đắp tuyệt đối, chỉ có khoảng cách tương đối quan trọng.

## Sử dụng nó

PyTorch 2.5+ tàu RoPE tiện ích trong `torch.nn.functional`Hầu hết mã sản xuất sử dụng`flash_attn`hoặc `xformers`nơi RoPE được áp dụng bên trong hạt nhân chú ý.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.**Tái quy mô `base`đến`base * (scale_factor)^(d/(d-2))`khi mở rộng từ 4K đến 16K+.
- **YaRN.**Sự phân tích thông minh hơn để bảo vệ sự chú ý vào các bối cảnh dài. Llama 3.1 128K sử dụng nó.
- **LongRoPE.**Phương pháp 2024 của Microsoft sử dụng tìm kiếm tiến hóa để chọn các yếu tố trên quy mô kích thước. Phi-3-Long sử dụng nó.
- **Position interpolation + fine-tuning.**Chỉ cần thu hẹp vị trí bằng nhân rộng và điều chỉnh tốt cho token 15B.

## Chuyển nó

Nhìn xem`outputs/skill-positional-encoding-picker.md`. Kỹ năng chọn một chiến lược mã hóa cho một mô hình mới do chiều dài bối cảnh mục tiêu, nhu cầu phân tích và ngân sách đào tạo.

## Các bài tập

1. **Easy.**Đặt đường chân lưng `PE`Matrix như là một bản đồ nhiệt cho `max_len=512, d=128`- Đảm nhận mô hình "những dải mở rộng hơn khi chỉ số kích thước tăng lên".
2. **Medium.**Thực hiện quy mô RoPE có ý thức NTK. Tập một LM nhỏ trên các chuỗi dài 256, sau đó thử nghiệm trên dài 1024 với và không quy mô. đo độ phức tạp.
3. **Hard.**Thực hiện ALiBi và RoPE trong cùng một mô-đun chú ý. Trình chuyển đổi 4 lớp trên một nhiệm vụ sao chép với chuỗi dài 512. Phân tích đến 2048 tại thời điểm thử nghiệm. So sánh sự suy giảm.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position. |
| Sinusoidal | "The original one" | `sin/cos` at geometric frequencies added to embeddings; doesn't extrapolate. |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance. |
| ALiBi | "Linear bias trick" | Add $-m \cdot \lvert i-j \rvert$ to attention scores; no embedding needed, great extrapolation. |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context at inference. |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed when context expands. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation that preserves attention entropy. |
| Extrapolation | "Works beyond trained length" | Can the position scheme serve correct output past `max_len` seen in training? |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) hình âm đạo gốc.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) Bảng giấy RoPE.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) hiện đại nhất RoPE quy mô.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) Llama 2 của Meta.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) phương pháp Microsoft được Phi-3-Long sử dụng và được trích dẫn trong phần Use It.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) Thực hiện cấp sản xuất của mỗi chương trình quy mô RoPE (đặc định, tuyến tính, động, YaRN, LongRoPE, Llama-3).
