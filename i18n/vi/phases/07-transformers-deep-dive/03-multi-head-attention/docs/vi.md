# Sự chú ý nhiều đầu

> Một đầu chú ý học một mối quan hệ một lần, tám đầu học tám đầu tự do, lấy nhiều hơn.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## Vấn đề

Một đầu tự chú ý đơn lẻ tính toán một matrix chú ý. Matrix đó nắm bắt một loại mối quan hệ  thường là một mối quan hệ giảm thiểu tổn thất trên bất kỳ tín hiệu đào tạo nào. Nếu dữ liệu của bạn có sự đồng thuận đối tượng-tên, tham chiếu đồng, bài phát biểu dài và phân đoạn ngữ pháp tất cả được dính vào nhau, một đầu đơn lẻ bôi chúng thành một phân phối tối đa mềm duy nhất và mất một nửa tín hiệu.

Sự cố từ bài báo Vaswani năm 2017: chạy một số chức năng chú ý song song, mỗi chức năng có dự đoán Q, K, V của riêng mình, và kết nối các đầu ra.$d_{\text{model}} / n_{\text{heads}}$- Tỷ lệ tổng số vẫn không thay đổi.

Sự chú ý đa đầu là mặc định của mọi bộ biến đổi trong 2026 tàu. Vấn đề duy nhất là về * bao nhiêu đầu và liệu các phím và giá trị có chia sẻ dự đoán (Thăm tâm nhóm, Thăm tâm đa câu hỏi, Thăm tâm trần trần đa đầu).

## Khái niệm

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.**Nhận đi`X`hình dạng`(N, d_model)`. Dự án đến Q, K, V mỗi hình dạng `(N, d_model)`- Tái tạo lại`(N, n_heads, d_head)`nơi $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$- Chuyển vào`(n_heads, N, d_head)`- Tôi không biết.

**Attend in parallel.**Điệu suất điểm-đánh giá trong mỗi đầu.`(N, d_head)`Các đầu hoạt động trên các vùng phụ khác nhau của việc nhúng và không bao giờ nói chuyện trong quá trình tính toán sự chú ý.

**Concatenate and project.**Lầu đầu quay lại `(N, d_model)`và nhân bằng một matrix đầu ra học `W_o`hình dạng`(d_model, d_model)`- `W_o`là nơi mà đầu người được pha trộn.

**Why it works.**Mỗi đầu có thể chuyên môn mà không cạnh tranh với các đầu khác về ngân sách đại diện. Các nghiên cứu thăm dò từ năm 20192024 cho thấy các vai trò đầu khác nhau: đầu vị trí, đầu tham gia vào mã thông báo trước đó, đầu sao chép, đầu thực thể có tên, đầu cảm ứng (được đặt nền tảng cho việc học trong bối cảnh).

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA là mặc định hiện đại bởi vì nó cắt giảm bộ nhớ cache KV bằng một nhân tố của $N/G$MLA đi xa hơn bằng cách nén K/V vào một không gian ẩn, sau đó chiếu lại vào thời gian tính toán  chi phí FLOPs, tiết kiệm nhiều bộ nhớ hơn.

```figure
multihead-split
```

## Hãy xây dựng nó

### Bước 1: chia đầu khỏi sự chú ý đơn đầu mà chúng ta đã có

Hãy lấy `SelfAttention`từ Bài học 02 và lấn nó bằng một cặp chia/concat.`code/main.py`cho một thực hiện numpy; logic là:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Một hình dạng lại và một chuyển thể. Không vòng lặp. Đây chính xác là những gì PyTorch làm dưới`nn.MultiheadAttention`- Tôi không biết.

### Bước 2: chạy điểm quy mô- sản phẩm chú ý mỗi đầu

Mỗi đầu đều có một mảnh của riêng mình của Q, K, V. Sự chú ý trở thành một bộ đống:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Trên thiết bị thực`Qh @ Kh.transpose(...)`là một `bmm`GPU nhìn thấy một bộ hình dạng đơn lẻ`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`Thêm đầu là miễn phí.

### Bước 3: Phân tích nhóm-Phân tích chú ý

Chỉ có các dự báo khóa và giá trị thay đổi.`n_heads`nhóm; K và V nhận `n_kv_heads < n_heads`nhóm và được lặp lại để phù hợp với:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Theo kết luận , điều này tiết kiệm được trí nhớ bởi vì chỉ có`n_kv_heads`Các bản sao sống trong cache KV, không `n_heads`Llama 3 70B sử dụng 64 đầu truy vấn với 8 đầu KV  một bộ thu nhỏ cache 8x.

### Bước 4: kiểm tra những gì mỗi đầu học được

Đọc MHA trên một câu ngắn với 4 đầu.`(N, N)`bạn sẽ thấy các đầu khác nhau chọn ra cấu trúc khác nhau ngay cả với sự khởi tạo ngẫu nhiên đó là một phần tín hiệu, một phần chu trình đối xứng quay trong các tiểu không gian.

## Sử dụng nó

Trong PyTorch, phiên bản một dòng:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA từ PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?**Quy tắc ngón tay từ các mô hình sản xuất vào năm 2026:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`Hầu như luôn luôn hạ cánh ở 64 hoặc 128. Đó là đơn vị của một đầu có thể "xem".$\sqrt{d_{\text{head}}}$; đi trên 256 và bạn sẽ mất lợi ích "nhiều chuyên gia nhỏ".

## Chuyển nó

Nhìn xem`outputs/skill-mha-configurator.md`. Kỹ năng này khuyến cáo số lượng đầu, số lượng đầu kv và chiến lược chiếu cho một biến thể mới với ngân sách tham số, chiều dài chuỗi và mục tiêu triển khai.

## Các bài tập

1. **Easy.**Hãy lấy MHA từ `code/main.py`và thay đổi`n_heads`từ 1 đến 16 với `d_model=64`- Làm việc này có thể giúp đỡ, làm cho cao điểm, hoặc làm tổn thương?
2. **Medium.**Thực hiện MQA (một đầu KV được chia sẻ trên tất cả các đầu truy vấn). đo số lượng số parameter giảm so với MHA đầy đủ. Xét số lượng KV-thủ nhớ nhỏ gọn ở suy luận cho N = 2048.
3. **Hard.**Thực hiện một phiên bản nhỏ của Multi-head Latent Attention: nén K, V đến một cấp độ`r`- Lưu trữ trong cache KV, giải nén vào thời điểm chú ý.`r`bộ nhớ cache vượt qua dưới 1/8 của MHA đầy đủ trong khi chất lượng vẫn trong vòng 1 bit của việc xác thực ppl?

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$ with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) mô hình đầu đa đầu ban đầu.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) giấy tờ MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) cách chuyển đổi MHA thành GQA sau khi đào tạo.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) MLA và tại sao nó đánh bại MHA / GQA trên bộ nhớ cache.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) nhìn cơ học về những gì đầu thực sự làm.
