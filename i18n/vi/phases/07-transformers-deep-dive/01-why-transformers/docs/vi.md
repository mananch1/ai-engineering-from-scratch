# Tại sao Transformers  Các vấn đề với RNN

> RNN xử lý các token một lần. Transformers xử lý tất cả các token cùng một lúc. Đặt cược kiến trúc duy nhất đó đã thay đổi mọi đường cong quy mô trong học sâu sau năm 2017.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## Vấn đề

Trước năm 2017, mọi mô hình trình tự hiện đại trên hành tinh  ngôn ngữ, dịch, ngôn ngữ  là một mạng lưới thần kinh tái phát. LSTM và GRU đã giành được các tiêu chuẩn dịch thuật tương đương với ImageNet trong nửa thập kỷ.

Họ có ba điểm yếu chết người. tính toán theo trình tự có nghĩa là bạn không thể song song dọc theo trục thời gian: token`t+1`cần trạng thái ẩn từ token `t`Một chuỗi 1.024 token có nghĩa là 1.024 bước liên tục trên GPU có thể thực hiện 1.000.000 hoạt động điểm nổi mỗi chu kỳ. Thời gian đào tạo tường-thành tròn theo chiều dài chuỗi trên phần cứng được thiết kế cho sự song song.

Các gradient biến mất có nghĩa là thông tin 50 token trở lại đã bị nén thông qua 50 không tuyến tính. Các đơn vị lặp lại được gài (LSTM, GRU) làm mềm sự đập vỡ nhưng không bao giờ loại bỏ nó.

Các trạng thái ẩn chiều rộng cố định có nghĩa là bộ mã hóa đã ép toàn bộ chuỗi nguồn thành một vector trước khi bộ mã hóa thấy bất cứ điều gì. Không quan trọng liệu nguồn là 5 token hay 500; nút thắt chai là hình dạng tương tự.

Bài báo năm 2017 "Trông tâm là tất cả những gì bạn cần" đề xuất một điều cực đoan: bỏ lại sự tái phát hoàn toàn. Hãy để mỗi vị trí chăm sóc mọi vị trí khác song song. Tập trong một số nhân tử lớn thay vì 1.024 thứ tự.

Kết quả thống trị mọi phương thức vào năm 2026. Ngôn ngữ (GPT-5, Claude 4, Llama 4), thị giác (ViT, DINOv2, SAM 3), âm thanh (Whisper), sinh học (AlphaFold 3), robot (RT-2).

## Khái niệm

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck.**Một RNN tính toán $h_t = f(h_{t-1}, x_t)$Mỗi bước phụ thuộc vào bước trước.$h_5$trước đây$h_4$Trên các GPU hiện đại với hơn 10.000 lõi song song, điều này lãng phí 99% silicon trên một chuỗi dài.

**Attention as a broadcast.**- Lưu ý về sự chú ý của chính mình`output_i = sum_j(a_ij * v_j)`cho mỗi cặp $(i, j)$cả nxn sự chú ý của các bộ viền chứa trong một loạt các matmul. không bước phụ thuộc vào một bước khác. GPUs thích nó.

**The speedup is not a constant.**Đó là sự khác biệt giữa $O(N)$độ sâu hàng loạt và$O(1)$Trong thực tế, các bộ biến chuyển tập luyện nhanh hơn 510x mỗi thời đại trên phần cứng phù hợp ở N=512, và khoảng cách mở rộng theo chiều dài chuỗi cho đến khi bạn chạm vào $O(N^2)$tường bộ nhớ của sự chú ý (mà Flash Attention sau đó sửa chữa  xem Bài học 12).

**What transformers cost.**Tăng cường bộ nhớ chú ý như $O(N^2)$Đối với bối cảnh 2K, tốt. Đối với bối cảnh 128K, bạn cần cửa sổ trượt, ngoại phân RoPE, flash chú ý, hoặc biến thể chú ý tuyến tính.$O(N)$trong cả thời gian và trí nhớ; những người biến đổi đổi thời gian với trí nhớ và sau đó giành lại thời gian thông qua sự song song.

**The inductive bias shift.**RNN giả định địa phương và tính gần đây. Các biến thể không giả định gì  mỗi cặp là ứng cử viên để chú ý. Đó là lý do tại sao các biến thể cần nhiều dữ liệu hơn để đào tạo tốt nhưng mở rộng hơn khi họ có nó. Chinchilla (2022) đã chính thức hóa điều này: nếu có đủ token, một biến thể luôn đánh bại một RNN với số lượng tham số bằng nhau.

```figure
rnn-vs-parallel
```

## Hãy xây dựng nó

Không có mạng thần kinh ở đây  chúng tôi mô phỏng nút thắt chai lõi theo số để bạn cảm thấy khoảng trống trên máy tính xách tay của bạn.

### Bước 1: đo độ sâu hàng loạt

Nhìn xem`code/main.py`Chúng ta xây dựng hai hàm. Một mã hóa một chuỗi như một chuỗi các sự gia tăng (serial, như một RNN). Một mã hóa nó như một giảm song song (broadcast, như sự chú ý).

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Chúng ta có thể ghi lại thời gian của cả hai trên chuỗi lên đến 100.000 yếu tố.$O(N)$Ngay cả trong Python tinh khiết, sự giảm độ tập trung vượt qua nó ở độ dài ≥ 1.000 bởi vì Python `sum()`được thực hiện bằng C và lặp lại mà không có chi phí thông dịch viên trên mỗi bước.

### Bước 2: tính toán các hoạt động lý thuyết

Cả hai thuật toán đều thêm N. Sự khác biệt là * độ sâu phụ thuộc *: bao nhiêu hoạt động phải xảy ra theo trình tự trước khi tiếp theo có thể bắt đầu. RNN độ sâu = N. Độ sâu chú ý = log(N) với một giảm cây, hoặc 1 với một quét song song. Độ sâu, không phải số op, quyết định thời gian GPU.

### Bước 3: quy mô thực nghiệm trên các chuỗi dài

Chúng tôi in một bảng thời gian làm cho $O(N)$Trong máy tính xách tay Mac 2026, các chuỗi dưới 1.000 yếu tố quá nhanh để đo. Các chuỗi 100.000 cho thấy một quét tuyến tính sạch. Đánh giá điều đó đến một biến thể mã thông báo 16,384 với một tương đương LSTM 12 tầng và bạn thấy tại sao tập luyện đồng hồ tường là một chất chặn trong năm 2016.

## Sử dụng nó

Khi nào để chọn một RNN vào năm 2026:

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

Các mô hình không gian nhà nước (SSM) như Mamba về cơ bản là RNN với các tham số cấu trúc cho họ tốt nhất của cả hai: $O(N)$Tự do học tập của các nhà khoa học đã được thực hiện trong các năm nay, và các nhà khoa học đã được nghiên cứu về các phương pháp học học.

## Chuyển nó

Nhìn xem`outputs/skill-architecture-picker.md`. Kỹ năng chọn một kiến trúc cho một vấn đề chuỗi mới do độ dài, thông suất và hạn chế ngân sách đào tạo.

## Các bài tập

1. **Easy.**Nhận đi`rnn_style`từ `code/main.py`và thay thế trạng thái ẩn scalar bằng một chiều dài-64 vector của trạng thái ẩn.
2. **Medium.**Thực hiện một tổng tiền tố song song (Hillis-Steele scan) trong Python tinh khiết. Kiểm tra nó tạo ra cùng một đầu ra số như một quét hàng loạt trên chiều dài 1024.
3. **Hard.**Đưa giảm kiểu chú ý vào PyTorch trên GPU. Thời gian cả hai khi bạn lau dọc chuỗi từ 64 đến 65.536.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
| Attention | "Let tokens look at each other" | Weighted sum $\sum_j a_{ij} v_j$ where $a_{ij}$ comes from a similarity score between positions $i$ and $j$. |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = $O(N^2)$ in sequence length; Flash Attention hides the constants, not the scaling. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) bài báo đã giết chết sự tái phát trong NLP chính thống.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) nơi sự chú ý được sinh ra, được gắn vào một RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) giấy LSTM gốc, để ghi lại.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) câu trả lời tái phát hiện đại cho các bộ biến đổi.
