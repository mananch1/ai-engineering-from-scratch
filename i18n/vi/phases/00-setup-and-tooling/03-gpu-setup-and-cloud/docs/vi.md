# Thiết lập GPU & Cloud

> Trình luyện trên CPU là tốt cho việc học. Trình luyện cho thực tế cần một GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Mục tiêu học tập

- Kiểm tra khả năng GPU địa phương bằng cách sử dụng `nvidia-smi`và API CUDA của PyTorch
- Cấu hình Google Colab với GPU T4 cho các thí nghiệm dựa trên đám mây miễn phí
- Đánh giá nhân số matrix trên CPU vs GPU và đo tốc độ tăng tốc
- Đếm mô hình lớn nhất phù hợp với VRAM của bạn bằng cách sử dụng quy tắc ngón tay fp16

## Vấn đề

Hầu hết các bài học trong giai đoạn 1-3 chạy tốt trên CPU. Nhưng một khi bạn bắt đầu đào tạo CNN, biến đổi, hoặc LLM (phase 4+), bạn cần tăng tốc GPU. Một cuộc đào tạo kéo dài 8 giờ trên CPU mất 10 phút trên GPU.

Bạn có ba tùy chọn: GPU địa phương, GPU đám mây, hoặc Google Colab (không).

## Khái niệm

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## Hãy xây dựng nó

### Tùy chọn 1: NVIDIA GPU địa phương

Hãy kiểm tra xem có có gì không.

```bash
nvidia-smi
```

Lắp đặt PyTorch với CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Tùy chọn 2: Google Colab

1. Đi đi[colab.research.google.com](https://colab.research.google.com)
2. Thời gian chạy > Thay đổi kiểu thời gian chạy > T4 GPU
3. Đi chạy`!nvidia-smi`để xác minh

Lên lên sổ ghi chép từ khóa học này trực tiếp đến Colab.

### Tùy chọn 3: GPU đám mây

Đối với Lambda Labs, RunPod hoặc Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Không có GPU?

Hầu hết các bài học đều hoạt động trên CPU. Những người cần GPU sẽ nói như vậy và bao gồm các liên kết Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Build It: GPU vs CPU benchmark

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Các bài tập

1. Động cơ chuẩn ở trên và so sánh thời gian CPU vs GPU
2. Nếu bạn không có GPU, chạy nó trên Google Colab và so sánh
3. Kiểm tra lượng bộ nhớ GPU bạn có và ước tính mô hình lớn nhất bạn có thể phù hợp (quyền ngón tay: 2 byte cho mỗi tham số cho fp16)

### Những gợi ý

**Exercise 1.**Hãy chạy mã từ phần Build It như hiện tại, sau đó ghi lại thời gian CPU, thời gian GPU và dòng tăng tốc.

**Exercise 2.**Trong Colab, chọn Runtime > Thay đổi loại runtime > GPU, sau đó dán điểm tham chiếu vào một tế bào.`!nvidia-smi`trong một tế bào khác trước tiên; nó phải hiển thị một tên GPU trước khi kết quả tham chiếu có ý nghĩa gì.

**Exercise 3.**Đọc bộ nhớ tổng thể với `torch.cuda.get_device_properties(0).total_memory / 1e9`, như trong `code/gpu_check.py`Chia bộ nhớ thành byte bằng 2 để có được số parameter cho fp16, sau đó giữ một biên, bởi vì các kích hoạt và ngữ cảnh CUDA cũng sử dụng bộ nhớ.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
