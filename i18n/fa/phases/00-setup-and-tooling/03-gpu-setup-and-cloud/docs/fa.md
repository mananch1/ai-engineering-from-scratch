# تنظیم GPU و ابر

> آموزش در مورد پردازنده برای یادگیری خوب است آموزش برای واقعی نیاز به GPU

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## اهداف یادگیری

- دسترسی GPU محلی را با استفاده از `nvidia-smi`و API CUDA PyTorch
- تنظیم Google Colab با یک GPU T4 برای آزمایش های رایگان مبتنی بر ابر
- ضرب ماتریکس بر روی CPU در مقابل GPU را نشان دهید و سرعت را اندازه گیری کنید
- با استفاده از قانون انگشت fp16 بزرگترین مدل را که در VRAM شما قرار می گیرد تخمین بزنید

## مشکل

بیشتر درس های مرحله ی ۱ تا ۳ در CPU خوب انجام می شود. اما هنگامی که شما شروع به آموزش CNNs، ترانسفورماتورها یا LLM ها (فاز ۴+) می کنید، نیاز به سرعت GPU دارید. یک دوره آموزشی که ۸ ساعت در CPU طول می کشد، ۱۰ دقیقه در GPU طول می کشد.

شما سه گزینه دارید: GPU محلی، GPU ابر یا Google Colab (بزدگی).

## مفهوم

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

## آن را بسازید

### گزینه ی ۱: GPU NVIDIA محلی

اگه یکی داری چک کن

```bash
nvidia-smi
```

PyTorch رو با CUDA نصب کن:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### گزینه دوم: گوگل کولاب

1. برو[colab.research.google.com](https://colab.research.google.com)
2. زمان اجرا > نوع زمان اجرا را تغییر دهید > GPU T4
3. فرار کن`!nvidia-smi`برای تایید

دفترچه های يادداشت از اين دوره رو به طور مستقیم به کولاب اپلود کن

### گزینه 3: گپتوپ های ابر

برای لامپدا لابراتوارها، RunPod یا Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### هيچ گپيوي نداره

بیشتر درس ها روی CPU کار می کنند. آنهایی که نیاز به GPU دارند اینو می گویند و لینک های Colab را نیز شامل می کنند.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## ساخت آن: GPU vs CPU benchmark

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

## تمرینات

1. معیار بالا را اجرا کنید و زمان CPU در مقابل GPU را مقایسه کنید
2. اگر GPU ندارید، آن را روی Google Colab اجرا کنید و مقایسه کنید
3. بررسی کنید که چقدر حافظه GPU دارید و بزرگترین مدل را که می توانید قرار دهید تخمین بزنید (قاعده انگشت: 2 بایت در هر پارامتر برای fp16)

### اشاره ها

**Exercise 1.**کد را از بخش ساخت آن به همان شکل اجرا کنید، سپس زمان CPU، زمان GPU و خط سرعت را بنویسید. آن را دومین بار اجرا کنید؛ اولین اجرای GPU شامل هزینه های راه اندازی است، بنابراین اعداد دوم هستند که برای مقایسه.

**Exercise 2.**در Colab، زمان اجرا > نوع زمان اجرا را تغییر دهید > GPU را انتخاب کنید، سپس معیار را به یک سلول پیوند دهید. اجرا کنید `!nvidia-smi`در سلول دیگری اول؛ باید نام GPU را قبل از اینکه نتیجه معیار چیزی به معنی باشد نشان دهد.

**Exercise 3.**حافظه کل رو با `torch.cuda.get_device_properties(0).total_memory / 1e9`، مثل`code/gpu_check.py`حافظه را به بائتهایی به 2 تقسیم کنید تا تعداد پارامتر ها برای fp16 بدست آید، سپس حاشیه ای را نگه دارید، زیرا فعال سازی ها و زمینه CUDA نیز از حافظه استفاده می کنند.

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
