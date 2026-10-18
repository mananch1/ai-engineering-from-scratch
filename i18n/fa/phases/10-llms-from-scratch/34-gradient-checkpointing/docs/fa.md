# کنترل درجه بندی و بازرس فعال سازی

> Backprop هر فعال سازی میانگین را حفظ می کند. در پارامترهای 70B و زمینه 128K که 3 TB فعال سازی در هر رتبه است. چکپینت FLOPs را برای حافظه تجارت می کند: به جای ذخیره مجدد محاسبه می کند. سوال این است که کدام بخش ها را رها کنیم، و پاسخ "همه آنها" نیست.

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## مشکل

آموزش یک ترانسفورماتور ذخیره می کند، برای هر لایه، ورودی برای هر عملیات که به عقب متمایز می شود: ورودی توجه، پروژکتورهای Q / K / V، ورودی نرم، ورودی FFN، ورودی نورم و جریان باقیمانده. برای یک لایه با اندازه پنهان `d`، طول دنباله`L`، دسته`B`، اين به دستور`12 * B * L * d`شناور در هر لایه

برای`d=8192, L=8192, B=1`، که 800 MB / لایه در BF16 است. یک مدل 64 لایه 51 GB از فعال سازی است و این قبل از اینکه شما ضرب با اندازه میکروباتش، قبل از اضافه کردن توجه نرمmax میانگین (`L^2`در هر سر) و قبل از اینکه نسخه های جزئی تنزور متوازمی را فاکتور کنید.

هزینه: وزن BF16 به علاوه حالت بهینه سازی ممکن است در 80GB قرار گیرد، اما فعال سازی شما را از آن دور می کند. کنترل درجه بندی (به عنوان تعدد مجدد فعال سازی) راه حل استاندارد است. اکثر فعال سازی ها را رها کنید؛ در طول عقب حرکت را دوباره انجام دهید تا آنها را به عقب برگردانید. هزینه: FLOPs اضافی. مزیت: کاهش حافظه با نسبت بخش های نقطه بازرسی به کل لایه ها.

با انجام این کار، چکپینت به طور نابینا، حدود 33 درصد هزینه FLOP های پیشروی را در هر مرحله افزایش می دهد. به خوبی انجام شده.  چکپینت انتخابی در هر "انتخاب هوشمند" کورتیکانتی و همکاران.  شما 5 برابر حافظه را برای کمتر از 5 درصد FLOP ذخیره می کنید. و با FP8 matmuls، FSDP offload، و MoE موازی متخصص این واقعا مهم است: شما نمی توانید نه حافظه و نه محاسبه تلف شده را پرداخت کنید.

## مفهوم

### آنچه که افراد عقب مانده واقعاً نیاز دارند

`output = layer(input)`. خواهان عقب نشینی`grad_input`و`grad_params`براي محاسبه اين موارد لازم است:

- `input`(برای محاسبه`grad_params = input.T @ grad_output`برای لایه های خطی)
- برخی از مشتق های واسطه فعال سازی (متغیر ReLU/GELU/softmax بستگی به ارزش فعال سازی دارد)

گذرنامه جلو به طور اتوماتیک اینو توی نمودار اوتوگراید ذخیره می کنه`tensor.retain_grad()`و هر عملیاتی که نیاز به ورودی داره یک مرجع رو نگه داره

### همه جا بازداشت

شبکه رو به قسمت ها تقسیم کن`N`در طول پیش، تنها *دخل* را به هر بخش ذخیره کنید. هنگامی که عقب نیاز به میانگین ها دارد، عبور پیش بخش را برای تحقق آنها تکرار کنید، سپس متمایز کنید.

مثال: ترانسفورماتور 32 لایه به 32 بخش 1 لایه تقسیم شده است.

- حافظه: 32 ورودی لایه (بچه) در مقابل 32 * (حجم فعال سازی در هر لایه) (بسیار).
- محاسبه اضافی: 1 واحد بیشتر به جلو در هر بخش، یعنی ~33٪ بیشتر از کل FLOP های جلو (از آنجا که عقب دو برابر به جلو است، مرحله کامل به جای 1 + 2 = 3 به 1 + 1 + 2 = 4 واحد می شود).

این نسخه اصلی چین و همکارانش است.`sqrt(L)`لايه هاي تعادل بين حافظه و محاسبات$L=64$، اين هشت نقطه بازي هست

### چکپوینت انتخابی (کورتیکانتی 2022)

همه فعال کردن ها به اندازه ي ديگه اي هم قيمت ندارن`B*L*L*heads`و به شکل * مربع* با طول دنباله رشد می کند.`B*L*4d`و خطی رشد می کند. برای دنباله های طولانی نرم حداکثر برتری دارد.

چکپینکنگ انتخابی فعال سازی های ارزان قیمت را حفظ می کند (پیش بینی های خطی، باقیمانده ها) و فقط آنهایی را که گران هستند (انتباه) دوباره محاسبه می کند.$O(L^2)$حافظه

Megatron-Core این را به عنوان "انتخاب" بازتغییر فعال سازی پیاده سازی می کند. در اکثر راه های آموزش مرز 2024 + استفاده می شود.

### تخلیه

جایگزین برای بازتغییر: فعال سازی های کشتی به RAM CPU بین جلو و عقب. نیاز به باند PCIe; مفید است زمانی که باند بیکار بیش از هزینه های ریماتریالیزاسیون است. استراتژی های مخلوط رایج هستند: چک پوند برخی لایه ها، بارگذاری دیگران.

FSDP2 به عنوان یک گزینه درجه اول تخلیه می کند. تخلیه زمانی که GPU در حافظه گیرنده است، درخشش می یابد اما انتقال CPU-GPU فضای سر دارد.

### مدل هزینه را دوباره محاسبه کنید

هر مرحله ای که با چکپوینت ساده ای هر لحظه`k`لایه های خارج از `L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

با چک پوائنٹنگ انتخابی فقط هسته توجه را دوباره محاسبه می کنید، نه کل لایه:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### مدل ذخیره حافظه

حجم فعال سازی در هر لایه: `A`. برای`L`لایه ها، حافظه فعال سازی کل: `L * A`. .

نقطه کنترل کامل (سگمان اندازه 1): فقط ذخیره کنید `L * input_volume`(~`L * 1/10 A`برای یک ترانسفورماتور استاندارد)`9 * L * A * 1/10`. .

هر لحظه`k`لایه ها: ذخیره سازی`L/k * A`و اضافه`k-1`ارزش لایه ها در بخش فعال.

در`k = sqrt(L)`, حافظه و هزینه های بازرس هر دو مقیاس با `sqrt(L)` بهترین معامله برای لایه های هزینه یکسانی.

### وقتی که نمی توانیم به نقطه بازرسی برویم

- درونی ترین لایه های مرحله ی خط لوله ها در حال پرواز هستند
- لایه های اول و آخر اگر بر محاسبه مرحله تسلط داشته باشند (در ترانسفورماتورها نادر است).
- هسته های توجه که قبلاً از FlashAttention استفاده می کنند  Flash قبلاً سرعت نرم را دوباره محاسبه می کند، بنابراین چکپوائنتینگ سطح لایه اضافی کمی اضافه می کند.

### الگوهای پیاده سازی

1. **Function wrapper:**یک بخش را در`torch.utils.checkpoint.checkpoint(fn, input)`فقط فروشگاه هاي پيترچ`input`، همه چيز ديگه رو به عقب حساب ميکنه

2. **Decorator-based:**لایه های برچسب را به عنوان نقطه کنترل قابل بررسی قرار دهید؛ مربی در زمان تنظیم تصمیم می گیرد که کدام بخش ها بسته می شوند.

3. **Manual explicit recompute:**خودت پس پشتو بنويس و بهشون دستور بده`recompute_forward`که به صورت دوگانه با ورودی ذخیره شده، به صورت پیش رو می شود.

هر سه نتیجه عملکردی مشابهی دارند. بسته بندی ها اصطلاح استاندارد هستند.

### تعامل با TP / PP / FP8

- **Tensor parallel:**ورودی های نقطه بازرسی باید در محاسبه مجدد جمع آوری یا توزیع شوند؛ هزینه های ارتباطی را مدیریت کنید.
- **Pipeline parallel:**الگوی معمول این است که هر مرحله خط لوله را به جلو چک کنید تا میکروباتش های ترتیب برعکس بتوانند حافظه فعال سازی را دوباره استفاده کنند.
- **FP8 recompute:**تاریخچه های amax که در طول بازخورد به روز می شوند باید با پیش بینی های اصلی یا انحرافات مقیاس FP8 مطابقت داشته باشند. اکثر چارچوب ها مقیاس را به صورت فوری ضبط می کنند.

```figure
activation-recompute
```

## آن را بسازید

### مرحله ی اول: یک مدل اسباب بازی با قطعات

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### مرحله دوم: ساده و عقب نشینی که نیاز به تمام فعال سازی ها دارد

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### مرحله سوم: هر نقطه ی حافظه

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### مرحله چهارم: مدل هزینه

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### مرحله پنجم: محاسب حافظه

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### مرحله 6: اندازه مطلوب بخش

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### مرحله هفتم: تصمیم گیری در نقطه بازرسی انتخابی

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## ازش استفاده کن

- **torch.utils.checkpoint**.`from torch.utils.checkpoint import checkpoint` بسته بندی کانونیکی در PyTorch. یک تابع را بسته می کند؛ تنها ورودی ها را ذخیره می کند، به عقب محاسبه می کند.
- **Megatron-Core activation recomputation**: حمایت می کند`selective`،`full`و`block`روش های استاندارد در آموزش مرز 2024+
- **FSDP2 offload**.`module.to_empty(device="cpu")`با`offload_policy`در FSDP2 فعال سازی ها را به جای بازتعداد به CPU تقسیم می کند.
- **DeepSpeed ZeRO-Offload**: بارگذاری CPU برای حالت های بهینه سازی و فعال سازی، تکمیل چکپینکنگ.

## -باده

این درس به ما کمک می کند`outputs/prompt-activation-recompute-policy.md` یک پیامک که پیکربندی مدل شما (طبقات، پنهان، seq، دسته) و حافظه GPU موجود را می گیرد و یک سیاست بازخورد هر لایه را صادر می کند (هیچ / انتخابی / کامل / تخلیه).

## تمرینات

1. دقت رو چک کن.`model_forward`+ `model_backward`(فعاليت کامل)`model_forward_checkpointed`+ `model_backward_checkpointed`(قطعات) gradients پارامتر باید یکسان با دقت ماشین باشد.

2. اندازه قطعه پاکش`k`از 1 تا `L`. نقشه فلاپ بالا و حافظه رو پيدا کن

3. کنترل انتخابی را پیاده سازی کنید: ورودی ماژول توجه را ذخیره کنید اما نه میانگین آن را. اندازه گیری FLOP Overhead vs. full-layer checkpointing برای یک مدل 32 لایه در seq=8192.

4. اضافه کردن بار تخلیه. ورودی بخش را به یک "بوفر CPU" شبیه سازی شده (یک لیست جداگانه) ذخیره کنید. "بند وایت PCIe" را به عنوان بایت / زمان اندازه گیری کنید و نقطه تعادل بین بار تخلیه و بازتعداد را پیدا کنید.

5. یک ترانسفورماتور واقعی PyTorch را با و بدون آن بنچ مارک کنید `torch.utils.checkpoint`. اندازه گیری حافظه (به وسیله `torch.cuda.max_memory_allocated`) و زمان قدم زدن

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | "Save memory by redoing forward" | Store segment inputs only; recompute intermediates during backward to get gradient-support tensors |
| Activation recomputation | "Same as checkpointing" | The HPC-flavored name for the same technique |
| Segment size (k) | "How many layers per checkpoint" | Number of layers whose intermediates are dropped and rematerialized together |
| Selective checkpointing | "Korthikanti's trick" | Recompute only expensive-to-store activations (attention softmax); keep cheap ones |
| Full checkpointing | "The naive version" | Recompute every layer's intermediates in every segment |
| Block checkpointing | "Coarse-grained" | Checkpoint whole transformer blocks; largest granularity |
| FLOP overhead | "The compute tax" | Extra FLOPs per step = (recompute FLOPs) / (fwd + bwd FLOPs); 33% naive, 5% selective |
| Activation offload | "Ship to CPU" | Move activations to CPU RAM across forward->backward; alternative to recompute |
| sqrt-L rule | "The classical optimum" | For uniform-cost layers, optimal checkpoint spacing is $\sqrt{L}$ layers |
| Attention-softmax volume | "The O(L^2) problem" | $L^2 \cdot \text{heads} \cdot \text{batch}$ floats; dominates activation memory at long contexts |

## خواندن بیشتر

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174)-- کاغذ اصلی که کنترل گرادینتی را رسمی کرد
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198)-- بازتعداد انتخابی فعال سازی و تحلیل رسمی هزینه
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645)-- رویکرد جایگزین حافظه ثابت از طریق حالت برگشت مجدد
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840)-- فعال کردن تخلیه در مقیاس
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html)-- API استاندارد
- [Megatron Bridge activation recomputation documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)-- حالت های انتخابی، کامل و بلاک
