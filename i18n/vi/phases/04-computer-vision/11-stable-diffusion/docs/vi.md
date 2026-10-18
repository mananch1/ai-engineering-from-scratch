# Sự pha trộn ổn định  Kiến trúc & Định vị

> Stable Diffusion là một DDPM chạy trong không gian ẩn của một VAE được đào tạo trước, điều chỉnh trên văn bản thông qua sự chú ý chéo, lấy mẫu bằng một giải pháp ODE xác định nhanh, và được điều khiển bằng hướng dẫn không có phân loại.

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 10 (Diffusion), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~75 minutes

## Mục tiêu học tập

- Theo dõi năm phần của một đường ống dẫn truyền ổn định: VAE, mã hóa văn bản, U-Net, lập trình, kiểm tra an toàn  và mỗi phần thực sự làm gì
- Giải thích sự phân tán ẩn và lý do tại sao việc đào tạo trong không gian ẩn 4x64x64 (nhưng là hình ảnh 3x512x512) làm giảm tính toán 48x mà không mất chất lượng
- Sử dụng `diffusers`để tạo ra hình ảnh, chạy hình ảnh-to-photos, inpainting, và ControlNet dẫn đầu
- Phân phối tinh tế Stable Diffusion với LoRA trên một bộ dữ liệu tùy chỉnh nhỏ và tải bộ chuyển đổi LoRA khi suy luận

## Vấn đề

Việc đào tạo DDPM trực tiếp trên hình ảnh RGB 512x512 là tốn kém. Mỗi bước đào tạo trở lại qua một U-Net thấy giá trị đầu vào 3x512x512 = 786,432, và lấy mẫu 50+ đi về phía trước qua cùng một U-Net. Ở mức chất lượng của Stable Diffusion 1.5 (được phát hành năm 2022), sự pha trộn không gian pixel sẽ cần khoảng 256 tháng đào tạo GPU và 10-30 giây mỗi hình ảnh trên GPU tiêu dùng.

Trận thuật khiến cho việc mở văn bản sang hình ảnh trở nên thực tế là**latent diffusion**(Rombach et al., CVPR 2022). Đào tạo một VAE mà lập bản đồ một hình ảnh 3x512x512 đến một tensor ẩn 4x64x64 và trở lại, sau đó thực hiện sự pha trộn trong không gian ẩn đó.`(3*512*512)/(4*64*64) = 48x`- Phân tích giảm từ 10 giây xuống dưới 2 giây trên cùng một GPU.

Hầu như mọi mô hình tạo hình ảnh hiện đại  SDXL, SD3, FLUX, HunyuanDiT, Wan-Video  là mô hình phân tán ẩn với sự thay đổi trên mã hóa tự động, trình báo (U-Net hoặc DiT), và điều kiện văn bản. Tìm hiểu Stable Diffusion và bạn đã học được mẫu.

## Khái niệm

### Đường ống

```mermaid
flowchart LR
    TXT["Text prompt"] --> TE["Text encoder<br/>(CLIP-L or T5)"]
    TE --> CT["Text<br/>embedding"]

    NOISE["Noise<br/>4x64x64"] --> UNET["UNet<br/>(denoiser with<br/>cross-attention<br/>to text)"]
    CT --> UNET

    UNET --> SCHED["Scheduler<br/>(DPM-Solver++,<br/>Euler)"]
    SCHED --> LATENT["Clean latent<br/>4x64x64"]
    LATENT --> VAE["VAE decoder"]
    VAE --> IMG["512x512<br/>RGB image"]

    style TE fill:#dbeafe,stroke:#2563eb
    style UNET fill:#fef3c7,stroke:#d97706
    style SCHED fill:#fecaca,stroke:#dc2626
    style IMG fill:#dcfce7,stroke:#16a34a
```

- **VAE** tự động mã hóa đóng băng. Mã hóa biến hình ảnh thành laten (được sử dụng cho img2img và đào tạo).
- **Text encoder** CLIP text encoder (SD 1.x/2.x), CLIP-L + CLIP-G (SDXL), hoặc T5-XXL (SD3/FLUX).
- **U-Net** trình báo. Có các lớp chú ý chéo tham gia từ laten đến văn bản nhúng ở mọi mức độ độ độ phân giải.
- **Scheduler** thuật toán lấy mẫu (DDIM, Euler, DPM-Solver++).
- **Safety checker** bộ lọc NSFW / nội dung bất hợp pháp tùy chọn trên hình ảnh đầu ra.

### Hướng dẫn không có phân loại (CFG)

Điều kiện văn bản đơn học `epsilon_theta(x_t, t, c)`cho mỗi lần gọi `c`CFG đào tạo cùng một mạng với `c`giảm 10% thời gian (được thay thế bằng một nhúng trống), đưa ra một mô hình duy nhất dự đoán cả tiếng ồn có điều kiện và không điều kiện.

```math
\epsilon = \epsilon_{\text{uncond}} + w \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})
```

`w`là thang đo hướng dẫn. `w=0`là không có điều kiện,`w=1`là hoàn toàn có điều kiện,`w>1`đẩy đầu ra hướng tới "còn điều kiện hơn trên prompt" với chi phí của sự đa dạng.`w=7.5`- Tôi không biết.

CFG là lý do text-to-image hoạt động ở chất lượng sản xuất. Nếu không có CFG, các yêu cầu định hướng sản xuất yếu; với nó, các yêu cầu thống trị.

### Xơ cấu không gian trần gian

VAE's 4-channel latente không chỉ là một hình ảnh nén. Đây là một đa dạng nơi toán học tương ứng với các chỉnh sửa ngữ nghĩa (kỹ thuật nhanh + phân tích cả hai sống ở đây), và nơi U-Net phân tán đã được đào tạo để chi tiêu toàn bộ ngân sách mô hình hóa của nó. Việc giải mã một laten tình cờ 4x64x64 không tạo ra một hình ảnh trông tình cờ  nó tạo ra rác, bởi vì chỉ có một số lượng nhỏ cụ thể của các laten giải mã cho hình ảnh hợp lệ.

Hai hậu quả:

1. **Img2img**= mã hóa hình ảnh thành ẩn, thêm tiếng ồn một phần, chạy biểu thị, giải mã.
2. **Inpainting**= giống như img2img nhưng các tên chỉ cập nhật các vùng che giấu; các vùng không che giấu được giữ ở mật mã ẩn.

### Kiến trúc U-Net

SD U-Net là một phiên bản lớn của TinyUNet từ Bài học 10 với ba bổ sung:

- **Transformer blocks**ở mỗi độ phân giải không gian, chứa sự chú ý tự tính + sự chú ý qua văn bản nhúng.
- **Time embedding**qua MLP trên mã hóa sinus.
- **Skip connections**giữa mã hóa và mã hóa khi độ phân giải phù hợp.

Tổng tham số trong SD 1.5: ~860M. SDXL: ~2.6B. FLUX: ~12B. Sự nhảy trong các param chủ yếu là trong các lớp chú ý.

### LoRA tinh chỉnh

Việc điều chỉnh hoàn chỉnh của Stable Diffusion cần 20 GB VRAM và cập nhật các tham số 860M. LoRA (Lower-Rank Adaptation) giữ cho mô hình cơ bản được đóng băng và tiêm các matrices phân hủy cấp độ nhỏ vào các lớp chú ý. Một bộ điều chỉnh LoRA cho SD thường là 10-50 MB, chạy trong 10-60 phút trên một GPU tiêu dùng duy nhất, và tải vào thời gian suy luận như một sửa đổi giảm.

```
Original: W_q : (d_in, d_out)   frozen
LoRA:     W_q + alpha * (A @ B)   where A : (d_in, r), B : (r, d_out)

r is typically 4-32.
```

LoRA là cách phân phối hầu hết các bản nhạc của cộng đồng. CivitAI và Hugging Face lưu trữ hàng triệu bản nhạc.

### Các lịch trình bạn sẽ thấy

- **DDIM** xác định, ~ 50 bước, đơn giản.
- **Euler ancestral** Stochastic, 30-50 bước, một chút sáng tạo hơn các mẫu.
- **DPM-Solver++ 2M Karras** định nghĩa, 20-30 bước, sản xuất mặc định.
- **LCM / TCD / Turbo** mô hình phù hợp và các biến thể chưng cất; 1-4 bước với chi phí của một số chất lượng.

Thay đổi lịch trình là một thay đổi đơn tuyến trong `diffusers`và đôi khi sửa chữa các vấn đề mẫu mà không cần đào tạo lại.

```figure
cv3-latent-compression
```

## Hãy xây dựng nó

Bài học này sử dụng`diffusers`End-to-end thay vì xây dựng lại Stable Diffusion từ đầu. Các mảnh bạn cần xây dựng lại (VAE, mã hóa văn bản, U-Net, lập trình) là chủ đề của bài học của riêng họ; ở đây mục tiêu là thông thạo với API sản xuất.

### Bước 1: Giao dịch văn bản sang hình ảnh

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    prompt="a dog riding a skateboard in tokyo, studio ghibli style",
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("dog.png")
```

`float16`giảm VRAM một nửa mà không có sự mất chất lượng rõ ràng. `num_inference_steps=25`với các kết hợp DPM-Solver++ mặc định `num_inference_steps=50`với DDIM.

### Bước 2: Thay đổi lập trình

```python
from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
```

Các nhà lập trình được tách khỏi trọng lượng U-Net. Bạn có thể tập luyện về DDPM và lấy mẫu với bất kỳ lập trình viên nào.

### Bước 3: Hình ảnh sang hình ảnh

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

init_image = Image.open("dog.png").convert("RGB").resize((512, 512))
out = img2img(
    prompt="a dog riding a skateboard, oil painting",
    image=init_image,
    strength=0.6,
    guidance_scale=7.5,
).images[0]
```

`strength`là bao nhiêu tiếng ồn phải thêm trước khi bỏ đi (0.0 = không thay đổi, 1.0 = tái tạo đầy đủ).

### Bước 4: Đơn sơn

```python
from diffusers import StableDiffusionInpaintPipeline

inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

image = Image.open("dog.png").convert("RGB").resize((512, 512))
mask = Image.open("dog_mask.png").convert("L").resize((512, 512))

out = inpaint(
    prompt="a cat",
    image=image,
    mask_image=mask,
    guidance_scale=7.5,
).images[0]
```

Các pixel trắng trong mặt nạ là khu vực để tái tạo.

### Bước 5: LoRA

```python
pipe.load_lora_weights(
    "artificialguybr/studioghibli-redmond-1-5v-studio-ghibli-lora-for-liberteredmond-sd-1-5",
    weight_name="StudioGhibliRedmond-15V-LiberteRedmond-StdGBRedmAF-StudioGhibli.safetensors",
)
pipe.fuse_lora(lora_scale=0.8)

image = pipe(prompt="a village square, StdGBRedmAF, Studio Ghibli").images[0]
```

Câu khởi động từ thẻ mô hình (`StdGBRedmAF, Studio Ghibli`) bật phong cách. `lora_scale`kiểm soát sức mạnh; 0,0 = không có hiệu ứng, 1,0 = hiệu ứng đầy đủ. `fuse_lora`làm cho bộ điều chỉnh nạp vào trọng lượng để tăng tốc, nhưng ngăn chặn sự thay đổi.`pipe.unfuse_lora()`trước khi tải một bộ chuyển đổi khác.

### Bước 6: đào tạo LoRA (phác thảo)

LoRA thực sự được đào tạo trong `peft`hoặc `diffusers.training`- Khía sơ:

```python
# Pseudocode
for step, batch in enumerate(dataloader):
    images, prompts = batch
    latents = vae.encode(images).latent_dist.sample() * 0.18215

    t = torch.randint(0, num_train_timesteps, (batch_size,))
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    text_emb = text_encoder(tokenizer(prompts))

    pred_noise = unet(noisy_latents, t, text_emb)  # LoRA weights injected here

    loss = F.mse_loss(pred_noise, noise)
    loss.backward()
    optimizer.step()
```

Chỉ có các matrices LoRA nhận gradient; U-Net cơ sở, VAE và mã hóa văn bản được đóng băng. Với kích thước lô 1 và điểm kiểm tra gradient, điều này phù hợp với 8 GB VRAM.

## Sử dụng nó

Trong sản xuất, những quyết định mà bạn thực sự đưa ra:

- **Model family**: SD 1.5 cho các bản nhạc cộng đồng nguồn mở, SDXL cho độ trung thực cao hơn, SD3 / FLUX cho trạng thái hiện đại và các yêu cầu cấp phép nghiêm ngặt.
- **Scheduler**: DPM-Solver++ 2M Karras cho 20-30 bước, LCM-LoRA khi độ trễ dưới 1s.
- **Precision**`float16`trên 4080/4090, `bfloat16`trên A100 và mới hơn, `int8`(via `bitsandbytes`hoặc `compel`) khi VRAM bị chặt chẽ.
- **Conditioning**: văn bản đơn giản hoạt động; để kiểm soát mạnh hơn, thêm ControlNet (có thể, độ sâu, tư thế) trên đầu của đường ống cơ sở.

Đối với thế hệ hàng loạt, `AUTO1111`- `ComfyUI`là các công cụ của cộng đồng; cho các API sản xuất,`diffusers`+ `accelerate`hoặc `optimum-nvidia`với TensorRT compilation.

## Chuyển nó

Bài học này mang lại:

- `outputs/prompt-sd-pipeline-planner.md` một lời nhắc chọn SD 1.5 / SDXL / SD3 / FLUX cộng với lập trình và độ chính xác do ngân sách thời gian trễ, mục tiêu trung thành và hạn chế cấp phép.
- `outputs/skill-lora-training-setup.md` một kỹ năng viết một cấu hình đào tạo LoRA đầy đủ cho một bộ dữ liệu tùy chỉnh bao gồm tiêu đề, cấp bậc, kích thước lô và tốc độ học tập.

## Các bài tập

1. **(Easy)**Tạo cùng một prompt với `guidance_scale`trong `[1, 3, 5, 7.5, 10, 15]`- Mô tả hình ảnh thay đổi như thế nào.
2. **(Medium)**Hãy chụp ảnh thật, xem qua.`StableDiffusionImg2ImgPipeline``strength`trong `[0.2, 0.4, 0.6, 0.8, 1.0]`Nguyên nhân nào giữ lại thành phần trong khi thay đổi phong cách? Tại sao 1.0 bỏ qua đầu vào hoàn toàn?
3. **(Hard)**Cải tạo LoRA trên 10-20 hình ảnh của một đối tượng (một con vật nuôi, logo, một nhân vật) và tạo ra những cảnh mới với đối tượng đó trong chúng.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Latent diffusion | "Diffuse in latents" | Run the entire DDPM in the VAE latent space (4x64x64) instead of pixel space (3x512x512); 48x compute saving |
| VAE scale factor | "0.18215" | Constant that rescales the VAE's raw latent to roughly unit variance; hardcoded in every SD pipeline |
| Classifier-free guidance | "CFG" | Mix conditional and unconditional noise predictions; the single most impactful inference knob |
| Scheduler | "Sampler" | The algorithm that turns noise + model predictions into a denoised latent trajectory |
| LoRA | "Low-rank adapter" | Small rank-decomposition matrices that fine-tune attention layers without touching base weights |
| Cross-attention | "Text-image attention" | Attention from latent tokens to text tokens; injects prompt information at every U-Net level |
| ControlNet | "Structure conditioning" | A separately-trained adapter that steers SD with an extra input (canny, depth, pose, segmentation) |
| DPM-Solver++ | "The default scheduler" | Second-order deterministic ODE solver; best quality at low step counts (20-30) in 2026 |

## Đọc thêm

- [High-Resolution Image Synthesis with Latent Diffusion (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752) giấy Stable Diffusion; bao gồm mọi loại bỏ hợp lý cho thiết kế
- [Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598) giấy CFG
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) LoRA là NLP đầu tiên; nó chuyển sang SD hầu như không có thay đổi
- [diffusers documentation](https://huggingface.co/docs/diffusers) tham chiếu cho mỗi đường ống SD / SDXL / SD3 / FLUX
