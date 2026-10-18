# Tầm nhìn từ ngữ mở  CLIP

> Hãy tập hợp mã hóa hình ảnh và mã hóa văn bản để các cặp phù hợp (phần hình ảnh, tiêu đề) đến cùng một điểm trong không gian chia sẻ. Đó là toàn bộ thủ thuật.

**Type:** Build + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 17 (Self-Supervised)
**Time:** ~45 minutes

## Mục tiêu học tập

- Giải thích kiến trúc hai tháp của CLIP và mục tiêu đào tạo tương phản
- Sử dụng CLIP (hoặc SigLIP) được đào tạo trước để phân loại không bắn mà không cần đào tạo cụ thể về nhiệm vụ
- Thực hiện phân loại chụp không từ đầu: mã hóa các class prompt, tính toán cosine tương tự, lấy argmax
- Sự phân biệt giữa CLIP, SigLIP, OpenCLIP và mô hình tầm nhìn LLaVA/LLaMA  mỗi mô hình này là gì vào năm 2026

## Vấn đề

Các phân loại truyền thống là từ vựng đóng kín: mô hình ImageNet lớp 1000 chỉ có thể dự đoán 1000 nhãn. Mỗi loại mới đòi hỏi dữ liệu được dán nhãn và một đầu được đào tạo lại.

CLIP (Radford et al., OpenAI 2021) cho thấy rằng đào tạo trên 400M (hình ảnh, tiêu đề) cặp được cạn ra từ web tạo ra một mô hình có thể phân loại thành bất kỳ bộ các loại nào theo suy luận, được mô tả hoàn toàn bằng ngôn ngữ tự nhiên. Bạn đưa ra một lớp học mới bằng cách viết một câu.

Khả năng chuyển đổi không ảnh là lý do tại sao mọi hệ thống thị giác hiện đại bắt đầu với một điểm kiểm soát CLIP-family. Khám phá (Grounding DINO, OWL-ViT), phân đoạn (CLIPSeg, SAM), tìm kiếm, điều chỉnh nội dung, VLM và tạo văn bản-được hình ảnh đều dựa trên nhúng chung kiểu CLIP.

## Khái niệm

### Hai tháp

```mermaid
flowchart LR
    IMG["Image"] --> IENC["Image encoder<br/>(ViT-L/14)"] --> IEMB["Image embedding<br/>(1024,)"]
    TXT["Caption"] --> TENC["Text encoder<br/>(transformer)"] --> TEMB["Text embedding<br/>(1024,)"]
    IEMB --> SIM["Cosine similarity"]
    TEMB --> SIM

    style IENC fill:#dbeafe,stroke:#2563eb
    style TENC fill:#fef3c7,stroke:#d97706
    style SIM fill:#dcfce7,stroke:#16a34a
```

Cả hai bộ mã hóa kết thúc bằng một dự án tuyến tính đến cùng một chiều sâu nhúng (512 cho CLIP-B/32, 1024 cho CLIP-L/14).

### Mục tiêu

Với một loạt $N$(hình, caption) cặp, xây dựng một $N \times N$Các mã hóa được tập hợp để các đường vạch (cặp phù hợp) có sự tương đồng cao và các đường vạch ngoài (không phù hợp) có sự tương đồng thấp.

```
sim_matrix = image_embeddings @ text_embeddings.T / tau

loss_i2t = cross_entropy(sim_matrix,       targets=arange(N))
loss_t2i = cross_entropy(sim_matrix.T,     targets=arange(N))
loss = (loss_i2t + loss_t2i) / 2
```

Tương đối bởi vì cả việc lấy lại từ hình ảnh sang văn bản và văn bản sang hình ảnh đều phải hoạt động. `tau`(giảm nhiệt độ) thường được học như là một tham số scalar, khởi đầu đến 0,07.

### Siglip: một tổn thất tốt hơn

SigLIP (Zhai et al., 2023) đã thay thế softmax bằng sigmoid mỗi cặp:

```math
\begin{aligned}
\text{loss} &= \mathop{\mathrm{mean}}\limits_{\text{pairs}} \log\left(1 + \exp(-y_{ij} \cdot \text{sim}_{ij})\right) \\
y_{ij} &= \begin{cases} +1 & \text{if matching} \\ -1 & \text{otherwise} \end{cases}
\end{aligned}
```

Thiệt hại mỗi cặp loại bỏ sự bình thường hóa cấp lô mà CLIP yêu cầu. SigLIP đào tạo tốt hơn ở kích thước lô nhỏ và phù hợp hoặc vượt quá CLIP ở dữ liệu bằng nhau.

### Định dạng không bắn

Với CLIP được đào tạo:

1. Đối với mỗi lớp, soạn một lời nhắc: "một bức ảnh của một { lớp}".
2. Mã hóa tất cả các lệnh lớp bằng mã hóa văn bản -> `T`hình dạng (C, d).
3. Mã hóa hình ảnh thử nghiệm -> `I`hình dạng (1, d).
4. Tương tự = `I @ T.T`hình dạng (1, C).
5. Argmax -> lớp dự đoán.

Các vấn đề kỹ thuật nhanh. OpenAI đã xuất bản 80 mẫu nhanh cho ImageNet ("một bức ảnh của {}", "một bức ảnh mờ của {}", "một bản phác thảo của {}", ...).

### Khi các mô hình CLIP được sử dụng vào năm 2026

- **Zero-shot classification** sử dụng trực tiếp.
- **Image retrieval** mã hóa tất cả hình ảnh một lần, nhúng truy vấn tại suy luận.
- **Text-conditioned detection** Địa điểm DINO, OWL-ViT lắp một tháp văn bản CLIP xung quanh một máy dò.
- **Text-conditioned segmentation** CLIPSeg; SAM sử dụng các đầu vào văn bản thông qua CLIP.
- **VLMs** LLaVA, Qwen-VL, InternVL dây một CLIP-chủ hình ảnh gia đình mã hóa thành một LLM.
- **Text-to-image gen** Sự pha trộn ổn định, điều kiện DALL-E 3 trên các bản ghi văn bản CLIP.

Khi bạn có một không gian nhúng chung, mỗi nhiệm vụ thị giác + ngôn ngữ trở thành một tính toán khoảng cách.

```figure
clip-contrastive
```

## Hãy xây dựng nó

### Bước 1: Một mô hình nhỏ hai tháp

CLIP thực sự là ViT + biến đổi. Đối với bài học này các tháp là MLP nhỏ trên các tính năng được khai thác trước để tín hiệu đào tạo được nhìn thấy trên CPU.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, img_in=128, txt_in=64, emb=64):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(img_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.text_proj = nn.Sequential(nn.Linear(txt_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

    def forward(self, img_feats, txt_feats):
        i = F.normalize(self.image_proj(img_feats), dim=-1)
        t = F.normalize(self.text_proj(txt_feats), dim=-1)
        return i, t, self.logit_scale.exp()
```

Hai dự đoán, phát ra chia sẻ độ mờ, nhiệt độ học được.

### Bước 2: Khối thấu

```python
def clip_loss(image_emb, text_emb, logit_scale):
    N = image_emb.size(0)
    sim = logit_scale * image_emb @ text_emb.T
    targets = torch.arange(N, device=sim.device)
    l_i = F.cross_entropy(sim, targets)
    l_t = F.cross_entropy(sim.T, targets)
    return (l_i + l_t) / 2
```

Tương đối. Skala logit_scale cao hơn = Softmax sắc nét hơn = tự tin hơn nhưng có nguy cơ bất ổn.

### Bước 3: Định dạng 0-shot

```python
@torch.no_grad()
def zero_shot_classify(model, image_feats, class_text_feats, class_names):
    """
    image_feats:      (N, img_in)
    class_text_feats: (C, txt_in)   one averaged embedding per class
    """
    i = F.normalize(model.image_proj(image_feats), dim=-1)
    t = F.normalize(model.text_proj(class_text_feats), dim=-1)
    sim = i @ t.T
    pred = sim.argmax(dim=-1)
    return [class_names[p] for p in pred.tolist()]
```

Đây là thủ tục chụp không chính xác được sử dụng với một điểm kiểm soát CLIP sản xuất.

### Bước 4: Kiểm tra sức khoẻ

```python
torch.manual_seed(0)
model = TwoTower()

img = torch.randn(8, 128)
txt = torch.randn(8, 64)
i, t, scale = model(img, txt)
loss = clip_loss(i, t, scale)
print(f"batch size: {i.size(0)}   loss: {loss.item():.3f}")
```

Lối mất sẽ gần như là `log(N) = log(8) = 2.08`cho một mô hình được khởi tạo ngẫu nhiên  mục tiêu giao hợp giao hợp khi chưa được học được cấu trúc.

## Sử dụng nó

OpenCLIP là mặc định của cộng đồng vào năm 2026:

```python
import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = tokenizer(["a photo of a dog", "a photo of a cat", "a photo of a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(probs)
```

SigLIP mới hơn, đào tạo tốt hơn ở quy mô nhỏ, và được ưa thích cho công việc mới:`google/siglip-base-patch16-224`- Nhìn cả hai mặt.

## Chuyển nó

Bài học này mang lại:

- `outputs/prompt-zero-shot-class-picker.md` một lời nhắc thiết kế các mẫu lớp cho CLIP không chụp được một danh sách các lớp và một miền.
- `outputs/skill-image-text-retriever.md` một kỹ năng xây dựng một chỉ số nhúng hình ảnh với bất kỳ điểm kiểm tra CLIP nào, hỗ trợ truy vấn theo văn bản và truy vấn theo hình ảnh.

## Các bài tập

1. **(Easy)**Sử dụng một OpenCLIP ViT-B/32 được đào tạo trước và thực hiện phân loại chụp không trên CIFAR-10 với bộ yêu cầu mẫu 80.
2. **(Medium)**So sánh một mẫu đơn ("một bức ảnh của {}") so với 80 mẫu trung bình nhúng trên cùng một nhiệm vụ CIFAR-10.
3. **(Hard)**Xây dựng chỉ số lấy lại hình ảnh không chụp: nhúng 1.000 hình ảnh bằng CLIP, xây dựng chỉ số FAISS, truy vấn bằng mô tả ngôn ngữ tự nhiên. Báo cáo lấy lại recall@5 cho 20 truy vấn được giữ trong bạn viết bằng tay.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Two-tower | "Dual encoder" | Separate image and text encoders ending in a shared-dim projection head |
| Zero-shot | "No task-specific training" | Classify into classes described only by text at inference; no labels touched |
| Temperature / logit_scale | "tau" | Learned scalar that scales the similarity matrix before softmax |
| Prompt template | "A photo of a {}" | Natural-language wrapper around class names; averaging many templates boosts zero-shot accuracy |
| CLIP | "Image+text model" | The 2021 OpenAI model; vocabulary of the field in 2026 |
| SigLIP | "Sigmoid CLIP" | Swaps softmax for per-pair sigmoid; trains better at small batches |
| OpenCLIP | "Open reproduction" | Community-trained CLIP variants on LAION; production default for open-source pipelines |
| VLM | "Vision-language model" | A CLIP-family encoder plus an LLM, trained to answer questions about images |

## Đọc thêm

- [CLIP: Learning Transferable Visual Models from Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [SigLIP: Sigmoid Loss for Language-Image Pre-Training (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) cơ sở mã cộng đồng
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) giấy, với các điểm tham khảo tính năng so với các mô hình CLIP và MAE
