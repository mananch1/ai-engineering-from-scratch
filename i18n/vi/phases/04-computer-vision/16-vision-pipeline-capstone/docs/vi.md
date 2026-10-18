# Xây dựng một đường ống dẫn tầm nhìn hoàn chỉnh  Capstone

> Một hệ thống thị giác sản xuất là một chuỗi các mô hình và quy tắc được đan xen với các hợp đồng dữ liệu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lessons 01-15
**Time:** ~120 minutes

## Mục tiêu học tập

- Thiết kế một đường ống thị giác sản xuất phát hiện các đối tượng, phân loại chúng và phát ra JSON  cấu trúc với mỗi đường mòn thất bại được xử lý
- Kết nối một bộ phát hiện (Mask R-CNN hoặc YOLO), một bộ phân loại (ConvNeXt-Tiny), và một hợp đồng dữ liệu (Pydantic) vào một dịch vụ
- Đánh dấu chuẩn đường ống kết thúc đến kết thúc và xác định nút thắt đầu (thường là xử lý trước, sau đó là máy dò)
- Gửi dịch vụ FastAPI tối thiểu chấp nhận tải lên hình ảnh, chạy đường ống dẫn và trả lại phát hiện với phân loại

## Vấn đề

Các mô hình thị giác cá nhân hữu ích; các sản phẩm thị giác là chuỗi của chúng. Một kiểm toán kệ bán lẻ là một bộ dò cộng với một phân loại sản phẩm cộng với một đường ống OCR giá. Đường lái tự động là một bộ dò 2D cộng với một bộ dò 3D cộng với một bộ phân đoạn cộng với một bộ theo dõi cộng với một lập kế hoạch. Một màn hình trước y tế là một bộ phân đoạn cộng với một bộ phân loại khu vực cộng với một UI lâm sàng.

Cáp dây là phần tách biệt một nguyên mẫu ML từ một sản phẩm. Mỗi giao diện giữa các mô hình là một nơi mới cho lỗi. Mỗi chuyển đổi phối hợp, mỗi chuẩn hóa, mỗi kích thước mặt nạ là ứng cử viên thất bại im lặng. Một đường ống là mạnh như giao diện yếu nhất của nó.

Bạch đá này thiết lập đường ống dẫn khả thi tối thiểu: phát hiện + phân loại + đầu ra cấu trúc + một lớp phục vụ. Mọi thứ khác trong các khe trong giai đoạn 4 vào bộ xương này: thay đổi Mask R-CNN cho YOLOv8, thêm đầu OCR, thêm một nhánh phân đoạn, thêm một bộ theo dõi. Kiến trúc ổn định; các mảnh có thể cắm.

## Khái niệm

### Đường ống

```mermaid
flowchart LR
    REQ["HTTP request<br/>+ image bytes"] --> LOAD["Decode<br/>+ preprocess"]
    LOAD --> DET["Detector<br/>(YOLO / Mask R-CNN)"]
    DET --> CROP["Crop + resize<br/>each detection"]
    CROP --> CLS["Classifier<br/>(ConvNeXt-Tiny)"]
    CLS --> AGG["Aggregate<br/>detections + classes"]
    AGG --> SCHEMA["Pydantic<br/>validation"]
    SCHEMA --> RESP["JSON response"]

    REQ -.->|error| RESP

    style DET fill:#fef3c7,stroke:#d97706
    style CLS fill:#dbeafe,stroke:#2563eb
    style SCHEMA fill:#dcfce7,stroke:#16a34a
```

Hai giai đoạn mô hình đắt tiền, còn 5 giai đoạn khác là nơi sinh sống của côn trùng.

### Hợp đồng dữ liệu với Pydantic

Mỗi đường biên giới mô hình trở thành một đối tượng được đánh dấu. Điều này biến những thất bại im lặng thành những thất bại lớn.

```
Detection(
    box: tuple[float, float, float, float],   # (x1, y1, x2, y2), absolute pixels
    score: float,                              # [0, 1]
    class_id: int,                             # from detector's label map
    mask: Optional[list[list[int]]],           # RLE-encoded if present
)

PipelineResult(
    image_id: str,
    detections: list[Detection],
    classifications: list[Classification],
    inference_ms: float,
)
```

Khi một máy dò trả lại các hộp trong `(cx, cy, w, h)`thay vì `(x1, y1, x2, y2)`, xác nhận của Pydantic thất bại ở biên giới và bạn tìm ra ngay lập tức thay vì debugging một cây trồng dòng chảy xuống đó lặng lẽ trả lại các vùng trống.

### Khi độ trễ đi

Ba sự thật có trong hầu hết các đường ống thị giác:

1. **Preprocessing is often the biggest single block.**Việc giải mã JPEG, chuyển đổi không gian màu, đổi kích thước  chúng được gắn với CPU và dễ quên.
2. **The detector dominates GPU time.**70-90% thời gian GPU là trong phát hiện trước đi.
3. **Postprocessing (NMS, RLE encode/decode) is cheap on GPU, expensive on CPU.**Luôn luôn ghi hình với mục tiêu thực tế.

Biết phân phối là điều biến tối ưu hóa thành một danh sách ưu tiên.

### Các chế độ thất bại

- **Empty detections** trả lại danh sách trống, không bị hỏng.
- **Out-of-bounds boxes** Nhấn vào kích thước hình ảnh trước khi cắt.
- **Tiny crops** bỏ phân loại cho các hộp nhỏ hơn số lượng nhập tối thiểu của phân loại.
- **Corrupt upload** 400 phản ứng với mã lỗi cụ thể, không phải 500.
- **Model load failure** thất bại khi khởi động dịch vụ, không phải khi yêu cầu đầu tiên.

Một đường ống sản xuất xử lý từng loại này mà không cần viết chung `try/except`Mỗi thất bại đều có một mã tên và một phản ứng.

### Nhóm

Một dịch vụ sản xuất phục vụ nhiều khách hàng. Khám phát hiện và phân loại trên các yêu cầu nhân lượng. Sự đổi mới: thời gian trễ thêm từ chờ đợi một lô để lấp đầy. Thiết lập điển hình: thu thập yêu cầu cho đến 20ms, lô cùng nhau, xử lý, phân phối phản ứng. `torchserve`và `triton`làm điều này tự nhiên; các dịch vụ nhỏ với tải trọng dự đoán được lăn bộ máy vi-batcher của riêng họ.

```figure
v4-vision-pipeline
```

## Hãy xây dựng nó

### Bước 1: Hợp đồng dữ liệu

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Detection(BaseModel):
    box: Tuple[float, float, float, float]
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    mask_rle: Optional[str] = None


class Classification(BaseModel):
    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)


class PipelineResult(BaseModel):
    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float
```

Năm giây mã tiết kiệm một giờ để cố định bất kỳ đường ống nào nghiêm trọng.

### Bước 2: Một lớp đường ống tối thiểu

```python
import time
import numpy as np
import torch
from PIL import Image

class VisionPipeline:
    def __init__(self, detector, classifier, class_names,
                 device="cpu", min_crop=32):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.min_crop = min_crop

    def preprocess(self, image):
        """
        image: PIL.Image or np.ndarray (H, W, 3) uint8
        returns: CHW float tensor on device
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        return self.detector([image_tensor])[0]

    @torch.no_grad()
    def classify(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        return list(zip(cls.tolist(), scores.tolist()))

    def run(self, image, image_id="anonymous"):
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        det = self.detect(tensor)

        crops = []
        detections = []
        valid_indices = []
        for i, (box, score, cls) in enumerate(zip(det["boxes"], det["scores"], det["labels"])):
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(score),
                class_id=int(cls),
            ))
            if (x2 - x1) < self.min_crop or (y2 - y1) < self.min_crop:
                continue
            crop = tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
            valid_indices.append(i)

        class_preds = self.classify(crops)

        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=self.class_names[cls_id],
                score=float(cls_score),
            ))

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=(time.perf_counter() - t0) * 1000,
        )
```

Mỗi giao diện được gõ, mỗi đường hỏng đều có một quyết định xử lý cụ thể.

### Bước 3: Đưa một máy dò và một bộ phân loại

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny

# Use ImageNet-pretrained weights for a realistic pipeline without training
detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
classifier = convnext_tiny(weights="DEFAULT")
class_names = [f"imagenet_class_{i}" for i in range(1000)]

pipe = VisionPipeline(detector, classifier, class_names)

# Smoke test with a synthetic image
test_image = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
result = pipe.run(test_image, image_id="demo")
print(result.model_dump_json(indent=2)[:500])
```

### Bước 4: Dịch vụ FastAPI

```python
from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO

app = FastAPI()
pipe = None  # initialised on startup

@app.on_event("startup")
def load():
    global pipe
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    classifier = convnext_tiny(weights="DEFAULT").eval()
    pipe = VisionPipeline(detector, classifier, class_names=[f"c{i}" for i in range(1000)])

@app.post("/detect")
async def detect_endpoint(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported image type")
    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="cannot decode image")
    result = pipe.run(img, image_id=file.filename or "upload")
    return result.model_dump()
```

Đi cùng `uvicorn main:app --host 0.0.0.0 --port 8000`- Thử nghiệm với `curl -F 'file=@dog.jpg' http://localhost:8000/detect`- Tôi không biết.

### Bước 5: Đánh dấu đường ống dẫn

```python
import time

def benchmark(pipe, num_runs=20, image_size=(400, 600)):
    img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)
    pipe.run(img)  # warm up

    stages = {"preprocess": [], "detect": [], "classify": [], "total": []}
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tensor = pipe.preprocess(img)
        t1 = time.perf_counter()
        det = pipe.detect(tensor)
        t2 = time.perf_counter()
        crops = []
        for box in det["boxes"]:
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            if (x2 - x1) >= pipe.min_crop and (y2 - y1) >= pipe.min_crop:
                crop = tensor[:, y1:y2, x1:x2]
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                )[0]
                crops.append(crop)
        pipe.classify(crops)
        t3 = time.perf_counter()
        stages["preprocess"].append((t1 - t0) * 1000)
        stages["detect"].append((t2 - t1) * 1000)
        stages["classify"].append((t3 - t2) * 1000)
        stages["total"].append((t3 - t0) * 1000)

    for stage, times in stages.items():
        times.sort()
        print(f"{stage:12s}  p50={times[len(times)//2]:7.1f} ms  p95={times[int(len(times)*0.95)]:7.1f} ms")
```

Khả năng đầu ra điển hình trên CPU: quá trình xử lý trước ~ 3 ms, phát hiện 300-500 ms, phân loại 20-40 ms, tổng cộng 350-550 ms. Trên GPU, phát hiện là 20-40 ms và quá trình xử lý trước + phân loại bắt đầu quan trọng hơn về mặt tương đối.

## Sử dụng nó

Các mẫu sản xuất hội tụ với cùng cấu trúc, cộng với:

- **Model versioning** luôn ghi tên mô hình và trọng lượng hash trong phản ứng.
- **Per-request trace IDs** ghi lại thời gian từng giai đoạn cho mỗi yêu cầu để bạn có thể liên quan các phản ứng chậm với các giai đoạn.
- **Fallback path** nếu phân loại viên không hoạt động, trả lại các phát hiện mà không có phân loại thay vì thất bại trong toàn bộ yêu cầu.
- **Safety filters** Các bộ lọc NSFW / PII chạy sau khi phân loại, trước khi phản ứng rời khỏi dịch vụ.
- **Batch endpoint** a `/detect_batch`chấp nhận một danh sách các URL hình ảnh để xử lý hàng loạt.

Đối với sản xuất phục vụ, `torchserve`- `Triton Inference Server`, và`BentoML`xử lý việc phân phối, phiên bản, số liệu và kiểm tra sức khỏe.`FastAPI`trực tiếp là tốt cho các nguyên mẫu và các sản phẩm quy mô nhỏ.

## Chuyển nó

Bài học này mang lại:

- `outputs/prompt-vision-service-shape-reviewer.md` một lời nhắc xem lại mã của dịch vụ thị giác cho vi phạm hình thức hợp đồng / phản ứng và đặt tên lỗi phá vỡ đầu tiên.
- `outputs/skill-pipeline-budget-planner.md` một kỹ năng, với tính đến độ trễ và thông qua mục tiêu, chỉ định ngân sách thời gian cho mỗi giai đoạn đường ống và đánh dấu giai đoạn nào sẽ bỏ lỡ ngân sách đầu tiên.

## Các bài tập

1. **(Easy)**Tiêu chuẩn đường ống trên 10 hình ảnh từ bất kỳ bộ dữ liệu mở nào.
2. **(Medium)**Thêm một trường đầu ra mặt nạ vào `Detection`và mã hóa nó như là RLE. Kiểm tra JSON ở dưới 1MB ngay cả cho một hình ảnh 10 đối tượng.
3. **(Hard)**Thêm một bộ vi xử lý trước bộ phân loại: thu thập các loại cây trồng trong thời gian lên đến 10 ms, phân loại tất cả chúng trong một cuộc gọi GPU, trả lại kết quả mỗi yêu cầu. đo tăng thông qua ở 5 yêu cầu đồng thời mỗi giây và độ trễ được thêm vào.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Pipeline | "The system" | An ordered chain of preprocessing, inference, and postprocessing steps with a typed interface between each pair |
| Data contract | "The schema" | Pydantic / dataclass definitions that every stage input and output conforms to; catches integration bugs at the boundary |
| Preprocessing | "Before the model" | Decoding, colour conversion, resizing, normalising; usually the biggest CPU time sink |
| Postprocessing | "After the model" | NMS, mask resize, threshold, RLE encode; cheap on GPU, expensive on CPU |
| Microbatcher | "Collect then forward" | Aggregator that waits a fixed window for multiple requests, runs a single batched forward pass |
| Trace ID | "Request id" | Per-request identifier logged at every stage so slow requests can be traced end-to-end |
| Failure code | "Named error" | Specific error code per failure class instead of generic 500; enables client retry logic |
| Health check | "Readiness probe" | Cheap endpoint that reports whether the service can answer; loadbalancers rely on this |

## Đọc thêm

- [Full Stack Deep Learning — Deploying Models](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/) tổng quan kinh điển về việc triển khai ML sản xuất
- [BentoML docs](https://docs.bentoml.com) phục vụ khung với batching, phiên bản và métrics
- [torchserve docs](https://pytorch.org/serve/) Thư viện phục vụ chính thức của PyTorch
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) phục vụ hiệu suất cao với hỗ trợ hàng và nhiều mô hình
