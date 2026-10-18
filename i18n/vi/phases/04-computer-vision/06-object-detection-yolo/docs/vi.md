# Khám phá đối tượng  YOLO từ đầu

> Khám phá là phân loại cộng với hồi quy, chạy ở mọi vị trí trong bản đồ tính năng, sau đó được dọn dẹp bằng cách đàn áp không tối đa.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification), Phase 4 Lesson 05 (Transfer Learning)
**Time:** ~75 minutes

## Mục tiêu học tập

- Giải thích thiết kế lưới và neo mà biến phát hiện thành một vấn đề dự đoán dày đặc và nói ra mỗi số trong tensor đầu ra có nghĩa là gì
- Lượng giao thông giữa các hộp và thực hiện việc xóa không tối đa từ đầu
- Xây dựng một cái đầu kiểu YOLO tối thiểu trên đỉnh xương sống đã được huấn luyện trước, bao gồm các loại, tính đối tượng và mất mát thấu trường
- Đọc một hàng số phát hiện (precision@0.5, recall, mAP@0.5, mAP@0.5:0.95) và chọn nút nào để xoay tiếp theo

## Vấn đề

Việc phân loại nói "hình ảnh này là một con chó". Việc phát hiện nói "có một con chó ở các pixel (112, 40, 280, 210), có một con mèo ở (400, 180, 560, 310), và không có gì khác trong khung. " Một thay đổi cấu trúc đó  dự đoán một số lượng thay đổi của hộp có nhãn thay vì một nhãn mỗi hình ảnh  là những gì mà mọi hệ thống tự trị, mọi sản phẩm giám sát, mọi trình phân tích bố bố trình, và mọi đường tầm nhìn nhà máy phụ thuộc vào.

Khám phá cũng là nơi mà mọi sự đổi mới kỹ thuật trong tầm nhìn xuất hiện cùng một lúc. Bạn muốn các hộp chính xác (đầu quay trở), bạn muốn lớp đúng cho mỗi hộp (đầu phân loại), bạn muốn mô hình biết khi không có gì để phát hiện (điểm đối tượng), và bạn muốn chính xác một dự đoán cho mỗi đối tượng thực (không áp lực tối đa). Trượt bất kỳ một trong những điều này và đường ống hoặc bỏ lỡ các đối tượng, báo cáo các hộp ảo giác, hoặc dự đoán cùng một đối tượng mười lăm lần trong các vị trí hơi khác nhau.

YOLO (You Only Look Once, Redmon et al. 2016) là thiết kế đã thực hiện tất cả các hoạt động này trong thời gian thực bằng cách thực hiện nó bằng cách vượt qua một con đường con đường, và các quyết định cấu trúc tương tự vẫn là xương sống của các máy dò hiện đại (YOLOv8, YOLOv9, YOLO-NAS, RT-DETR).

## Khái niệm

### Khám phá như dự đoán mật độ

Một bộ phân loại sẽ đưa ra các số C cho mỗi hình ảnh.`(S x S x (5 + C))`số cho mỗi hình ảnh, nơi S là kích thước lưới không gian.

```mermaid
flowchart LR
    IMG["Input 416x416 RGB"] --> BB["Backbone<br/>(ResNet, DarkNet, ...)"]
    BB --> FM["Feature map<br/>(C_feat, 13, 13)"]
    FM --> HEAD["Detection head<br/>(1x1 convs)"]
    HEAD --> OUT["Output tensor<br/>(13, 13, B * (5 + C))"]
    OUT --> DEC["Decode<br/>(grid + sigmoid + exp)"]
    DEC --> NMS["Non-max suppression"]
    NMS --> RESULT["Final boxes"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a
```

Mỗi một trong số đó`S * S`các tế bào lưới dự đoán `B`Các hộp.

- 4 số mô tả hình học: `tx, ty, tw, th`- Tôi không biết.
- Số 1 là điểm đối tượng: "có một đối tượng tập trung trong tế bào này không?"
- Số C là xác suất lớp.

Tổng số mỗi tế bào: `B * (5 + C)`. Đối với VOC với `S=13, B=2, C=20`, đó là 50 con số cho mỗi tế bào.

### Tại sao lưới và neo

Sự lùi lại đơn giản sẽ dự đoán `(x, y, w, h)`cho mỗi đối tượng như một phối hợp tuyệt đối. Điều đó khó khăn cho một mạng conv vì dịch hình ảnh không nên dịch tất cả các dự đoán bằng cùng một số lượng  mỗi đối tượng được neo không gian. lưới trả lời điều này bằng cách gán mỗi hộp thực tại căn bản cho tế bào lưới trung tâm của nó rơi vào; chỉ có tế bào đó chịu trách nhiệm cho đối tượng đó.

Các neo giải quyết một vấn đề thứ hai. một con 3x3 không thể dễ dàng quay lại một hộp rộng 500 pixel ra khỏi một tế bào tính năng trường 16 pixel. thay vào đó, chúng tôi xác định trước `B`hình dạng hộp trước (những neo) cho mỗi tế bào và dự đoán các đợt nhỏ từ mỗi neo. mô hình học cách chọn neo đúng và đẩy nó thay vì lùi lại từ không có gì.

```
Anchor box priors (example for 416x416 input):

  small:   (30,  60)
  medium:  (75,  170)
  large:   (200, 380)

At each grid cell, every anchor emits (tx, ty, tw, th, obj, c_1, ..., c_C).
```

Các máy dò hiện đại thường sử dụng FPN với các bộ neo khác nhau cho mỗi độ phân giải  neo nhỏ trên bản đồ độ phân giải cao nông, neo lớn trên bản đồ độ phân giải thấp sâu.

### Dự đoán giải mã

Các nguyên liệu`tx, ty, tw, th`không phải là các định vị hộp; chúng là các mục tiêu hồi quy phải được chuyển đổi trước khi vẽ:

```math
\begin{aligned}
\text{centre } x &= (\sigma(t_x) + \text{cell}_x) \cdot \text{stride} \\
\text{centre } y &= (\sigma(t_y) + \text{cell}_y) \cdot \text{stride} \\
\text{width} &= \text{anchor}_w \cdot \exp(t_w) \\
\text{height} &= \text{anchor}_h \cdot \exp(t_h)
\end{aligned}
```

`sigmoid`giữ trung tâm của sự thay đổi bên trong tế bào. `exp`cho phép chiều rộng mở ra khỏi neo mà không cần một dấu hiệu đảo. `stride`Dành cách giải mã này là giống nhau trong mọi phiên bản YOLO kể từ v2.

### Tỷ lệ

Métric tương đồng phổ biến của phát hiện giữa hai hộp:

```math
\mathrm{IoU}(A, B) = \frac{\mathop{\mathrm{area}}(A \cap B)}{\mathop{\mathrm{area}}(A \cup B)}
```

$\mathrm{IoU} = 1$nghĩa là giống nhau; $\mathrm{IoU} = 0$IoU giữa dự đoán và hộp thực tại cơ bản là những gì quyết định liệu một dự đoán có được tính là một tích cực thực (thường là$\mathrm{IoU} \ge 0.5$IoU giữa hai dự đoán là những gì NMS sử dụng để deduplicate.

### Phong trào không tối đa

Một mạng conv được đào tạo trên các neo lân cận thường dự đoán các hộp chồng chéo cho cùng một đối tượng. NMS giữ được dự đoán độ tin cậy cao nhất và xóa bất kỳ dự đoán nào khác với IoU trên ngưỡng.

```
NMS(boxes, scores, iou_threshold):
    sort boxes by score descending
    keep = []
    while boxes not empty:
        pick the top-scoring box, add to keep
        remove every box with IoU > iou_threshold to the picked box
    return keep
```

Đường ngưỡng điển hình: 0,45 cho phát hiện đối tượng.`soft-NMS`- `DIoU-NMS`, hoặc học cách đàn áp trực tiếp (RT-DETR) nhưng mục đích cấu trúc là giống nhau.

### Sự mất mát

Lối mất YOLO là ba lỗ cộng với trọng lượng:

```math
\begin{aligned}
L = {} & \lambda_{\text{coord}} \cdot L_{\text{box}}(\text{pred}, \text{target}, \text{where obj}=1) \\
+ {} & \lambda_{\text{obj}} \cdot L_{\text{obj}}(\text{pred}, 1, \text{where obj}=1) \\
+ {} & \lambda_{\text{noobj}} \cdot L_{\text{obj}}(\text{pred}, 0, \text{where obj}=0) \\
+ {} & \lambda_{\text{cls}} \cdot L_{\text{cls}}(\text{pred}, \text{target}, \text{where obj}=1)
\end{aligned}
```

Chỉ có các tế bào chứa một đối tượng góp phần vào sự mất mát thu hồi và phân loại hộp. Các tế bào không có đối tượng chỉ góp phần vào sự mất mát đối tượng (đọc cho mô hình giữ im lặng). `lambda_noobj`thường nhỏ (~ 0,5) vì phần lớn các tế bào trống rỗng và nếu không sẽ thống trị tổng tổn thất.

Các biến thể hiện đại thay đổi mất hộp MSE cho CIoU / DIoU (được tối ưu hóa IoU trực tiếp), sử dụng mất tập trung cho sự mất cân bằng lớp học, và cân bằng đối tượng với mất tập trung chất lượng.

### Các số liệu phát hiện

Độ chính xác không chuyển sang phát hiện.

- **Precision@IoU=0.5** trong số các dự đoán được tính là tích cực, bao nhiêu là thực sự đúng.
- **Recall@IoU=0.5** của các vật thể thực, chúng tôi tìm thấy bao nhiêu.
- **AP@0.5** diện tích đường cong thu hồi chính xác ở ngưỡng IoU 0,5; một số cho mỗi lớp.
- **mAP@0.5:0.95** trung bình AP trên ngưỡng IoU 0,5, 0,55, ..., 0,95.

Báo cáo tất cả bốn. Một máy dò có độ mạnh trên mAP@0.5 nhưng yếu trên mAP@0.5:0.95 đang định vị khá nhưng không chặt chẽ; sửa chữa với sự mất mát giảm thấu trường tốt hơn. Một máy dò có độ chính xác cao và hồi tưởng thấp quá bảo thủ; giảm ngưỡng độ tin cậy hoặc tăng trọng lượng đối tượng.

```figure
object-detection-nms
```

## Hãy xây dựng nó

### Bước 1:

Nó là con ngựa của toàn bộ bài học.`(x1, y1, x2, y2)`định dạng.

```python
import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

Trả lại một `(N_a, N_b)`Matrix của các IoU theo đôi. Sử dụng nó chống lại một hộp thực tại mặt đất duy nhất bằng cách làm cho một trong các mảng hình dạng `(1, 4)`- Tôi không biết.

### Bước 2: Thiết bị không tối đa

```python
def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
```

Định nghĩa,`O(N log N)`từ loại, và phù hợp với hành vi của `torchvision.ops.nms`trên các đầu vào giống nhau.

### Bước 3: Mã hóa và giải mã hộp

Chuyển đổi giữa các tọa độ pixel và `(tx, ty, tw, th)`các mục tiêu mà mạng thực sự lùi lại.

```python
def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

Thử nghiệm: mã hóa một hộp sau đó mã hóa  bạn nên lấy lại một cái gì đó rất gần với nguyên bản (trong khi ngược sigmoid không hoàn toàn đảo ngược khi `tx`không nằm trong phạm vi sau sigmoid).

### Bước 4: Một đầu YOLO tối thiểu

Một 1x1 conv trên một bản đồ tính năng, định dạng lại để `(B, S, S, num_anchors, 5 + C)`- Tôi không biết.

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y
```

Hình dạng đầu ra: `(N, H, W, num_anchors, 5 + C)`- Mức độ cuối cùng vẫn còn.`[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`- Tôi không biết.

### Bước 5: Giới thiệu sự thật

Đối với mỗi hộp chân lý, hãy quyết định cái nào.`(cell, anchor)`là người chịu trách nhiệm.

```python
def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj
```

Sự lựa chọn neo là "IoU hình dạng tốt nhất với sự thật mặt đất"  một proxy rẻ tiền phù hợp với nhiệm vụ YOLOv2/v3. v5 và sau sử dụng các chiến lược tinh vi hơn (sự phù hợp với nhiệm vụ, động lực k) để tinh chỉnh cùng một ý tưởng.

### Bước 6: Ba lỗ

```python
def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}
```

Năm siêu tham số mà mỗi hướng dẫn YOLO mã hóa hoặc xóa.`lambda_coord=5, lambda_noobj=0.5`Nhìn lại giấy YOLOv1 gốc và vẫn hoạt động như một mặc định hợp lý.

### Bước 7: Đường ống dẫn dẫn

Giải mã đầu đầu ra nguyên chất, áp dụng sigmoid/exp, ngưỡng đối tượng và NMS.

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

Đó là con đường eval hoàn chỉnh: đầu -> giải mã -> ngưỡng -> NMS.

## Sử dụng nó

`torchvision.models.detection`Các máy dò sản xuất có cấu trúc khái niệm tương tự.

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")
```

Đối với đường ống dẫn suy luận thời gian thực,`ultralytics`(YOLOv8/v9) là tiêu chuẩn: `from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`. Mô hình xử lý mã hóa và NMS nội bộ và trả lại cùng một `boxes / scores / labels`3 lần mà anh xây dựng ở trên.

## Chuyển nó

Bài học này mang lại:

- `outputs/prompt-detection-metric-reader.md` một lời nhắc nhở biến một `precision, recall, AP, mAP@0.5:0.95`xếp vào một chẩn đoán một dòng và thử nghiệm tiếp theo hữu ích nhất.
- `outputs/skill-anchor-designer.md` một kỹ năng mà, với một tập hợp dữ liệu của các hộp thực tại cơ bản, chạy k- trung bình trên `(w, h)`và trả lại các bộ neo theo cấp FPN cộng với số liệu thống kê bảo hiểm bạn cần để chọn số lượng neo đúng.

## Các bài tập

1. **(Easy)**Thực hiện`box_iou`và chạy nó chống lại `torchvision.ops.box_iou`trên 1.000 cặp hộp ngẫu nhiên.`1e-6`- Tôi không biết.
2. **(Medium)**Cảng`yolo_loss`cho một phiên bản sử dụng `CIoU`Box loss thay vì MSE. Cho thấy trên một tập dữ liệu tổng hợp 100 hình ảnh rằng CIoU hội tụ với một mAP tốt hơn cuối cùng @ 0.5: 0.95 so với MSE trong cùng một số thời đại.
3. **(Hard)**Thực hiện suy luận đa quy mô: cung cấp cùng một hình ảnh ở ba độ phân giải thông qua mô hình, hợp nhất các dự đoán hộp, và chạy một NMS duy nhất ở cuối. đo mAP nâng so với suy luận quy mô duy nhất trên một tập hợp được giữ.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Anchor | "Box prior" | A pre-defined box shape at each grid cell from which the network predicts deltas instead of absolute coordinates |
| IoU | "Overlap" | Intersection-over-union of two boxes; the universal similarity measure in detection |
| NMS | "Deduplicate" | Greedy algorithm that keeps highest-score predictions and removes overlapping ones above a threshold |
| Objectness | "Is there something here" | Per-anchor, per-cell scalar predicting whether an object is centred in that cell |
| Grid stride | "Downsample factor" | Pixels per grid cell; a 416-px input with a 13-grid head has stride 32 |
| mAP | "Mean average precision" | Average of the area under the precision-recall curve, averaged over classes and (for COCO) IoU thresholds |
| AP@0.5 | "PASCAL VOC AP" | Average precision with IoU threshold 0.5; the lenient version of the metric |
| mAP@0.5:0.95 | "COCO AP" | Average over IoU thresholds 0.5..0.95 step 0.05; the strict version and current community standard |

## Đọc thêm

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) giấy thành lập; mỗi YOLO kể từ đó là một sự tinh chỉnh của cấu trúc này
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) giấy giới thiệu các đầu kiểu FPN đa quy mô; vẫn là sơ đồ rõ ràng nhất
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com) tham chiếu sản xuất hiện tại; bao gồm các định dạng tập dữ liệu, bổ sung, công thức đào tạo
- [The Illustrated Guide to Object Detection (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904) Tour tiếng Anh đơn giản tốt nhất của vườn thú toàn bộ máy dò; vô giá để hiểu cách DETR, RetinaNet, FCOS và YOLO liên quan
