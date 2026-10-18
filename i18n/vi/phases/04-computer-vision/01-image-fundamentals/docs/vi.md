# Chữ hình ảnh cơ bản  Pixel, kênh, không gian màu

> Một hình ảnh là một tensor của các mẫu ánh sáng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 Lesson 12 (Tensor Operations), Phase 3 Lesson 11 (Intro to PyTorch)
**Time:** ~45 minutes

## Mục tiêu học tập

- Giải thích làm thế nào một cảnh liên tục được phân loại thành pixel và tại sao các quyết định lấy mẫu/quantization đặt trần trên mỗi mô hình dòng chảy xuống
- Đọc, cắt và kiểm tra hình ảnh như các mảng NumPy và chuyển đổi một cách ruột giữa các bố cục HWC và CHW
- Chuyển đổi giữa RGB, thang xám, HSV và YCbCr và biện minh tại sao mỗi không gian màu tồn tại
- Sử dụng xử lý trước cấp độ pixel (tự chuẩn hóa, chuẩn hóa, thay đổi kích thước, kênh đầu tiên) chính xác như các mô hình thị giác PyTorch được đào tạo trước mong đợi

## Vấn đề

Mỗi bài báo bạn sẽ đọc, mỗi cân nặng trước khi được đào tạo bạn sẽ tải xuống, mỗi API tầm nhìn bạn sẽ gọi giả định một mã hóa cụ thể của đầu vào.`uint8`hình ảnh nơi mô hình muốn `float32`và nó vẫn sẽ chạy  và lặng lẽ sản xuất rác. Đưa BGR vào một mạng được đào tạo trên RGB và độ chính xác sụp đổ mười điểm. Đưa một mô hình kênh - đầu vào cuối khi nó mong đợi kênh - đầu tiên và lớp conv đầu tiên coi chiều cao như một kênh tính năng. Không có điều này ném một lỗi. Nó chỉ làm hỏng métrics của bạn và bạn dành một tuần săn tìm một lỗi sống trong cách bạn tải tập tin.

Một cú xoay không phức tạp khi bạn biết nó đang trượt qua. Phần khó khăn là "một hình ảnh" có nghĩa là những điều khác nhau với một máy ảnh, một máy giải mã JPEG, PIL, OpenCV, torchvision và một hạt nhân CUDA. Mỗi ngăn xếp có trục tựa riêng của nó, phạm vi byte và quy tắc kênh. Một kỹ sư tầm nhìn không thể giữ cho các tàu thẳng này bị vỡ đường ống dẫn.

Bài học này cố định nền tảng để phần còn lại của giai đoạn có thể xây dựng trên nó. Đến cuối bạn sẽ biết một pixel là gì, tại sao có ba số cho mỗi pixel thay vì một, "tự chuẩn hóa với thống kê ImageNet" thực sự làm gì, và làm thế nào để di chuyển giữa hai hoặc ba bố cục mà mọi bài học khác trong giai đoạn này sẽ giả định.

## Khái niệm

### Các đường ống quá trình xử lý trước hoàn toàn trong một cái nhìn

Mỗi hệ thống thị giác sản xuất đều là một chuỗi biến đổi đảo ngược. Một bước sai và mô hình sẽ thấy một đầu vào khác so với nó được đào tạo.

```mermaid
flowchart LR
    A["Image file<br/>(JPEG/PNG)"] --> B["Decode<br/>uint8 HWC"]
    B --> C["Convert<br/>colorspace<br/>(RGB/BGR/YCbCr)"]
    C --> D["Resize<br/>shorter side"]
    D --> E["Center crop<br/>model size"]
    E --> F["Divide by 255<br/>float32 [0,1]"]
    F --> G["Subtract mean<br/>Divide by std"]
    G --> H["Transpose<br/>HWC → CHW"]
    H --> I["Batch<br/>CHW → NCHW"]
    I --> J["Model"]

    style A fill:#fef3c7,stroke:#d97706
    style J fill:#ddd6fe,stroke:#7c3aed
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#bfdbfe,stroke:#2563eb
```

Hai hộp màu đỏ và xanh là nơi 80% các lỗi âm thanh sống: thiếu tiêu chuẩn hóa và bố cục sai.

### Một pixel là một mẫu, không phải là một hình vuông

Một cảm biến máy ảnh đếm photon rơi vào một lưới các máy dò nhỏ. Mỗi máy dò tích hợp ánh sáng trong một phần nhỏ của giây và phát ra điện áp tương xứng với số lượng photon chạm vào nó.

```
Continuous scene                 Sensor grid                     Digital image
(infinite detail)                (H x W detectors)               (H x W integers)

    ~~~~~                        +--+--+--+--+--+                 210 198 180 155 120
   ♪ ♪ Tôi đã làm gì ♪ ♪ ♪ Tôi đã làm gì ♪
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   ~~~~~                         |  |  |  |  |  |                 195 185 170 148 112
                                 +--+--+--+--+--+                 188 180 165 145 108
```

Có hai lựa chọn xảy ra ở bước này và họ cố định trần nhà trên mọi thứ dòng chảy xuống:

- **Spatial sampling**cho biết có bao nhiêu máy dò cho mỗi độ của cảnh. quá ít, và các cạnh trở nên đốm (aliasing). quá nhiều, và lưu trữ và tính toán nổ.
- **Intensity quantization**8 bit cho 256 mức và là tiêu chuẩn cho hiển thị. 10, 12, 16 bit cho gradient mịn hơn và vật chất cho hình ảnh y tế, HDR, và ống dẫn cảm biến nguyên liệu.

Một pixel không phải là một hình vuông màu với diện tích. Nó là một phép đo duy nhất. Khi bạn thay đổi kích thước hoặc xoay, bạn đang lấy lại mẫu lưới đo đó.

### Tại sao ba kênh

Một máy dò đếm photon trên toàn bộ quang phổ có thể nhìn thấy  đó là thang xám. Để có được màu sắc, cảm biến bao phủ lưới với một hình ảnh của các bộ lọc đỏ, xanh lá cây và xanh lá cây. Sau khi làm phân tích, mỗi vị trí không gian có ba số nguyên: phản ứng của máy dò lọc màu đỏ, lọc màu xanh lá cây và lọc màu xanh lá cây gần đó. Ba số nguyên là bộ ba RGB của một pixel.

```
One pixel in memory:

    (R, G, B) = (210, 140, 30)   <- reddish-orange

An H x W RGB image:

    shape (H, W, 3)     stored as   H rows of W pixels of 3 values
                                    each in [0, 255] for uint8
```

Ba không phải là phép thuật. Các máy ảnh độ sâu thêm một kênh Z. Các vệ tinh thêm băng hồng ngoại và cực tím. Quét nghiệm y tế thường có một kênh (X-quang, CT) hoặc nhiều kênh (lớn quang phổ). Số lượng kênh là trục cuối cùng; các lớp conv học để pha trộn qua nó.

### Hai công ước bố trí: HWC và CHW

Tăng-tơ giống nhau, hai thứ tự.

```
HWC (height, width, channels)           CHW (channels, height, width)

   W ->                                    H ->
  +-----+-----+-----+                     +-----+-----+
H |R G B|R G B|R G B|                   C |R R R R R R|
| +-----+-----+-----+                   | +-----+-----+
v |R G B|R G B|R G B|                   v |G G G G G G|
  +-----+-----+-----+                     +-----+-----+
                                          |B B B B B B|
                                          +-----+-----+

   PIL, OpenCV, matplotlib,              PyTorch, most deep learning
   almost every image file on disk       frameworks, cuDNN kernels
```

CHW tồn tại vì các hạt nhân convolution trượt qua H và W. Giữ trục kênh trước tiên có nghĩa là mỗi hạt nhân thấy một trình độ 2D liền kề mỗi kênh, mà vectorizes sạch sẽ.

Việc chuyển đổi một dòng bạn sẽ gõ một ngàn lần:

```
img_chw = img_hwc.transpose(2, 0, 1)      # NumPy
img_chw = img_hwc.permute(2, 0, 1)        # PyTorch tensor
```

Layout bộ nhớ, hình ảnh hóa:

```mermaid
flowchart TB
    subgraph HWC["HWC — pixels stored interleaved (PIL, OpenCV, JPEG)"]
        H1["row 0: R G B | R G B | R G B ..."]
        H2["row 1: R G B | R G B | R G B ..."]
        H3["row 2: R G B | R G B | R G B ..."]
    end
    subgraph CHW["CHW — channels stored as stacked planes (PyTorch, cuDNN)"]
        C1["plane R: entire H x W of red values"]
        C2["plane G: entire H x W of green values"]
        C3["plane B: entire H x W of blue values"]
    end
    HWC -->|"transpose(2, 0, 1)"| CHW
    CHW -->|"transpose(1, 2, 0)"| HWC
```

### Phạm vi byte và dtype

Ba hội nghị chủ yếu:

| Convention | dtype | Range | Where you see it |
|------------|-------|-------|------------------|
| Raw | `uint8` | [0, 255] | Files on disk, PIL, OpenCV output |
| Normalized | `float32` | [0.0, 1.0] | After `img.astype('float32') / 255` |
| Standardized | `float32` | roughly [-2, +2] | After subtracting mean and dividing by std |

Các mạng convolutional được đào tạo trên các đầu vào tiêu chuẩn hóa.`mean=[0.485, 0.456, 0.406]`- `std=[0.229, 0.224, 0.225]`là trung bình toán học và lệch tiêu chuẩn của ba kênh trên bộ đào tạo ImageNet đầy đủ, được tính trên [0, 1] pixel bình thường.`uint8`trong một mô hình dự kiến lơ lửng tiêu chuẩn là sự thất bại im lặng phổ biến nhất trong tầm nhìn ứng dụng.

### Không gian màu sắc và lý do tại sao chúng tồn tại

RGB là định dạng chụp nhưng nó không phải lúc nào cũng là đại diện hữu ích nhất cho một mô hình.

```
 RGB               HSV                       YCbCr / YUV

 R red             H hue (angle 0-360)       Y luminance (brightness)
 G green           S saturation (0-1)        Cb chroma blue-yellow
 B blue            V value/brightness (0-1)  Cr chroma red-green

 Linear to         Separates color from      Separates brightness from
 sensor output     brightness. Useful for    color. JPEG and most video
                   color thresholding, UI    codecs compress the chroma
                   sliders, simple filters   channels harder because the
                                             human eye is less sensitive
                                             to chroma detail than to Y.
```

Đối với hầu hết các đài CNN hiện đại, bạn cung cấp RGB. Bạn gặp các không gian khác khi:

- **HSV** mã CV cổ điển, phân đoạn dựa trên màu sắc, cân bằng trắng.
- **YCbCr** đọc nội dung JPEG, đường ống video, mô hình siêu độ phân giải chỉ hoạt động trên Y.
- **Grayscale** OCR, mô hình tài liệu, bất kỳ trường hợp nào màu sắc là biến phiền nhiễu thay vì tín hiệu.

Tỷ lệ xám từ RGB là một tổng số cân nặng, không phải là trung bình, bởi vì mắt con người nhạy cảm hơn với màu xanh lá cây hơn là màu đỏ hoặc xanh dương:

```math
Y = 0.299 R + 0.587 G + 0.114 B \qquad \text{(ITU-R BT.601, the classic weights)}
```

### Tỷ lệ khía cạnh, quy mô lại và phân cực

Mỗi mô hình đều có kích thước đầu vào cố định (224x224 cho hầu hết các phân loại ImageNet, 384x384 hoặc 512x512 cho các máy dò hiện đại).

- **Resize shorter side, then center crop** công thức tiêu chuẩn của ImageNet. Giữ tỷ lệ hình ảnh, ném đi một dải của các pixel cạnh.
- **Resize and pad** bảo tồn tỷ lệ hình ảnh và mỗi pixel, thêm thanh đen.
- **Resize directly to target** kéo dài hình ảnh. rẻ, làm biến dạng hình học, tốt cho nhiều nhiệm vụ phân loại.

Phương pháp phân cực quyết định cách tính các pixel trung gian khi lưới mới không phù hợp với lưới cũ:

```
Nearest neighbour     fastest, blocky, only choice for masks/labels
Bilinear              fast, smooth, default for most image resizing
Bicubic               slower, sharper on upscaling
Lanczos               slowest, best quality, used for final display
```

Quy tắc ngón tay: hàng tỷ đường cho đào tạo, hai khối hoặc lanczos cho tài sản bạn sẽ xem, gần nhất cho bất cứ thứ gì có chứa các thẻ nhận dạng lớp nguyên số.

```figure
conv-output-size
```

## Hãy xây dựng nó

### Bước 1: Xây dựng một tensor hình ảnh và kiểm tra hình dạng của nó

Bắt đầu với một hình ảnh tổng hợp xác định để phòng thí nghiệm đầu tiên chạy offline chỉ với NumPy. Việc giải mã tập tin là một ranh giới riêng biệt: một khi một máy giải mã JPEG hoặc PNG trả về các byte RGB, mỗi hoạt động tensor dưới đây đều giống nhau.

```python
import numpy as np

def synthetic_rgb(h=128, w=192, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    r = (np.sin(xx * 6) * 0.5 + 0.5) * 255
    g = yy * 255
    b = (1 - yy) * xx * 255
    rgb = np.stack([r, g, b], axis=-1) + rng.normal(0, 6, (h, w, 3))
    return np.clip(rgb, 0, 255).astype(np.uint8)

arr = synthetic_rgb()

print(f"type:   {type(arr).__name__}")
print(f"dtype:  {arr.dtype}")
print(f"shape:  {arr.shape}     # (H, W, C)")
print(f"min:    {arr.min()}")
print(f"max:    {arr.max()}")
print(f"pixel at (0, 0): {arr[0, 0]}")
```

Tạo sản lượng dự kiến: `shape: (H, W, 3)`- `dtype: uint8`, phạm vi`[0, 255]`Đó là biểu diễn được giải mã theo quy luật cho dù các byte đến từ một máy ảnh, một máy giải mã hình ảnh, hoặc một máy phát điện tổng hợp này.

### Bước 2: Chia kênh và sắp xếp lại

Lấy R, G, B riêng biệt, sau đó chuyển đổi từ HWC sang CHW cho PyTorch.

```python
R = arr[:, :, 0]
G = arr[:, :, 1]
B = arr[:, :, 2]
print(f"R shape: {R.shape}, mean: {R.mean():.1f}")
print(f"G shape: {G.shape}, mean: {G.mean():.1f}")
print(f"B shape: {B.shape}, mean: {B.mean():.1f}")

arr_chw = arr.transpose(2, 0, 1)
print(f"\nHWC shape: {arr.shape}")
print(f"CHW shape: {arr_chw.shape}")
```

Ba phẳng thang xám, một trên mỗi kênh. CHW chỉ sắp xếp lại các trục; không cần có bản sao dữ liệu nghiêm ngặt khi bố cục bộ nhớ cho phép.

### Bước 3: Chuyển đổi thang xám và HSV

Đánh giá số lượng xám, sau đó là một bản RGB đến HSV.

```python
def rgb_to_grayscale(rgb):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb.astype(np.float32) @ weights).astype(np.uint8)

def rgb_to_hsv(rgb):
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.max(rgb_f, axis=-1)
    cmin = np.min(rgb_f, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 0
    argmax = np.argmax(rgb_f, axis=-1)
    rmax = mask & (argmax == 0)
    gmax = mask & (argmax == 1)
    bmax = mask & (argmax == 2)
    h[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6
    h[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2
    h[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4
    h = h * 60.0

    s = np.divide(delta, cmax, out=np.zeros_like(delta), where=cmax > 0)
    v = cmax
    return np.stack([h, s, v], axis=-1)

gray = rgb_to_grayscale(arr)
hsv = rgb_to_hsv(arr)
print(f"gray shape: {gray.shape}, range: [{gray.min()}, {gray.max()}]")
print(f"hsv   shape: {hsv.shape}")
print(f"hue range: [{hsv[..., 0].min():.1f}, {hsv[..., 0].max():.1f}] degrees")
print(f"sat range: [{hsv[..., 1].min():.2f}, {hsv[..., 1].max():.2f}]")
print(f"val range: [{hsv[..., 2].min():.2f}, {hsv[..., 2].max():.2f}]")
```

Hue xuất hiện trong độ, bão hòa và giá trị trong [0, 1].`hsv_full`Hội nghị.

### Bước 4: Tự chuẩn hóa, chuẩn hóa và đảo ngược nó

Đi từ các byte nguyên liệu đến các tensor chính xác một mô hình ImageNet được đào tạo trước mong đợi, sau đó quay lại.

```python
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_imagenet(rgb_uint8):
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)
    return x

def deprocess_imagenet(chw_float32):
    x = chw_float32.transpose(1, 2, 0)
    x = x * std + mean
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x

x = preprocess_imagenet(arr)
print(f"preprocessed shape: {x.shape}     # (C, H, W)")
print(f"preprocessed dtype: {x.dtype}")
print(f"preprocessed mean per channel:  {x.mean(axis=(1, 2)).round(3)}")
print(f"preprocessed std  per channel:  {x.std(axis=(1, 2)).round(3)}")

roundtrip = deprocess_imagenet(x)
max_diff = np.abs(roundtrip.astype(int) - arr.astype(int)).max()
print(f"roundtrip max pixel diff: {max_diff}    # should be 0 or 1")
```

Tỷ lệ trung bình mỗi kênh nên gần bằng không, std gần một.`transforms.Normalize`gọi là làm dưới nắp.

### Bước 5: Tái kích thước từ đầu

Các vòng hàng xóm gần nhất mỗi liên kết đầu ra đến một pixel nguồn. Sự phân cực hai tuyến tìm thấy bốn pixel xung quanh và pha trộn chúng theo khoảng cách. Cả hai thực hiện dưới đây sử dụng các liên kết liên kết với điểm cuối để pixel nguồn đầu tiên và cuối cùng vẫn được cố định.

```python
def resize_coordinates(source_length, target_length):
    if target_length == 1:
        return np.zeros(1, dtype=np.float32)
    return np.linspace(0, source_length - 1, target_length, dtype=np.float32)

def nearest_resize(image, target_height, target_width):
    y = np.rint(resize_coordinates(image.shape[0], target_height)).astype(int)
    x = np.rint(resize_coordinates(image.shape[1], target_width)).astype(int)
    return image[y[:, None], x[None, :]]

def bilinear_resize(image, target_height, target_width):
    y = resize_coordinates(image.shape[0], target_height)
    x = resize_coordinates(image.shape[1], target_width)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    wy = (y - y0)[:, None, None]
    wx = (x - x0)[None, :, None]

    source = image.astype(np.float32)
    top = source[y0[:, None], x0[None, :]] * (1 - wx)
    top += source[y0[:, None], x1[None, :]] * wx
    bottom = source[y1[:, None], x0[None, :]] * (1 - wx)
    bottom += source[y1[:, None], x1[None, :]] * wx
    result = top * (1 - wy) + bottom * wy
    return np.clip(np.rint(result), 0, 255).astype(image.dtype)

target_height = arr.shape[0] * 3
target_width = arr.shape[1] * 3
nearest = nearest_resize(arr, target_height, target_width)
bilinear = bilinear_resize(arr, target_height, target_width)

def local_roughness(x):
    gy = np.diff(x.astype(float), axis=0)
    gx = np.diff(x.astype(float), axis=1)
    return float(np.abs(gy).mean() + np.abs(gx).mean())

for name, out in [("nearest", nearest), ("bilinear", bilinear)]:
    print(f"{name:>8}  shape={out.shape}  roughness={local_roughness(out):6.2f}")
```

Điểm gần nhất cao nhất trên độ thô lỗ vì nó giữ cạnh cứng. Bilinear mịn hơn vì mỗi pixel mới kết hợp hai vị trí trên mỗi trục. Người đồng hành chạy mở rộng ý tưởng phân tách tương tự đến bốn hàng xóm trên mỗi trục với hạt nhân khối Catmull-Rom, sau đó in tất cả ba kết quả mà không cần thư viện hình ảnh.

## Sử dụng nó

PyTorch thực hiện các hoạt động tương tự trên các tensor có tính năng thiết bị. Mã dưới đây sẽ đổi kích thước bên ngắn hơn, lấy một crop trung tâm, chuẩn hóa mỗi kênh và tạo ra tensor NCHW mà mô hình được đào tạo trước mong đợi.

```python
import torch
import torch.nn.functional as F

image_hwc = torch.from_numpy(synthetic_rgb(256, 320))
batch = image_hwc.permute(2, 0, 1).unsqueeze(0).float() / 255.0

height, width = batch.shape[-2:]
scale = 256 / min(height, width)
resized_height = round(height * scale)
resized_width = round(width * scale)
batch = F.interpolate(
    batch,
    size=(resized_height, resized_width),
    mode="bilinear",
    align_corners=False,
    antialias=True,
)

top = (resized_height - 224) // 2
left = (resized_width - 224) // 2
batch = batch[:, :, top:top + 224, left:left + 224]

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
batch = (batch - mean) / std

print(f"tensor dtype: {batch.dtype}")
print(f"batched shape: {tuple(batch.shape)}")
print(f"per-channel mean: {batch.mean(dim=(0, 2, 3)).tolist()}")
print(f"per-channel std:  {batch.std(dim=(0, 2, 3)).tolist()}")
```

Bốn bước, theo thứ tự chính xác này: chuyển đổi các byte sang nổi và đổi HWC thành NCHW, thay đổi kích thước bên ngắn hơn thành 256, lấy một vùng trung tâm 224x224, sau đó trừ trung bình ImageNet và chia bằng sự lệch chuẩn của nó.

## Chuyển nó

Bài học này mang lại:

- `outputs/prompt-vision-preprocessing-audit.md` một lời nhắc biến bất kỳ thẻ mô hình hoặc thẻ tập hợp dữ liệu nào thành danh sách kiểm tra các biến số tiền xử lý chính xác mà một nhóm phải tôn trọng.
- `outputs/skill-image-tensor-inspector.md` một kỹ năng, nếu xem xét bất kỳ tensor hình ảnh hoặc mảng nào, báo cáo dtype, bố cục, phạm vi, và liệu nó có trông nguyên liệu, bình thường hoặc chuẩn hóa hay không.

## Các bài tập

1. **(Easy)**Tạo một 2x2 RGB `uint8`chuyển đổi HWC thành CHW và trở lại, in cả hai hình dạng, và chứng minh chuyến đi trở lại bảo tồn mọi giá trị.
2. **(Medium)**Hãy viết`standardize(img, mean, std)`và ngược của nó mà cùng nhau đi qua một `roundtrip_max_diff <= 1`Các chức năng của bạn phải hoạt động trên một hình ảnh duy nhất trong HWC và trên một lô trong NCHW với cùng một cuộc gọi.
3. **(Hard)**Hãy lấy một tensor tiêu chuẩn ImageNet 3 kênh và chạy nó qua một con số 1x1 mà học được một hỗn hợp trọng lượng của RGB vào một kênh thang xám.`[0.299, 0.587, 0.114]`, đóng băng chúng, và xác minh đầu ra phù hợp với hướng dẫn của bạn `rgb_to_grayscale`Những biến đổi không gian màu cổ điển nào khác có thể được viết là 1x1 xoắn?

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Pixel | "A coloured square" | One sample of light intensity at one grid location — three numbers for colour, one for grayscale |
| Channel | "The colour" | One of the parallel spatial grids stacked into an image tensor; last axis in HWC, first in CHW |
| HWC / CHW | "The shape" | Axis orderings for an image tensor; disk and PIL use HWC, PyTorch and cuDNN use CHW |
| Normalize | "Scale the image" | Divide by 255 so pixels live in [0, 1] — necessary but not sufficient |
| Standardize | "Zero-center" | Subtract mean and divide by std per channel so the input distribution matches what the model was trained on |
| Grayscale conversion | "Average the channels" | A weighted sum with coefficients 0.299/0.587/0.114 that matches human luminance perception |
| Interpolation | "How resize picks pixels" | The rule that decides output values when the new grid does not align with the old one — nearest for labels, bilinear for training, bicubic for display |
| Aspect ratio | "Width over height" | The ratio that distinguishes "resize and pad" from "resize and stretch" |

## Đọc thêm

- [Charles Poynton — A Guided Tour of Color Space](https://web.archive.org/web/20251220000525/https://poynton.ca/PDFs/Guided_tour.pdf) cách xử lý kỹ thuật rõ ràng nhất về lý do tại sao có quá nhiều không gian màu sắc và khi mỗi một trong số đó quan trọng
- [PyTorch Vision Transforms Docs](https://pytorch.org/vision/stable/transforms.html) toàn bộ đường ống biến đổi bạn thực sự tạo ra trong sản xuất
- [How JPEG Works (Colt McAnlis)](https://www.youtube.com/watch?v=F1kYBnY6mwg) một tour du lịch trực quan sắc nét của mẫu phụ chroma, DCT, và tại sao JPEG mã hóa YCbCr thay vì RGB
- [ImageNet Preprocessing Conventions (torchvision models)](https://pytorch.org/vision/stable/models.html) nguồn gốc của sự thật cho `mean=[0.485, 0.456, 0.406]`và tại sao mọi người mẫu trong vườn thú đều mong đợi nó
