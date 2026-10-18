# Các mô hình ngôn ngữ video: Các biểu tượng thời gian và đặt đất

> Video không phải là một đống ảnh. Một clip 5 giây có thứ tự nguyên nhân, động từ hành động và thời gian sự kiện mà mô hình hình ảnh không thể đại diện. Video-LLaMA (Zhang et al., tháng 6 năm 2023) đã xuất khẩu video-LLM mở đầu tiên với nền tảng âm thanh-quan hình. VideoChat và Video-LLaVA đã mở rộng mô hình. Đến năm 2025, TMRoPE của Qwen2.5-VL đã thu hẹp khoảng cách với các mô hình độc quyền biên giới. Mỗi hệ thống giải quyết các token thời gian khác nhau  Q-former mỗi clip, concat-pool mỗi khung, TMRoPE mỗi token. Bài học này đọc các mẫu, xây dựng một mẫu khung đồng nhất so với động lực, và đánh giá về các nhiệm vụ đặt đất thời gian.

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## Mục tiêu học tập

- Giải thích lý do tại sao mã hóa vị trí thời gian thay đổi hiệu suất video VLM độc lập với mã hóa tầm nhìn.
- So sánh mẫu khung đồng nhất, động-FPS, và các sự kiện-động hành trên token-per-second so với độ chính xác đất.
- Mô tả các thiết kế Q-ex-per-clip (Video-LLaMA) vs pooled-per-frame (Video-LLaVA) vs M-RoPE-per-token (Qwen2.5-VL).
- Hãy nêu tên bốn tiêu chuẩn video: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## Vấn đề

Một video 1 phút ở 30 FPS là 1800 khung hình. Với 196 mã thông báo trực quan mỗi khung hình (ViT-B ở 224), đó là 352k mã thông báo lớn hơn bất kỳ bối cảnh LLM nào trong kỷ nguyên 2024 .

Có ba chiến lược giảm thiểu:

1. Các khung mẫu nhỏ (1-8 FPS tùy thuộc vào nội dung).
2. Đặt các mã đệm của mỗi khung một cách hung hăng (3x3 hoặc 4x4 pool hàng tỷ).
3. Nén qua một máy Q-former lấy clip 16 khung và phát ra 64 token.

Mỗi trade-off khác nhau. Subsampling mất chi tiết thời gian. Pooling mất chi tiết không gian. Q-former mất cả hai một chút nhưng tiết kiệm token.

Mã hóa vị trí thời gian là trục khác: mô hình làm thế nào biết khung 5 đến trước khung 6? Các tùy chọn bao gồm RoPE thời gian 1D đơn giản (Video-LLaMA), nhúng thời gian học (Video-LLaVA), và TMRoPE (Qwen2.5-VL, 3D đầy đủ).

## Khái niệm

### Video-LLaMA: Q-former mỗi clip + nhánh âm thanh

Video-LLaMA (2023) là video-LLM mở đầu tiên.

- 16 khung hình clip với 2 FPS (vì vậy 8 giây).
- Các tính năng ViT mỗi khung -> Video Q-former phục vụ qua tất cả 16 khung -> 32 truy vấn được học -> LLM.
- Chi nhánh âm thanh song song: hình dạng sóng -> ImageBind âm thanh mã hóa -> Audio Q-former -> 32 truy vấn -> LLM.

Nguyên: lý luận âm thanh-thông quan.

### VideoChat và Video-LLaVA

VideoChat giữ ý tưởng Video-LLaMA nhưng bỏ âm thanh và đơn giản hóa. Video-LLaVA (Lin et al., 2023) đào tạo một bộ mã hóa thị giác duy nhất trên cả hình ảnh và khung video ("sự sắp xếp trước khi chiếu"), cung cấp một đại diện thống nhất. Cả hai đều là mã hóa CLIP đóng băng + MLP + LLM.

Cả hai đều là hệ thống khung hình 8-16.

### Qwen2.5-VL và TMRoPE

Qwen2.5-VL đã giới thiệu TMRoPE  Temporal-Modality Rotary Position Embedding. Mỗi mã đệm mang một vị trí (t, h, w) nơi t là dấu thời gian thực tế (không phải chỉ số khung).

Sự khác biệt chính từ việc nhúng thời gian đơn giản:

- Thời gian tuyệt đối, không chỉ số. mô hình thấy "tới 4,2 giây" chứ không phải "tới khung 15".
- Mỗi token quay không phải mỗi clip, mỗi token hình ảnh quay tự do theo dấu thời gian của nó.
- Nếu bạn lấy mẫu ở 2 FPS ở đây và 4 FPS ở đó, TMRoPE xử lý khoảng cách không đồng đều một cách tự nhiên.

TMRoPE cho phép truy vấn "tại giây nào mèo nhảy?" mô hình có thể phát ra "tới 4,2 giây". Video-LLaMA chỉ có thể nói "trước khi trong clip".

### Các chiến lược lấy mẫu khung

Một dạng: mẫu N khung đồng đều trong thời gian. đơn giản, mất đỉnh chuyển động.

FPS động: mẫu thích ứng dựa trên cường độ chuyển động. Phân tích luồng quang học hoặc khung chọn các phân đoạn chuyển động cao để lấy mẫu dày đặc hơn. Qwen2.5VL chạy trên điều này.

Động cơ sự kiện: chạy một máy dò nhẹ, lấy mẫu hơn về nơi xảy ra hành động.

Keyframe + context: mẫu ở ranh giới chụp + một vài khung lân cận. Được sử dụng cho nội dung phim.

### Tỷ lệ hợp nhất mỗi khung

Với 1 FPS và 576 token mỗi khung hình, một clip 5 phút là 172.800 token.

3x3 pool hàng tỷ đường giảm xuống còn 64 token mỗi khung -> 19.200 token trong 5 phút. điểm dễ dàng cho hầu hết các nhiệm vụ.

Đặt một nhóm mạnh mẽ hơn (6x6 -> 16 token mỗi khung) cho các dòng công việc của đại lý nơi chi tiết không gian ít quan trọng hơn.

### Bốn tiêu chuẩn video

- VideoMME: hiểu biết video toàn diện, ngắn + trung bình + dài.
- TempCompass: lý luận thời gian tinh tế, "trước" / "sau" câu hỏi.
- EgoSchema: video người đầu tiên dài.
- Video-MMMU: các câu hỏi video đa phương pháp đa ngành.

Một đánh giá video-VLM đầy đủ chạm đến tất cả bốn. Họ nhấn mạnh các trục khác nhau  TempCompass là tất cả về đặt hàng, EgoSchema là về 3 + phút lý luận, VideoMME trải dài thời gian.

### Các định dạng đầu ra trục trặc

Các định dạng đầu ra cho việc đặt đất thời gian:

- "Căn cưng nhảy quanh dấu hiệu 4 giây". Mời phân tích nhưng không chính xác.
- JSON được cấu trúc: `{"event": "jump", "start": 4.1, "end": 4.3}`- Qwen2.5VL chạy theo.
- Dựa trên token: đặc biệt `<time>4.1</time>`- Đơn vị nội bộ của Qwen2.5VL.

Xây dựng dựa trên token chính xác nhất cho việc sử dụng dòng chảy. định dạng đầu ra JSON của Qwen2.5VL phân tích trực tiếp.

### 2026 thực hành tốt nhất

Đối với VLM video vào năm 2026:

- Bộ mã hóa: SigLIP 2 với M-RoPE hoặc TMRoPE (Qwen2.5-VL).
- Phân mẫu khung: FPS động (1-4 tùy thuộc vào chuyển động) với nắp khung tối đa.
- Phân hợp mỗi khung: 3x3 hàng tỉ.
- Kết quả: cấu trúc JSON với các trường thời gian + sự kiện.
- Các điểm chuẩn: VideoMME + TempCompass cho chung; EgoSchema cho tầm xa.

```figure
video-temporal-patches
```

## Sử dụng nó

`code/main.py`bao gồm:

- Các mẫu khung FPS đồng nhất và động.
- Một đánh giá thời gian đồ chơi: với sự kiện "thực tế cơ bản" tại thời gian T và một sản xuất mô hình, ghi điểm chính xác với dung nạp.
- Một so sánh giữa Video-LLaMA (16 khung hình, Q-ex), Video-LLaVA (8 khung hình, MLP), Qwen2.5-VL (dynamic FPS + TMRoPE).

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/skill-video-vlm-frame-planner.md`. Với một nhiệm vụ video (phòng dõi, nhận dạng hành động, định vị thời gian, tổng kết), nó chọn mẫu khung, yếu tố hợp nhất, định dạng đầu ra và mức độ chính xác dự kiến.

## Các bài tập

1. Để xem 3 phút, chọn đồng phục vs FPS động.

2. TMRoPE thêm vào những gì cụ thể mà một bảng nhúng thời gian đơn giản không thể làm?

3. Viết một sơ đồ JSON cho việc đặt đất thời gian mà một VLM có thể học được phát ra. Bao gồm các trường hợp lỗi.

4. Đọc phần 3 của Video-LLaVA về "Sự sắp xếp trước khi chiếu". Tại sao điều này tốt hơn so với việc đào tạo các bộ mã hóa hình ảnh và video riêng biệt?

5. Với bảng xếp hạng VideoMME, khoảng cách giữa mô hình mở hàng đầu và mô hình độc quyền hàng đầu vào năm 2026 là gì?

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Temporal grounding | "Time-localized answers" | VLM outputs a specific timestamp range for when an event happens |
| TMRoPE | "Time-Multimodal RoPE" | 3D rotary position with absolute timestamps, used by Qwen2.5-VL |
| Dynamic FPS | "Motion-aware sampling" | Sample more frames in high-motion segments, fewer in static ones |
| Frame pooling | "Spatial compress per frame" | Reduce patches per frame with bilinear interpolation before the LLM |
| Video Q-former | "Clip compressor" | Cross-attention bottleneck mapping N frames to K learned queries |
| VideoMME | "Video bench" | Comprehensive short/medium/long video benchmark, 2500+ samples |

## Đọc thêm

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
