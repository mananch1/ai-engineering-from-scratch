# Phân tích âm thanh thời gian thực

> Các đường ống hàng xử lý một tập tin. Các đường ống hàng thời gian thực xử lý 20 millisecond tiếp theo trước khi 20 giây tiếp theo đến. Mỗi AI trò chuyện, studio phát sóng và bot điện thoại sống và chết bằng ngân sách trễ này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Vấn đề

Bạn muốn một trợ lý giọng nói cảm thấy sống. thời gian trễ chuyển đổi cuộc trò chuyện của con người là ~ 230 ms (trầm lặng để trả lời). Bất cứ điều gì trên 500 ms cảm thấy robot; trên 1500 ms cảm thấy vỡ. Ngân sách cho một đầy đủ **hear → understand → respond → speak**vòng vào năm 2026 là:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) đã đồng hồ 200 ms đầy đủ duplex. GPT-4o-time (2024) đồng hồ ~ 320 ms. Các đường ống nước ngập vào năm 2022 được vận chuyển với 2500 ms. Sự cải tiến 10x đến từ ba kỹ thuật: (1) truyền khắp nơi, (2) đường ống không đồng bộ với kết quả một phần, (3) sản xuất bị gián đoạn.

## Khái niệm

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.**Đường độ âm thanh trong thời gian thực được chuyển đổi thành khối kích thước cố định.

**Ring buffer.**Bộ đệm vòng tròn kích thước cố định. Dòng sản xuất viết khung mới, dây tiêu dùng đọc. ngăn chặn phân bổ trong đường nóng. kích thước ≈ độ trễ tối đa × tốc độ mẫu; vòng 16 kHz 2 giây = 32.000 mẫu.

**VAD (Voice Activity Detection).**Gates hoạt động dòng chảy xuống khi không ai nói. Silero VAD 4.0 (2024) chạy < 1 ms mỗi khung hình 30 ms trên CPU. `webrtcvad`là sự thay thế cũ hơn.

**Streaming ASR.**Các mô hình phát ra bản ghi âm một phần khi âm thanh đến. Parakeet-CTC-0.6B trong chế độ phát trực tuyến (NeMo, 2024) thực hiện 25% WER với độ trễ 320 ms. Whisper-Streaming (Macháček et al., 2023) phân đoạn Whisper cho gần phát trực tuyến với độ trễ ~ 2 s.

**Interruption.**Khi người dùng nói trong khi trợ lý đang nói, bạn phải (a) phát hiện sự đột nhập, (b) dừng TTS, (c) loại bỏ các kết quả LLM còn lại. Tất cả trong vòng 100 ms, hoặc người dùng nhận thấy trợ lý điếc.

**WebRTC Opus transport.**20 ms khung hình, 48 kHz, tốc độ bit thích ứng 8128 kbps. tiêu chuẩn cho trình duyệt và di động. LiveKit, Daily.co, Pion là các gói 2026 để xây dựng ứng dụng thoại.

**Jitter buffer.**Các gói mạng đến không đúng giờ / muộn. bộ đệm jitter sắp xếp lại và làm mượt mà; khoảng trống nhỏ quá → nghe thấy, quá lớn → độ trễ. 6080 ms điển hình.

### Thị trường chung

- **Thread contention.**Các mô hình nặng GIL + của Python có thể làm mất đi dây chuyền âm thanh. Sử dụng thư viện âm thanh C-callback (hỗ máy âm thanh, PortAudio) và giữ Python khỏi con đường nóng.
- **Sample-rate conversion latency.**Phân tích lại bên trong đường ống thêm 520 ms. Hoặc lấy lại mẫu trước hoặc sử dụng một mẫu lại không trễ (PolyPhase, `soxr_hq`().
- **TTS priming.**Ngay cả TTS nhanh như Kokoro cũng có tốc độ nóng lên 100200 ms khi yêu cầu đầu tiên.
- **Echo cancellation.**Không có AEC, đầu ra TTS quay lại vào micrô và kích hoạt ASR trên giọng nói của bot. WebRTC AEC3 là mặc định nguồn mở.

```figure
nyquist-aliasing
```

## Hãy xây dựng nó

### Bước 1: bơm vòng

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Công suất xác định độ trễ tối đa. 32.000 mẫu ở 16 kHz = 2 s.

### Bước 2: Cổng VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Thay thế bằng Silero VAD trong sản xuất:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Bước 3: phát ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Bước 4: xử lý gián đoạn

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Hinges trên I / O async và phát TTS hủy bỏ. WebRTC peerconnection.stop() trên các bài hát âm thanh là cách truyền thống.

## Sử dụng nó

Số 2026:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## Những bẫy

- **Buffering 500 ms to be safe.**Buffer là tầng độ trễ của bạn.
- **Not pinning threads.**Phục hồi âm thanh trên một chuỗi ưu tiên thấp hơn UI = lỗi dưới tải.
- **TTS chunks too small.**Các mảnh dưới 200 ms làm cho các vật thể vocoder được nghe thấy. 320 ms là điểm ngọt ngào.
- **No jitter buffer.**Các mạng lưới thực sự là căng thẳng; mà không làm trơn, bạn có thể bị bùng nổ.
- **Single-shot error handling.**Các ống dẫn âm thanh phải không bị tai nạn.

## Chuyển nó

Cứ như `outputs/skill-realtime-designer.md`Thiết kế một đường ống âm thanh thời gian thực với ngân sách độ trễ cụ thể cho mỗi giai đoạn.

## Các bài tập

1. **Easy.**Đi chạy`code/main.py`. Mô phỏng một bộ đệm vòng + năng lượng VAD; in độ trễ giai đoạn cho một dòng 10 giây giả.
2. **Medium.**Sử dụng `sounddevice`, xây dựng một vòng lặp qua mà xử lý micro của bạn trong 20 ms khung hình và in trạng thái VAD tại mỗi khung hình.
3. **Hard.**Xây dựng một thử nghiệm echo duplex đầy đủ với `aiortc`: browser → WebRTC → Python → WebRTC → browser. đo độ trễ kính-vàng với xung 1 kHz.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## Đọc thêm

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) Chúc rắc gần như đang phát sóng.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) Full duplex 200 ms latency.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) sản xuất âm thanh đại lý dàn nhạc.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) Sub-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) Pháo âm thanh trong mã nguồn mở.
