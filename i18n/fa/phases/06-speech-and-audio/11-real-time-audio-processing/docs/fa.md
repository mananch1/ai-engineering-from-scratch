# پردازش صوتی در زمان واقعی

> لوله های دسته ای یک فایل را پردازش می کنند. لوله های زمان واقعی 20 میلی ثانیه آینده را قبل از رسیدن 20 ثانیه بعدی پردازش می کنند. هر هوش مصنوعی مکالمه ای، استودیو پخش و ربات تلفن با این بودجه تاخیر زندگی می کند و می می میرد.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## مشکل

شما می خواهید یک دستیار صوتی که احساس زنده است. تاخیر زمان گیری مکالمه انسانی حدود 230 ms است. هر چیزی بالاتر از 500 ms به نظر می رسد رباتیک است. بالاتر از 1500 ms به نظر می رسد شکسته است. بودجه برای یک کامل **hear → understand → respond → speak**در سال 2026 این حلقه:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

موشی (کیوتای، 2024) 200 ms کامل دوگونی را انجام داد. ساعت های GPT-4o در زمان واقعی (2024) ~ 320 ms. لوله های Cascaded در سال 2022 با 2500 ms ارسال شدند. بهبود 10x از سه تکنیک ناشی شد: (1) پخش در همه جا ، (2) لوله های غیر هماهنگ با نتایج جزئی ، (3) تولید قطع می شود.

## مفهوم

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.**جریان صوتی زمان واقعی به عنوان بلوک های اندازه ثابت. انتخاب رایج: 20 ms (320 نمونه در 16 kHz). همه چیز در جریان پایین باید با این کادانس همراه باشد.

**Ring buffer.**پوشک دایره ای با اندازه ثابت. رشته تولید کننده فریم های جدید را می نویسد، رشته مصرف کننده را می خواند. از اختصاص در مسیر داغ جلوگیری می کند. اندازه ≈ حداکثر تاخیر × نرخ نمونه؛ حلقه 2 ثانیه 16 kHz = 32000 نمونه.

**VAD (Voice Activity Detection).**گیتس در زمان صحبت کردن با هیچ کس کار نمی کند. Silero VAD 4.0 (2024) در CPU <1 ms در هر فریم 30 ms اجرا می شود. `webrtcvad`این گزینه قدیمی تر است.

**Streaming ASR.**مدل هایی که در هنگام ورود صوتی نقل و نقل جزئی را منتشر می کنند. Parakeet-CTC-0.6B در حالت جریان (NeMo، 2024) 25% WER را در تاخیر 320 ms انجام می دهد. Whisper-Streaming (Macháček و همکارانش، 2023) قطعات Whisper برای نزدیک به جریان در تاخیر ~ 2 ثانیه را تشکیل می دهد.

**Interruption.**وقتی کاربر در حالی که دستیار صحبت می کند صحبت می کند، باید (ا) بارج-این را تشخیص دهید، (ب) TTS را متوقف کنید، (ج) محصول LLM باقی مانده را کنار بگذارید. همه اینها در عرض 100 ms، یا کاربر دستیار ناشنوا را درک می کند.

**WebRTC Opus transport.**20ms فریم، 48 kHz، سرعت بایت سازنده 8128 kbps. استاندارد برای مرورگر و تلفن همراه. LiveKit، Daily.co، Pion استک های 2026 برای ساخت برنامه های صدا هستند.

**Jitter buffer.**بسته های شبکه از راه رفته / دیر می رسند. بفر jitter تغییر ترتیب می دهد و صاف می شود؛ شکاف های کوچک → شنیدنی، تاخیر بیش از حد بزرگ. 6080 ms معمول است.

### گات ها

- **Thread contention.**مدل های سنگین GIL + پایتون می توانند موضوع صوتی را از بین ببرند. از کتابخانه صوتی C-callback (جهاز صوتی، PortAudio) استفاده کنید و پایتون را از مسیر داغ دور نگه دارید.
- **Sample-rate conversion latency.**نمونه برداری مجدد در داخل لوله 520 ms اضافه می کند. یا نمونه گیری مجدد از قبل یا استفاده از نمونه گیری مجدد صفر تاخیر (PolyPhase، `soxr_hq`)
- **TTS priming.**حتی TTS سریع مثل Kokoro هم در اولین درخواست 100200 ms گرم می شود. مدل کیش + گرم کردن آن با یک اجرا ساختگی قبل از اولین باری واقعی.
- **Echo cancellation.**بدون AEC، TTS output دوباره وارد میکروفون می شود و ASR را در صدای خود ربات فعال می کند. WebRTC AEC3 پیش فرض منبع باز است.

```figure
nyquist-aliasing
```

## آن را بسازید

### مرحله اول: باز کننده حلقه

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

ظرفیت مشخص می کنه حداکثر تاخیر بفرنگ 32000 نمونه در 16 kHz = 2 ثانیه

### مرحله دوم: دروازه VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

جایگزین Silero VAD در تولید:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### مرحله 3: پخش ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### مرحله 4: کنترل قطع

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

به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به عنوان یک برنامه ای که به طور کلی به طور کلی به عنوان یک برنامه ای که به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور کلی به طور ممکن است.

## ازش استفاده کن

دسته 2026:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## دام ها

- **Buffering 500 ms to be safe.**بفر کف تاخير شماست.
- **Not pinning threads.**بازگشت صدا در یک رشته اولویت پایین تر از UI = خرابی در زیر بار.
- **TTS chunks too small.**قطعات زیر 200 ms باعث می شود آثار Vocoder شنیده شود. قطعات 320 ms نقطه خوشایند هستند.
- **No jitter buffer.**شبکه های واقعی عصبانی هستند؛ بدون نرم کردن شما پاپ می گیرید.
- **Single-shot error handling.**لوله هاي صداي بايد ضد تصادف باشند.

## -باده

پس از`outputs/skill-realtime-designer.md`طراحی یک خط صدا در زمان واقعی با بودجه تاخیر کنکریتی در هر مرحله.

## تمرینات

1. **Easy.**فرار کن`code/main.py`. یک بازنده حلقه + انرژی VAD را شبیه سازی می کند؛ تاخیر مرحله ای را برای جریان 10 ثانیه جعلی چاپ می کند.
2. **Medium.**استفاده کردن`sounddevice`، یک حلقه عبور ایجاد کنید که میکروفون شما را در 20ms فریم پردازش می کند و حالت VAD را در هر فریم چاپ می کند.
3. **Hard.**با استفاده از  یک تست دوپلیکس کامل را بسازید`aiortc`: مرورگر → WebRTC → پایتون → WebRTC → مرورگر. طول طول گلس به گلس را با یک نبض 1 kHz اندازه گیری کنید.

## اصطلاحات کلیدی

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## خواندن بیشتر

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) "سسپر" که نزدیک به جریان داره
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) دوپلیکس کامل 200 ms تاخیر
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده سازنده
- [Silero VAD repo](https://github.com/snakers4/silero-vad) sub-1 ms VAD، آپاچی 2.0
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) حذف بازگشت در زیر منبع باز
