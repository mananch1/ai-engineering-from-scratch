# Processamento de áudio em tempo real

> As pipelines em batch processam um arquivo. As pipelines em tempo real processam os próximos 20 milissegundos antes que os próximos 20 cheguem. Toda IA conversacional, estúdio de transmissão e bot de telefonia vive e morre com este orçamento de latência.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## O problema

Você quer um assistente de voz que se sinta vivo. A latência de conversas humanas é de ~ 230 ms (silêncio-resposta). Qualquer coisa acima de 500 ms parece robótica; acima de 1500 ms parece quebrado. O orçamento para uma completa **hear → understand → respond → speak**Loop em 2026 é:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) obteve 200 ms de duplex completo. GPT-4o-relógio real (2024) relógios ~ 320 ms. Os oleodutos em cascata em 2022 foram enviados a 2500 ms. A melhoria de 10 x veio de três técnicas: (1) streaming em todos os lugares, (2) oleodutos assíncronos com resultados parciais, (3) geração interrompida.

## O conceito

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.**O áudio em tempo real é transmitido como blocos de tamanho fixo.

**Ring buffer.**O filtro de produção escreve novos quadros, o filtro de consumo lê. Impede a alocação no caminho quente. Dimensão ≈ latência máxima × taxa de amostra; um anel de 2 segundos de 16 kHz = 32.000 amostras.

**VAD (Voice Activity Detection).**Os gates funcionam em baixo fluxo quando ninguém está falando. Silero VAD 4.0 (2024) executa < 1 ms por 30 ms de frame na CPU. `webrtcvad`É a alternativa mais antiga.

**Streaming ASR.**Modelos que emitem transcrições parciais à medida que chega o áudio. Parakeet-CTC-0.6B em modo de streaming (NeMo, 2024) faz 25% WER em 320 ms de latência.

**Interruption.**Quando o usuário fala enquanto o assistente está falando, você deve (a) detectar a barge-in, (b) parar o TTS, (c) descartar o resultado restante do LLM. Tudo isso dentro de 100 ms, ou o usuário percebe o assistente surdo.

**WebRTC Opus transport.**20 ms de quadros, 48 kHz, bitrate adaptativa 8128 kbps. padrão para navegador e móvel. LiveKit, Daily.co, Pion são as pilhas 2026 para a construção de aplicativos de voz.

**Jitter buffer.**Pacotes de rede chegam fora de ordem / atrasado. O buffer jitter reordena e suave; muito pequenas → lacunas audíveis, muito grande → latência. 6080 ms típico.

### Gotas comuns

- **Thread contention.**Os modelos pesados GIL + do Python podem deixar de usar o fio de áudio. Use uma biblioteca de áudio de chamadas C (dispositivo de áudio, PortAudio) e mantenha o Python fora do caminho quente.
- **Sample-rate conversion latency.**A re-sampulação dentro do gasoduto adiciona 520 ms. Ou re-sampulação antecipada ou utilização de um re-sampulador de latência zero (PolyPhase, `soxr_hq`)).
- **TTS priming.**Mesmo TTS rápido como Kokoro tem um aquecimento de 100200 ms em primeiro pedido. modelo de cache + aquecê-lo com uma jogada de maniquí antes da primeira virada real.
- **Echo cancellation.**Sem AEC, a saída TTS re-entrada no microfone e desencadeia ASR na própria voz do bot.

```figure
nyquist-aliasing
```

## Construí-lo

### Passo 1: tampão de anel

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

A capacidade determina a latença máxima de amortecimento. 32.000 amostras a 16 kHz = 2 s.

### Passo 2: Porta de VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Substituição por Silero VAD em produção:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Passo 3: streaming de ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Passo 4: Gestor de interrupção

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

Assincroniza-se a I/O e o streaming TTS cancelável.

## Usá-lo

A pilha de 2026:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## Encurralagens

- **Buffering 500 ms to be safe.**O amortecedor é o teu piso de latência.
- **Not pinning threads.**Recall de áudio em um fio de prioridade inferior à UI = falhas sob carga.
- **TTS chunks too small.**Os fragmentos sub-200 ms fazem os artefatos do vocoder sonoros. Os fragmentos 320 ms são o ponto ideal.
- **No jitter buffer.**As redes reais são nervosas; sem suavizar, você fica com um pop.
- **Single-shot error handling.**Os canais de áudio devem ser resistentes a acidentes.

## Envia-o

Salva como`outputs/skill-realtime-designer.md`- Projetar um canal de áudio em tempo real com orçamentos concretos de latência por etapa.

## Exercícios

1. **Easy.**Corra .`code/main.py`Simula um tampão de anel + energia VAD; Imprime latências de fase para uma corrente falsa de 10 segundos.
2. **Medium.**Usando`sounddevice`, construir um loop que processa o microfone em 20 ms de quadros e imprime estado VAD em cada quadro.
3. **Hard.**Construa um teste de eco duplex completo com `aiortc`: browser → WebRTC → Python → WebRTC → browser. Medir latência de vidro a vidro com um pulso de 1 kHz.

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## Mais leitura

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743)- Um sussurro quase que fluindo.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) Latência de 200 ms de duplex completo.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/)- Orquestração de agentes de produção.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) sub-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) cancelamento de eco em código aberto.
