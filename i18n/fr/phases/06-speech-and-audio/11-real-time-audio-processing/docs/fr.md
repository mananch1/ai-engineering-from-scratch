# Traitement audio en temps réel

> Les pipelines en série traitent un fichier. Les pipelines en temps réel traitent les 20 millisecondes suivantes avant l'arrivée des 20 suivantes. Chaque IA de conversation, studio de diffusion et robot téléphonique vit et meurt avec ce budget de latence.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Le problème

Vous voulez un assistant vocal qui se sent vivant. La latence de prise de tour de conversation humaine est d'environ 230 ms (silence à réponse). Tout ce qui dépasse 500 ms se sent robotique; au-dessus de 1500 ms se sent cassé. Le budget pour une pleine**hear → understand → respond → speak**la boucle en 2026 est:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) a réalisé 200 ms de double-duplex. GPT-4o en temps réel (2024) montre ~ 320 ms. Les pipelines en cascade en 2022 ont été expédiées à 2500 ms. L'amélioration de 10 fois est venue de trois techniques: (1) streaming partout, (2) pipeline asynchrone avec des résultats partiels, (3) génération interrompue.

## Le concept

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.**Les flux audio en temps réel sont des blocs de taille fixe.

**Ring buffer.**Bouffer circulaire de taille fixe. Le fil de producteur écrit de nouveaux cadres, le fil de consommation est lu. empêche l'allocation dans le chemin chaud. Taille ≈ latence maximale × taux d'échantillonnage; un anneau de 2 secondes 16 kHz = 32 000 échantillons.

**VAD (Voice Activity Detection).**Les portes fonctionnent en aval quand personne ne parle. Silero VAD 4.0 (2024) fonctionne < 1 ms par frame de 30 ms sur CPU. `webrtcvad`est l'alternative plus ancienne.

**Streaming ASR.**Modèles qui émettent des transcriptions partielles à l'arrivée de l'audio. Parakeet-CTC-0.6B en mode streaming (NeMo, 2024) fait 25% WER à 320 ms de latence.

**Interruption.**Lorsque l'utilisateur parle pendant que l'assistant parle, vous devez (a) détecter le barge-in, (b) arrêter le TTS, (c) jeter le reste de la sortie LLM. Tout cela dans un délai de 100 ms, ou l'utilisateur perçoit l'assistant sourd.

**WebRTC Opus transport.**20 ms de cadres, 48 kHz, bitrate adaptative 8128 kbps. Standard pour navigateur et mobile. LiveKit, Daily.co, Pion sont les piles 2026 pour la construction d'applications vocales.

**Jitter buffer.**Les paquets réseau arrivent en panne / en retard. Le tampon jitter réordonne et se déplace; trop petits → espaces audibles, trop grands → latence. 6080 ms typique.

### Les goutches communes

- **Thread contention.**Les modèles lourds GIL + de Python peuvent affamer le fil audio. Utilisez une bibliothèque audio C-callback (appareil sonore, PortAudio) et gardez Python hors du chemin chaud.
- **Sample-rate conversion latency.**Le reéchantillonnage à l'intérieur du pipeline ajoute 520 ms.`soxr_hq`)
- **TTS priming.**Même un TTS rapide comme Kokoro a un réchauffement de 100 à 200 ms sur première demande.
- **Echo cancellation.**Sans AEC, la sortie TTS rentre dans le micro et déclenche ASR sur la voix du robot.

```figure
nyquist-aliasing
```

## Faites-le

### Étape 1: tampon à anneaux

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

La capacité détermine la latence maximale du tampon. 32 000 échantillons à 16 kHz = 2 s.

### Étape 2: Porte de détection

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Remplacez par Silero VAD en production:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Étape 3: diffusion de l' ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Étape 4: gestionnaire d'interruption

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

Le lien est en continu avec le réseau TTS.

## Utilisez-le

La pile de 2026:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## Les pièges

- **Buffering 500 ms to be safe.**Le tampon est votre plancher de latence.
- **Not pinning threads.**Rappel d'audio sur un fil de priorité inférieure à l'interface utilisateur = défauts sous chargement.
- **TTS chunks too small.**Les pièces sous 200 ms rendent audibles les objets du vocoder.
- **No jitter buffer.**Les réseaux réels sont nerveux; sans lisser, on se fait des coups.
- **Single-shot error handling.**Les conduites audio doivent être résistantes aux chocs.

## La faire partir

- Je ne sais pas .`outputs/skill-realtime-designer.md`- Conception d'un pipeline audio en temps réel avec des budgets concrets de latence par étape.

## Exercices

1. **Easy.**On court .`code/main.py`Simulation d'un tampon d'anneau + VAD d'énergie; imprime les latences de l'étape pour un faux flux de 10 secondes.
2. **Medium.**En utilisant `sounddevice`, construire un passage à travers la boucle qui traite votre micro en 20 ms cadres et imprime l'état de VAD à chaque cadre.
3. **Hard.**Construisez un test d' écho duplex complet avec `aiortc`: navigateur → WebRTC → Python → WebRTC → navigateur. Mesurer la latence verre à verre avec un pulsation de 1 kHz.

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## Pour en savoir plus

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743)- Il y a des morceaux de "Whisper" qui circulent.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) La latence de 200 ms à double intégral.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) orchestration d'agents audio de production.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) sous-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) annulation de l'écho sous source ouverte.
