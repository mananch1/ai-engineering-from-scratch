# Modelos de Língua em Vídeo: Tokens Temporais e Grounding

> O vídeo não é uma pilha de fotos. Um clip de 5 segundos tem ordem causal, verbos de ação e cronometragem de eventos que um modelo de imagem não pode representar. Video-LLaMA (Zhang et al., junho 2023) enviou o primeiro open video-LLM com a terração audiovisual. O VideoChat e o Video-LLaVA escalaram o padrão. Até 2025, o TMRoPE da Qwen2.5-VL encerrou a lacuna com os modelos proprietários de fronteira. Cada sistema resolveu tokens temporais de forma diferente  Q-former por clip, concat-pool por frame, TMRoPE por token. Esta lição lê os padrões, constrói um amostragem de quadro uniforme versus dinâmico e avalia as tarefas de aterragem temporal.

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## Objetivos de aprendizagem

- Explique por que a codificação posicional temporal altera o desempenho do VLM de vídeo independentemente do codificador de visão.
- Compare a amostragem uniforme, dinâmica-FPS e de quadros orientados por eventos em tokens-por-segundo versus precisão de aterragem.
- Descreva Q-former-per-clip (Video-LLaMA) vs pooled-per-frame (Video-LLaVA) vs M-RoPE-per-token (Qwen2.5-VL) projetos.
- Cite os quatro critérios de referência para o vídeo: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## O problema

Um vídeo de 1 minuto a 30 FPS é de 1800 quadros. A 196 tokens visuais por quadro (ViT-B em 224), que é 352k tokens  maior do que qualquer contexto LLM da era de 2024.

Existem três estratégias de redução:

1. Quadros de submuestra (1-8 FPS dependendo do conteúdo).
2. Agrupar os tokens de patch de cada quadro de forma agressiva (3x3 ou 4x4 pool bilinear).
3. Compressão através de um Q-former que toma um clip de 16 quadros e sai 64 tokens.

Cada troca é diferente. sub-sampling perde detalhes temporais. pooling perde detalhes espaciais. Q-former perde um pouco, mas economiza tokens.

A codificação de posição temporal é o outro eixo: como o modelo sabe que o quadro 5 veio antes do quadro 6? As opções incluem RoPE temporal simples 1D (Video-LLaMA), incorporações temporais aprendidas (Video-LLaVA) e TMRoPE (Qwen2.5-VL, 3D completa).

## O conceito

### Video-LLaMA: Q-former por clip + ramo de áudio

Video-LLaMA (2023) foi o primeiro vídeo-LLM aberto. Arquitetura:

- Clip de 16 quadros a 2 FPS (de 8 segundos).
- Características de ViT por quadro -> Video Q-former que atende cruzando sobre todos os 16 quadros -> 32 consultas aprendidas -> LLM.
- Ramo de áudio paralelo: forma de onda -> Encoder de áudio ImageBind -> Áudio Q-former -> 32 consultas -> LLM.

Força: raciocínio conjunto audiovisual. Fraqueza: comprimento fixo do clip, sem fixação arbitrária de tempo.

### VideoChat e Video-LLaVA

O VideoChat manteve a ideia de Video-LLaMA, mas deixou cair o áudio e simplificou. O Video-LLaVA (Lin et al., 2023) treinou um único codificador visual em ambas as imagens e quadros de vídeo ("alinhamento antes da projeção"), dando uma representação unificada. Ambos são o codificador CLIP congelado + MLP + LLM.

Nenhum deles lida com vídeo longo.

### Qwen2,5-VL e TMRoPE

Qwen2.5-VL introduziu TMRoPE  Temporal-Modality Rotary Position Embedding. Cada patch token carrega uma posição (t, h, w) onde t é o timestamp real (não o índice de quadro).

Diferenças fundamentais do simples incorporamento temporal:

- O modelo vê "em 4,2 segundos" e não "no quadro 15".
- Cada token visual gira independentemente por sua marca de tempo.
- Compativel com FPS dinâmico. Se você amostrar a 2 FPS aqui e 4 FPS ali, o TMRoPE lida com o espaçamento desigual de forma nativa.

O TMRoPE permite que "em que segundo o gato salta?" consultas. O modelo pode emitir "em 4,2 segundos".

### Estratexias de amostragem de quadro

Uniforme: mostra N quadros uniformemente ao longo da duração.

FPS dinâmico: amostra adaptativamente com base na intensidade de movimento. fluxo óptico ou diferenciação de quadro escolhe segmentos de alta moção para amostragem mais densa. Qwen2.5-VL traz sobre isso.

Event-driven: executar um detector leve, amostrar mais onde a ação acontece.

Quadro de teclado + contexto: amostra em limites de tomadas + alguns quadros adjacentes.

### Reunião por quadro

A 1 FPS e 576 tokens por quadro, um clip de 5 minutos é de 172.800 tokens.

3x3 bilinear pool reduz para 64 tokens por quadro -> 19.200 tokens por 5 minutos.

Poem mais agressivamente (6x6 -> 16 tokens por quadro) para fluxos de trabalho de agentes onde o detalhe espacial importa menos.

### Os quatro critérios de referência de vídeo

- VideoMME: compreensão completa do vídeo, curto + médio + longo.
- TempCompass: raciocínio temporal de graus finos, perguntas "antes" / "depois".
- EgoSchema: vídeo em primeira pessoa de longo horizonte.
- Video-MMMU: perguntas multimodal de vídeo multidisciplinares.

Uma avaliação completa de vídeo-VLM atinge os quatro. Eles enfatizam diferentes eixos  TempCompass é tudo sobre encomendar, EgoSchema é sobre 3 + minutos de raciocínio, VideoMME abrange durações.

### Formatos de saída de aterragem

Formatos de saída para aterragem temporal:

- "O gato salta ao redor da marca de 4 segundos". Facil de analisar mas impreciso.
- JSON estruturado: `{"event": "jump", "start": 4.1, "end": 4.3}`O Qwen2.5VL treina isto.
- Baseado em tokens: especial `<time>4.1</time>`Os tokens estão entrelaçados com a resposta.

O formato de saída JSON do Qwen2.5VL analisa diretamente.

### 2026 melhores práticas

Para VLMs de vídeo em 2026:

- Encoder: SigLIP 2 com M-RoPE ou TMRoPE (Qwen2.5-VL).
- Amostragem de quadro: FPS dinâmico (1-4 dependendo do movimento) com tampa máxima do quadro.
- A combinação por quadro: 3x3 bilinear.
- Saída: JSON estruturado com campos de tempo + evento.
- Referências: VideoMME + TempCompass para geral; EgoSchema para longo horizonte.

```figure
video-temporal-patches
```

## Usá-lo

`code/main.py`inclui:

- Amplificadores de quadros de FPS uniformes e dinâmicos.
- Um avaliador de fixação temporal de brinquedos: dado um evento de "verdade básica" no tempo T e uma saída de modelo, pontuação de precisão com tolerância.
- Uma comparação entre o Video-LLaMA (16 quadros, Q-former), o Video-LLaVA (8 quadros, MLP), o Qwen2.5-VL (FPS dinâmico + TMRoPE).

## Envia-o

Esta lição produz`outputs/skill-video-vlm-frame-planner.md`. Dada uma tarefa de vídeo (monitorização, reconhecimento de ação, fixação temporal, resumo), ele escolhe o amostragem de quadro, o factor de agregação, o formato de saída e o nível de precisão esperado.

## Exercícios

1. Para uma demonstração de 3 minutos, escolha uniforme vs FPS dinâmico.

2. O TMRoPE acrescenta o que especificamente uma simples tabela de inserção temporal não pode fazer?

3. Escreva um esquema JSON para o aterramento temporal que um VLM possa aprender a emitir. Incluir casos de erro.

4. Leia a Seção 3 do Video-LLaVA sobre "Alignment Before Projection". Por que é melhor que treinar codificadores de imagem e vídeo separados?

5. Dada a classificação de vídeo-ME, qual é a diferença entre o modelo aberto superior e o modelo proprietário superior em 2026?

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Temporal grounding | "Time-localized answers" | VLM outputs a specific timestamp range for when an event happens |
| TMRoPE | "Time-Multimodal RoPE" | 3D rotary position with absolute timestamps, used by Qwen2.5-VL |
| Dynamic FPS | "Motion-aware sampling" | Sample more frames in high-motion segments, fewer in static ones |
| Frame pooling | "Spatial compress per frame" | Reduce patches per frame with bilinear interpolation before the LLM |
| Video Q-former | "Clip compressor" | Cross-attention bottleneck mapping N frames to K learned queries |
| VideoMME | "Video bench" | Comprehensive short/medium/long video benchmark, 2500+ samples |

## Mais leitura

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
