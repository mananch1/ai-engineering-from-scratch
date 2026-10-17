# APIs de lote  o desconto de 50% como padrão do setor

> Todos os principais fornecedores enviam uma API de lote asíncrono com um desconto de 50% e uma rotação de ~ 24 horas. OpenAI, Anthropic, Google e a maioria das plataformas de inferência (Fireworks batch tier, Together batch) implementam o mesmo padrão. Batch de pilha com cache rápida e canalizações de noite caem para ~10% do custo sincrono-descoberto. A regra é brutalmente simples: se não é interativo, pertence ao lote. Os canais de geração de conteúdo, classificação de documentos, extração de dados, geração de relatórios, rotulagem em massa, etiquetado de catálogos  qualquer coisa tolerante à latença de 24 horas é dinheiro deixado na mesa até que se transfira para batches. O padrão de produção de 2026 consiste em triagar cada nova carga de trabalho do LLM em três linhas: interativa (sincronizada com o caching), semi-interativa (fila assíncrona com fallback), lote (overnight, input em caching empilhado). Cargas de trabalho que fingem ser interativas mas toleram minutos de atraso desperdiçam mais.

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## Objetivos de aprendizagem

- Cite as três APIs de lote dos fornecedores (OpenAI, Anthropic, Google) e as garantias comuns de desconto de 50% + 24 horas de recuperação.
- Calcular o custo de empilhamento de lote + entrada em cache numa carga de trabalho de classificação durante a noite e comparar com a linha de base sincrónica de entrada não em cache.
- Triagem de uma carga de trabalho em lote interativo/semiinterativo/parcela e justifique a faixa de trabalho.
- Nomear as duas armadilhas: interatividade parcial (o utilizador espera mais rápido do que 24h) e derivação do esquema de saída (formato de arquivo de lote diferem por fornecedor).

## O problema

A sua equipa envia um canal de geração de relatórios noturno. 50.000 documentos, resuma cada um, agrupa os resumos, redige um resumo executivo.

O lote recebe 50% de desconto. Você também permite o cache de prompt no sistema de prompt (compartilhado em todas as chamadas 50k).

O batch é a alavanca mais barata no conjunto de ferramentas de custos de LLM que ninguém tira. A razão é principalmente organizacional: as equipes pensam "em tempo real" quando o SLA é realmente "até de manhã". Esta lição é sobre não deixar 90% da conta na mesa.

## O conceito

### As três APIs de lote

**OpenAI Batch API**A versão JSONL é uma versão de um ficheiro de vídeo que inclui uma lista de solicitações. Promete-se uma volta de 24 horas (geralmente 2-8 horas na prática). 50% de desconto em tokens de entrada e saída. `/v1/batches`As entradas elegíveis para o caché também recebem preços de entradas no caché.

**Anthropic Message Batches**JSONL upload, 24 horas de turnaround, desconto de 50%. Suporta`cache_control` As gravações no cache são explícitas, as leituras acontecem automaticamente dentro do lote.

**Google Vertex AI Batch Prediction**BigQuery ou GCS entrada. Desconto semelhante de 50% para Gemini. Integra com canais Vertex.

### Semântica: assíncrona, não lenta

Batch é "Eu prometo voltar dentro de 24 horas"  não "isso vai levar 24 horas". Tipicamente P50 é de 2 a 6 horas.

### Apilação com cache

Uma resumidação de 50k documentos com o mesmo sistema de 4K-token prompt:

- Sincrono não caché: 50000 × ($input × 4000 + $Output × 200) em taxas completas.
- Cachado sincrono: o sistema de solicitação em cache após a primeira escrita; restantes 49999 obtêm entrada 10 vezes mais barata.
- Batch caché: todas as indicações acima, mais desconto de 50% em leitura e escrita.

A pilha: batch + cache = ~ 10% da conta sincronizada não caçada. Qualquer carga de trabalho que seja executada durante a noite e tenha um prompt do sistema compartilhado deve usar isso.

### Classificação da carga de trabalho

**Interactive** usuário aguarda a resposta. TTFT importa. chamada sincrónica com cache imediato. Não pode batch.

**Semi-interactive** usuário envia uma tarefa, verifica em minutos. fila de sincronização com fallback para sincronizar se o lote não estiver disponível. Pense em indexação RAG de volume moderado.

**Batch** o usuário espera resultados "até manhã" ou "na próxima hora".

Erro comum: classificar tudo como interativo porque o pipeline é produção.

### A armadilha de interatividade parcial

Alguns recursos parecem interativos, mas toleram 5-10 minutos. Exemplo: um relatório de saúde do cliente noturno com botão "refrescir". Clique em "refrescir"; espere 10 minutos é bom. A equipe envia como sincrônico. 50 atualizações simultâneas custam 10 vezes o que custaria o batch-and-deliver-via-email.

A pergunta a ser feita: "O que significa 24 horas para este usuário?" Se a resposta for "não perceberão", faça o batch.

### A armadilha do esquema de saída

Os formatos de arquivos de lote diferem por fornecedor:

- JSONL, uma solicitação por linha.
- Antropico: JSONL, uma mensagem por linha; formato de resposta incorporado.
- Vertex: Tabela BigQuery ou prefixo GCS com TFRecord.

Escrever "um cliente de lote" em todos os provedores significa código de adaptador por fornecedor. Gateways que anunciam lote de multi-provedor (Portkey, LiteLLM em alguns níveis) ainda embalam o formato bruto.

### Números que você deve lembrar

- Desconto de lote entre os prestadores: 50% fixo em entrada + saída.
- SLA de reviravolta: 24 horas garantidas, 2 a 6 horas típicas P50.
- Batch empilhado + entrada em cache: ~10% do custo sincronizado não em cache.
- Regra de triagem da carga de trabalho: se a latência de 24 horas for aceitável, sempre em lote.

```figure
batch-lane-triage
```

## Usá-lo

`code/main.py`Calcula os custos em sincronização, sincronização + cache, lote e lote + cache para uma carga de trabalho de documentos de 50k. Relata economias em $ e em percentagem.

## Envia-o

Esta lição produz`outputs/skill-batch-triager.md`- Dadas as características da carga de trabalho, triagem em interativos/semi/parceiros e estimativas de poupança.

## Exercícios

1. Corra .`code/main.py`. Para um pipeline de 100k-doc com 3K-token system prompt e 500-token output, calcular a economia de pilha completa (batch + cache) vs sincronização linha de base.
2. Escolha três características num produto real que conhece, triando cada uma em interativo/semi/batch.
3. Um usuário reclama que o relatório demorou 3 horas. Foi um erro de triagem de lote ou um interativo legítimo? Escreva o critério de decisão.
4. O seu batch API return SLA é de 24h mas P99 é de 20 horas. Como você comunica isso ao usuário  qual é o comportamento do sistema downstream no caso de borda?
5. Computação de equilíbrio: em que comprimento de prefixo compartilhado o batch + cache torna-se mais barato do que executar durante a noite em sua própria GPU reservada?

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Batch API | "async discount" | 50% off with 24h turnaround |
| JSONL | "batch format" | One JSON request per line; OpenAI/Anthropic standard |
| Message Batches | "Anthropic batch" | Anthropic's batch API product name |
| Batch prediction | "Vertex batch" | Vertex AI's batch API product |
| Turnaround SLA | "24h promise" | Guarantee, not typical; typical is 2-6h |
| Workload triage | "interactivity decision" | Interactive / semi / batch routing decision |
| Output schema | "response format" | Per-provider JSONL layout; not portable |
| Stacked discount | "batch + cache" | ~10% of uncached sync bill when both apply |

## Mais leitura

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) Formatos JSONL e `/v1/batches`- Não, não.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) formato de lote e `cache_control`Interação.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini)Semântica de lote de Gémeos.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
