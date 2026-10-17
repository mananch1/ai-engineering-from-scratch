# Cachagem rápida e economia semântica

> **Pricing snapshot dated 2026-04.**As reivindicações numéricas abaixo refletem os cartões de taxas de fornecedores capturados na publicação desta lição; verifique contra os documentos vinculados antes de citarem-nos para baixo.

> O caching ocorre em duas camadas. L2 (nível de fornecedor) prompt/prefix caching reutiliza atenção KV para prefixos repetidos  Os documentos de caching de prompt da Anthropic anunciam até 90% de redução de custos e 85% de redução de latência em longos pedidos; para Claude 3.5 Sonnet lecturas de cache são $0.30/M vs $3,00/M fresco com um TTL de 5 minutos e um prêmio de escrita de 2x para a opção TTL de 1 hora (docs.anthropic.com, 2026-04). O caching de prompt do OpenAI aplica-se automaticamente para os tokens de prompt ≥1024 e os preços de entrada em caché com aproximadamente um desconto de 90% versus fresco (platform.openai.com, 2026-04); a taxa de caché exata por modelo depende do cartão de taxa ao vivo. L1 (nivel de aplicação) cache semântico salta o LLM inteiramente em incorporar hites de semelhança. Produtor "95% de precisão" refere-se à correção, não à taxa de impacto  as taxas de impacto da produção relatadas variam de 10% (chat aberto) até 70% (FAQ estruturado); nenhum dos provedores publica uma linha de base oficial, por isso trate-as como telemetria comunitária em vez de garantias. As armadilhas de produção: a paralelalização mata o cache (N solicitações paralelas emitidas antes da primeira escrita no cache podem inflar o gasto várias vezes), e o conteúdo dinâmico dentro do prefixo impede que o cache atinja inteiramente. ProjectDiscovery relatou que a taxa de hits passou de 7% para 74% (2025-11) movendo texto dinâmico do prefixo cacheável.

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Objetivos de aprendizagem

- Distinguir o caching de prompts/prefixos L2 (reutilização de KV no fornecedor) do caching semântico L1 (bypass de LLM em prompts semelhantes).
- Explica o Anthropic `cache_control`Marcação explícita e as duas opções TTL (5-min vs. 1 hora) com os seus multiplicadores de preço.
- Calcule as economias mensais esperadas, dada a taxa de acidente, a combinação de resposta/prompto e os preços dos tokens.
- Nomear o padrão anti-paralelalização que infla contas por 5-10x e o padrão anti-conteúdo dinâmico que desmorona taxa de hits.

## O problema

Você adiciona o cache de instruções ao seu serviço RAG. A conta permanece plana. Você mede a taxa de hits; é 7%. As instruções parecem estáticas, mas não são  o prompt do sistema inclui a data atual formatada para o minuto, um ID de solicitação e uma reordem de exemplo aleatório para a diversidade. Cada solicitação escreve uma nova entrada de cache, lê zero.

Separadamente, seu agente executa dez chamadas paralelas de ferramentas por pergunta de usuário. Todas as dez chegam ao provedor antes de a primeira escrita em cache ser concluída. Dez escreve, zero lê. Sua conta é 5-10 vezes o que "com cache" deveria custar.

O caching é um protocolo, não uma bandeira.

## O conceito

### L2  Cachagem de antecedentes/prefixos do fornecedor

O provedor armazena o KV de atenção para um prefixo cacheável e o reutiliza no próximo pedido que corresponda ao prefixo.

**Anthropic (Claude 3.5 / 3.7 / 4 series)**: explícito `cache_control`TTL: 5 minutos (costas de escrita 1,25x base) ou 1 hora (costas de escrita 2x base).$0.30/M on Claude 3.5 Sonnet vs $3,00/M fresco  10 vezes mais barato (docs.anthropic.com, a partir de 2026-04). As tarifas diferem por modelo (Opus/Haiku publicado separadamente); sempre verifique a página de preços ao vivo.

**OpenAI**O sistema de cache automático para as instruções ≥1024 tokens (platform.openai.com, 2026-04). Não há bandeira explícita. A entrada em cache é aproximadamente 10 vezes mais barata do que a nova nas atuais cartões de taxa gpt-4o/gpt-5. Nem os documentos nem as notas de lançamento publicam uma linha de base oficial de taxa de sucesso; relatórios comunitários agrupam cerca de 3060% com um projeto de prompt cuidadoso. Monitor `usage.cached_tokens`Para medir o seu.

**Google (Gemini)**: conteúdo em cache através de API explícito; 1M-token conteúdo significa que o cache paga ainda mais.

**Self-hosted (vLLM, SGLang)**: Fase 17 · 06 abrange o mesmo padrão em seu próprio cálculo.

### L1  Caching semântico a nível de aplicativo

Antes de ligar para o LLM, hash o prompt, embude-o e procure uma solicitação similar armazenada em cache (similaridade de cozinho acima do limiar, normalmente 0,95+).

Caso aberto: Redis Vector Similarity, GPTCache, Qdrant. Comércio: Portkey Cache, Helicone Cache.

As alegações de precisão do fornecedor se referem à frequência com que a resposta de cache retornada foi semanticamente apropriada, não à frequência com que você bate.

- Chat aberto: 10-15%.
- FAQ estruturada / apoio: 40-70%.
- Questões de código: 20-30% (variantes pequenas matam os hits).
- Agentes de voz que repetem as instruções: 50-80% (conjunto fixo de normalização de voz).

### O padrão anti-paralelalização

O seu agente faz 10 chamadas para ferramentas em paralelo. Todas as 10 têm o mesmo prompt do sistema de tokens 4K. As gravações do cache antropico são por pedido; a primeira gravação do cache completa cerca de 300 ms após o provedor ver o prompt. As solicitações de 2 a 10 chegam na mesma janela de milissegundos e cada uma vê o cache perdido. Você paga 10 prêmios de escrita, 0 descontos de leitura.

Correção: lote com sequencial-first  fazer pedido 1 sozinho, em seguida, disparar 2-10 uma vez que o caché de 1 tem povoado. Adiciona 300 ms para a primeira chamada de ferramenta; salva 5-10x a conta.

### O antipatrão de conteúdo dinâmico

O teu sistema de resposta parece:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Cada pedido é único, cada pedido é escrito, zero hits.

Correção: mover tudo realmente estático para o prefixo cacheable; adicionar conteúdo dinâmico após o limite do cache:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

O ProjectDiscovery passou de 7% para 74% da taxa de cache desta forma e publicou a anatomia.

### Batch de pilha + cache para cargas de trabalho noturnas

As APIs de lote (Fase 17 · 15) oferecem 50% de desconto em turnaround de 24 horas. A entrada em cache no topo lhe dá ~ 10x acima disso. As cargas de trabalho de classificação, rotulagem e geração de relatórios durante a noite podem cair para ~ 10% do custo sincrônico-unchedded por empilhamento.

### Números que você deve lembrar

Os pontos de preços são capturados 2026-04 dos documentos dos fornecedores vinculados e são verificados a cada poucos meses antes de dependerem deles.

- Leitura em cache antropico: $0,30/M no Claude 3.5 Sonnet, aproximadamente 10 vezes mais barato do que a entrada fresca (docs.anthropic.com).
- Prêmio de escrita de cache antropópica: 1,25x (5-min TTL) ou 2x (1-hora TTL).
- O caché automático OpenAI: aplica-se a pedidos ≥ 1024 tokens; entrada em caché com preço de aproximadamente 10% da entrada nova em cartões de taxa corrente (platform.openai.com).
- Taxa de hits de cache semântico (relatado pela comunidade): ~ 10% aberto chat; até ~ 70% FAQ estruturada. Não uma linha de base documentada pelo fornecedor.
- ProjectDiscovery: 7% → 74% taxa de acidentes movendo dinâmica do prefixo (blog do projeto, 2025-11).
- Anti-patrão de paralelalização: relatórios típicos de inflação de contas 510x quando N solicitações paralelas perdem a primeira escrita cache.

```figure
semantic-cache-hit
```

## Usá-lo

`code/main.py`Simula o cache L1 + L2 em cargas de trabalho mistas.

## Envia-o

Esta lição produz`outputs/skill-cache-auditor.md`- Tendo em conta o modelo e o tráfego, verifica a cachéabilidade e recomenda a reestruturação.

## Exercícios

1. Corra .`code/main.py`- O que é que a lei muda?
2. O seu sistema de solicitação tem uma data.
3. Calcule o equilíbrio de 1 hora de TTL (2x escrever) vs 5 minutos de TTL (1.25x escrever) dada a taxa de chegada do pedido.
4. O cache semântico no limiar de 0,95 atinge 20%. No 0,85 atinge 50% mas você vê respostas armazenadas em cache incorretas. Escolha o limiar certo e justifique.
5. Você faz 10 subqueries paralelas por pergunta de usuário. Reescrever para a facilidade de cache sem adicionar latência de ponta a ponta.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | Provider stores KV for repeated prefix |
| `cache_control` | "Anthropic cache marker" | Explicit attribute marking cacheable blocks |
| Cache write premium | "write tax" | Extra cost for first miss-to-cache (1.25x or 2x) |
| L1 semantic cache | "embedding cache" | App-level hash-and-embed before calling LLM |
| GPTCache | "LLM caching lib" | Popular OSS L1 cache library |
| Cache hit rate | "hits / total" | Fraction of requests served from cache |
| Parallelization anti-pattern | "the N-write trap" | N parallel requests miss cache N times |
| Dynamic content trap | "the time-in-prompt trap" | Dynamic bytes in prefix kill hit rate |
| RadixAttention | "intra-replica cache" | SGLang's prefix-cache implementation |

## Mais leitura

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) oficial `cache_control`semântica e TTL.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) comportamento de cache automático e elegibilidade.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
