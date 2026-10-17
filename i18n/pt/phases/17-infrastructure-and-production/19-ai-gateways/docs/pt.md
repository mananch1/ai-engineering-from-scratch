# Portes de inteligência artificial  LiteLLM, Portkey, Kong AI Gateway, Bifrost

> Um gateway fica entre os seus aplicativos e os provedores de modelos. As características principais são roteamento do provedor, retrocesso, retestes, limitação de taxa, referências secretas, observabilidade, guardrails. Divisão do mercado em 2026: **LiteLLM**é MIT OSS com mais de 100 provedores, compatível com OpenAI, mas quebra em torno de ~ 2000 RPS (8 GB de memória, falhas em cascata em benchmarks publicados); melhor para Python, <500 RPS, desenvolvimento / prototipagem. **Portkey**é posicionado no plano de controle (guardrails, redação de PII, detecção de jailbreak, trilhas de auditoria), foi Apache 2.0 de código aberto Março de 2026, 20-40 ms de latência overhead, $49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $Preço de 100/modelo/mês (máximo 5 no nível Plus); adequado para empresas se já estiver no Kong. **Bifrost**(Maxim AI)  Retemps automáticos com back-off configurável, fallback para Anthropic em OpenAI 429. **Cloudflare / Vercel AI Gateways** gerenciado, zero-ops, retry básico. Residência de dados impulsiona a decisão de auto-host; Portkey e Kong sentam no meio com OSS + opcional gerenciado.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## Objetivos de aprendizagem

- Enumerar as seis características principais do gateway (routing, fallback, retries, limites de taxa, segredos, observabilidade, barris).
- Mapear quatro gateways 2026 (LiteLLM, Portkey, Kong AI, Bifrost) para escalar limites e casos de uso.
- Cite o índice de referência Kong (228% contra Portkey, 859% contra LiteLLM) e explique por que é importante para > 500 RPS.
- Escolha auto-hosted versus gerenciado dada a residência de dados e orçamento de operações.

## O problema

O seu produto chama OpenAI, Anthropic e um Llama auto-hosted. Cada provedor tem um SDK diferente, modelo de erro, limite de taxa e esquema de auth. Você quer falha (se OpenAI 429, tente Anthropic), uma única loja de credenciais, observabilidade unificada e limites de taxa por inquilino.

Reinventando isso na camada de aplicativos, cada serviço é associado a cada provedor. Uma camada de gateway consolida-o em um processo com uma API (normalmente compatível com o OpenAI) que é divulgada aos provedores.

## O conceito

### Seis características principais

1. **Provider routing** OpenAI, Anthropic, Gemini, auto-hosted, etc. por trás de uma API.
2. **Fallback** em 429, 5xx, ou falha de qualidade, tente novamente em outro lugar.
3. **Retries**- O back-off exponencial, tentativas limitadas.
4. **Rate limits**- Por inquilino, por chave, por modelo.
5. **Secret references** extrair as credenciais do cofre no tempo de execução (nunca no aplicativo).
6. **Observability** ATRITUDOS OTEL + GenAI (Fase 17 · 13) + ATRITUÇÃO DE COSTOS.
7. **Guardrails** Reduzir PII, detectar jailbreak, filtros de tópicos permitidos.

### LiteLLM  MIT OSS, Python

- 100+ provedores, compatíveis com o OpenAI, configuração do roteador, retrocesso, observabilidade básica.
- Desfez cerca de 2000 RPS no ponto de referência de Kong; 8 GB de memória, falhas em cascata sob carga sustentada.
- Melhor ajuste: aplicativo Python, < 500 RPS, gateways de desenvolvimento/estagem, roteamento experimental.
- Custo: $0 para OSS; nível livre de nuvem existe.

### Portkey  posicionamento do plano de controlo

- Apache 2.0 OSS a partir de março de 2026.
- 20-40 ms por pedido de atraso.
- $49/mo para nível de produção com retenção + SLA.
- O melhor ajuste: indústrias regulamentadas que necessitam de barris de segurança + observabilidade agrupada.

### Kong AI Gateway  o jogo de escala

- Construído no Kong Gateway (produto de gateway API maduro, lua+OpenResty).
- O próprio índice de referência da Kong no equivalente a 12 CPUs: 228% mais rápido que o Portkey, 859% mais rápido que o LiteLLM.
- Preço: 100 dólares por modelo por mês, máximo 5 no nível Plus.
- Melhor ajuste: já em Kong; > 1000 RPS; disposto a licenciar.

### Bifrost (Maxim AI)

- Reintentos automáticos com back-off configurável.
- O Fallback para Anthropic no OpenAI 429 é uma receita canônica.
- Novos participantes, comerciais.

### Cloudflare AI Gateway / Vercel AI Gateway

- Re-ataque básico e observabilidade.
- Melhor ajuste: aplicativos JavaScript que servem Edge no Cloudflare/Vercel.
- Limitado em comparação com Kong/Portkey em barris e limites de taxa.

### Auto-hosted versus gerenciado

Residência de dados é a função forçante. saúde e finanças auto-host padrão (LiteLLM ou Portkey OSS ou Kong). produtos de consumo gerenciados por padrão (Cloudflare AI Gateway) ou de nível médio (Portkey gerenciado). híbrido: auto-hosted para inquilino regulamentado, gerenciado para outros.

### Orçamento de latência

- LiteLLM: 5-15 ms de carga normal.
- 20-40 ms em cima.
- 3 a 8 ms em cima.
- Cloudflare/Vercel: 1-3 ms de custo geral (vantagem de ponta).

A latência do gateway adiciona-se diretamente ao TTFT. Para TTFT P99 < 100 ms SLA, Kong ou Cloudflare. Para P99 < 500 ms, qualquer.

### Materia semântica de limite de taxa

O token-bucket simples funciona até uma escala moderada. Multi-tenant requer janela deslizante + allocação de explosão + tiering por tenente. LiteLLM navega token-bucket; Kong navega janela deslizante; Portkey navega em camadas.

### Gateway + observabilidade + roteamento compõe

Fase 17 · 13 (observabilidade) + 16 (routing modelo) + 19 (gateways) são a mesma camada na produção. Escolha uma ferramenta que cobre todas as três ou cableá-las cuidadosamente: a maioria das implementações de 2026 combina Helicone (observabilidade) ou Portkey (garda) com Kong (escala) para papéis divididos.

### Números que você deve lembrar

- LiteLLM: quebra em ~ 2000 RPS, memória de 8 GB.
- Portkey: 20-40 ms de custo; Apache 2.0 desde março de 2026.
- Kong: 228% mais rápido que Portkey, 859% mais rápido que LiteLLM.
- Preço Kong: 100 dólares por modelo por mês, 5 max no nível Plus.
- Cloudflare/Vercel: 1-3 ms de carga no limite.

```figure
mx-gateway-fallback
```

## Usá-lo

`code/main.py`Simula o roteamento de gateway com fallback em 3 provedores sob injeção 429/5xx. Relata latência, taxa de retiro e taxa de impacto de fallback.

## Envia-o

Esta lição produz`outputs/skill-gateway-picker.md`Dada a escala, a postura das operações, a conformidade, o orçamento de latência, escolhe um portal.

## Exercícios

1. Corra .`code/main.py`Configurar fallback de OpenAI→Anthropic→auto-hosted. Qual é a taxa de impacto esperada com taxa de erro de fornecedor de 5%?
2. O seu SLA é TTFT P99 < 200 ms em uma linha de base de 300 ms. Quais gateways mantêm dentro do orçamento?
3. Um cliente de saúde precisa de auto-hosting + redação de PII + auditoria.
4. Comparar LiteLLM vs Kong: a que limite máximo de RPS deve migrar uma equipa?
5. Desenhar uma política de limite de taxas para um SaaS multi-arrendatário: nível gratuito, nível de teste, nível pago. Token-bucket ou janela deslizante?

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | Process sitting between apps and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated log" | Immutable record of every LLM call |
| Token-bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding-window | "precise rate limit" | Time-windowed rate limiter; better fairness |

## Mais leitura

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
