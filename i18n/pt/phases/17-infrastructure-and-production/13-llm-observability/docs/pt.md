# Seleção de pilhas de observabilidade do MLL

> O mercado de observabilidade de 2026 divide-se em duas categorias. As plataformas de desenvolvimento (LangSmith, Langfuse, Comet Opik) combinam o monitoramento com avaliações, gestão de prompt, repetições de sessões. As ferramentas de gateway/instrumentamento (Helicone, SigNoz, OpenLLMetry, Phoenix) focam-se na telemetria. Langfuse é um núcleo licenciado pelo MIT com forte balanço OSS (50K eventos / mês nuvem gratuita). Phoenix é OpenTelemetry-native sob Licença Elastica 2.0  excelente para visualização drift/RAG, não um backend de produção persistente. O Arize AX usa uma integração Iceberg/Parquet de cópia zero, afirmando que é 100 vezes mais barato do que a observabilidade monolitica. LangSmith lidera para LangChain/LangGraph, $39/usuário/mês, auto-host em Enterprise apenas. O Helicone é baseado em proxy com configuração de 15-30 minutos, 100K de req/mo livre, mas menos profundidade em vestígios de agentes. Padrão de produção comum: Gateway (Helicone/Portkey) + plataforma de avaliação (Phoenix/TruLens) colada pela OpenTelemetry.

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Objetivos de aprendizagem

- Distinguir as plataformas de desenvolvimento (agrupadas: evals + prompts + sessões) das ferramentas de gateway/telemetria (só traços + métricas).
- Mapear seis principais ferramentas (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) para suas licenças, preços e casos de uso do sweet-spot.
- Explique o padrão de cola OpenTelemetry que permite combinar uma ferramenta de gateway com uma plataforma de avaliação separada.
- Nomear o diferenciador de custos de 2026 (abordagem de cópia zero do Arize AX vs ingestão monolitica) e indicar o multiplicador de 100x aproximado.

## O problema

Você enviou um recurso LLM. Funciona. Você não tem visibilidade em falhas rápidas, loop de ferramentas, regressões de latência, picos de custo ou taxa de sucesso de caché rápido. Você Google "observabilidade LLM" e obtém oito ferramentas todas alegando que resolvem o mesmo problema em três pontos de preço diferentes.

Eles não resolvem o mesmo problema. LangSmith responde "por que essa execução de LangGraph falhou?" Phoenix responde "meu pipeline RAG está a driftar?" Helicone responde "qual aplicativo está a queimar tokens?" Langfuse responde "eu posso auto-hostar a coisa toda?" Ferramentas diferentes, públicos diferentes.

A selecção envolve quatro eixos: pilha (LangChain? SDK bruto? multi-vendor?), tolerância à licença (apenas MIT? Elastic OK? multa comercial?), orçamento (nível gratuito? $100/mo? $1000/mo?), e auto-host (deve ser bom para ter? nunca?).

## O conceito

### Duas categorias

**Development platforms**O que é que você tem de fazer é fazer experiências, ver qual prompt funcionou, regressão de dados, um novo prompt contra antigos vencedores.

**Gateway/telemetry tools**Infraestrutura de inferência de chamadas  prompt, resposta, tokens, latência, modelo, custo. Helicone, SigNoz, OpenLLMetry, Phoenix. Minimalista. Pode ser combinado com uma ferramenta de avaliação separada através de OpenTelemetry.

### Balanço de Langfuse  OSS

- Core Apache / MIT licenciado; auto-host via Docker.
- Número gratuito em nuvem: 50 mil eventos por mês.
- Evals, gestão rápida, rastreamento, conjuntos de dados, cobertura razoável de todas as quatro características da plataforma de desenvolvimento.
- O ponto mais interessante: você quer recursos da classe LangSmith, mas deve ser auto-host ou ficar com a licença OSS.

### Phoenix (Arize)  Telemetria-primeira, OpenTelemetry-nativa

- Licença Elastica 2.0; auto-host trivial.
- Excelente em RAG e visualização de deriva.
- Não concebido como backend de produção persistente  observabilidade primordial no tempo de desenvolvimento.
- Ponto de interesse: desenvolvimento de tubos RAG, depuração de deriva, pares com uma porta de entrada separada para produção.

### Arize AX  a escala de jogo

- Integração de dados de zero cópia através de Iceberg/Parquet.
- A matemática: você armazena vestígios no seu próprio Parquet no S3; Arize lê diretamente.
- Ponto de espera: > 10 milhões de traços por dia, lago de dados existente, quer painéis de controle específicos para LLM sem preços Datadog.

### LangSmith  LangChain/LangGraph primeiro

- Commercial, 39 dólares por mês, auto-host só na Enterprise.
- É o melhor de classe para as pilhas LangChain e LangGraph.
- O ponto doce: equipa comprometida com a LangChain, dispostos a pagar.

### Helicone  baseado em proxy

- 15-30 minutos de configuração trocando o seu `OPENAI_API_BASE`Para o proxy do Helicone.
- Licenciado pelo MIT; 100 mil reais por mês gratuitos, pagos 20 dólares por mês.
- Inclui falhas, cache, limites de taxa  também atua como um gateway.
- Menos profundidade em agentes / traços de vários passos.
- Sweet spot: início rápido, aplicativo de pilha única, precisa de gateway + observabilidade em um.

### Opik (Comet)  Plataforma de desenvolvimento OSS

- Apache 2.0, totalmente sob controle.
- Características semelhantes ao Langfuse com patrimônio cometa.
- Equipes de ML já no Comet, querem observabilidade LLM no mesmo painel.

### SigNoz  OpenTelemetry-first APM completo

- Apache 2.0. lida com APM geral mais LLM através da OpenTelemetry.
- O ponto ideal: observabilidade unificada entre os serviços e as chamadas de LLM.

### A cola: OpenTelemetry + convenções semânticas da GenAI

A OpenTelemetry publicou convenções semânticas da GenAI no final de 2025 (`gen_ai.system`- Não .`gen_ai.request.model`- Não .`gen_ai.usage.input_tokens`O modelo de produção emergente:

1. Emitir OTel com convenções da GenAI de cada chamada de LLM.
2. Rota para o portal (Helicone / Portkey) para o dia-a-dia.
3. Dual-ship-to-evaluation (Phoenix/Langfuse) para regressões.
4. Arquivo no lago de dados (Iceberg) para análise a longo prazo através do Arize AX ou DuckDB.

### A armadilha: instrumentação na camada errada

Instrumentar dentro de sua estrutura de agente (por exemplo, adicionando traços LangSmith) o acopla a essa estrutura. Instrumentar na camada HTTP/OpenAI-SDK (através do OpenLLMetry ou do seu gateway) é portátil.

### Amostragem  não pode ficar tudo

A retenção de dados completos custa mais que as chamadas de LLM. Amostra por regras: 100% de erros, 100% de alto custo, 5% de sucesso.

### Números que você deve lembrar

- Nuvem livre Langfuse: 50 mil eventos por mês.
- LangSmith: 39 dólares por utilizador por mês.
- Helicone livre: 100 mil reais por mês.
- Arize AX afirmação: ~ 100 vezes mais barato do que monolitico em escala.
- Convenções da OpenTelemetry GenAI: 2025 transporte marítimo, 2026 amplamente adotada.

```figure
i4-otel-glue
```

## Usá-lo

`code/main.py`Simula um dia de rastreamento de 1M em todas as estratégias de retenção (100% ingestão, amostragem, amostragem + erros).

## Envia-o

Esta lição produz`outputs/skill-observability-stack.md`. Dada a pilha, a escala, o orçamento, a posição da licença, escolhe a ferramenta ((s).

## Exercícios

1. A sua equipa na LangChain quer a observabilidade auto-hostada do OSS.
2. Com 5M traços por dia com Datadog citar $ 150K por mês, calcular o equilíbrio para Arize AX.
3. Desenhar um atributo OpenTelemetry GenAI definido as diretrizes da sua organização devem ser obrigatórias em cada chamada de LLM.
4. Discutir se Phoenix sozinho é suficiente para a produção.
5. O helicóptero tem 20 ms de carga por proxy.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OpenLLMetry | "OTel for LLMs" | Open-source OpenTelemetry instrumentation for LLMs |
| GenAI conventions | "OTel attributes" | Standard OTel attribute names for LLM calls |
| LangSmith | "LangChain observability" | Commercial platform bundled with LangChain ecosystem |
| Langfuse | "OSS LangSmith" | MIT OSS with similar feature set |
| Phoenix | "Arize dev tool" | OpenTelemetry-native dev/eval platform |
| Arize AX | "scale observability" | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | "proxy observability" | HTTP proxy collecting LLM telemetry + gateway features |
| Opik | "Comet LLM" | Apache 2.0 OSS dev platform from Comet |
| Session replay | "trace rerun" | Replay a full agent session with tool calls |
| Eval | "offline test" | Running candidate model/prompt over labeled dataset |

## Mais leitura

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
