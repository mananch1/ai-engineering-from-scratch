# Capstone 11  LLM Observabilidade e Eval Dashboard

> O Langfuse foi aberto. Arize Phoenix publicou os mapas semconv da GenAI de 2026. O Helicone e o Braintrust duplicaram a atribuição de custos por utilizador. O OpenLLMetry do Traceloop tornou-se a instrumentação de facto do SDK. A forma de produção é ClickHouse para traços, Postgres para metadados, Next.js para UI e um pequeno exército de trabalhos de avaliação (DeepEval, RAGAS, LLM-judge) que correm sobre traços de amostra. Construir um auto-hosted, ingerir de pelo menos quatro famílias SDK, e demonstrar capturar uma regressão injetada em menos de cinco minutos.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:**P11 · P13 · P17 · P18
**Time:** 25 hours

## Problemas

Cada equipe de IA que executa o tráfego de produção em 2026 mantém um plano de observabilidade ao lado do modelo. A atribuição dos custos. Detecção de alucinações. Monitoramento de deriva. - O sinal de jailbreak. Painéis de controlo SLO. Alertas de vazamento de informação. As referências de código aberto  Langfuse, Phoenix, OpenLLMetry  convergiram nas convenções semânticas do OpenTelemetry GenAI como o esquema de ingestão. Agora você pode instrumentar OpenAI, Anthropic, Google, LangChain, LlamaIndex e vLLM com um SDK e enviar extensões compatíveis.

Você vai construir um painel de controle auto-hospedado que ingere de pelo menos quatro famílias de SDK, executa um pequeno conjunto de trabalhos de avaliação sobre traços amostragados, detecta a deriva e alertas.

## Conceptos

Ingest é OTLP HTTP. O SDK produz extensões GenAI-semconv: `gen_ai.system`- Não .`gen_ai.request.model`- Não .`gen_ai.usage.input_tokens`- Não .`gen_ai.response.id`- Não .`llm.prompts`- Não .`llm.completions`. Expira-se para a análise columnar na ClickHouse; os metadados (usuários, sessões, aplicativos) para a Postgres.

Os Evals executam como trabalhos em lote sobre os vestígios de amostras. DeepEval marca fidelidade, toxicidade e relevância da resposta. RAGAS marca métricas de recuperação quando o vestígio carrega contexto de recuperação. Jurados LLM personalizados executam verificações específicas de domínio (vaga de PII, resposta fora da política).

A detecção de deriva observa distribuições de espaço de inserção ao longo do tempo (divergência PSI ou KL em inserções rápidas) além de tendências de avaliação. Alertas alimentam o Prometheus Alertmanager e, em seguida, Slack / PagerDuty. A interface é Next.js 15 com Recharts.

## Arquitetura

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Estaca

- Ingestão: SDKs OpenTelemetry + convenções semânticas GenAI; transporte HTTP OTLP
- Colector: OpenTelemetry Collector com processador de amostragem de cauda (para controlo de custos)
- Armazenamento: ClickHouse para períodos, Postgres para metadados, S3 para arquivo de eventos brutos
- Evals: DeepEval, RAGAS 0.2, Arize Phoenix evaluator pack, jurado LLM personalizado
- Drift: PSI / KL em embutidos conjuntos de prompt (transformadores de sentença) semanalmente
- Alerta: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + ações do servidor
- SDKs suportados fora da caixa: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## Construí-lo

1. **Collector config.**OpenTelemetry Collector com o receptor HTTP OTLP, um monstrador de cauda que mantém 100% dos rastos errados e 10% dos sucessos, e exportadores para ClickHouse e S3.

2. **ClickHouse schema.**Tabela `spans`com colunas que refletem a GenAI semconv: `gen_ai_system`- Não .`gen_ai_request_model`- Não .`input_tokens`- Não .`output_tokens`- Não .`latency_ms`- Não .`prompt_hash`- Não .`trace_id`- Não .`parent_span_id`, mais JSON saco para cargas úteis longas. Adicionar índices secundários por user_id e app_id.

3. **SDK coverage test.**Escreva um pequeno aplicativo cliente usando cada SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) com o instrumento automático OpenLLMetry. Verifique se cada um produz extensões canônicas de GenAI que aterram na ClickHouse.

4. **Eval jobs.**Um trabalho programado lê traços de amostras de 15 minutos e executa a fidelidade, toxicidade e relevância de resposta do DeepEval.

5. **Custom LLM-judge.**Um juiz de vazamento de PII: se tiver uma resposta, chama um guarda LLM para avaliar a probabilidade de vazamento de PII.

6. **Drift detection.**O trabalho semanal calcula o PSI entre as incorporações de prompt desta semana e a linha de base de 4 semanas.

7. **Dashboard.**Next.js 15 com páginas: visão geral (spans/sec, custo/usuário, latência p95), rastreamento (busca + cascata), avaliações (trend de fidelidade, toxicidade), deriva (PSI ao longo do tempo), alertas.

8. **Alerting chain.**O exportador Prometheus lê os agregados de pontuação de eval e os percentilhas de latência; Alertmanager encaminha-se para Slack para alertas e PagerDuty para violações críticas.

9. **Regression probe.**Injecte um bug: o chatbot avaliado começa a vazamento de SSNs falsos 1% do tempo.

## Usá-lo

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Envia-o

`outputs/skill-llm-observability.md`Em uma aplicação de LLM, o painel de controle ingere seus vestígios, executa avaliações, alertas sobre deriva e apresenta a divisão custo/usuário no Next.js.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## Exercícios

1. Adicionar instrumentação personalizada para o framework Haystack. Verifique canônicos de extensão de pouso em ClickHouse com fiéis `gen_ai.*`- Os atributos.

2. Troca DeepEval para avaliadores Phoenix nos mesmos traços.

3. Aperte o detector de deriva: compute o PSI por app-id em vez de globalmente. Mostre traços de deriva por app.

4. Adicionar uma página "impacto do utilizador": custo por utilizador e taxa de falhas por utilizador com linhas de referência.

5. Estabelecer uma política de amostragem de cauda que mantenha 100% das pistas com toxicidade > 0,5 e uma amostra estratificada de 10% do resto.

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## Mais leitura

- [Langfuse](https://github.com/langfuse/langfuse) a plataforma de observabilidade de referência de núcleo aberto
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) referência alternativa com forte apoio à deriva
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) Família de SDK de auto-instrumentação
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) o esquema de ingestão
- [Helicone](https://www.helicone.ai) observabilidade hospedada alternada
- [Braintrust](https://www.braintrust.dev) plataforma alternativa de avaliação-primeira
- [ClickHouse documentation](https://clickhouse.com/docs) loja de comprimento columnar
- [DeepEval](https://github.com/confident-ai/deepeval) Biblioteca de avaliadores
