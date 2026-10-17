# Capstone 11  LLM Observabilidad y Eval Dashboard

> Langfuse se puso en el centro abierto. Arize Phoenix publicó los mapas semiconv de GenAI de 2026. Helicone y Braintrust se duplicaron en la atribución de costos por usuario. La OpenLLMetry de Traceloop se convirtió en la instrumentación de facto del SDK. La forma de producción es ClickHouse para rastros, Postgres para metadatos, Next.js para UI, y un pequeño ejército de trabajos de evaluación (DeepEval, RAGAS, LLM-judge) que recorren rastros muestrados. Construye uno auto-hosted, ingere de al menos cuatro familias de SDK, y demuestre capturar una regresión inyectada en menos de cinco minutos.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:**P11 · P13 · P17 · P18
**Time:** 25 hours

## El problema

Cada equipo de IA que dirige el tráfico de producción en 2026 mantiene un plano de observabilidad junto al modelo. La atribución de costes. Detección de alucinaciones. Monitoreo de la deriva. - Sí, la señal de jailbreak. Tablas de control de SLO. Las alarmas de fuga de información. Las referencias de código abierto  Langfuse, Phoenix, OpenLLMetry  convergieron en las convenciones semánticas de OpenTelemetry GenAI como el esquema de ingesta. Ahora puedes utilizar OpenAI, Anthropic, Google, LangChain, LlamaIndex y vLLM con un SDK y enviar extensiones compatibles.

Se construirá un panel de control auto-hostado que ingere al menos cuatro familias de SDK, ejecuta un pequeño conjunto de trabajos de evaluación sobre rastros muestrados, detecta la deriva y alertas. La barra de medición: dada una regresión inyectada deliberadamente (una solicitud que comienza a producir PII), el panel de control lo captura y dispara una alerta en menos de cinco minutos.

## Concepto

Ingest es OTLP HTTP. El SDK produce generaciones GenAI-semconv: `gen_ai.system`¿ Qué ?`gen_ai.request.model`¿ Qué ?`gen_ai.usage.input_tokens`¿ Qué ?`gen_ai.response.id`¿ Qué ?`llm.prompts`¿ Qué ?`llm.completions`. Se extiende en ClickHouse para análisis columnar; los metadatos (usuarios, sesiones, aplicaciones) se extienden en Postgres.

Los Evals se ejecutan como trabajos de lote sobre las huellas muestrada. DeepEval califica la fidelidad, toxicidad y relevancia de la respuesta. RAGAS califica las métricas de recuperación cuando el rastro lleva el contexto de recuperación. Los jueces de LLM personalizados ejecutan controles específicos de dominio (fuera de PII, respuesta fuera de la política).

La detección de deriva observa las distribuciones de espacio de incorporación a lo largo del tiempo (divergencia de PSI o KL en las incorporaciones rápidas) más las tendencias de puntaje de evaluación. Las alertas se alimentan con Prometheus Alertmanager y luego Slack / PagerDuty.

## Arquitectura

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

## El establo

- Ingesta: SDKs de OpenTelemetry + convenciones semánticas de GenAI; transporte HTTP OTLP
- Colector: OpenTelemetry Colector con procesador de muestreo de cola (para el control de costes)
- Almacenamiento: ClickHouse para períodos, Postgres para metadatos, S3 para archivo de eventos crudos
- Evals: DeepEval, RAGAS 0.2, paquete de evaluadores Arize Phoenix, juez de LLM personalizado
- Drift: PSI / KL en las incorporaciones rápidas conjuntas (transformadores de oraciones) semanalmente
- Alerta: Prometheus Alertmanager -> Slack / PagerDuty
- Interfaz de usuario: Next.js 15 App Router + Recharts + acciones del servidor
- SDKs soportados fuera de la caja: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## Construye el mismo

1. **Collector config.**OpenTelemetry Collector con el receptor HTTP OTLP, un muestreo de cola que guarda el 100% de las huellas de error y el 10% de los éxitos, y exportadores a ClickHouse y S3.

2. **ClickHouse schema.**Cuadro `spans`con columnas que reflejan el genAI semconv: `gen_ai_system`¿ Qué ?`gen_ai_request_model`¿ Qué ?`input_tokens`¿ Qué ?`output_tokens`¿ Qué ?`latency_ms`¿ Qué ?`prompt_hash`¿ Qué ?`trace_id`¿ Qué ?`parent_span_id`, más bolsa JSON para cargas útiles largas. Agregar índices secundarios por user_id y app_id.

3. **SDK coverage test.**Escriba una pequeña aplicación cliente utilizando cada SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) con el instrumento automático OpenLLMetry. Verifique si cada uno produce extensiones canónicas de GenAI que aterrizan en ClickHouse.

4. **Eval jobs.**Un trabajo programado lee las huellas de muestras de los últimos 15 minutos y ejecuta la fidelidad, toxicidad y relevancia de la respuesta de DeepEval.

5. **Custom LLM-judge.**Un juez de fuga de PII: si recibe una respuesta, llama a un guardia LLM para calificar la probabilidad de fuga de PII. Las respuestas de puntaje alto aterrizan en una cola de triaje.

6. **Drift detection.**El trabajo semanal calcula el PSI entre las incorporaciones conjuntas de este mes y el límite de 4 semanas.

7. **Dashboard.**Next.js 15 con páginas: visión general (tempo/segundo, costo/usuario, latencia p95), rastros (busca + cascada), evaluaciones (trend de fidelidad, toxicidad), deriva (PSI a lo largo del tiempo), alertas.

8. **Alerting chain.**El exportador Prometheus lee los agregados de puntajes de eval y los percentiles de latencia; Alertmanager recorre rutas a Slack para advertencias y PagerDuty para infracciones críticas.

9. **Regression probe.**Inyectar un bug: el chatbot evaluado comienza a filtrar SSNs falsos 1% del tiempo. Medir MTTR: desde el bug desplegado a la alerta Slack.

## Usalo

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Envío

`outputs/skill-llm-observability.md`En una aplicación de LLM, el panel ingere sus huellas, ejecuta evaluaciones, alertas sobre la deriva y presenta la desglose de costos/usuarios en Next.js.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## Los ejercicios

1. Añadir instrumentación personalizada para el marco Haystack. Verificar extensiones canónicas de aterrizaje en ClickHouse con fiel `gen_ai.*`Los atributos.

2. Cambiar DeepEval por evaluadores de Phoenix en las mismas pistas.

3. Agudice el detector de deriva: computa el PSI por app-id en lugar de globalmente. Muestre las pistas de deriva por app.

4. Añadir una página de "impacto de usuario": costo por usuario y tasa de fallas por usuario con líneas de referencia.

5. Establezca una política de muestreo de cola que mantenga el 100% de las huellas con toxicidad > 0,5 más una muestra estratificada del 10% del resto.

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## Leer más

- [Langfuse](https://github.com/langfuse/langfuse) la plataforma de observabilidad de referencia de núcleo abierto
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) referencia alternativa con fuerte apoyo a la deriva
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) Familia de SDK de instrumentos automáticos
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) el esquema de ingesta
- [Helicone](https://www.helicone.ai) observabilidad alojada alternativa
- [Braintrust](https://www.braintrust.dev) plataforma alternativa de evaluación-primera
- [ClickHouse documentation](https://clickhouse.com/docs) almacenaje de columnas
- [DeepEval](https://github.com/confident-ai/deepeval) Biblioteca de evaluadores
