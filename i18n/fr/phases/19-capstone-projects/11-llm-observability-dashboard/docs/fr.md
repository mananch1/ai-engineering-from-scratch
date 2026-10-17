# Capstone 11  LLM Observabilité et tableau de bord équivalent

> Langfuse est devenu un centre ouvert. Arize Phoenix a publié les cartographies semconv de la génération de 2026 . Helicone et Braintrust ont tous deux doublé leur coût par utilisateur. L'OpenLLMetry de Traceloop est devenu l'instrumentation de facto du SDK. La forme de production est ClickHouse pour les traces, Postgres pour les métadonnées, Next.js pour UI, et une petite armée de tâches d'évaluation (DeepEval, RAGAS, LLM-judge) qui se déroulent sur les traces échantillonnées. Construisez un auto-hébergé, ingérez au moins quatre familles de SDK et démontrez la capture d'une régression injectable en moins de cinq minutes.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:**P11 · P13 · P17 · P18
**Time:** 25 hours

## Problème

Chaque équipe d'IA qui dirige le trafic de production en 2026 maintient un plan d'observabilité à côté du modèle. L'attribution des coûts. Détection d'hallucinations. Surveillance de dérive. Le signal de jailbreak. Des tableaux de bord SLO. Des alertes de fuite de données. Les références open source  Langfuse, Phoenix, OpenLLMetry  convergent sur les conventions sémantiques OpenTelemetry GenAI comme le schéma d'ingestion. Vous pouvez maintenant utiliser OpenAI, Anthropic, Google, LangChain, LlamaIndex et vLLM avec un SDK et envoyer des spans compatibles.

Vous construirez un tableau de bord auto-hébergé qui absorbe au moins quatre familles de SDK, exécute un petit ensemble de tâches d'évaluation sur les traces échantillonnées, détecte la dérive et les alertes.

## Concept

Ingest est OTLP HTTP. Le SDK produit des étendues GenAI-semconv: `gen_ai.system`- Je suis là .`gen_ai.request.model`- Je suis là .`gen_ai.usage.input_tokens`- Je suis là .`gen_ai.response.id`- Je suis là .`llm.prompts`- Je suis là .`llm.completions`. Les données se déroulent dans ClickHouse pour l'analyse colonnalisée; les métadonnées (utilisateurs, sessions, applications) se déroulent dans Postgres.

Les Evals fonctionnent en tant que tâches de lot sur les traces échantillonnées. DeepEval note la fidélité, la toxicité et la pertinence des réponses. RAGAS note les mesures de récupération lorsque la trace comporte le contexte de récupération. Les juges LLM personnalisés effectuent des vérifications spécifiques au domaine (fuite de PII, réponse hors politique).

La détection de dérive surveille les distributions de l'espace d'intégration au fil du temps (divergence PSI ou KL sur les intégrations rapides) ainsi que les tendances de scores d'évaluation.

## Architecture

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

## La pile

- Ingest: SDK OpenTelemetry + conventions sémantiques GenAI; transport HTTP OTLP
- Collecteur: OpenTelemetry Collector avec processeur de prélèvement de couture (pour le contrôle des coûts)
- Stockage: ClickHouse pour les délais, Postgres pour les métadonnées, S3 pour l'archivage d'événements bruts
- Evals: DeepEval, RAGAS 0.2, Arize Phoenix évaluateur pack, juge spécialisé en LLM
- Drift: PSI / KL sur les intégrations rapides combinées (transformateurs de phrases) hebdomadaire
- Alerte: Prometheus AlertManager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + actions serveur
- Les SDK pris en charge hors boîte: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## Faites-le

1. **Collector config.**OpenTelemetry Collector avec le récepteur HTTP OTLP, un échantillon de queue qui conserve 100% des traces d'erreur et 10% des succès, et des exportateurs vers ClickHouse et S3.

2. **ClickHouse schema.**Tableau `spans`avec des colonnes reflétant le génome de l' ARM: `gen_ai_system`- Je suis là .`gen_ai_request_model`- Je suis là .`input_tokens`- Je suis là .`output_tokens`- Je suis là .`latency_ms`- Je suis là .`prompt_hash`- Je suis là .`trace_id`- Je suis là .`parent_span_id`, plus JSON sac pour les charges utiles longues. Ajouter des indices secondaires par user_id et app_id.

3. **SDK coverage test.**Écrivez une petite application client à l'aide de chaque SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) avec OpenLLMetry auto-instrument. Vérifiez que chaque produit génAI canonique étendue qui atterrissent dans ClickHouse.

4. **Eval jobs.**Un travail programmé lit les traces échantillonnées de 15 minutes et exécute la fidélité, la toxicité et la pertinence de la réponse de DeepEval.

5. **Custom LLM-judge.**Un juge de fuite d'informations: si vous avez une réponse, appelez un garde LLM pour évaluer la probabilité de fuite d'informations.

6. **Drift detection.**Le travail hebdomadaire compute le PSI entre les intégrations rapides de cette semaine et la ligne de base de 4 semaines.

7. **Dashboard.**Next.js 15 avec pages: vue d'ensemble (spans/seconde, coût/utilisateur, latence p95), traces (recherche + cascade), évaluations (trend de fidélité, toxicité), dérive (PSI au fil du temps), alertes.

8. **Alerting chain.**L'exportateur Prometheus lit les agrégats de scores d'évaluation et les percentiles de latence; Alertmanager accède à Slack pour les avertissements et PagerDuty pour les violations critiques.

9. **Regression probe.**Injecter un bug: le chatbot évalué commence à fuir de faux SSN 1% du temps. Mesurer MTTR: de bug déployé à l'alerte Slack.

## Utilisez-le

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## La faire partir

`outputs/skill-llm-observability.md`Le tableau de bord ingère ses traces, exécute des évaluations, des alertes sur la dérive et affiche la répartition des coûts/utilisateurs dans Next.js.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## Exercices

1. Ajoutez des instruments personnalisés pour le framework Haystack. Vérifiez les étendues canoniques de la terre dans ClickHouse avec fidèles `gen_ai.*`les attributs.

2. Échangez DeepEval pour les évaluateurs Phoenix sur les mêmes traces.

3. Aiguiser le détecteur de dérive: calculer le PSI par app-id plutôt que globalement.

4. Ajouter une page "impact sur l'utilisateur": coût par utilisateur et taux d'échec par utilisateur avec des étincelles.

5. Élaborer une politique de prélèvement de l'échantillon de la queue qui conserve 100% des traces avec une toxicité supérieure à 0,5 et un échantillon stratifié de 10% du reste.

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## Pour en savoir plus

- [Langfuse](https://github.com/langfuse/langfuse) la plateforme d'observabilité de référence à noyau ouvert
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) référence alternative avec un fort support de dérive
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) Famille de KDD d'instrumentation automatique
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) le schéma d'ingestion
- [Helicone](https://www.helicone.ai) Observabilité d'hébergement alternatif
- [Braintrust](https://www.braintrust.dev) plateforme alternative d'évaluation première
- [ClickHouse documentation](https://clickhouse.com/docs) magasin de couches de couches
- [DeepEval](https://github.com/confident-ai/deepeval) bibliothèque d'évaluation
