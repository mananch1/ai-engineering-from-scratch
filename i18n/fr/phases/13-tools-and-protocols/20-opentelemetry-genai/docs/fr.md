# OpenTelemetry GenAI  Traceur d'outils de suivi des appels de bout en bout

> Un agent appelle cinq outils, trois serveurs MCP et deux sous-agents. Il vous faut une trace de tout ça. Les conventions sémantiques OpenTelemetry GenAI (attributs stables dans v1.37 et plus) sont la norme 2026, nativement prise en charge par Datadog, Langfuse, Arize Phoenix, OpenLLMetry et AgentOps. Cette leçon nomme les attributs requis, traverse la hiérarchie de span (outil agent → LLM →), et envoie un émetteur stdlib span que vous pouvez brancher sur n'importe quel exportateur OTel.

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Nombre des attributs OTel GenAI requis pour une durée de LLM et une durée d'exécution des outils.
- Construisez une hiérarchie de traces qui couvre la boucle d'agent, l'appel LLM, l'appel à l'outil et l'expédition du client MCP.
- Décider quel contenu capturer (opt-in) ou rédiger (par défaut).
- Émettez des étendues à un collecteur local (Jaeger, Langfuse) sans réécrire le code de l'outil.

## Le problème

Un débogage de février 2026: l'utilisateur rapporte " mon agent prend parfois 30 secondes pour répondre; d'autres fois 3 secondes. " Aucun trace. Les journaux montrent l'appel LLM, mais pas le dépêchage de l'outil, pas le serveur MCP aller-retour, pas le sous-agent. Vous devinez. Finalement, vous découvrez: un serveur MCP accroche occasionnellement à un démarrage froid.

Sans tracer de bout en bout, on ne peut pas le trouver.

Les conventions se sont établies en 2025-2026 dans le cadre du groupe de conventions sémantiques OpenTelemetry. Ils définissent les noms d'attributs stables afin que Datadog, Langfuse, Phoenix, OpenLLMetry et AgentOps analysent toutes les mêmes étendues.

## Le concept

### Hiérarchie de la langue

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

Tout est sous une seule trace d'identité.

### Attributs requis

Pour les semestres 2025-2026,

- `gen_ai.operation.name` `"chat"`- Je suis là .`"text_completion"`- Je suis là .`"embeddings"`- Je suis là .`"execute_tool"`- Je suis là .`"invoke_agent"`- Je suis désolé .
- `gen_ai.provider.name` `"openai"`- Je suis là .`"anthropic"`- Je suis là .`"google"`- Je suis là .`"azure_openai"`- Je suis désolé .
- `gen_ai.request.model` chaîne de modèle demandée (par exemple `"gpt-4o-2024-08-06"`)
- `gen_ai.response.model` le modèle a effectivement servi.
- `gen_ai.usage.input_tokens`- Je suis là .`gen_ai.usage.output_tokens`- Je suis désolé .
- `gen_ai.response.id` identifiant de réponse du fournisseur pour la corrélation.

Pour les couches d'outils:

- `gen_ai.tool.name` identifiant de l'outil.
- `gen_ai.tool.call.id` l'identifiant d'appel spécifique.
- `gen_ai.tool.description` description de l'outil (facultatif).

Pour les intervalles d'agents:

- `gen_ai.agent.name`- Je suis là .`gen_ai.agent.id`- Je suis là .`gen_ai.agent.description`- Je suis désolé .

### Les espèces de spans

- `SpanKind.CLIENT`pour les appels qui franchissent une frontière de processus (fournisseur de MLL, serveur MCP).
- `SpanKind.INTERNAL`pour les étapes de la boucle de l'agent et l'exécution de l'outil.

### Capture de contenu à option

Par défaut, les intervalles portent des mesures et des délais  pas des invites ou des compléments.`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`Les résultats de la recherche ont été analysés en détail et en détail.

### Evénements sur les spans

Les événements au niveau des jetons peuvent être ajoutés en tant qu'événements de durée:

- `gen_ai.content.prompt` messages d'entrée.
- `gen_ai.content.completion` messages de sortie.
- `gen_ai.content.tool_call` appel à l'outil tel qu'enregistré.

Les événements dans l'ordre chronologique dans une période de reproduction détaillée.

### Les exportateurs

OTel est destiné à:

- **Jaeger / Tempo.**OSS, sur place.
- **Langfuse.**Spécifique en matière d'observabilité du LLM; visualise l'utilisation des jetons.
- **Arize Phoenix.**Evals + traçage combinés.
- **Datadog.**Commercial; parseurs natifs `gen_ai.*`les attributs.
- **Honeycomb.**Il est orienté vers les colonnes, facile à consulter.

Tout le monde parle OTLP, le format de fil.

### Propagation à travers les PCM

Lorsqu'un client MCP appelle un serveur, injectez l'en-tête traceparent W3C dans la requête. Streamable HTTP prend en charge les en-têtes standard. Stdio ne porte pas d'en-têtes HTTP nativement; la feuille de route 2026 de la spécification discute de l'ajout d'une `_meta.traceparent`champ sur les appels JSON-RPC.

Jusqu' à ce que les navires: inclure le traceparent dans le `_meta`Le serveur enregistre l'identifiant de trace.

### Les mesures

En plus des étendues, le génAI semconv définit les métriques:

- `gen_ai.client.token.usage` histogramme.
- `gen_ai.client.operation.duration` histogramme.
- `gen_ai.tool.execution.duration` histogramme.

Utilisez-les pour les tableaux de bord qui n'ont pas besoin de détails par appel.

### Couche AgentOps

AgentOps (fondé en 2024) est spécialisé dans l'observabilité de GenAI. Il embrasse des cadres populaires (LangGraph, Pydantic AI, CrewAI) pour émettre automatiquement des spans OTel. Utilisant si votre pile utilise un cadre pris en charge; utiliser l'instrumentation manuelle autrement.

```figure
t3-span-waterfall
```

## Utilisez-le

`code/main.py`émet des spans en forme d'OTel à un stdout (au format OTLP-JSON) pour un agent qui appelle un LLM, envoie deux outils et fait un MCP aller-retour. Aucun exportateur réel  la leçon se concentre sur la forme de spans et l'ensemble d'attributs. Collez la sortie dans un spectateur compatible avec OTLP ou lisez-la simplement.

À quoi regarder:

- L'identifiant de trace est partagé sur toutes les étendues.
- Les liens parent-enfant sont codés via `parentSpanId`- Je suis désolé .
- Il est nécessaire `gen_ai.*`Les attributs sont peuplés.
- La capture de contenu est désactivée par défaut; un scénario l'active via env var.

## La faire partir

Cette leçon produit `outputs/skill-otel-genai-instrumentation.md`- En raison d'une base de code des agents, la compétence produit un plan d'instrumentation: où ajouter des sphères, quelles attributs pour peupler et quels exportateurs cible.

## Exercices

1. On court .`code/main.py`- Comptez les intervalles et identifiez lequel est client vs interne.

2. Activer la capture de contenu (env var) et confirmer `gen_ai.content.prompt`et `gen_ai.content.completion`Les événements apparaissent.

3. Ajoutez la métrique d' exécution des outils `gen_ai.tool.execution.duration`et l'émet en tant qu'échantillon d'histogramme par appel.

4. Propagation d'un traceparent d'un agent parent dans une demande de MCP `_meta.traceparent`Vérifiez que le serveur MCP voit la même trace d'identification.

5. Lisez la spécification OTel GenAI semconv. Identifiez un attribut énuméré dans le semconv que le code de cette leçon n'émite PAS. Ajoutez-le.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | Open standard for traces, metrics, logs |
| GenAI semconv | "GenAI semantic conventions" | Stable attribute names for LLM / tool / agent spans |
| `gen_ai.*` | "The attribute namespace" | All GenAI attributes share this prefix |
| Span | "Timed operation" | A unit of work with a start, end, and attributes |
| Trace | "Cross-span ancestry" | Tree of spans sharing a trace id |
| SpanKind | "CLIENT / SERVER / INTERNAL" | Hints about span direction |
| OTLP | "OpenTelemetry Line Protocol" | Wire format for exporters |
| Opt-in content | "Prompt / completion capture" | Off by default; env var to enable |
| traceparent | "W3C header" | Propagates trace context across services |
| Exporter | "Backend-specific shipper" | Component that sends spans to Jaeger / Datadog / etc. |

## Pour en savoir plus

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) Conventions canoniques pour les étendues, les mesures et les événements de GenAI
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) Liste des attributs de la MLL et de l'exécution des outils
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) au niveau des agents `invoke_agent`débit
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) Source de vérité hébergée sur GitHub
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) Accès à l'intégration de la production
