# Sélection de la pile d'observabilité du LLM

> Le marché de l'observabilité de 2026 est divisé en deux catégories. Les plateformes de développement (LangSmith, Langfuse, Comet Opik) regroupent la surveillance avec des évaluations, une gestion rapide, des répétitions de session. Les outils de passerelle/instrumentation (Helicone, SigNoz, OpenLLMetry, Phoenix) se concentrent sur la télémétrie. Langfuse est un noyau sous licence MIT avec un fort équilibre OSS (50K événements / mois en nuage gratuit). Phoenix est né en OpenTelemetry sous licence élastique 2.0  excellent pour la visualisation drift/RAG, pas un backend de production persistant. Arize AX utilise une intégration Iceberg/Parquet à copie zéro, affirmant une observabilité monolithique 100 fois moins chère. LangSmith est à la tête de LangChain/LangGraph, 39 $/utilisateur/mo, auto-hébergement dans Enterprise seulement. Helicone est basé sur le proxy avec 15-30 minutes de configuration, 100K de req / mo libre, mais moins de profondeur sur les traces d'agent. Modèle de production commun: Gateway (Helicone/Portkey) + plateforme d'évaluation (Phoenix/TruLens) collée à OpenTelemetry.

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Objectifs d'apprentissage

- Distinguer les plateformes de développement (agrégées en valeurs: évaluations + invitations + sessions) des outils de passerelle/télémétrie (seulement traces + métriques).
- Mettez en page six outils majeurs (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) pour les licences, les prix et les cas d'utilisation des points doux.
- Expliquez le modèle d'enclos OpenTelemetry qui vous permet de combiner un outil de passerelle avec une plateforme d'évaluation distincte.
- Nommez le différenciateur de coûts de 2026 (approche zéro copie d'Arize AX par rapport à l'ingestion monolithique) et indiquez le multiplicateur de 100x approximatif.

## Le problème

Vous avez envoyé une fonctionnalité de LLM. Elle fonctionne. Vous n'avez pas de visibilité sur les pannes rapides, les boucles d'outils, les régressions de latence, les pics de coûts ou le taux de succès de la mise en cache rapide. Vous recherchez "l'observabilité de LLM" et obtenez huit outils tous prétendant résoudre le même problème à trois prix différents.

Ils ne résolvent pas le même problème. LangSmith répond: "Pourquoi cette exécution LangGraph a-t-elle échoué?" Phoenix répond: "Mon pipeline RAG dérive-t-elle?" Helicone répond: "Quelle application brûle des jetons?" Langfuse répond: "Peut-on auto-héberger l'ensemble?"

Le choix implique quatre axes: stack (longchain? raw SDK? multi-vendor?), tolérance aux licences (m.t. seulement? élastique OK? fine commerciale?), budget (niveau gratuit? $100/mo? $1000/mo?), et l'hébergeur (deux ?

## Le concept

### Deux catégories

**Development platforms**Vous pouvez faire des expériences, voir quel prompt a fonctionné, régression d'un nouveau prompt contre d'anciens gagnants.

**Gateway/telemetry tools**Les appels d'infrastructure sont des appels  prompt, réponse, jetons, latence, modèle, coût. Helicone, SigNoz, OpenLLMetry, Phoenix. Minimaliste. Peut être combiné avec un outil d'évaluation séparé via OpenTelemetry.

### Langfuse  équilibre de l'OSS

- Licence Apache / MIT; hébergeur automatique via Docker.
- Le niveau gratuit dans le cloud: 50 000 événements par mois.
- Evals, gestion rapide, traces, ensembles de données, couverture raisonnable des quatre fonctionnalités de la plateforme de développement.
- Bon point: vous voulez des fonctionnalités de la classe LangSmith mais vous devez vous héberger ou rester sous licence OSS.

### Phoenix (Arize)  Télémétrie-première, OpenTélémétrie-native

- Licence élastique 2.0; auto-hébergeur trivial.
- Excellent pour la visualisation RAG et la dérive.
- Non conçu comme un backend de production persistant  principalement observable au cours du développement.
- Bon point: développement de pipelines RAG, débogage à dérive, paires avec une passerelle séparée pour la production.

### Arize AX  le jeu de l'échelle

- Intégration de la mer de données à copie zéro via Iceberg/Parquet.
- Les mathématiques: vous stockez des traces dans votre propre parc sur S3; Arize lit directement.
- Bon point: > 10 millions de traces par jour, lac de données existant, veulent des tableaux de bord spécifiques à la LLM sans tarification Datadog.

### LangSmith  LangChain/LangGraph d'abord

- Commercial, 39 $ par mois, hébergeur personnel sur Enterprise.
- Le meilleur de sa catégorie pour les piles LangChain et LangGraph.
- Un groupe engagé dans LangChain, prêt à payer.

### Helicone  minimum viable à base de proxy

- 15 à 30 minutes de configuration en échangeant votre `OPENAI_API_BASE`à un proxy de l'hélicone.
- Licence MIT; 100 000 requêtes par mois gratuites, payées 20 $ par mois+.
- Inclut le failover, le caching, les limites de tarifs  agit également comme une passerelle.
- Moins de profondeur sur les traces d'agent / multi-étape.
- Bon point: démarrage rapide, application à pile unique, besoin de passerelle + observabilité en un.

### Opik (Comet)  Plateforme de développement OSS

- Apache 2.0, entièrement sous réserve d'exploitation.
- Une caractéristique similaire à Langfuse avec héritage de la comète.
- Les équipes de ML qui sont déjà sur Comet veulent l'observabilité de la LLM dans le même panneau.

### SigNoz  OpenTelemetry-first APM complet

- Apache 2.0 gère l'APM général plus le LLM via OpenTelemetry.
- Un point positif: une observabilité unifiée entre les services et les appels de LLM.

### La colle: OpenTelemetry + conventions sémantiques GenAI

OpenTelemetry a publié des conventions sémantiques de GenAI fin 2025 (`gen_ai.system`- Je suis là .`gen_ai.request.model`- Je suis là .`gen_ai.usage.input_tokens`Les outils qui consomment OTel peuvent interagir.

1. Émettez OTel avec les conventions de GenAI à chaque appel de LLM.
2. Route vers la passerelle (Helicone / Portkey) pour le quotidien.
3. Dual-ship à évaluer la plateforme (Phoenix / Langfuse) pour les régressions.
4. Archivage dans le lac de données (Iceberg) pour une analyse à long terme via Arize AX ou DuckDB.

### Le piège: instrumentation dans la mauvaise couche

L'instrumentation à l'intérieur de votre framework d'agent (par exemple, l'ajout de traces LangSmith) vous associe à ce framework.

### On ne peut pas tout garder.

Pour les demandes de plus de 1 million par jour, la rétention complète coûte plus cher que les appels de LLM. échantillon par règles: 100% d'erreurs, 100% de coûts élevés, 5% de succès.

### Les chiffres que vous devriez vous rappeler

- Nuage Langfuse gratuit: 50 000 événements par mois.
- LangSmith: 39 $ par mois.
- Helicone gratuite: 100 000 réactifs par mois.
- Arize AX revendique: ~ 100 fois moins cher que le monolithique à l'échelle.
- Conventions de l'OpenTelemetry GenAI: 2025 navigation, 2026 largement adopté.

```figure
i4-otel-glue
```

## Utilisez-le

`code/main.py`Simulation d'une journée de 1M sur les stratégies de rétention (100% ingestion, prélèvement, prélèvement + erreurs).

## La faire partir

Cette leçon produit `outputs/skill-observability-stack.md`. Compte tenu de la pile, de l'échelle, du budget, de la posture de licence, choisit l'outil (s).

## Exercices

1. Votre équipe sur LangChain veut une observation auto-hébergée par OSS.
2. À 5M traces par jour avec Datadog citées 150K $ par mois, calculer l'équilibre pour Arize AX.
3. Conception d'un attribut OpenTelemetry GenAI fixé par la directive de votre organisation devrait être mandaté à chaque appel de LLM.
4. Discutez si Phoenix est suffisant pour la production.
5. L'hélicone est de 20 ms. à P99 TTFT 300 ms, est-ce acceptable ?

## Les termes clés

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

## Pour en savoir plus

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
