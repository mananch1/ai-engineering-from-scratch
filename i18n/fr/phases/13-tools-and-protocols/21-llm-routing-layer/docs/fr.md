# L'équipement de routage de la ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de ligne de

> Le verrouillage par fournisseur est cher. Différentes charges de travail d'appel d'outils s'adaptent à différents modèles. Les passerelles de routage donnent une surface API, des retries, des pannes, le suivi des coûts et des barreaux. Trois archétypes dominent 2026: LiteLLM (auto-hébergé open source), OpenRouter (SaaS géré), Portkey (de qualité de production, open source en mars 2026). Cette leçon nomme les critères de décision et marche sur une passerelle de routage stdlib.

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## Objectifs d'apprentissage

- Distinguer les options d'itinérance auto-hébergées, gérées et de production.
- Implementer une chaîne de retrait qui tente à nouveau les défaillances des fournisseurs dans un ordre prioritaire défini.
- Suivre le coût par demande et l'utilisation des jetons entre les fournisseurs.
- Décider entre LiteLLM, OpenRouter et Portkey pour une restriction de production donnée.

## Le problème

Scenarios où le routage du fournisseur est important:

1. **Cost.**Claude Sonnet coûte 3 fois plus cher que Haiku, pour une tâche de triage, Haiku est suffisant, pour une tâche de synthèse, Sonnet en vaut la peine.

2. **Failover.**OpenAI a une mauvaise heure, toutes les demandes échouent, vous voulez un retour automatique à Anthropic sans redéployer.

3. **Latency.**Une interface de chat en direct a besoin d'un jeton rapide, un résumé de lot ne le fait pas.

4. **Compliance.**Les utilisateurs de l'UE doivent rester dans les régions de l'UE.

5. **Experimentation.**A/B deux modèles sur la même charge de travail.

Le codage manuel de tout cela par intégration est répétitif. Une passerelle de routage donne une API compatible avec OpenAI et gère le reste.

## Le concept

### Forme de proxy compatible avec OpenAI

Tout le monde parle OpenAI.`/v1/chat/completions`, accepte le schéma OpenAI, et interne proxy à Anthropic / Gémeaux / Cohere / Ollama / quoi que ce soit.

### Les pseudonymes de modèle

Au lieu d'un identifiant instantané, votre code dit `our_smart_model`Lorsque un fournisseur envoie une nouvelle génération, vous changez le côté serveur du pseudonyme; votre code ne touche rien.

### Chaînes de renversement

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Les retries comptent contre un budget pour que les cascades de retrait ne provoquent pas de coûts explosifs.

### Cachage sémantique

Les commandes identiques ou presque identiques sont enregistrées dans un cache au lieu du fournisseur. Les économies sur les boucles d'agent répétées peuvent être de 30 à 60%. Les clés sont basées sur l'intégration; les commandes presque identiques partagent une fente de cache.

### Rennes de garde

Niveau de la passerelle:

- **PII redaction.**Passage à base de Regex ou ML avant d'envoyer des instructions.
- **Policy violations.**Rejetez les invites contenant un contenu interdit.
- **Output filters.**- Pour les fuites.

Portkey et Kong ont tous deux des barreaux de protection.

### Limits de taux par clé

Une clé API = une équipe. Les budgets par clé empêchent une équipe de consommer le quota partagé. La plupart des passerelles le supportent.

### Compromises d'hébergement personnel par rapport à des compromis gérés

| Factor | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | Open source, Python | Managed SaaS | Open source (Mar 2026) + managed |
| Setup | Deploy a proxy | Sign up | Either |
| Providers | 100+ | 300+ | 100+ |
| Billing | Your own keys | OpenRouter credits | Your own keys |
| Observability | OpenTelemetry | Dashboard | Full OTel + PII redaction |
| Best for | Teams that want full control | Rapid prototyping | Production with compliance |

LiteLLM gagne quand vous avez une équipe SRE et que vous voulez la souveraineté des données. OpenRouter gagne quand vous voulez un seul abonnement et pas d'infrastructure. Portkey gagne quand vous avez besoin de barrières et de conformité hors de la boîte.

### Tracking des coûts

Chaque demande est accompagnée .`provider`- Je suis là .`model`- Je suis là .`input_tokens`- Je suis là .`output_tokens`- Multipliez par prix par modèle par jeton (extrait d'une feuille de prix que le gateway maintient).

### MCP plus routage

Une passerelle peut router les appels LLM ET les demandes d'échantillonnage MCP. Lorsque le modèle de la demande d'échantillonnagePréférences préfère un modèle spécifique, la passerelle se traduit vers le bon backend. C'est là que la phase 13 · 17 (MCP gateway) et la passerelle de routage de cette leçon fusionnent parfois en un seul service.

### Stratégies de routage

- **Static priority.**Première de la liste; renoncez à l'erreur.
- **Load balancing.**Ronde-robin ou pondérée.
- **Cost-aware.**Choisissez le modèle le moins cher pour répondre à la latence / qualité.
- **Latency-aware.**Choisissez le modèle le plus rapide en N minutes.
- **Task-aware.**Les routes de classification rapide codant à un modèle, résumant à un autre.

```figure
tp-router-failover
```

## Utilisez-le

`code/main.py`Il utilise une passerelle de routage en ~ 150 lignes: accepte les demandes en forme d'OpenAI, traduit en coups par fournisseur, exécute une chaîne de rétroaction prioritaire, suit le coût par demande et applique un passe de rédaction d'IIP aux entrées.

À quoi regarder:

- `ROUTES`dict: alias -> liste des fournisseurs concrets par ordre prioritaire.
- Retour à la boucle de retour sur 5xx.
- Le tracker de coûts multiplie l'utilisation des jetons par taux par modèle.
- Le rédacteur d'informations personnelles nettoie les modèles en forme de SSN avant de les transférer.

## La faire partir

Cette leçon produit `outputs/skill-routing-config-designer.md`. Compte tenu du profil de charge de travail (latence, coût, conformité), le compétences choisit LiteLLM / OpenRouter / Portkey et produit une configuration de routage.

## Exercices

1. On court .`code/main.py`- déclencher le scénario de panne; confirmer que le second fournisseur est en panne et que le coût est correctement attribué.

2. Ajouter la mise en cache sémantique: SHA256 de la demande est une clé de recherche; les clics de mise en cache reviennent instantanément. Mesurer les économies de coûts sur un appel répété.

3. Ajouter un classifiateur rapide qui route "code ... " aux invites d'un alias favorisant l'intelligence et " résumer ... " aux invites d'un alias favorisant la vitesse.

4. Conception de budgets par équipe: chaque équipe dispose d'un plafond de dépenses mensuels; Gateway refuse les demandes une fois le plafond atteint. Choisissez une granularité d'application (par demande ou en fenêtre).

5. Lisez les documents LiteLLM, OpenRouter et Portkey côte à côte.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Routing gateway | "LLM proxy" | One-API-surface layer in front of many providers |
| OpenAI-compatible | "Speaks the OpenAI schema" | Accepts `/v1/chat/completions` shape, translates to any backend |
| Model alias | "our_smart_model" | Name in your code that the gateway maps to a concrete model |
| Fallback chain | "Retry list" | Ordered list of providers attempted on failure |
| Semantic caching | "Prompt-embedding cache" | Key is embedding of the prompt; near-duplicates share a cache hit |
| Guardrails | "Input/output filters" | Redact PII, reject policy violations |
| Per-key rate limit | "Team budget" | Quota scoped to an API key |
| Cost tracking | "Per-request spend" | Aggregate token usage x price per model |
| LiteLLM | "The open proxy" | Self-hostable OSS routing gateway |
| OpenRouter | "The managed SaaS" | Hosted gateway with credit-based billing |
| Portkey | "The production option" | Open-source + managed with guardrails built in |

## Pour en savoir plus

- [LiteLLM — docs](https://docs.litellm.ai/) passerelle de routage auto-hébergée
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) SaaS de routage géré
- [Portkey — docs](https://portkey.ai/docs) routage de la production avec barreaux
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) Guide de décision
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) Enquête auprès des fournisseurs
