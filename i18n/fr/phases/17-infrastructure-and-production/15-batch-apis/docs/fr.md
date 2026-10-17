# Les API de lot  la réduction de 50% en tant que norme de l'industrie

> Chaque fournisseur majeur envoie une API de lot asynchrone avec une réduction de 50% et une rotation de 24 heures. OpenAI, Anthropic, Google et la plupart des plateformes d'inférence (batch tier de Fireworks, batch ensemble) mettent en œuvre le même schéma. Les lots de stockage avec un caching rapide et les pipelines de nuit diminuent à ~10% du coût synchrone-non-caché. La règle est brutalement simple: si elle n'est pas interactive, elle appartient au lot. Les lignes de production de contenu, la classification des documents, l'extraction de données, la production de rapports, l'étiquetage en vrac, l'étiquetage du catalogue  tout ce qui tolère une latence de 24 heures est de l'argent laissé sur la table jusqu'à ce qu'il passe au lot. Le modèle de production de 2026 consiste à trier chaque nouvelle charge de travail du LLM en trois voies: interactive (synchrone avec le caching), semi-interactive (couche asynchrone avec le fallback), lot (sur la nuit, entrée en cache empilée). Les charges de travail qui prétendent être interactives mais tolèrent les minutes de latence gaspillent le plus.

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## Objectifs d'apprentissage

- Nombre des trois API de lot de fournisseurs (OpenAI, Anthropic, Google) et des garanties communes de réduction de 50% + 24h de retour.
- Calculer le coût de l'empilage de lot + entrée en cache sur une charge de travail de classification de nuit et comparer avec la ligne de base synchrone-non-chassée.
- Troiser une charge de travail en lot interactif / semi-interactif / lot et justifier la voie.
- Nombre des deux pièges: interactivité partielle (l'utilisateur s'attend à une vitesse supérieure à 24h) et dérive du schéma de sortie (format de fichier de lot diffère par fournisseur).

## Le problème

Votre équipe envoie un pipeline de génération de rapports par nuit. 50 000 documents, résumez chacun, regroupez les résumés, rédigez un résumé exécutif.

Le lot vous donne 50% de réduction. Vous activez également la mise en cache rapide sur le système de demande (partagé sur tous les 50k appels).

Le lot est le levier le moins cher dans le kit de coûts de la LLM que personne ne tire. La raison est principalement organisationnelle: les équipes pensent "en temps réel" lorsque le SLA est en fait "dans la matinée".

## Le concept

### Les trois API de lot

**OpenAI Batch API**: JSONL fichier téléchargement avec une liste de demandes. Promise de 24 heures de retour (habituellement ~ 2-8 heures dans la pratique). 50% de réduction sur les jetons d'entrée et de sortie. `/v1/batches`Les entrées éligibles au caché ont également des prix de l'entrée en caché en haut.

**Anthropic Message Batches**JSONL téléchargement, 24 heures de retour, 50% de réduction.`cache_control` les écritures en cache sont explicites, les lectures se produisent automatiquement dans le lot.

**Google Vertex AI Batch Prediction**: BigQuery ou GCS entrée. 50% de réduction similaire pour Gemini. Intégré avec les pipelines Vertex.

### Sémantique: asynchrone, pas lente

Le lot est "Je promets de revenir dans les 24 heures"  pas "cela prendra 24 heures". Le P50 typique est de 2 à 6 heures.

### - L' emplacement de la mise en cache

Un résumé de document 50K avec le même système de jetons 4K:

- Les données de l'équipe de surveillance sont définies dans le tableau de bord.$input × 4000 + $en fonction des taux de débit de l'énergie.
- Caché synchrone: le système de demande est mis en cache après la première rédaction; les 49999 restants obtiennent 10 fois moins cher entrée.
- Partie mise en cache: toutes les données ci-dessus plus 50% de réduction sur la lecture et l'écriture.

La pile: lot + cache = ~10% de synchronisation de facture non caché. Toute charge de travail qui fonctionne du jour au lendemain et a une mise en service système partagée devrait utiliser ceci.

### Triation de la charge de travail

**Interactive**L'utilisateur attend la réponse. TTFT importe. Appel synchrone avec cache rapide. Pas de lot.

**Semi-interactive** l'utilisateur soumet une tâche, vérifie en quelques minutes.

**Batch** l'utilisateur s'attend à des résultats "d'ici le matin" ou "à l'heure prochaine".

Erreur commune: classer tout comme interactif parce que le pipeline est la production.

### Le piège de l'interactivité partielle

Certaines fonctionnalités semblent interactives mais tolèrent 5 à 10 minutes. Exemple: un rapport de santé du client par nuit avec le bouton "refresh". L'utilisateur clique sur refresh; attendre 10 minutes est bien. L'équipe les expédie en synchronisation. 50 refreshs simultanés coûtent 10 fois ce que coûterait le partage et la livraison par e-mail.

La question à poser: " Que signifie 24 heures pour cet utilisateur ? " Si la réponse est " ils ne le remarqueront pas ", partagez-la.

### Le piège du schéma de sortie

Les formats de fichiers de lot diffèrent par fournisseur:

- JSONL, une requête par ligne.
- Anthropic: JSONL, un message par ligne; format de réponse intégré.
- Vertex: Table BigQuery ou préfixe GCS avec TFRecord.

L'écriture de "un client de lot" entre les fournisseurs signifie le code de l'adaptateur par fournisseur.

### Les chiffres que vous devriez vous rappeler

- Réduction par lots entre les fournisseurs: 50% fixe sur entrée + sortie.
- SLA de retour: 24 heures garanties, 2 à 6 heures typiques P50.
- Partie empilée + entrée en cache: ~10% du coût non caché de synchronisation.
- Règle de triation de la charge de travail: si la latence de 24h est acceptable, toujours en lots.

```figure
batch-lane-triage
```

## Utilisez-le

`code/main.py`Compute les coûts sur synchronisation, synchronisation + cache, lot et lot + cache pour une charge de travail de 50 000 documents.

## La faire partir

Cette leçon produit `outputs/skill-batch-triager.md`- compte tenu des caractéristiques de la charge de travail, trier en lots/semi/interactifs et estimer les économies.

## Exercices

1. On court .`code/main.py`. Pour un pipeline de 100k-doc avec une réponse système de 3K-token et une sortie de 500-tokens, calculer l'économie de la pile complète (batch + cache) par rapport à la ligne de base de synchronisation.
2. Choisissez trois caractéristiques dans un produit réel que vous connaissez.
3. Un utilisateur se plaint que leur rapport a pris 3 heures.
4. Votre SLA de retour API de lot est de 24h mais P99 est de 20h. Comment communiquer cela à l'utilisateur  quel est le comportement du système en aval sur le boîtier de bord?
5. Compute l'équilibre: à quelle longueur de préfixe partagé le lot + cache devient moins cher que de fonctionner du jour au lendemain sur votre propre GPU réservé ?

## Les termes clés

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

## Pour en savoir plus

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) Format JSONL et `/v1/batches`la sémantique.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) format de lot et `cache_control`l'interaction.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) Sémantique de lot de Gémeaux.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
