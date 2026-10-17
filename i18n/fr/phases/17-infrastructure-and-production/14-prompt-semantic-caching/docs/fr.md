# Le caching rapide et l'économie du caching sémantique

> **Pricing snapshot dated 2026-04.**Les revendications numériques ci-dessous reflètent les cartes de taux des fournisseurs capturées à la publication de cette leçon; vérifiez les documents liés avant de les citer en aval.

> Le caching se fait à deux niveaux. L2 (niveau fournisseur) le caching de prompt/prefix réutilise attention KV pour les préfixes répétés  Les documents de caching de prompt d'Anthropic annoncent une réduction de coût allant jusqu'à 90% et une réduction de latence de 85% sur les longues demandes; pour Claude 3.5 Sonnet, les lectures de cache sont $0.30/M vs $3,00/M frais avec un TTL de 5 minutes et une prime de rédaction de 2 fois pour l'option TTL d'une heure (docs.anthropic.com, 2026-04). Le caching rapide OpenAI s'applique automatiquement pour les instructions ≥ 1024 jetons et les prix de l'entrée caché à environ 90% de réduction par rapport au frais (platform.openai.com, 2026-04); le taux exact de caching par modèle dépend de la carte de taux en direct. L1 (app-level) cache sémantique saute le LLM entièrement sur l'intégration de hits de similitude. Vendor "95% accuracy" désigne la correction des correspondances, pas le taux de succès  les taux de succès de production rapportés vont de 10% (chat ouvert) à 70% (FAQ structurée); aucun fournisseur ne publie une ligne de base officielle, alors traitez-les comme une télémétrie communautaire plutôt que comme des garanties. Les pièges de production: la parallélisation tue le caching (N requêtes parallèles émises avant la première écriture du cache peuvent gonfler les dépenses plusieurs fois), et le contenu dynamique à l'intérieur du préfixe empêche les hits du cache entièrement. ProjectDiscovery a rapporté passer de 7% à 74% de taux de succès (2025-11) en déplaçant du texte dynamique hors du préfixe cacheable.

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Objectifs d'apprentissage

- Distinguer le caching de prompts/préfixes L2 (reutilisation de KV chez le fournisseur) du caching sémantique L1 (bypass de LLM sur des prompts similaires).
- Expliquez à l'anthropique `cache_control`marquage explicite et les deux options TTL (5 min contre 1 heure) avec leurs multiplicateurs de prix.
- Comptez les économies mensuelles attendues compte tenu du taux de succès, du mix prompt/response et des prix des jetons.
- Nommez le modèle anti-parallélisation qui gonfle les factures de 5 à 10 fois et le modèle anti-contenu dynamique qui s'effondre taux de frappe.

## Le problème

Vous ajoutez le caching rapide à votre service RAG. La facture reste plate. Vous mesurez le taux de succès; il est de 7%. Vos demandes semblent statiques mais elles ne le sont pas.

Par ailleurs, votre agent effectue 10 appels parallèles par requête utilisateur. Tous les 10 arrivent au fournisseur avant la fin de la première mise en cache. 10 écrit, zéro est lu. Votre facture est de 5 à 10 fois plus cher que ce que "avec la mise en cache" était censé coûter.

Le caching est un protocole, pas un drapeau, deux couches, deux modes de défaillance différents.

## Le concept

### L2  mise en cache des prompts/préfixes du fournisseur

Le fournisseur stocke le KV d'attention pour un préfixe cacheable et le réutilise à la prochaine demande qui correspond au préfixe.

**Anthropic (Claude 3.5 / 3.7 / 4 series)**: explicite `cache_control`TTL: 5 minutes (écriture coûte 1,25x base) ou 1 heure (écriture coûte 2x base).$0.30/M on Claude 3.5 Sonnet vs $3,00/M frais  10 fois moins cher (docs.anthropic.com, à partir de 2026-2004). Les tarifs diffèrent par modèle (Opus/Haiku publié séparément); vérifiez toujours la page de prix en direct.

**OpenAI**: mise en cache automatique pour les instructions ≥1024 jetons (platform.openai.com, 2026-04). Aucun drapeau explicite. L'entrée en cache est environ 10 fois moins chère que la fraîche sur les cartes de taux gpt-4o/gpt-5 actuelles. Ni les documents ni les notes de sortie ne publient une ligne de base officielle de taux d'accident; les rapports communautaires se regroupent autour de 3060% avec une conception rapide soigneuse. Moniteur `usage.cached_tokens`Pour mesurer les vôtres.

**Google (Gemini)**: le caching de contexte via une API explicite; le caching de contexte 1M-token signifie que le caching paie encore plus.

**Self-hosted (vLLM, SGLang)**: La phase 17 · 06 couvre RadixAttention  le même schéma à votre propre calcul.

### L1  Cachage sémantique au niveau de l'application

Avant d'appeler le LLM, faites le hash de la demande, encassez-la et recherchez une demande en cache similaire (semblance de cousin au-dessus du seuil, généralement 0,95 +).

Le code de base est le code de base de la base de données.

Les revendications de précision du fournisseur se réfèrent à la fréquence à laquelle la réponse en cache retournée était sémantiquement appropriée  et non à la fréquence de frappe.

- Chat ouvert: 10 à 15%.
- Questions fréquentes structurées / soutien: 40-70%.
- Questions de code: 20-30% (petites variantes tuent les hits).
- Agents de voix répéter des instructions: 50-80% (configuration fixe de normalisation vocale).

### Le modèle anti-parallélisation

Votre agent fait 10 appels d'outils en parallèle. Tous les 10 ont la même requête système 4K-token. Les écritures de cache anthropic sont par requête; la première écriture de cache se termine environ 300 ms après que le fournisseur a vu la requête. Les demandes 2-10 arrivent dans la même fenêtre de milliseconde et chacune voit le cache manquer. Vous payez 10 primes d'écriture, 0 rabais de lecture.

Correction: lot avec séquentiel-first  faire la demande 1 seul, puis le feu 2-10 une fois que le cache 1 a été peuplé. Ajout de 300 ms au premier appel de l'outil; économise 5-10x la facture.

### Le modèle anti-contenu dynamique

Votre commande de système ressemble à:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Chaque requête est unique, chaque requête écrit, nul succès.

Correction: déplacer tout ce qui est vraiment statique vers le préfixe cacheable; ajouter du contenu dynamique après la limite de cache:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery est passé de 7% à 74% de taux de cache de cette façon et a publié l'anatomie.

### Charge de pile + cache pour les charges de travail de nuit

Les API de lot (phase 17 · 15) offrent une réduction de 50% à la rotation de 24 heures. Les entrées cachées en haut vous donnent ~ 10 fois plus. Les charges de travail de classification, d'étiquetage et de génération de rapports peuvent diminuer de ~ 10% du coût synchrone-non-caché par stacking.

### Les chiffres que vous devriez vous rappeler

Les points de prix sont capturés 2026-04 à partir des documents des fournisseurs liés et dérivent tous les quelques mois  vérifier à nouveau avant de s'y fier.

- Lire en cache: 0,30 $/M sur Claude 3.5 Sonnet, environ 10 fois moins cher que les entrées fraîches (docs.anthropic.com).
- Prémium d'écriture de cache anthropic: 1,25x (5 min TTL) ou 2x (1 heure TTL).
- OpenAI automatique de mise en cache: s'applique aux signaux de commande ≥ 1024; entrée en cache au prix d'environ 10% de l'entrée fraîche sur les cartes de taux courants (platform.openai.com).
- Taux de succès du cache sémantique (reporté par la communauté): ~10% de chat ouvert; jusqu'à ~70% de FAQ structurées. Pas de base documentée par le fournisseur.
- ProjectDiscovery: 7% → 74% de taux de succès en déplaçant la dynamique hors préfixe (blog du projet, 2025-11).
- Anti-pattern de parallélisation: rapports typiques d'inflation de factures 510x lorsque N requêtes parallèles manquent la première cache écrire.

```figure
semantic-cache-hit
```

## Utilisez-le

`code/main.py`Les rapports de taux de rebond, facturation et montre la pénalité de parallélisation.

## La faire partir

Cette leçon produit `outputs/skill-cache-auditor.md`- compte tenu du modèle et du trafic, il vérifie la cachéabilité et recommande une restructuration.

## Exercices

1. On court .`code/main.py`- Quel est le prix de la facture ?
2. Votre commande système a une date.
3. Calculer le break-even pour 1 heure TTL (2 fois écrit) contre 5 minutes TTL (1,25 fois écrit) compte tenu du taux d'arrivée de votre demande.
4. Le cache sémantique à 0,95 atteint 20%. à 0,85 il atteint 50% mais vous voyez des réponses cachées incorrectes. Choisissez le bon seuil et justifiez.
5. Vous faites 10 requêtes parallèles par requête utilisateur. Réécrivez pour la facilité de mise en cache sans ajouter de latence de bout en bout.

## Les termes clés

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

## Pour en savoir plus

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) officiel `cache_control`la sémantique et les TTL.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) comportement de mise en cache automatique et admissibilité.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
