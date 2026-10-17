# Portkey, Kong, Bifrost

> Une passerelle se trouve entre vos applications et les fournisseurs de modèles. Les principales caractéristiques sont le routage des fournisseurs, le retrait, les retries, la limitation des tarifs, les références secrètes, l'observabilité, les barreaux.**LiteLLM**est MIT OSS avec plus de 100 fournisseurs, compatible avec OpenAI, mais se décompose autour de ~ 2000 RPS (8 Go de mémoire, défaillances en cascade dans les benchmarks publiés); le mieux pour Python, < 500 RPS, développement / prototypage. **Portkey**est positionné sur le plan de contrôle (gardiens, édition d'informations personnelles, détection de jailbreak, pistes d'audit), est passé Apache 2.0 open-source mars 2026, 20-40 ms latence overhead, $49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $100/modèle/mois prix (max 5 sur le niveau Plus); adapté pour l'entreprise si vous êtes déjà sur Kong. **Bifrost**(Maxim AI)  retries automatiques avec configurable backkoff, retour à Anthropic sur OpenAI 429. **Cloudflare / Vercel AI Gateways** géré, zéro-opérations, réessayer de base. La résidence des données conduit la décision de l'hébergeur; Portkey et Kong sont au milieu avec OSS + géré facultatif.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## Objectifs d'apprentissage

- Enumérer les six caractéristiques de la passerelle principale (routage, retrait, retrait, limites de vitesse, secrets, observabilité, barreaux).
- Mettez en page quatre passerelles 2026 (LiteLLM, Portkey, Kong AI, Bifrost) pour élargir les plafonds et les cas d'utilisation.
- Citez le point de référence Kong (228% contre Portkey, 859% contre LiteLLM) et expliquez pourquoi il importe pour >500 RPS.
- Choisissez auto-hébergé par rapport à géré en fonction de la résidence des données et du budget des opérations.

## Le problème

Votre produit appelle OpenAI, Anthropic et un Llama auto-hébergé. Chaque fournisseur a un SDK différent, un modèle d'erreur, une limite de tarification et un schéma d'auth. Vous voulez un failover (si OpenAI 429, essayez Anthropic), un seul magasin de crédits, une observabilité unifiée et des limites de tarification par locataire.

Une nouvelle invention de cette technologie dans la couche d'applications coupe chaque service à chaque fournisseur.

## Le concept

### Six caractéristiques fondamentales

1. **Provider routing** OpenAI, Anthropic, Gemini, auto-hébergé, etc. derrière une seule API.
2. **Fallback** sur 429, 5xx, ou défaillance de qualité, réessayez ailleurs.
3. **Retries**- Des tentatives limitées.
4. **Rate limits** par locataire, par clé, par modèle.
5. **Secret references** retirer les informations d'identification de la caisse en temps d'exécution (ne jamais dans l'application).
6. **Observability** ATTEL + GenAI (phase 17 · 13) + attribution des coûts.
7. **Guardrails** Rédition d'informations personnelles, détection de jailbreak, filtres d'objet permis.

### L'utilisation de l'application est également possible.

- 100+ fournisseurs, compatibles avec OpenAI, configuration du routeur, rétroaction, observabilité de base.
- Des pannes d'environ 2000 RPS dans le repère de Kong; 8 Go de mémoire, défaillances en cascade sous charge soutenue.
- Meilleure taille: application Python, < 500 RPS, dev/staging gateways, routage expérimental.
- Coût: 0 $ pour OSS; niveau libre du cloud existe.

### Portkey  positionnement du plan de contrôle

- Apache 2.0 OSS à partir de mars 2026. Garde-rails, rédaction des informations personnelles, détection de jailbreak, pistes d'audit.
- 20 à 40 ms par demande de latence.
- 49 $ par mois pour le niveau de production avec Rétention + SLA.
- Le meilleur ajustement: les industries réglementées nécessitant des barreaux + la capacité d'observation regroupée.

### Kong AI Gateway  le jeu de l'échelle

- Construit sur Kong Gateway (produit de passerelle API mature, lua+OpenResty).
- Le propre indice de référence de Kong sur l'équivalent de 12 CPU: 228% plus rapide que Portkey, 859% plus rapide que LiteLLM.
- Prix: 100 $ par mois, maximum 5 sur le niveau Plus.
- Le meilleur ajustement: déjà sur Kong; > 1000 RPS; prêt à licencier.

### Bifrost (AI maximale)

- Réessayer automatiquement avec un backoff configurable.
- Le retour à Anthropic sur OpenAI 429 est une recette canonique.
- Nouveau participant, commercial.

### Cloudflare AI Gateway / Vercel AI Gateway

- Gestion, zéro opération, réessayer et observer.
- Le meilleur ajustement: les applications JavaScript à bord sur Cloudflare/Vercel.
- Limitée par rapport à Kong/Portkey sur les barreaux et les limites de vitesse.

### Autogestion par rapport à gestion

La résidence des données est la fonction de contrainte. Les soins de santé et les finances sont auto-hébergés par défaut (LiteLLM ou Portkey OSS ou Kong). Les produits de consommation sont gérés par défaut (Cloudflare AI Gateway) ou de niveau moyen (Portkey managed). Hybride: auto-hébergé pour les locataires réglementés, géré pour d'autres.

### Budget de la latence

- L'utilisation de la ligne de conduite est généralement limitée à 5 à 15 ms.
- 20 à 40 ms au-dessus.
- 3 à 8 ms au-dessus.
- Cloudflare/Vercel: 1 à 3 ms de charge générale (avantage de bord).

La latence de la passerelle s'ajoute directement au TTFT. Pour TTFT P99 < 100 ms SLA, Kong ou Cloudflare. Pour P99 < 500 ms, n'importe quel.

### Matériel de la sémantique des limites de taux

Le jeton-bucket simple fonctionne à une échelle modérée. Multi-locataire nécessite une fenêtre coulissante + allocation de débordement + tirage par locataire. LiteLLM envoie un jeton-bucket; Kong envoie une fenêtre coulissante; Portkey envoie des navires classés.

### Gateway + observabilité + routage composé

La phase 17 · 13 (observabilité) + 16 (routing modèle) + 19 (gateways) sont la même couche de production. Choisissez un outil qui couvre les trois ou les câble soigneusement: la plupart des déploiements 2026 combinent Helicone (observabilité) ou Portkey (garderelles) avec Kong (échelle) pour les rôles divisés.

### Les chiffres que vous devriez vous rappeler

- LiteLLM: rupture à environ 2000 RPS, 8 Go de mémoire.
- Portkey: 20 à 40 ms en charge générale; Apache 2.0 depuis mars 2026.
- Kong: 228% plus rapide que Portkey, 859% plus rapide que LiteLLM.
- Prix Kong: 100 $ par modèle par mois, 5 max sur le niveau Plus.
- Cloudflare/Vercel: 1 à 3 ms de charge à l'extrémité.

```figure
mx-gateway-fallback
```

## Utilisez-le

`code/main.py`Simulation du routage de passerelle avec des retards sur 3 fournisseurs sous injection 429/5xx. Rapporte la latence, le taux de réessayer et le taux de retards.

## La faire partir

Cette leçon produit `outputs/skill-gateway-picker.md`Compte tenu de l'ampleur, de la posture des opérations, de la conformité, du budget de latence, choisit une passerelle.

## Exercices

1. On court .`code/main.py`. Configurer le retrait de OpenAI→Anthropic→auto-hébergé. Quel est le taux de succès attendu à 5% de taux d'erreur du fournisseur?
2. Votre SLA est TTFT P99 < 200 ms sur une ligne de base de 300 ms. Quelles passerelles restent dans le budget ?
3. Un client de soins de santé a besoin d'un hébergement personnel + de rédaction des données personnelles + d'audit.
4. Comparer LiteLLM vs Kong: à quel plafond RPS une équipe devrait migrer ?
5. Conceptionner une politique de limite de tarifs pour un SaaS multi-locataire: niveau gratuit, niveau d'essai, niveau payant.

## Les termes clés

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

## Pour en savoir plus

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
