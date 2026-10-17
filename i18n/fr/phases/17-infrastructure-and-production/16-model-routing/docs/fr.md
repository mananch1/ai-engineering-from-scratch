# Le modèle de routage comme un primitif de réduction des coûts

> Un courtier dynamique évalue chaque demande (type de tâche, longueur de jeton, similarité d'intégration, confiance) et envoie des requêtes simples à un modèle bon marché, en augmentant les requêtes complexes à un modèle frontalier. On l'appelle aussi cascade de modèle. Des études de cas de production montrent une réduction de 20 à 60% des coûts à l'aide de l'iso-qualité dans les déploiements aux États-Unis/Royaume-Uni/UE; une amélioration de 30% de l'efficacité de routage sur le SaaS à volume élevé se transforme en économies annuelles de six chiffres. Le contexte de 2026 est que les prix des inférences LLM ont chuté ~ 10x par an  un jeton de classe GPT-4 est allé de $20/M to ~$0,40/M de fin 2022 à 2026. La plupart des dépôts sont plus efficaces pour les piles (phase 17 · 04-09), et non pour le matériel. Le routage est la façon dont vous convertissez cette baisse de prix en marge sans régression du produit. Le mode d'échec est le dérivé du modèle bon marché: le trajet pousse 40% vers un modèle plus faible, la qualité diminue de 3 à 5% sur les tâches de raisonnement, personne ne remarque pendant un quart. Les itinéraires de passerelle par métriques de qualité en ligne, pas seulement les ensembles d'évaluation hors ligne.

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 minutes

## Objectifs d'apprentissage

- Expliquez le cascade de modèle: bon marché-premier avec vérification de la confiance, escalade sur la confiance faible.
- Enumérer les quatre signaux de routage (classification des tâches, longueur rapide, intégration de la similitude avec un ensemble connu de dur, confiance en soi à partir du premier passage).
- Calculer le coût combiné attendu à la fraction de routage cible et la tolérance aux pertes de qualité.
- Nombre de mesures de surveillance de la dérive (portée de qualité en ligne) qui prennent le modèle bon marché.

## Le problème

Votre service coûte 80 000 $ par mois sur GPT-5. Vos analyses montrent que 70% des questions sont simples: "qu'est-ce que l'heure est à Paris?" "réphrasez cette phrase". Un modèle de classe Haiku les gère parfaitement à 3% du coût. 30% ont besoin du raisonnement de GPT-5.

Si vous roulez les 70% vers le bon marché et 30% vers le coûteux, votre facture diminue de 65% à la même qualité du produit.

## Le concept

### Quatre signaux de routage

1. **Task classification**: simple/complexe/codegen/math/chat. Peut être un classifiant basé sur des règles, un petit LLM (Haiku-class à 0,25 $/M), ou intégrer la similitude avec des seaux étiquetés.

2. **Prompt length**Les commandes <500 de marque n'ont généralement pas besoin de frontière pour la cohérence.

3. **Embedding similarity to known-hard set**: si la requête est proche (cosine > 0,88) d'un seau de dureté connue, escalader directement à la frontière.

4. **Self-confidence from first-pass**Si les tests de logs du modèle montrent une faible confiance OU s'il refuse OU les sorties de langage de couverture, réessayez à la frontière.

### Trois modèles

**Pre-route**(classifiateur en avant): ~5-10ms de latence ajoutée; plus rapide dans l'ensemble.

**Cascade**(primaire moins cher, augmentation sur faible confiance): ~ 1,2x la latence médiane (exécution moins chère plus vérification), ~ 2x sur augmentation.

**Ensemble route**(exécution à bas prix et frontière en parallèle pour un échantillon, choix de modèle de récompense): la plus haute qualité, le coût le plus élevé; utilisation uniquement pour des A/B critiques.

### Mise en œuvre

Les passerelles d'IA (phase 17 · 19) exposent le routage.`router`Portkey a des gardiens + routage. Kong AI Gateway a un routage basé sur des plugins. Le marché de modèle OpenRouter expose une API de recommandation.

Le code de base est basé sur le code de base.

### La courbe des prix de 2026

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

La plupart des améliorations sont en service de l'efficacité  les leçons de base de la phase 17 · 04-09 ont été transformées en baisses de coûts du côté du fournisseur.

### La dérive est le risque réel

Votre route envoie 40% au modèle bon marché. Au cours de six mois, la répartition des tâches change (les utilisateurs deviennent plus sophistiqués, posent des questions plus longues). Le routeur ne remarque pas parce que son classifiateur a été formé sur les données du premier trimestre. La qualité diminue silencieusement. Personne ne se plaint assez fort. Vous découvrez dans un benchmark concurrentiel que vous avez perdu.

Route de passerelle selon les mesures de qualité en ligne:

- Les pouces des utilisateurs vers le haut/ vers le bas par route.
- Juge automatique de la LLM sur un échantillon de détention (5%) par route.
- Taux d'escalade: si la cascade monte de plus de 30%, le modèle bon marché est sur-routé.
- Taux de refus par route.

### Les chiffres que vous devriez vous rappeler

- 2026 économie de routage à iso-qualité: 20 à 60% d'études de cas.
- Réduction du prix des LLM 2022-2026: ~ 10 fois par an.
- Niveau GPT-4 2022 contre 2026: ~$20/M → ~$0,40/M.
- Impact de la latence en cascade: ~ 1,2x la médiane, ~ 2x l'escalade (~ 10% du trafic).

```figure
model-cascade-router
```

## Utilisez-le

`code/main.py`Les données de l'étude de la Commission sont fournies par le rapport de l'Union européenne, qui a été établi en décembre 2014.

## La faire partir

Cette leçon produit `outputs/skill-router-plan.md`- En raison de la charge de travail et du budget de qualité, choisit un modèle de routage et des signaux.

## Exercices

1. On court .`code/main.py`À quel étage de précision la cascade va-t-elle avant le trajet ?
2. Votre base d'utilisateurs est de 30% d'entreprise (queries complexes), 70% de niveau gratuit (simple).
3. Une route réduit la qualité de 2% mais économise 40%.
4. Mettre en œuvre un contrôle de confiance en utilisant des logprobs d'API OpenAI / Anthropic.
5. Au cours de six mois, le taux d'escalade passe de 8% à 22%.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## Pour en savoir plus

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://web.archive.org/web/20260413143335/https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) passerelle multimodale avec des primitifs de routage.
