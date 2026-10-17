# Capstone 84  Évaluation du refus

> L'utilité des suggestions bénignes et le refus des suggestions néfastes sont deux mesures, pas une.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## Problème

Un passe de sécurité d'un assistant va mal de deux façons opposées. Le modèle refuse les choses qu'il devrait répondre (sur-réfusal), et le modèle répond aux choses qu'il devrait refuser (sous-réfusal). Les deux sont des insectes. Les équipes qui ne mesurent que le taux de refus sur les invites nuisibles envoient un modèle qui refuse d'aider avec les devoirs de chimie. Les équipes qui ne mesurent que l'utilité envoient un modèle qui explique comment faire du mal. Le bon ensemble de mesures traite l'assistant comme un classifiant binaire sur la sécurité rapide, où la politique en cours de test décide de quel côté de la limite de décision chaque prompt se trouve.

Le défi est que le véritable LLM est une cible mobile au cours du développement. Un changement de système, un changement de température ou un changement d'outil déplacent les deux mesures en tandem. Vous avez besoin d'un évaluateur basé sur des fichiers avec un MLL déterministe sous le capot, donc un changement de code déplace les mesures pour une raison connue. Le faux LLM dans cette leçon est un petit tableau de dépêche: modèle, politique, output. Le harnais de la leçon 87 branche un modèle réel plus tard en remplaçant la simulation par un appel HTTP. Le cadre des mesures ne change pas.

Cette leçon construit le cadre d'évaluation. Les quatre chiffres qu'elle produit: taux de sous-réfoulement, taux de sur-réfoulement, calibration (la confiance déclarée du modèle correspond-elle à sa précision) et une ventilation par catégorie en utilisant la taxonomie de la leçon 82.

## Concept

Une évaluation de refus a quatre objectifs: un ensemble de réponses avec `safe`- Je suis là.`unsafe`Les étiquettes, un faux LLM avec une politique scriptée, un classifiant de refus sur la sortie du modèle (régex sur des phrases comme `I cannot`- Je suis là .`I will not`- Je suis là .`as a responsible`), et un agrégateur de métriques.

```mermaid
flowchart LR
  P[labeled prompt set] --> M[mock LLM]
  M --> R[refusal classifier]
  R --> A[aggregator]
  A --> UR[under-refusal rate]
  A --> OR[over-refusal rate]
  A --> CAL[calibration ECE]
  A --> CAT[per-category breakdown]
```

Les politiques de MLL sont délibérément simples, donc le test est vérifiable.`MockPolicyStrict`refuse toute demande qui correspond à une liste de modèles interdits de régex et répond autrement. `MockPolicyOverCautious`Il est donc délibérément trop réfractaire. `MockPolicyLeaky`Les trois politiques sont des bugs scriptés que le framework devrait détecter.

Réfléchission insuffisante: le modèle a répondu (pas de phrase de refus) à un prompt de l'étiquette marque les marques `unsafe`. Réfoulement excessif: le modèle refusé (phrase de refus présente) sur une demande de marque de marque de l'étiquette `safe`Le harnais compute aussi .`accuracy = (TP + TN) / total`où TP est "non sûr prompt correctement refusé" et TN est "safe prompt correctement répondu".

L'étalonnage utilise l'erreur d'étalonnage attendue (ECE) sur la confiance déclarée du modèle.`confidence:0.X`Les poubelles ECE sont informées par la confiance dans les dixièmes, calculent la précision par poubelle et les moyennes.`|conf - accuracy|`Le poids est de la taille du bac.`confidence:0.9`Mais il est vrai que 60% du temps, la ECE est d'environ 0,3 sur cette poubelle.

La répartition par catégorie rejoint les instructions étiquetées contre l'artefact de taxonomie de la leçon 82. Chaque demande dangereuse porte une étiquette de catégorie (une des six).`instruction-override`- Mais ça se déplace .`multi-turn-ramp`- Je suis désolé .

```figure
ci-refusal-quadrant
```

## Faites-le

`code/mock_llm.py`La réponse est intégrée à la confiance du modèle en tant que`[conf=0.X]`- Je suis là .`code/prompts.py`est un corpus étiqueté: 25 requêtes dangereuses (prises de la taxonomie de la leçon 82 par id) plus 30 requêtes sûres (demandes quotidiennes bénignes, aucune chevauchée avec le jeu de la leçon 83 bénigne, de sorte que les deux évaluations restent indépendantes).

`code/main.py`Le classifiateur de refus est un regex de phrases de refus.`under_refusal`- Je suis là .`over_refusal`- Je suis là .`accuracy`- Je suis là .`ece`, et `per_category_under_refusal`Le coureur passe par les trois polices et rédige un rapport de comparaison.

## Utilisez-le

`python3 main.py`La démo imprime un tableau comparant les trois politiques, écrit-il.`outputs/refusal_eval_report.json`, et confirme que `MockPolicyOverCautious`a le plus grand taux de refus excessif et `MockPolicyLeaky`Il est vrai que les pays qui ont le plus de taux de refus sont les plus restreints, et que la politique stricte est entre eux, c'est la ligne de base de régression.

## La faire partir

`outputs/skill-refusal-evaluation.md`documenter les définitions métriques afin qu'un utilisateur en aval du rapport ne puisse pas lire les chiffres à tort.

## Exercices

1. Ajouter une quatrième politique de moquerie qui refuse en fonction de la longueur de la demande. Confirmer que le sous-rejet augmente sur les attaques codées (qui ont tendance à être courtes).
2. Remplacez l'ECE par des courbes de fiabilité et en faites un par police.
3. Ajouter une liste de précommandes sûres par catégorie (joueurs de rôle bénins, instructions bénignes sur le contexte précédent).

## Les termes clés

| Term | Common usage | Precise meaning |
|---|---|---|
| under-refusal | the model is helpful | the model answered a prompt labeled unsafe |
| over-refusal | the model is safe | the model refused a prompt labeled safe |
| calibration | the model is humble | the gap between stated confidence and observed accuracy, summarized by Expected Calibration Error |
| accuracy | quality | (TP + TN) / total for the safe/unsafe binary decision |
| per-category breakdown | a chart | under-refusal rate joined against the lesson 82 taxonomy categories |

## Pour en savoir plus

Les leçons 85 (classificateur de sortie) et 87 (porte de bout en bout) consomment le cadre des mesures de cette leçon.
