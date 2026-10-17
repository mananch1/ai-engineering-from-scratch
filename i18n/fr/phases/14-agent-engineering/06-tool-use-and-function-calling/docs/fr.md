# Utilisation des outils et appel à la fonction

> Toolformer (Schick et coll., 2023) a commencé à s'auto-surveiller en annotant les outils. Berkeley Function Calling Leaderboard V4 (Patil et coll., 2025) définit la barre 2026: 40% agent, 30% multi-tour, 10% en direct, 10% non-vivant, 10% hallucination.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## Objectifs d'apprentissage

- Expliquez le signal d'entraînement auto-surveillé de Toolformer: gardez les annotations de l'outil seulement lorsque l'exécution réduit la perte de jeton suivant.
- Nommez les cinq catégories d'évaluation de la BFCL V4 et les mesures que chaque catégorie prend.
- Implémenter un registre d'outils stdlib avec validation de schéma, coercition d'arguments et sandboxing d'exécution.
- Découvrez les trois problèmes ouverts de 2026: la chaîne d'outils à long horizon, la prise de décision dynamique et la mémoire.

## Le problème

L'utilisation des outils précoces a posé la question: le modèle peut-il prédire une appel de fonction correcte? l'utilisation des outils moderne demande: peut-il le modèle de chaîne d'outils sur 40 étapes, avec mémoire, avec observabilité partielle, avec récupération des défaillances des outils, sans halluciner des outils qui n'existent pas?

Les modèles peuvent apprendre à appeler des outils avec une autorégulation.BFCL V4 définit l'objectif d'évaluation de 2026 .L'écart entre eux est celui des agents de production spatiale dans lesquels vivent.

## Le concept

### Le groupe de travail de l'équipe de formation (Schick et coll., NeurIPS 2023)

Idée: laissez le modèle annoter son propre corpus de pré-entraînement avec des appels API candidats. Pour chaque candidat, exécutez-le. Gardez la annotation seulement si l'inclusion du résultat de l'outil réduit les pertes sur le jeton suivant.

Les outils couverts: calculateur, système d'évaluation de qualité, moteurs de recherche, traducteur, calendrier.

Les modèles plus petits sont touchés par les annotations des outils; les modèles plus grands gagnent. C'est pourquoi les modèles frontaliers 2026 ont une forte utilisation des outils, tandis que la plupart des modèles 7B nécessitent une mise en forme explicite de l'utilisation des outils pour être fiables.

### Le tableau de classement des fonctions de Berkeley V4 (Patil et coll., ICML 2025)

BFCL est l'évaluation de facto de 2026 .

- **Agentic (40%)** trajectories de l'agent complet: mémoire, multiples tours, décisions dynamiques.
- **Multi-Turn (30%)** des conversations interactives avec des chaînes d'outils.
- **Live (10%)** réelles demandes d'information envoyées par l'utilisateur (distribution plus difficile).
- **Non-Live (10%)** cas d'essai synthétiques.
- **Hallucination (10%)** détecter quand aucun outil ne doit être appelé.

V3 introduit une évaluation basée sur l'état: après une séquence d'outils, vérifiez l'état réel de l'API (par exemple: "le fichier a-t-il été créé?") plutôt que de correspondre à l'AST des appels de l'outil. V4 a ajouté des catégories de recherche Web, de mémoire et de sensibilité au format.

Résultats de la recherche: l'appel à la fonction à tour unique est presque résolu. Les défaillances se concentrent dans la mémoire (le transport de contexte à travers les tours), la prise de décision dynamique (le choix d'outils basé sur des résultats antérieurs), les chaînes à long horizon (départ après plus de 20 étapes) et la détection des hallucinations (rejet de passer un appel quand aucun outil ne convient).

### Schéma d' outil

Chaque fournisseur a un schéma. Ils diffèrent dans les détails mais partagent la même forme:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Utilisation anthropologique `input_schema`Directement.`function.parameters`Les deux acceptent le schéma JSON. Les descriptions sont portables  le modèle les lit pour choisir l'outil correct. Les descriptions d'outils malveillants sont la première cause de défaillances de choix de mauvais outils.

### Validation des arguments

Ne faites confiance à aucun appel d'outil.

1. **Type coercion.**Le modèle peut retourner une chaîne "5" où le schéma dit int. Forcer si non ambigu; rejeter si non.
2. **Enum validation.**Si le schéma dit:`status in {"open", "closed"}`et émissions de modèle `"in_progress"`, rejeter avec une erreur descriptive.
3. **Required fields.**champ requis manquant -> observation d'erreur immédiate de retour au modèle, pas un crash.
4. **Format validation.**Les dates, les courriels, les adresses URL  valident avec des parseurs de béton, pas avec des regex.

Chaque défaillance de validation doit renvoyer une observation structurée afin que le modèle puisse réessayer avec la forme correcte.

### Appels parallèles à l'outil

Les fournisseurs modernes prennent en charge les appels parallèles d'outils en un seul tour d'assistant.

1. Le modèle émet 3 appels d' outils avec des distincts `tool_use_id`S.
2. Le temps d'exécution les exécute (en parallèle si indépendant).
3. Chaque résultat revient comme un `tool_result`bloc corrélateur par `tool_use_id`- Je suis désolé .

Règle d'ingénierie: traiter les identifiants de corrélation comme des charges, les échanger et vous obtenez un routage de mauvais outil à mauvais résultat.

### La boîte à sable

L'exécution de l'outil est la limite de la boîte à sable. Voir leçon 09 pour plus de détails.`run_shell(cmd)`est un drapeau rouge; spécifique `git_status()`C'est plus sûr.

```figure
tool-routing
```

## Faites-le

`code/main.py`met en œuvre un registre des outils de forme de production:

- Validateur de sous-ensemble JSON Schema (stdlib uniquement).
- L'enregistrement de l'outil avec description, schéma d'entrée, délai et exécuteur.
- La coercition des arguments et la validation enum.
- Envoi parallèle d'outils avec identifiants de corrélation.
- Observations d'erreur en tant que chaînes structurées.

- Je vais le faire.

```
python3 code/main.py
```

La trace montre un mini agent appelant trois outils à tour de rôle, avec un appel délibérément malformé qui est rejeté avec une erreur descriptive sur laquelle le modèle peut agir.

## Utilisez-le

Chaque fournisseur a son propre schéma d'outils  Anthropic, OpenAI, Gemini, Bedrock. Utilisez une couche de traduction (OpenAI Agents SDK, Vercel AI SDK, adaptateur d'outils LangChain) si vous avez besoin de plusieurs fournisseurs.

## La faire partir

`outputs/skill-tool-registry.md`génère un catalogue d'outils, un schéma et un registre pour un domaine de tâche donné. Inclut des contrôles de qualité de la description (la description de chaque outil indique-t-elle au modèle quand l'utiliser?).

## Exercices

1. Ajouter un outil "no-op" qui permet au modèle de refuser explicitement d'utiliser tout autre outil. Mesurer sur un test d'hallucination similaire à BFCL.
2. Où la coercition commence-t-elle à cacher de vrais insectes ?
3. Ajouter un temps d'arrêt par outil et un disjoncteur (rejeter l'outil pendant 60 ans après 3 défaillances consécutives).
4. Lisez la description de BFCL V4. Choisissez une catégorie (par exemple "multi-turn") et exécutez 10 invites d'exemple par l'intermédiaire de votre agent.
5. Portes le validateur de stdlib à Pydantic ou Zod.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Structured-output tool invocation with validated schema |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — keep tool calls whose results reduce next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Function signature for the model" | name, description, JSON Schema of arguments |
| tool_use_id | "Correlation ID" | Ties a tool call to its result; essential for parallel dispatch |
| Hallucination detection | "Know when not to call" | V4 category: refuse to call when no tool fits |
| Argument coercion | "String-to-int repair" | Narrow fixes for predictable schema-mismatch; reject if ambiguous |
| Sandboxing | "Tool execution boundary" | Per-tool read/write surface, network, timeout, memory cap |

## Pour en savoir plus

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) annotation d'outils autosuvisés
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) référence d'évaluation 2026
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) Schéma d'outil de production dans le SDK Claude Agent
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) type d'outil de fonction et Garderrails
