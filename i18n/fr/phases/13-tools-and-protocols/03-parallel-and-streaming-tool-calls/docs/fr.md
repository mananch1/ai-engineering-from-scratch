# Appels parallèles et diffusion en continu avec des outils

> Trois recherches météorologiques indépendantes sont sérialisées. Exécutez-les en parallèle et le temps total s'effondre au plus lent appel unique. Chaque fournisseur frontalier émet maintenant plusieurs appels d'outils en un seul tour. Le paiement est réel; la plomberie est subtile. Cette leçon marche les deux faces: le ventilateur parallèle et le réassemblage des arguments en streaming, en mettant l'accent sur le piège de corrélation id.

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Expliquez pourquoi .`parallel_tool_calls: true`Il existe et quand désactiver.
- Corréler les blocs d'arguments en streaming à l'identifiant de l'appel d'outil droit pendant le ventilateur parallèle.
- Rassembler partiellement `arguments`les chaînes dans JSON complet sans parser tôt.
- Exécutez un indice météorologique de trois villes qui montre la latence séquentielle par rapport à la latence parallèle.

## Le problème

Sans appels parallèles, un agent répondant à " quel est le temps à Bengaluru, Tokyo et Zurich " fait ceci:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

Trois voyages aller-retour de la LLM, chacun payant également la latence de l'exécuteur.

Avec des appels parallèles:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

Un voyage aller-retour LLM. Le temps d'exécution est le maximum des trois, pas la somme. Les critères de référence de production sur OpenAI, Anthropic et Gemini montrent une réduction de 60 à 70% de la vitesse de la paroi sur les charges de travail de ventilateur.

Le prix est la complexité de la corrélation. Lorsque les trois appels sont terminés hors d'ordre, vos résultats doivent être conformes `tool_call_id`En effet, les résultats de l'exécution de la requête sont en partie des éléments de l'argument JSON, et les résultats sont en cours de diffusion.

## Le concept

### Permetant le parallèle

- **OpenAI.** `parallel_tool_calls: true`activé par défaut.`false`Pour forcer la série.
- **Anthropic.**Parallèlement via `disable_parallel_tool_use: false`(par défaut sur Claude 3.5 et plus).`true`Pour la série.
- **Gemini.**Toujours en parallèle .`tool_config.function_calling_config.mode = "AUTO"`Laissez le modèle décider.

Désactiver le parallèle lorsque les outils ont des dépendances d'ordre (`create_file`Alors ...`write_file`), lorsque la sortie d'un appel informe la sortie d'un autre ou lorsque le limitateur de fréquence ne peut pas gérer le ventilateur.

### Corrélation de l'ID

Chaque appel que le modèle émet a un`id`Chaque résultat que l'hôte renvoie doit contenir le même identifiant.

- **OpenAI.** `tool_call_id`sur chaque message de rôle d'outil.
- **Anthropic.** `tool_use_id`sur chacun `tool_result`Le bloc.
- **Gemini.** `id`sur chacun `functionResponse`(Gemini 3 et plus; Gemini 2 correspondant par nom qui a rompu pour les appels parallèles du même nom).

### Exécution d'appels simultanément

L'hôte exécute l'exécuteur de chaque appel sur son propre fil, coroutine ou travailleur à distance.`asyncio.gather`L'ordre de réalisation est imprévisible  l'identifiant est l'identifiant.

Un bug commun: répondre avec des résultats dans l'ordre de la liste d'appels au lieu de l'ordre de finalisation.`tool_call_id`, mais si un résultat est abandonné ou dupliqué, la soumission hors ordre rend le débogage plus difficile.

### Appels d' outils de diffusion

Quand le modèle est en streaming,`arguments`Trois flux séparés de pièces pour trois appels parallèles interférent sur le fil.

Forme par fournisseur:

- **OpenAI.**Chaque morceau est`choices[0].delta.tool_calls[i].function.arguments`La pièce porte`index`(position dans la liste des appels). Vous accumulez par index, lire `id`quand il apparaît pour la première fois, et parser JSON quand `finish_reason = "tool_calls"`- Je suis désolé .
- **Anthropic.**Les événements de streaming sont `message_start`, puis un .`content_block_start`par bloc avec type `tool_use`(contenant un identifiant, un nom, une entrée vide). `content_block_delta`événements de transport `input_json_delta`Des morceaux.`content_block_stop`ferme chaque bloc.
- **Gemini.** `streamFunctionCallArguments`(Gemini 3 et plus) émet des morceaux avec un `functionCallId`Avant Gemini 3, le streaming renvoyait un appel complet à la fois.

### JSON partiel et le piège de partage précoce

Vous ne pouvez pas analyser .`arguments`JSON partiel comme `{"city": "Beng`La bonne passerelle est le signal de fin d'appel du fournisseur: OpenAI `finish_reason = "tool_calls"`, Anthropic's `content_block_stop`C'est seulement après que tu es arrivé.`json.loads`Une approche plus robuste utilise un parseur JSON incrémentiel qui produit des événements à mesure que la structure est complétée; le guide de streaming d'OpenAI le recommande pour UX qui montre un indicateur de "pensation" en direct. Le comptage de brace est peu fiable en tant que test de complétude (les braces à l'intérieur des chaînes citées ou le contenu échappé provoquent des faux positifs) et ne doit être utilisé que comme un heuristique de débogage informel.

### Résultats hors commande

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

La réponse de l'hôte doit toujours citer les identifiants:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

L'ordre dans la réponse n'a pas d'importance pour la précision sur OpenAI ou Anthropic. Gemini accepte toute commande tant que les identifiants correspondent.

### Indice de référence: séquentiel par rapport parallèle

Le harnais dans `code/main.py`La séquence le fonctionne en 1800 ms total. Le parallèle le fonctionne en max ((400, 600, 800) = 800 ms. La différence est constante, pas proportionnelle, donc les économies augmentent avec le nombre d'outils.

Avertissement dans le monde réel: les appels parallèles stressent les API en aval. Une extension de 10 voies vers un service limité de taux échouera. La phase 13 · 17 couvre la contrainte au niveau du gateway; une nouvelle sémantique d'essai est prévue pour une phase future.

### Retour de ventilateur-horloge de mur

Si le modèle lui-même diffuse des flux, vous pouvez commencer à exécuter dès que les arguments d'un appel sont complets, plutôt que d'attendre que tous les appels soient terminés. Il s'agit d'une optimisation des documents OpenAI mais pas tous les SDK sont exposés.

```figure
tp-parallel-fanout
```

## Utilisez-le

`code/main.py`La première fonctionne trois appels météorologiques simulés en séquence et en parallèle en utilisant`concurrent.futures.ThreadPoolExecutor`La seconde moitié reproduit une fausse réponse de streaming `arguments`pour trois appels parallèles interleavés sur un flux  et les réassemble par id avec `StreamAccumulator`Pas de Master, pas de réseau, juste la logique de réassemblage.

À quoi regarder:

- Le temporiseur séquentiel atteint 1,8 seconde. Le temporiseur parallèle atteint 0,8 seconde sur les mêmes latences fausses.
- L'accumulateur gère les morceaux arrivant hors ordre en tamponnant par ID et en analysant seulement lorsque le JSON de chaque appel est complet.
- L'exécuteur démarre dès que les arguments d'un ID sont définis, pas après la fin de tous les flux.

## La faire partir

Cette leçon produit `outputs/skill-parallel-call-safety-check.md`. En raison d'un registre des outils, les audits des compétences permettent de comparer les outils qui sont sûrs, qui ont des dépendances de commande et qui dépasseraient les limites des taux en aval  renvoyer un registre révisé avec un outil `parallel_safe`Les drapeaux.

## Exercices

1. On court .`code/main.py`Confirmer que le rapport parallèle-sequentiel est approximatif `max/sum`(les courbes réelles deviennent légèrement de l'idéal en raison de la planification des fils, de la sérialisation et du coût de l'harmonisation).

2. Élargir l'accumulateur pour gérer un cas "appel a été annulé en milieu de cours de route" en laissant tomber son tampon et émettant un `cancelled`Quel fournisseur a explicitement documenté cette affaire ?`content_block_stop`La sémantique et les ouvertures d'AI `finish_reason: "length"`Le comportement.

3. Remplacez la piscine de fil par `asyncio.gather`Vous devriez voir de petites victoires sur asynchrony en raison du coût de commutation de contexte inférieur, mais seulement si les exécutants font de vraies opérations d'entrée/sortie.

4. Choisissez deux outils qui ne doivent pas être parallélisés (p. ex. `create_file`Alors ...`write_file`         `ordering_dependency`Il s'agit de la machine minimale pour la planification consciente de la dépendance, que formalise une future phase d'ingénierie des agents.

5. Lisez la section d'appels parallèles à la fonction d'OpenAI et celle d'Anthropic `disable_parallel_tool_use`Documents. Identifiez le type d'outil réel où Anthropic recommande de désactiver le parallélisme.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Parallel tool calls | "Fan-out in one turn" | Model emits multiple tool calls in a single assistant message |
| `parallel_tool_calls` | "OpenAI's flag" | Enable or disable multi-call emission |
| `disable_parallel_tool_use` | "Anthropic's inverse" | Opt-out flag; default is parallel enabled |
| Tool call id | "Correlation handle" | Per-call identifier the result message must echo |
| Accumulator | "Stream buffer" | Per-id string buffer for partial `arguments` chunks |
| Out-of-order completion | "Fastest first" | Parallel calls finish in unpredictable order; ids are the glue |
| Dependency graph | "Ordering constraints" | Tools whose outputs feed into inputs of other tools; cannot parallelize |
| Parse-early trap | "JSON.parse exploded" | Attempting to parse an incomplete `arguments` string |
| `streamFunctionCallArguments` | "Gemini 3 feature" | Streamed argument chunks with unique id per call |
| Completion-order reply | "Don't wait for all" | Reply with results as they arrive, keyed by id |

## Pour en savoir plus

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) comportement par défaut et le drapeau de désactivation
- [Anthropic — Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) `disable_parallel_tool_use`et le partage des résultats
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) Appels parallèles liés à l'id de Gémeaux 3
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) Rassemblement de arguments en morceaux pour les flux OpenAI
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) `content_block_delta`avec `input_json_delta`
