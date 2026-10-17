# Fonction appelant plongée profonde  OpenAI, Anthropic, Gémeaux

> Les trois fournisseurs frontaliers ont convergé sur la même boucle d'appels d'outils en 2024 et ont ensuite divergé sur tout le reste.`tools`et `tool_calls`Utilisation anthropologique`tool_use`et `tool_result`Les Gémeaux utilisent`functionDeclarations`Cette leçon différencie les trois côtés de sorte que le code qui est envoyé sur un fournisseur ne se rompt pas lorsque vous le portez.

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Indiquez les trois différences de forme entre les charges utiles appelant la fonction OpenAI, Anthropic et Gemini (déclaration, appel, résultat).
- Translatez une déclaration d'outil sur les trois formats de fournisseur et prédisez où les contraintes de mode strict différeront.
- Utilisation `tool_choice`dans chaque fournisseur pour forcer, interdire ou sélectionner automatiquement les appels d'outils.
- Connaître les limites d'erreur par fournisseur (compte d'outils, profondeur de schéma, longueur d'argument) et les signatures d'erreur émises par chacun lorsqu'une limite est violée.

## Le problème

La forme d'une demande d'appel à fonction varie d'un fournisseur à l'autre.

**OpenAI Chat Completions / Responses API.**Tu passes .`tools: [{type: "function", function: {name, description, parameters, strict}}]`La réponse du modèle contient `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`où `arguments`est une chaîne JSON que vous devez analyser.`strict: true`) impose la conformité au schéma par décoding restreint.

**Anthropic Messages API.**Tu passes .`tools: [{name, description, input_schema}]`La réponse est:`content: [{type: "text"}, {type: "tool_use", id, name, input}]`- Je suis là .`input`est déjà analysé (un objet, pas une chaîne).`user`message contenant une `{type: "tool_result", tool_use_id, content}`Le bloc.

**Google Gemini API.**Tu passes .`tools: [{functionDeclarations: [{name, description, parameters}]}]`(néger sous `functionDeclarations`La réponse est:`candidates[0].content.parts: [{functionCall: {name, args, id}}]`où `id`est unique dans Gemini 3 et plus pour la corrélation parallèle.`{functionResponse: {name, id, response}}`- Je suis désolé .

La même boucle, différents noms de champs, différentes niches, différentes conventions de chaîne contre objet, différents mécanismes de corrélation.

Cette leçon construit un traducteur qui unit les trois formats en une déclaration canonique outil et des routes à la limite.

## Le concept

### La structure commune

Chaque fournisseur a besoin de cinq choses:

1. **Tool list.**Nom, description et schéma d'entrée par outil.
2. **Tool choice.**Forcer un outil spécifique, interdire les outils, ou laisser le modèle décider.
3. **Call emission.**Exit structuré nommant l'outil et les arguments.
4. **Call id.**Corréler la réponse à l'appel correct (matériaux pour le parallèle).
5. **Result injection.**Un message ou un blocage qui relie le résultat à l'appel.

### Différences de forme, champ par champ

| Aspect | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Declaration envelope | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema field | `parameters` | `input_schema` | `parameters` |
| Response container | `tool_calls[]` on assistant message | `content[]` of type `tool_use` | `parts[]` of type `functionCall` |
| Arguments type | stringified JSON | parsed object | parsed object |
| Id format | `call_...` (OpenAI generates) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| Result block | role `tool`, `tool_call_id` | `user` with `tool_result`, `tool_use_id` | `functionResponse` with matching `id` |
| Force-a-tool | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Forbid tools | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | schema-is-schema (always enforced) | `responseSchema` at request level |

### Les limites que vous atteindrez

- **OpenAI.**128 outils par requête. profondeur de schéma 5. chaîne d'arguments <= 8192 octets.`$ref`- Non , pas du tout .`oneOf`- Je suis là.`anyOf`- Je suis là.`allOf`avec chevauchement, chaque propriété répertoriée dans `required`- Je suis désolé .
- **Anthropic.**64 outils par demande. La profondeur du schéma est effectivement illimitée mais une limite pratique 10.
- **Gemini.**64 fonctions par requête. Les types de schéma sont un sous-ensemble OpenAPI 3.0 (légère divergence de JSON Schema 2020-12).

### `tool_choice`comportement

Trois modes que tout le monde prend en charge, nommés différemment.

- **Auto.**Le modèle choisit l'outil ou le texte.
- **Required / Any.**Le modèle doit appeler au moins un outil.
- **None.**Le modèle ne doit pas appeler les outils.

Plus un mode unique pour chaque fournisseur:

- **OpenAI.**Forcer un outil spécifique par nom.
- **Anthropic.**Forcer un outil spécifique par nom; `disable_parallel_tool_use`Le drapeau sépare le single versus le multi.
- **Gemini.** `mode: "VALIDATED"`enroute chaque réponse à travers un validateur de schéma, quelle que soit l'intention du modèle.

### Appels parallèles

Les ouvertures d'AIA `parallel_tool_calls: true`(par défaut) émet plusieurs appels dans un message d'assistant. Vous les exécutez tous et répondez avec un message de rôle d'outil en lots contenant une entrée par`tool_call_id`- Anthropic a toujours fait un appel unique .`disable_parallel_tool_use: false`(par défaut à partir de Claude 3.5) permet multi. Gemini 2 a permis des appels parallèles mais n'a pas donné d'identifiants stables; Gemini 3 ajoute des UUID afin que les réponses hors ordre se corrélations nettement.

### Retour en continu

Les trois appels supportent les appels en streaming.

- **OpenAI.**Les morceaux de delta de `tool_calls[i].function.arguments`Vous accumulez jusqu'à ce que`finish_reason: "tool_calls"`- Je suis désolé .
- **Anthropic.**Événements de démarrage/delta/arrêt de blocage. `input_json_delta`Les morceaux portent des arguments partiels.
- **Gemini.** `streamFunctionCallArguments`(nouveau dans Gemini 3) émet des morceaux avec un `functionCallId`pour que plusieurs appels parallèles puissent intervenir.

La phase 13 · 03 est consacrée à la réassemblage parallèle + en streaming.

### Erreurs et réparations

Les erreurs d'argument invalide sont également différentes.

- **OpenAI (non-strict).**Retours de modèle `arguments: "{bad json}"`Si votre analyse JSON échoue, vous injectez un message d'erreur et réappellez.
- **OpenAI (strict).**La validation se produit lors du décoding; JSON non valide est impossible mais `refusal`peut apparaître.
- **Anthropic.** `input`peut contenir des champs inattendus; le schéma est conseillé. Valider côté serveur.
- **Gemini.**La particularité de OpenAPI 3.0: `enum`sur les champs d'objets ignorés silencieusement; validez-vous.

### Le modèle de traducteur

Une déclaration canonique d'outil dans votre code ressemble à ceci (vous choisissez la forme):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

Trois petites fonctions le traduisent en trois formes fournisseurs.`code/main.py`Il est nécessaire de faire une mise en réseau, mais pas le format HTTP.

Les équipes de production enveloppent ce traducteur dans `AbstractToolset`(IA de l'Aï)`UniversalToolNode`(LangGraph), ou `BaseTool`La phase 13 · 17 expose une passerelle qui expose une API en forme d'OpenAI devant l'une des trois.

```figure
function-call-args
```

## Utilisez-le

`code/main.py`définit un canonique `Tool`Il analyse ensuite une réponse de fournisseur faite à la main de chaque forme dans le même objet d'appel canonique, démontrant que la sémantique est identique sous la peau.

À quoi regarder:

- Les trois blocs de déclaration ne diffèrent que par les noms de boîtes et de champs.
- Les trois blocs de réponse diffèrent selon l'endroit où l'appel se trouve (niveau supérieur `tool_calls`- Je suis là .`content[]`- Le bloc,`parts[]`entrée).
- Un .`canonical_call()`extraits de fonction `{id, name, args}`de toutes les trois formes de réponse.

## La faire partir

Cette leçon produit `outputs/skill-provider-portability-audit.md`. En raison d'une intégration d'appels à fonction contre un fournisseur, la compétence produit un audit de portabilité: quel fournisseur limite ses dépendances, quels champs doivent être renommés et quels sont les défaillances lorsqu'ils sont portés à l'autre fournisseur.

## Exercices

1. On court .`code/main.py`et vérifier que les trois JSONs de déclaration fournisseur tous sérialisent la même base `Tool`modifier l'outil canonique pour ajouter un paramètre enum et confirmer que seul le traducteur Gemini a besoin de gérer la quirk OpenAPI.

2. Ajouter un `ListToolsResponse`parser pour chaque fournisseur qui extrait la liste des outils un modèle renvoie après une `list_tools`OpenAI n'en a pas nativement un; notez cette asymétrie.

3. Mise en œuvre `tool_choice`Conversion: carte d' un canonique `ToolChoice(mode="force", tool_name="x")`dans les trois formes fournisseurs.`mode="any"`et `mode="none"`Vérifiez le tableau des différences.

4. Choisissez l'un des trois fournisseurs et lisez son guide d'appel à fonction de bout en bout.`strict`, Anthropic `disable_parallel_tool_use`, Gémeaux `function_calling_config.allowed_function_names`- Je suis désolé .

5. Écrivez un vecteur de test: un appel d'outil dont les arguments violent le schéma déclaré. Exécutez-le dans le validateur de chaque fournisseur (le stdlib dans le cours 01 fonctionnera comme un proxy) et enregistrer les erreurs de feu. Document que vous utiliserez dans la production pour la stricteur.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Provider-level API for structured tool-call emission |
| Tool declaration | "Tool spec" | Name + description + JSON Schema input payload |
| `tool_choice` | "Force / forbid" | Auto / required / none / specific-name modes |
| Strict mode | "Schema enforcement" | OpenAI flag that constrains decoding to match schema |
| `tool_use` block | "Anthropic's call shape" | Inline content block with id, name, input |
| `functionCall` part | "Gemini's call shape" | A `parts[]` entry containing name, args, and id |
| Arguments-as-string | "Stringified JSON" | OpenAI returns args as a JSON string, not an object |
| Parallel tool calls | "Fan-out in one turn" | Multiple tool calls in one assistant message |
| Refusal | "Model declines" | Strict-mode-only refusal block instead of a call |
| OpenAPI 3.0 subset | "Gemini schema quirk" | Gemini uses a JSON-Schema-like dialect with minor differences |

## Pour en savoir plus

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) référence canonique comprenant le mode strict et les appels parallèles
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) `tool_use`et `tool_result`sémantique de bloc
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) appels parallèles, identifiants uniques et sous-ensemble OpenAPI
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) La surface de l'entreprise de Gémeaux
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) Détails de l'application des régimes de mode strict
