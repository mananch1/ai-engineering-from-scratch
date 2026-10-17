# L'interface des outils  Pourquoi les agents ont besoin d'une entrée/entrée structurée

> Un modèle de langage produit des jetons. Un programme prend des actions. L'écart entre ces deux est l'interface des outils: un contrat qui permet au modèle de demander une action et à l'hôte de l'exécuter.`tools/call`; Les pièces de tâche d'A2A  est un codage différent de la même boucle en quatre étapes.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## Objectifs d'apprentissage

- Expliquez pourquoi un LLM qui ne peut générer que du texte ne peut pas, par lui-même, prendre des mesures contre le monde réel.
- Dessinez la boucle d'appel d'outil en quatre étapes (décrivez → décidez → exécutez → observez) et nommez le propriétaire de chaque étape.
- Écrivez une description d'outil en trois parties: nom, entrée de schéma JSON et fonction d'exécuteur déterministe.
- Distinguer les outils purs et les outils ayant des effets secondaires et expliquer pourquoi la fraction est importante pour la sécurité.

## Le problème

Un LLM émet une distribution de probabilité sur le prochain jeton. C'est la surface de sortie entière. Si vous demandez à un modèle de chat " quelle est la météo à Bengaluru en ce moment ", il peut écrire une phrase plausible, mais il ne peut pas dial dans une API météo. La phrase peut être correcte par hasard ou trois jours de pérennité.

Le but de l'interface des outils est de combler ce fossé. Le programme hôte  votre agent runtime, Claude Desktop, ChatGPT, Cursor, ou un script personnalisé  annonce une liste d'outils appelables au modèle. Le modèle, lorsqu'il décide qu'une action est nécessaire, émet une charge utile structurée nommant un outil et ses arguments. L'hôte analyse la charge utile, exécute l'outil en réalité et renvoie le résultat. La boucle continue jusqu'à ce que le modèle décide qu'il n'y a plus besoin d'appels.

La première version de ce contrat a été expédiée en juin 2023 comme paramètre "fonctions" d'OpenAI.`tool_use`Les blocs dans Claude 2.1.`functionDeclarations`Un modèle de déploiement de l'outil JSON est utilisé pour la déploiement de l'outil.

La boucle de quatre étapes est l'invariable sous tout cela.

## Le concept

### Étape 1: décrire

L'hôte déclare chaque outil avec trois champs.

- **Name.**Un identifiant stable et lisible par machine.`get_weather`Pas "chose météo".
- **Description.**Un bref résumé en langage naturel en un paragraphe. "Utilisez-le lorsque l'utilisateur demande les conditions actuelles d'une ville spécifique.
- **Input schema.**Un objet de schéma JSON (projet 2020-12) décrivant les arguments de l'outil.

Le modèle reçoit la liste. Les fournisseurs modernes sérialisent ces déclarations dans le système de demande en utilisant un modèle spécifique au fournisseur, de sorte que vous, en tant qu'appelant, ne traitez que du formulaire structuré.

### Deuxième étape: décider

Compte tenu du message de l'utilisateur et des outils disponibles, le modèle choisit l'un des trois comportements.

1. **Answer directly**Pas de appel à l'outil.
2. **Call one or more tools.**Émettez des objets d'appel structurés.`parallel_tool_calls: true`(par défaut sur OpenAI et Gemini, opt-in sur Anthropic) le modèle peut émettre plusieurs appels en un seul tour.
3. **Refuse.**Les sorties structurées en mode strict peuvent produire une`refusal`bloc au lieu d'appeler.

Une charge utile d' appel d' outil comporte trois champs stables: un appel `id`, un outil `name`, et un JSON `arguments`L'id existe pour que l'hôte puisse corréler le résultat ultérieur à l'appel spécifique, ce qui importe lorsque les appels parallèles sont en panne.

### Étape 3: exécuter

L'hôte reçoit l'appel, valide les arguments contre le schéma déclaré et exécute l'exécuteur. Les arguments invalides signifient que le modèle a halluciné un champ ou utilisé le type incorrect  un mode d'échec très courant sur les modèles faibles. Les hôtes de production font une des trois choses sur des arguments invalides: échouent rapidement et font surface à l'erreur du modèle, réparent le JSON avec un parseur restreint ou réessayer le modèle avec l'erreur de validation incluse dans le prompt.

L'exécuteur lui-même est un code ordinaire. Python, TypeScript, une commande shell, une requête de base de données. Il produit un résultat, qui est généralement une chaîne, mais peut être n'importe quelle valeur JSON ou un bloc de contenu structuré (texte, image ou référence de ressources en MCP). Le résultat doit être sérialisable.

### Étape 4: observer

L' hôte ajoute le résultat de l' outil à la conversation (comme `tool`message de rôle avec correspondance `id`Le modèle a maintenant l'outil de sortie dans le contexte et peut produire une réponse finale ou demander plus d'appels.

### La confiance s' est brisée

Les outils sont disponibles en deux saveurs qui sont importantes pour la sécurité.

- **Pure.**- Uniquement lisible, déterministe, sans effets secondaires.`get_weather`- Je suis là .`search_docs`- Je suis là .`get_current_time`- On peut appeler spéculativement.
- **Consequential.**Il mutera l'état, dépensera de l'argent, touchera les données des utilisateurs. `send_email`- Je suis là .`delete_file`- Je suis là .`execute_trade`- Il doit être fermé.

La "Règle des deux" de Meta pour la sécurité des agents 2026 indique qu'un seul tour peut combiner au plus deux de: entrées non fiables, données sensibles, action conséquente. L'interface de l'outil est l'endroit où vous mettez en œuvre cette règle  en rejetant les appels, en exigeant la confirmation de l'utilisateur ou en augmentant les champs d'application. Voir la phase 13 · 15 pour le chapitre complet de la sécurité et la phase 14 · 09 pour les politiques d'autorisation au niveau des agents.

### Où vit la boucle

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

Les noms des colonnes changent, mais pas la structure.

### Pourquoi ne pas demander au modèle d'émettre JSON ?

" Demandez au modèle de répondre en JSON " était le modèle d'appel pré-fonction. Il échoue de 5 à 15% du temps sur les modèles frontaliers et bien plus sur les modèles plus petits. Les modes d'échec comprennent des braces manquantes, des virgules en arrière, des champs hallucinés et des types erronés. Vous avez ensuite besoin d'un JSON réparation passe, une nouvelle tentative ou un décodeur restreint.

L'appel à la fonction native est préférable pour trois raisons. Tout d'abord, le fournisseur entraîne le modèle de bout en bout sur la forme exacte de l'appel, de sorte que le taux de validité JSON monte à 98 à 99% en mode strict. Deuxièmement, la charge utile de l'appel est située dans sa propre fente de protocole, pas à l'intérieur du texte libre  de sorte qu'un appel d'outil ne fuit jamais dans la réponse visible de l'utilisateur. Troisièmement, les fournisseurs imposent la conformité des schémas avec le décoding restreint (mode strict d'OpenAI, mode strict d'Anthropic `tool_use`, des Gémeaux `responseSchema`La validation de la production est garantie.

La phase 13 · 02 traverse les trois API fournisseurs côte à côte.

### Les interrupteurs de circuit

La boucle se termine lorsque le modèle cesse d'émettre des appels ou que l'hôte atteint un nombre maximum de tours. Les hôtes de production le fixent à 5 à 20 tours. Au-delà de cela, vous êtes presque certainement dans une boucle que le modèle ne peut pas sortir. Claude Code est par défaut à 20; OpenAI Assistants à 10; le mode agent de Cursor à 25.

L'alternative  boucles illimitées  apparaît tous les six mois comme "agent a dépensé 400 $ en appels API au cours de la nuit" post mortem.

La phase 14 · 12 couvre en profondeur la récupération des erreurs et l'auto-guérison; la phase 17 couvre les limites du taux de production.

### Où la phase 13 va d'ici

- Les leçons 02 à 05 polissent la surface de l'appel à l'outil au niveau du fournisseur.
- Les leçons 06 à 14 généralisent la boucle en MCP.
- Les leçons 15 à 18 défendent la boucle contre les serveurs hostiles, les utilisateurs adversaires et les surfaces auth distantes non authentifiées.
- Les leçons 19 à 22 étendent le modèle à la collaboration entre agents, à l'observabilité, au routage et à l'emballage.
- La leçon 23 propose un écosystème complet utilisant chaque primitif.

Chaque leçon restante est une élaboration de cette boucle en quatre étapes.

```figure
tp-tool-loop
```

## Utilisez-le

`code/main.py`Une fausse fonction " décideur " simule le modèle en correspondant des modèles sur le message de l'utilisateur; l'exécuteur, le validateur de schéma et l'harnais d'observation de la phase sont réels.

À quoi regarder:

- Le registre des outils contient trois champs par outil: nom, description, schéma et référence d'exécuteur.
- Le validateur est un sous-ensemble minimal de schéma JSON (types, requis, enum, min/max) écrit uniquement en stdlib.
- Les agents de production ont besoin de ce type de disjoncteur.

## La faire partir

Cette leçon produit `outputs/skill-tool-interface-reviewer.md`. Compte tenu d'un projet de définition d'outil (nom + description + schéma + contour d'exécuteur), la compétence l'audit pour la pertinence de la boucle: est-ce que le nom est stable en machine, est-ce que la description est un bref usage complet, est-ce que le schéma utilise correctement JSON Schema 2020-12, et est-ce que la classification pure-versus-consequentielle est explicite.

## Exercices

1. Ajouter un quatrième outil à `code/main.py`appelée `get_stock_price(ticker)`. Écrivez sa description comme "Utilisez lorsque l'utilisateur demande un prix actuel des actions par ticker. N'utilisez pas pour les prix historiques ou les résumés du marché".

2. Brisez le validateur de schéma.`arguments`l'objet manque un champ requis, et confirmez que l'hôte le rejette avant l'exécution. Ensuite, passez un appel avec un champ inconnu supplémentaire. Décidez: l'hôte doit-il rejeter ou ignorer? Justifiez votre choix avec un argument de sécurité.

3. Chaque outil de la harnais doit être classé comme pur ou conséquent.`consequential: true`le signal des entrées du registre qui en ont besoin, et changer la boucle pour imprimer une ligne "confirmerait avec l'utilisateur" chaque fois qu'un outil conséquent est choisi.

4. Dessinez la boucle en quatre étapes sur papier avec la table de colonne fournisseur ci-dessus remplie pour votre client préféré (Claude Desktop, Cursor, ChatGPT ou une pile personnalisée).

5. Lisez le guide d'appel à fonction d'OpenAI de haut en bas. Identifiez le champ qui se trouve dans la demande mais pas dans la boucle en quatre étapes comme présenté ici. Expliquez ce qu'il ajoute et pourquoi il est pratique plutôt que nécessaire.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tool | "A thing the model can call" | A triple of name + JSON-Schema-typed input + executor function |
| Function calling | "Native tool use" | Provider-level API support for emitting structured tool calls instead of prose |
| Tool call | "The model's request to act" | A JSON payload with `id`, `name`, `arguments` emitted by the model |
| Tool result | "What the tool returned" | The executor's output, wrapped in a `tool` role message with matching id |
| Parallel tool calls | "Many calls at once" | Multiple call objects in one model turn, independent and orderable by id |
| Strict mode | "Guaranteed JSON" | Constrained decoding that forces the model's output to validate against the declared schema |
| Pure tool | "Read-only tool" | No side effects; safe to re-run |
| Consequential tool | "Action tool" | Mutates external state; requires gate, audit, or user confirmation |
| Four-step loop | "The tool-call cycle" | describe → decide → execute → observe |
| Host | "Agent runtime" | The program that holds the tool registry, calls the model, and runs the executor |

## Pour en savoir plus

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) référence canonique pour les déclarations d'outils et les formes d'appels à l'aide de l'OpenAI
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)- C'est le cas de Claude.`tool_use`- Je suis là .`tool_result`format de bloc
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) `functionDeclarations`et la sémantique parallèle dans Gemini
- [Model Context Protocol — Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) la généralisation actuelle sans état et agnostique des fournisseurs de l'interface des outils
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) le dialecte du schéma chaque API d'outil moderne parle
