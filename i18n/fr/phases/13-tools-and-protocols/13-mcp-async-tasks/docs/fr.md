# Extension des tâches du MCP: travail durable sur un noyau sans État

> Le MCP sans État ne signifie pas que chaque opération doit être terminée dans une seule demande. L'extension officielle Tasks donne à un travail de longue durée une poignée durable explicite. Un serveur peut retourner cette poignée à partir de `tools/call`, n' importe quelle instance peut répondre `tasks/get`, et les données du client arrivent par le biais de `tasks/update`sans réanimer les sessions de protocole.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 11 (stateless MRTR), Phase 13 · 12 (elicitation)
**Time:** ~90 minutes

## Objectifs d'apprentissage

- Distinguer le transport de protocole sans état de l'état de tâche d'application durable.
- - On négocie`io.modelcontextprotocol/tasks`l' extension des capacités par demande et `server/discover`- Je suis désolé .
- Retourner une adresse serveur `CreateTaskResult`avec `resultType: "task"`seulement après une création durable.
- Enquête avec `tasks/get`, remplir les tâches de commande avec `tasks/update`, et demander l' annulation de la coopération avec `tasks/cancel`- Je suis désolé .
- Retirez les plus âgés .`tasks/status`- Je suis là .`tasks/result`, et `tasks/list`- les hypothèses.
- Abonnez-vous aux notifications de tâches facultatives via `subscriptions/listen`sur un flux SSE de réponse POST.
- Expiration de la tâche modèle, redémarrage de la récupération, déduplication de la clé d'entrée et erreurs d'exécution correctement.

## Pourquoi les tâches sont une extension

Les tâches ont été introduites pour la première fois comme une fonction de base expérimentale en 2025-11-25.`io.modelcontextprotocol/tasks`L'extension afin que les clients et les serveurs puissent opter pour le cycle de vie supplémentaire sans étendre le protocole de base pour tout le monde.

La spécification d'extension reste une surface de projet même si c'est la maison officielle actuelle de Tasks.

Utiliser une tâche lorsque l'opération présente une ou plusieurs des propriétés suivantes:

- Il peut survivre à une période de temps ordinaire de demande.
- Une file d'attente ou un système d'emploi externe possède déjà l'exécution.
- Le client doit se remettre après son propre redémarrage.
- L'opération s'arrête pour l'entrée de l'utilisateur ou du modèle pendant l'exécution.
- L'annulation et la récupération durable des résultats sont des exigences du produit.

Ne créez pas une tâche pour une recherche déterministe bon marché. Une manche, la persistance, les sondages, l'expiration et l'annulation sont une vraie complexité.

## Le code de base des statuts, application de l'État

Le MCP 2026-07-28 est retiré `initialize`- Je suis là .`notifications/initialized`, les sessions de protocole, et `Mcp-Session-Id`- Ce n'est pas interdit.

Un id de tâche est l'état explicite de l'application:

- Le serveur le persiste avant de le renvoyer.
- Le client peut le stocker et le refaire après le redémarrage.
- L'identifiant peut être redirigé vers n'importe quelle copie soutenue par le même magasin durable.
- L'autorisation est vérifiée sur chaque méthode de tâche.
- L'expiration et la suppression sont définies par des champs de tâches, et non par une durée de vie du transport.

Cela est fonctionnellement différent de l'état caché attaché à une connexion.

Gardez quatre vies séparées:

| State | Lifetime | Where it belongs |
|---|---|---|
| Protocol metadata | One request | `params._meta`, validated again on every call |
| Transport work | One stdio request or HTTP response | In-flight coordinator with a bounded deadline |
| MRTR continuation | One retry sequence | Integrity-protected `requestState`, plus replay controls when needed |
| Durable task | Across requests, replicas, restarts, and reconnects | Shared application store keyed by an authorized `taskId` |

Le déplacement d'un enregistrement de tâche dans la mémoire de processus ne rend pas MCP état.`tasks/get`Persistez avant de retourner la poignée, puis faites en sorte que chaque méthode de tâche résolve le même enregistrement partagé sous les contrôles du locataire et du principal.

## Les négociations sur la capacité

Le client annonce un soutien à chaque demande éligible:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

Le serveur retourne exact `supportedVersions`, les capacités,`ttlMs`, et `cacheScope`de `server/discover`En effet, elle annonce des outils, elle implique aussi des obligations obligatoires.`tools/list`Ce résultat renvoie une déterministe`generate_report`descripteur, objet valide `inputSchema`- Je suis là .`resultType: "complete"`, les métadonnées d'identité du serveur, et les indices de cache public.

Une méthode de tâche d' un client qui n' a pas déclaré les retours d' extension `-32021`, la capacité requise du client manquant, avec `data.requiredCapabilities`à `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`Une chaîne de protocole non prise en charge est retournée .`-32022`avec exactitude `supported`et `requested`données; une version manquante ou non à chaîne est renvoyée `-32602`- Je suis désolé .

Une enveloppe sans JSON-RPC `id`est une notification. Le récepteur peut la traiter, mais elle n'émite pas de résultat JSON-RPC ou d'erreur. Un adaptateur HTTP par flux renvoie `202 Accepted`sans organisme pour une notification acceptée.

Pour le moment, seulement `tools/call`Conceptez votre abstraction interne afin que les types de requêtes futurs ne nécessitent pas de réécriture de stockage.

## Création de tâches dirigées par le serveur

Le vieux drapeau du client .`params._meta.task.required`Le client déclare le support d'extension, puis le serveur décide si une certaine`tools/call`devient une tâche.

La demande:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Réponse:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

Le serveur ne doit pas retourner cette poignée avant un `tasks/get`Dans un magasin éventuellement cohérent, attendez la visibilité de lecture avant de répondre. sinon un client peut recevoir un id valide et obtenir immédiatement " pas trouvé ".

Une réponse à la tâche n'est pas demandée dans le sens où le client ne demande pas le mode de tâche.

## La forme de la tâche

Chaque tâche comporte:

- `taskId`: identifiant stable généré par le serveur;
- `status`Le numéro de la liste:`working`- Je suis là .`input_required`- Je suis là .`completed`- Je suis là .`cancelled`ou `failed`Le dépôt de la commission
- `createdAt`et `lastUpdatedAt`: timestamps ISO 8601;
- `ttlMs`: durée d'expiration à compter de la création, ou `null`pour une limite non annoncée;
- optionnel `pollIntervalMs`: la cadence minimale suggérée des sondages du serveur;
- optionnel `statusMessage`: contexte d'utilisation ou de modèle.

Les champs spécifiques à l'état ne sont affichés que lorsqu'ils sont pertinents:

- `input_required`inclut `inputRequests`- Je suis désolé .
- `completed`inclut les documents de la demande initiale `result`la forme.
- `failed`inclut un JSON-RPC `error`- Je suis désolé.

Le client doit honorer .`pollIntervalMs`Un serveur peut limiter les taux de sondage plus agressif et modifier l'intervalle au cours de la durée de la tâche.

## Enquête avec `tasks/get`

Le client demande une photo d' instant:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

`tasks/get`Il est donc toujours le résultat.`resultType: "complete"`La tâche enlisée peut encore avoir`status: "working"`ou `status: "input_required"`- Je suis désolé .

Cette distinction empêche un bug parseur commun:

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

Il n' y a pas de`tasks/result`Quand la tâche est terminée, la prochaine`tasks/get`La réponse est conforme à l'original `CallToolResult`sous `result`- Le numéro de la liste:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

L' extérieur .`resultType`dit le `tasks/get`RPC terminé.`result.resultType`Il faut un discriminateur niché.`CallToolResult`Il doit aussi porter sa propre .`io.modelcontextprotocol/serverInfo`Cette leçon l'inclut au lieu de stocker une charge utile non typiée.

Il n' y a pas de`tasks/list`Les serveurs sans session ne peuvent pas en toute sécurité déduire quelles tâches appartiennent à une liste de connexion.

## Entrée lors de l'exécution de tâches

L'entrée de tâche et le MRTR de base sont similaires, mais utilisent des continuations différentes.

### Entrée nécessaire avant la création de tâches

Le noyau de retour `resultType: "input_required"`à partir de l' original `tools/call`Le client le remplit et tente à nouveau l'appel original.

### Entrée nécessaire après création de tâche

Définissez la tâche à `input_required`- Je suis là .`tasks/get`dévoile les faits remarquables `inputRequests`, et le client envoie des réponses par l' intermédiaire de `tasks/update`Le client ne réessaye pas l' original .`tools/call`- Je suis désolé .

Une photo:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

Mise à jour:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

La réponse à la réussite est une reconnaissance vide plus `resultType: "complete"`Le changement d'état peut être cohérent, donc le client continue à faire des sondages ou à écouter.

Chacun d' eux .`inputRequests`La clé doit être unique pour toute la durée de vie de la tâche.`tasks/get`Les instantanés peuvent afficher la même clé en suspens; les clients déduplicent l'interface utilisateur et les serveurs ignorent les réponses pour les clés inconnues, remplacées ou déjà remplies.`input_required`jusqu'à ce que toutes les clés requises soient répondues.

## L'annulation est coopérative

`tasks/cancel`Les travailleurs qui ont été arrêtés ne peuvent pas être autorisés à effectuer des travaux avant la fin de leur travail, à ne pas tenir compte de l'annulation ou de la transition.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Pour les trois méthodes de tâche,`Mcp-Name`Les miroirs`params.taskId`. Il ne répète pas le nom de la méthode JSON-RPC. `code/main.py`centraliser cette règle en `make_http_request`- Je suis désolé .

Le travailleur de leçon honore immédiatement l'annulation, faisant des appels répétés idempotents.

Ne pas utiliser `notifications/cancelled`Cette notification appartient à la demande d'annulation, pas à des tâches durables.

La distinction est importante à la limite de routage. L'annulation de requête cible une opération JSON-RPC en vol ou sa réponse HTTP à la demande.`tools/call`Il est déjà revenu .`resultType: "task"`La demande est complète et la fermeture de son transport ne peut pas nommer ou arrêter le travail durable. `tasks/cancel`Il est un nouveau RPC autorisé.`params.taskId`, reflète cette identité dans`Mcp-Name`, résolve le backend de la tâche, enregistre l'intention de l'annulation de la coopérative et renvoie un avis de confirmation sans prétendre que le travailleur s'est arrêté.

Une passerelle doit donc conserver les coordonnées des requêtes et les routes des tâches dans différentes tables. La table des requêtes peut disparaître lorsque la réponse est terminée. La route des tâches doit survivre jusqu'à l'expiration de l'état terminal et de la conservation. [Lesson 29: MCP Reliability, Cancellation, and Flow Control](../../29-mcp-reliability-cancellation-and-flow-control/docs/en.md)construit la course, le temps d'arrêt, l'idempotence, la contrainte et réessayez les règles pour les deux chemins.

## Notifications facultatives

Les sondages sont la base. Un client qui veut des mises à jour push envoie`subscriptions/listen`Pour Streamable HTTP, il s'agit d'un POST dont la réponse est un flux SSE à échelle de requête. Il n'y a pas de flux d'événements GET indépendant et aucune session de protocole à maintenir en vie.

Le serveur reconnaît les identifiants acceptés avec `notifications/subscriptions/acknowledged`et peut ensuite envoyer des instantanés complets à travers `notifications/tasks`- Le relevé et chaque notification de tâche`io.modelcontextprotocol/subscriptionId`dans `_meta`, égale à la `subscriptions/listen`chaque notification de tâche est équivalente à ce que `tasks/get`Il reviendrait à ce moment-là.

Les clients doivent toujours déclarer l'extension Tasks. Ils doivent se reconnecter et reprendre à partir d'id de tâche durables plutôt que de dépendre de la répétition d'événements ou `Last-Event-ID`- Je suis désolé .

## Sémantique de l'échec

Utilisez les deux couches d'erreur correctement.

### Erreur de protocole

Paramètres de méthode invalide ou un id de tâche inconnu renvoyer une erreur JSON-RPC, communément `-32602`. Des retours de soutien à l' extension manquants `-32021`avec l'objet de capacité requis.

### Résultat de l'exécution des tâches

- Un résultat normal avec `isError: true`est toujours une `completed`la tâche parce que l'appel à l'outil a produit son résultat défini.
- Une erreur JSON-RPC lors de l' exécution différée fait la tâche `failed`et stocke cette erreur JSON-RPC dans `error`- Je suis désolé .
- Le refus de l'utilisateur peut entraîner `cancelled`, un résultat de refus complet, ou un autre résultat sûr spécifique au domaine.

## Durable, expirant et propriétaire

Persistez au moins l'identifiant de tâche, le statut, les timestamps, ttl, l'intervalle de sondage, la propriété d'exploitation originale, le résultat ou l'erreur, les demandes de saisie en suspens et toutes les clés d'entrée émises.

La clé de stockage doit inclure ou résoudre un locataire et un principal autorisé.`tasks/get`- Je suis là .`tasks/update`- Je suis là .`tasks/cancel`, et souscription.

`ttlMs`Un client peut le traiter comme un backstop lorsqu'une tâche a cessé de produire des mises à jour observables. Un serveur peut échouer et supprimer une tâche expirée plus tard. Ne le décris pas comme une promesse de conserver un résultat fini pendant plusieurs millisecondes après sa réalisation.

Utilisez des écritures ou des transactions atomiques. La leçon écrit un fichier temporaire et le renomme atomquement. Un service multi-replica devrait utiliser un magasin durable partagé et un bail ou un contrôle de concurrences équivalent.

```figure
tp-task-lifecycle
```

## Faites-le

`code/main.py`met en œuvre un service de tâches déterministe:

- `server/discover`Retour `supportedVersions`, des indices de cache, et l'extension des tâches.
- `tools/list`renvoie une définition déterministe, cacheable `generate_report`Descripteur avec un schéma d'entrée valide.
- `tools/call`crée et persiste la tâche avant de revenir `resultType: "task"`- Je suis désolé .
- Une nouvelle instance de service recharge la même tâche, démontrant le redémarrage de la récupération.
- `tasks/get`renvoie des instantanés complets de la tâche.
- Le travailleur déménage de `working`à `input_required`- Je suis désolé .
- `tasks/update`accepte une réponse du formulaire et renvoie une confirmation complète vide.
- Le travailleur stocke un nid`CallToolResult`avec ses propres `resultType`et l'identité du serveur, puis des transitions vers `completed`- Je suis désolé .
- `tasks/cancel`est idempotent dans cette mise en œuvre.
- Les ensembles de constructeur HTTP `Mcp-Name`à `params.taskId`pour `tasks/get`- Je suis là .`tasks/update`, et `tasks/cancel`- Je suis désolé .
- Les aides à la notification utilisent `notifications/subscriptions/acknowledged`et `notifications/tasks`, tous deux marqués avec l'identifiant de demande d'écoute.
- Les notifications sans ID ne produisent aucune réponse JSON-RPC.

Le travailleur avance explicitement au lieu de dormir dans un fil de fond. Cela rend chaque transition d'état déterministe et garde l'exemple de protocole séparé de la mécanique de file d'attente.

## Utilisez-le

À partir de la racine du référentiel:

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

Sequence de résultats attendue:

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

Vérifiez aussi que `tasks/status`- Je suis là .`tasks/result`, et `tasks/list`méthode de retour non trouvée dans le service moderne.
Vérifiez ça .`tools/list`est déterministe et chaque méthode de tâche HTTP actuelle reflète son id de tâche à travers `Mcp-Name`- Je suis désolé .

## La faire partir

`outputs/skill-task-store-designer.md`Il produit maintenant une conception consciente de l'extension: négociation de capacités, création durable avant retour, méthodes actuelles, flux de mise à jour des entrées, propriété, expiration, annulation, abonnement et migration des méthodes expérimentales supprimées.

## Exercices

1. Ajoutez une deuxième clé d'entrée en attente. Envoyez une partie `tasks/update`et prouver que la tâche reste `input_required`jusqu'à ce que les deux clés soient répondues.
2. Ajouter la propriété du locataire au magasin et rejeter un identifiant de tâche valide présenté par le principal authentifié incorrectement.
3. Ajouter un contrat de location de travailleurs à expiration.
4. Implementer un adaptateur SSE de réponse POST pour `subscriptions/listen`. Ne pas ajouter GET, `Last-Event-ID`, ou un en-tête de session.
5. Ajouter le nettoyage d'expiration. Sélectionner une tâche expirée d'un identifiant de tâche malformé sans fuite d'existence entre les locataires.

## Les termes clés

| Term | Meaning in the current extension |
|------|----------------------------------|
| Tasks extension | Optional `io.modelcontextprotocol/tasks` capability for durable async work |
| `CreateTaskResult` | Server-directed `resultType: "task"` response to an eligible request |
| `tasks/get` | Poll a full current task snapshot, including terminal result or pending input |
| `tasks/update` | Submit responses to a task's outstanding `inputRequests` |
| `tasks/cancel` | Acknowledge cooperative cancellation intent |
| `input_required` | Task status indicating client input is outstanding |
| `pollIntervalMs` | Server-suggested minimum delay before another poll |
| `ttlMs` | Expiry duration measured from task creation |
| Durable-before-return | Rule that the task id must resolve before its handle is sent |
| `notifications/tasks` | Optional full task snapshot delivered on a subscribed SSE response |

## Compatibilité avec l'héritage

La surface expérimentale de 2025-11-25 utilisait l'augmentation des tâches demandées par le client, `tasks/status`- Je suis là .`tasks/result`, et facultatif `tasks/list`Un client actuel utilise la capacité d'extension, accepte les poignées dirigées par le serveur, les sondages`tasks/get`, fournit des informations avec `tasks/update`, et lit le résultat final de l'instantané de la tâche.

## Pour en savoir plus

- [Official MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
