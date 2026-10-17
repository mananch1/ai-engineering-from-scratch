# La fiabilité, l'annulation et le contrôle des flux des MCP

> Un identifiant de demande corrélate un message. Il ne rend pas un effet secondaire sûr, arrêter un travailleur, ou protéger un flux d'un consommateur lent.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 09 and 13
**Time:** ~120 minutes

## Objectifs d'apprentissage

- Implémenter le signal d'annulation correct pour stdio et Streamable HTTP.
- Résoudre les courses de finition et d'annulation sans envoyer de messages après l'annulation.
- Annulation de demande séparée de durable `tasks/cancel`la sémantique.
- Faites des décisions à nouveau à partir d'effets secondaires et de clés explicites d'idempotence.
- Réservez les files d'attente de progrès tout en conservant les réponses finales.
- Récupérer les flux en reconnectant, en réétablissant et en se remettant en colère.

## Le problème

Le chemin heureux cache les bugs les plus chers des systèmes distribués.

Un client appelle un outil. Le serveur commence à travailler. Un progrès arrive. Un proxy tamponne le flux. Le client atteint son délai et se déconnecte. Le serveur termine une milliseconde plus tard. Le client tente à nouveau avec un nouvel ID JSON-RPC. La mutation se déroule deux fois.

Chaque composant a fonctionné localement, le système a échoué à l'échelle mondiale.

MCP définit le comportement des messages et du transport, mais votre application possède toujours:

- les budgets de temps;
- l'indépendance des entreprises;
- les files d'attente délimitées;
- la classification des essais de réapprovisionnement;
- l'état de la tâche durable;
- Les États membres doivent également rétablir et rétablir des politiques.

Cette leçon construit ces décisions dans un simulateur déterministe.
Aucune prise, aucun défaut aléatoire.
Un test de fil synchronisé oblige deux clients de registre à rivaliser
pour la même clé d'indemnité.

## La demande d'annulation est spécifique au transport

L'intention est la même dans tous les transports: le client n'a plus besoin d'un résultat en vol.

### studio

stdio utilise un canal bidirectionnel partagé. Un client envoie une notification:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": 41,
    "reason": "User closed the operation"
  }
}
```

La notification est de feu et oublier. Le serveur n'émite aucune réponse JSON-RPC à elle.

Le serveur doit arrêter de travailler, libérer des ressources et éviter d'envoyer une réponse pour la demande annulée. Il peut ignorer l'annulation lorsque la demande est inconnue, terminée déjà ou ne peut pas être arrêtée en toute sécurité.

Les notifications d'annulation mal formées, inconnues et déjà terminées sont ignorées.

### HTTP par flux

Le client annule en fermant le flux de réponse de cette demande.

Ne pas poster `notifications/cancelled`pour une demande HTTP ordinaire.

Une fois que le serveur a observé la déconnexion, il doit cesser de fonctionner et ne pas envoyer de messages supplémentaires pour cette demande.

### L' annulation par le serveur est étroite

Un serveur n' utilise pas `notifications/cancelled`En studio, l'annulation par serveur est réservée à la fin d'une`subscriptions/listen`Gardez ce chemin séparé de l'annulation ordinaire des demandes de client.

## L'annulation est une course

Deux commandes d'événements sont valides.

### L'annulation gagne

```text
request starts
client sends cancellation signal
server marks request cancelled
worker reaches completion
server suppresses the response
```

### La fin gagne

```text
request starts
worker commits the result
server sends the response
cancellation arrives late
server ignores the late notification
```

Le client doit également ignorer une réponse tardive pour une demande qu'il a déjà abandonnée.

```figure
mcp-reliability-race
```

La leçon est `RequestCoordinator`stocke un état terminal. `complete()`une annulation tardive ne peut pas modifier un enregistrement terminé.

## Les temps de sortie nécessitent deux horloges

Un seul temporiseur d'inactivité ne suffit pas.

Utilisez deux limites:

1. **Idle timeout.**Combien de temps la demande peut ne pas produire d'activité utile.
2. **Maximum timeout.**Le budget absolu de l'horloge de mur à partir du début de la demande.

Le progrès peut réinitialiser l'horloge inactive.

```text
start: 0 ms
progress: 400 ms
progress: 800 ms
progress: 1200 ms
idle timeout: 500 ms
maximum timeout: 2000 ms
```

À 1500 ms, la demande est toujours active car la dernière progression n'a que 300 ms d'ancienneté.

Le progression est facultatif. Un serveur peut accepter un jeton de progression et ne pas émettre de mises à jour.

Les valeurs de progrès du MCP doivent augmenter. Les notifications s'arrêtent après la fin ou l'annulation.

## La demande d' annulation n' est pas `tasks/cancel`

Ces mécanismes résolvent différentes vies.

| Mechanism | Target | Signal | What success means |
|-----------|--------|--------|--------------------|
| Request cancellation on stdio | One in-flight RPC | `notifications/cancelled` | Client abandoned the request; server should stop if practical |
| Request cancellation on HTTP | One in-flight response stream | Close the stream | Client abandoned the request; server should stop if practical |
| `tasks/cancel` | One durable Task | Ordinary MCP request | Server acknowledged cancellation intent |

Un succès .`tasks/cancel`Le résultat ne prouve pas que le travailleur a arrêté.`working`jusqu'à ce qu'un poste de contrôle de travail observe le drapeau.

Ne supprimez pas l'état de tâche durable lorsque la connexion HTTP se ferme. La raison de créer une tâche est que son cycle de vie survit à une seule demande et à une seule connexion.

## Un nouvel identifiant JSON-RPC n'est pas idempotence

Les identifiants JSON-RPC corrélent les demandes et les réponses. Ils n'identifient pas une opération commerciale.

Supposons qu' un client dépose une charge avec un id `41`, perd la réponse, et tente à nouveau avec id `42`Le serveur voit deux messages différents. Sans une clé d'application, il ne peut pas savoir qu'ils représentent une seule facturation.

Une clé d'idempotence identifie l'intention d'entreprise:

```json
{
  "name": "charge_account",
  "arguments": {
    "account": "acct-7",
    "cents": 1200,
    "idempotencyKey": "checkout-7"
  }
}
```

Les serveurs stockent:

- la clé;
- une empreinte digitale des arguments d'opération;
- le résultat promis.

La même clé et les mêmes arguments renvoient le résultat stocké. La même clé avec des arguments différents est rejetée. Cela empêche une réutilisation accidentelle de la clé de muter une opération d'entreprise différente.

### La limite du registre doit être atomique et durable

Cette séquence est dangereuse:

```text
check key
run mutation
store result
```

Deux travailleurs peuvent observer une clé manquante et exécuter la mutation.
après l'effet mais avant le magasin crée la même ambiguïté lors de la réessay.

La leçon utilise un registre SQLite avec un fichier. `BEGIN IMMEDIATE`réalise la série
vérification de la clé, effet d'affaires simulé, compteur d'exécution et résultat stocké dans
Deux connexions de registre indépendantes en course avec la même clé
Il est donc nécessaire d'observer un résultat engagé et une exécution.
Le livre de l'homme garde ce dossier.

Chaque valeur retour est reconstituée à partir de JSON stockée.
l'objet mutable détenu par le registre, de sorte que le changement d'un dictionnaire retourné ne peut pas être effectué
corrupt les résultats de lecture ultérieures.

L'effet de travail du simulateur est le compteur de réception et d'exécution à l'intérieur du
Une transaction SQLite réelle, un déploiement ou un appel API externe est
La production ne doit pas être faite de manière atomique simplement en écrivant une table locale.
une transaction partagée dans une base de données, une boîte de réception transactionnelle ou un fournisseur en amont
Une serrure de processus seule ne protège pas
plusieurs réplices ou survivre à un redémarrage.

### Matrice de réessayer

Classifier les retries avant de les mettre en œuvre.

| Class | Example | Retry rule |
|------|---------|------------|
| Safe | Deterministic read with no side effect | Retry with a new JSON-RPC id after the failure boundary is understood |
| Conditional | Mutation with a durable idempotency key | Retry with the same key and identical arguments |
| Unsafe | Mutation without business deduplication | Do not retry automatically; reconcile first |

Annotations d' outils telles que `readOnlyHint`et `idempotentHint`Le contrat d'application et la mise en œuvre du serveur décident de la sécurité.

## La répression fait partie de la justice

Un producteur SSE peut générer des progrès plus rapidement qu'un client, un proxy ou un réseau ne peuvent les consommer.

Utilisez une file d'attente limitée et définissez ce qui peut être perdu.

La progression est remplaçable. Une valeur de progression ultérieure remplace une valeur antérieure pour le même token. Une réponse JSON-RPC finale n'est pas remplaçable.

Le tampon de leçons s'applique à cette politique:

1. Coalice les progrès adjacents pour le même token.
2. Arrêtez les progrès les plus anciens quand la capacité est atteinte.
3. Marquez le courant comme ayant besoin d'une rééducation.
4. Gardez la réponse finale.
5. Refuser un état où la préservation de la réponse finale nécessiterait la chute d'une autre réponse finale.

C'est une perte limitée avec une récupération explicite.

### Buffrage par procuration

Un serveur peut diffuser correctement tandis qu'un proxy inverse conserve des événements dans un tampon.

Pour une réponse SSE, envoyez:

```http
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

La spécification HTTP 2026 Streamable recommande `X-Accel-Buffering: no`les proxies compatibles livrent les événements immédiatement.

Pour les flux de longue durée silencieux, émettre périodiquement un commentaire SSE:

```text
:
```

Le client ignore les lignes de commentaires, les intermédiaires voient le trafic et sont moins susceptibles de fermer une connexion inactive.

Ne réinitialisez pas le délai d'arrêt sémantique d'une opération simplement parce qu'un commentaire de transport est arrivé.

## Reconnecter signifie refaire

Le système HTTP en continu moderne ne prend pas en charge le système SSE réalisable via `Last-Event-ID`- Je suis désolé .

Après une`subscriptions/listen`débits de courant:

1. Ouvrez une nouvelle demande d'écoute avec un nouvel identifiant JSON-RPC.
2. Retournez le filtre d'abonnement souhaité.
3. Remporter les outils, les ressources, les instructions ou les tâches affectés à partir de méthodes autorisées.
4. Déduplicer l'état de l'application par des identifiants stables.
5. Ne reproduisez pas une mutation dangereuse juste parce que sa réponse a été perdue.

Le plan de récupération de l' échantillon définit explicitement `sendLastEventId`Il y a des sources à réécrire.

### Éviter une reconnexion du troupeau

Si 10 000 clients se reconnectent exactement en une seconde, le serveur de récupération échoue à nouveau.

Utilisez un backoff exponentiel avec jitter et un cap. La leçon calcule le jitter déterministe à partir de l'identifiant du client et du numéro d'essai afin que les tests restent reproducibles:

```text
attempt 0: up to 250 ms
attempt 1: up to 500 ms
attempt 2: up to 1000 ms
...
cap: 8000 ms
```

La production peut utiliser la cryptographie sécurisée ou le temps de fonctionnement aléatoire.

## Faites-le

`code/main.py`construit cinq petits composants de fiabilité.

### `RequestCoordinator`

- démarre une demande en vol avec des délais vacants et maximaux;
- émet des notifications monotones de progrès;
- produit le signal d'annulation stdio ou HTTP correct;
- ignore les notifications d'annulation invalides;
- met en évidence les courses de terminaux d'annulation et d'achèvement;
- Réserve l'annulation par le serveur pour les abonnements à l'émission.

### `MutationLedger`

- démontre que deux identifiants JSON-RPC s'exécutent deux fois sans clé d'entreprise;
- utilise une transaction SQLite basée sur des fichiers pour la vérification des clés, l'effet simulé,
  compteur d'exécution et engagement de résultat;
- déduplicates des arguments correspondants sous une clé d'idempotence à travers des arguments indépendants
  connexions de registre;
- rejette une clé réutilisée avec des arguments différents;
- Retourne des copies défensives et préserve les dossiers engagés à travers la réouverture.

### `DurableTaskService`

- reconnaît une demande d'annulation;
- Il garde la tâche.`working`jusqu'à un poste de contrôle des travailleurs;
- démontre pourquoi la reconnaissance n'est pas définitive.

### `BoundedSseBuffer`

- se fusionnent ou diminuent les progrès sous pression;
- les documents indiquant qu'une réaffectation autoritaire est requise;
- ne laisse jamais tomber la réponse finale.

### Des aides à la récupération

- retourner les en-têtes SSE sûrs par procuration et les commentaires conservateurs;
- créer un plan de reconnexion et de réaménagement;
- répétition avec des déterminations exponentielles de back-off et de jitter.

## Utilisez-le

À partir de la racine du référentiel:

```bash
cd phases/13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/code
python3 main.py
python3 -m unittest discover tests -v
```

La démo est en cours de réalisation des deux côtés de la course centrale, exécute une transaction
une mutation déduplicée dans un registre temporaire avec fichier, surcharge une limite
Buffer de progression, et montre une tâche durable qui déménage de l'annulation reconnue
à l'annulation observée par les travailleurs.

## Laboratoire interactif

Faites quatre commandes d'événements sans ajouter de sommeil.

1. Commencez la demande `A`, annuler, puis appeler`complete()`- Je suis désolé .
2. Commencez la demande `B`, le compléter, puis livrer l'annulation.
3. Commencez la demande `C`, émettez des progrès avant chaque délai vacant, puis franchissez le délai maximum.
4. Commencez la demande `D`sur HTTP par flux et fermer son flux de réponse.

Enregistrement pour chaque scénario:

- l'état de la demande de terminal;
- si une réponse finale existe;
- le signal d'annulation placé sur le fil;
- Ce qui est un événement que le client doit ignorer.

Alors changez .`D`L'opération est identique, mais le signal d'annulation doit changer.

## Laboratoire de pratique

Ajouter un `reserve_inventory`mutation à `MutationLedger`- Je suis désolé .

Les exigences:

1. La clé lie le SKU, la quantité, le locataire et le nom de l'exploitation.
2. Une nouvelle tentative avec la même clé et les mêmes arguments renvoie la première réservation.
3. Une nouvelle tentative avec une quantité modifiée échoue sans autre réserve.
4. Une exécution qui a été commise mais a perdu sa réponse peut être réconciliée par clé.
5. Le résultat n'enregistre pas de données secrètes ou de paiement.
6. La réessayer automatique est désactivée lorsque le client n'a pas fourni de clé.
7. Ajoutez une décomposition simulée de l'abonnement et réinitialisez le dossier d'inventaire avant de décider de ce qu'il faut faire ensuite.
8. Démarrez deux connexions de registre à une barrière et soumettez la même clé
   - Une réserve a été faite.
9. Muter le premier objet de réservation retourné.
   le résultat stocké n'a pas changé.
10. Fermez et rouvrez le fichier de registre, puis réconcilier la réservation par clé.

Gardez le laboratoire honnête: si l'inventaire vit dans un autre service, expliquez si
que le service accepte la même clé d'idempotence ou qu'une boîte de réception transactionnelle
Les ponts de l'engagement local à l'effet à distance.

## Artéfacts expédiés

`outputs/skill-mcp-reliability-reviewer.md`Il est un outil de révision de fiabilité. Donnez-lui une opération MCP, le transport, la politique de délai, le comportement de retrait, la politique de file d'attente et le plan de récupération. Il renvoie un tableau de course, la classification de retrait, la limite d'idempotence, les contrôles de contrôle de débit et les dispositifs de défaillance.

## Vérifiez

La leçon est complète lorsque ces déclarations sont vraies:

- L' annulation du studio envoie `notifications/cancelled`et ne reçoit aucune réponse.
- L'annulation HTTP diffusée ferme le flux de demande et n'envoie pas de POST d'annulation.
- L'annulation avant la fin supprime la réponse finale.
- Le service complet avant annulation préserve la réponse et ignore la résiliation tardive.
- Le progrès peut réinitialiser le temps de travail, mais jamais le maximum.
- Un nouveau JSON-RPC id seul exécute la mutation à nouveau.
- Une clé d'idempotence et des arguments identiques exécutent une fois sous une concurrent
  Une course à deux connexions.
- Un enregistrement engagé survit à la réouverture et la répétition renvoie une copie défensive.
- La mutation d'un résultat retourné ne peut pas modifier le résultat stocké.
- Le tampon délimité reste dans la capacité et préserve la réponse finale.
- Reconnect utilise une nouvelle demande, ne l'envoie pas `Last-Event-ID`, et réétablit l'état affecté.
- `tasks/cancel`Le travail est effectué en fonction de la durée de la période de validité.

## Mode de défaillance de la production

| Failure | Observable symptom | Correct response |
|---------|--------------------|------------------|
| HTTP client POSTs cancellation notification | Server and client disagree about request lifetime | Close the request's SSE response stream |
| Server responds after accepted cancellation | Client receives an unusable late result | Stop work and suppress further messages when cancellation wins |
| Progress resets every deadline | Hung work survives forever | Keep a separate absolute maximum timeout |
| New RPC id treated as deduplication | Charge, deployment, or deletion runs twice | Add a durable application idempotency key |
| Key check and effect are separate | Concurrent workers both observe a missing key | Commit key claim, effect record, and result atomically |
| In-memory ledger used across replicas | Restart or another worker forgets prior commits | Use shared durable storage or upstream idempotency |
| Stored mutable result returned directly | Caller mutation corrupts later replays | Serialize committed results and return defensive copies |
| Key reused with changed arguments | One key aliases two business intents | Store and compare an argument fingerprint |
| Unbounded progress queue | Memory rises with a slow consumer | Coalesce and drop replaceable progress within a bound |
| Final response dropped under pressure | Client cannot know the request outcome | Reserve capacity or evict progress, never the final response |
| Proxy buffers SSE | Progress arrives in bursts or after timeout | Disable buffering and configure compatible proxy timeouts |
| `Last-Event-ID` assumed | Client resumes from state the server does not support | Reconnect with a new request and refetch |
| Every client reconnects immediately | Recovery creates another outage | Use capped exponential backoff with jitter |
| Task ack treated as final cancellation | Worker keeps running after UI says stopped | Poll the Task until a terminal status |

## Connexion Capstone

La pierre angulaire de l'écosystème d'outils devrait considérer la fiabilité comme une preuve exécutable et non comme un paragraphe dans un diagramme d'architecture.

Il faut les objets suivants:

- une transcription de course d'annulation pour chaque transport;
- une table de réessai pour chaque mutation exposée;
- un enregistrement de la clé d'idempotence et un dispositif de non-coïncidence;
- une transcription simultanée de la même clé, une vérification de réouverture et une vérification de mutation;
- un résultat de surcharge de tampon limité;
- les en-têtes SSE à reverse-proxy et la politique de l'inaction;
- un plan de reconnexion qui énumère les méthodes de réaménagement autorisées;
- une trace durable d'annulation de tâche lorsque la pierre angulaire utilise des tâches.

Une demande verte dans un processus local ne prouve que le bon chemin. La pierre angulaire est prête à la production lorsque les réponses perdues, l'annulation tardive, les consommateurs lents et les troupeaux reconnectés ont des résultats déterministes.

## Les termes clés

| Term | Meaning |
|------|---------|
| Request cancellation | Abandonment of one in-flight MCP request |
| Cancellation race | Competition between terminal completion and cancellation events |
| Idle timeout | Limit since the last useful request activity |
| Maximum timeout | Absolute limit from request start, unaffected by progress |
| Idempotency key | Application identifier that deduplicates one business intent |
| Atomic ledger | Durable boundary that commits the key claim, effect record, and result as one unit |
| Backpressure | Control applied when producers outpace consumers |
| Progress coalescing | Replacing older progress with a newer authoritative value |
| Refetch | Reading current state again after a stream gap |
| Jitter | Deliberate variation that spreads retries across time |

## Pour en savoir plus

- [MCP Cancellation](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/cancellation)
- [MCP Progress](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/progress)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Tasks Extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
