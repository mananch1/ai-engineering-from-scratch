# Entrée des PPC sans État et du registre

> Un gateway doit rendre chaque route explicite. Le protocole 2026-07-28 lui donne des limites de méthode, nom, version, capacité, identité, cache et trace sans séance de transport.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 15 (security), Phase 13 · 16 (authorization)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Rassembler plusieurs serveurs MCP derrière un seul point final 2026-07-28 sans affinité de session.
- Valider les métadonnées et les en-têtes de routage par requête avant la politique ou la transmission.
- Fusion d'outils avec des espaces de noms stables, ordre déterministe, broches descripteurs, RBAC et caching privé.
- Traitez les enregistrements du registre comme des preuves de découverte qui nécessitent toujours une politique d'admission.
- SSE à l'échelle de la demande de route, `subscriptions/listen`, MRTR retries, et les appels d'extension des tâches correctement.
- Isolez les poignées de main et le support de session de la voie moderne.

## Le problème

Connecter un client directement à un serveur est simple.

- Quels serveurs sont autorisés ?
- Quel directeur peut voir et appeler chaque outil ?
- Que se passe-t-il quand deux arrière-plan exposent le même nom ?
- Comment les modifications apportées aux descripteurs sont-elles examinées?
- Où sont appliquées les limites de taux et les événements d'audit?
- Une instance peut-elle répondre à la prochaine demande ?

Un gateway se trouve entre les clients et les serveurs MCP de base. Il présente un point final MCP, applique des politiques transversales et transmet les demandes approuvées.

Les conceptions de passerelles plus anciennes multiplie souvent une session client en plusieurs sessions backend et réécrit `Mcp-Session-Id`Le noyau 2026-07-28 n'a pas de sessions protocoles.

## Le concept

### Le chemin de la porte moderne

Pour chaque demande:

1. Identifier le principal de l'autorisation de transport.
2. Valider`MCP-Protocol-Version`- Je suis là .`Mcp-Method`- Je suis là .`Mcp-Name`, et `params._meta`- Je suis désolé .
3. Autoriser le principe, la ressource, la méthode, l'outil et les arguments.
4. Appliquer la politique de description, de registre, de taux et de données.
5. Créer une nouvelle demande autonome pour le backend sélectionné.
6. Valider le résultat de l'arrière-plan et retourner un résultat de passerelle.
7. Enregistrer un événement d'audit sans enregistrer des secrets.

Aucune étape n'a besoin d'une session de protocole cachée. L'état d'application peut toujours exister dans les bases de données, les poignées explicites, les tâches ou l'état MRTR protégé par l'intégrité.

### La politique de mise en œuvre est la principale décision de la passerelle

L'admission décide de la version de backend qui peut entrer dans le gateway. Il n'autorise pas un appel en direct. Pour chaque demande, le gateway recompte la politique du principal authentifié, de l'émetteur et de la ressource, du locataire, de la méthode et du nom correspondants, des arguments normalisés, de la pin d'expression admise, de la santé actuelle du backend, de l'intersection des capacités, de la classification des données, de l'état des tarifs et de toute approbation liée à l'action.

Un enregistrement de registre peut rester actif pendant que le rôle d'un utilisateur est révoqué. Un descripteur peut rester coincé pendant qu'un argument de destination franchit une limite du locataire. Un backend peut rester approuvé pendant que la politique d'incident met en quarantaine les appels en changement d'état. La politique de fonctionnement est donc la principale décision de permettre ou de refuser, avec le registre et la preuve de descripteur comme entrées.

Ne cachez pas une décision d'autoriser sous une connexion ou un identifiant de session supprimé. Si la politique n'est pas disponible, suivez une politique de défaillance déclarée par classe d'opération. Une défaillance sécurisée consiste à ne pas fermer pour les changements d'état et les lectures sensibles, tandis que les voies de lecture publiques explicitement approuvées ne peuvent utiliser une politique de dernière connaissance de courte durée que lorsque leur modèle de risque le permet. Enregistrer quelle version de politique et quel chemin d'échec a pris la décision, puis valider le résultat de l'arrière-plan avant de le retourner.

### Un point d'extrémité POST

Le protocole HTTP est utilisé pour envoyer chaque message JSON-RPC via POST:

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

La passerelle peut retourner JSON ou SSE scopé par requête pour ce POST. GET et DELETE retourner 405 pour les demandes modernes. `Mcp-Session-Id`et `Last-Event-ID`ne créent pas d'autorité, d'affection ou de comportement répétitif.

Les valeurs de titre et de corps doivent être d'accord.`-32020`Cela permet aux équilibrateurs de charge, aux passerelles et aux limitateurs de vitesse de router sans analyser tout le corps tout en préservant l'intégrité de bout en bout.

Valider dans un ordre exact: JSON-RPC et types de métadonnées, égalisation d'en-tête et de corps, puis prise en charge de la version correspondante.`-32020`Si l'en-tête et le corps conviennent d'une version non prise en charge, renvoyez HTTP 400 avec `-32022`et `data`Exactement .`{"supported":["2026-07-28"],"requested":"<actual>"}`Une méthode inconnue renvoie HTTP 404 avec `-32601`- Je suis désolé .

`ProtocolError`porte facultatif `data`, et la passerelle la sérialise dans l'objet d'erreur JSON-RPC.`id`Une notification HTTP acceptée renvoie 202 avec un corps vide.

### Implémenter la découverte à chaque couche

La passerelle s' implique `server/discover`Il découvre chaque backend afin de connaître les versions de protocole, les capacités et les extensions.

Exemple de résultat de la passerelle:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

Une fonctionnalité de backend n'est pas automatiquement sûre d'exposer. Une fonctionnalité de gateway sans chemin de backend n'est pas utile pour la publicité.

`serverInfo`Les données de diagnostic et d'affichage sont auto-déclarées.

### Les capacités du client par demande

Chaque demande transmise a besoin d' un courant `_meta`enveloppe:

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

Ne copiez pas aveuglément les capacités du client externe à un backend. La passerelle est le client du backend. La publicité est uniquement des fonctionnalités que la passerelle médiera correctement.

### L'espacement des noms déterministe

Fusion des outils de backend sous des noms publics stables:

```text
notes.search
notes.create
issues.list
issues.open
```

Gardez une carte du nom public au nom de l'outil d'arrière-plan et original. Ne choisissez jamais la première ou la dernière collision. Un nom public fait partie du contrat d'approbation et d'audit, donc le changer est une migration.

`tools/list`Les résultats de l'analyse de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur`cacheScope: private`- Un délimité .`ttlMs`réduit la charge de découverte de backend sans permettre à une liste spécifique à l'utilisateur de fuir dans les contextes d'autorisation.

Chaque descripteur d'outil exposé comprend un nom stable, une description et une racine d'objet `inputSchema`. L'espacement de noms ne peut pas supprimer les champs de descripteur requis.`resultType`, les métadonnées d'identité du serveur, et des indices de cache.

### Des descripteurs approuvés par le pin

Au moment de l'admission, canonizez le descripteur complet et stockez son digeste sous le nom public qualifié.

Si elle change:

- Retirez-le de `tools/list`- Je suis désolé .
- Rejetez les appels directs.
- - Émettez un audit.
- Requérir une nouvelle approbation politique ou humaine avant de mettre à jour le pin.

Une passerelle est un point central d'application utile, mais elle ne transforme pas un descripteur à première vue en un descripteur sûr.

### Les registres aident à découvrir, pas à décider

Un registre`server.json`Les données de base sont fournies par le service de la publication.

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

Les métadonnées de publication ne portent pas la décision de sécurité du portail.

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

La passerelle vérifie le `server.json`La porte d'entrée a encore besoin d'une politique d'admission.

Pour chaque backend admis, enregistrer:

- Identifiant le registre et le dossier.
- L'espace de noms ou la preuve de domaine vérifiés par l'éditeur.
- Le transport et le point de destination sont autorisés.
- Version en pin ou politique de mise à niveau approuvée.
- Digestion d'un objet ou d'un descripteur.
- Émetteur et ressource de l'autorisation.
- Réviseur, heure d'approbation et expiration.

N'acceptez pas un serveur parce que son nom d'affichage ressemble à un produit familier. Ne traitez pas la présence du registre comme une revue de sécurité opérationnelle. Les serveurs privés peuvent être admis à travers le même schéma de preuve même lorsqu'ils ne figurent jamais dans un registre public.

Cette leçon met en œuvre la solution de la porte d'entrée: joindre les preuves de publication à l'admission locale avant qu'un backend ne devienne routable. [Lesson 30: MCP Registry Supply Chain, Admission, Drift, and Rollback](../../30-mcp-registry-supply-chain-and-drift/docs/en.md)construit le plan de contrôle complet pour la preuve exacte de l'espace de noms, l'origine des artefacts, les broches immuables, la dérive du descripteur en direct, la réconciliation de l'état du registre, un registre d'admission équivautant à des manipulations et un retour à l'emploi fondé sur des preuves.

### Médiation des accréditations

La passerelle authentifie ses appelants et authentifie séparément les backends.

Gardez ces obligations explicites:

```text
outer principal -> gateway role and policy
backend issuer + resource -> backend registration and token
```

Ne jamais passer le jeton de passerelle externe à un backend. Ne jamais réutiliser un jeton backend à un émetteur ou une ressource différent. Si un outil agit au nom d'un utilisateur final, préserver cette délégation avec un modèle d'échange ou de revendications conçu plutôt que de se faire passer pour l'utilisateur avec une carte de crédit de service partagée.

### Limits de taux sans séances

Les limites clés par principal authentifié, émetteur, ressource, outil public, classe de coûts et fenêtre de temps.

Avant de consommer des travaux coûteux, appliquez une validation peu coûteuse et décidez si les appels refusés comptent pour les limites d'abus, les quotas d'affaires ou les deux.

### Audit de la chaîne de décision

Enregistrer suffisamment pour reconstruire un appel:

- Identifiants de demande et de trace.
- Principaux et émetteur authentifiés.
- Un outil public et une route de backend.
- Une version de pin descripteur.
- Décision politique et raison.
- La classe de latence et de résultats.
- Identifiant de la ronde MRTR ou de la tâche, le cas échéant.

Les jetons de rédaction, les codes d'autorisation, les jetons de rafraîchissement, les secrets bruts et les arguments sensibles inutiles.

### SSE à échelle de demande

Un POST normal peut retourner SSE scopé de demande lorsque des flux de travail sont effectués pendant cette seule demande.

Ne créez pas un flux GET séparé et ne promettez pas de répliquer Last-Event-ID.

### Notifications de changements à long terme

Pour les notifications de changement de liste et de ressources, un client actuel envoie `subscriptions/listen`Les filtres de notification utilisent les champs plats exacts `toolsListChanged`- Je suis là .`promptsListChanged`- Je suis là .`resourcesListChanged`, et `resourceSubscriptions`- Le numéro de la liste:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Le premier événement reconnaît le sous-ensemble pris en charge. Son identifiant d'abonnement est l'id JSON-RPC de la demande qui a ouvert le flux:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

La passerelle renvoie alors uniquement les types de changements reconnus.`io.modelcontextprotocol/subscriptionId`dans `params._meta`. Il n'y a pas de répétition automatique ou de réécoute automatique. Lors de la reconnexion, le client rouvre l'abonnement et renouvelle les listes sur lesquelles il s'appuie.

Le chemin moderne remplace `resources/subscribe`- Je suis là .`resources/unsubscribe`Gardez ces derniers dans un chemin plus ancien, avec des versions fermées.

### MRTR à travers une passerelle

Quand un backend revient `resultType: input_required`, la passerelle ne peut transmettre ce résultat que si le client externe prend en charge la demande d'entrée nécessaire.`requestState`octet par octet, à moins que la passerelle ne mette fin délibérément à l'interaction et ne la réémette.

Le client réessaye l' outil public original avec un ID JSON-RPC frais et `inputResponses`- la passerelle autorise à nouveau la reprise, vérifie le même itinéraire public, puis envoie une nouvelle demande de backend.

### Routing des tâches d'extension

Les tâches sont une extension officielle identifiée par `io.modelcontextprotocol/tasks`Ils ne remplacent pas les séances de base.

Le client déclare l'extension à l'intérieur des capacités du client par demande, et la passerelle l'annonce en découverte seulement quand elle peut préserver le cycle de vie de bout en bout.`tools/call`, le backend seul décide si le résultat ordinaire est retourné ou `resultType: task`. Un résultat de tâche porte `taskId`- Je suis là .`status`, timestamps,`ttlMs`, et une option `pollIntervalMs`La tâche doit être déjà lisible durablement avant l'envoi de ce résultat.

La passerelle enregistre l'itinéraire principal et l'arrière-plan authentifiés pour l'identifiant de tâche opaque.`tasks/get`- Je suis là .`tasks/update`, et `tasks/cancel`utilisation des appels `params.taskId`comme `Mcp-Name`, qui donne aux intermédiaires une clé de routage. `tasks/get`Retour `resultType: complete`avec l'état de tâche actuel et incarne le résultat final ou l'erreur de protocole dans un état terminal. `tasks/update`envoie à clé `inputResponses`pour les entrées de tâche en suspens et renvoie une confirmation complète vide. `tasks/cancel`est une intention de coopération avec une reconnaissance totale vide, pas une garantie que le travail cesse.

Ne pas mettre en œuvre de nouvelles `tasks/list`ou `tasks/result`Les méthodes de recherche sont des méthodes de recherche, qui sont des méthodes de recherche, et qui sont des méthodes de recherche.`tasks/get`; le client répond à ces questions en leur adressant des réponses `tasks/update`Le client vote toujours à l'intervalle suggéré; la création de tâches reste orientée vers le serveur.

L'état de route de tâche durable est les données d'application clés par la manche de tâche, pas une session de protocole.

### Limite de compatibilité

Si la passerelle doit servir un client ou un backend plus ancien:

- Détectez l'ère explicitement.
- Gardez l'initialisation, les séances de transport, les flux GET, les abonnements aux ressources et l'ancien vocabulaire de tâches dans un adaptateur ancien.
- Ne divulguez jamais un identifiant de session dans un routage ou une autorisation modernes.
- Je préfère une sonde de découverte limitée et une politique explicite de rétroaction à une dégradation silencieuse.

```figure
t3-gateway-funnel
```

## Faites-le

`code/main.py`Il est utilisé pour la mise en œuvre d'une passerelle de protocole en cours de processus et de deux serveurs backend.`tools/list`, routage à espace nommé, Registre `server.json`Plus l'état d'admission externe, les pinces descripteurs, le RBAC, les limites de taux de base, les décisions d'audit et un modèle `subscriptions/listen`Confirmation de l'ESA.

Le modèle reçoit des corps de requêtes analysés, des en-têtes de routage et une identité de porteur authentifiée.`Content-Type`ou le plein `Accept`connectez-le à l'adaptateur HTTP diffusable de la leçon 09 qui nécessite `Content-Type: application/json`et une `Accept`valeur contenant les deux `application/json`et `text/event-stream`- Je suis désolé .

- Je vais le faire.

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

La démo imprime l'ID de la demande externe et l'ID de la demande de l'arrière-plan frais afin que le hop sans état soit visible.

## Utilisez-le

Remplacez les objets de fond en cours de processus par de vrais clients de protocole de courant.

- - Le dossier d'admission avant la connexion.
- Découverte de l'arrière avant l'exposition des capacités.
- Nom public qualifié avant autorisation.
- Pin descripteur avant liste ou appel.
- Les métadonnées à la demande sont fraîches avant de les transférer.
- Validation du résultat avant retour.

## La faire partir

Cette leçon va à l' air .`outputs/skill-gateway-bootstrap.md`Il produit une conception de passerelle moderne couvrant l'entrée, la découverte, l'admission, les espaces de noms, l'autorisation, le caching, le streaming, les abonnements, le MRTR, les tâches, l'observabilité et l'isolement hérité.

## Exercices

1. Ajouter un contexte de trace aux métadonnées extérieures et transmises de la demande et enregistrer la corrélation dans l'événement d'audit.
2. Ajouter un arrière-plan et une route capables de tâches `tasks/get`par ID de tâche en `Mcp-Name`- Je suis désolé .
3. Changez un descripteur de backend et prouvez que la découverte et l'appel direct sont bloqués.
4. Ajouter une capacité de serveur spécifique au principal et expliquer pourquoi la découverte doit rester en cache privée.
5. Écrire une interface adaptateur ancienne sans ajouter aucun état ancien à la moderne `Gateway`- La classe.

## Les termes clés

| Term | Meaning |
|------|---------|
| MCP gateway | Policy and routing server between clients and backend MCP servers |
| Admission record | Evidence and policy decision allowing one backend into the gateway |
| Qualified tool name | Stable public route such as `notes.search` |
| Descriptor pin | Approved digest checked during discovery and dispatch |
| Private cache scope | Cached result restricted to one authorization context |
| Request-scoped SSE | Streaming response attached to one POST request |
| `subscriptions/listen` | Client-opened SSE stream for selected long-lived change notifications |
| Task route | Application mapping from an opaque task id to its backend |
| Legacy adapter | Explicit version-gated boundary for old handshake and session behavior |

## Pour en savoir plus

- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
