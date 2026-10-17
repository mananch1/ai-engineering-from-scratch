# Transports MCP: stdio et HTTP en streaming sans État

> Le transport transporte des messages MCP. Il ne fournit pas l'état du protocole manquant.`2026-07-28`, stdio local et HTTP diffusable à distance portent toutes deux des demandes d'auto-décription.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07 and 08
**Time:** ~65 minutes

## Objectifs d'apprentissage

- Choisissez stdio pour les processus en enfant local et HTTP diffusable pour les services réseau.
- Implémenter le contrat HTTP Streamable moderne à point unique, uniquement POST.
- Réfléchissez et validez les en-têtes de version, de méthode et de nom de MCP contre le corps JSON-RPC.
- Livraison de l' SSE à la demande et à longue durée de vie `subscriptions/listen`les courants correctement.
- Migrons les déploiements HTTP+SSE basés sur les sessions et anciens sans présenter le comportement d'ancienneté comme moderne.

## Le problème

Les versions précédentes de Streamable HTTP combinent la négociation du protocole avec le comportement de connexion et de session.`Mcp-Session-Id`, exposer un flux GET autonome, accepter le DELETE pour la fin de la session et reprendre l'ESS avec `Last-Event-ID`- Je suis désolé .

MCP `2026-07-28`Chaque requête peut atterrir sur n'importe quel travailleur en bonne santé parce que sa version de protocole et les capacités du client voyagent dans le corps de la requête.

Le résultat est plus facile à étaler et plus facile à raisonner. Cela signifie également qu'un serveur qui enseigne le transport 2025 comme courant enseigne le mauvais modèle de défaillance et de sécurité.

## Le concept

### studio

La liaison stdio est pour un sous-processus lancé par le client:

- Le client écrit un message UTF-8 JSON-RPC par ligne à stdin.
- Le serveur écrit un message UTF-8 JSON-RPC par ligne à stdout.
- Le serveur écrit des diagnostics à Stderr.
- Le serveur sort rapidement sur le stdin EOF.
- Chaque demande moderne contient des versions et des capacités de client dans `params._meta`- Je suis désolé .

Le processus peut être utilisé pour de nombreux appels, mais il ne s'agit pas d'une session de protocole moderne. S'il sort de manière inattendue, les demandes en vol sont perdues. Rémarquez le processus, redécouvrez, réenregistrer, rouvrir les abonnements et réessayer des opérations sécurisées avec de nouvelles identifiants de requête.

### HTTP diffusable en 2026-07-28

Un serveur moderne expose un point final du MCP, tel que `/mcp`, qui accepte POST.

Chaque requête ou notification JSON-RPC est un nouveau HTTP POST. Le corps contient un message JSON-RPC. Les clients n'envoient pas de réponses JSON-RPC au serveur.

Pour une demande, le serveur renvoie:

- `Content-Type: application/json`avec une réponse JSON-RPC; ou
- `Content-Type: text/event-stream`avec les notifications relatives à cette demande, suivie de la réponse finale JSON-RPC.

Pour une notification acceptée, le serveur retourne `202 Accepted`sans corps.

Les clients font de la publicité pour les deux types de réponses:

```http
Accept: application/json, text/event-stream
```

### Le terme "POST seulement" signifie "POST seulement".

Le HTTP Streamable moderne n'a pas de flux GET autonome et aucun point d'extrémité de session DELETE.

- `GET /mcp`Retour `405 Method Not Allowed`- Je suis désolé .
- `DELETE /mcp`Retour `405 Method Not Allowed`- Je suis désolé .
- `Mcp-Session-Id`est ignoré et jamais inventé ou écho.
- `Last-Event-ID`est ignoré parce que les flux modernes ne sont pas réutilisables.

Si un flux scopé par la demande est cassé avant sa réponse finale, le client a perdu cette demande en vol. Il peut émettre une nouvelle demande avec un nouvel id JSON-RPC lorsque la reprise est sûre. Il ne doit pas tenter de reprendre le flux.

### Validation de l'origine

Les serveurs valident `Origin`Si l'en-tête est présent et n'est pas explicitement autorisé, retourner `403 Forbidden`Un client non navigateur peut omettre `Origin`, ce qui est autorisé par les règles officielles du transport.

Les serveurs locaux doivent être liés à `127.0.0.1`Les services réseau ont toujours besoin d'authentification et d'autorisation pour chaque demande.

Utilisez une correspondance exacte d'origine après la configuration canonique.`origin.startswith("https://trusted.example")`sont dangereuses parce qu'elles peuvent accepter des suffixes contrôlés par l'attaquant.

### En-tête de métadonnées HTTP requis

Chaque demande POST moderne comprend:

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

Règles de titre:

- `MCP-Protocol-Version`est exigé et doit être égal `params._meta.io.modelcontextprotocol/protocolVersion`- Je suis désolé .
- `Mcp-Method`est nécessaire et doit être égal au JSON-RPC `method`- Je suis désolé .
- `Mcp-Name`est nécessaire pour `tools/call`- Je suis là .`resources/read`, et `prompts/get`- Je suis désolé .
- `Mcp-Name`égale `params.name`ou `params.uri`pour `resources/read`- Je suis désolé .
- Les valeurs d'en-tête sont sensibles aux cas même si les noms d'en-tête sont insensibles aux cas.

Non sûre ou non ASCII `Mcp-Name`Les valeurs utilisent la sentinelle UTF-8 Base64 exacte:

```text
=?base64?{Base64EncodedValue}?=
```

Le serveur décode cette valeur avant de la comparer avec le corps.

En-tête miroir manquant, malformé ou mal correspondant retour HTTP `400`avec code JSON-RPC `-32020`. Si l'en-tête et le corps conviennent d'une version que le serveur ne prend pas en charge, retournez HTTP `400`avec `-32022`et des données d'erreur exactes telles que `{"supported":["2026-07-28"],"requested":"2027-01-01"}`- Je suis désolé .

Une méthode moderne inconnue renvoie HTTP `404`avec JSON-RPC `-32601`. Le corps JSON-RPC est important car un client à double époque l'utilise pour distinguer une erreur moderne d'une erreur de point final héritée.

### SSE à échelle de demande

Un serveur peut choisir SSE pour une seule demande de longue durée:

```text
POST tools/call id=41
  <- notifications/progress related to id=41
  <- notifications/progress related to id=41
  <- JSON-RPC response id=41
stream closes
```

Le serveur ne doit pas envoyer de requêtes JSON-RPC indépendantes sur ce flux. L'échantillonnage, l'exécution et les interactions racines utilisent des résultats de requête de plusieurs tours.

Ne pas ajouter d' identifiants d' événement SSE pour la répétition. `Last-Event-ID`La révision n'est pas une révision moderne.

### Changements de longue durée utilisant les abonnements/écoute

Les notifications de modification utilisent une demande ouverte par le client, et non un GET indépendant:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

La réponse POST est un flux SSE de longue durée.`notifications/subscriptions/acknowledged`- la confirmation, chaque notification de changement et le résultat final`io.modelcontextprotocol/subscriptionId`dans `_meta`Le serveur peut émettre des commentaires SSE en tant que conservateurs.`subscriptions/listen`avec un nouveau identifiant de demande et réaffecte les données affectées.

`resources/subscribe`et `resources/unsubscribe`Ne les utilisez pas dans une connexion moderne.

### État de demande explicite

La suppression des sessions de protocole n'interdit pas les flux de travail avec état. Le serveur peut imprimer une poignée d'état opaque et la retourner comme résultat d'outil normal. Le client passe cette poignée comme argument explicite sur les appels ultérieurs.

La liaison des poignées au principal authentifié, les rendant inessables, expirant et autorisant toute utilisation. Cela rend l'état visible sur la couche d'application au lieu de le cacher dans l'affinité du transport.

L'échec causé par l'état de réplication cachée est mécanique:

1. La requête A atteint la réplique 1 et crée un projet dans la mémoire de ce processus.
2. La réponse ne renvoie pas un projet de manœuvre car la mise en œuvre suppose que la connexion identifie le projet.
3. La demande B est un nouveau POST et atteint la réplique 2.
4. La réplique 2 a des métadonnées protocoles valides mais aucun moyen de nommer ou de charger le projet, donc le flux de travail échoue ou lit l'objet local incorrect.
5. Le routage collant semble corriger le symptôme jusqu'à ce qu'une redémarrage, un déploiement, un réprogramming ou un décalage déplacent la prochaine demande.

La limite correcte a deux parties. Le contexte du protocole reste dans chaque demande. L'état d'application durable vit dans un magasin partagé sous une poignée de serveur remise au client. L'appel suivant fournit le traitement, toute copie charge le même enregistrement, et l'autorisation lie le enregistrement au principal et au locataire authentifiés. La mémoire de réplication peut mettre un enregistrement en cache, mais elle ne peut pas être la seule copie requise pour être correcte.

Choisissez le mécanisme d'état par durée de vie. Les variables locales de demande peuvent servir à un appel. Une courte continuation MRTR peut utiliser l'intégrité protégée `requestState`Une tâche de projet ou durable nécessite une manipulation explicite plus une persistance partagée, une expiration, un contrôle de la concurrence et une idempotence.

### Compatibilité HTTP à deux époques

Un client qui prend en charge des serveurs modernes et anciens tente d'abord un POST moderne.`400`- Je suis là .`404`ou `405`, il inspecte le corps:

- Une erreur JSON-RPC moderne reconnue prouve que le serveur est moderne. Corrigez la demande ou réessayez une version annoncée. Ne pas dégrader.
- Un corps vide ou une réponse non reconnue peut indiquer un serveur HTTP+SSE ancien.`endpoint`événement.

Un serveur peut prendre en charge les deux époques pendant la migration en routant les métadonnées modernes vers la mise en œuvre moderne POST uniquement et en conservant des terminaux hérités séparés pour les anciens clients.`2026-07-28`- Je suis désolé .

```figure
tp-transport-handshake
```

## Utilisez-le

`code/main.py`Il met en œuvre un serveur HTTP Streamable moderne et fini avec la bibliothèque standard Python. Il valide les en-têtes d'origine et miroirés, ignore les en-têtes de session supprimés, renvoie JSON pour les appels normaux et démontre un finit `subscriptions/listen`Le courant SSE.

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

La sonde vérifie:

- l'origine non valable est rejetée;
- la découverte réussit sans identifiant de session;
- `Mcp-Session-Id`et `Last-Event-ID`sont ignorées;
- Les résultats de l' incohérence d' en-tête `-32020`Le dépôt de la commission
- Retour de version non prise en charge `-32022`avec exactitude `supported`et `requested`les données;
- une notification sans identité acceptée renvoie HTTP `202`sans corps;
- Retour à la Retour et à la Retour à la Retour `405`Le dépôt de la commission
- `subscriptions/listen`est un flux de réponse POST dont la confirmation, les notifications et le résultat final portent son identifiant d'abonnement.

## La faire partir

Cette leçon va à l' air .`outputs/skill-mcp-transport-migrator.md`Il supprime les sessions de protocole modernes, ajoute la validation du corps d'en-tête, remplace le GET autonome par `subscriptions/listen`, et garde tout pont hérité visiblement séparé.

## Exercices

1. Retirez `Mcp-Method`à partir d' un message public.`400`et erreur `-32020`- Je suis désolé .
2. Envoyez la version correspondante de l' en-tête et du corps `2027-01-01`Confirmer le HTTP`400`, erreur `-32022`, et des données exactes `{"supported":["2026-07-28"],"requested":"2027-01-01"}`- Je suis désolé .
3. Envoyez une sentinelle Base64 .`Mcp-Name`Pour une URI de ressource non ASCII. Confirmer que la valeur décodée est comparée à `params.uri`- Je suis désolé .
4. Brisez le flux d'écoute fini avant sa réponse finale, rééditez-le avec un nouvel identifiant JSON-RPC et réécrivez-le avec des outils.
5. Ajouter une poignée de flux de travail explicite à l'outil ping. Lier à un sujet d'autorisation sans utiliser l'affinité de connexion.

## Les termes clés

| Term | Meaning |
|------|---------|
| stdio | Newline-delimited JSON-RPC over a client-launched subprocess |
| Streamable HTTP | Single endpoint where each modern message is a new POST |
| Request-scoped SSE | POST response stream containing related notifications and final response |
| `subscriptions/listen` | Long-lived POST request for opted-in change notifications |
| Header mismatch | HTTP `400` and JSON-RPC `-32020` when mirrored headers disagree with body |
| Origin validation | DNS-rebinding defense for incoming connections, not authentication |
| Explicit state handle | Application token passed as an ordinary argument instead of hidden session state |
| Legacy bridge | Separate earlier-era behavior kept only for compatibility |

## Pour en savoir plus

- [MCP Transport Overview](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
