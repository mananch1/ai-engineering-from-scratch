# Construire un client MCP: découverte, routage et double époque de retour

> Un client MCP moderne répète son contrat à chaque demande. Sa décision de compatibilité la plus difficile est de savoir quand un ancien serveur est vraiment vieux et quand un serveur moderne rapporte une erreur corrigable.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07
**Time:** ~85 minutes

## Objectifs d'apprentissage

- Construire chaque MCP `2026-07-28`une demande avec des métadonnées actuelles.
- Probe des serveurs stdio avec `server/discover`et sélectionnez une version mutuellement prise en charge.
- Autoriser une sonde d'héritage limitée uniquement pour les pairs explicitement autorisés.
- Acceptez une ère d' héritage seulement après avoir validé un positif `initialize`résultat pour une révision appuyée.
- Fusion des listes d'outils déterministes sans écraser silencieusement les collisions.
- Route appels à la paire qui possède chaque outil sans inventer des sessions de protocole.

## Le problème

Un agent hôte parle généralement à plus d'un serveur MCP. Il doit découvrir chaque serveur, fusionner les catalogues des outils, résoudre les noms dupliqués, les appels de route et récupérer des défaillances de transport.

Le `2026-07-28`La révision rend l'état stable plus simple parce que chaque demande est autonome.

- un serveur moderne qui prend en charge la version préférée;
- un serveur moderne qui renvoie une version reconnue ou une erreur d'en-tête;
- Un serveur hérité dont vous n' avez jamais entendu parler .`server/discover`Le dépôt de la commission
- Un serveur ancien qui reste silencieux jusqu' à ce qu' il reçoive `initialize`- Je suis désolé .

Traiter chaque erreur de sonde comme héritage est dangereux. Une demande moderne malformée, un serveur surchargé, un processus mort et un serveur ancien peuvent tous produire le même temps ou la fermeture de la connexion. Ces signaux sont ambiguës. Le client doit combiner l'intention explicite de l'opérateur avec des preuves de protocole positifs avant de choisir l'ère héritage.

## Le concept

### Une session de pairs, pas une session de protocole

Garder un enregistrement par rapport au transport pour chaque processus ou point final du serveur:

- fonction de poignée de transport ou d'envoi;
- l'ère et la version du protocole sélectionnés;
- les capacités du serveur découvertes pour la dernière fois;
- dernière liste d'outils déterministes;
- les identifiants de demande en attente de corrélation;
- santé des transports.

C'est la comptabilité du client. Ce n'est pas l'état de session du protocole. Sur le MCP moderne, le serveur reçoit toujours la version et les capacités actuelles à chaque demande.

### Construire chaque demande moderne à partir de zéro

```python
def modern_request(request_id, method, params, version, capabilities):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": version,
                "io.modelcontextprotocol/clientCapabilities": capabilities,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            },
        },
    }
```

Ne joignez pas les métadonnées à un objet de connexion une fois et supposez qu'il a atteint le fil.

### Découverte moderne

`server/discover`renvoie les versions prises en charge, les capacités du serveur, les instructions, les indices de cache et l'identité du serveur recommandée. Un client choisit la version moderne la plus haut en charge mutuellement.

La découverte est facultative pour un client moderne uniquement, mais elle est recommandée sur stdio. Certains serveurs anciens acceptent une opération avant l'initialisation, donc l'envoi `tools/list`La première peut produire un succès ambigu. `server/discover`crée une frontière d'ère propre.

### La sonde de compatibilité avec le studio

Un client de studio à deux époques envoie `server/discover`Il existe trois classes de résultats:

1. **DiscoverResult.**Le serveur est moderne. Sélectionnez une version mutuellement prise en charge et continuez avec les métadonnées par demande.
2. **Recognized modern error.**Le serveur est moderne.`-32022`, choisissez parmi `data.supported`Pour les erreurs d'en-tête ou de capacité, corrigez la demande.`initialize`- Je suis désolé .
3. **Ambiguous signal.**Une erreur JSON-RPC non reconnue, une échéance de temps, une connexion fermée ou une réponse vide n'identifient pas une époque.

Les erreurs de protocole modernes reconnues comprennent:

- `-32020`HeaderIncohérent
- `-32021`Faute de capacité requise
- `-32022`Protocol non pris en chargeVersion

Les erreurs modernes reconnues restent modernes même lorsque le pair est sur la liste d'allowness.`initialize`Ce serait une dégradation.

Ne traitez pas`-32601`Il n'est possible de faire une enquête par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un seul analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste par un analyste

### L'autorisation est l'intention de l'opérateur, pas une preuve

La compatibilité héréditaire doit être une propriété explicite d'une configuration de pair fichée:

```python
client.add_server("archive", archive_transport, allow_legacy=True)
```

Passer le temps à la sélectionner.`allow_legacy=True`Il échoue après une découverte ambiguë et ne reçoit jamais .`initialize`- Je suis désolé .

L'autorisation de l'internaute est accordée par le client, mais pas par l'ère.`initialize`dans le cadre d'une date limite imposée par le transport, il exige alors tout ce qui suit:

- un JSON-RPC `2.0`la réponse avec l'identifiant de la demande correspondante;
- Exactement une .`result`et non`error`Le dépôt de la commission
- - Il y a`protocolVersion`dans le jeu de révision de l'héritage configuré du client;
- une valeur d' objet `capabilities`champ;
- - Il y a`serverInfo`objet avec une chaîne non vide `name`et `version`les champs.

Un délai, la fermeture de la connexion, la réponse à l'erreur, le résultat malformé, l'identifiant inégalé ou la révision non prise en charge échouent à fermer. Seul un résultat positif structurellement valide sélectionne l'ère héréditaire. Le code passe `legacy_probe_timeout_ms`à l'adaptateur de transport; un véritable stdio ou adaptateur HTTP doit appliquer cette date limite plutôt que de simplement l'enregistrer.

Conservez l'ère sélectionnée en cache pour le transport.

### L'héritage est une branche de compatibilité

Une fois que la sonde délimité renvoie des preuves de l'héritage positif valides, le client utilise la version d'héritage sélectionnée exactement comme définie par cette révision:

1. Vérifiez l'enveloppe de réponse et l'identifiant de corrélation.
2. Vérifiez que la révision négociée est dans l'ensemble de l'héritage configuré.
3. Enregistrer les capacités validées et l'identité du serveur.
4. Envoyez-moi .`notifications/initialized`seulement après que tous les chèques aient passé.
5. Utilisez des formes de demande héritées pour cette durée de vie du transport.

Cette branche existe pour l'interopérabilité avec des pairs connus. Ce n'est pas la conception par défaut pour de nouveaux serveurs ou de nouvelles demandes. Si le transport redémarre ou que son point d'extrémité change, jetez le cache de l'ère des pairs et négociez à nouveau.

### Outils de découverte et de mise en cache

Pour chaque paire actif, appelez `tools/list`Un résultat moderne comprend `resultType`- Je suis là .`ttlMs`, et `cacheScope`- Respecter le signe de fraîcheur dans le contexte de l'autorisation correcte.

Les clients doivent traiter un disparu .`resultType`à partir d' un serveur ancien comme `"complete"`Ne nécessite pas de champs de cache modernes sur une réponse d'une ère précédente négociée.

Le serveur doit retourner l'ordre déterministe. Le client doit également trier avant la fusion afin que l'ordre du registre local ne dépend pas du moment de démarrage du processus.

### fusion de l'espace de noms à risque de collision

Deux serveurs peuvent exposer les deux .`search`. Choisir une politique déclarée:

1. **Prefix on collision.**Gardez le premier nom canonique et exposez les collisions ultérieures comme `<server>/<tool>`- Je suis désolé .
2. **Reject on collision.**Ne chargez pas le double et ne faites pas apparaître une erreur de configuration claire.
3. **Silent overwrite.**Ne l'utilisez jamais, il cache quel serveur reçoit une action sélectionnée par le modèle.

Le modèle voit le nom canonique.`tools/call`utilise le nom local déclaré par le serveur propriétaire.

### Routage d'une appel

Le routage est une simple recherche:

```text
canonical tool name
  -> peer name + local tool name
  -> new JSON-RPC request id
  -> modern request metadata or explicit legacy shape
  -> matching response id
```

Ne pas envoyer d'appel lorsque le transport de son propriétaire n'est pas disponible.`tools/list`. Les demandes modernes perdues en vol lors d'un transport défectueux peuvent être retenues avec un nouvel identifiant JSON-RPC lorsque la politique de sécurité de l'opération le permet.

### Notifications et abonnements

Les modifications modernes des listes et des ressources ne sont effectuées que sur une liste ouverte par le client `subscriptions/listen`Le client envoie le filtre de notification, attend `notifications/subscriptions/acknowledged`, et corréle les événements avec l'identifiant de demande d'écoute dans les métadonnées de notification.

Lorsqu'il est déconnecté, ouvrez une nouvelle demande d'écoute et réaffectez les listes ou ressources pertinentes.`Last-Event-ID`- Je suis désolé .

### Aucune demande initiée par le serveur

Les serveurs modernes n'appellent pas le client avec des demandes indépendantes JSON-RPC pour le prélèvement d'échantillons, l'élicitation ou la racine.`input_required`, et le client tente à nouveau la demande originale après avoir rempli les demandes d'entrée intégrées.

Ne bloquez pas le lecteur de réponse de l'utilisateur pendant la commande.

```figure
tp-client-merge
```

## Utilisez-le

`code/main.py`Il utilise des fonctions de pair en cours de processus afin que les décisions du protocole restent visibles. Il se connecte à deux pairs modernes et à un paire hérité délibérément autorisé, puis fusionne et route leurs outils.

```bash
cd code
python3 main.py
python3 -m unittest discover tests -v
```

Les tests prouvent les limites que les démos normaux manquent:

- les demandes modernes répéteront les métadonnées;
- `-32022`réessaye la découverte moderne sans initialisation;
- les erreurs modernes reconnues ne sont jamais rebaptisées, même pour un paire autorisé;
- Les temps de sortie, la fermeture de la connexion, les réponses vides et les erreurs non reconnues ne déclenchent pas `initialize`sans permis;
- Un coéquipier autorisé devient héritier seulement après un coéquipier valide et soutenu `initialize`résultat;
- les résultats antérieurs mal formés et non soutenus rendent le produit non disponible;
- une époque sélectionnée avec succès est mise en cache pour la durée de vie du transport.

## La faire partir

Cette leçon va à l' air .`outputs/skill-mcp-client-harness.md`Il comporte le timbre de requête moderne, la négociation de l'ère studio, la fusion déterministe de l'espace de noms, le routage et une branche de compatibilité héréditaire fermées en échec.

## Exercices

1. Faites une fausse restitution de serveur .`-32022`Confirmer que le client échoue au lieu d'envoyer `initialize`- Je suis désolé .
2. Permettez un faux serveur hérité, faites-le limité `initialize`- Je vais vous montrer que le groupe reste.`unknown`et non disponible.
3. Ajouter `cacheScope: "private"`Confirmer que le client ne partage jamais le résultat caché d'un contexte avec l'autre.
4. Modifiez la politique de collision en rejet et faites échouer le démarrage avec les deux noms de pairs dans l'erreur.
5. Ajouter une finite `subscriptions/listen`En cas de perte de flux, écoutez à nouveau avec un nouvel identifiant de requête et des outils de rééducation.

## Les termes clés

| Term | Meaning |
|------|---------|
| Peer | Client-side record for one server transport and its discovered data |
| Protocol era | Modern per-request metadata or legacy initialization semantics |
| Discovery probe | Initial `server/discover` used to identify the stdio era |
| Recognized modern error | Error that proves modern behavior and forbids legacy fallback |
| Legacy allowlist | Operator configuration permitting one bounded compatibility probe for a pinned peer |
| Positive legacy evidence | Valid, correlated `initialize` result for an explicitly supported legacy revision |
| Merged namespace | Canonical tool names across all active peers |
| Collision policy | Prefix or reject rule for duplicate tool names |
| Era cache | Selected modern or legacy behavior stored for one transport peer |
| Transport recovery | Restart or reconnect, rediscover, relist, and retry safely with a new id |

## Pour en savoir plus

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
