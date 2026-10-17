# Les principes fondamentaux du MCP: demandes de stateless et JSON-RPC

> Le MCP moderne n'a pas de poignée de main et aucune session de protocole. Chaque requête doit contenir suffisamment de métadonnées pour être comprise, autorisée, rouverte et retestée par elle-même.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 01 through 05
**Time:** ~55 minutes

## Objectifs d'apprentissage

- Distinguer les primitifs du serveur de MCP de ses fonctionnalités côté client.
- Construire des requêtes et réponses JSON-RPC 2.0 valides pour MCP `2026-07-28`- Je suis désolé .
- Attachez la version du protocole, les capacités du client et l'identité du client à chaque demande.
- Utilisation `server/discover`et la main`UnsupportedProtocolVersionError`sans serrer la main.
- Suivre une demande indépendante de la validation à un résultat complet.

## Le problème

Un serveur MCP peut recevoir deux demandes consécutives de différents clients, avec des capacités différentes, sur le même processus ou le même serveur HTTP. Si le serveur se souvient de ce que la requête précédente a déclaré, il peut appliquer les mauvaises autorisations ou retourner la mauvaise forme du fil.

MCP `2026-07-28`Le protocole est sans état, le serveur doit décider comment traiter la requête actuelle à partir de la requête actuelle, pas de l'historique de connexion.

La première séquence était la connexion, la seconde la poignée de main, la troisième les opérations.

1. Le client envoie une demande d'auto-décription.
2. Le serveur valide la version et les capacités de cette demande.
3. Le serveur gère la méthode.
4. Le serveur renvoie un résultat typé ou une erreur JSON-RPC.

La demande suivante répète le même processus à partir de zéro.

## Le concept

### Les serveurs primitifs

Les serveurs MCP exposent trois primitives primaires:

1. **Tools**sont des actions contrôlées par modèle, découvertes avec `tools/list`et invoqué avec `tools/call`- Je suis désolé .
2. **Resources**sont des données adressées aux IRU, découvertes avec `resources/list`et récupéré avec `resources/read`- Je suis désolé .
3. **Prompts**sont des modèles réutilisables, découverts avec `prompts/list`et rendu par `prompts/get`- Je suis désolé .

Les racines, l'échantillonnage et l'exploitation forestière restent dans le `2026-07-28`Les plans de compatibilité sont dépassés. Les nouvelles implémentations devraient utiliser des outils ou des ressources explicites pour les racines, des API directes du fournisseur de modèles pour le prélèvement d'échantillons et stderr ou OpenTelemetry pour l'enregistrement. L'exécution reste disponible par le biais de demandes de plusieurs voyages ronds, où un serveur renvoie une demande d'entrée et le client tente à nouveau l'opération initiale. Un serveur moderne ne démarre jamais une demande JSON-RPC indépendante.

### Enveloppes JSON-RPC

MCP utilise JSON-RPC 2.0:

- Demande: `{jsonrpc, id, method, params}`
- Réponse: `{jsonrpc, id, result}`ou `{jsonrpc, id, error}`
- La notification: `{jsonrpc, method, params}`sans aucun`id`

La demande `id`Il ne crée pas de session de protocole.

### Metadata requise requise

Chaque demande moderne est accompagnée d' une`_meta`objet à l' intérieur `params`- Le numéro de la liste:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
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

La version du protocole et les capacités du client sont nécessaires. L'identité du client est recommandée. Il s'agit d'une affichage et de débogage automatiques, pas d'une carte de sécurité.

Le serveur ne doit pas déduire une de ces valeurs à partir d'une demande antérieure, d'un processus stdio, d'une connexion HTTP ou d'une en-tête de transport seule.

### Résultats complets et identité du serveur

Chaque résultat moderne réussi inclut`resultType`. Un résultat final normal utilise `"complete"`Les serveurs doivent également s'identifier dans les métadonnées des résultats:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`- Je suis là .`resources/list`- Je suis là .`prompts/list`- Je suis là .`resources/templates/list`- Je suis là .`resources/read`, et `server/discover`Les résultats sont cachés.`ttlMs`et `cacheScope`Une défaillance sécurisée est`ttlMs: 0`et `cacheScope: "private"`. Les éléments de liste doivent avoir un ordre déterministe afin que les réponses équivalentes produisent des clés de cache stables et un contexte de modèle stable.

### Découverte sans serrer la main

Chaque serveur moderne doit mettre en œuvre `server/discover`Le client peut appeler avant une autre méthode pour obtenir:

- `supportedVersions`
- serveur `capabilities`
- utilisation facultative `instructions`
- l' identité du serveur en conséquence `_meta`
- indices de cache

La découverte est utile, mais ce n'est pas une passerelle.`tools/list`Premièrement, parce que cette demande comporte déjà sa version et ses capacités de protocole.

Si la version demandée n'est pas prise en charge, le serveur renvoie le code JSON-RPC `-32022`avec:

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

Le client sélectionne une version moderne mutuellement prise en charge et réessaye avec un nouveau identifiant de requête JSON-RPC.

### Un cycle de vie de la demande

Suivez une demande moderne dans cet ordre:

1. Parser une enveloppe JSON-RPC.
2. Confirme `jsonrpc`est `"2.0"`, une`id`Il existe.`method`est une chaîne, et `params`est un objet.
3. Requérir l' objet de chaîne de version et de capacité dans `params._meta`; les métadonnées malformées ou manquantes sont `-32602`- Je suis désolé .
4. Dans une limite HTTP, comparez la version, la méthode et les en-têtes de nom applicables avec le corps.`-32020`même si l'une des deux valeurs de version n'est pas prise en charge.
5. Une fois l'égalité établie, rejeter une version correspondante mais non soutenue avec `-32022`- Je suis désolé .
6. Vérifiez les capacités requises, puis parcourez `method`et valider les arguments spécifiques à la méthode.
7. Authentifier et autoriser l'opération de béton avant son exploitation.
8. Retournez un résultat complet avec l'identité du serveur.
9. Oubliez les métadonnées du protocole.

Cette ordonnance empêche deux composants d'interpréter des appels différents.`Mcp-Name: notes.read`pendant que l' origine exécute `params.name: notes.delete`Il conserve également des entrées malformées, la confusion des en-têtes, la négociation de versions, l'échec de la capacité, l'autorisation et l'échec du gestionnaire comme preuve distincte.

Fermer stdin ou une réponse HTTP met fin à l'activité de transport.

### Compatibilité explicite avec l'héritage

Les versions à travers `2025-11-25`utilisation `initialize`- Je suis là .`notifications/initialized`Ce comportement est toujours pertinent lorsqu'un client à double époque parle à un ancien serveur.

Gardez les périodes séparées. Une demande moderne est identifiée par les métadonnées requises par requête. Une connexion héritée est sélectionnée uniquement par le chemin de retour documenté.`initialize`comme défaut pour un `2026-07-28`Le serveur.

L'appartenance à un État  a donc une signification spécifique à l'ère.`2026-07-28`, il s'agit d'une invariance de protocole: chaque demande ordinaire est interprétable indépendamment et aucune session de MCP n'existe.`2025-11-25`Une mise en œuvre à deux époques n'est pas une machine d'état permissif. C'est un noyau moderne sans état à côté d'un adaptateur d'état isolé, avec une décision de sélection explicite avant l'exécution de l'un des parseurs.

Aucun des deux sens n'interdit l'état d'application durable. Un flux de travail, une tâche ou un projet peut vivre derrière une poignée opaque dans un magasin partagé. Le client envoie cette poignée comme entrée ordinaire, et chaque réplique authentifie et autorise son utilisation. Le contexte du protocole ne doit pas fuir dans ce magasin en remplacement de la session supprimée.

```figure
mcp-tool-call
```

## Utilisez-le

`code/main.py`crée, valide, trace et envoie des messages MCP modernes sans cadre.

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Attention à trois invariants dans la sortie:

- Chaque demande répète son`_meta`les champs.
- Tout résultat réussi est`resultType: "complete"`et inclut l'identité du serveur.
- Le résultat de la liste est ordonné de manière déterministe et a des indices de cache explicites.

## La faire partir

Cette leçon va à l' air .`outputs/skill-mcp-handshake-tracer.md`. Le nom historique du fichier reste stable, mais l'artefact est maintenant un traceur de requêtes sans état. Il contrôle chaque message de manière indépendante et n'étiquette le trafic d'épaississement de main hérité que lorsqu'il est réellement présent.

## Exercices

1. Modifier la version du protocole d'une demande en `2027-01-01`Confirmez que le code d' erreur est `-32022`et les données annoncent la version prise en charge.
2. Retirez `io.modelcontextprotocol/clientCapabilities`Confirmer que le serveur n'utilise pas les capacités de la première demande.
3. Inverser le registre des outils en mémoire.`tools/list`retourne toujours le même ordre déterministe.
4. Le changement`cacheScope`de `public`à `private`- Expliquer les contextes d'autorisation qui peuvent réutiliser la réponse dans chaque cas.
5. Ajouter une option `clientInfo`test d'omission. La demande devrait rester valide parce que l'identité du client est recommandée et non requise.

## Les termes clés

| Term | Meaning |
|------|---------|
| Stateless protocol | Every request supplies the metadata needed to interpret it |
| Request metadata | Version, client capabilities, and recommended client identity in `params._meta` |
| `server/discover` | Mandatory server method for versions, capabilities, instructions, and identity |
| `resultType` | Discriminator on every successful modern result |
| Cacheable result | Result that includes required `ttlMs` and `cacheScope` hints |
| Protocol era | Modern per-request metadata or legacy connection-scoped initialization |
| Transport lifetime | Process, connection, or response-stream lifetime, not protocol session state |
| `-32022` | Unsupported protocol version error with requested and supported versions |

## Pour en savoir plus

- [MCP Architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP Base Protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
