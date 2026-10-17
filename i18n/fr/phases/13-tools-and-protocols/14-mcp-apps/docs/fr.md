# Applications de MCP sur le protocole des stateless

> Un résultat interactif est toujours un outil MCP et un échange de ressources. Le noyau 2026-07-28 rend cet échange autonome, tandis que l'extension Apps ajoute la surface du navigateur sandboxé.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Faites de la publicité dans les applications MCP via `server/discover`et des capacités d'extension à la demande.
- Déclarer une`ui://`une ressource sur un outil avant que l'outil ne soit appelé.
- Retourner les résultats complets des outils et des ressources sur le fil sans état de 2026-07-28.
- Séparer les applications `ui/initialize`message de pont de la poignée de main du noyau MCP supprimée.
- Appliquer la validation de l'origine, le sandboxing, le CSP et les autorisations de privilège minimum.

## Le problème

Un résultat texte peut décrire une chronologie.

MCP Apps résout le problème de présentation avec une extension optionnelle.`ui://`L'hôte peut récupérer et examiner cette ressource avant que l'outil ne s'exécute, la rendre dans un iframe sandboxé et médier toutes les actions de l'application via un pont JSON-RPC.

Le protocole de base a été modifié en 2026-07-28. Ne remplissez pas une application dans l'ancien cycle de vie de la connexion:

- Il n' y a pas de noyau .`initialize`demande ou `notifications/initialized`la notification.
- Il n' y a pas de`Mcp-Session-Id`- Je vous en prie.
- Chaque requête contient une version de protocole et des capacités de client en `params._meta`- Je suis désolé .
- Un serveur implémenté `server/discover`afin que les clients puissent inspecter les versions, les capacités de base et les extensions.
- Chaque résultat réussi a un résultat .`resultType`discriminateur.
- HTTP diffusable utilise un POST par requête. Les points d'entrée modernes GET et DELETE retournent 405.

Le pont Apps a encore une méthode nommée `ui/initialize`Il appartient au dialecte post-message iframe.

## Le concept

### Deux protocoles, une fonctionnalité

Gardez les couches explicites:

1. Le noyau du MCP porte `server/discover`- Je suis là .`tools/list`- Je suis là .`tools/call`- Je suis là .`resources/list`, et `resources/read`- Je suis désolé .
2. L'extension MCP Apps déclare l'interface utilisateur et définit le pont iframe-host.
3. Les règles de la boîte à sable du navigateur limitent ce que l'interface utilisateur peut atteindre.

L' identifiant d' extension est `io.modelcontextprotocol/ui`. Les deux pairs optent. Un client envoie une extension de support à l'intérieur de l' objet de capacités à chaque demande:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

`clientInfo`Il s'agit de données auto-déclarées, et non d'une identité d'autorisation.

### Découvrez avant de rendre

Le résultat de la découverte du serveur annonce l'extension:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

Le serveur doit prendre en charge la découverte. Un client n'est pas obligé d'appeler la découverte avant chaque action parce que chaque action porte ses propres capacités.

### Déclarer l'interface utilisateur sur la définition de l'outil

Le contrat moderne des applications lie une interface utilisateur à l' outil en `tools/list`- Le numéro de la liste:

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

Il s'agit de métadonnées pré-appelées délibérément. L'hôte peut précharger, mettre en cache et examiner la sécurité de l'HTML avant qu'un résultat ne demande de l'afficher.`_meta.ui.resourceUri`forme.

`tools/list`est cacheable dans le noyau courant.`ttlMs`, et `cacheScope`- Utilisez`private`lorsque les outils visibles varient selon l'utilisateur ou le jeton.

### Retournez les données, puis laissez l'hôte lier la vue

L'appel à l' outil renvoie le contenu ordinaire plus les données structurées:

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

L'hôte sait déjà quelle vue appartient à l'outil. Évitez d'inventer un nouveau bloc de contenu juste pour répéter l'URI.

### Servir l'application comme ressource

Le serveur annonce `resources`Il est donc nécessaire de mettre en œuvre les conditions de l'application de la directive.`resources/list`L'entrée de la liste déterministe comprend l'URI canonique, un nom stable, une description et un type MIME.`resultType`, les métadonnées d'identité du serveur, `ttlMs`, et `cacheScope`, comme la liste des outils déterministes.

L' hôte envoie `resources/read`. Sur le HTTP en streaming, la demande a:

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

Les valeurs d'en-tête et le corps JSON-RPC doivent être compatibles.`-32020`- Je suis désolé .

Le résultat contient les indices de ressources HTML et de cache:

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### Cache des ressources de l'interface utilisateur en tant que contenu exécutable

Une ressource d'application n'est pas interchangeable avec la prose ordinaire. Son entrée de cache peut exécuter du code bridge, rendre des données d'outil et demander des actions médiées par l'hôte.`ui://`URI, identité et version du serveur admis, digestion du contenu des ressources et contexte d'autorisation lorsque `cacheScope`Ne réutilisez jamais une ressource d'application privée entre les principaux principaux, car le HTML ou ses métadonnées de politique peuvent différer même lorsque l'URI est identique.

Invalider l' entrée lorsque `ttlMs`l'outil est épuisé.`_meta.ui.resourceUri`modification de liaison, modification de la version du serveur ou de la pin de descripteur admis, ou une souscription reconnue pour modifier les ressources nomme l'URI. Recharger et réappliquer CSP et révision des autorisations avant de remonter. Un iframe obsolète ne doit pas conserver des autorisations plus larges simplement parce qu'une nouvelle version de ressource n'a pas encore été chargée.

### Rejeter l'ambiguïté du fil avant la politique de fonctionnalité

La validation a un ordre délibéré. Validez d'abord la forme JSON-RPC et nécessitez des métadonnées de protocole de chaîne plus une carte de capacité du client objet. Ensuite, comparez les en-têtes de routage avec le corps.

| Condition | HTTP | JSON-RPC error |
|-----------|------|----------------|
| Header and body version, method, or name disagree | 400 | `-32020` |
| Header and body agree on an unsupported version | 400 | `-32022`, with `data` exactly `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read` lacks the Apps extension capability | 400 | `-32021`, with `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| Method is unknown | 404 | `-32601` |

Une notification JSON-RPC n' a pas été publiée `id`Un message HTTP accepté renvoie 202 avec un corps vide. Une erreur peut modifier l'état HTTP, mais elle ne peut toujours pas créer un corps d'erreur JSON-RPC pour une notification.

### La boîte à sable est une limite, pas un verdict de confiance

Un hôte contrôle l'iframe. L'application ne peut pas lire directement les cookies hôte, le stockage local ou la page DOM. Tout travail privilégié doit traverser le pont.

Utilisez les paramètres suivants:

- Laissez toutes les listes de domaines CSP vides, puis ajoutez uniquement les origines dont l'application a besoin. Utilisez `connectDomains`pour le retrait, XHR et WebSocket; utilisation `resourceDomains`pour les scripts, les styles, les images et les polices.
- Rassemblez le code et les données lorsque cela est pratique.
- Ne demandez pas de permis de localisation à moins que vous n'ayez besoin d'un appareil photo, d'un microphone ou d'une fonctionnalité visible.
- - Le pin`postMessage`à l'origine exacte et rejeter les événements de toute autre origine.
- Traitez les arguments des outils, les résultats des outils, le texte des ressources et les messages de pont comme des entrées non fiables.
- Garder le consentement de l'utilisateur dans l'hôte.

Ne copiez pas un fichier .`sandbox`L'hôte doit choisir des drapeaux en fonction du modèle d'origine de l'application et de sa propre conception d'isolement.

Un domaine autorisé est toujours un chemin d'exfiltration. `connectDomains: ["https://api.example.com"]`signifie que tout script qui s'exécute à l'intérieur de l'application peut y envoyer des données autorisées. La correspondance exacte de l'origine empêche la confusion de destination, mais elle ne décide pas si la charge utile est appropriée. Gardez l'accès à la connexion vide par défaut, évitez de placer des jetons porteurs dans l'iframe, éliminez les opérations de narrow proxy via l'hôte lorsque cela est pratique, limitez les tailles de réponse et de demande et vérifiez quelle action de l'utilisateur a causé chaque demande sortante. Traiter`resourceDomains`séparément de `connectDomains`; l'autorisation de charger une police ou un script ne devrait pas autoriser le chargement arbitraire de données.

### Le pont Apps a son propre cycle de vie

Le pont Apps est un dialecte JSON-RPC sur `postMessage`- Il peut échanger .`ui/initialize`et `ui/*`Les notifications et les méthodes de base peuvent être représentées par des proxy tels que `tools/call`- Je suis désolé .

La vue envoie `ui/initialize`avec `appInfo`et une `appCapabilities`l'objet. L'hôte renvoie ses capacités et le contexte de l'hôte.`ui/notifications/initialized`L'hôte doit attendre cette notification des applications avant d'envoyer des messages à la vue.

Cette poignée de main locale crée un pont entre un iframe et un cadre hôte. Elle ne négocie pas la version du protocole MCP, ne crée pas d'état de serveur ou ne crée pas une session de transport.`notifications/initialized`a été supprimé, tandis que les applications `ui/notifications/initialized`Une demande de base générée par un appel à l'outil relié est une nouvelle demande autonome avec un nouvel identifiant JSON-RPC et des métadonnées complètes de la demande.

### Contextes, actions et révocation de l'hôte

L'hôte reste l'autorité après l'initialisation du pont. Une vue peut demander une action d'outil, une navigation, une utilisation de planche à clips ou un autre effet privilégié uniquement grâce à une fonctionnalité que l'hôte annonce. L'hôte valide la demande typée, l'utilisateur actuel, la cible et les arguments, applique une politique d'approbation et peut la refuser. Un bouton cliquer et un message de pont valide expriment l'intention; aucun ne donne l'autorité.

Traiter le thème, la taille et l'accessibilité comme des changements de contexte hôte plutôt que des entrées de rendu unique:

- Appliquez les jetons de couleur et de typographie fournis par l'hôte, puis réagissez lorsque les préférences de thème ou de contraste changent.
- Laissez la vue indiquer les dimensions souhaitées, mais laissez le capteur d'accueil et appliquez la taille d'iframe afin que le contenu ne puisse pas échapper à sa mise en page ou créer des superpositions trompeuses.
- Préserver l'ordre du clavier, la mise au point visible, les noms accessibles, l'état du lecteur d'écran, le contraste suffisant, le zoom et le comportement à motion réduite à l'intérieur de l'iframe.
- Re-test transfert de mise au point entre les commandes hôte et les commandes affichage après redimensionnement et redirigement.

Les capacités peuvent être révoquées pendant que l'application est ouverte parce que l'utilisateur change de compte, modifie de politique, un serveur est mis en quarantaine ou que l'hôte restreint son consentement.`ui/initialize`. Lors de la révocation, rejeter les appels privilégiés en attente, arrêter l'activité réseau qui ne correspond plus à la politique, effacer l'état rendu sensible et remettre ou revenir au texte lorsque la ressource de l'interface utilisateur elle-même n'est plus admise.

### Le renversement fait partie du contrat.

Un serveur conscient des applications peut toujours servir les hôtes qui ne font pas de publicité pour l'extension de l'interface utilisateur:

- Retournez le même outil sans `_meta.ui`dans `tools/list`- Je suis désolé .
- Gardez un résultat texte utile pour `tools/call`- Je suis désolé .
- Je refuse .`resources/read`pour l'interface utilisateur avec une erreur de capacité manquante.
- Ne jamais supposer qu'un iframe existe lors de la décision de la finalisation de l'outil.

```figure
t3-ui-sandbox
```

## Faites-le

`code/main.py`Il valide l'enveloppe de requête actuelle et les valeurs de routage HTTP Streamable, annonce les applications via `server/discover`, liste les outils et les ressources, exécute l'outil et sert une ressource HTML autonome.

Le modèle reçoit déjà des corps analysés et des en-têtes de routage.`Content-Type`ou `Accept`. Utilisez la leçon 09 pour l' adaptateur HTTP en continu complet qui nécessite `Content-Type: application/json`et une `Accept`valeur contenant les deux `application/json`et `text/event-stream`- Je suis désolé .

- Je vais le faire.

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Inspectez quatre choses dans la sortie:

1. Chaque appel est indépendant.
2. Chaque demande a été faite .`_meta`les capacités.
3. `resources/list`renvoie un descripteur stable avant la lecture de toute ressource.
4. Chaque résultat a`resultType`et les métadonnées d'identité du serveur.
5. Aucun identifiant de session de base n'apparaît.

## Utilisez-le

Commencez par `server/discover`- Confirme .`io.modelcontextprotocol/ui`apparaît dans la carte d'extension du serveur.`tools/list`La première réponse déclare la ressource. la seconde reste un outil utilisable uniquement en texte.

Lire `ui://notes/timeline.html`- Recherchez dans le HTML .`hostOrigin`et le `event.origin`Ces deux lignes sont la preuve visible minimale que le pont n'utilise pas de cible à carte blanche.

## La faire partir

Cette leçon va à l' air .`outputs/skill-mcp-apps-spec.md`. Utilisez-le pour examiner un contrat d'application avant d'écrire le code-cadre. Il oblige l'auteur à indiquer l'enveloppe principale actuelle, la négociation d'extension, le retrait, la ressource UI, la politique de cache, le CSP, les autorisations, les méthodes de pont et la limite de consentement.

## Exercices

1. Modifiez la capacité du client à une carte d'extension vide.`tools/list`conserve l'outil mais supprime la liaison de l'interface utilisateur.
2. Envoyez-moi .`Mcp-Name: ui://notes/other.html`avec un corps qui lit la ligne de temps.`-32020`- Je suis désolé .
3. Modifier la ressource en `cacheScope: private`. Décrire la condition spécifique à l'utilisateur qui la justifie.
4. Mettez le script à `https://static.example.com/app.js`- Ajouter cette origine à `resourceDomains`et expliquer le risque de la nouvelle chaîne d'approvisionnement.
5. Ajouter un`notes_open`l'outil et le bouton de route cliquez sur l'hôte. Gardez l'approbation de l'utilisateur dans l'hôte.

## Les termes clés

| Term | Meaning |
|------|---------|
| MCP Apps | Optional extension for interactive HTML rendered by an MCP host |
| `io.modelcontextprotocol/ui` | Extension identifier advertised by both peers |
| `ui://` | Resource scheme for an App's UI template |
| `text/html;profile=mcp-app` | MIME type for MCP App HTML |
| `server/discover` | Current RPC for protocol and capability discovery |
| `resources/list` | Mandatory resource listing method when the server advertises resources |
| `resultType` | Required discriminator for modern successful results |
| `ui/initialize` | First Apps bridge request, separate from removed core initialization |
| `ui/notifications/initialized` | Apps View readiness notification sent after the host responds |
| CSP | Browser policy that restricts scripts, styles, images, and network origins |
| Text fallback | Tool behavior retained for a host without Apps support |

## Pour en savoir plus

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Apps overview](https://modelcontextprotocol.io/extensions/apps/overview)
- [MCP Apps build guide](https://modelcontextprotocol.io/extensions/apps/build)
- [Official extension support matrix](https://modelcontextprotocol.io/extensions/client-matrix)
