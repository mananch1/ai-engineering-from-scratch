# Résultats et requêtes du MCP: Context adressable pour les serveurs sans État

> Les outils effectuent des opérations. Les ressources exposent le contenu adressable. Il invite les modèles de messages sélectionnés par l'utilisateur. Un bon serveur MCP garde ces contrats séparés et prévisibles.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07 (Building an MCP Server), Phase 13, Lesson 09 (MCP Transports)
**Time:** ~60 minutes

## Objectifs d'apprentissage

- Choisissez entre les outils, les ressources et les instructions de l'intention du consommateur.
- Faire de la publicité sur la ressource et la surface rapide par voie obligatoire `server/discover`- Je suis désolé .
- Construire une déterministe `resources/list`et `prompts/list`Les résultats.
- Appliquer `ttlMs`et `cacheScope`sans fuite de données spécifiques à l'utilisateur.
- Retourner l' erreur JSON-RPC `-32602`pour une URI de ressource invalide ou inconnue.
- Ouvrez une`subscriptions/listen`Retour de réponse POST et corréler chaque événement par ID d'abonnement.
- Traiter le contenu des ressources et les modèles de demande comme une sortie de serveur non fiable.

## Commencez par le consommateur

La façon la plus simple d'abuser de MCP est de commencer par le code d'implémentation. Une requête de base de données devient un outil parce que les fonctions sont familières. Un flux de travail réutilisable devient une ressource parce qu'il est stocké dans un fichier. Un prompt devient une politique cachée parce que l'hôte peut l'injecter.

Commencez par qui choisit et ce qu'ils attendent.

| Primitive | Primary intent | Selection owner | Typical result |
|---|---|---|---|
| Tool | Perform an operation | Model or application | Structured action result |
| Resource | Read content at a URI | Host, application, or user | Text or binary content |
| Prompt | Start a reusable message workflow | User through host UI | One or more prompt messages |

Une note à `notes://note-1`est une ressource car elle est un contenu adressable. `delete_note`est un outil parce qu'il change l'état. `review_note`est une demande parce qu'un utilisateur choisit un flux de travail de révision préparé.

Ne faites pas apparaître une seule opération comme les trois pour paraître complète.

## Le dossier des statuts de 2026-07-28

Cette leçon vise la révision du protocole MCP `2026-07-28`. Il n'y a pas de serrage de main d'initialisation ou de session de protocole dans ce profil.`_meta`Les clés.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Un serveur doit mettre en œuvre `server/discover`- Ses résultats annoncent
les versions, les capacités de ressources et de prompt, l'identité de la mise en œuvre, et
Un client peut appeler une autre méthode directement, mais la découverte lui donne
une capture instantanée stable avant de créer une interface utilisateur.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

Un résultat normal est annoncé .`"resultType": "complete"`La réponse`_meta`identifie la mise en œuvre de la prestation de services avec `io.modelcontextprotocol/serverInfo`. Ces informations sont utiles pour le diagnostic. Ce n'est pas une identité d'authentification.`-32022`avec la révision demandée et les révision prises en charge par le serveur.

Le contrat sans statut modifie vos instincts de conception. Une liste ne peut pas dépendre d'un appel antérieur sur une connexion. L'autorisation peut modifier le jeu visible parce que les informations d'identification sont une entrée requise, mais l'historique de connexion ne doit pas.

## Les ressources sont des contrats URI stables

Une ressource est le contenu identifié par un URI. Conceptualisez l'URI avant le gestionnaire.

Les propriétés de l' IRU sont bonnes:

- Suffisamment stable pour marquer des livres ou passer entre les demandes.
- Nommé à l'emplacement du serveur.
- Indépendante d'un identifiant de processus ou d'une connexion.
- Validée avant l'accès au stockage.
- Autorisé à chaque lecture.

`notes://note-1`est mieux que `note-1`Un serveur de fichiers peut utiliser `file://`URI, mais il doit toujours vérifier les limites des répertoires configurés après avoir résolu les liens symboliques et les segments relatifs.

`resources/list`Retourne les ressources actuellement visibles à l'appelant. trier par une clé stable telle que URI. L'ordre déterministe empêche les sauts de cache bruyants, le changement de snapshots et les interfaces d'accueil qui sautent entre les mises à jour.

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read`renvoie un ou plusieurs éléments de contenu. Un URI inconnu n'est pas une lecture vide réussie. La spécification des ressources actuelle attribue des URI de ressources invalides ou inconnus à des paramètres invalides JSON-RPC, code `-32602`- Je suis désolé .

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

Cette distinction permet à un client de séparer l'absence d'un document vide valide.

### Templates de ressources

Un modèle de ressources décrit une famille d'URI paramétrisés. Utilisez-en une lorsque vous énumérez chaque élément concret serait cher ou illimité.`notes://projects/{project}/decisions/{decision}`indique à un client comment former une adresse valide sans retourner chaque décision.

Un modèle ne fait pas défaut de validation. Analysez les variables, appliquez des autorisations, mettez en œuvre des limites de longueur et de caractères et construisez des requêtes de stockage avec des paramètres typés. Ne concateniez jamais une queue URI arbitraire dans un chemin du système de fichiers ou une déclaration de base de données.

### Le contenu n'est pas une instruction fiable

Le texte de la ressource peut contenir une injection rapide, des secrets, des commandes trompeuses ou des balises malformées. L'hôte doit préserver la provenance et traiter le contenu de la ressource comme des données. Le serveur doit limiter la taille du contenu, renvoyer un type MIME précis, rédiger des champs auxquels l'appelant ne peut pas accéder et éviter de renvoyer des enregistrements non liés.

## Les commandes sont des modèles contrôlés par l'utilisateur

Les requêtes MCP sont conçues pour une sélection explicite de l'utilisateur. Un hôte peut les rendre sous forme de commandes de tablier, d'éléments de menu ou de boutons de flux de travail. Le protocole ne nécessite pas une interface utilisateur unique.

`prompts/list`chaque prompt a besoin d'un nom stable, d'une description utile et de déclarations d'arguments qui permettent à l'hôte de collecter des entrées avant `prompts/get`- Je suis désolé .

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get`Il ne remplace pas les instructions système de l'hôte. L'hôte décide comment les messages retournés entrent dans le contexte du modèle et garde sa propre politique de confiance à la plus haute priorité.

Valider les arguments de prompt à la limite du serveur. Une URI de prompt doit passer la même vérification d'autorisation qu'une lecture directe de ressource. Ne faites pas d'un prompt un canal secondaire autour de l'accès à la ressource.

## Les indices de cache font partie de la correction

`ttlMs`indique au client combien de temps un résultat peut être réutilisé. `cacheScope`décrivant qui peut partager cette valeur caché.

| Scope | Meaning | Typical use |
|---|---|---|
| `public` | May be reused across users when authorization permits | Public prompt catalog |
| `private` | Bound to the requesting user or credential context | User-owned note content |

Choisissez un TTL à partir du taux de changement des données et des dommages causés par la latence.

Le MCP ne définit que `public`et `private`comme `cacheScope`Pour un résultat secret ou en mutation rapide, retourner `cacheScope: "private"`avec `ttlMs: 0`, puis appliquer toute règle stricte de non-entreposage dans la politique de cache hôte. `no-store`n'est pas elle-même un MCP `cacheScope`La valeur.

Les indices de cache ne remplacent jamais l'autorisation. Une clé de cache doit inclure toutes les dimensions de la demande qui modifient la visibilité, y compris le curseur localisateur, l'utilisateur, la portée, le lieu et la pagination. Si un cache partagé ne peut pas exprimer ces dimensions en toute sécurité, utilisez `private`avec un TTL zéro et une politique de non-boutique au niveau de l'hôte.

## Les abonnements Utilisez un flux de réponse ouvert par le client

Le modèle d' abonnement moderne remplace le précédent `resources/subscribe`RPC et le vieil point final de l'événement HTTP GET.

Le client envoie `subscriptions/listen`En HTTP Streamable, il s'agit d'un POST dont la réponse reste ouverte en tant que flux SSE.`notifications`Un serveur ne doit pas fournir de types de notification qui n'ont pas été demandés.

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

L'ID de demande est l'ID d'abonnement. Avant tout événement demandé, le serveur envoie `notifications/subscriptions/acknowledged`Son filtre ne contient que le sous-ensemble accepté par le serveur.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

Chaque événement ultérieur sur ce flux contient les mêmes métadonnées.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

La notification indique que la ressource a changé.`resources/read`Il n'assume pas que l'événement contient le nouveau document.

Plusieurs abonnements peuvent partager un canal stdio. L'ID d'abonnement permet au client de les démultiplexer. En HTTP, la fermeture du flux de réponse annule l'abonnement. Un serveur qui termine le flux retourne gracieusement une finale `resultType: "complete"`la réponse corrélative à la demande initiale.

Ne pas utiliser un flux d'abonnement comme session de protocole. Une lecture ultérieure est toujours une demande complète qui peut atteindre n'importe quelle instance de serveur en bonne santé.

```figure
t3-primitive-sort
```

## Laboratoire interactif

Utilisez la figure pour classer cinq fonctionnalités d'un tracker de projet: détails de la question, créer une question, modèle de révision de sprint, politique du projet et issue de clôture. Déterminez ensuite quelles listes peuvent être mises en cache publiquement, quelles lectures doivent rester privées et quelles ressources méritent des notifications de mise à jour.

Pour chaque classification, nommez le choix. Si le modèle effectue une action, utilisez un outil. Si un hôte lit un contenu à adresses URI, utilisez une ressource. Si l'utilisateur démarre un flux de travail de message préparé, utilisez un prompt.

## Laboratoire de pratique

Exécutez le simulateur à partir de la racine du référentiel:

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Vérifiez la transcription dans cet ordre:

1. Confirme `server/discover`Il publie la révision actuelle et les deux capacités.
2. Confirmer que les résultats de la liste sont triés et utilisés `resultType: "complete"`- Je suis désolé .
3. Confirmer la liste et lire les résultats contiennent des indices de cache intentionnels.
4. Modifiez l' URI de lecture en `notes://missing`et observez`-32602`- Je suis désolé .
5. Confirmer la confirmation de l'abonnement précède l'événement de ressource.
6. Confirmer l' événement et fermer gracieusement les deux porter l' ID d' abonnement `5`- Je suis désolé .

Le modèle Python n'ouvre pas une connexion HTTP réelle. Il représente les messages qu'un SDK doit placer sur le flux de réponse à la demande.

## Artéfacts expédiés

`outputs/skill-primitive-splitter.md`est une revue de conception réutilisable pour la sélection primitive MCP. Il vérifie maintenant la découverte déterministe, la portée du cache, le comportement URI non valide et les filtres d'abonnement modernes.

La leçon est aussi des navires .`assets/primitive-split.svg`, une version statique de la limite de primitive et d'abonnement pour l'étude hors ligne.

## Vérifiez

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Résultat attendu: le programme principal imprime une transcription JSON et la commande de test rapporte au moins douze tests passés.

## Connexion Capstone

Utilisez ce contrat lorsque votre serveur capstone expose des connaissances adressables en plus des actions. Inclure un instantané de catalogue déterministe, une lecture de ressource autorisée, une résolution rapide, un cas URI invalide et une transcription d'abonnement.

Votre preuve doit montrer qu'aucune liste ne dépend de l'historique de connexion et qu'un événement d'abonnement ne donne jamais accès à la ressource sous-jacente.

## Exercices

1. Ajouter un `notes://projects/{project}/notes/{id}`le modèle de ressource et de valider les deux variables.
2. Ajouter une page à `resources/list`tout en préservant l'ordre déterministe.
3. Modifier une ressource en `cacheScope: "private"`avec `ttlMs: 0`, ajouter une politique de non-boutique au niveau de l'hôte, et expliquer la menace qui justifie les deux contrôles.
4. Ajouter un abonnement à la liste de changements et prouver qu' aucun événement n' est envoyé lorsque le filtre est omis `promptsListChanged`- Je suis désolé .
5. Créez deux abonnements simultanés et prouvez que chaque événement porte le bon identifiant de demande.
6. Ajouter une autorisation sujet au gestionnaire de lecture et prouver une entrée de cache ne peut pas croiser des sujets.

## Les termes clés

- **Resource:**Contenu adressé aux URI exposé par un serveur MCP.
- **Prompt:**Un modèle de message contrôlé par l'utilisateur exposé par un serveur MCP.
- **Deterministic list:**Un résultat de découverte avec un membre stable et une commande pour les mêmes entrées de demande.
- **`ttlMs`:**La durée de fraîcheur est en millisecondes.
- **`cacheScope`:**La limite de partage pour un résultat caché.
- **`subscriptions/listen`:**Une demande de longue durée dont le flux de réponse fournit des notifications explicitement filtrées.
- **Subscription ID:**L'identifiant de demande d'écoute original, répété dans les métadonnées de notification.
- **Invalid parameters:**Erreur JSON-RPC `-32602`, utilisé pour un URI de ressource non valide ou inconnu.
- **Unsupported protocol version:**Erreur JSON-RPC `-32022`, y compris `supported`et `requested`Les révision.
- **`server/discover`:**Métode de serveur obligatoire qui renvoie les révisions, les capacités, l'identité et les indices de cache optionnels pris en charge.

## Pour en savoir plus

- [MCP 2026-07-28 Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 2026-07-28 Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP 2026-07-28 Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Caching](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching)
