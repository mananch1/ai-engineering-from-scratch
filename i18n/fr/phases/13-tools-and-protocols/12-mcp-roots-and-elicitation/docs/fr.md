# L'étendue explicite et l'obtention d'un statut de stateless

> Les racines sont dépassées dans MCP 2026-07-28 et n'ont jamais été une boîte à sable de sécurité. Mettez la portée dans les arguments ou les URIs de ressources visibles, autorisez-les sur le serveur et utilisez MRTR lorsqu'un outil a vraiment besoin de l'entrée de l'utilisateur. L'utilisateur voit la décision, le modèle voit la poignée et toute instance du serveur peut traiter la répétition.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 11 (stateless MRTR)
**Time:** ~60 minutes

## Objectifs d'apprentissage

- Remplacez les racines dépassées par des paramètres explicites de l'espace de travail, des URIs de ressources ou une configuration de serveur.
- Indices de portée séparés de l'autorisation, de la contention du chemin et du sandboxing du système d'exploitation.
- Mode de livraison `elicitation/create`par un MRTR `input_required`Le résultat.
- La publicité de l'appui à l'exécution dans les capacités du client par demande et le rejet des modes non pris en charge.
- Valider`accept`- Je suis là .`decline`, et `cancel`Les résultats sont différents.
- Lier la confirmation destructive à un principal authentifié, les arguments originaux, le nombre de candidats et l'expiration.

## Deux problèmes qui se ressemblent

Un outil de notes reçoit cette demande: " Supprimer l'ancien rapport TPS. "

Le serveur doit répondre à deux questions différentes.

1. Quel espace de travail cette opération peut toucher ?
2. Quelle des trois notes correspondantes l'utilisateur voulait dire ?

Le premier est la portée et l'autorisation. Le second est la désambiguation interactive. Le mélange des deux conduit à des conceptions dangereuses, comme traiter un dossier fourni par le client comme une preuve que l'appelant peut supprimer tout ce qui est à l'intérieur.

## Les racines sont une surface de migration

Les modifications précédentes de MCP permettaient à un client de faire de la publicité Roots et de notifier un serveur lorsque la liste changeait.

Le MCP 2026-07-28 est déprécié `roots/list`et `notifications/roots/list_changed`Pour les nouveaux modèles, je préfère l'un de ces remplacements explicites:

- Une .`workspaceUri`ou `directory`argument d'outil lorsque la portée varie par appel.
- Une URI de ressource lorsque l'opération cible déjà une ressource.
- Configuration du serveur lorsqu'un déploiement possède un espace de travail fixe.
- Un système de fichiers de processus ou de fichiers emprisonnés lorsque le code doit être techniquement incapable de s'échapper.

Si une intégration existante de 2026 à 2728 est encore nécessaire `roots/list`pendant la fenêtre de dépréciation, le serveur l'intègre dans MRTR `inputRequests`Il ne doit pas envoyer une demande inverse en direct. C'est un adaptateur de migration; les nouveaux opérateurs devraient accepter une portée explicite.

Le modèle peut voir et répéter une poignée explicite.

### La règle des trois couches

Une URI explicite ne s'autorise toujours pas.

1. **Authorization:**Est-ce que ce directeur authentifié est autorisé à utiliser cet espace de travail ?
2. **Containment:**L'URI cible normalisée reste-t-elle à l'intérieur de la limite de l'espace de travail autorisé?
3. **Sandbox:**Le système d'exploitation peut-il empêcher un serveur compromis de s'échapper ?

Le serveur exécutable conserve une liste d'allowlist des URIs autorisés de l'espace de travail, normalise les chemins codés en pourcentage, vérifie une limite réelle du composant chemin et vérifie à nouveau la contention immédiatement avant la suppression.

Les vérifications naïves de préfixe de chaîne sont erronées:

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

Les deux chemins hostiles commencent par une chaîne trompeuse. Normalizez d'abord, puis comparez les composants du chemin. Un serveur de système de fichiers de production doit également se défendre contre les courses de liens symboliques et la sémantique du chemin spécifique à la plateforme.

## L'élicitation est toujours présente, mais la livraison a changé

L' élicitation est la fonction client actuelle pour collecter les entrées des utilisateurs pendant `tools/call`- Je suis là .`prompts/get`ou `resources/read`. Le nom de la méthode reste `elicitation/create`Ce qui a changé, c'est la direction du fil.

Un serveur 2026-07-28 n'envoie pas une requête JSON-RPC inverse.`InputRequiredResult`- Le numéro de la liste:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

L'hôte rend le formulaire. L'utilisateur peut accepter, refuser explicitement ou rejeter. Le client réessaye ensuite l'original `tools/call`avec une nouvelle pièce d'identité:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Il n'y a pas de session de protocole entre les deux appels. Le serveur vérifie l'état de l'écho, valide la réponse contre le schéma attendu, vérifie que la note sélectionnée était dans le jeu de candidats signé, autorise à nouveau l'espace de travail, vérifie à nouveau la contention, puis supprime.

## Les négociations de capacité sont effectuées sur demande

Un client qui prend en charge l'obtention de formulaires déclare:

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

Une capacité d'évitement vide,`"elicitation": {}`, reste équivalent à un soutien à la compatibilité uniquement sous forme.`"elicitation": {"form": {}}`Il prend également en charge le mode formulaire.`"elicitation": {"url": {}}`Le serveur ne doit pas intégrer un mode absent des capacités de la demande actuelle, même si une demande antérieure l'a annoncé.

Chaque demande est aussi accompagnée .`io.modelcontextprotocol/protocolVersion`. Une version manquante ou non-string est retournée `-32602`Une chaîne non prise en charge revient .`-32022`avec exactitude `supported`et `requested`données. les données manquantes ou uniquement par URL sont retournées `-32021`avec `data.requiredCapabilities`à `{"elicitation":{"form":{}}}`- Je suis désolé .

Une enveloppe sans JSON-RPC `id`est une notification. le traiter sans émettre une réponse de succès ou d'erreur JSON-RPC. sur HTTP en streaming, une notification acceptée reçoit `202 Accepted`sans corps.

`clientInfo`Il doit être inclus pour le diagnostic, mais il est autodéclaré et ne peut pas identifier l'utilisateur pour autorisation.

Le serveur implémentera `server/discover`et les retours `supportedVersions`, les capacités,`ttlMs`, et `cacheScope`avec `resultType: "complete"`Il ne fait pas de publicité pour ce design moderne.`tools/list`Ce résultat renvoie la déterministe`notes_delete`un objet valide `inputSchema`, les métadonnées d'identité du serveur, et les indices de cache public.

## Mode de formulaire

Le mode de formulaire utilise un schéma JSON restreint conçu pour les dialogues utilisables. La racine est un objet et ses propriétés sont des champs primitifs plats ou des matrices enum pris en charge. Les objets profondément nichés et les schèmes de documents à usage général ne font pas partie d'un dialogue de confirmation.

Utiliser le mode formulaire pour:

- choisir l'un des candidats;
- la confirmation d'une opération destructive;
- la collecte de préférences non sensibles;
- La collecte d'un petit nombre de valeurs doit être déterminée par l'utilisateur et non par le modèle.

Ne pas utiliser le mode formulaire pour les mots de passe, les clés API, les jetons d'accès ou les informations de paiement. Ces secrets passeraient par le client MCP et pourraient atteindre les journaux ou le contexte du modèle.

Le serveur valide à nouveau le contenu retourné.

## Mode d' URL

Le mode URL envoie une URL Web sécurisée pour une interaction hors bande:

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

Utilisez-le lorsque des informations sensibles doivent aller directement à un flux Web contrôlé par le serveur, comme l'autorisation d'un tiers. Le client affiche la destination complète et obtient son consentement avant de l'ouvrir. Il ne doit pas pré-coller l'URL.

Un `accept`réponse signifie que l'utilisateur a accepté d'ouvrir l'URL. Il ne prouve pas que le flux externe a été terminé. Lors de la nouvelle tentative, le serveur vérifie son propre état et complète ou renvoie un autre `input_required`Le résultat.

L'exécution d'URL n'est pas un remplacement de l'autorisation entre le client MCP et le serveur MCP. C'est pour une interaction externe que le serveur MCP doit effectuer au nom de l'utilisateur. Le serveur doit lier l'utilisateur du navigateur au même principe authentifié qui a commencé l'opération MCP.

## Les branches de réponse

Traiter les actions comme des décisions relatives au produit et non comme des aliases:

| Action | Meaning | Safe server behavior |
|--------|---------|----------------------|
| `accept` | User submitted the interaction | Validate content and continue |
| `decline` | User explicitly refused | Return a complete, non-error refusal outcome |
| `cancel` | User dismissed or could not finish | Stop safely and allow a later retry |

Ne jamais interpréter le contenu manquant comme un consentement.

## Protéger l'état destructeur du MRTR

La liste des candidats ne peut pas être enregistrée uniquement dans une valeur Base64 non signée ou non.

La leçon signale une charge utile de l'État contenant:

- le principal authentifié;
- méthode d'origine;
- digestion de `workspaceUri`et `title`Le dépôt de la commission
- les identifiants de notes autorisés indiqués dans le formulaire;
- phase de fonctionnement;
- une durée de validité courte.

Avant la mutation, le serveur vérifie également l'enregistrement de notes en direct. Cela capture les courses de suppression et une cible déplacée hors de l'espace de travail après que le formulaire a été affiché.

Pour une action financière unique ou irréversible, le HMAC seul n'empêche pas la reproduction d'un état valide dans les délais de son expiration. Conserver et consommer un nonce exactement une fois dans un magasin de répétition partagé par chaque instance de traitement. La leçon injecte un stock limité et taillé en TTL et conserve sa revendication atomique tout en effectuant la suppression en mémoire. Une base de données de production devrait associer la demande nonce et la mutation dans une transaction ou une limite de rédaction conditionnelle équivalente.

Valider l'interaction avant de réclamer le nonce.`cancel`Il ne produit aucune mutation et laisse l'état réactif jusqu'à expiration.`decline`est terminal, donc la leçon consomme le nonce sans supprimer quoi que ce soit.

```figure
t3-roots-boundary
```

## Faites-le

`code/main.py`démontre une modernité `notes_delete`outil:

- `tools/list`renvoie un descripteur déterministe et cacheable avec l'espace de travail et le schéma de titre requis.
- La portée est explicite `workspaceUri`- Je suis d'accord.
- La configuration du serveur autorise cet espace de travail pour le principal de la leçon.
- La normalisation URI rejette la confusion des préfixes et le traversage codé.
- Toute suppression destructive nécessite une élimination en mode forme.
- L' élicitation se déplace à l' intérieur .`resultType: "input_required"`- Je suis désolé .
- Signé .`requestState`lie la liste exacte des candidats et les arguments originaux.
- Un magasin de répétition injecté rejette le même état accepté ou refusé dans toutes les instances de serveur.
- La nouvelle tentative utilise un nouveau identifiant de demande et renvoie `resultType: "complete"`- Je suis désolé .

Le stockage de données est en mémoire, le comportement du protocole est facile à inspecter.

## Utilisez-le

À partir de la racine du référentiel:

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

Les points de contrôle attendus:

- Discovery fait de la publicité pour des outils sans Roots.
- Les résultats de la découverte des outils `notes_delete`avec `resultType`, l'identité du serveur, et des indices de cache.
- Requête d' id`1`renvoie le formulaire en `inputRequests.delete_choice`- Je suis désolé .
- Requête d' id`2`l'état signé et complète la suppression.
- Un chemin de préfixe et un chemin de traversée codé échouent tous les deux à la contention.
- Un titre modifié ne peut pas réutiliser l'état de confirmation original.
- Un déclin laisse la note inchangée.
- Deux objets serveur partageant l'état de note et de répétition ne peuvent pas exécuter une confirmation.
- Les déclarations de formulaire vides et explicites fonctionnent, tandis que le support uniquement pour URL renvoie les informations exactes `-32021`les exigences relatives aux formulaires.
- Les défaillances de version non prises en charge utilisent la version exacte `-32022`la forme des données.
- Une notification sans id ne produit aucune réponse JSON-RPC.

## La faire partir

`outputs/skill-elicitation-form-designer.md`Il refuse de traiter les racines dépassées comme une boîte à sable ou de collecter des secrets via le mode formulaire.

## Exercices

1. Remplacez le magasin de lecture en mémoire par SQLite. Utilisez une transaction pour revendiquer le nonce et supprimer la note, puis prouvez que deux processus ne peuvent pas tous les deux s'engager.
2. Ajouter `url`Les données de référence sont fournies par les autorités compétentes pour les informations et les informations nécessaires à la mise en œuvre des procédures de certification.`inputResponses`- Je suis désolé .
3. Remplacez la carte de notes en mémoire par une base de données temporaire SQLite.
4. Ajouter une politique de lien symbolique pour une mise en œuvre réelle du système de fichiers. Expliquer pourquoi la contention léxicale URI seule ne peut pas arrêter une fuite de lien symbolique.
5. Conçuez un adaptateur 2025-11-25 qui correspond à la sortie du gestionnaire MRTR moderne à l'exécution initiale du serveur hérité.

## Les termes clés

| Term | Meaning in 2026-07-28 |
|------|------------------------|
| Roots | Deprecated informational workspace hints, not authorization or sandboxing |
| Explicit scope | Workspace, directory, or resource handle visible in request arguments |
| Containment | Normalized path-component check that keeps a target inside a boundary |
| Elicitation | Client feature for obtaining user input during an MCP operation |
| Form mode | In-band structured user input using a restricted flat schema |
| URL mode | Out-of-band interaction for sensitive or external workflows |
| MRTR | Stateless input-required result followed by a fresh retry |
| `requestState` | Opaque state echoed exactly and integrity-checked by the server |
| Decline | Explicit user refusal |
| Cancel | Dismissal or incomplete interaction without approval |

## Compatibilité avec l'héritage

Pour un paire fixé à 2025-11-25, `roots/list`- Je suis là .`notifications/roots/list_changed`, et le serveur en direct initié `elicitation/create`Étiquettez l'héritage de l'adaptateur. Ne laissez pas une liste Root ancienne contourner l'autorisation du serveur, et ne portez pas les hypothèses de session de protocole dans le gestionnaire moderne.

## Pour en savoir plus

- [MCP 2026-07-28 Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots deprecation](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
