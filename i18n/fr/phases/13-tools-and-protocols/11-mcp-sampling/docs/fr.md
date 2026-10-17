# Entrée du modèle MCP: échantillonnage de la migration et du MRTR des apatrides

> MCP 2026-07-28 déprécie le prélèvement d'échantillons pour de nouveaux modèles et supprime le canal de demande serveur-client. Si un flux de travail existant a encore besoin du modèle du client, le serveur renvoie un `input_required`Le logiciel de logique devient explicite, limité et sans état à la couche de protocole.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Expliquez pourquoi l'échantillonnage est dépassé dans MCP 2026-07-28 et choisissez l'intégration directe par défaut du modèle pour les nouveaux serveurs.
- Implémenter un flux de travail de compatibilité qui comporte `sampling/createMessage`par le biais des demandes de plusieurs voyages aller-retour (MRTR).
- Mettez la révision du protocole et les capacités du client dans chaque demande `_meta`- Je suis désolé.
- Retour `resultType: "input_required"`et réessayez la méthode originale avec un nouvel identifiant JSON-RPC.
- Protéger l'intégrité `requestState`et le lier au principal, à la méthode, aux arguments et à l'expiration.
- Les boucles liées aux modèles assistés avec des contrôles de capacité, une approbation, une validation de réponse et une limite ronde.

## La décision prise avant le protocole

Un outil comme `summarize_repo`Il faut deux types de travail:

1. Travail déterministe: liste des fichiers, lecture des fichiers autorisés, validation des chemins et assemblage du contenu.
2. Modèle de travail: sélectionner des fichiers représentatifs et synthétiser le résumé.

Vous avez maintenant deux architectures valides.

### Nouveau serveur: intégrer directement avec un fournisseur de modèle

C'est la version par défaut actuelle. Le serveur possède la sélection de modèle, les informations d'identification, les budgets, les retries et l'observabilité.`tools/call`résultat pour le client du MCP.

Choisissez ceci lorsque le serveur est déjà un service hébergé ou lorsque le comportement du modèle prévisible est plus important que d'utiliser le modèle de l'hôte.

### Flux de travail d'échantillonnage existant: migrer vers MRTR

L'échantillonnage existe toujours pendant sa fenêtre de dépréciation. Un serveur ciblant 2026-07-28 ne peut pas envoyer un live `sampling/createMessage`La demande est renvoyée au client.`InputRequiredResult`- Je suis désolé .

Choisir ce chemin de compatibilité uniquement lorsque vous utilisez le modèle du client et les informations d'identification est une exigence réelle du produit.

## Le contrat de l'appartenance à une nation

Le protocole de juillet 2026 n' a pas de`initialize`- Le taux de change est de 0,5%.`notifications/initialized`, et non `Mcp-Session-Id`Chaque demande contient les informations qui vivaient dans la poignée de main:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Le serveur valide la révision sur chaque demande. Une version manquante ou non-string est des paramètres invalides, `-32602`Une chaîne non prise en charge revient .`-32022`avec des données exactes `{"supported":["2026-07-28"],"requested":"<client version>"}`Une capacité d' échantillonnage manquante est retournée .`-32021`avec `data.requiredCapabilities`à `{"sampling":{}}`- Je suis désolé .

Une enveloppe sans JSON-RPC `id`est une notification. Le récepteur peut la traiter, mais elle n'émite ni une réponse de succès ni une réponse d'erreur. Un adaptateur HTTP diffusable renvoie `202 Accepted`sans organisme pour une notification acceptée.

Le serveur met également en œuvre `server/discover`avec le juste`supportedVersions`la clé, les capacités, `ttlMs`, et `cacheScope`afin qu'un client puisse apprendre et cacher le contrat du serveur avant d'appeler un outil.`tools`, le serveur met également en œuvre les obligations obligatoires `tools/list`- C' est déterministe .`summarize_repo`le descripteur comprend un objet valide `inputSchema`- Je suis là .`resultType: "complete"`, les métadonnées d'identité du serveur, et les indices de cache public.

Chaque résultat moderne réussi a un facteur de discrimination:

- `resultType: "complete"`signifie que l'opération est terminée.
- `resultType: "input_required"`signifie que le client doit répondre aux demandes intégrées et réessayer.
- Les extensions peuvent définir des types de résultats supplémentaires.`"task"`Dans la leçon 13.

## Une ronde MRTR

Le serveur ne peut pas appeler le client pendant la gestion de la demande. Il renvoie le résultat suivant:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

Le client vérifie qu'il prend en charge le Sampling, applique ses politiques d'approbation et de modèle, et obtient une réponse de modèle.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

La répétition n'est pas une continuation d'une session de protocole, mais une nouvelle demande qui répète la méthode et les arguments originaux, ajoutant seulement les arguments de la ronde actuelle `inputResponses`, et échos .`requestState`octet par octet.

Le MRTR est autorisé uniquement sur `tools/call`- Je suis là .`prompts/get`, et `resources/read`Un serveur ne doit pas revenir .`input_required`à partir de méthodes non liées.

## État multi-roundable

Cette leçon a besoin de deux modèles:

1. `pick_files`renvoie un tableau JSON.
2. `summary`retourne la prose finale.

Chaque réessay ne contient que les réponses de cette ronde. Le serveur met donc la phase et les données intermédiaires validées dans la prochaine ronde `requestState`- Je suis désolé .

Traiter cette valeur comme contrôlée par l'attaquant.

- le principal authentifié, non déclaré par lui-même `clientInfo`Le dépôt de la commission
- la méthode d'origine;
- un résumé des arguments originaux;
- une durée de validité courte;
- la phase actuelle et les valeurs intermédiaires validées.

Utilisez HMAC lorsque la confidentialité n'est pas requise. Utilisez le cryptage authentifié lorsque le client ne doit pas lire l'état. Rejetez une mauvaise signature, une valeur expirée, un changement de principe ou un changement d'arguments avec `-32602`- Je suis désolé .

Le client ne doit pas analyser ou modifier `requestState`Son seul travail est de faire écho à la chaîne exacte lors de la nouvelle tentative.

## Les préférences de modèle sont des indices

`costPriority`- Je suis là .`speedPriority`, et `intelligencePriority`Les préférences de la clientèle sont indépendantes, elles ne sont pas une répartition de probabilité et ne doivent pas être sumées à une seule.

Je le garde .`includeContext`à`"none"`Si vous conservez un flux d'échantillonnage ancien, les autres modes contextuels augmentent le risque de fuite et sont eux-mêmes dépassés.

## Les invariables de sécurité

Le client est la limite de confiance pour les demandes d'échantillonnage intégrées.

- Montrez à l'utilisateur ce que le serveur demande au modèle lorsque la politique exige l'approbation.
- Un serveur malveillant peut créer une boucle de dépense.
- Valider chaque réponse d'échantillonnage avant de l'utiliser comme nom de fichier, URL ou entrée d'outil.
- Limiter les octets et les jetons par tour.
- Refuser une demande d'entrée qui n'a pas été déclarée dans les capacités actuelles du client.
- Gardez les résultats du modèle hors des décisions d'autorisation.
- Enregistrez la méthode d'origine et la clé de demande d'entrée sans enregistrer le contenu de prompt sensible.

`clientInfo`et `serverInfo`Les données de l'affichage et du diagnostic sont des métadonnées.

```figure
t3-sampling-flip
```

## Faites-le

`code/main.py`met en œuvre le flux complet à deux tours sans package tiers:

- `server/discover`Retour `supportedVersions`, annonce le support des outils et renvoie des indices de cache.
- `tools/list`renvoie une définition déterministe, cacheable `summarize_repo`descripteur avec un schéma d'entrée d'objet.
- `tools/call`valides les métadonnées par demande.
- Le premier résultat s' intègre `sampling/createMessage`pour la sélection des fichiers.
- La première réessaye de valider le résultat du modèle et d'intégrer une deuxième demande.
- Protégé par le HMAC `requestState`Il est intermédiaire entre les demandes indépendantes.
- Le résultat final utilise `resultType: "complete"`- Je suis désolé .

Le faux modèle d'hôte rend l'exemple déterministe.`fake_host_model`La machine d'état du côté du serveur devrait rester déterministe et testable.

## Utilisez-le

À partir de la racine du référentiel:

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

Les points de contrôle attendus:

- Discovery renvoie un résultat complet avec `ttlMs`et `cacheScope`- Je suis désolé .
- La découverte d' outils renvoie le même descripteur trié avec `resultType`, l'identité du serveur, et des indices de cache.
- Les capacités manquantes et les versions non prises en charge utilisent exact `-32021`et `-32022`données d'erreur.
- Une notification sans id ne produit aucune réponse JSON-RPC.
- Les identifiants de demande sont `[1, 2, 3]`, prouvant que chaque tour MRTR est indépendant.
- Les deux premiers résultats sont `input_required`- Je suis désolé .
- Le résultat final est `complete`et contient les fichiers sélectionnés et le résumé.
- Le changement des arguments originaux lors d'une nouvelle tentative échoue à la vérification de l'état de la demande.

## La faire partir

`outputs/skill-sampling-loop-designer.md`Il décide d'abord si le Sampling doit être supprimé en faveur de l'intégration directe du modèle. Si la compatibilité est requise, il produit les tours MRTR, la liaison d'état, la porte de capacité, le budget, la validation et le plan de suppression.

## Exercices

1. Modifier la réponse de sélection de fichiers à JSON non valide. Confirmer les retours du serveur `-32602`au lieu de faire confiance à la production du modèle.
2. Le changement`audience`Expliquez pourquoi l'état scellé bloque la réutilisation des demandes croisées.
3. Ajoutez une troisième ronde qui demande à l'hôte de critiquer le résumé.
4. Supprimer le Sampling en remplaçant le faux appel-retour de l'hôte par un adaptateur de modèle appartenant au serveur.
5. Ajouter un test d'expiration à l'aide d'une valeur d'état qui est une seconde après sa date limite.

## Les termes clés

| Term | Meaning in 2026-07-28 |
|------|------------------------|
| Sampling | Deprecated feature that asks the client's model for a completion |
| MRTR | Stateless retry pattern for client input required during a request |
| `InputRequiredResult` | Result with `resultType: "input_required"` |
| `inputRequests` | Server-assigned map of embedded elicitation, sampling, or roots requests |
| `inputResponses` | Current round's client results keyed like `inputRequests` |
| `requestState` | Opaque server state echoed exactly by the client and verified by the server |
| `resultType` | Required discriminator for modern MCP results |
| Direct model integration | Recommended replacement for new servers that need model inference |
| Capability gate | Rule that prevents sending an embedded request the client did not advertise |
| Loop budget | Maximum rounds, tokens, bytes, time, and spend allowed for the operation |

## Compatibilité avec l'héritage

Un client fixé à 2025-11-25 peut encore utiliser l' ancien serveur initié `sampling/createMessage`Ne faites pas de la voie de session l'architecture d'un serveur 2026-07-28.

Les KDD officiels peuvent traduire la modernité `input_required`Ce shim est une limite de compatibilité, pas la permission d'ajouter une nouvelle logique dépendante de la session.

## Pour en savoir plus

- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP Sampling deprecation](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
