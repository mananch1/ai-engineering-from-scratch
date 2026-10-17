# Construire un serveur MCP: Python et TypeScript sans état

> Un serveur MCP moderne ne se souvient pas d'une poignée de main. Il valide les métadonnées sur chaque demande, exécute un traitement et renvoie un résultat typé.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## Objectifs d'apprentissage

- Implementation obligatoire `server/discover`pour les MCP `2026-07-28`- Je suis désolé .
- Valider la version du protocole et les capacités du client à chaque demande.
- Exposez les outils, les ressources et les instructions avec un ordre de liste déterministe.
- Retour `resultType`, l'identité du serveur, et des indices de cache sur les résultats corrects.
- Servir le même contrat sans État sur le studio à ligne délimitée en Python et TypeScript.

## Le problème

Un serveur qui stocke les capacités du client après le premier message est facile à construire et difficile à utiliser. Le même processus peut servir des clients séquentiels. Une demande à distance peut atterrir sur un travailleur différent. Une déclaration de capacité périmée peut fuir le comportement à travers les limites d'autorisation.

MCP `2026-07-28`L'application peut toujours conserver des notes durables, des emplois ou des manipulations d'état explicites. Ce qu'elle ne peut pas conserver est l'état de protocole caché qui change la façon dont une demande ultérieure est décodée.

Cette leçon construit un serveur de notes deux fois. Les versions Python et TypeScript utilisent seulement leurs bibliothèques standard pour le noyau du protocole.

## Le concept

### La boucle de dépêche moderne

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

Trois règles de studio sont toujours importantes:

- Écrivez seulement des messages JSON-RPC à stdout.
- Délimitez les messages avec une nouvelle ligne et versez chaque réponse.
- Sortez immédiatement lorsque le stdin atteindra l'EOF.

La durée de vie du processus est une durée de vie du transport.

### Validation de la demande

Chaque demande doit contenir:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Les deux premiers champs sont requis. `clientInfo`Il est recommandé de valider une forme d'identité actuelle, mais de ne pas la considérer comme une authentification.

Si la version n'est pas prise en charge, renvoyez le code `-32022`avec `requested`et `supported`. Les métadonnées de la demande manquantes sont des paramètres invalides, code `-32602`Ne remplis jamais les champs manquants d'un appel précédent.

### Découverte obligatoire

Les serveurs modernes doivent être mis en œuvre `server/discover`. Un résultat complet de découverte comprend des versions modernes prises en charge, des capacités, des instructions facultatives, des conseils de cache et l'identité du serveur en conséquence `_meta`- Le numéro de la liste:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

Discovery ne déverrouille pas le serveur.`tools/list`sans appeler la découverte parce que `tools/list`contient déjà les mêmes métadonnées de la demande.

### Les outils

`tools/list`Le système de référencement stable améliore la mise en cache des réponses et maintient le contexte du modèle stable.`ttlMs`et `cacheScope`- Je suis désolé .

`tools/call`renvoie des blocs de contenu et `isError`. Utiliser une erreur JSON-RPC lorsque l'enveloppe du protocole ou les paramètres de méthode sont invalides. Utiliser `isError: true`lorsque l'invocation d'un outil valide est exécutée mais que l'outil lui-même échoue.

Les annotations des outils restent des indices, pas des mesures d'application:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

L'hôte doit les utiliser pour la confirmation et la présentation.

### Les ressources

`resources/list`renvoie des descripteurs d'URI stables. `resources/read`Retourne le contenu typé.`2026-07-28`, donc les deux comprennent `ttlMs`et `cacheScope`- Je suis désolé .

Utilisation `cacheScope: "private"`Un cache partagé ne doit pas réutiliser une réponse privée dans des contextes d'autorisation.

La livraison moderne de changement n' utilise pas `resources/subscribe`Un client ouvre .`subscriptions/listen`et les demandes `resourceSubscriptions`La leçon 10 construit ce flux.

### Les instructions

`prompts/list`est cacheable et déterministe. `prompts/get`Le résultat du prompt rendu est complet, mais il ne fait pas partie de la liste cacheable ou des résultats de lecture qui nécessitent des indices de cache.

### Chaque résultat réussi est taillé

Les exemples utilisent un emballage pour chaque succès:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

Liste, lecture et découverte de gestionnaires ajouter `ttlMs`plus `cacheScope`La centralisation de cet emballage empêche un gestionnaire d'omettre silencieusement les champs de résultats modernes.

### Aucune demande initiée par le serveur

Un serveur moderne peut envoyer des notifications liées à une demande de client ou des notifications sur une client ouverte `subscriptions/listen`Il ne doit pas envoyer sa propre demande JSON-RPC.

Lorsqu'un gestionnaire a besoin d'échantillonnage, d'élicitation ou d'entrée de racine, il renvoie un `input_required`Le client remplit les demandes d'entrée intégrées et tente à nouveau la méthode d'origine avec un nouvel id de demande.

### Compatibilité explicite avec l'héritage

Un serveur à double époque peut également mettre en œuvre le `2025-11-25`Il choisit un comportement moderne quand il est nécessaire.`_meta`champs sont présents et comportement hérité lorsqu' il reçoit `initialize`- Je suis désolé .

Ne mettez pas un `2026-07-28`Ne marquez pas modernes `resultType`Le code de cette leçon est délibérément moderne, de sorte que ses invariants restent visibles.

```figure
t3-dispatch-loop
```

## Utilisez-le

Exécutez la démo et les tests finis du serveur Python:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

Exécutez le port TypeScript avec un coureur TypeScript:

```bash
npx tsx main.ts --demo
```

La démo envoie `server/discover`La liste de toutes les requêtes de base est complétée par un code de base, qui répertorie chaque élément primitif, invoque les outils et affiche une erreur de version non prise en charge.

## La faire partir

Cette leçon va à l' air .`outputs/skill-mcp-server-scaffolder.md`Il produit un plan de serveur moderne avec un contrat de découverte, une validation par demande, des listes de cache déterministiques et un adaptateur d'héritage isolé optionnel.

## Exercices

1. Supprimer les capacités d'une demande et prouver que le serveur n'utilise pas à nouveau la déclaration de la demande précédente.
2. Retourner le `TOOLS`- Je suis là .`PROMPTS`Confirmez que tous les résultats de la liste restent stables.
3. Ajoutez une destructive `notes_delete`l'outil et nécessitent une vérification de l'autorisation à l'intérieur de l'exécuteur.`destructiveHint`comme une seule indication de l'expérience.
4. Ajouter `resources/templates/list`avec `ttlMs`- Je suis là .`cacheScope`, et l'ordre déterministe.
5. Construire un adaptateur d' héritage séparé pour `2025-11-25`Ajoutez des tests qui prouvent qu'une demande moderne ne l'entrera jamais.

## Les termes clés

| Term | Meaning |
|------|---------|
| Stateless server | Handles each request from its own metadata without protocol-session memory |
| `server/discover` | Mandatory modern method that advertises versions and capabilities |
| Complete result | Successful modern result with `resultType: "complete"` |
| Cacheable result | Discovery, list, or resource-read result with `ttlMs` and `cacheScope` |
| Deterministic list | Same logical registry produces the same item order |
| Server identity | Recommended `io.modelcontextprotocol/serverInfo` in result `_meta` |
| Tool error | Valid tool call that returns content with `isError: true` |
| Protocol error | Invalid JSON-RPC or MCP request returned through `error` |

## Pour en savoir plus

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
