# Sécurité MCP: métadonnées empoisonnées, routage et état MRTR

> L'appartenance à un État ne signifie pas être sans confiance, mais chaque demande expose les preuves dont un serveur et une passerelle ont besoin pour valider l'appel de manière indépendante.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~60 minutes

## Objectifs d'apprentissage

- Traiter les descriptions des outils, les annotations, les informations du client et les informations du serveur comme des données non fiables.
- Détecter l'empoisonnement des métadonnées, les changements de descripteur et les collisions de noms entre serveurs.
- Valider les métadonnées de la demande 2026-07-28 et les en-têtes de routage HTTP diffusibles.
- Protéger le MRTR `requestState`contre la manipulation et lier la confirmation à des arguments exacts.
- Appliquer des limites d'autorisation et de taux à un principal, et non à une session de protocole supprimée.

## Le problème

Un modèle lit les descriptions des outils pour décider à quoi appeler. Un routeur lit les noms des outils pour décider où envoyer une demande. Un utilisateur lit les étiquettes pour décider ce qu'il approuve. Un descripteur malveillant peut cibler les trois.

Les instructions de sécurité officielles du MCP sont directes: les descriptions et les annotations doivent être traitées comme non fiables à moins qu'elles ne proviennent d'un serveur de confiance. Même alors, la confiance en déploiement peut changer. Une mise à jour du serveur, un paquet compromis, une erreur de registre ou une fusion de passerelle peuvent modifier ce que le modèle voit.

Le protocole actuel modifie également la limite de sécurité. En 2026-07-28, il n'y a pas de poignée de main de base et aucune séance de transport.`Mcp-Session-Id`n'est pas une conception actuelle.

## Le concept

### Sept surfaces d'attaque à vérifier

Utilisez une liste concrète au lieu de l'instruction vague pour être prudent.

1. **Metadata poisoning.**Une description contient des instructions qui ne sont pas liées au comportement de l'outil déclaré.
2. **Descriptor rug pull.**Une modification de nom, de description, de schéma ou d'annotation approuvée précédemment.
3. **Cross-server shadowing.**Deux arrière-plan exposent le même nom d'outil non qualifié et le routage choisit silencieusement l'un.
4. **Header and body confusion.** `Mcp-Method`ou `Mcp-Name`ne partage pas la demande JSON-RPC.
5. **Capability escalation.**Un concurrent revendique une extension ou une fonction client et le serveur erre cette déclaration d'autorisation.
6. **MRTR state tampering.**Un client change .`requestState`, répond à une question différente, ou réutilise la confirmation avec des arguments différents.
7. **Supply-chain identity confusion.**Un nom d'affichage familier est traité comme une preuve de l'identité de l'éditeur ou du serveur.

Ces surfaces se chevauchent. Le pinage de hachage aide à modifier le descripteur mais ne prouve pas que le premier descripteur était sûr. Le scan statique capte des phrases évidentes mais pas des instructions subtiles. L'espacement de noms empêche une classe de collision mais pas un serveur malveillant avec espace de noms.

### L'enveloppe de demande actuelle est une preuve, pas une identité

Chaque demande de 2026-07-28 contient:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

Valider la version et la forme de la capacité sur chaque demande. Utilisez les capacités pour choisir une forme de réponse compatible.`clientInfo`Il est autodéclaré.

Le même avertissement s' applique à `io.modelcontextprotocol/serverInfo`Il est utile pour les journaux et le débogage. Ce n'est pas un certificat, une preuve de registre ou une décision d'autorisation.

### Valider le routage avant la politique

Pour `tools/call`, HTTP par flux comprend:

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

La méthode d'en-tête doit être égale à la méthode du corps.`params.name`- Rejeter les désaccords avec`-32020`avant de sélectionner un backend, d'appliquer RBAC ou de consommer un jeton à limite de taux.

Cet ordre ferme une ambiguïté commune: un composant autorise le corps tandis qu'un autre le trace par l'en-tête.

La validation par fil suit une séquence exacte. Validez les types de JSON-RPC et de métadonnées, comparez les valeurs de l'en-tête avec le corps, puis vérifiez si la version correspondante est prise en charge. Une en-tête incompatible renvoie HTTP 400 avec `-32020`Si l'en-tête et le corps conviennent d'une version non prise en charge, renvoyez HTTP 400 avec `-32022`et `data`Exactement .`{"supported":["2026-07-28"],"requested":"<actual>"}`Une méthode inconnue renvoie HTTP 404 avec `-32601`- Je suis désolé .

Chaque objet d' erreur est facultatif `data`Lorsque le contrat a besoin d'informations structurées sur le recouvrement, une notification ne peut être effectuée.`id`Une notification HTTP acceptée renvoie 202 avec un corps vide.

### Enfoncer l'ensemble du descripteur

Un seul hash de description manque de changements de schéma et d'annotation.

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

Conservez le digeste sous une clé qualifiée comme `notes.export`, ainsi que des preuves et des délais d'approbation de l'éditeur en dehors de cet exemple de jouet.

À chaque rafraîchissement:

- - La clé inconnue: quarantaine jusqu'à révision.
- La même clé, différente digestion: quarantaine comme tirage au tapis jusqu'à réapprobation.
- Nom non qualifié dupliqué: nécessite un espacement de noms déterministe.
- Tapeur de scanner: bloquez et examinez le descripteur complet.

L'égalité de hash prouve la stabilité, pas la sécurité.

### La numérisation statique est un tricot

Des modèles simples peuvent marquer les balises de rôle, les délais d'instruction, la dissimulation, l'accès secret et les destinations de réseau obscurcées.

Une description sécurisée peut contenir une phrase marquée dans un avertissement légitime. Une description malveillante peut éviter chaque phrase. Traitez la sortie du scanner comme une preuve de révision, pas un score d'innocence automatique.

### Espace de noms avant fusion

Supposons que deux serveurs exposent les deux .`search`Ne laissez jamais l'ordre de découverte décider qui gagne.

```text
notes.search
issues.search
```

Le nom qualifié est le nom de la passerelle publique. Enregistrer le cartographie de l'arrière-plan séparément.`Mcp-Name`le routage fait référence au même objet.

### Les capacités sont des déclarations de compatibilité

À la demande `clientCapabilities`Il indique à un serveur quel protocole peut être traité par le client.

L'autorisation provient toujours de la politique de base et de ressources authentifiée.

1. Authentifier les informations de transport.
2. Valider la version, les en-têtes et la forme de la demande.
3. Vérifiez la compatibilité des capacités.
4. Autoriser le principal, l'outil, les ressources et les arguments.
5. Exécuter ou demander des entrées d'utilisateur.

### Protéger la confirmation MRTR sans état

Un outil conséquent peut nécessiter une confirmation de l'utilisateur. Le MCP actuel utilise des demandes de plusieurs voyages de retour au lieu d'un appel de retour du serveur au client.

Première réponse:

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Export notes to archive?",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

Le client obtient l'entrée et réessaye la méthode d'origine avec un nouveau identifiant JSON-RPC:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Chacun d' eux .`inputRequests`la valeur est une demande intégrée complète avec `method`et `params`. Sa clé doit correspondre à l' entrée correspondante dans `inputResponses`Une élicitation de formulaire utilise une racine d' objet`requestedSchema`, et le client doit avoir déclaré la capacité d'obtention du formulaire avant que le serveur ne le demande.

La capacité actuelle dispose de deux déclarations de formulaire valides. `{"elicitation":{}}`Il est également possible de faire des efforts pour obtenir des résultats.`{"elicitation":{"form":{}}}`Une déclaration uniquement sur URL comme `{"elicitation":{"url":{}}}`Le serveur retourne HTTP 400 avec `-32021`et `data.requiredCapabilities`égale à `{"elicitation":{"form":{}}}`- Je suis désolé .

Traiter`requestState`En cas de problème, le code de leçon utilise HMAC et l'argument exact pour rendre la limite visible.

Le registre nonce ne doit pas vivre à l'intérieur d'un seul objet de passerelle. Le modèle exécutable injecte un magasin de lecture limité et taillé TTL qui peut être partagé par plusieurs instances de passerelle. Sa revendication atomique est la limite d'exécution: seule une acceptation validée ou un déclin terminal explicite consomme l'état. Une réponse malformée ou `cancel`Une flotte de production a besoin de la même demande conditionnelle dans un stockage durable partagé.

Ne pas stocker le contexte de confirmation caché dans une session de protocole. Toute instance de serveur devrait être capable de valider la nouvelle tentative.

### Règle de deux pour les appels à haut risque

Classifier un appel sur trois axes:

- Il consomme des entrées non fiables.
- Il peut accéder à des données sensibles.
- Il provoque une action extérieure conséquente.

Une seule étape automatique ne doit pas combiner les trois. Divisez-la, réduisez les privilèges ou demandez une entrée explicite de l'utilisateur via MRTR. Il s'agit d'une heuristique de conception, pas d'une capacité de protocole.

### Réduire l'autorité avant l'exécution

L'appartenance à un État n'est pas une question de sécurité. Elle supprime l'historique caché du protocole, mais une demande autonome peut toujours demander à un gestionnaire débordé de pouvoir de fuir des données ou de faire un changement irréversible.

1. **Typed verb.**Exposer une opération limitée telle que `archive_note`Pas un générique .`run`ou `request`outil qui peut exprimer des pouvoirs non liés.
2. **Validated arguments.**Utilisez un schéma fermé où il est pratique de rejeter des champs inconnus, de normaliser les identifiants une fois, de fixer les tailles de plafond et de valider la destination, le locataire et la propriété des ressources avant l'évaluation des politiques.
3. **Current authorization.**Lier le principal authentifié au verbe exact, à la ressource, à l'environnement et aux arguments normalisés.
4. **Action-bound approval.**Pour une appel conséquente, attachez l'approbation à un digeste du verbe typé et des arguments normalisés, plus la politique principale, expiration et une fois.
5. **First-class refusal.**Le rejet du modèle, l'approbation expirée, le déclin des utilisateurs et la destination dangereuse comme résultats ordinaires qui n'exécutent aucun effet secondaire.
6. **Redacted audit evidence.**Enregistrer qui a demandé, qui a admis descripteur et version de politique ont été utilisés, quelle cible normalisée a été autorisée, pourquoi la décision a permis ou refusé, et si l'exécution a commencé.

Chaque étape restreint ce que le composant suivant peut faire. Le gestionnaire final doit recevoir une commande de domaine déjà validée, pas du texte brut du modèle plus de grandes informations d'identification. Répétez toute la chaîne lors d'une nouvelle tentative MRTR, d'une mise à jour de tâche ou d'un appel transmis par passerelle. Une approbation antérieure ne transforme pas les demandes ultérieures en trafic de session de confiance.

### Les voies d'interaction actuelles et les voies d'interaction héritées

Roots, Sampling et Logging sont dépassés pour les nouvelles implémentations 2026-07-28. Un gateway peut conserver le code de la demande du canal ancien uniquement comme chemin de compatibilité avec les versions.

Ne construisez pas une nouvelle défense autour d'un limitateur d'échantillonnage par session. Appliquez des quotas au principal, à l'émetteur, à la ressource, à l'outil et à la fenêtre de temps authentifiés. Pour les travaux interactifs actuels, inspectez les demandes et les réponses de saisie MRTR.

### Vérifie des transports sans État

- Accepter les messages MCP modernes au point final unique de la poste.
- Retour 405 pour GET et DELETE modernes.
- Ne pas faire de la mousse ou dépendre de `Mcp-Session-Id`- Je suis désolé .
- Ignorez les sessions précédentes et répétez les en-têtes comme entrées d'autorité.
- Retourner JSON ou SSE à la demande pour ce POST.
- Utilisation `subscriptions/listen`uniquement pour les notifications de changement de longue durée acceptées.

```figure
tp-tool-poisoning
```

## Faites-le

`code/main.py`Il canonifie et pinne les descripteurs complets des outils, rapporte l'empoisonnement et l'ombrage des métadonnées, valide l'enveloppe de demande moderne et les valeurs de routage, et effectue une exportation confirmée en deux rounds avec une signature`requestState`et un magasin de répétition partagée injectable.

Le modèle démarre après qu'un adaptateur HTTP ait analysé le corps JSON et les en-têtes de routage.`Content-Type`ou `Accept`. Connectez le même dispatcher à l'adaptateur HTTP Streamable complet de la leçon 09 , ce qui nécessite `Content-Type: application/json`et une `Accept`valeur contenant les deux `application/json`et `text/event-stream`- Je suis désolé .

- Je vais le faire.

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

L'échantillon mutera intentionnellement un descripteur.`input_required`réaction et une nouvelle tentative de non-état.

## Utilisez-le

Remplacez`SAFE_TOOLS`En plus de cela, vous pouvez utiliser des informations de référence pour les informations de référence, en utilisant une image standardisée de vos propres serveurs agréés.

Lors d'une passerelle, effectuez les mêmes vérifications pendant la découverte et à nouveau avant l'expédition. Un cache peut réduire le travail de découverte, mais une approbation en cache doit expirer ou être invalidée lorsque le descripteur change.

## La faire partir

Cette leçon va à l' air .`outputs/skill-mcp-threat-model.md`Il produit un modèle de menace de protocole courant sur les métadonnées, le routage, les capacités, les autorisations, le MRTR, le caching, le registre et les limites de compatibilité.

## Exercices

1. Lier le principal authentifié et la décision d'autorisation en cours à l'état MRTR scellé, puis rejeter une nouvelle tentative sous un principal différent.
2. Remplacez le magasin de lecture en mémoire par un insert conditionnel persistant et prouvez que deux processus ne peuvent pas tous deux revendiquer une nonce.
3. Injecter une défaillance après la réclamation de répétition mais avant une exportation simulée. Définir et tester la règle de transaction ou d'idempotence qui rend la récupération sûre.
4. Changez l'outil `inputSchema`Confirmez que le pin de tout le descripteur le capte.
5. Ajouter une politique qui refuse la mise en cache publique lorsque `tools/list`Les différences sont selon le principal.
6. Modélisez un serveur plus ancien derrière la passerelle.`2025-11-25`branche de la compatibilité.

## Les termes clés

| Term | Meaning |
|------|---------|
| Metadata poisoning | Instructions or deceptive claims embedded in a tool descriptor |
| Rug pull | Change to a previously approved descriptor |
| Tool shadowing | Ambiguous routing caused by duplicate unqualified names |
| Header mismatch | Routing header and JSON-RPC body disagreement, error `-32020` |
| Hash pin | Digest of the complete approved descriptor |
| MRTR | Stateless response and retry pattern for server-requested input |
| `requestState` | Opaque round-trip value that must be treated as untrusted input |
| Capability declaration | Statement of protocol compatibility, not authorization |
| Implicit form support | An empty `elicitation` capability object, equivalent to form support |
| Qualified tool name | Stable gateway name such as `notes.search` |

## Pour en savoir plus

- [MCP security and trust guidance](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Deprecated features](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
