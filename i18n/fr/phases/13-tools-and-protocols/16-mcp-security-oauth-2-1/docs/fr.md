# Autorisation du MCP: CIMD, obligation de l'émetteur, PKCE et step-up

> Une demande de MCP à distance est sans statut, mais son autorisation n'est pas anonyme.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security)
**Time:** ~90 minutes

## Objectifs d'apprentissage

- Découvrez les serveurs d'autorisation à travers les métadonnées de ressources protégées.
- Préférer les documents de métadonnées d'identification de client à l'enregistrement dynamique de client dépassé.
- Déclare la bonne`application_type`lorsque la voie de compatibilité DCR est inévitable.
- Valider la réponse à l' autorisation `iss`et isoler les informations d'identification par émetteur.
- Utilisez PKCE, indicateurs de ressources, validation du public et champs de mesure supplémentaires.
- Envoyer des demandes autorisées au MCP 2026-07-28 sans séances de protocole.

## Le problème

Un serveur MCP distant peut lire des enregistrements privés, écrire des systèmes externes ou déclencher des travaux coûteux.

- Quel serveur d'autorisation a délivré la carte d'identité ?
- Pour quelle ressource MCP est le jeton ?
- Quel client et quel URI redirigé a terminé le flux ?
- Quelles opérations l'utilisateur a-t-il approuvées ?
- Cette demande correspond- elle toujours à cette approbation?

Le profil d'autorisation 2026-07-28 durcit l'inscription des clients et la gestion des émetteurs.`application_type`sur le RCR, valide les réponses des émetteurs de la RFC 9207 et interdit la réutilisation des certificats de crédit entre les émetteurs.

Ces règles complètent le noyau de l'apathie.`Mcp-Session-Id`- Je suis désolé .

## Le concept

### Connaître les trois rôles

- **MCP client:**envoie des demandes au nom d'un propriétaire de ressource.
- **MCP resource server:**accepte le jeton d'accès et sert le point final du MCP.
- **Authorization server:**authentifie le propriétaire de la ressource, collecte le consentement et émet des jetons.

Le serveur de ressources et le serveur d'autorisation peuvent être exploités ensemble, mais garder leurs identifiants et leurs responsabilités de validation séparés.

### L'autorisation s'applique à HTTP

La spécification d'autorisation MCP s'applique aux transports basés sur HTTP. Un serveur local de studio fonctionne sous la limite de confiance du processus et du système d'exploitation.

Pour le HTTP en streaming à distance, envoyez le jeton porteur dans le `Authorization`Ne jamais le mettre dans l'URL.

### Commencez par les métadonnées de ressources protégées

Le serveur de ressources publie les métadonnées RFC 9728:

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

Le client démarre à partir de l'URL de la ressource MCP, récupère ce document, sélectionne un serveur d'autorisation annoncé, puis récupère les métadonnées OAuth ou OpenID Connect de ce serveur.

Préserver le chemin de la ressource lors de la construction de l'URL bien connue RFC 9728. Pour la ressource `https://notes.example.com/mcp`, cette leçon utilise `https://notes.example.com/.well-known/oauth-protected-resource/mcp`- Je dépose le`/mcp`le suffixe peut sélectionner des métadonnées pour une ressource protégée différente de la même origine.

Ne devinez pas le serveur d'autorisation à partir d'un nom d'hôte. Ne suivez pas un émetteur découvert dans un corps d'erreur non validé. Gardez une politique pour laquelle l'émetteur est prêt à faire confiance au client.

### Vérifiez les métadonnées du serveur d'autorisation

Les métadonnées doivent exposer les points d'extrémité et les contrôles pris en charge:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

Requérir S256 pour PKCE. Enregistrer la chaîne d'émetteur exacte. Cette valeur exacte devient la clé pour l'enregistrement et le stockage des jetons.

### Suivre la priorité de l'enregistrement

Utilisez les informations du client prérégistrées lorsque le client a déjà une relation explicite avec l'émetteur sélectionné. Sinon, préférez les documents de métadonnées de l'ID du client lorsque le serveur d'autorisation annonce le support. Utilisez DCR uniquement comme la rétroaction de compatibilité dépassée, puis demandez des informations du client si aucun de ces mécanismes n'est disponible.

### Préférer les documents de métadonnées d' identifiant de client

Un document de métadonnées d'identification de client donne au serveur d'autorisation une URL HTTPS qui est à la fois l'identifiant du client et l'emplacement de ses métadonnées:

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

Le serveur d'autorisation récupère et valide le document.`client_id`doit être une URL HTTPS avec un chemin, et la valeur à l'intérieur du document doit être égale à cette URL exactement.`client_id`- Je suis là .`client_name`, et `redirect_uris`- Je suis là .`application_type`Il est également nécessaire de prendre en compte les exigences de la CIMD, notamment en ce qui concerne les méthodes de détection des risques.

Traiter la récupération du document comme une opération sensible aux SSRF. Résoudre et valider la destination, rejeter les adresses de retour en boucle, privées, locales de lien et autrement interdites, vérifier à nouveau après les redirections et les changements DNS, limiter les redirections, octets et temps, nécessiter JSON, et seulement selon des contrôles de cache HTTP validés. Traiter `client_name`et d'autres champs d'affichage en tant que texte non fiable.

Le CIMD élimine la nécessité de fabriquer un nouvel identifiant dynamique pour chaque premier contact.

### DCR est un chemin de compatibilité

L'enregistrement dynamique du client reste disponible pour les anciens serveurs d'autorisation, mais il est dépassé pour les nouvelles implémentations de MCP.

Lorsque vous utilisez DCR, déclarez `application_type`- Le numéro de la liste:

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- Utilisation des clients de bureau, mobile, ligne de commande et loopback `native`- Je suis désolé .
- Utilisation des applications de navigateur hébergées à distance `web`et des redirections HTTPS à distance.

L' omission du champ peut être par défaut `web`dans une mise en œuvre d'enregistrement OpenID Connect et faire une redirection en boucle légitime échouer.

Gardez le code DCR derrière une décision explicite de rétractation. Ne retirez pas silencieusement après une défaillance arbitraire de validation CIMD. Cela pourrait transformer une défaillance de sécurité en un chemin d'inscription plus faible.

### Les informations d'identification obligatoires à l'émetteur

Stocker le matériel d'inscription mis en mémoire par l'émetteur sous l'émetteur exact:

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

Si la découverte de ressources protégées change à partir de `https://auth-one.example`à `https://auth-two.example`, réévaluer la confiance. Ne jamais envoyer le secret du client du premier émetteur, l'identifiant du client DCR, le jeton d'accès à l'enregistrement, le jeton de rafraîchissement ou le jeton d'accès au second. Les clients prérégistrés et DCR doivent utiliser les informations d'identification émises pour le nouvel émetteur.

Un identifiant client CIMD est différent car il s'agit d'une URL HTTPS auto-hébergée, et non d'une pièce d'identité élaborée par un serveur d'autorisation. La même URL CIMD est portable: un nouvel émetteur de confiance récupère et valide le document sans réenregistrement DCR. Les réponses et les jetons d'autorisation sont toujours validés et stockés sous le nouvel émetteur.

### Code d'autorisation avec PKCE

Le flux interactif est:

1. Générer une entropie élevée `code_verifier`- Je suis désolé .
2. Dériver le S256 `code_challenge`- Je suis désolé .
3. Envoyez la demande d' autorisation avec exactitude `client_id`- Je suis là .`redirect_uri`- Je suis là .`scope`- Je suis là .`code_challenge`, et `resource`- Je suis désolé .
4. Recevoir une réponse d' autorisation contenant `code`et, lorsqu'il est prévu, `iss`- Je suis désolé .
5. Valider`iss`contre l'émetteur enregistré exact avant d'utiliser un champ de réponse.
6. Échangez le code avec `code_verifier`, le même redirection URI, et le même `resource`- Je suis désolé .
7. Conservez le jeton résultant en dessous `(issuer, resource)`- Je suis désolé .

Le `resource`Le paramètre de RFC 8707 apparaît dans les demandes d'autorisation et de jetons.

### Valider`iss`exactement

La RFC 9207 empêche une réponse d'autorisation d'un émetteur d'être confondue avec une réponse d'un autre.

Quand ?`iss`Si le code est présent, comparez-le à l'émetteur enregistré sans repli, sans changement de trailing-slash, sans suppression de port par défaut ou normalization de codage en pourcentage.

Un serveur d' autorisation qui inclut `iss`annonces `authorization_response_iss_parameter_supported: true`Les clients actuels valident toujours un cadeau .`iss`Même quand cette annonce manque.

### Valider le public sur le serveur MCP

Le serveur de ressources n'accepte que les jetons émis pour lui-même:

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

Les jetons invalides, expirés, émis à tort ou de mauvais public reçoivent 401. Le serveur MCP ne doit pas accepter ou transférer un jeton destiné à un autre service.

### Requérir la portée de courant la plus petite

Si un outil ultérieur demande plus, le serveur renvoie 403 avec un défi de portée autorisé:

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

Le client explique la nouvelle autorisation, obtient le consentement, effectue un nouveau flux d'autorisation avec le jeu de champ combiné et tente à nouveau la demande MCP avec un nouveau identifiant JSON-RPC.

Ne supposez pas que la portée contestée soit un sous-ensemble de `scopes_supported`Le défi est autoritaire pour l'opération actuelle.

### Autorisation et câble MCP sans État

Un appel d' outil autorisé contient toujours l' enveloppe de demande actuelle complète:

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Le jeton autorise le principal, les métadonnées de la demande négocient le comportement du protocole, aucune ne remplace l'autre.

Valider le fil dans un ordre fixe: JSON-RPC et types de métadonnées, égalisation d'en-tête et de corps, puis support du protocole.`-32020`Si l'en-tête et le corps conviennent d'une version non prise en charge, renvoyez HTTP 400 avec `-32022`et `data`Exactement .`{"supported":["2026-07-28"],"requested":"<actual>"}`Une méthode inconnue renvoie HTTP 404 avec `-32601`- Je suis désolé .

Chaque erreur de demande, y compris 401 jetons invalides et 403 champs insuffisants, est une enveloppe d'erreur JSON-RPC avec la demande initiale `id`. Les informations de récupération structurée sont dans une erreur facultative `data`- le président .`WWW-Authenticate`Une notification n'a pas de code de réponse HTTP.`id`Une notification HTTP acceptée renvoie 202 avec un corps vide.

Le serveur implémentera `server/discover`Il est également obligatoire de mettre en œuvre les outils de publicité.`tools/list`Les descripteurs d'outils ont des noms, des descriptions et des racines d'objets stables.`inputSchema`La liste est déterministe et rend.`resultType`, les métadonnées d'identité du serveur, une limite `ttlMs`, et `cacheScope`- la recherche et une liste d'outils indépendants de l'utilisateur peuvent être disponibles avant l'autorisation.

### Pas de passe symbolique

Un serveur MCP ne doit pas transférer le jeton d'accès MCP du client vers une API en aval. Obtenir un jeton en aval séparé avec le bon public ou utiliser une conception explicite de jeton-échange. La validation du public ne fonctionne que lorsque les services refusent des jetons échangés pour quelqu'un d'autre.

### Les jetons de rafraîchissement

Les jetons de rafraîchissement sont facultatifs. Lorsqu'ils sont émis, conservez-les confidentiellement et clés par émetteur et ressource. Ne supposez pas qu'ils existent.

```figure
t3-scope-stepup
```

## Faites-le

`code/main.py`est un protocole en cours de processus et un simulateur d'autorisation. Il implémente la découverte de ressources protégées, les métadonnées du serveur d'autorisation, l'inscription à la CIMD, le retrait de DCR avec version, les vérifications de type d'application, PKCE, la validation de l'émetteur, les jetons liés aux ressources, l'augmentation de la portée,`server/discover`- Je suis là .`tools/list`, et une demande d'outil sans État.

Le modèle reçoit des corps de requêtes analysés et des en-têtes de routage.`Content-Type`ou `Accept`. Connectez-le à l'adaptateur HTTP diffusable de la leçon 09 qui nécessite `Content-Type: application/json`et une `Accept`valeur contenant les deux `application/json`et `text/event-stream`- Je suis désolé .

- Je vais le faire.

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

La sortie montre la découverte en premier, l'inscription au CIMD, une lecture ordinaire, deux étapes séparées de la portée et le stockage des identifiants par l'émetteur.

## Utilisez-le

Mettre en place des objets du simulateur sur les composants de production:

- `ResourceServer.protected_resource_metadata`devient le point final de la RFC 9728.
- `AuthorizationServer.metadata`devient la découverte RFC 8414 ou OpenID Connect.
- `Client.enroll`devient une résolution CIMD plus une branche explicite de compatibilité DCR.
- Les informations de confiance des clients émises par l' émetteur et `tokens_by_issuer_resource`Une URL CIMD peut rester portable tant que ses résultats d'autorisation restent liés à l'émetteur.
- `ResourceServer.handle`devient un middleware qui valide les en-têtes actuels de MCP, les jetons et la portée des outils avant l'envoi tout en conservant chaque erreur de demande dans une enveloppe JSON-RPC correspondante.

## La faire partir

Cette leçon va à l' air .`outputs/skill-oauth-scope-planner.md`Il conçoit désormais la priorité de l'inscription, le stockage des identifiants liés à l'émetteur, le type d'application, le PKCE, les indicateurs de ressources, les défis de portée et la limite actuelle des demandes de statut.

## Exercices

1. Ajouter la rotation du jeton de rafraîchissement et rejeter la réutilisation du jeton de rafraîchissement précédent.
2. Ajouter une liste d'autorisation d'émetteur. Lors du changement d'émetteur, réutilisez uniquement une URL CIMD portable; refuser toutes les identifiants et jetons précédemment émis par l'émetteur.
3. Ajouter une expiration aux codes d'autorisation et confirmer un échange tardif échoue.
4. Construisez une variante de client Web avec un redirection HTTPS à distance et comparez ses métadonnées DCR avec le client natif.
5. Ajouter une deuxième ressource sous le même émetteur. Confirmer que son jeton d'accès ne peut pas être utilisé à la première ressource.

## Les termes clés

| Term | Meaning |
|------|---------|
| Protected-resource metadata | RFC 9728 document that identifies the resource and authorization servers |
| CIMD | HTTPS metadata document whose URL is the OAuth client identifier |
| DCR | Deprecated dynamic client enrollment retained for compatibility |
| `application_type` | `native` or `web`, used to validate redirect URI rules |
| PKCE | Verifier and S256 challenge that protect an intercepted authorization code |
| `iss` | RFC 9207 authorization response issuer identifier |
| Resource indicator | RFC 8707 parameter that binds a token request to an MCP resource |
| Audience | Resource for which a token is valid |
| Step-up | New consent and token issuance for an additional current-operation scope |
| Issuer-bound credentials | Registration and token records isolated by exact authorization server issuer |

## Pour en savoir plus

- [MCP 2026-07-28 authorization specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707: Resource Indicators for OAuth 2.0](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207: OAuth 2.0 Authorization Server Issuer Identification](https://www.rfc-editor.org/rfc/rfc9207)
- [OAuth Client ID Metadata Document draft](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)
