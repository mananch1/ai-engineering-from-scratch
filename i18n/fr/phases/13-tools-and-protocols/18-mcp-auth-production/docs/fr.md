# Autorités de la production du PCM: inscription et jetons liés à l'émetteur

> La leçon 16 construit la machine d'état OAuth 2.1. Cette leçon durcit ses limites de production pour MCP 2026-07-28: Documents de métadonnées d'identification de client d'abord, enregistrement dynamique dépassé uniquement pour la compatibilité, validation de l'émetteur d'autorisation-réponse, identifiants de client à clé émetteur, mise à jour JWKS et jetons pinés par le public sur chaque demande stateless.
> > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
> **Spec note (2026-07-28):**L'enregistrement dynamique du client est dépassé en faveur des documents de métadonnées d'identification du client.`application_type`Un client valide une RFC 9207 présente `iss`Les données de référence sont fournies par les autorités compétentes.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time:** ~90 minutes

## Objectifs d'apprentissage

- Découvrez un serveur d'autorisation à travers les métadonnées RFC 8414 et vérifiez le contrat.
- Enregistrer un document de métadonnées de l'identifiant de client et isoler le DCR dépassé comme une rétroaction.
- Valider la RFC 9207 `iss`, les enregistrements clés par émetteur de serveur d'autorisation et les jetons clés liés aux ressources par émetteur plus ressource.
- En cas de mise en cache et de mise à jour des clés JWKS dans un calendrier afin que la vérification de la signature survienne au déploiement de la clé.
- En cas de défaillance de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de
- Choisissez la validation JWT ou l'introspection de jeton, définissez la fraîcheur de révocation et échouez en toute sécurité lorsque les dépendances d'identité ne sont pas disponibles.
- Séparer le serveur d'autorisation, le serveur de ressources et le client afin que chacun ne fasse que ses propres contrôles.
- Contrôlez un serveur d'autorisation contre une liste de contrôle de déploiement et refusez une inscription ou une réutilisation de jetons non sûrs.

## Le problème

Le simulateur de leçon 16 exécute OAuth 2.1 en mémoire.

La première lacune est l'inscription et l'isolement des accréditations.**Client ID Metadata Document**Le client utilise une URL HTTPS avec un chemin qu'il contrôle comme son identifiant, et le serveur d'autorisation tire les métadonnées.`application_type`. Le client stocke les enregistrements dans le cadre de l' émetteur du serveur d' autorisation et les jetons d' accès dans le cadre de la `(issuer, resource)`Un émetteur modifié signifie une nouvelle inscription, et une ressource différente signifie un jeton lié séparément au public.

Le deuxième écart est la rotation de la clé. La validation JWT dépend des clés de signature du serveur d'autorisation, publiées sous forme de JSON Web Key Set (JWKS). Le serveur d'autorisation les fait tourner selon un calendrier (souvent à l'heure, parfois plus rapidement en cas de réponse à l'incident). Un serveur MCP qui récupère JWKS une fois au démarrage valide bien jusqu'à la fenêtre de rotation  puis chaque demande échoue jusqu'à ce que le redémarrage. Les câbles de production JWKS sont une valeur mise en cache avec un travail de mise à jour qui sur écrit le cache avant l'expiration des clés précédentes, plus un retrait de la cache pour le cas où un jeton signé par une clé plus récente que le cache arrive.

La troisième lacune est la liaison entre le public. La leçon 16 introduit des indicateurs de ressources RFC 8707.`token.aud`Il est la seule défense contre un serveur MCP en amont (ou un client malveillant détenant un jeton destiné à un serveur) en reproduisant ce jeton contre un autre serveur dans le même filet de confiance.

Cette leçon trace chaque lacune sur une surface en béton. Le document de métadonnées est un point d'extrémité HTTP. La mise à jour du cache JWKS est un travail programmé plus un cache de valeur clé. La validation JWT est une routine que le serveur de ressources exécute avant de déployer un outil. Gardez les trois rôles séparés et chacun ne fait appliquer que les contrôles qu'il possède: le serveur d'autorisation émet et rotate les clés, le serveur de ressources cache et valide, le client découvre et s'inscrit.

## Scope: Enforcement de la production après leçon 16

[Lesson 16: MCP Security with OAuth 2.1](../../16-mcp-security-oauth-2-1/docs/en.md)Cette leçon ne définit pas un deuxième flux OAuth. Elle commence après l'existence de ces contrats et demande comment un serveur de ressources déployé les met en œuvre pendant la rotation de clé, la validation de jetons opaques, la révocation, l'échec de dépendance, le déploiement et la réponse aux incidents.

La limite de production est plus étroite et plus opérationnelle:

- Un chemin JWT vérifie un émetteur fiché, un algorithme, une clé de signature, un public, des demandes de temps et des champs de chaque demande tout en rafraîchissant JWKS en toute sécurité.
- Un chemin de jeton opaque appelle le point final d'introspection authentifié de l'émetteur et valide l'état actif, le public ou la ressource, l'expiration, le sujet et les champs de validation retournés.
- La politique de révocation définit la rapidité avec laquelle une carte d'identité doit cesser de fonctionner et quel cache peut retarder ce fait.
- La politique de défaillance décide de ce qui se passe lorsque l'infrastructure de découverte, JWKS, introspection ou révocation n'est pas disponible.
- Les données probantes indiquent que les métadonnées de l'émetteur, l'ensemble de clés ou la réponse à l'introspection, les réclamations de jetons, la version de la politique et la raison du refus ont conduit le résultat sans stocker le jeton.

Cette distinction permet de maintenir les leçons comptables. La leçon 16 prouve le flux. La leçon 18 prouve qu'un jeton reste digne de confiance, ou est refusé, après qu'il ait atteint un véritable chemin de demande MCP.

## Le concept

### RFC 8414  Méta-données du serveur d'autorisation OAuth

Un document à `/.well-known/oauth-authorization-server`décrivant tout ce dont un client a besoin:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

Un client qui a reçu une chaîne d' URL de ressources MCP découverte: `oauth-protected-resource`à partir de la RFC 9728 (document du serveur de ressources) nomme l'émetteur, puis `oauth-authorization-server`Le client ne code jamais une URL d'autorisation.

Pour un identifiant de ressource avec un chemin, insérez le segment bien connu avant ce chemin.`https://mcp.example.com/team/server`résolve les métadonnées de ressources protégées à `https://mcp.example.com/.well-known/oauth-protected-resource/team/server`- Appendice`/.well-known/...`après que la voie de la ressource soit incorrecte.

Le contrat que vous vérifiez avant de faire confiance à un IDP pour MCP:

- `code_challenge_methods_supported`inclut `S256`La spécification est explicite: si ce champ est **absent**, le serveur d' autorisation ne prend pas en charge PKCE et le client **MUST**refuser de procéder.
- `grant_types_supported`inclut `authorization_code`et rejette `password`et `implicit`- Je suis désolé .
- Au moins un chemin d'inscription est disponible: `client_id_metadata_document_supported: true`(CIMD, préférentiel), un client préréglé, ou `registration_endpoint`(compatibilité dépréciée avec la RFC 7591).
- Si vous`authorization_response_iss_parameter_supported`est vrai, le client exige le RFC 9207 retourné `iss`et le compare exactement avec l'émetteur enregistré avant la redirection.
- `response_types_supported`C' est exactement ça.`["code"]`pour l'AOuth 2.1.

Si vous`S256`Si le serveur MCP refuse de déployer contre cette IdP  il n'y a pas de mode dégradé pour PKCE. Si *ni l'un ni l'autre* chemin d'inscription est annoncé et que vous n'avez pas de pré-enregistrement `client_id`, vous ne pouvez pas non plus vous inscrire; le manifeste de déploiement est erroné, pas le code.

### RFC 9728 (recapitulation)  Méta-données de ressources protégées

Le document est le seul endroit où un client cherche à trouver les serveurs d'autorisation de confiance par * ce * serveur MCP. Un seul serveur MCP peut accepter des jetons de plusieurs IdPs (un pour le personnel, un pour les partenaires).

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Documents de métadonnées d'identification du client (par défaut recommandé)

CIMD inverse l'enregistrement de *push* à *pull*. Au lieu de demander au serveur d'autorisation de cocher une`client_id`, le client utilise une URL HTTPS qu' il contrôle **as**- le`client_id`. L'URL se résolve à un document de métadonnées JSON; le serveur d'autorisation le récupère sur demande pendant le flux OAuth.`app.example.com`, il fait confiance au client servi de`https://app.example.com/client.json`Pas de retour et de retour.`client_id`espace de noms à l'échappement, aucun état par serveur à garder en synchronisation.

Le document de métadonnées hébergé par le client:

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

Le `client_id`la valeur dans le document **MUST**correspondant à l' URL à partir de laquelle il est servi (le serveur d'autorisation le vérifie; les désaccords sont rejetés).`client_id_metadata_document_supported: true`dans ses métadonnées RFC 8414.

Pour le contrat actuel du CIMD, `client_id`- Je suis là .`client_name`, et une non-vide `redirect_uris`L'identifiant client est une URL HTTPS absolue avec un chemin. `application_type`Les données de référence peuvent être incluses, mais ce n'est pas un champ obligatoire de la CIMD.`application_type`dans le chemin de CIMD préféré.

Deux faits de sécurité dont la spécification est claire:

- **SSRF.**Le serveur d'autorisation récupère une URL fournie par l'attaquant. Il doit se défendre contre la contrefaçon de requêtes du côté du serveur (pas de récupération vers les terminaux internes / administrateurs).
- **localhost impersonation.**La CIMD seule ne peut pas empêcher un attaquant local de réclamer l'URL de métadonnées d'un client légitime et de lier les données `localhost`Le serveur d'autorisation **MUST**afficher clairement le nom d' hôte de redirection URI pendant le consentement et **SHOULD**- Je vous préviens .`localhost`- seulement redirige.

Parce que CIMD n'a pas besoin d'état côté serveur, il n'y a pas de registraire pour se tenir debout comme DCR le demande.

Si l'opérateur du serveur d'autorisation a déjà fourni un identifiant client, utilisez cet enregistrement à l'échelle de l'émetteur avant d'essayer l'inscription automatique.

### RFC 7591: inscription à la compatibilité dépassée

DCR est dépassé dans la révision 2026-07-28. Gardez-le uniquement pour les serveurs d'autorisation qui ne peuvent pas consommer CIMD et où la pré-enregistrement est impracticable.

```json
POST /register
Content-Type: application/json

{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Le serveur répond par `client_id`et une `registration_access_token`pour les mises à jour ultérieures:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`application_type`Un client de bureau en boucle déclare`native`; un client hébergé sur un serveur déclare `web`et utilise des URIs de redirection HTTPS. `token_endpoint_auth_method: none`est la bonne option par défaut pour un client public natif.`client_id`uniquement, avec la PKCE fournissant la preuve de possession.

Trois pièges de production:

- Le point final d'enregistrement doit être limité par IP source.`client_id`Vérifiez les limites de tarifs avant que le registraire ne traite la demande.
- `software_statement`La moqueur de la leçon la saute; la production câble une étape de vérification qui rejette les enregistrements non signés de tout autre que localhost redirect URIs.
- Le `registration_access_token`Le vol de ce jeton signifie que l'attaquant peut réécrire les URIs de redirection du client.

### RFC 8707 (récapitulation)  Indicateurs de ressources

Leçon 16 établit la forme.`resource=<canonical-mcp-url>`, et le serveur MCP vérifie `token.aud`L'URI canonique est l'identifiant le plus spécifique pour le serveur: il utilise un schéma minuscule et un hôte, aucun fragment et, conventionnellement, aucune tache de trail.**not**La spécification la conserve lorsqu'elle est nécessaire pour identifier un serveur MCP individuel. `https://mcp.example.com`- Je suis là .`https://mcp.example.com/mcp`- Je suis là .`https://mcp.example.com:8443`, et `https://mcp.example.com/server/mcp`Choisissez un par serveur et pin`aud`(La moque de cette leçon utilise des publics à l'hôte nu comme`https://notes.example.com`pour une courte durée; un déploiement qui héberge plusieurs serveurs MCP sous une même origine les distingue par chemin.)

### RFC 7636 (recapitulation)  PKCE

Le PKCE est obligatoire dans l'AOuth 2.1.`code_challenge`et `code_verifier`. Le serveur rejette toute demande de jeton sans vérificateur ou avec un vérificateur qui ne correspond pas au défi stocké.

### Profil d'autorisation du MCP 2026-07-28

La révision actuelle du MCP maintient la limite entre le serveur et la ressource OAuth tout en rendant le transport du MCP stéréotique. Il n'y a pas de session de protocole pour mettre en cache une décision d'identité. La couche d'autorisation valide donc chaque demande de manière indépendante:

- Implémenter les métadonnées de ressources protégées de la RFC 9728 et fournir leur localisation soit par l'intermédiaire de l'application `WWW-Authenticate: Bearer resource_metadata="..."`en-tête sur un 401 **or**l' URI bien connu `/.well-known/oauth-protected-resource`(SEP-985 a rendu l'en-tête facultatif avec une rétroaction bien connue).`authorization_servers`champ **MUST**nommer au moins un serveur.
- Acceptez des jetons uniquement via `Authorization: Bearer ...`sur**every**requête  jamais dans une chaîne de requête, jamais validée uniquement au début de la session.
- Valider`aud`- Je suis là .`iss`- Je suis là .`exp`, et les champs de champs requis par demande.**MUST**valider que le jeton a été émis spécifiquement pour lui (audience); une absence ou une disparition `aud`est rejetée, jamais traitée comme une carte blanche.
- Au 401/403, retour `WWW-Authenticate: Bearer`transport `error=...`, le `resource_metadata="<PRM-URL>"`paramètre (l'URL du document de métadonnées, *non* la ressource nue), et `scope="..."`sur`insufficient_scope`(403). Note: le paramètre est `resource_metadata`, un pointeur de découverte  il n' y en a pas `resource`paramètre dans le défi.
- Le serveur d' autorisation accepte la découverte **either**RFC 8414 OAuth métadonnées **or**OpenID Connect Discovery 1.0; les clients doivent essayer les deux suffixes bien connus dans l'ordre de priorité.
- Le client (et non le serveur) se défend contre **mix-up attacks**: il enregistre les attentes `issuer`avant de rediriger et de valider le `iss`La valeur retournée dans la réponse d'autorisation réelle (RFC 9207) avant de racheter le code.`code_verifier`où qu'il soit dirigé.
- Une carte de crédit client appartient à un émetteur de serveur d'autorisation.`client_id`, jeton d'enregistrement ou jeton d'accès.
- Le CIMD est le mécanisme de recrutement préféré.`application_type`- Je suis désolé .

Le projet OAuth 2.1 est le substrat; la surface est la RFC 8414/7591/8707/9728/9207 + la RFC 7636 + la CIMD; la spécification MCP est le profil.

### Liste de contrôle des capacités de déploiement

Les tables de fonctionnalités fournisseurs deviennent obsolètes rapidement. Inspect les métadonnées retournées par le serveur d'autorisation que vous allez réellement déployer à la place.

| Check | Required decision |
|---|---|
| Discovered issuer | Exact HTTPS issuer expected by policy |
| PKCE | `S256` advertised; otherwise stop |
| Enrollment | CIMD preferred, pre-registration accepted, DCR only as deprecated compatibility |
| Authorization response | Validate RFC 9207 `iss` when present or advertised |
| Resource binding | Token request carries `resource`; resource server requires the matching `aud` |
| Credential storage | Key client IDs and registration credentials by issuer; key access tokens by issuer plus resource |
| DCR compatibility | Declare `native` or `web`; reject redirect URIs that do not fit the declared application type |

Ne déduisez pas le support d'un nom de produit ou d'un niveau de prix.

### Modèle de rafraîchissement JWKS (rotation à l'AS, rafraîchissement au serveur de ressources)

Gardez deux verbes séparés, car les confondre est un vrai bug de production:

- **Rotate**Le serveur de ressources n'a pas de part à cela et ne peut pas le faire  il ne détient pas les clés privées de l'IDP.
- **Refresh**est ce que fait le *serveur de ressources*: re-`GET`C'est la seule action JWKS qu'un serveur de ressources ait jamais effectuée.

Le mode d'échec de production est un cache obsolète. Résolvez-le avec un travail de mise à jour programmé plus un cache de valeur clé. Le serveur de ressources exécute un travail (cron, temporisateur, quel que soit votre temps d'exécution) qui, à un intervalle fixe, récupère `<issuer>/.well-known/jwks.json`et les sur-écrits `cache[issuer] = {keys, fetched_at}`Le validateur lit à partir de ce cache.`kid`est manquant dans les déclencheurs du cache **one**Cette opération traite deux cas à la fois: la mise à jour programmée et les fenêtres de chevauchement de clés où un jeton signé par une nouvelle clé arrive avant la prochaine mise à jour programmée.

Le renversement .**must be a re-fetch, never a rotate**Si vous faites passer le chemin de cache-miss à un rotate-and-mint, deux choses se brisent: (1) le codage d'une clé fraîche produit un `kid`que *toujours* ne correspond pas au jeton, donc la recherche échoue de toute façon; et (2) un attaquant qui pulvérisent des jetons avec le hasard `kid`Les valeurs forcent une série illimitée de créations clés  un DoS auto-infligé.`kid`Le coût est au plus d'une récupération gaspillée.

La forme du cache:

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

Deux touches à la fois est l'état stable.`k_2026_04`) avant de retirer le précédent (`k_2026_03`), de sorte que les jetons émis sous l'ancienne clé restent valables jusqu'à leur expiration.`kid`- Je suis désolé .

### La routine de validation

Le serveur MCP exécute la validation avant de déployer un outil.`code/main.py`utilisations:

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate`Décode le JWT, résout la clé de signature du cache JWKS (renouvellement une fois sur une défaillance), vérifie la signature, puis vérifie `iss`contre la liste d'autorisation, `aud`contre la ressource canonique de ce serveur, `exp`, et la portée requise  retourner un `WWW-Authenticate`La mise en œuvre d'une seule routine sur le serveur de ressources signifie que chaque point d'entrée (tout appel à l'outil, chaque transport) passe par les mêmes contrôles; il n'y a pas de chemin qui atteint un outil sans avoir d'abord validé.

### Les jetons opaques utilisent l'introspection, pas la spéculation

Tous les jetons d'accès ne sont pas des JWT. Si l'émetteur documente un jeton opaque, le serveur de ressources ne peut pas le décoder en revendications fiables. Il envoie le jeton au point final d'introspection RFC 7662 de l'émetteur via un backchannel authentifié et exige`active: true`, le contexte de l'émetteur attendu, le public ou la ressource exacte des PCM, les demandes de temps non expirés et les champs d'application requis par l'outil concret.

L'introspection en cache par émetteur, un digeste de jeton à sens unique et une ressource MCP. Ne jamais utiliser le jeton clair comme un journal ou une étiquette de cache. Réservez une entrée cache positive par la date d'expiration des jetons la plus tôt possible, les instructions de cache de l'émetteur et l'objectif de fraîcheur de révocation du déploiement. Gardez la mise en cache négative suffisamment courte pour que le jeton nouvellement émis ne reste pas faussement inactif. Un résultat pour une ressource ne peut pas autoriser une autre ressource même lorsque la chaîne de jetons opaque est identique.

Ne choisissez pas le mode de validation parmi les contenus de jetons contrôlés par l'attaquant. Pin JWT par rapport au comportement d'introspection à la métadonnées validées de l'émetteur et la configuration de déploiement. Sur le chemin de JWT, pin accepté algorithmes et fiables `jwks_uri`; ne suivez jamais une URL ou un algorithme clé sélectionnés uniquement par l'en-tête du jeton.

### La révocation est un contrat de fraîcheur

RFC 7009 permet à un client de demander à un serveur d'autorisation de révoquer un jeton. Cette demande ne supprime pas les copies déjà mises en cache par chaque serveur de ressources. Définir le délai de révocation maximum acceptable et faire en sorte que chaque cache le respecte.

Les déploiements de jetons opaques peuvent obtenir une révocation plus rigoureuse en observant chaque appel à haut risque ou en utilisant un cache positif court. Les déploiements JWT autonomes combinent généralement de courtes durées d'accès aux jetons avec la révocation des jetons de rafraîchissement, la retraite des clés pour les incidents à l'échelle de l'émetteur et un sujet, une session ou une liste déniforme des jetons pour le refus local d'urgence. Un JWT signé reste cryptographiquement valide jusqu'à expiration à moins que le serveur de ressources ne dispose d'une preuve externe de révocation actuelle.

Le démarrage, la désactivation du compte, le retrait du consentement et la réponse aux incidents sont des déclencheurs différents, mais doivent converger sur une déclaration mesurable: après la fenêtre de révocation déclarée, chaque réplique refuse la carte d'identité.

### L'échec de la dépendance nécessite une décision déclarée

Ne jamais improviser la politique de disponibilité à l'intérieur d'un gestionnaire d'exception.

| Failure | Safe production behavior |
|---|---|
| Scheduled JWKS refresh fails, known `kid` remains in a still-valid bounded cache | Continue only within the declared stale-on-error window and emit degraded health evidence |
| Token has an unknown `kid` and the one allowed refresh fails | Reject; never accept an unverifiable signature |
| Introspection is unavailable | Fail closed for protected calls; do not convert network failure into `active: true` |
| Protected-resource or issuer metadata changes unexpectedly | Stop new enrollment and token acquisition; keep only explicitly pinned, unexpired configuration under a bounded incident policy |
| Revocation endpoint is unavailable | Report logout or revocation as incomplete, retain the credential locally as unusable when possible, and do not claim global revocation succeeded |
| Clock source or claim type is invalid | Reject rather than widening skew until the token passes |

Une défaillance est une erreur opérationnelle avec la politique de santé et de réessayer. Une mauvaise signature, émetteur, audience, expiration ou champ d'application est un refus d'autorisation. Aucun ne parvient au gestionnaire de l'outil et aucun ne devrait fuir le contenu des jetons dans les preuves d'audit.

### Retour de lecture du public (restriction des privilèges des jetons d'accès)

Le serveur A (`notes.example.com`) et le serveur B (`tasks.example.com`L'attaquant prend le jeton de notes d'un utilisateur et le redémarre contre le serveur B.

Le validateur du serveur B:

1. Décodez JWT, apportez JWKS par `kid`- Je ne sais pas.
2. Vérifiez`iss`contre ses métadonnées de ressources protégées `authorization_servers`. (Passage  même IDP.)
3. Vérifiez`aud == "https://tasks.example.com"`(Fail à faire le token)`aud`est `https://notes.example.com`(dont le nom est
4. Retour 401 avec `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`- Je suis désolé .

La revendication du public est la seule défense contre cette attaque à la couche de protocole. Le sauter pour la performance est l'erreur de production la plus courante; le validateur doit fonctionner sur chaque demande, pas seulement au début de la session.**access-token privilege restriction**: un serveur MCP `MUST`rejeter tout symbole qui ne le mentionne pas dans l'auditoire.

> **Naming note.**La spécification réserve le terme "député confus" pour un problème lié mais distinct: un serveur MCP agissant en tant qu' autorité de contrôle **proxy**à une API tierce, en utilisant un ID client statique, qui transmet un jeton sans obtenir le consentement de l'utilisateur par client.**plus**ne pas passer le jeton entrant à l' API en amont (serveur MCP `MUST`Il obtient son propre jeton en amont séparé).

### Attaques mixtes (une défense du côté client que le serveur ne peut pas fournir)

Un client parle à de nombreux serveurs d'autorisation au cours de sa vie. Un AS malveillant peut essayer de faire racheter le code d'autorisation d'un AS honnête au point final du jeton de l'attaquant.

1. Avant de rediriger, le client enregistre les résultats attendus `issuer`à partir des métadonnées AS validées.
2. Sur la réponse d'autorisation, le client compare les retours `iss`Paramètre par rapport à l'émetteur enregistré (comparaison simple de chaînes, aucune normalisation) avant d'envoyer le code n'importe où.
3. Ne correspond pas (ou `iss`absent lorsque l' AS a fait la publicité `authorization_response_iss_parameter_supported`) → rejeter, et ne pas même afficher le `error`les champs.

PKCE ne peut pas arrêter de se confondre, car le client lui remet ses`code_verifier`C'est pourquoi la spécification enregistre l'émetteur par demande aux côtés du vérificateur PKCE et `state`- Je suis désolé .

### Mode d'échec

- **Stale JWKS.**Le validateur rejette les jetons valides après que l'AS ait fait tourner une clé. La correction est le schéma cron-refresh + cache-miss-refetch ci-dessus.
- **Rotate-as-fall-back.**Le câblage du chemin de cache-miss vers un rotate-and-mint au lieu d'un re-fetch est un vrai bug: il ne produit jamais le manque `kid`, et il devient contrôlé par l' attaquant .`kid`Les valeurs de base doivent être intégrées dans un système de gestion de la création de clés.`refresh-jwks`- Je suis désolé .
- **Missing `aud` claim.**Certains IP sont défavorisés à omettre `aud`à moins que `resource`Le validateur doit rejeter les jetons manquant `aud`On ne traite pas l'absence comme un jeu de cartes.
- **Mix-up via missing `iss` check.**Un client qui ne valide pas la RFC 9207 `iss`Le paramètre autorisation-réponse contre l'émetteur qu'il a enregistré avant de rediriger peut être dirigé vers le rachat d'un code d'AS honnête au point final de jeton d'un attaquant.
- **Scope upgrade race.**Deux flux de mise à niveau simultanés pour le même utilisateur peuvent à la fois réussir et produire deux jetons d'accès avec des champs différents. Le validateur doit utiliser le jeton présenté sur la demande, et non rechercher "la portée actuelle de l'utilisateur"  qui crée une fenêtre TOCTOU.
- **Registration token theft.**Une fuite .`registration_access_token`La mise à jour de l'URI permet à l'attaquant de réécrire les URIs redirigés.
- **`iss` not pinned.**Un validateur qui accepte tout.`iss`permet à un attaquant de créer son propre serveur d'autorisation, d'enregistrer un client pour le public cible et d'émettre des jetons.`authorization_servers`la liste est la liste d'autorisation; appliquer.
- **Credential or token cache collision.**Un client qui ne détecte les enregistrements que par ressource peut présenter l'identité d'un serveur d'autorisation à un autre. Un client qui détecte les jetons d'accès que par émetteur peut reproduire un jeton au mauvais public.`(issuer, resource)`, et se réinscrire chaque fois que l'émetteur change.

```figure
t3-jwks-rotate
```

## Utilisez-le

`code/main.py`passe le flux de production complet avec stdlib Python et trois rôles: `AuthorizationServer`- Je suis là .`ResourceServer`, et `Client`Le flux:

À partir de la racine du référentiel, exécuter:

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Le premier commandement imprime l'inscription et la validation des jetons liés à l'émetteur
Le deuxième rapport rapporte dix-huit contrôles passés.
l'écouteur de réseau ou écrit des informations d'identification.

1. Le serveur d' autorisation publie les métadonnées RFC 8414 à `/.well-known/oauth-authorization-server`- Je suis désolé .
2. Le client MCP appelle le point final des métadonnées et vérifie ses options d'inscription (`client_id_metadata_document_supported`pour le CIMD, `registration_endpoint`pour les RDC) et `S256`Le soutien de la PKCE.
3. Le client vérifie la pré-enregistrement de l'émetteur, sinon il s'inscrit avec son HTTPS Client ID Metadata Document.
4. Le client enregistre l'émetteur validé, crée un défi S256, reçoit un code d'autorisation unique plus `iss`, valide l' émetteur retourné et échange le code avec le vérificateur original et la RFC 8707 `resource`indicateur.
5. Le client MCP appelle un outil sur le serveur MCP avec `Authorization: Bearer ...`- Je suis désolé .
6. Le serveur MCP fonctionne `validate`, résolvant la clé de signature du cache JWKS.
7. L'IDP fait tourner une clé; la mise à jour programmée entraîne le JWKS dans le cache.
8. L'appel suivant est validé contre les touches actualisées sans redémarrage, et le jeton précédent est toujours validé pendant la fenêtre de chevauchement.
9. Une tentative de répétition du public contre une autre ressource MCP obtient 401 avec`audience mismatch`et une `resource_metadata`Le pointeur.

La JWT utilise ici HS256 avec un secret partagé (donc la leçon fonctionne uniquement sur stdlib). La production utilise RS256 ou EdDSA avec le modèle JWKS ci-dessus; la logique de validation est autrement identique.`refresh_jwks`lis directement la liste des clés du serveur d'autorisation; par fil, c'est un HTTP `GET`à `jwks_uri`- Je suis désolé .

## La faire partir

Cette leçon produit `outputs/skill-mcp-auth.md`. Compte tenu d'une configuration du serveur MCP et d'un ensemble de capacités IdP, la compétence émet la surface d'auth pour se tenir debout  les métadonnées de la ressource protégée, le chemin d'inscription à utiliser (CIMD, pré-inscription ou DCR fallback), le calendrier de mise à jour de JWKS, la cartographie de portée et les règles de refus d'appliquer lorsque l'IdP ne prend pas en charge le profil RFC complet.

## Exercices

1. On court .`code/main.py`- Suivez le flux. Notez comment l'IDP tourne une clé à l'étape 6, la planifiée `refresh_jwks`Retire le jeu publié et valident à la fois l'ancien jeton (fenêtre de chevauchement) et un jeton nouveau sans redémarrage.

2. Ajouter un nouveau IDP aux métadonnées de la ressource protégée `authorization_servers`Liste. Émettre un jeton signé par le nouveau IDP et confirmer que le validateur l'accepte. Émettre un jeton signé par un IdP non inscrit et confirmer que le validateur rejette avec `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`- Je suis désolé .

3. Ajouter une vérification de limite de taux à `register_client`Utilisez un jeton-bucket par IP source détenu dans un petit dicté en clés IP.

4. Lisez RFC 7591 et identifiez deux champs dans la leçon `/register`Le gestionnaire ne valide pas. Ajoutez la validation.`software_statement`et `redirect_uris`Régime d'IRU.)

5. Ajouter un deuxième serveur d'autorisation. Confirmer que le client stocke une inscription séparée à clé émetteur et refuse de réutiliser le jeton du premier émetteur ou `client_id`- Je suis désolé .

6. Prouvez la correction du DoS. Envoyez un jeton au validateur avec un jeton aléatoire.`kid`et confirmer `refresh_jwks`Le nombre de clés du serveur d'autorisation ne croît pas, puis le nombre de clés est redirigé vers un "rotate-and-mint" et regardez le nombre de clés augmenter par faux jeton.

7. Exercer des RDC dépassées avec les deux `native`et `web`Confirmer un client Web avec un redirection HTTP URI et un client natif sans redirection loopback exacte sont rejetés.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| ASM | "OAuth metadata document" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "Client metadata URL" | Client ID Metadata Document: an HTTPS URL used as the `client_id`; the AS pulls the JSON. Preferred enrollment in MCP 2026-07-28 |
| DCR | "Self-service client registration" | RFC 7591 `POST /register`; deprecated for current MCP and retained only for compatibility |
| JWKS | "Public keys for JWT validation" | JSON Web Key Set, fetched from `jwks_uri`, indexed by `kid` |
| Rotate vs refresh | "Updating the keys" | *Rotate* = AS mints/retires signing keys; *refresh* = resource server re-fetches the published set. Resource servers only ever refresh |
| Resource indicator | "Audience parameter" | RFC 8707 `resource` parameter pinning the token to one server |
| `aud` claim | "Audience" | JWT claim the validator compares against the canonical resource URL |
| Audience replay | "Token replay" | Token issued for Server A presented to Server B; defended by audience validation (spec: access-token privilege restriction) |
| Confused deputy | "Proxy token misuse" | An MCP proxy with a static client ID forwarding a token without per-client consent; distinct from audience replay |
| Mix-up attack | "Wrong token endpoint" | Client steered to redeem an honest AS's code at an attacker's endpoint; defended client-side via RFC 9207 `iss` |
| `iss` allow-list | "Trusted authorization servers" | The set named in protected-resource metadata's `authorization_servers` |
| `resource_metadata` | "Where to find the PRM doc" | `WWW-Authenticate` parameter naming the RFC 9728 metadata URL on a 401/403 |
| Public client | "Native or browser client" | OAuth client with no `client_secret`; PKCE compensates |
| `WWW-Authenticate` | "401/403 response header" | Carries `Bearer error=...` directives that drive client recovery |

## Pour en savoir plus

- [MCP authorization specification (2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)- le profil actuel d'autorisation des PMP
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)- CIMD, validation de l'émetteur, dépréciation des RCR et changements de clé de crédit de l'émetteur
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) contrat de découverte
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) RDC (route de retour)
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) preuve de possession du client public
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) Le public s'arrête
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) découverte de serveur de ressources
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) le `iss`Paramètre qui protège contre les attaques de confusion
- [RFC 7662: OAuth 2.0 Token Introspection](https://datatracker.ietf.org/doc/html/rfc7662)
- [RFC 7009: OAuth 2.0 Token Revocation](https://datatracker.ietf.org/doc/html/rfc7009)
