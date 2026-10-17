# Ingénierie de conformité MCP: versionnement, preuve et opérations

> Un serveur n'est pas conforme parce que son cheminement heureux a fonctionné à travers un SDK. La conformité vit au fil, aux limites de version, à travers les intermédiaires et pendant le retour.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 17 (gateways), Phase 13 · 30 (registry admission)
**Time:** ~100 minutes

## Objectifs d'apprentissage

- Transformer les règles de MCP en transcriptions de fil dorées et négatives.
- Restez strict .`2026-07-28`comportement séparé de l'héritage limité de la chute.
- Distinguer les champs additifs inconnus d' un champ non valide inconnu `resultType`- Je suis désolé .
- Comparer les preuves JSON-RPC brutes avec une vue normalisée par SDK.
- Prouvez l'intégrité de l'en-tête et du corps à travers une véritable limite de proxy.
- Les sorties de la porte avec la transcription édité, la santé et les preuves de retour.

## Le problème

Votre client appelle .`tools/list`Il passe le test d'intégration.

Ce résultat laisse sans réponse des questions importantes:

- La requête contenait-elle des métadonnées modernes par protocole de demande ?
- Je l' ai fait .`MCP-Protocol-Version`- Je suis là .`Mcp-Method`, et `Mcp-Name`correspondent au corps JSON-RPC ?
- La réponse contenait-elle une valeur valide ?`resultType`sur le fil, ou le SDK en a synthétisé un ?
- Le client conserverait-il un futur champ additif?
- Une erreur reconnue de nos jours pourrait- elle déclencher accidentellement une poignée de main?
- Un proxy a-t-il préservé l'état d'origine et l'erreur JSON-RPC ?
- Le sérialisateur de notification a-t-il émis une réponse interdite ?
- Les opérations peuvent-elles prouver pourquoi une libération a été promue ou retenu sans garder des secrets ?

Conformité est un ensemble d'invariants observables. Construisez un harnais qui capture ces invariants avant que le trafic de production ne les découvre.

```figure
mcp-conformance-operations
```

## Commencez par les épisodes de la version

MCP `2026-07-28`Les données de base sont fournies par les utilisateurs.`params._meta.io.modelcontextprotocol/protocolVersion`et `params._meta.io.modelcontextprotocol/clientCapabilities`Les clés avec un espace nommé comptent .`protocolVersion`ou `clientCapabilities`Les aliases sont malformés. Lorsque les en-têtes de routage réfléchis sont présents à la limite HTTP, leurs valeurs doivent être conformes au corps JSON-RPC.`resultType`- Je suis désolé .

Les versions à travers `2025-11-25`Utilisez l'ère d'initialisation antérieure.`resultType`est interprété comme complet seulement après que le client a sélectionné cette période antérieure.

Ne créez pas un validateur permissif qui accepte les deux formes à la fois.

| Branch | Entry evidence | Missing `resultType` | Initialization |
|---|---|---|---|
| Modern | Successful `server/discover` or recognized modern response | Invalid | Not the default path |
| Legacy | Configured allowlist plus a valid legacy `initialize` result after an inconclusive modern probe | Interpreted as complete | Required by that era |

La séparation empêche un paire moderne malformé d'être récompensé par une validation plus faible.

### Mode strict

Le mode strict exige une preuve de comportement moderne.`server/discover`Une erreur JSON-RPC moderne reconnue prouve aussi. Corrigez la demande ou arrêtez. Ne jamais dégrader parce que le serveur est revenu`-32020`- Je suis là .`-32021`ou `-32022`- Je suis désolé .

### Mode de rétroaction

Le mode Fallback effectue une sonde moderne limitée. Un délai, une réponse vide, une connexion fermée ou une réponse non reconnue sont inconclusifs. Cela ne prouve pas que le pair est hérité. Seul un point final explicitement configuré ou alloué pour la compatibilité peut ensuite recevoir une sonde limitée héritée, et le client sélectionne la branche héritée seulement après avoir validé la validation de cette sonde.`initialize`résultat et révision négociée de l'héritage.

Une erreur moderne reconnue contient des informations de correction utiles. Le dégradement après cela peut cacher un désaccord d'en-tête, une déclaration de capacité manquante ou une version non prise en charge.

Cela empêche un attaquant, une panne ou un filtrage de proxy de forcer la rétrogradation en abandonnant la réponse moderne.

En outre, il est possible de noter l'époque sélectionnée à côté de chaque transcription.

## Construisez un corpus de transcriptions

Un fichier de transcription enregistre ce qui a franchi la frontière, pas seulement l'appel SDK:

```json
{
  "name": "golden-modern-list",
  "era": "modern",
  "headers": {
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/list"
  },
  "request": {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
      }
    }
  },
  "responseStatus": 200,
  "responseBody": {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "resultType": "complete",
      "tools": []
    }
  }
}
```

Gardez deux classes de fixations.

### Des transcriptions en or

Les transcriptions dorées prouvent un comportement acceptable:

- une demande de découverte ou de méthode moderne avec des métadonnées et des en-têtes correspondants
- résultat complet avec les champs requis
- `input_required`résultat lorsque la méthode peut demander plus d'entrée
- résultat d'extension seulement après que la capacité correspondante ait été annoncée
- résultat hérité sans `resultType`, mais seulement dans l' époque de l' héritage sélectionné
- traitement des notifications sans réponse JSON-RPC

Une transcription en or est précise, pas grande.

### Transcriptions négatives

Les transcriptions négatives prouvent le comportement de refus:

- décalage entre la tête et le corps
- manquantes capacités par demande
- version du protocole non prise en charge
- manquant de modernes `resultType`
- inconnu ou non annoncé `resultType`
- réponse `jsonrpc`autre que `2.0`ou une ID qui diffère en valeur ou en type JSON
- une réponse contenant les deux `result`et `error`, ou aucune
- une erreur sans un nombre entier `code`et la corde`message`
- une erreur de protocole connue mappée à l'état HTTP incorrect
- réponse délivrée pour une notification
- enveloppe JSON-RPC malformée
- effondrement par proxy d'une erreur de protocole

Pour chaque cas négatif, affichez la limite de rejet et le code d'erreur stable. L'appel a échoué est trop faible.`-32020`peuvent tous deux ressembler à un échec tout en racontant des histoires complètement différentes aux opérateurs.

Le fichier de désaccord d'en-tête doit inclure la réponse HTTP 400 JSON-RPC réelle du serveur avec l'identifiant de la requête correspondante et le code d'erreur `-32020`- Appliquer automatiquement chaque fois que le validateur local observe`HeaderMismatch`; ne faites pas de la vérification de réponse un signal de fixation facultatif. Un cas avec HTTP 500 et aucun corps échoue même lorsque le code de rejet local était correct. Un harnais qui s'arrête après que son propre validateur de demande lance a testé seulement lui-même, pas le comportement du fil du serveur.

Le projet officiel de conformité MCP est utile comme suite externe et référence versionnée. Gardez également vos transcriptions locales. Ils capturent votre proxy, SDK, authentification, extensions et chemin de sortie, que une suite générale ne peut pas connaître.

## Les valeurs de titre doivent correspondre au corps du RPC

Dans le HTTP Streamable moderne, les intermédiaires peuvent router ou faire appliquer la politique en utilisant des en-têtes miroirés. Le corps JSON-RPC reste la source de vérité du protocole.

Valider dans cet ordre:

1. Analyse et validation des types de enveloppe et de métadonnées JSON-RPC.
2. Comparer`MCP-Protocol-Version`avec `params._meta.io.modelcontextprotocol/protocolVersion`- Je suis désolé .
3. Comparer`Mcp-Method`avec `method`- Je suis désolé .
4. Lorsque la méthode a un nom de routage, comparez `Mcp-Name`avec la valeur corporelle correspondante.
5. Une fois l'égalité établie, décidez si la version et l'ensemble de capacités correspondants sont pris en charge.

Cet ordre distingue l' incompatibilité .`-32020`de la version non prise en charge `-32022`Il empêche également une passerelle d'autoriser le nom de l'en-tête pendant que l'origine exécute un nom de corps différent.

Les noms de champs HTTP sont insensibles aux cas, tandis que leurs valeurs restent sensibles aux cas. Normalisez les noms d'en-tête avant de rechercher et rejettez les duplicates contradictoires. Pour un espace blanc non sûr, non ASCII ou à la tête ou à la traîne `Mcp-Name`, décodez l' exact`=?base64?{Base64EncodedValue}?=`UTF-8 sentinel avant de le comparer avec le corps. rejetez un sentinel incomplet, non valide Base64, non valide UTF-8, ou une valeur crue non sûre avec `-32020`. L'espace blanc environnant est invalide même lorsque le corps contient les mêmes caractères, car cette valeur nécessite une codage sentinelle avant le transport.

Un intermédiaire peut rejeter un HTTP malformé avant qu'une demande n'atteigne le serveur MCP, de sorte que son échec peut être une erreur HTTP sans JSON-RPC. Capturez si un rejet est venu de l'intermédiaire ou de l'origine. Le serveur MCP d'origine doit utiliser le contrat d'erreur de protocole lorsqu'il traite une demande JSON-RPC valide.

## Les champs inconnus ne sont pas des résultats inconnus

La compatibilité avec la méthode de l'avenir exige deux règles différentes.

### champs additifs inconnus

Objets de résultat et `_meta`Les cartes peuvent acquérir des champs. Un validateur doit conserver ou ignorer un champ additif en fonction de son rôle, sauf si ce champ viole un contrat réservé. L'échantillon conserve le résultat brut complet dans la preuve et accepte `futureHint`à côté d'un résultat connu.

Si vous êtes un proxy transparent, la conservation d'un champ inconnu est généralement plus sûre que le dépouillement. Si vous êtes un client d'application, l'ignorance peut être valide. Votre test différentiel devrait toujours révéler que le SDK l'a omis, donc le comportement est délibéré.

### On ne sait pas .`resultType`

`resultType`Les résultats modernes utilisent des méthodes de recherche de base.`complete`ou `input_required`. Une extension ne peut ajouter une autre valeur que lorsque sa capacité a été annoncée.`task`dans le cadre de la capacité négociée.

Un discriminateur inconnu ou non annoncé ne peut pas être traité en toute sécurité comme complet. Le client ne sait pas le cycle de vie qu'il rejetterait.

La même réponse brute peut donc contenir un champ inconnu acceptable et un type de résultat inconnu inacceptable.

Le discriminateur n'est que la première couche.`tools/list`Le résultat a besoin d' une`tools`array dont les descripteurs ont des noms uniques non vides, des descriptions utiles et la racine d'objet `inputSchema`Les valeurs`task`Le résultat est valable uniquement pour un éligible `tools/call`avec la capacité des tâches et exige `taskId`, état connu, création et mise à jour des timestamps, et `ttlMs`, plus un intervalle de vote facultatif valide.`completion/complete`Le résultat exige une `completion`objet avec une valeur de chaîne ne dépassant pas 100 valeurs, un nombre entier non négatif facultatif `total`qui n'est pas inférieur aux valeurs retournées, et un boolean facultatif `hasMore`Une bonne orthographe .`resultType`ne peut pas faire un conformant de charge utile malformé.

## La variante de notification

Une notification JSON-RPC n' a pas été publiée `id`. Le récepteur ne doit pas envoyer une réponse de succès ou d'erreur JSON-RPC.

Pour une forme de notification HTTP acceptée, le harnais s'attend à un HTTP `202`avec un corps vide.`2026-07-28`n'explique pas les notifications de base client-serveur sur HTTP en streaming. L'échantillon utilise une notification d'extension de cours espacée uniquement pour tester l'invariable de sérialisation unidirectionnelle. Ne le présente pas comme une nouvelle méthode de base.

Testez le sérialisateur, pas seulement le manipulateur.`None`Pendant que le middleware l'enveloppe dans un objet JSON de succès.

## Ajouter un SDK différentiel

Les SDK transforment souvent les objets en langage pratique.

Pour chaque appareil à haut risque, capturer:

1. Statut brut, en-têtes et corps de réponse avant le décoding du SDK.
2. La valeur de retour normalisée par SDK ou l'exception.
3. La projection sémantique attendue pour l'ère sélectionnée.
4. Les champs soulevés, synthétisés, dépouillés ou modifiés par le SDK.

L'échantillon permet la suppression de la comptabilité bancaire connue uniquement par SDK, comme `resultType`- Je suis là .`_meta`- Je suis là .`ttlMs`, et `cacheScope`Il rapporte une chute de la charge utile de l'application.`futureHint`Parce que ce champ sémantique inconnu a disparu.

Ne supposez pas que chaque différence est un bug SDK. L'important est de rendre la transformation visible. Décidez si votre composant est un point d'extrémité de l'application, qui peut ignorer un champ additif, ou un intermédiaire transparent, qui devrait le préserver.

Exécutez le différentiel contre chaque SDK et version que vous expédez. Si deux SDK normalisent la même transcription différemment, la politique de sortie devrait indiquer quel comportement est acceptable plutôt que de choisir la sortie la plus pratique après le fait.

## Capture de preuves par procuration

La plupart des défaillances de PCM de production se produisent dans plus d'un processus.

| View | Minimum evidence |
|---|---|
| Ingress | request headers, JSON-RPC body, content type, authenticated route, receive time |
| Origin | forwarded headers and body digest, origin status, response headers and body |
| Egress | client-visible status, headers, body, and send time |

L'échantillon détecte deux transformations courantes:

- une erreur HTTP 400 ou 404 JSON-RPC d'origine devient un proxy générique 500
- le corps de sortie JSON-RPC diffère du corps d'origine

Ajouter des affirmations spécifiques au déploiement pour le type de contenu, `Accept`Il est possible de prendre des informations sur les deux côtés de la terminaison TLS lorsque la politique le permet.

## Rédaction avant que les preuves ne quittent la mémoire

La rédaction fait partie des opérations de conformité, pas un travail de nettoyage ultérieur. Appliquez-la avant la sérialisation, le hachage, les journaux, les objets de test ou les téléchargements de défaillance.

Le cas d'échantillon plie les noms des clés et supprime les séparateurs avant de les correspondre, puis remplace récursivement les valeurs sous des clés telles que `Authorization`- Je suis là .`Cookie`- Je suis là .`Set-Cookie`- Je suis là .`X-Api-Key`- Je suis là .`accessToken`- Je suis là .`clientSecret`- Je suis là .`registrationAccessToken`- Je suis là .`token`- Je suis là .`password`- Je suis là .`secret`, et `api_key`- la canonisation et le denyliste doivent utiliser la même forme afin que les variantes camel, hyphénées, soulignées et ponctuées ne puissent pas contourner la politique de l'autre.`query`peuvent toujours contenir des données personnelles ou réglementées.

La mise en cache des données de preuve éditées. Gardez les captures brutes uniquement dans un système approuvé de courte durée lorsque une enquête spécifique les nécessite. Un digeste prouve quel groupe édité a conduit à la décision; il ne révèle pas la valeur supprimée.

## Faites de la santé et du retour une partie de la porte

La conformité au protocole est nécessaire mais pas suffisante pour la libération. Un candidat conformant peut toujours démarrer, fuir la mémoire ou surcharger une dépendance.

Définir une fenêtre de santé avant le déploiement:

- nombre minimum d'échantillons
- taux d'erreur maximum
- le pourcentage de latence maximal
- limitations de saturation ou de ressources
- durée de l'observation
- comparaison avec la ligne de base admise

Définir aussi les preuves de retour avant le déploiement:

- version préalable exacte
- digeste des preuves d'admission
- SHA-256 artifèque et broches de description
- état actuel du registre
- résultat de santé actuel
- procédure de restauration de la route
- une attestation sur ces champs exacts d'une identité de contrôleur de décharge fiable

Exiger que l'objectif de réintégration soit vérifié et sain avant la promotion, et pas seulement après l'échec du candidat.

Si un candidat échoue et que l'objectif de réouverture manque de cette preuve, arrêtez la circulation au lieu de deviner.

Ne réduisez pas la préparation aux vérifications de véracité, telles que la version non vide, `healthy: "yes"`L'échantillon nécessite des types exacts, un statut actif, trois digests SHA-256, un signataire de confiance et une attestation HMAC-SHA-256 valide sur la charge utile complète de roulement. Sa clé démo déterministe est un fichier non secret. Injectez une clé protégée, un résultat de vérification KMS ou un vérificateur d'attestation de clé publique à la limite de sortie en production.

La porte de sortie refuse également la transcription vide, le différentiel SDK ou les preuves proxy. Chaque source doit contenir des digestes de preuves valides.

## Faites-le

- Je vous ai donné le harnais de la bibliothèque standard.

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
```

La démo exécute exactement quinze transcriptions dorées et négatives, y compris des résultats de finalisation valides et malformées, compare un résultat brut avec une vue SDK, inspecte un proxy qui a effondré une erreur d'origine, évalue la santé, authentifie les preuves de retour et sélectionne cette cible.

Forme attendue:

```json
{
  "transcriptsPassed": 15,
  "transcriptsTotal": 15,
  "sdkDroppedFields": ["futureHint"],
  "proxyIssues": [
    "proxy collapsed a protocol error into HTTP 500",
    "proxy changed the origin JSON-RPC body"
  ],
  "releaseAction": "rollback",
  "evidenceDigest": "..."
}
```

Lire `code/main.py`dans cet ordre:

1. `validate_request()`met en œuvre des règles de demande et de titre spécifiques à l'ère.
2. `validate_result()`Il est important de noter que les données de référence sont disponibles dans les pays tiers.
3. `select_era()`Il met en œuvre une politique de retrait stricte et limitée.
4. `run_transcript()`évalue les fixations dorées et négatives.
5. `compare_sdk_view()`- les différences de normalisation.
6. `inspect_proxy()`Comparer les preuves d'entrée, d'origine et d'exode.
7. `redact()`Il élimine les secrets évidents avant de déchiffrer les preuves.
8. `rollback_evidence_ready()`valide les champs de pin exacts et l'attestation de libération fiable.
9. `ReleaseGate.evaluate()`s'ajoute à la conformité non vide, au SDK, au proxy, à la santé et aux preuves de retour.

## Utilisez-le

Remplissez le harnais à quatre points:

1. À chaque changement de mise en œuvre avec un adaptateur d'essai en cours.
2. Contre les binaires client et serveur construits sur le transport réel.
3. Par le proxy ou la passerelle déployée dans un environnement de mise en scène.
4. Pendant le déploiement des canaries avec des preuves de santé et de retour.

Gardez les mêmes noms de cas stables à travers les couches. `negative-header-body-mismatch`Les données de preuve seront différentes parce que la limite a changé; l'exigence ne devrait pas.

Conservez les schémas de fixation dans le contrôle de version. Conservez les preuves de course éditées dans votre système de sortie. Conservez les captures brutes de courte durée uniquement sous contrôle d'accès aux incidents.

## Laboratoire interactif

### Laboratoire A: prouver la frontière de l'ère

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/code
python3 -q
```

Je vais courir .

```python
from main import *
validate_result({"tools": []}, "legacy")
validate_result({"tools": []}, "modern")
```

L' appel héréditaire est inférieur .`complete`L' appel moderne soulève`ProtocolViolation`- Maintenant , le test de rétroaction:

```python
select_era({"kind": "timeout"}, "fallback")
select_era(
    {"kind": "timeout"},
    "fallback",
    legacy_allowed=True,
    legacy_evidence={"kind": "initialize_success", "protocolVersion": LEGACY_VERSION},
)
select_era({"kind": "jsonrpc_error", "code": -32021}, "fallback")
```

Le premier délai ne se ferme pas parce que le silence n'est pas une preuve d'héritage. Le deuxième appel sélectionne l'héritage uniquement parce que la configuration le permet et qu'un résultat d'initialisation hérité valide a été observé.

### Laboratoire B: champ additif versus discriminateur

```python
validate_result({"resultType": "complete", "tools": [], "futureHint": True}, "modern")
validate_result({"resultType": "future_mode", "tools": []}, "modern")
```

Le premier résultat préserve `futureHint`Le second est rejeté parce que le facteur de discrimination du cycle de vie est inconnu.

### Laboratoire C: inspecter une transformation du SDK

```python
compare_sdk_view(
    {"resultType": "complete", "tools": [], "futureHint": {"mode": "new"}},
    {"tools": []},
)
```

Décidez si votre composant peut être ignoré `futureHint`Envoyez le choix dans la politique de libération.

### Laboratoire D: réparer le proxy

Modifiez l'échange de démo pour que l'exode préserve le statut d'origine et le corps.`python3 main.py`Les problèmes de proxy devraient disparaître, mais le différentiel SDK bloque toujours la promotion.`futureHint`dans la vue du SDK et observez le changement d'action à `promote`quand chaque source de preuves passera.

## Laboratoire de pratique

Ajouter des transcriptions SSE à la demande à l'arsenal.

Les exigences:

- Capture de l'état de réponse, du type de contenu, des événements SSE commandés et de la fin du flux.
- Prouver que chaque événement JSON-RPC a un résultat ou une erreur valide spécifique à l'ère.
- Ajouter un cas négatif pour un proxy qui tamponne l'ensemble du flux avant de le rediriger.
- Ajouter un cas négatif pour un événement SSE dont l'identifiant JSON-RPC diffère de la demande.
- Rédaction des données sur l'événement avant de rédiger des preuves.
- Inclure la durée du flux, la latence du premier événement et le nombre d'événements dans la fenêtre santé.
- Faites en sorte que la porte de sortie choisit seulement une cible de retour prouvée lorsque le flux échoue.

Le succès signifie que le même cas se déroule directement et par le proxy, avec un rapport qui identifie la limite exacte qui a changé le comportement.

## Artéfacts expédiés

Cette leçon va à l' air .`outputs/skill-mcp-conformance-release-gate.md`. Utilisez-le pour transformer un serveur, un client, une passerelle ou un changement de SDK en une matrice de conformité versionnée et une décision de sortie.

## Vérifiez

Exécutez la suite démo et déterministe:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La vérification doit prouver:

- chaque transcription dorée et négative incluse atteint son résultat attendu
- Les requêtes modernes nécessitent les clés de métadonnées avec des noms précis
- Les noms des en-têtes HTTP sont correspondants de manière insensible aux cas et codés `Mcp-Name`les valeurs sont décodées exactement
- Le code de déséquilibre entre l'en-tête et le corps renvoie le code de déséquilibre moderne
- la version de réponse, l'identifiant, l'exclusivité du résultat ou de l'erreur, la forme de l'erreur et la cartographie HTTP sont validés
- les exigences relatives à la liste des outils, à la tâche et à la charge utile de réalisation spécifiques à la méthode sont appliquées
- chaque observation .`HeaderMismatch`nécessite un HTTP 400 JSON-RPC réel `-32020`réponse
- crues `Mcp-Name`l'espace blanc est rejeté pendant les voyages de retour dans l'espace blanc avec un code sentinelle exact
- une disparition `resultType`est valable uniquement pour l'ère de l'héritage sélectionnée
- champs additifs survivent à la validation brute tandis que les types de résultats inconnus échouent
- Les types de résultats d'extension nécessitent leur capacité annoncée
- Les erreurs modernes reconnues ne causent jamais de rétrécissement.
- les notifications ne produisent aucune réponse JSON-RPC
- La suppression de la comptabilité du SDK et la perte de champ sémantique sont distinguées
- L'erreur de proxy est détectée et les informations sont éditées récursivement sur camelCase et les variantes de séparateur
- La promotion nécessite une transcription non vide, un SDK, un proxy et des preuves opérationnelles saines
- La promotion et le retour à l'emploi nécessitent tous deux une cible de retour à l'emploi authentifiée, fixée, active et saine

## Mode de défaillance de la production

| Failure | What the weak test reports | What the harness must prove |
|---|---|---|
| SDK synthesizes a missing discriminator | “tools/list passed” | Raw modern result lacked `resultType` and is invalid |
| Client downgrades after `-32021` | “legacy retry worked” | Recognized modern error forbids fallback |
| Unknown result type treated as complete | “response parsed” | Unadvertised lifecycle discriminator is rejected |
| Proxy authorizes one tool and origin executes another | “request reached server” | `Mcp-Name` equals the body routing name at every hop |
| Harness throws before reading the server response | “header mismatch test passed” | HTTP 400 and JSON-RPC `-32020` response are captured and validated |
| Proxy turns origin 400 into generic 500 | “upstream error” | Origin and egress statuses and JSON-RPC bodies are preserved |
| Notification middleware emits `{result: null}` | “handler returned none” | Final egress body is empty and no JSON-RPC response exists |
| SDK strips an additive field | “typed objects match” | Raw and normalized views show the exact dropped field |
| Failure artifact leaks a bearer token | “debug bundle uploaded” | Redaction occurred before hashing, logging, or upload |
| Credential key style bypasses redaction | “denylist contains api_key” | CamelCase and separator variants share one canonical denylist form |
| Canary has no samples but appears healthy | “zero errors” | Minimum sample count is enforced |
| Rollback selects an unknown build | “previous deployment restored” | Target version, admission digest, pins, status, and health are present |

## Règlement d'exploitation

Test des octets que vous envoyez, des octets de chaque intermédiaire, la sémantique exposée par chaque SDK et les opérations de preuve seront utilisées sous pression. La compatibilité est une branche explicite. Le retour à l'emploi est une action de libération appuyée par des preuves.

## Pour en savoir plus

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP version negotiation](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Official MCP conformance project](https://github.com/modelcontextprotocol/conformance)
