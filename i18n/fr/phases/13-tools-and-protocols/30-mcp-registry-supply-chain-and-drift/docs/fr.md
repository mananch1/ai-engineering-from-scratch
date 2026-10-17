# Chaîne d'approvisionnement du registre MCP: admission, dérive et retour

> L'entrée du registre vous indique ce qu'un éditeur a déclaré.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 17 (gateways and registries), Phase 13 · 18 (production authentication)
**Time:** ~90 minutes

## Objectifs d'apprentissage

- Publication séparée du registre, provenance du paquet, découverte en temps d'exécution et approbation locale.
- Vérifiez un espace de noms du serveur MCP sans faire confiance au nom dans son propre dossier.
- Pin publication immuable, source d'exécution, provenance et preuve de descripteur en direct.
- Détecter les changements d'état du registre et la dérive de l'heure d'exécution après l'admission.
- Retourner le routage à une version précédemment admise sans réécrire l'historique.
- Gardez un registre d'admission qui explique chaque décision.

## Le problème

Vous trouverez`com.example/inventory`La description est correcte, le serveur répond.`server/discover`- Je suis désolé .

Ce n'est pas un fait, c'est une chaîne de faits de différentes autorités:

1. Un éditeur authentifié pour un espace de noms a soumis un enregistrement.
2. Un registre de colis servait un artefact avec une identité et une digestion spécifiques.
3. Un terminal en cours d'exécution a rapporté une version du protocole, des capacités, des outils et des informations du serveur de diagnostic.
4. Votre organisation a décidé que cette combinaison exacte était autorisée.

Une publication valide peut toujours être dépréciée. Une balise de paquet peut indiquer un artefact inattendu si vous ne le digérez pas. Un serveur peut ajouter un outil destructeur après examen. Un déroutement peut discrètement choisir une version qui n'a jamais été admise.

Le contrôleur d'admission est un contrôleur d'admission avec des preuves à chaque limite.

## Le registre est un indice, pas votre système d'approbation

Le registre officiel du MCP stocke les métadonnées du serveur.`server.json`enregistrer des noms d'une version du serveur et déclarer un ou plusieurs paquets ou des terminaux distants. Les règles de publication ajoutent une authentification de l'espace de noms, des vérifications de propriété du package, des règles de registre restreintes et une position de métadonnées étroite de l'éditeur.

Ces contrôles répondent aux questions de publication.

| Boundary | Question | Evidence owner |
|---|---|---|
| Namespace | Was the publisher allowed to use this name? | Registry authentication plus your verified namespace input |
| Record | What did the publisher declare for this version? | Immutable `server.json` digest |
| Execution source | Which package or remote endpoint will execute? | Declared source fields, verified ownership result, transport, and trusted digest |
| Runtime | What does the endpoint expose now? | `server/discover` and tool descriptors |
| Admission | Did your policy approve this exact set? | Local pin and ledger entry |
| Operations | Is it still safe, and what can replace it? | Drift checks, status sync, health, and rollback route |

La version du schéma du Registre et la version du protocole MCP sont indépendantes.`2025-12-11`schéma du serveur alors que le serveur en direct prend en charge MCP `2026-07-28`Ne jamais en déduire l'un de l'autre.

```figure
mcp-registry-admission
```

## Sept contrôles dans une seule décision d'admission

### 1. Vérification de l'espace de noms

Les noms officiels du Registre utilisent des espaces de noms authentifiés. Un domaine vérifié peut mapper à un préfixe de domaine inversé. Par exemple, le contrôle de `example.com`peut établir `com.example/*`- Je suis désolé .

Ne pas accepter une vérification de préfixe de chaîne:

```python
server_name.startswith("com.example")
```

Ce qui accepte aussi `com.exampleevil/tool`- Partagez le nom à`/`En effet, les données de référence sont les données de référence de l'espace de noms, et les données de référence sont les données de référence de l'espace de noms, et les données de référence sont les données de référence de l'espace de noms, et les données de référence sont les données de référence de l'espace de noms, et les données de référence sont les données de référence de l'espace de noms, et les données de référence sont les données de référence de l'espace de noms, et les données de référence sont les données de référence de l'espace de noms, et les données de référence sont les données de référence de l'espace de noms, et les données de référence sont les données de référence de l'espace de référence.

Les espaces de noms supportés par GitHub et les espaces de noms supportés par domaine utilisent différents chemins d'authentification. Normalisez l'un ou l'autre chemin dans une entrée d'admission: la chaîne exacte de l'espace de noms vérifiée.

### 2. Résultats de la recherche

Pour un enregistrement de colis, la déclaration et l'artefact récupéré doivent se joindre sur des champs explicites:

- type de registre de colis
- identifiant de colis
- version de l'emballage
- résultat de propriété vérifié
- digeste d' artefact téléchargé

Valider également le transport de colis déclaré. Un enregistrement avec un seul point d'extrémité distant est valide et ne doit pas être rejeté pour manque d'un colis. Pour une source distante, joindre l'URL déclarée et le type de transport à la propriété de point d'extrémité vérifiée indépendamment et un digeste de la connexion fiable ou des preuves de déploiement.

Le code de leçon prend en charge les deux types de source et hashes la source sélectionnée avec la source du Registre, le nom du serveur, la version du Registre, le digeste des enregistrements et le digeste des preuves.

Ne jamais accepter un digeste fourni uniquement par l'artefact que vous essayez de vérifier, mais le calculer à une limite de récupération fiable ou le recevoir auprès d'un service de colis dont vous validez le résultat de vérification.

### 3. - Je ne suis pas le seul à avoir fait la décision.

Les versions du registre sont des identifiants de publication uniques. Les métadonnées publiées sont immuables. Un enregistrement modifié nécessite une nouvelle version. La version sémantique est recommandée, mais le registre ne l'exige pas et n'accepte pas les gammes de versions.

Ça veut dire ...`^1.4`n'est pas un pin d'admission. ni lastest. Un pin utile contient:

```json
{
  "server": "com.example/inventory",
  "version": "1.0.0",
  "recordDigest": "...",
  "source": {"kind": "package", "registryType": "pypi"},
  "sourceDigest": "...",
  "toolsetDigest": "...",
  "provenanceDigest": "...",
  "registryStatus": "active"
}
```

En fixant plusieurs couches, vous pouvez identifier quelle limite a changé. Une modification de la digestion d'enregistrement dans la même version du Registre est une défaillance de l'intégrité du Registre.

### 4. Détection de dérive en direct

L'entrée doit observer le serveur qui recevra réellement le trafic.`server/discover`, liste ou obtenir autrement les descripteurs d'outils exposés par votre chemin de confiance, et vérifier:

- `2026-07-28`Il est là .`supportedVersions`
- toutes les capacités requises localement sont présentes
- chaque descripteur d'outil a l'identité et la surface du schéma requis
- le digeste de descripteur normalisé correspond à la broche admise lors de contrôles ultérieurs

Le résultat facultatif `_meta["io.modelcontextprotocol/serverInfo"]`La valeur est un contexte d'affichage, de journal et de débogage auto-déclaré. Enregistrez-le comme preuve de diagnostic, mais n'utilisez jamais pour établir l'espace de noms, la propriété du package, la propriété du point final, l'admission ou toute autre décision de sécurité.`serverInfo`Alias à l' extérieur `_meta`n'est pas le domaine contractant et ne doit pas être promu comme preuve diagnostique.

Normalizer uniquement les champs dont l'ordre n'a pas de signification. L'échantillon triera la liste des outils par nom stable avant de faire un hachage, de sorte qu'un changement inoffensif de l'ordre de liste ne provoque pas de dérive. Il ne rejette pas les champs descripteurs. Un nouvel outil, un schéma modifié, une description modifiée ou de nouvelles annotations modifient la broche.

L'échantillon traite les descripteurs malformés et tout changement de digestion des descripteurs comme dérivant, quarantaine la broche, enlève son itinéraire actif et bloque cette version comme une cible de retour.

### 5. Le statut du registre est en état de vie

L' API du Registre attache un niveau de réponse `_meta`Objet à côté de chaque enregistrement serveur.`_meta["io.modelcontextprotocol.registry/official"]`- Passez la réponse .`_meta`l' admission et la lecture `_meta["io.modelcontextprotocol.registry/official"].status`- Une direct .`_meta.status`La valeur n'est pas la forme officielle du fil. Ne confond pas les métadonnées de réponse avec les données du registre de publication `_meta`- Le statut peut être:

- `active`: retourné par défaut et admissible à l' admission locale
- `deprecated`: toujours détectable avec un avertissement, mais pas plus un choix automatique sûr
- `deleted`: caché par défaut alors que son dossier historique reste disponible par le biais de vues supprimées ou incrémentielles

Si une version active devient obsolète ou supprimée, quarantainez son pin et arrêtez de lui router de nouveaux travaux. Gardez les preuves. La suppression de la liste par défaut n'est pas une autorisation pour effacer votre piste d'audit.

Les métadonnées personnalisées fournies par l' éditeur appartiennent uniquement à `_meta.io.modelcontextprotocol.registry/publisher-provided`Les données métadonnées de réponse gérées par le registre sont séparées.

### 6. Le retour à l'emploi signifie la restauration de la route

Une publication immuable n'est pas édité pendant le rollback.

Une cible sûre doit:

1. Avoir un dossier d'admission complété.
2. Vous avez toujours un statut de registre actif dans votre police.
3. Ne pas être mis en quarantaine par des preuves de sécurité ou de temps de course.
4. Toujours résolvez le paquet coincé et le décritor en direct.
5. Passez les contrôles médicaux actuels.

L'échantillon se concentre sur les trois premières conditions. Un véritable reconcileur doit récupérer le paquet et vérifier le point final en direct avant l'activation.

### 7. Ajouter un registre d'admission

Une base de données d'admission indique ce qui est actif.

Chaque entrée d'échantillon contient une séquence, un moment, un événement, un serveur, une version, un résultat, des raisons, des preuves, le hash de l'entrée précédente et son propre hash.

Ce n'est pas magique, mais évident. Les annuaires périodiques d'ancrage sont placés dans un domaine de confiance séparé, comme les métadonnées de sortie signées ou le stockage de la rédaction unique. Restrictez qui peut ajouter. Gardez les jetons d'autorisation, les identifiants de package, les arguments d'outils et les données de terminaux privés hors de la preuve.

## Faites-le

Le contrôleur est activé .`code/main.py`Il utilise uniquement la bibliothèque standard Python.

Commencez par la démonstration finie:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
```

La démonstration effectue cinq opérations:

1. Admettez-le`1.0.0`avec un espace de noms, une provenance de paquets, un protocole, des capacités et des outils correspondants.
2. Admettez-le`1.1.0`et le rendre actif.
3. Observez un outil de suppression inattendu en temps d'exécution.
4. Observer le statut du registre pour `1.1.0`devenir `deprecated`- Je suis désolé .
5. Retourner le routage à l' encore admis `1.0.0`Je suis en train de vous dire.

Forme attendue:

```json
{
  "admitted": [true, true],
  "driftAllowed": false,
  "rollbackAllowed": true,
  "activeVersion": "1.0.0",
  "ledgerValid": true
}
```

Lisez la mise en œuvre dans cet ordre:

1. `namespace_for_domain()`et `namespace_matches()`établir l'autorité de dénomination exacte.
2. `digest()`et `normalized_tools()`produire des preuves déterministes.
3. `RegistryAdmissionController.admit()`Les résultats de la recherche sont les suivants:
4. `check_live()`comparer une nouvelle observation avec la broche.
5. `observe_registry_status()`les versions de quarantaine dont l'état de registre change.
6. `rollback()`n'active qu'une cible admissible précédemment admise.
7. `AdmissionLedger.verify()`détecte les changements dans l'historique enregistré.

## Utilisez-le

Mettez le contrôleur entre la découverte et le routage:

```text
Registry sync -> artifact verifier -> live discovery -> admission controller -> route table
                                               |                 |
                                               v                 v
                                          evidence store    admission ledger
```

Utilisez des identités séparées pour ces tâches. Un synchroniseur de registre a besoin d'accès à la lecture des métadonnées. Un vérificateur d'artefact a besoin d'accès à la récupération de paquets. Un reconcileur de route a besoin d'autorisation pour activer une pin approuvée. Aucun d'entre eux n'a besoin de toutes les informations d'identification.

Faites expliciter l'état de déploiement. Approuvé signifie la politique de preuve adoptée. Active signifie que la route la sélectionne actuellement. Quarantié signifie qu'il ne peut pas recevoir de nouveaux travaux. Superseded signifie qu'une autre version admise est active. Ne codifiez pas les quatre significations dans un seul booléen.

Exécutez l' admission avant d' exposer un serveur dans `tools/list`- Dans le cas contraire, un client peut découvrir un outil pendant l'écart entre la publication et l'évaluation des politiques.

## Laboratoire interactif

Vous verrez une frontière s'effondrer à la fois.

### Laboratoire A: collision de l'espace de noms

Ouvrez une coquille Python depuis le répertoire de code:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/code
python3 -q
```

Alors, courez:

```python
from main import namespace_matches
namespace_matches("com.example/inventory", "com.example")
namespace_matches("com.exampleevil/inventory", "com.example")
```

Le premier résultat est `True`; le second est `False`. Remplacez la comparaison exacte par `startswith`Il faut donc vérifier pourquoi le deuxième nom traverse la frontière.

### Laboratoire B: dérive de descripteur

```python
from main import *
times = iter(f"2026-08-21T12:00:{n:02d}+00:00" for n in range(10))
c = RegistryAdmissionController(clock=lambda: next(times))
meta = {OFFICIAL_META_KEY: {"status": "active"}}
c.admit(sample_record("1.0.0"), meta, "com.example", evidence_for("1.0.0"), sample_live("1.0.0"))
c.check_live("com.example/inventory", "1.0.0", sample_live("1.0.0", True))
```

Vérifiez les raisons et l'état de l'itinéraire. Le dossier du colis et du registre n'a pas changé. La surface de l'outil de fonctionnement l'a fait, le contrôleur a donc mis en quarantaine et désactivé la broche. C'est pourquoi le contrôle de la chaîne d'approvisionnement doit continuer après l'installation.

### Laboratoire C: état et réouverture

Admettez-le`1.1.0`, marquer dépassé, et essayez les deux cibles de retour:

```python
c.admit(sample_record("1.1.0"), meta, "com.example", evidence_for("1.1.0"), sample_live("1.1.0"))
c.observe_registry_status("com.example/inventory", "1.1.0", "deprecated")
c.rollback("com.example/inventory", "1.1.0", "unsafe retry")
c.rollback("com.example/inventory", "1.0.0", "restore known release")
c.ledger.verify()
```

La cible mise en quarantaine est rejetée, le pin actif précédent est accepté, le registre reste valide.

## Laboratoire de pratique

Étendre le contrôleur avec une passerelle d'homologation pour deux personnes.

Les exigences:

- Les approbations sont conservées comme des références signées à des preuves, et non comme des noms changeables dans la broche.
- Exiger deux identités différentes de l' examinateur pour un ensemble d' outils qui contient un outil avec `destructiveHint: true`- Je suis désolé .
- Rejeter les identités de réviseurs.
- Conserver la tentative d'admission initiale dans le registre lorsque l'approbation est incomplète.
- Ajoutez des tests pour zéro, un, deux exemplaires et deux approbations distinctes.
- Ne mettez pas de jour les signatures, les informations d'identification ou les arguments privés complets.

Le succès signifie qu'un outil destructeur ne peut pas devenir actif tant que les deux identités n'ont pas approuvé le recueil exact, le paquet et l'ensemble d'outils.

## Artéfacts expédiés

Cette leçon va à l' air .`outputs/skill-mcp-registry-admission.md`. Utilisez-le comme un répertoire plat et réutilisable lors de l'examen d'une nouvelle version du Registre ou de l'enquête sur la dérive. Il définit les entrées, les règles de refus, le paquet de preuves, la reconciliation de statut et la preuve de rétroaction sans dépendre des noms des classes d'échantillon.

## Vérifiez

Exécutez la démonstration et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La vérification doit prouver:

- limites exactes de l'espace de noms rejeter les préfixes similaires
- Seul le statut officiel du registre avec espace nommé peut rendre une version éligible
- l'emballage non vérifié ou non correspondant et les preuves à distance sont rejetées
- Les métadonnées de l'éditeur ne peuvent pas se faire passer pour des métadonnées gérées par le Registre
- l'ordre des outils est normalisé sans cacher les changements de descripteur
- les structures de colis et d'outils malformés sont rejetées en toute sécurité
- `serverInfo`reste diagnostique et ne fournit jamais l'autorité d'admission
- Descriptor dérive quarantaines, désactive et bloque le retour à la broche
- changements de statut des broches actives de quarantaine
- le retour ne peut pas sélectionner une version mise en quarantaine ou inconnue
- une manipulation du registre est détectée

## Mode de défaillance de la production

| Failure | Why it happens | Required response |
|---|---|---|
| Name looks valid but namespace was never authenticated | Policy trusted record text | Reject until a trusted namespace verifier supplies the exact prefix |
| Same package coordinate returns new bytes | Mutable upstream or compromised distribution | Stop activation, retain both digests, investigate the fetch boundary |
| “Latest” changes without review | Floating selection escaped the pin | Resolve only exact admitted versions and digests |
| New tool appears after approval | Runtime drift or a different deployment | Quarantine the route and capture a fresh descriptor observation |
| Deprecated version remains active | Status sync is missing or delayed | Reconcile status on a schedule and before activation |
| Deleted record disappears from default sync | Client requested only active records | Use incremental or deleted-aware reconciliation and preserve local history |
| Rollback target was never admitted | Route control and approval state are disconnected | Refuse rollback and run a new admission for the target |
| Ledger verifies locally after an attacker rewrites all entries | Hash chain has no external anchor | Publish signed ledger heads to a separate trust domain |
| Evidence contains bearer tokens or tool arguments | Logging copied whole requests | Redact at collection time and store only the minimum proof |

## Règlement d'exploitation

Réponses de publication  Cette identité peut-elle publier ce nom? Réponses d'admission Vous voulez exécuter cet artefact exact et exposer ce comportement exact? Gardez ces décisions séparées, fixez chaque joint et faites en sorte que le retour choisisse des preuves plutôt que la mémoire.

## Pour en savoir plus

- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [Official Registry OpenAPI contract](https://registry.modelcontextprotocol.io/openapi.yaml)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
