# Contrats et contenu des outils de MCP

> Un outil n'est sûr d'automatiser que lorsque la découverte, les arguments, les résultats, la pagination et le transport de métadonnées sont d'accord sur un contrat.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07, 09, and 10
**Time:** ~120 minutes

## Objectifs d'apprentissage

- Définir les entrées et sorties des outils avec JSON Schema 2020-12.
- Valider les résultats structurés sans supposer qu'ils sont des objets JSON.
- Choisissez entre le texte, l'image, l'audio, les liens vers les ressources et les ressources intégrées.
- Rejeté dangereux`x-mcp-header`définitions avant qu'un outil n'atteigne le modèle.
- Encodez les valeurs de paramètre-tête et vérifiez la parité exacte de tête-à-corps.
- Paginaison transversale du curseur sans interpréter les valeurs du curseur.
- Lié et autorisé`completion/complete`- Je vous propose des suggestions.

## Le problème

Appeler une fonction Python est facile. Appeler une capacité à distance via un hôte d'IA est un problème de contrat.

Le serveur publie un descripteur. Le client transforme ce descripteur en contexte de modèle et en interface utilisateur. Le modèle crée des arguments. Un gateway peut router la demande à partir d'en-têtes miroirées. Le serveur exécute l'outil. Le client décide ensuite si le résultat est suffisamment sûr et valide pour revenir au modèle.

Une seule limite faible corrompt toute la chaîne.

Considérez cinq erreurs:

- Le descripteur dit que le résultat est un objet, mais le serveur renvoie un tableau.
- Le client arrête la pagination lorsque `nextCursor`est une chaîne vide.
- Un paramètre de jeton est reflété dans une en-tête HTTP et devient visible pour les intermédiaires.
- Une valeur de routage Unicode est envoyée sous forme d'en-tête brut, puis le gateway et l'origine interprètent différents octets.
- Un point final de termination suggère un environnement de production à un appelant qui n'a pas accès à celui-ci.

Aucune de ces défaillances ne peut être corrigée par une meilleure mise en œuvre.

## Le pipeline de contrat

Traitez chaque appel d' outil comme cinq portes:

1. **Discover.**Lisez une liste déterministe et paginée d'outils.
2. **Admit.**Valider chaque descripteur et appliquer la politique de sécurité locale.
3. **Invoke.**Valider les arguments et créer des métadonnées de transport.
4. **Execute.**Exécutez le gestionnaire et classez correctement les défaillances.
5. **Consume.**Valider les blocs de contenu et les sorties structurées avant l'utilisation du modèle.

```figure
mcp-contract-pipeline
```

L'hôte possède les portes d'entrée et de consommation. Un serveur ne peut forcer un client à faire confiance à ses annotations, schémas ou sorties.

## Le schéma JSON est une limite de temps d'exécution

Dans le MCP `2026-07-28`- Je suis là .`inputSchema`et `outputSchema`utiliser JSON Schema.`$schema`est absent, le dialecte par défaut est 2020-12.

Le schéma d'entrée doit être un objet schéma.

```json
{
  "type": "object",
  "additionalProperties": false
}
```

C' est plus strict que ...`{ "type": "object" }`, qui accepte des propriétés arbitraires.

Un schéma de sortie est facultatif. Une fois qu'un serveur en publie un, chaque outil complet
résultat s' engage à retourner conforme `structuredContent`, y compris les résultats
avec `isError: true`. Le drapeau d'erreur classe le résultat de l'exécution; il ne
Les clients doivent plutôt valider le résultat
de faire confiance au descripteur.

### Le contenu structuré est une valeur JSON

Ne codez pas dur `structuredContent`comme un dictionnaire.

- un objet;
- un ensemble;
- une chaîne;
- un numéro;
- un booléen;
- `null`- Je suis désolé .

Cet outil renvoie un tableau:

```json
{
  "name": "tag_catalog",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "array",
    "items": {"type": "string"}
  }
}
```

Son résultat est valable:

```json
{
  "resultType": "complete",
  "content": [
    {
      "type": "text",
      "text": "[\"contracts\", \"mcp\", \"stateless\"]"
    }
  ],
  "structuredContent": ["contracts", "mcp", "stateless"],
  "isError": false
}
```

Pour la compatibilité, les résultats structurés doivent également inclure des JSON sérialisés dans un bloc de texte. Le texte n'est pas la source de validation. `structuredContent`- Je suis.

### Un petit validateur enseigne toujours la limite

La leçon utilise un sous-ensemble délibéré de JSON Schema parce qu'il reste à l'intérieur de la bibliothèque standard Python.

- les types objet, tableau, chaîne, entier, nombre, booléen et nul;
- les propriétés requises;
- `additionalProperties: false`Le dépôt de la commission
- les éléments du tableau;
- les valeurs enum;
- longueur minimale de la chaîne.

Il ne s'agit pas d'un remplacement pour un validateur de production complet.

## Les blocs de contenu coûtent différemment

Le `content`L'ensemble peut combiner plusieurs types de contenu.

| Type | Use it for | Main boundary |
|------|------------|---------------|
| `text` | Human and model-readable summaries | Treat text as untrusted output |
| `image` | Visual evidence encoded as base64 | Validate media type and size |
| `audio` | Spoken or recorded output encoded as base64 | Validate media type and duration limits |
| `resource_link` | A URI the client may fetch later | Reauthorize the later resource read |
| `resource` | Data embedded directly in the result | Enforce payload and content limits now |

Un lien vers une ressource n' est pas la preuve que la ressource apparaît dans `resources/list`Le client applique toujours sa politique de ressources lorsqu'il suit l'URI.

Une ressource intégrée évite un autre voyage aller-retour mais augmente la taille de la réponse actuelle. Utilisez des liens pour des objets importants ou en mutation indépendante. Utilisez des ressources intégrées pour de petites preuves qui doivent voyager atomiquement avec le résultat.

La leçon est `evidence_bundle`Le client valide chaque bloc avant d'accepter le résultat.

## `x-mcp-header`Est-ce que le routage des métadonnées

Une propriété à l' intérieur .`inputSchema`peut déclarer `x-mcp-header`. Sur HTTP en streaming, le client reflète cet argument dans `Mcp-Param-{name}`- Je suis désolé .

```json
{
  "region": {
    "type": "string",
    "x-mcp-header": "Region"
  }
}
```

Avec `region: "eu-west"`, le transport peut émettre:

```http
Mcp-Param-Region: eu-west
```

L'annotation existe pour qu'un équilibreur de charge, une passerelle ou un moteur de politique puisse rouvrir sans analyser le corps JSON.

Le protocole limite la note:

- le nom de l'en-tête n'est pas vide et suit la syntaxe des jetons de nom de champ HTTP;
- les noms des en-têtes sont uniques sans égard au cas;
- le type de propriété est une chaîne, un entier ou un booléen;
- `number`n'est pas autorisé;
- l' annotation n' apparaît que sur un membre direct de `inputSchema.properties`Le dépôt de la commission
- Les valeurs entières restent à l'intérieur `-9007199254740991`à travers `9007199254740991`- Je suis désolé .

La règle de localisation est syntaxique et fermé.
Il ne s'agit pas seulement des propriétés que votre validateur comprend.
annotation sous l'objet niché `properties`, une `oneOf`branche,`items`, une
définition atteinte par `$ref`, ou tout schéma de sortie.
ne transforme pas le nœud de référence en une propriété directe de niveau supérieur.

Cette leçon ajoute une politique de déploiement: rejeter les descripteurs qui reflètent des noms tels que `password`- Je suis là .`secret`- Je suis là .`token`- Je suis là .`api_key`ou `authorization`. La spécification officielle conseille aux auteurs du serveur de ne pas refléter les paramètres sensibles. Un client peut transformer ce conseil en une règle d'admission dure.

Vérifiez le nom de l'en-tête, pas sa valeur.`Mcp-Param-Region`tout en gardant`eu-west`hors de l'événement d'audit.

### Les valeurs de codage avant de créer des en-têtes HTTP

Une valeur de paramètre ne peut se déplacer en texte ordinaire que lorsqu'elle est une chaîne non vide
de caractères ASCII visibles de `!`à travers `~`et ne ressemble pas à la
Tout le reste utilise la même forme:

```text
=?base64?{Base64UTF8}?=
```

`Base64UTF8`est base 64 standard sur les octets UTF-8 exacts. Ne pas couper,
Encode Unicode, chaînes vides, espaces,
les onglets, les caractères de commande, CR ou LF, l'espace blanc avant ou derrière, et tout autre
valeur commençant par `=?base64?`Encore une fois, le code d'une valeur ressemblant à une sentinelle est
ce qui permet au récepteur de récupérer le texte original littéral au lieu de décoder
Il est aussi syntaxique.

Les booléens rendent en minuscules .`true`ou `false`- Intégres rendus à la base 10 et
doit rester dans la plage de nombres entiers sécurisés JavaScript.
sont rejetés au lieu d'être arrondis par un intermédiaire.

### Le serveur vérifie la copie miroirée

La génération d'en-tête est seulement la moitié du client.
le serveur doit:

1. trouver reconnu `Mcp-Param-*`les noms sans égard au cas de titre-nom;
2. déchiffrer la forme exacte de base64 sentinelle lorsqu'elle est présente;
3. comparer exactement le texte décodé avec l'argument corps JSON correspondant;
4. rejeter une absence, une duplication, une inattendue, une déformation ou une non-coïncidence
   - Le titre reconnu avant l'expédition.

Le rejet est HTTP `400`avec code d'erreur JSON-RPC `-32020`- Ni les
La valeur du corps et sa forme d'en-tête codée ne figurent pas dans le dossier d'audit.
nom d'en-tête reconnu et catégorie de rejet uniquement.

`code/main.py`Il est possible de modéliser directement cette frontière. [Lesson 09](../../09-mcp-transports/)
couvre l'ordre de validation HTTP Streamable plus large, y compris la méthode et
Parité protocole-version.

## Les maîtres de page sont opaques

Les opérations de liste MCP utilisent la pagination du curseur. Le serveur sélectionne la taille de la page et le format du curseur. Le client obtient une décision:

```python
if result.get("nextCursor") is None:
    break
cursor = result["nextCursor"]
```

N' écrivez pas ceci:

```python
if not result.get("nextCursor"):
    break
```

Une chaîne vide est un cursor valide. La vérité s'arrêterait trop tôt.

Les clients ne doivent pas décoder un curseur, l'augmenter, le comparer à un curseur précédent pour la commande ou en déduire un numéro de page. Un serveur peut signer un curseur, le lier à une version de catalogue ou le cartographier à l'état privé. C'est le détail de la mise en œuvre du serveur.

Le serveur d' échantillon retourne délibérément `""`Le client doit envoyer cette valeur exacte à la deuxième demande.

```text
<first request with no cursor>
<second request with cursor "">
```

Les curseurs invalides produisent des paramètres invalides JSON-RPC, code `-32602`- Je suis désolé .

## L'achèvement est une autorisation

`completion/complete`Il est utile pour les formulaires interactifs, mais il peut fuir des noms que les méthodes de liste ordinaires protègent.

Une demande de complémentation nomme une référence et l'argument qui est complété:

```json
{
  "method": "completion/complete",
  "params": {
    "ref": {
      "type": "ref/prompt",
      "name": "deployment_review"
    },
    "argument": {
      "name": "environment",
      "value": "st"
    }
  }
}
```

Le résultat renvoie au maximum 100 valeurs et peut être rapporté `total`plus `hasMore`- Je suis désolé .

Appliquer la même limite d'autorisation utilisée par le prompt ou la ressource référencé.`development`et `staging`- Seul un opérateur peut recevoir`production`- Je suis désolé .

La réalisation de la production doit également:

- validation des entrées;
- filtration à l'attention des appelants;
- demande de déposer auprès du client;
- limitation de taux sur le serveur;
- les comptes de résultats limités;
- les journaux qui n'exposent pas de valeurs sensibles de suggestion.

L'achèvement est une aide, pas un détour de découverte.

## Deux couches d'erreur

Gardez les erreurs de protocole séparées des erreurs d'exécution de l'outil.

Utiliser une erreur JSON-RPC lorsque la demande MCP ne peut pas être envoyée correctement:

- nom de l'outil inconnu;
- forme de demande déformée;
- les métadonnées manquantes de la demande;
- un curseur non valide.

Utilisez un résultat complet avec `isError: true`lorsque l'invocation est arrivée à l'outil et que l'outil rapporte une défaillance exploitable:

- une source de rapport n'est pas disponible;
- une date est hors de la plage de support;
- une règle d'entreprise rejette l'opération demandée.

Les modèles peuvent souvent réparer une erreur d'exécution d'outil. Ils ne peuvent pas réparer un serveur qui a violé son propre schéma de sortie.

Si l'outil déclare un schéma de sortie, modélisez une défaillance actionable à l'intérieur de celui-ci.
Le schéma.`route_report`l'échec renvoie sa région demandée avec
`accepted: false`, aux côtés du texte d' erreur lisible par l' homme et `isError: true`- Je suis désolé .

## Faites-le

`code/main.py`construit les deux côtés de la frontière avec la bibliothèque standard Python.

Le serveur implémentera:

- la validation des métadonnées du MCP par demande;
- `server/discover`disposant d'outils et de capacités de finalisation;
- déterministe `tools/list`la pagination;
- quatre descripteurs d'outils, dont un qui doit être rejeté;
- sortie structurée de tableau;
- chaque type de bloc de contenu actuel de l'outil;
- une passerelle de parité HTTP diffusée qui décode les en-têtes de paramètres reconnus et
  renvoie HTTP `400`plus JSON-RPC `-32020`sur une non-coïncidence;
- l'achèvement autorisé et limité.

Le client met en œuvre:

- l'admission de descripteurs;
- arbre plein`x-mcp-header`la validation du placement et la politique de champ sensible;
- une codage exacte de la valeur UTF-8 ASCII ou base64 visible clairement;
- une boucle de curseur opaque qui suit une chaîne vide;
- argumentation et validation des résultats;
- validation du bloc de contenu;
- les événements d'audit d'en-tête contenant des noms mais pas des valeurs.

Le descripteur délibérément dangereux est l'enseignement des données.

## Utilisez-le

À partir de la racine du référentiel:

```bash
cd phases/13-tools-and-protocols/28-mcp-tool-contracts-and-content/code
python3 main.py
python3 -m unittest discover tests -v
```

Les impressions démo ont accepté les outils, le descripteur rejeté, les deux pagination
requêtes, contenu de tableau structuré, types de blocs de contenu, en-tête miroir
les noms, que ce soit la valeur requise de codage, le statut de parité HTTP, et
les valeurs de finalisation filtrées par l'appelant.

## Laboratoire interactif

Ouvert .`code/main.py`et localiser `TOOLS`- Je suis désolé .

1. Le changement`tag_catalog.outputSchema.type`de `array`à `object`- Je suis désolé .
2. L'équipe doit rejeter le tableau retourné.
3. Retournez le schéma.
4. Gardez la première page `nextCursor`comme `""`, puis faire la dernière page retourner
   `nextCursor: None`au lieu d'omettre le champ.
5. Exécutez les tests et comparez la trace du curseur.
6. Ajouter `x-mcp-header: "Authorization"`à une propriété de chaîne.
7. L'admission de la description de confirmation la rejette avant l'invocation.
8. Essayez .`region`valeurs contenant Unicode, une nouvelle ligne, des espaces environnants, et
   le texte littéral `=?base64?SGVsbG8=?=`Décoder chaque en-tête émis et prouver
   La valeur originale survit exactement.
9. Mettez la note en dessous `oneOf`- Je suis là .`items`, ou une `$ref`définition.
   chaque descripteur est rejeté même si cette branche n'est jamais utilisée par la démo.
10. Retirez l'en-tête reconnu ou modifiez sa valeur décodée.
    état des retours de limite `400`et code JSON-RPC `-32020`- Je suis désolé .

Le but n'est pas de mémoriser une forme JSON, mais de regarder chaque passerelle échouer à la limite qui la possède.

## Laboratoire de pratique

Élargir le laboratoire de contrat avec un`search_evidence`outil.

Les exigences:

1. Son schéma d' entrée accepte `query`- Je suis là .`limit`, et un coffre-fort .`region`champ de routage.
2. Son schéma de sortie est une gamme d'objets avec `uri`- Je suis là .`title`, et `score`- Je suis désolé .
3. Le résultat comprend un texte de compatibilité et un lien vers les ressources par élément.
4. Les arguments rejettent les propriétés inconnues.
5. `limit`est limité par la validation de la demande.
6. Un appelant sans accès à un URI ne voit jamais cet URI à travers la finition ou la sortie de l'outil.
7. Les tests comprennent un score non conforme, une annotation invalidée de l'en-tête et une liste de deux pages.
8. Les tests de valeur d'en-tête couvrent les caractères ASCII, Unicode, contrôle visibles,
   espace blanc, texte ressemblant à un sentinel, et les deux limites entières sécurisées par JavaScript.
9. Le fichier HTTP accepte les noms d'en-tête insensés à l'affaire mais rejette les manquants
   ou des valeurs reconnues non correspondantes avec le statut `400`et code `-32020`- Je suis désolé .

## Artéfacts expédiés

`outputs/skill-mcp-contract-reviewer.md`Il est une compétence de révision plate et réutilisable. Donnez-lui un descripteur d'outil, des résultats d'échantillon, un comportement de pagination et une politique de finalisation. Il renvoie une décision d'admission, un plan de validation des résultats, une politique d'en-tête et des tests d'échec concrets.

## Vérifiez

La leçon est complète lorsque ces déclarations sont vraies:

- `tools/list`renvoie le même ordre logique sur les appels répétés.
- Le client effectue une deuxième demande lorsque `nextCursor`est `""`- Je suis désolé .
- Le descripteur de titre sensible dangereux est exclu tandis que d'autres outils restent disponibles.
- Un tableau passe son schéma de sortie de tableau.
- Un objet ne répond pas à ce même schéma d'un tableau.
- Les résultats d'erreur ne peuvent pas omettre ou violer un schéma de sortie publié.
- Le texte, l'image, l'audio, le lien vers les ressources et les blocs de ressources intégrés valident.
- Les événements d'audit en en-tête contiennent des noms et aucune valeur.
- ASCII visible simple reste simple; Unicode, contrôle, rembourrage, vide, et
  Les valeurs ressemblant à des sentinelles vont et viennent à travers le codage exact de base64 UTF-8.
- Les nombres entiers reflétés en dehors de la plage de sécurité JavaScript sont rejetés.
- Annotations dans `oneOf`- Je suis là .`items`, objets en nid, `$ref`définitions, ou
  les régimes de sortie sont rejetés lors de l'admission.
- Les noms d'en-tête reconnus insensibles au cas ne passent que lorsque la valeur décodée
  correspond exactement au corps; des copies manquantes ou non correspondantes produisent HTTP `400`
  et JSON-RPC `-32020`- Je suis désolé .
- L' analyse ne revient jamais .`production`- Je suis désolé .
- Une défaillance de l' outil utilise `isError: true`; une appel de protocole malformée utilise JSON-RPC `error`- Je suis désolé .

## Mode de défaillance de la production

| Failure | What the learner sees | Correct response |
|---------|-----------------------|------------------|
| Client assumes object output | Valid arrays fail or are silently wrapped | Validate against the published schema without object-only types |
| Empty cursor treated as false | Final pages disappear | Continue whenever `nextCursor` is present and non-null |
| Sensitive value mirrored | Secret appears in proxy, WAF, or trace data | Reject the descriptor and keep secrets in protected request data |
| Raw Unicode or whitespace mirrored | Gateway and origin disagree or the value is normalized | Use exact base64 UTF-8 sentinel encoding and compare after decoding |
| Annotation hidden in a schema branch | A client misses routing metadata during admission | Traverse the entire schema tree and allow only direct top-level properties |
| Large integer mirrored | JavaScript intermediary rounds the routing value | Reject values outside the JavaScript safe integer range |
| Header and body disagree | Gateway routes one target while the origin executes another | Reject before dispatch with HTTP `400` and JSON-RPC `-32020` |
| Output schema ignored | Downstream code consumes corrupt structure | Validate before model or application use |
| Resource link trusted automatically | Caller follows an unauthorized URI | Reauthorize every resource read |
| Completion shares global suggestions | Hidden tenant names leak | Filter by caller, reference, and authorization |
| Tool annotations treated as policy | Destructive operation bypasses confirmation | Enforce authorization and approval outside annotations |
| One malformed tool breaks discovery | Entire server becomes unavailable | Reject the bad descriptor and admit valid tools independently |

## Connexion Capstone

La phase 13 a besoin d'une passerelle qui peut fusionner des outils de plusieurs serveurs.

Utilisez l' artefact pour classer quatre pièces de preuves de pierre angulaire:

- une découverte déterministe et complètement paginée;
- la validation du descripteur avant l'exposition du modèle;
- la sortie structurée validée plus les blocs de contenu limités;
- compléter et router les métadonnées qui préservent les limites d'autorisation.

Ne prétendre pas à la compatibilité de la passerelle d' une solution réussie `tools/call`Capturez le descripteur, la trace de page, l'ensemble d'outils admis, l'ensemble d'outils rejetés et un résultat validé.

## Les termes clés

| Term | Meaning |
|------|---------|
| `inputSchema` | JSON Schema object defining accepted tool arguments |
| `outputSchema` | Optional JSON Schema defining `structuredContent` |
| `structuredContent` | Any JSON value produced by a tool result |
| Content block | Typed text, image, audio, resource link, or embedded resource |
| `x-mcp-header` | Schema annotation that mirrors a primitive argument into Streamable HTTP metadata |
| Opaque cursor | Server-issued pagination token whose value the client does not interpret |
| Completion reference | Prompt name or resource URI/template whose argument is being completed |
| Admission | Client decision to expose or reject a discovered descriptor |

## Pour en savoir plus

- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Completion](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion)
- [MCP Pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)
- [MCP Streamable HTTP Parameter Headers](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#custom-headers-from-tool-parameters)
