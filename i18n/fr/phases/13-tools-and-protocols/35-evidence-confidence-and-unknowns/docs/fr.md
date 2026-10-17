# Des preuves, une confiance et des inconnues dans l'analyse binaire

> Une découverte sur un programme compilé est un enregistrement avec un hash, une confiance, une autorité et une liste de ce qu'il ne peut pas montrer.

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~105 minutes

## Objectifs d'apprentissage

- Construire un dossier de preuves basé sur les données de réa `Evidence`avec la confiance observée, dérivée et déduite.
- Compute les identifiants de contenu-hash qui restent les mêmes lorsque le fichier passe à un autre dossier.
- Enregistrer ce que chaque fournisseur ne peut pas montrer comme des limites qui voyagent avec le dossier.
- Refuser d'indiquer un résultat jusqu'à ce qu'une fonction observée le supporte.
- Conserver les inconnus résiduels en tant qu'enregistrements révisés rejoints par une chaîne de hachage.
- Le fournisseur de cache répond sous une clé exacte et refuse une répétition obsolète.

## Le problème

Vous demandez à un agent ce que vous voulez .`grade`dans `ledger-x86_64-O1.o`Le réviseur demande comment l'agent sait. L'agent a un bloc de texte que Ghidra a imprimé. Pour le chemin par défaut, ce texte lit`uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);`Les octets à ce moment-là disent`xor eax, eax`- Je suis désolé .

L'expression et l'instruction produisent tous les deux 0 pour chaque entrée. L'agent ne savait pas cela. Il a fait confiance au texte parce qu'un outil l'a produit.[NDSS BAR 2026](https://www.ndss-symposium.org/wp-content/uploads/bar2026-58.pdf)Il a cité quatre faiblesses récurrentes: le biais de formation, la confiance excessive en les observations, les limites de contexte et la persistance des plans.[arXiv 2607.12507](https://arxiv.org/abs/2607.12507))

Réa 6.5.0 donne à l'agent un vocabulaire pour cela.`Evidence`enregistrement avec un `confidence`de `observed`- Je suis là .`derived`ou `inferred`, une`authority`, et une liste de `limitations`. La compétence de l'emballage dit à l'agent: "Chaque conclusion doit distinguer les observations, les inférences et les inconnus. Citez les identifiants de preuve, préservez les limites et la couverture incomplète, et ne jamais implique que l'analyse statique observé exécution".

Trois choses réal vous laissent. Ses instantanés cachent les résultats exacts de la requête, pas l'analyse. Sa vérification passe une revendication lorsqu'une dimension de comparaison est inchangée, et son propre commentaire dit qu'un passe " n'étale pas l'équivalence globale de la mise en œuvre. " Son registre d'obligation prend des candidats uniquement à partir de JavaScript et de graphiques gérés, de sorte qu'une fonction native n'atteint jamais.`verified`à travers elle.

Cette leçon construit la pièce manquante en Python simple. Vous écrivez un livre qui stocke tous les faits sur le dispositif comme un enregistrement avec un hash de contenu. Vous faites des prédictions sur`percent_change`et `grade`Vous importez deux enregistrements réels et l'instantané de laquelle ils sont venus, et votre grand livre recompte chaque hash réé écrit.

```figure
re-evidence-ledger
```

## Trois mots pour la confiance, cinq pour l'autorité

`confidence`est un enum avec trois valeurs, jamais un nombre. `authority`Il est important de noter que les résultats obtenus par le producteur ne sont pas suffisants pour que les producteurs ne fassent pas de`shipped-artifact`avec `observed`- Je suis désolé .

| Value | Meaning on the fixture | Records in this lesson |
|---|---|---|
| `observed` | A fact read from bytes or from a run | the ELF reader's section and symbol reads, the two Ghidra records, 17 host runs |
| `derived` | A fact computed from observed records | the switch table model, the division model, every unknown mutation |
| `inferred` | An expectation that no observation states yet | 18 predictions |
| `shipped-artifact` | The file on disk was the source | every read of `ledger-x86_64-O1.o` |
| `controlled-replay` | A run you controlled produced the fact | the host runs of `harness.c` |
| `analyst-inference` | Your own judgement | predictions and unknown mutations |
| `historical-reference` | An older source tree or document | none here |
| `external-service` | A service outside the file and the host | none here |

Deux confusions coûtent le plus de temps. Le premier traite le pseudocode comme une observation du comportement. C'est une observation de ce que Ghidra a imprimé. Le second traite une liste vide comme une preuve d'absence.`code/main.py`- Il encode les deux règles.

## Le récit

Un enregistrement contient exactement seize champs, et réa 6.5.0 rejette tout autre champ. `code/main.py`Le 16ème champ, qui est le même que le réel, est le même que le réel.`relationship_coverage`, indique la taille d'une liste complète d'appelants ou de appelés.`complete`- Je suis là .`partial`ou `unknown`ou `null`lorsque l'opération ne rapporte pas de relations d'appel.

| Field | What it holds |
| --- | --- |
| `evidence_id` | `ev_` and 64 hex digits, a hash of the content |
| `subject` | `name`, `digest.sha256`, `format`, `architecture`, `local_path`, or `null` |
| `provider` | `id`, `name`, `version` or `null` |
| `analysis_profile` | the provider's settings and their digest, or `null`, always present |
| `predicate_type` | a tag, `rea.analysis` by default |
| `operation`, `parameters` | the tool and its input |
| `raw_result`, `normalized_result` | provider output and the canonical result |
| `confidence`, `authority` | the enums above |
| `environment` | `id`, `platform`, `architecture`, `isolation`, or `null` |
| `limitations` | plain sentences |
| `relationship_coverage` | `status` `complete`, `partial`, or `unknown` with diagnostics, or `null` |
| `locations` | `address`, `address-range`, `artifact-path`, `file-offset`, `file-offset-range` |
| `evidence_links` | the `ev_` ids this record rests on |

C' est le vrai disque pour lequel réa a écrit .`grade`Le dossier complet est`code/fixtures/rea-evidence-grade.json`, 37 573 octets.

```json
{
  "evidence_id": "ev_67b6d3f992eaf49a2e16de084b6ebfdd9884e8c5c7f75e175b7bb20f4f748ce1",
  "subject": {
    "name": "ledger-x86_64-O1.o",
    "digest": { "sha256": "9a4a8082ba5e285a732552981a6935e0c5255725664f380e37c44ebfb04d803f" },
    "format": "elf",
    "architecture": "x86_64",
    "local_path": "/private/tmp/aiefs-re/phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns/code/fixtures/ledger-x86_64-O1.o"
  },
  "provider": { "id": "ghidra", "name": "Ghidra", "version": "12.1.4" },
  "analysis_profile": {
    "provider": { "id": "ghidra", "name": "Ghidra", "version": "12.1.4" },
    "parameters": { "jump_table_evidence": "typed-case-default-blocks-v1", "import_mode": "ephemeral-source-immutable" },
    "digest": "9c7f6b24be69294e61da046e4583e3856cb9e40433b149c1bf8bb534d2fe03ef"
  },
  "operation": "analyze_function",
  "parameters": { "procedure": "grade" },
  "confidence": "observed",
  "authority": "shipped-artifact",
  "environment": null,
  "limitations": [
    "Default-space addresses use lowercase 0x-prefixed hexadecimal; other address spaces use <percent-encoded-space>:0x<hex>.",
    "Resolved reference metadata identifies computed, indirect, external, call, jump, and data edges; unresolved targetless flows remain unknown, and function classifications distinguish thunks and externals."
  ],
  "relationship_coverage": {
    "status": "complete",
    "diagnostics": [
      "Only resolved Ghidra call references are returned; unresolved computed or indirect calls remain unknown, while function classifications distinguish thunks and externals."
    ]
  },
  "locations": [],
  "evidence_links": []
}
```

La vraie .`parameters`L'objet du profil a 25 clés.`limitations`Une limite fait partie du contenu, donc en la supprimant, l'identifiant est modifié, et votre grand livre rejetterait l'enregistrement.

Notez ce que le récit ne dit pas. `environment`est `null`Parce que Ghidra a lu un dossier et n'a rien trouvé.`locations`est vide car cette opération rapporte des adresses à l' intérieur `normalized_result`- Je suis là .`relationship_coverage`a le statut `complete`Le diagnostic indique que seuls les références d'appels Ghidra résolus sont renvoyées, et que les appels informatisés ou indirects non résolus restent inconnus.

## L'ID est un mélange de contenu, et le chemin n'est pas dans le même

réacteurs `evidence_id`comme `ev_`plus SHA-256 sur le JSON canonique RFC 8785 d'une projection du dossier.`digest`- Je suis là .`format`, et `architecture`et des gouttes .`name`et `local_path`Il tombe .`relationship_coverage`lorsque sa valeur est `null`, donc un enregistrement sans relations d'appel conserve la même identité que avant que le champ n'existe.`raw_result`et chaque limite est hashée.

RFC 8785 tries les clés d'objet, supprime l'espace blanc et fixe les formes de nombre et de chaîne. Pour les enregistrements dont les nombres sont entiers, la bibliothèque standard de Python produit les mêmes octets:

```python
def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
```

`code/main.py`Les deux enregistrements réels et l'instantané ne contiennent pas de floats, donc la recette reproduit chaque hash réel:

| rea value | Recomputed by `code/main.py` |
|---|---|
| `ev_67b6d3f992eaf49a…` for `grade` | equal |
| `ev_e88b10dc0c6400e5…` for `percent_change` | equal |
| profile digest `9c7f6b24be69294e…` over `{provider, parameters}` | equal |
| `query_26122aebe9d8abae…` for the `grade` snapshot entry | equal |
| `query_0f95ef38038d3666…` for the `percent_change` entry | equal |

Pourquoi exclure le chemin ? la même observation des mêmes octets obtient le même id sur n'importe quelle machine et dans n'importe quel dossier.`local_path`dans la liste ci-dessus, les noms d'un dossier sur un Mac. L' id ne change pas lorsque vous copiez le fichier à `/home/you/re`- Je suis là .`rebind_subject_path()`Il refuse quand on a un disque.`sha256`- Je suis là .`format`ou `architecture`réal 6.5.0 fait la même chose avec `rebindEvidenceToTargetPath`lorsqu'il reproduit un enregistrement d'impression instantanée pour une cible sur une nouvelle voie.

`validate_evidence()`Il rejette également un enregistrement dont le nom est identifié par un identifiant sémantique.`analysis_profile.provider`diffère de son `provider`- Le test .`test_tampered_id_and_mismatched_profile_are_rejected`modifie une ligne de pseudocode à l'intérieur de la vraie `grade`l'enregistrement et regarde le livre refuser.

## Observez vous- même l'objet

Avant tout fournisseur, le registre lit le dossier. `observe_object()`analyse les en-têtes de section ELF64, `.symtab`- Je suis là .`.strtab`, et `.rela.text`avec `struct.unpack_from`Il en écrit un .`inspect_object`enregistrement et un `read_function_bytes`enregistrement par symbole de fonction.`observed`avec autorité `shipped-artifact`, parce que la source était le fichier sur le disque.

| Record | Content | Id |
|---|---|---|
| `inspect_object` | five sections with offsets, sizes, and hashes, four functions, three relocations | `ev_645e0a145f28bd5c…` |
| `read_function_bytes percent_change` | 16 bytes at `.text+0x00` | `ev_2572a7d254ad15cb…` |
| `read_function_bytes grade` | 46 bytes at `.text+0x10`, SHA-256 `ffcb75c2a7bb1104…` | `ev_a467c2427bfc0c56…` |
| `read_function_bytes total` | 31 bytes at `.text+0x40` | `ev_b1e845cea2543073…` |
| `read_function_bytes grade_label` | 63 bytes at `.text+0x60` | `ev_c50041be26e6b7f4…` |

`llvm-objdump -t`est d'accord: `grade`à`0000000000000010`avec taille `2e`Les trois délocalisations en`.rela.text`sont`R_X86_64_PC32`contre `.rodata`à des compensations `0x36`- Je suis là .`0x84`, et `0x93`, avec des additions `0x10`- Je suis là .`0x24`, et `-0x4`Chaque disque contient un`file-offset-range`Localisation et une limitation: "Les octets de section sont lus à partir du fichier sur le disque. Le lecteur n'applique aucune relocalisation et n'exécute aucun code".

La leçon 32 construit le lecteur dans son intégralité. Voici 60 lignes, parce que cette leçon est sur ce que vous enregistrez sur la lecture.

## Dérivez la table et dites que vous l'avez dérivée

Les 46 octets de `grade`contiennent quatre modèles que la leçon 33 décode:

| Bytes | Instruction | Offset in `grade` |
|---|---|---|
| `48 69 c0 67 66 66 66` | `imul rax, rax, 0x66666667` | 3 |
| `48 c1 f9 22` | `sar rcx, 34` | 13 |
| `83 c1 fa` | `add ecx, -6` | 23 |
| `83 f9 04 77 0c` | `cmp ecx, 4` then `ja` | 28 |

Un multiplicateur par `0x66666667`Le dossier contient le numéro 10.`derive_grade_model()`écrit ce fait comme une limitation: "Le divisor 10 est une dérivation du multiplicateur 0x666667 et du changement 34. Le fichier contient le numéro 10. "

Le `add ecx, -6`Le premier cas est de 6.`cmp ecx, 4`avec un saut non signé admet cinq cas, 6 à 10.

Le `lea`à`.text+0x33`effectue le déménagement avec addend `0x10`. Un PC32 mesure la relocalisation à partir de la fin du déplacement de 4 bytes, de sorte que le tableau commence à `.rodata+0x14`Cinq mots de petite enjeu .`1, 2, 3, 4, 4`- Le commutateur .`ledger.c`Le compilateur a transformé le commutateur en une table de recherche des valeurs de retour.

Le dossier résultant `ev_605106bff2d9998f…`Il a`confidence` `derived`, et son `evidence_links`nommer les deux enregistrements observés sur lesquels il repose. supprimer l'enregistrement de octets du livre et l'enregistrement dérivé a un lien suspendu. La figure ci-dessus montre cet état lorsque vous désactivez`ev_a467c242…`- Je suis désolé .

Comparons avec Ghidra, son pseudocode pour les mêmes octets:

```c
uint grade(int param_1)
{
  uint uVar1;
  uint uVar2;

  uVar2 = param_1 / 10 - 6;
  uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);
  if (uVar2 < 5) {
    uVar1 = *(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4);
  }
  return uVar1;
}
```

Ghidra a appliqué le déménagement et nommé la table `DAT_001000b4`, qui est `.rodata+0x14`après avoir été placé `.text`à`0x100000`Il a récupéré la division par 10 et les cinq cas. Pour la valeur par défaut, il a imprimé une expression qui est toujours 0 et ne dit jamais 0.`native_value_flow`marque ces opérations `is_dead: true`Le récit est une fidèle observation de Ghidra.

Le `percent_change`Le dossier montre l'autre échec.`undefined1 [16] percent_change(int param_1,int param_2,ulong param_3)`La source dit:`int percent_change(int before, int after)`- Je suis là .`idiv`écrit `EDX:EAX`, et le chemin de zéro sort .`EDX`Le décompilateur a donc déduit un retour de 16 octets et un troisième paramètre.`native_api.return_type.confidence`pour `grade`est `"low"`- Traiter une signature récupérée comme une conclusion du fournisseur, et l'enregistrer de cette façon.

## Les prédictions sont des récits

Une prédiction est une affirmation sur un comportement qu'aucune observation n'a déclaré. `claim_suite()`Il en construit dix-huit à partir des modèles dérivés et de la source. `record_prediction()`stocke chacun comme un disque avec `confidence` `inferred`et `authority` `analyst-inference`, lié aux dossiers sur lesquels il repose.

| Claim | Predicted value | Rests on |
|---|---|---|
| `percent_change(12400, 15500)` | 25, from 310000 / 12400 | division model `ev_076a9167…` |
| `percent_change(0, 5)` | 0, the zero guard | division model |
| `percent_change(100, 50)` | -50 | division model |
| `percent_change(-7, 3)` | -142, because `idiv` truncates toward zero | division model |
| `grade(55)` | 0, index -1 fails the unsigned compare | table model `ev_605106bf…` |
| `grade(60)`, `grade(69)` | 1, 1 | table model |
| `grade(70)`, `grade(85)` | 2, 3 | table model |
| `grade(95)`, `grade(100)`, `grade(105)` | 4, 4, 4 | table model |
| `grade(110)`, `grade(-5)` | 0, 0 | table model |
| `total(1, 2, 3)`, `total()` | 6, 0 | source record `ev_2f1e6d25…` |
| `grade_label(85)` | `"B"` | source record and table model |
| `grade(score)` for 90 to 100 | 4 for all eleven inputs | table model |

`predict()`utilise la sémantique C, pas la sémantique Python. `c_div()`se réduit à zéro, et `wrap32()`Le code de Python est de 32 bits.`//`Ronde vers l'infini négatif, donc `(3 - -7) * 100 // -7`La leçon 36 rend cette grille beaucoup plus grande.

Chaque prédiction ouvre un résidu inconnu.`observed`preuve avec autorité `controlled-replay`et recommande une sonde: `execute_function`La question pour la demande de première classe est "La note ((95) == 4 se tient-elle lorsque la fonction compilée est exécutée?"

## Vérifiez leur équivalence

`code/fixtures/harness.c`inclut `ledger.c`et imprime un résultat par invocation. `observe_runs()`compilée avec la première de `clang`- Je suis là .`cc`ou `gcc`chaque entrée dans la grille une fois. chaque course devient un record:

```json
{
  "operation": "execute_function",
  "parameters": { "function": "grade", "args": [95] },
  "normalized_result": { "result": 4, "stdout": "4\n", "exit_code": 0 },
  "confidence": "observed",
  "authority": "controlled-replay",
  "environment": { "id": "host-process", "platform": "darwin", "architecture": "arm64", "isolation": "process" },
  "limitations": [
    "The run executed a host build of ledger.c, not the x86-64 object. The claim about the object rests on the shared source and on the compiler.",
    "One run observes one input. It proves nothing about other inputs."
  ]
}
```

Le sujet d'un enregistrement d'exécution est le fichier binaire de harnais, un fichier Mach-O arm64 sur ce Mac, avec son propre SHA-256.`course-harness`avec la ligne de version du compilateur, ici `Apple clang version 17.0.0 (clang-1700.6.3.2)`- Le .`evidence_links`nommer le dossier source `ev_2f1e6d251543d017…`, parce que la course et l' objet partagent`ledger.c`Je ne sais pas.

Lire la première limitation deux fois. L'objet x86-64 n'a jamais fonctionné sur cette machine. Ce qui a fonctionné était une deuxième construction de la même source pour une autre architecture. C'est toujours une observation, et c'est la plus forte disponible sans émulateur, donc la revendication passe avec cette phrase attachée. Un rapport qui l'omit a perdu les informations dont le lecteur a besoin.

Les 17 entrées uniques sont toutes en accord avec les prédictions.`code/fixtures/run-evidence.json`détient ces 17 enregistrements et le registre source, exporté avec `--record`En 2026-10-13. Quand aucun compilateur n'est présent,`build_ledger()`Il est vrai que les pays de l'Europe centrale et orientale ont déjà fait preuve de la même volonté.`runs`le champ.

## Règles de promotion

Une réclamation passe de la déduction à l'appui uniquement par des démarches observées. `check_claim()`Il cherche`execute_function`enregistrer avec la même fonction et les mêmes arguments, comparer chaque résultat avec la prédiction et compter les entrées couvertes:

| Status | Condition | rea's completion ledger |
|---|---|---|
| `pass` | every input has an agreeing run and none disagrees | `pass` |
| `fail` | any run disagrees | `fail` |
| `unknown` | some inputs have runs, others have none | `unknown` |
| `unsupported` | no input has a run | `unsupported` |

`promote_claim()`C'est la forme stricte.`ClaimError`quand un identifiant est `inferred`ou `derived`. Il se lève lorsqu'un id nomme une suite d'une autre fonction ou d'autres arguments. Il se lève lorsque la liste est vide et lorsque les files d'attente couvrent moins d'entrées que la revendication.`grade(55)`, et aucun ne peut le promouvoir.`ev_7be90c0c…`dans la figure et voir la réclamation tomber à `unsupported`Alors que deux enregistrements sur `grade`Restez ici.

La demande de portée montre le cas partiel. Seuls deux de ses onze entrées, 95 et 100, apparaissent dans la grille de fonctionnement. `check_claim()`Retour `unknown`avec `covered`2 sur 11, et `promote_claim()`Les données de référence sont les données de référence de l'ensemble des données de référence.

`completion_ledger()`Les miroirs de réa `EvidenceCompletionLedger`Son numéro d' identification est`ecl_`plus le hash de ses dossiers, et `complete`est vrai seulement lorsque `pass`égale `total`Les empreintes de démonstration .`ecl_da64933ea13befed…`avec `complete: false`, à cause de cette revendication de rang unique.

## Les inconnus sont des enregistrements avec une chaîne de hachage

réa stocke un résidu inconnu comme un enregistrement strict. `code/main.py`Il conserve les mêmes champs et les mêmes invariants.

L' identité est `unk_`plus SHA-256 sur`{domain, question, scope_digest}`Deux analystes qui posent la même question sur le même fichier obtiennent la même identité.`scope_digest`est le SHA-256 de l'objet.

Chaque changement est une nouvelle révision.`previous_revision_digest` `null`. Chaque révision ultérieure nomme le `revision_digest`de celui qui l'a précédé, et le digeste couvre tout le champ sauf lui-même. `verify_unknown_chain()`Il vérifie que chaque lien pointe vers la révision précédente. Il vérifie que les preuves de mutation de chaque nom de révision sont dans le grand livre.

Chaque mutation est elle-même un enregistrement de preuve, comme le fait réellement le fournisseur `rea-unknown-registry`- Nos porteurs .`confidence` `derived`- Je suis là .`authority` `analyst-inference`La limite est: " Une mutation du registre enregistre ce que l'analyste a décidé.

C'est pourquoi les 81 enregistrements de la démo comprennent 36 enregistrements de mutations dérivées: 18 ouvertures et 18 mises à jour.

Trois invariants sont appliqués à chaque révision:

- - Je ne suis pas là .`resolved`Un inconnu porte un `resolution`, et seule la révision 1 n'a pas de prédécesseur.
- Une .`contradicted`noms inconnus au moins un identifiant contradictoire et aucun identifiant n'apparaît dans les deux listes.
- Une .`verified`la résolution nomme au moins un identifiant de preuve avec la confiance et l'autorité requises.

La troisième règle est celle où le registre refuse le plus souvent. Les notes de flux de travail de rea disent "Seules les preuves observées qualifiantes peuvent vérifier une résolution". `resolve_unknown()`rejette donc le modèle de tableau, rejette un enregistrement observé avec autorité `shipped-artifact`, et accepte le record de course `ev_f4ffbe8f922ec7fe…`. La révision résolue pour `grade(95)`est la révision 2, et son `previous_revision_digest`est `ffca840b55a8a188…`, le résumé de la révision 1.

`update_unknown()`Il prend`expected_revision`et augmente `revision-conflict`Quand il est démodé, comme ré's `update_unknown`La réaction correcte est celle que donnent les notes de Rea: relire la révision actuelle et décider à nouveau.

Un inconnu reste ouvert après la démo:`unk_f2eb0d7f84c80a19…`, "Actuelle la note ((score) == 4 pour chaque score de 90 à 100 tenir quand la fonction compilée fonctionne ?" Son statut est `investigating`, ses identifiants de soutien comprennent les deux courses, et sa sonde nomme toujours les neuf entrées sans course.

## Un cache de réponses n'est pas une base de données d'analyse

réa a trois choses qui ressemblent à un cache.`-readOnly -deleteProject`et supprimé à la fin.`AnalysisSnapshotCache`contient jusqu'à 10 000 entrées pour une identité binaire.`target`- Je suis là .`binding`- Je suis là .`entries`- Je suis là .`workflow_entries`, et une `evidence_bundle`- Je suis désolé .

La clé cache est `query_`plus SHA-256 sur`{target, binding, operation, parameters}`- Je suis là .`target`est `{sha256, kind, format, architecture}`, et une `raw-binary`La cible a aussi`entry_point`- Je suis là .`binding`Le fournisseur et l'ensemble du profil d'analyse, digest inclus.`jump_table_evidence`de `typed-case-default-blocks-v1`à une `v2`, et la digestion change, donc chaque clé change.

`code/fixtures/rea-snapshot.json`est le fichier réé écrit pendant les deux exécutions, 142.690 octets. réé l'a écrit avec des autorisations réservées au propriétaire, et la copie engagée est lisible par tout le monde. Il a deux entrées et un paquet de preuves avec les mêmes deux enregistrements. `AnswerCache.import_snapshot()`recompte chacun `query_id`Il refuse une photo de l'un de ses fichiers.`target`ou `binding`diffère de l'actif, avec la formulation de réa `profile_mismatch`Il refuse une photo de qui`format_version`- Ce n' est pas vrai .`course-answer-cache-1`- Je suis désolé .

Trois mesures de cette machine.`grade`La requête a pris 8,6 secondes, avec une importation de Ghidra fraîche.`percent_change`la requête contre le même instantané a pris 8,5 secondes, parce qu'une requête différente est une cache manquée.`grade`la requête a pris 2,1 s et a rendu le même enregistrement avec le même `evidence_id`. La répétition en mémoire en réa ajoute la limite " Chargé à partir d'un instantané d'analyse locale de réa; cet appel n'a pas réexécuté le fournisseur. " Notre `lookup()`ajoute une phrase du même genre, afin que le lecteur puisse distinguer une répétition d'une nouvelle réponse.

Une mutation met fin au cache. Dans rea 6.5.0, un renom ou un commentaire efface le cache d'instantané avant que rea envoie l'appel au fournisseur. Une réponse ratée ne peut pas prouver que la rédaction a été réouverte, donc une rédaction ratée arrête également la répétition.

## Pourquoi un agent a besoin de ça ?

Un agent qui conduit réa par MCP voit le dossier complet de chaque appel et peut citer`evidence_id`La recherche de 2026 indique où cela va mal. La confiance excessive en la sortie des outils était l'une des quatre faiblesses de l'étude NDSS BAR. Dans l'étude de texte planté, montrant le contenu binaire comme données seules, 15 propositions dangereuses de 40 étaient toujours laissées, et les vérifications de provenance avec l'autorisation de l'outil les ont retirées.[arXiv 2606.06838](https://arxiv.org/abs/2606.06838)Dans les binaires avec des centaines de fonctions, seulement 10 à 15% ont reçu une analyse réelle avant que l'agent ne produise des éclats.

Le registre répond à chacun de ces questions avec un enregistrement, et l'agent ne rapporte que ce que le registre contient.`unsupported`Une réclamation avec quelques courses est`unknown`Un texte du fournisseur est `observed`Donnez à l'agent le dossier des compétences.`outputs/`et faire du livre le seul endroit où il écrit des conclusions, qui s'applique à Claude, Cursor, Codex, OpenClaw, Hermes, ou à tout agent qui lit SKILL.md.

## Faites-le

Exécutez la démo de la bibliothèque standard:

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/main.py
```

La démo lit l'objet, importe les deux enregistrements réels, vérifie l'instantané réel et dérive les deux modèles.

```json
{
  "records": 81,
  "by_confidence": { "observed": 25, "derived": 38, "inferred": 18 },
  "rea_snapshot": { "entries": 2, "replay_hit": true, "stale_profile_refused": true },
  "grade_table": [1, 2, 3, 4, 4],
  "runs": "live: Apple clang version 17.0.0 (clang-1700.6.3.2)",
  "claims": [ { "claim_id": "grade(95)", "status": "pass", "covered": "1/1" }, { "claim_id": "grade(90..100)", "status": "unknown", "covered": "2/11" } ],
  "completion_ledger_id": "ecl_da64933ea13befed9483f8b00447bb3f4ceb74324c01e2515eff03f3af3ca750",
  "complete": false,
  "unknowns": { "resolved": 17, "investigating": 1 }
}
```

Sans un compilateur C,`runs`Il lit`imported 17 recorded run records from run-evidence.json`et le reste est le même.

Lire `code/main.py`dans cet ordre:

1. `canonical_json()`et `compute_evidence_id()`Gardez la recette de hash et la projection sémantique.
2. `create_evidence()`et `validate_evidence()`construire et vérifier un dossier.
3. `EvidenceLedger.record()`et `import_bundle()`conserver des dossiers et rejeter les conflits.
4. `open_unknown()`- Je suis là .`update_unknown()`, et `resolve_unknown()`maintenir la chaîne de révision.
5. `observe_object()`Lire l'ELF et `derive_grade_model()`Il lit la table.
6. `claim_suite()`- Je suis là .`predict()`, et `record_prediction()`écrire le niveau déduit.
7. `observe_runs()`chaque harnais est un enregistrement observé.
8. `check_claim()`- Je suis là .`promote_claim()`, et `completion_ledger()`appliquer les règles de promotion.
9. `AnswerCache`Les clés répondent comme les instantanés de Rea et refusent les anciens.

## Utilisez-le

Gardez un registre par cible et par enquête. Mettez le SHA-256 de la cible dans chaque sujet et le champ de la question dans chaque inconnu.`provider`Lorsque vous reconstruisez le fichier, le sujet est digéré et chaque ancien enregistrement devient une preuve d'un autre fichier.

Écrivez des limites au moment où vous les connaissez. Le lecteur qui n'applique aucune relocalisation dit cela. La course d'une construction hôte dit cela. Le modèle qui infère un diviseur dit cela. Un enregistrement écrit sans ses limites semble plus fort que les faits ne le permettent, et un lecteur ultérieur ne peut pas réparer cela.

Les chaînes et les pseudocodes de la cible sont des textes que la cible contrôle.`_meta.provenance.untrusted_content: true`Les instructions du serveur indiquent au modèle de traiter le contenu cible comme des données et de signaler toute demande d'exécution, d'installation, de téléchargement ou d'envoi de quelque chose comme une tentative d'injection rapide.

## Laboratoire interactif

### Laboratoire A: recomputer une réaction

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns/code
python3 -q
```

Je vais courir .

```python
from main import *
record = load_evidence_file(FIXTURES / "rea-evidence-grade.json")
record["evidence_id"] == compute_evidence_id(record)
record["subject"]["local_path"] = "/home/you/re/ledger-x86_64-O1.o"
record["evidence_id"] == compute_evidence_id(record)
record["limitations"].pop()
record["evidence_id"] == compute_evidence_id(record)
```

Les deux premières comparaisons sont vraies. Le chemin n'est pas dans le hash. Le troisième est faux, parce qu'une limitation est contenue.

### Laboratoire B: refus d'être promu

```python
ledger, report = build_ledger(live_runs=False)
claim = [c for c in claim_suite(ledger.find(operation="derive_switch_model")[0], ledger.find(operation="derive_arithmetic_model")[0], ledger.find(operation="read_source")[0]) if c.claim_id == "grade(55)"][0]
promote_claim(ledger, claim, [ledger.find(operation="derive_switch_model")[0]["evidence_id"]])
```

L' appel s' élève `ClaimError`et nomme l'identifiant dérivé.

```python
run = run_records_for(ledger, "grade", [55])[0]
promote_claim(ledger, claim, [run["evidence_id"]])["status"]
```

Le statut est `pass`, et les retournés `limitations`porter la sentence de construction d'hôte.

### Laboratoire C: refuser une photo dépassée

```python
import json
snapshot = json.loads(REA_SNAPSHOT_PATH.read_text())
target, binding = rea_snapshot_binding(snapshot)
AnswerCache(target, binding).import_snapshot(snapshot)
stale = dict(binding["analysis_profile"]["parameters"], jump_table_evidence="typed-case-default-blocks-v2")
AnswerCache(target, {"provider": binding["provider"], "analysis_profile": make_profile(binding["provider"], stale)}).import_snapshot(snapshot)
```

Le premier rapport d'importation 2. Le second rapport augmente `CacheError`avec `profile_mismatch`Une chaîne de version a changé, et l'instant-shot entier est refusé.

### Laboratoire D: casser la chaîne

```python
unknown_id = [u for u, r in ledger.unknowns.items() if r[-1]["status"] == "investigating"][0]
ledger.verify_unknown_chain(unknown_id)
ledger.unknowns[unknown_id][0]["severity"] = "critical"
ledger.verify_unknown_chain(unknown_id)
```

Le premier appel revient, le second monte.`EvidenceIntegrityError`, parce que la révision 1 ne correspond plus à sa propre digestion et la révision 2 fait référence à une digestion qui n'existe plus.

## Laboratoire de pratique

Fermez l'ouverture inconnue et ajoutez un deuxième fournisseur.

Les exigences:

- Extension `run_grid()`alors les 9 entrées manquantes de la demande de plage sont exécutées, puis confirmez les mouvements inconnus à `resolved`à la révision 3.
- Ajouter `llvm-objdump -d`en tant que deuxième fournisseur avec sa propre`provider`l' objet et enregistrer sa liste de `grade`comme `observed`- Je suis désolé .
- Écrivez une`derived`enregistrement qui indique les deux listes, Ghidra et llvm-objdump, décode les mêmes 14 instructions, et relier les deux.
- Ouvrez un inconnu pour `grade_label(105)`avec gravité `high`, puisque l' indice du tableau 4 est `labels[4]`, et résoudre avec une course.
- Exporter le livre avec `export_bundle()`Il faut l'importer dans un nouveau registre et prouver que chaque chaîne est toujours vérifiée.

Le succès signifie un rapport dans lequel chaque phrase cite un id du paquet et celui de l' un.`unknown`La réclamation est partie.

## Leçon d'artisanat

Cette leçon produit `outputs/skill-evidence-ledger.md`Il indique à l'agent comment écrire chaque découverte en tant qu'enregistrement avec confiance, autorité, limitations et liens.`pass`- Je suis là .`fail`- Je suis là .`unknown`ou `unsupported`- Je suis désolé .

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Les 28 tests prouvent:

- réalistes`evidence_id`Les valeurs, le profil et les deux instantanés `query_id`Les valeurs sont recomputées à partir du contenu
- un chemin modifié garde l' id et une limite modifiée le modifie
- un enregistrement corrompu, un fournisseur de profil non correspondant, un flot et une valeur enum inconnue sont rejetés
- une copie des dossiers `duplicate`, un record contradictoire monte, et une importation de paquets est atomique
- les importations enregistrées en série avec 18 enregistrements et aucun lien suspendu
- Un inconnu s' ouvre à la révision 1, refuse une mise en mode `expected_revision`, et se résolve seulement avec observé `controlled-replay`preuve
- une révision modifiée rompt la chaîne
- La promotion refuse les preuves déduites, dérivées, vides et non liées, et passe avec un accord
- couverture partielle est `unknown`et une course en désaccord est `fail`
- le cache clique sur la clé exacte et manque sur d'autres paramètres
- le cache refuse une version de profil accidentée, une cible erronée, une version de format erroné et une clé altérée
- Le lecteur de l' ELF est d' accord avec `llvm-objdump`sur les compensations, les tailles, les octets et les délocalisations, et le tableau se lit `1, 2, 3, 4, 4`
- Les prédictions utilisent la division C, et les courses enregistrées sont d'accord avec les 17 revendications de seule entrée
- Avec un compilateur présent, les émissions en direct sont également d'accord

## Mode de défaillance de la production

| Failure | What the weak report says | What the ledger requires |
|---|---|---|
| Pseudocode read as behaviour | "grade returns 0 for failing scores" | an `execute_function` record for that input, or `unsupported` |
| Decompiler signature trusted | "percent_change returns 16 bytes" | the signature marked as the provider's inference with its `low` confidence |
| Empty list read as absence | "the function has no callers" | an `unknown` status, or a coverage field when the tool provides one |
| Path hashed into the id | "the record changed" after a copy | the semantic projection without `name` and `local_path` |
| Limitation dropped from a copy | a record that looks stronger than the source | id mismatch and rejection on import |
| Partial grid rounded up | "all scores from 90 pass" | `unknown` with `covered` 2 of 11 |
| Host run presented as the object's run | "the x86-64 object returned 25" | the host-build limitation carried on the claim |
| Unknown closed by a sentence | "resolved, see above" | a `verified` resolution with observed ids of the required authority |
| Stale revision overwritten | two analysts' edits collide | `revision-conflict` and a reread |
| Old snapshot replayed after a release | answers from a changed contract | `profile_mismatch` on import |
| Cache kept after a rename | a replay that hides a metadata write | invalidation before dispatch and a failed export |

## Règlement d'exploitation

Enregistrez ce que vous avez lu, ce que vous avez calculé et ce que vous attendez comme trois types d'enregistrements avec trois mots. Fermez un inconnu avec un id, jamais avec une phrase. Rejouez une réponse seulement sous la clé exacte avec laquelle elle a été stockée, et refuser quand une chaîne de version diffère.

## Pour en savoir plus

- [rea 6.5.0 source, `src/domain/evidence.ts` and `src/domain/residualUnknown.ts`](https://github.com/morluto/rea)
- [RFC 8785, JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
- [Ghidra 12.1.4 release](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Agent Failure Patterns in CTF Reverse Engineering, NDSS BAR 2026](https://www.ndss-symposium.org/wp-content/uploads/bar2026-58.pdf)
- [When Binaries Talk Back: Representation-Confusion Attacks on LLM-Assisted Reverse Engineering, arXiv 2607.12507](https://arxiv.org/abs/2607.12507)
- [Agentic decompilation study, arXiv 2606.06838](https://arxiv.org/abs/2606.06838)
- [When LLM Decompilers Recompile More and Preserve Less, arXiv 2609.05370](https://arxiv.org/abs/2609.05370)
- réa 6.5.0 source à la balise `3b79af2`que cette leçon décrit: `src/domain/evidence.ts`- Je suis là .`.agents/skills/reverse-engineer-anything/SKILL.md`- Je suis là .`docs/cli.md`- Je suis là .`src/domain/reconstructionVerification.ts`- Je suis là .`src/application/ReconstructionObligationCandidates.ts`- Je suis là .`docs/tool-design.md`- Je suis là .`src/domain/evidenceCompletionLedger.ts`- Je suis là .`src/domain/residualUnknown.ts`- Je suis là .`src/application/investigation/UnknownEvidence.ts`- Je suis là .`.agents/skills/reverse-engineer-anything/references/evidence-workflows.md`- Je suis là .`src/application/binary/AnalysisSnapshotCache.ts`- Je suis là .`src/domain/analysisSnapshot.ts`- Je suis là .`src/application/binary/BinarySession.ts`- Je suis là .`src/application/binary/BinarySessionRecords.ts`- Je suis là .`docs/mcp-contracts.md`- Je suis désolé .
