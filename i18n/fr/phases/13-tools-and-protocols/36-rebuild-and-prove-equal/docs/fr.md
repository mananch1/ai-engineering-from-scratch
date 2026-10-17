# Reconstruire et prouver l'égalité: tests différentiels, comparaison en octets et boucle de preuves

> Passer vos tests prouve votre accord sur vos tests, donc l'égalité a besoin d'une grille qui atteint chaque bord et octets qui correspond après le masquage de relocalisation.

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 34 (lifting to pseudocode), Phase 13 · 35 (evidence, confidence, and unknowns)
**Time:** ~110 minutes

## Objectifs d'apprentissage

- Écrivez une reconstruction de la liste des instructions et nommez le dossier de preuve d'où elle provient.
- Construisez une grille d'entrée avec une classe par danger: zéro garde, valeurs négatives, division de tronc, débordement de 32 bits et piège de division.
- Exécutez l'original et la reconstruction en un programme et enregistrez chaque cellule, y compris un piège.
- Récupérez la reconstruction avec les drapeaux originaux et comparez les octets de fonction après avoir masqué les champs de relocalisation.
- Lisez les constantes à lesquelles une relocation pointe et comparez-les, de sorte qu'un faux littéral ne peut pas passer.
- Conduisez le serveur MCP de Rea depuis un petit client de studio et enregistrez chaque étape avec son identifiant de preuve.
- Mente un dossier de preuve fermé par défaut dont l'identification est un hash sur ses allégations et les liens de preuve.

## Le problème

Vous avez demandé à Rea le dossier de fonctionnement de `percent_change`dans notre `ledger-x86_64-O1.o`Ghidra 12.1.4 a renvoyé un pseudocode, et l'agent a écrit ce C dessus:

```c
int percent_change(int before, int after) {
    if (before == 0) return 0;
    return (int)(((long)after - before) * 100 / before);
}
```

Quatre tests passés:`(100, 150)`donne 50, `(200, 100)`donne -50, `(0, 5)`donne 0, `(50, 100)`L'agent a aussi été reconstruit.`grade`comme quatre `if`Les tests sont passés, six autres tests ont été confirmés.

Ce rapport ne répond à aucune de ces questions:

- Qu' est-ce que le texte original revient quand `(after - before) * 100`ne peut pas être utilisé dans 32 bits ?
- Qu'est-ce que le résultat original est de 110?
- La reconstruction se compose-t-elle sur les mêmes 16 octets que l'original ?
- Lorsque deux fonctions compilées ont les mêmes octets, leurs délocalisations pointent-elles vers les mêmes constantes?
- De quel dossier provient la reconstruction, et l'agent a-t-il lu l'annonce ou le pseudocode ?
- Quel est le statut de la preuve sur un hôte sans compilateur C ?

Chaque cible de cette leçon est un programme que nous avons écrit et compilé.

```figure
re-36-prove-equal
```

## Ce que la preuve dit

La boucle d' agent est en train de bouger .`code/main.py`J'ai fait six appels à réa 6.5.0 sur le studio.`code/fixtures/runs/agent-run.json`Le serveur a répondu .`initialize`avec la version du protocole `2025-11-25`Trois appels ont rapporté des preuves:

| Step | Tool | Time | Evidence id |
|---|---|---|---|
| 2 | `analyze_function percent_change` | 6.31 s | `ev_e88b10dc0c6400e51bb6f7fdc3eaea8d2c97cc02463f0db3e6afeefc273197cd` |
| 3 | `analyze_function grade` | 0.02 s | `ev_67b6d3f992eaf49a2e16de084b6ebfdd9884e8c5c7f75e175b7bb20f4f748ce1` |
| 4 | `read_bytes 0x1000b4, 20` | 0.00 s | `ev_c26146cfedf4c1ee146d8ff00cfa9a5d3f5335bdd9c97ccf4e1af5f6ea3ed6bc` |

Le premier appel a été payé pour l'importation et l'analyse automatiques de Ghidra.`confidence: "observed"`- Je suis là .`authority: "shipped-artifact"`, et le SHA-256 de l'objet,`9a4a8082ba5e285a732552981a6935e0c5255725664f380e37c44ebfb04d803f`Ghidra charge l' objet relocalisable avec `.text`à`0x100000`Je suis un homme .`percent_change`Il est assis à`0x100000`et `grade`à`0x100010`- Je suis désolé .

Le pseudocode de `percent_change`est où le `long`provenant de:

```c
undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)
{
  long lVar1;
  undefined1 auVar3 [16];
  if (param_1 != 0) {
    lVar1 = (long)((param_2 - param_1) * 100);
    auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
    auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
    return auVar3;
  }
  ...
}
```

Le décompilateur a vu .`cltd`et `idivl`écrire les deux `edx`et `eax`et a modélisé une valeur de retour de 16 octets avec une division de 64 bits.

```
0x100000: TEST EDI, EDI
0x100002: JZ 0x0010000d
0x100004: SUB ESI, EDI
0x100006: IMUL EAX, ESI, 0x64
0x100009: CDQ
0x10000a: IDIV EDI
0x10000c: RET
0x10000d: XOR EAX, EAX
0x10000f: RET
```

Chaque opération est de 32 bits de large.`IMUL EAX, ESI, 0x64`multiplié par 32 bits et conserve la moitié basse. `CDQ`- les signes s' étendent`eax`dans `edx`, et `IDIV EDI`une réparation qui suit la liste est `100 * (after - before) / before`dans `int`- Je suis désolé .

Le `grade`Le pseudocode nomme une table,`*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`, et une expression par défaut illisible construite à partir de `0x66666667`La liste montre la division par 10 comme `IMUL RAX, RAX, 0x66666667`- Je suis là .`SAR RCX, 0x22`- Je suis là .`SHR RAX, 0x3f`- Je suis là .`ADD ECX, EAX`- Alors ...`ADD ECX, -0x6`- Je suis là .`CMP ECX, 0x4`, et `JA`Gardez des dizaines de 6 à 10.`XOR EAX, EAX`avant la comparaison est la valeur par défaut de 0. `LEA RCX, [0x1000b4]`et `MOV EAX, dword ptr [RCX + RAX*0x4]`lire une table de cinq entrées de 4 bytes.

Étape 4 lire ces 20 octets: `01000000 02000000 03000000 04000000 04000000`La table est là .`1, 2, 3, 4, 4`Pour dix, 6, 7, 8, 9, 10.`switch`dans `ledger.c`, avec `case 10`et `case 9`partage de la valeur 4.

Les deux preuves figurant dans les pages de preuve de réa ont commencé à partir de la liste et de `read_bytes`Le premier pseudocode de DX-Ball était`FUN_00406400(void)`Il n'y a pas d'entrée de pile, et la vue des instructions l'a corrigé.

## Écrivez les candidats

Le dossier `code/fixtures/candidates/`Il contient huit reconstructions, une fonction par fichier. Quatre suivent la liste ou compiler au même code. Quatre sont plausibles et erronés, ou erronés dans la façon qu'un test écrit à la main trouve.

| File | Source | What it does |
|---|---|---|
| `percent_change_mul_first.c` | `100 * (after - before) / before` | the listing in C |
| `percent_change_split.c` | `(after * 100 - before * 100) / before` | different text, same bytes after clang folds it |
| `percent_change_wide.c` | `(int)(((long)after - before) * 100 / before)` | the pseudocode's `long`, divides before it wraps |
| `percent_change_div_first.c` | `(after - before) / before * 100` | divides first, so rounding differs |
| `grade_switch.c` | `switch (score / 10)` with the five cases | the original source |
| `grade_table.c` | `unsigned slot = score / 10 - 6; slot < 5 ? points[slot] : 0` | the listing in C |
| `grade_thresholds.c` | `if (score >= 90) return 4; ...` | no upper bound on the top band |
| `grade_case10_wrong.c` | `case 10: return 3;` | one table entry wrong |

Chaque fichier est compilé avec la commande de fixation, `clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c`Les objets compilés sont commis à côté des sources de sorte que les tests se déroulent sur un hôte sans clang.

## 1ère étape: Exécuter et comparer

La définition de McKeeman de 1998 des tests différentiels nécessite "deux systèmes comparables ou plus" alimentés par "cas de test générés mécaniquement", et traite un résultat différent, un accrochage ou un crash comme un candidat à un bug. Il a nommé la partie difficile: " Le plus gros problème de l'essai est d'évaluer le résultat d'un test. " La fonction originale est l'oracle ici. Nous n'avons pas besoin de connaître la bonne réponse. On a besoin de l'original et de la reconstruction pour ne pas être en désaccord nulle part sur une grille qui atteint tous les dangers.

### Conception du réseau

`input_grid()`dans `code/main.py`Il construit 92 cellules.`percent_change`Il faut chaque paire de dix .`before`Les valeurs et sept `after`les valeurs:

```
before: 0, 1, -1, 3, -3, 7, 100, -100, 2147483647, -2147483648
after:  0, 5, -5, 105, 536870911, 2147483647, -2147483648
```

Pour `grade`Il faut 22 points: les deux côtés de chaque bord de bande, négatifs, 0, 110, et les deux `int`Les limites.`classify()`le surcoulant a 32 cellules, négative 17, limite 14, plaine 12 et garde zéro 7. le troncage négatif a 5, arrondissement 2, au-delà du commutateur 2, et piège 1.

Chaque classe existe pour une raison.`TEST EDI, EDI`Les cellules négatives vérifient la division troncante:`idiv`ronde vers zéro, donc `-9 / 10`est de 0 et `-10 / 10`est -1. Les cellules rotatives vérifient que le multiplicateur se produit avant la division, donc `(105 - 100) * 100 / 100`Les cellules de surcoût produisent`(after - before) * 100`La cellule de piège unique rend le produit enveloppé égal`INT_MIN`avec `before`égal à -1.

Yang, Chen, Eide et Regehr ont construit Csmith pour générer des programmes sans comportement indéfini. Un désaccord oracle sur un comportement indéfini est un bruit et jamais un rapport de bug. Le débordement signé est indéfini en C. Notre oracle est un objet compilé avec un comportement fixe, donc les cellules de débordement sont dans la portée.

### Le conducteur et le piège

`code/driver.c`Lire une cellule par ligne de stdin et imprime les entrées, le résultat original et le résultat de reconstruction. `build_harness()`compilera le candidat avec `-Dpercent_change=cand_percent_change -Dgrade=cand_grade`Dans un hôte x86-64 Linux, l'original est l'objet fixe lui-même.`ledger.c`Les rapports ont été compilés de manière native et les dossiers du rapport ont été enregistrés `original_kind: fixture-source`- Je suis désolé .

Le conducteur installe un `SIGFPE`avec `sigsetjmp`et `siglongjmp`Pour ...`before = -1`et `after = 536870911`, la différence est 536870912 et le produit 32 bits se termine à -2147483648.`idiv`Le quotient ne s'inscrit pas dans les 32 bits (Intel SDM Volume 2, IDIV).`trap`- Je suis désolé .

Sur l'Arche64, `sdiv`Le récit indique lequel a été exécuté.

### Résultats

`prove_behaviour(PLAUSIBLE)`construit le `long`Les dix cas naïfs passent, la grille échoue sur 28 cellules.

| Function | Inputs | Class | Original | Rebuild |
|---|---|---|---|---|
| `percent_change` | 3, 2147483647 | overflow | -133 | -1431655899 |
| `percent_change` | 100, 2147483647 | overflow | -101 | 2147483547 |
| `percent_change` | 2147483647, 5 | overflow | 0 | -99 |
| `grade` | 110 | beyond the switch | 0 | 4 |
| `grade` | 2147483647 | beyond the switch | 0 | 4 |

Toutes les 26 `percent_change`Les défaillances sont des cellules de débordement avec `|before| > 1`Pour ...`before`Le coefficient de 64 bits est réduit à la valeur enveloppée, de sorte que ces cellules s'accordent par accident.`grade`échec parce que l'interrupteur original n'a pas de raison pour des dizaines de 11 et plus, et `110 / 10`C'est 11 ans.

`prove_behaviour(EQUAL)`construit `percent_change_mul_first.c`et `grade_switch.c`Toutes les 92 cellules sont d'accord.`percent_change_div_first.c`Il échoue dans deux des quatre cas naïfs, qui est le seul cas que le test écrit à la main trouve.

`oracle_disagreements()`Il vérifie une seconde chose.`percent_change_model()`et `grade_model()`sont des modèles Python de la liste, avec `wrap32()`pour le multiplicateur de 32 bits et `idiv32()`Le rapport compare le modèle avec l'original observé sur chaque cellule. Zéro désaccords signifie que le modèle et le binaire sont d'accord sur chaque cellule. Le modèle est une preuve dérivée et la course est une preuve observée. Le rapport les garde dans des champs distincts.

## Deuxième étape: recompiler et comparer

Le deuxième étage demande si la reconstruction, compilée avec le même compilateur et les mêmes drapeaux, produit les mêmes octets. Le projet DX-Ball a rapporté "63 octets correspondants" pour son assistant. Le projet TH04 a rapporté un match de 2,139 octets. Chacun a utilisé le compilateur de sa propre période. réelle elle-même ne peut pas le faire.`compare_functions`le texte du fournisseur de dérivés de ligne et attache la limite "L'assemblage est texte opaque du fournisseur; la normalisation de la relocalisation n'est pas disponible".

### Lire l'objet

`ElfObject`dans `code/main.py`est un objet ELF64 relocalisable avec `struct.unpack_from`Il vérifie la magie, la classe de 64 bits, les données de petite enjeu,`e_machine`62, et `e_type`1. Puis il passe les en-têtes de section, la table de chaînes, `.symtab`, et chaque `.rela`Dans le fichier,`percent_change`commence à`.text`décompensé 0 avec la taille 16 et `grade`à`0x10`avec la taille 46. `total`commence à`0x40`avec la taille 31 et `grade_label`à`0x60`avec la taille 63.

Les 16 octets de `percent_change`sont`85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3`La fonction n'a pas de délocalisation.`grade`a un: décompensé `0x26`à l'intérieur de la fonction, type `R_X86_64_PC32`, symbole `.rodata`, additionnel`0x10`Il appartient au 7 bytes .`leaq`à l' équilibre `0x23`, dont les quatre derniers octets sont le champ de déplacement.

### Masque, puis compare les dossiers

Dans un objet RELA, l'addend vit dans le dossier de relocalisation, et les quatre octets de champ sont zéro.`grade_switch.o`Un simple byte comparer passe alors que les deux délocalisations nomment des lieux différents. `mask_relocations()`Le champ est nul de tous les deux côtés, donc la même méthode s'applique aux objets REL et à Mach-O, où le champ porte l'addend.

Alors ...`prove_bytes()`Comparer les enregistrements de délocalisation par section de compensation, type et cible.`.rodata+0x10`et le candidat dit:`.rodata-0x4`L' addendum diffère parce que l' objet candidat n' a pas de`labels.rel`Une addition différente est un décalage différent, pas une fonction différente.

### Lisez les constantes les points de relocalisation à

Pour un champ relatif au PC qui termine son instruction, le processeur ajoute l'adresse de l'instruction suivante, de sorte que la cible est le symbole plus addend plus 4.`.rodata+0x14`, et dans le candidat c' est`.rodata+0x0`- Je suis là .`read_table()`Il y a cinq entrées de 4 bytes.`1, 2, 3, 4, 4`- Je suis désolé .

Le nombre cinq vient de la liste:`CMP ECX, 0x4`avec `JA`admettant les indices 0 à 4, et `[RCX + RAX*0x4]`lisent 4 octets par index. `CLAIMS`dans `code/main.py`enregistrement de cette forme pour `grade`et enregistrent zéro déménagement pour `percent_change`- Je suis désolé .

Cette étape existe en raison d'un bug documenté par le projet TH08. Sa comparaison plus ancienne "a reproduit l'adresse cible dans le champ de relocation de l'instruction, puis a comparé les octets d'instruction sans vérifier les données littérales flottantes à cette adresse. source `0.0`Il pourrait donc paraître exact contre l' objectif `128.0`." Le correcteur a vérifié 1 548 références littérales flottantes et a corrigé douze autres. `grade_case10_wrong.c`Son code de 46 octets équivaut à l'original, son enregistrement de relocation a la même forme, et sa table est`1, 2, 3, 4, 3`Seul le contrôle constant échoue.

### Résultats

| Candidate | Size | Masked bytes | Relocation records | Table | Status |
|---|---|---|---|---|---|
| `percent_change_mul_first` | 16 vs 16 | equal | none | none | pass |
| `percent_change_split` | 16 vs 16 | equal | none | none | pass |
| `percent_change_wide` | 16 vs 26 | size differs | none | none | fail |
| `percent_change_div_first` | 16 vs 18 | size differs | none | none | fail |
| `grade_switch` | 46 vs 46 | equal | same shape | `1,2,3,4,4` both | pass |
| `grade_table` | 46 vs 46 | equal | same shape | `1,2,3,4,4` both | pass |
| `grade_thresholds` | 46 vs 39 | size differs | 1 vs 0 | none | fail |
| `grade_case10_wrong` | 46 vs 46 | equal | same shape | `1,2,3,4,4` vs `1,2,3,4,3` | fail |

Deux résultats méritent un deuxième regard.`percent_change_split.c`a un texte source différent et des octets identiques, car clang traite le débordement signé comme indéfini et se replie `after * 100 - before * 100`en un multiplier. `grade_table.c`Il correspond également octet par octet, de sorte que le compilateur a atteint la même table à partir d'un index d'un tableau et d'un `switch`. Une correspondance en octets prouve que ce compilateur, avec ces drapeaux, a produit le même code du candidat.

La taille est un diagnostic et non une acceptation. Les pages de preuve de rea enregistrent un candidat de 1 825 octets avec la bonne longueur et 467 octets différents.`wide`reconstruire est de 26 octets parce que `movslq`- Je suis là .`imulq`- Je suis là .`cqto`, et `idivq`La liste nous avait dit la largeur avant toute compilation.

## L' agent est en train de se détourner

65 des outils MCP de rea n'ont pas de commande CLI, et `verify_reconstruction`C'est l'un d'entre eux.`McpStdioClient`dans `code/main.py`C'est le plus petit client qui les atteint.`npx -y rea-agents@6.5.0 mcp`, écrit une requête JSON-RPC par ligne à stdin, et lit un objet JSON par ligne de stdout sur un fil.`initialize`avec la version du protocole `2025-11-25`Alors ...`notifications/initialized`Alors ...`tools/call`. La date limite par défaut par appel est de 330 s, le même nombre réal permet un démarrage à froid Ghidra.

`agent_loop()`Il fait les six appels et enregistre chacun:

```json
{
  "step": 2,
  "tool": "analyze_function",
  "arguments": {"procedure": "percent_change"},
  "seconds": 6.31,
  "is_error": false,
  "evidence_id": "ev_e88b10dc0c6400e51bb6f7fdc3eaea8d2c97cc02463f0db3e6afeefc273197cd",
  "confidence": "observed",
  "authority": "shipped-artifact",
  "limitations": ["Resolved reference metadata identifies computed, indirect, external, call, jump, and data edges; unresolved targetless flows remain unknown, and function classifications distinguish thunks and externals.", "..."],
  "result_sha256": "..."
}
```

Les dossiers complets des preuves sont stockés sous leur identité dans le même fichier.`table_address()`trouve le `LEA`L'opérand dans le `grade`l' énumération et l' alimentation `read_bytes`- Je suis là .`open_binary`- Je suis là .`get_evidence_bundle`, et `close_binary`ne renvoient pas de dossier de preuve et sont enregistrés avec un identifiant nul.

`run_agent(live=True)`réa lorsque `npx`Il est sur le chemin et `GHIDRA_INSTALL_DIR`Il est possible de modifier le nom d'un répertoire, sinon il reproduit le fichier et dit ça sur stderr.`code/tests/fake_rea_server.py`est un serveur alternatif qui répond à chaque `tools/call`Le client et la boucle s'y opposent à chaque test, sans réa, Ghidra ou Java installés.

### Ce que le vérificateur de réa peut et ne peut pas dire

ré's `verify_reconstruction`Les données de référence sont les données de référence de la fonction de référence de la fonction de référence de la fonction de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence de référence`compare_functions`Une affirmation comportementale compare deux captures de processus enregistrées à partir de `compare_process_captures`. Lorsqu'un enregistrement source ne couvre pas complètement les appels, une réclamation concernant les appels est `unknown`Rien à l'intérieur de réa ne compile une fonction reconstruite ou l'exécute sur votre grille d'entrée.

Le registre des obligations de rea s'attend à un dossier de preuve avec `predicate_type: "rea.reconstruction-proof"`- Je suis là .`operation: "verify_reconstruction_obligations"`- Je suis là .`confidence: "observed"`, et `normalized_result.passed === true`Aucun outil public ne l'émet.`build_proof()`dans `code/main.py`le mélange:

```json
{
  "predicate_type": "rea.reconstruction-proof",
  "operation": "verify_reconstruction_obligations",
  "confidence": "observed",
  "authority": "controlled-replay",
  "normalized_result": {
    "passed": true,
    "status": "pass",
    "claims": [
      {"claim_id": "percent_change.behaviour", "kind": "behavioral", "status": "pass", "evidence_ids": ["ev_e88b10dc..."]},
      {"claim_id": "grade.behaviour", "kind": "behavioral", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]},
      {"claim_id": "percent_change.bytes", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_e88b10dc..."]},
      {"claim_id": "grade.bytes", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]},
      {"claim_id": "grade.referenced-constants", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]}
    ]
  },
  "evidence_links": ["ev_e88b10dc...", "ev_67b6d3f9..."],
  "evidence_id": "ev_..."
}
```

L' identité est `ev_`Plus le SHA-256 du disque sans l'identifiant, sur JSON compact à clé triée. `aggregate()`est fermé en cas de défaillance: `fail`Il gagne, alors.`unknown`, et une liste de demandes est vide `unknown`Sur un hôte sans compilateur C , les deux revendications de comportement sont `unknown`, le dossier dit`unknown`même si les deux demandes de octets passent. `authority`est `controlled-replay`, parce que la preuve vient de notre propre harnais et pas du seul fichier objet.

## Ce que la recherche dit

Liu, Raff et Micinski ont mesuré huit systèmes de décompilation dans neuf configurations en 2026. Les candidats qui ont passé tous les tests que l'indice de référence inclut sont toujours divergents sur les entrées fuzzed. Le taux était de 4,9% dans l'ensemble et de 13% pour un système. Sur 300 fonctions de bibliothèque GitHub et 287 fonctions basées sur CVE, le modèle de raffinement le plus fort a augmenté le taux de création de Ghidra de 75% à 90%. Sur les mêmes fonctions, le taux de correspondance comportementale est passé de 74% à 62%. Ils ont repéré la cause à des champs, des types, des calées et des gardes inventés qui ont remplacé les inconnus visibles qu'un décompilateur traditionnel laisse derrière lui.

Notre .`percent_change_wide.c`Le pseudocode offrait une`long`Le groupe de recherche a été créé par le groupe de recherche de Ghidra, qui a réussi à obtenir des résultats de recherche de la recherche de la recherche de la recherche de l'information sur la technologie de l'information et de l'information sur la technologie de l'information.

DecompileBench a mesuré 23 400 fonctions à partir de 130 programmes réels. Il a trouvé que les méthodes LLM étaient 52,2% moins précises que les décompilateurs industriels, avec des scores de compréhensibilité plus élevés.

Les trois preuves que le champ utilise sont les trois pattes que vous pouvez construire. Phoenix et DREAM recompilent la sortie et exécutent les tests du programme d'origine. McKeeman et Csmith exécutent des entrées générées par rapport à l'original avec des entrées conservées à l'intérieur du comportement défini. Ramos et Engler vérifient deux fonctions C avec une exécution symbolique limitée, ce qui prouve l'égalité sur tous les chemins explorés jusqu'à une taille d'entrée. Cette leçon construit les deux premières. Le troisième est le laboratoire de pratique dans le registre des preuves de la leçon 35 quand une grille est trop petite pour être fiable.

## Faites-le

- Je vous ai donné le harnais de la bibliothèque standard.

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/main.py
```

Avec le clang sur le chemin, la démo compile les deux paires de candidats et exécute 92 cellules pour chacun. Il recompile huit candidats, reproduit l'agent exécuté, et remplit le dossier de preuve. Sans clang il lit les objets engagés et rapporte les réclamations de comportement comme`unknown`- Je suis désolé .

Forme prévue sur un arm64 Mac:

```json
{
  "host": "aarch64 Darwin",
  "originalKind": "fixture-source",
  "gridCells": 92,
  "plausible": {"naivePassed": true, "gridMismatches": 28, "oracleDisagreements": 0},
  "equal": {"naivePassed": true, "gridMismatches": 0, "oracleDisagreements": 0},
  "legTwo": {
    "percent_change_mul_first": {"status": "pass", "reasons": []},
    "percent_change_wide": {"status": "fail", "reasons": ["size 16 vs 26 bytes"]},
    "grade_case10_wrong": {"status": "fail", "reasons": ["referenced constants differ: [1, 2, 3, 4, 4] vs [1, 2, 3, 4, 3]"]}
  },
  "agentRun": {"mode": "replayed fixture", "server": {"name": "rea", "version": "6.5.0", "protocolVersion": "2025-11-25"}},
  "proof": {"status": "pass", "evidence_id": "ev_..."}
}
```

Lire `code/main.py`dans cet ordre:

1. `wrap32()`- Je suis là .`c_div()`- Je suis là .`idiv32()`, et les deux `*_model()`Les fonctions contiennent la sémantique entière de la liste.
2. `input_grid()`et `classify()`construire et marquer les cellules 92.
3. `build_harness()`et `execute()`compiler un programme à partir de l'original, les candidats renommés, et `driver.c`, puis faire fonctionner les cellules.
4. `prove_behaviour()`renvoie le rapport de première étape avec des désaccords et des désaccords de modèle.
5. `ElfObject`Il lit les en-têtes, les symboles et `.rela`Les dossiers.`mask_relocations()`et `read_table()`préparer la comparaison.
6. `prove_bytes()`renvoie le rapport de la jambe à deux avec des tailles, des intervalles masqués, la forme de la relocalisation et des tableaux.
7. `McpStdioClient`et `agent_loop()`réalisez et enregistrez chaque étape.
8. `build_proof()`et `aggregate()`- le dossier de défaillance.

## Utilisez-le

Pour prouver que la nouvelle fonction est égale, répétez ces étapes:

1. Apportez-le .`analyze_function`Pour la fonction, et enregistrer l'identifiant de la preuve.
2. Écrivez le candidat à partir des largeurs d'opération de la liste.
3. Lisez chaque fois que vous multipliez, ajoutez, soustraisez et divisez dans la liste. Ajoutez des entrées de grille qui enveloppent chacune à sa largeur.
4. Ajoutez les deux côtés de chaque constante de comparaison dans la liste.`grade`Ce sont 6 et 4 après le soustrait.
5. Ajoutez tous les diviseurs qui peuvent piéger sur la cible, et enregistrer quelle architecture a exécuté la grille.
6. Faites-moi courir les deux jambes et lisez les raisons avant le statut.
7. Gardez le dossier à côté du candidat.`evidence_links`indiquer à un lecteur sur lequel repose la demande.

Récouter les deux pattes après une mise à jour du compilateur.

## Laboratoire interactif

### Laboratoire A: le modèle et la course

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal/code
python3 -q
```

Je vais courir .

```python
from main import *
percent_change_model(3, INT_MAX)
percent_change_model(-1, 536870911, "x86_64")
percent_change_model(-1, 536870911, "aarch64")
report = prove_behaviour(PLAUSIBLE)
report.naive_passed, len(report.mismatches), report.mismatches[0]
```

Le premier appel donne -133, le produit enveloppé divisé par 3. Le second donne `'trap'`Le rapport montre que dix cas naïfs ont été passés et 28 cellules grille ont échoué.`report.oracle_disagreements`avec une liste vide.

### Laboratoire B: lire l'objet

```python
obj = ElfObject(ORIGINAL_OBJECT)
grade = obj.function("grade")
grade.size, grade.data[0x26:0x2a].hex(), grade.relocations
read_table(obj, grade.relocations[0], 5, 4)
```

Les octets de champ sont `00000000`- Les noms des délocalisations`.rodata`avec ajout `0x10`, et la table est`[1, 2, 3, 4, 4]`- Je suis désolé .

### Laboratoire C: briser une constante

```python
report = prove_bytes("grade", CANDIDATES / "grade_case10_wrong.o")
report.masked_equal, report.relocations_match, report.tables_equal, report.reasons()
```

Les octets correspondent, les enregistrements correspondent, et les tables ne le font pas.`grade_case10_wrong.c`Alors ...`case 10`retourne 4 encore, le recompiler avec la commande de fixation, et regarder le changement d'état à `pass`- Je suis désolé .

### Laboratoire D: réaction

Avec réa 6.5.0 et Ghidra 12.1.4 installés:

```bash
GHIDRA_INSTALL_DIR=/opt/homebrew/Cellar/ghidra/12.1.4/libexec \
JAVA_HOME=/opt/homebrew/opt/openjdk@21 \
python3 code/main.py --live
```

La première .`analyze_function`Il faut quelques secondes pendant que Ghidra importe l'objet. Comparer les nouveaux identifiants de preuve avec la mise en œuvre engagée. Les mêmes octets sous le même profil d'analyse donnent les mêmes identifiants, car l'id est un hash de contenu qui exclut le chemin du fichier.

## Laboratoire de pratique

Étendre les deux jambes à `total`et `grade_label`du même objet.

Les exigences:

- Pour `total`, l'entrée est un tableau et un compte. Construisez des cellules avec un compte 0, un compte négatif, un élément, et des sumes qui se terminent à 32 bits.
- Pour `grade_label`, la fonction appelle `grade`en ligne et se lit `labels.rel`, une table de cinq compensations relatives de chaînes avec leurs propres délocalisations dans `.rela.rodata`- Je suis désolé .
- Ajouter un `CLAIMS`entrée pour `grade_label`avec deux relocalisations et une table de cinq entrées de 4 bytes, et comparer les chaînes auxquelles les entrées pointent.
- Enregistrez les cellules sur lesquelles le modèle et la course ne sont pas d'accord, le cas échéant, et expliquez chacune de ces cellules à partir de la liste.
- Ajouter une réclamation par étape par fonction au dossier de preuve et garder l'ensemble de défaillance fermé.

Le succès signifie que le dossier de preuve énumère quatre fonctions, chaque réclamation porte un identifiant de preuve, et `grade_label`échoue lorsqu'une chaîne d'étiquette est différente.

## Les compétences en matière de sortie

Cette leçon produit `outputs/skill-rebuild-and-prove-equal.md`. Utilisez-le pour transformer la preuve de fonction et la reconstruction d'un candidat en une preuve à deux pattes et un enregistrement de preuve.

## Vérifiez

Exécutez la démo et la suite de test:

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Les 35 tests prouvent:

- `c_div()`se tronque vers le zéro et `wrap32()`suit le complément de deux
- Le modèle donne -133 pour `(3, INT_MAX)`- Je suis là .`trap`pour la cellule de piège sur x86-64 et -2147483648 sur AArch64
- La grille a 92 cellules et chaque classe de danger, et les cas naïfs évitent chaque classe dure
- le lecteur ELF trouve quatre fonctions aux compensations et tailles enregistrées
- `percent_change`est de 16 octets sans délocalisation et `grade`Il en a un .`R_X86_64_PC32`à`0x26`
- les octets du champ de relocation sont zéro et le masquage ne change rien dans cet objet RELA
- les huit candidats engagés atteignent le statut attendu, avec les raisons indiquées
- `grade_case10_wrong`passe des octets masqués et ne parvient que à vérifier la constante
- Les courses enregistrées de la jambe 1 montrent dix passes naïves, 28 désaccords et zéro désaccord de modèle
- avec un compilateur, les deux paires reconstruisent et redémarrent avec le même résultat
- l'agent est dirigé vers l'objet engagé par SHA-256, dans l'ordre suivant:`ev_`Les identifiants de chaque étape de la preuve
- le `read_bytes`record est égal à la table que le lecteur ELF trouve
- le client exploite le faux serveur et enregistre les mêmes identifiants de preuve que la vraie course
- un résultat d'erreur du serveur entre dans le dossier d'exécution
- `aggregate()`est fermé en défaut et l'identifiant de preuve recompte à partir de son contenu

## Mode de défaillance de la production

| Failure | What the weak report says | What the harness must prove |
|---|---|---|
| Rebuild keeps the decompiler's `long` | "ten tests pass" | 26 overflow cells differ from the original |
| Thresholds replace a switch | "every grade band passes" | 110 and larger return 0 in the original |
| Divide before multiply | "formula is algebraically equal" | rounding cells differ, naive cases fail |
| Trap cell skipped | "no crashes" | x86-64 raises `#DE`, AArch64 returns `INT_MIN`, the report names the host |
| Byte compare without masking | "bytes equal" | relocation records name different offsets |
| Masking without reading constants | "bytes equal after masking" | `1,2,3,4,3` differs from `1,2,3,4,4` |
| Size taken as acceptance | "same length" | differing bytes inside the same length |
| Different compiler or flags | "bytes differ, rebuild wrong" | the same source gives different bytes, so the compare is void |
| Model treated as observation | "model agrees" | the run is observed, the model is derived, disagreements are listed |
| Claim without evidence id | "verified" | every claim links an `ev_` record of the function it covers |
| No compiler on the CI host | "passed" | behaviour claims are `unknown` and the aggregate is `unknown` |
| Pseudocode read as source | "decompiler says long" | the listing shows `IMUL EAX`, `CDQ`, `IDIV EDI` |

## Règlement d'exploitation

Écrivez la reconstruction de la liste et nommez le registre d'où elle est venue. Appliquez-la contre l'original sur une grille qui enveloppe chaque intermédiaire et traverse chaque constante de comparaison. Remplissez-la avec les drapeaux originaux, masquez les champs de relocalisation, comparez les enregistrements et lisez les constantes qu'ils pointent vers. Enregistrez chaque réclamation avec son identifiant de preuve, et laissez une réclamation non vérifiée rester inconnue.

## Pour en savoir plus

- [McKeeman, Differential Testing for Software, Digital Technical Journal 10(1), 1998](https://www.cs.tufts.edu/comp/150FP/archive/bill-mckeeman/DifferentailTesting.pdf)
- [Yang, Chen, Eide, Regehr, Finding and Understanding Bugs in C Compilers, PLDI 2011](https://doi.org/10.1145/1993498.1993532)
- [Ramos, Engler, Practical, Low-Effort Equivalence Verification of Real Code, CAV 2011](https://doi.org/10.1007/978-3-642-22110-1_55)
- [Schwartz, Lee, Woo, Brumley, Native x86 Decompilation Using Semantics-Preserving Structural Analysis, USENIX Security 2013](https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/schwartz)
- [Liu, Raff, Micinski, When LLM Decompilers Recompile More and Preserve Less, arXiv 2609.05370, 2026](https://arxiv.org/abs/2609.05370)
- [Gao et al., DecompileBench, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1194/)
- [Tan et al., LLM4Decompile, arXiv 2403.05286, 2024](https://arxiv.org/abs/2403.05286)
- [System V gABI, Chapter 4, Relocation](https://www.sco.com/developers/gabi/latest/ch4.reloc.html)
- [x86-64 psABI repository, relocation types](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2, IDIV](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Arm Architecture Reference Manual for A-profile, DDI 0487, SDIV](https://developer.arm.com/documentation/ddi0487/latest)
- [rea 6.5.0 MCP contracts](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/mcp-contracts.md)
- [rea 6.5.0 reconstruction obligation ledgers](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/reconstruction-obligation-ledgers.md)
