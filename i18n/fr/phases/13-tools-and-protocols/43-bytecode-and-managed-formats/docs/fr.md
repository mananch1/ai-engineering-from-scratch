# Au-delà du code natif: Python, JVM, .NET et Bytecode WebAssembly

> Un fichier de code octal est un programme plus les notes du compilateur à ce sujet, et cette leçon construit les lecteurs qui montrent quelles notes chaque conteneur conserve.

**Type:** Build
**Languages:** Python, Java, C#, WebAssembly text
**Prerequisites:** Phase 13 · 32 (ELF from scratch), 33 (disassembly and control flow)
**Time:** ~150 minutes

## Objectifs d'apprentissage

- Lisez le 16 octets `.pyc`en-tête de la PEP 552 et démarcher l'objet code sans le `marshal`module, pour les fichiers écrits par quatre versions interprètes.
- Parser un fichier de classe JVM à partir du JVMS Chapitre 4: pool constant, champs, méthodes, le `Code`attribut, et les tables qui `javac -g`- Il y a aussi.
- Démontrez les codes opcodes de la JVM en `Ledger.class`et prouver la liste contre `javap -c`sur quatre constructions de la même source.
- Lisez une section par section de module WebAssembly, décodez chaque corps de fonction et prouvez le résultat en fonction de `wasm-objdump -d`- Je suis désolé .
- Passez les métadonnées ECMA-335 d' un ensemble .NET du répertoire de données PE à la `MethodDef`les lignes et le CIL de chaque méthode, vérifiés par rapport à la bibliothèque de lecteur utilisée par ILSpy.
- Mesurer, pour une fonction dans cinq conteneurs, les noms, types et structures de contrôle que chaque format conserve et quels il perd.

## Le problème

Vous avez construit .`ledger.py`et trouvé`ledger.cpython-314.pyc`dans `__pycache__`Vous avez demandé à un agent ce qu'il y avait dans le dossier, l'agent a mis en place un décompiler.`Unsupported opcode`et une ligne qui disait:`# WARNING: Decompyle incomplete`L'agent a conclu que le dossier était obscurci.

Le fichier n'a pas été obscurci.`2b 0e`C'est le nombre que CPython 3.14 écrit, et les tables de code op de décompiler s'arrêtent à 3.13.

La leçon 33 montre ce que le code natif perd: les noms des paramètres, les types et les`switch`qui est devenu une table de saut. Les conteneurs de code octogène perdent moins, et chacun perd quelque chose de différent. Un pyc conserve tous les noms de variables et aucun type. Un fichier de classe ne conserve les types et les noms locaux que lorsque vous demandez. Un module de Wasm conserve la structure de contrôle et dépose les noms à moins qu'une section personnalisée ne les porte. Un assemblage .NET conserve les noms de paramètres dans une table et laisse les noms locaux dans un fichier séparé. Vous ne pouvez pas juger la sortie d'un décompilateur sur aucun d'entre eux jusqu'à ce que vous sachiez avec quelles notes le conteneur a dû commencer.

Cette leçon construit quatre lecteurs dans la bibliothèque standard Python. Chacun lit le conteneur à partir de son premier octet, énumère une fonction et est vérifié par rapport à l'outil du format sur les accessoires que vous avez construits.`grade`de `ledger`côte à côte dans les cinq conteneurs.

```figure
re-43-five-containers
```

## Le pyc Header est quatre mots

PEP 552 définit l'en-tête comme quatre mots de 32 bits. Le premier est le nombre magique. Le second est un champ de bit.`check_source`L'interprète hashes la source à l'importation et compare. Avec bit 0 clair, les mots trois et quatre sont le source mtime et la taille de la source. Avec bit 0 ensemble, ils contiennent un SipHash de 64 bits de la source.

Le nombre magique vit dans`Include/internal/pycore_magic_number.h`La règle de Python 3.11 est que Python 3.n commence à 2900 + 50n, et chaque modification de code octal au sein d'un cycle de libération en ajoute un.`\r\n`Python 3.15.0 a été publié le 2026-10-09 et n'est pas installé sur la machine qui a construit ces fixtures, donc son numéro provient du fichier d'en-tête et non d'un fichier ici.

Les appareils contiennent deux fichiers pyc par interprète. `compile_pyc.py`Je les ai écrites avec `py_compile.compile(..., dfile="ledger.py")`afin que le nom du fichier stocké soit `ledger.py`Les 16 premiers octets des fichiers 3.14:

```text
checked-hash   2b 0e 0d 0a  03 00 00 00  c7 16 c7 fc c5 1d 81 01
timestamp      2b 0e 0d 0a  00 00 00 00  8d 60 cf 6a  f1 01 00 00
```

Lisez l'en-tête de l'étemplaire mot par mot. `03 00 00 00`ou `00 00 00 00`C'est le mot de drapeau.`8d 60 cf 6a`est 0x6acf608d = 1791975565, le mtime de `ledger.py`quand le dossier a été construit. `f1 01 00 00`Le fichier de hash vérifié contient des drapeaux 3 et huit bytes hash.

| Interpreter | Magic bytes | Word | File bytes | `grade.co_code` bytes | `grade` instructions |
|---|---|---|---|---|---|
| 3.11.14 | `a7 0d 0d 0a` | 3495 | 1054 | 80 | 31 |
| 3.12.12 | `cb 0d 0d 0a` | 3531 | 937 | 62 | 26 |
| 3.13.12 | `f3 0d 0d 0a` | 3571 | 943 | 70 | 26 |
| 3.14.6 | `2b 0e 0d 0a` | 3627 | 965 | 96 | 35 |

`read_pyc_header()`dans `code/main.py`renvoie ces champs et nomme la version du mot. `importlib.util.MAGIC_NUMBER`donne les mêmes quatre octets pour l'interprète en cours d'exécution. `python3 code/main.py compare`compile la source avec chaque `python3.11`à `python3.15`trouvé sur `PATH`Pour chacun, il imprime l'en-tête que le lecteur voit à côté des faits que l'interprète rapporte sur lui-même.

## Le maréchal écrit le code Objet champ par champ

Le 16 octets de chaque fichier est `e3`- C' est ça .`0x80 | ord('c')`: le code de type marshal pour un objet de code avec `FLAG_REF`l'ensemble, qui dit au lecteur de se souvenir de l'objet pour plus tard `r`Les documents du marshal indiquent que le format est délibérément indocumenté et que les objets de code ne sont pas compatibles entre les versions.`Python/marshal.c`répertorie les codes de type: `N`Aucun.`i`une int de 32 bits, `s`en octets, `z`et `Z`courts chaînes ASCII, `(`et `)`- Les deux autres.`c`un objet de code, `r`Les conteneurs réservent leur place de référence devant leurs enfants, de sorte que le lecteur fait de même.

`w_complex_object()`Il est écrit un objet de code dans un ordre fixe.`argcount`- Je suis là .`posonlyargcount`- Je suis là .`kwonlyargcount`- Je suis là .`stacksize`- Je suis là .`flags`comme 32 bits de l'ins, alors `code`- Je suis là .`consts`- Je suis là .`names`- Je suis là .`localsplusnames`- Je suis là .`localspluskinds`- Je suis là .`filename`- Je suis là .`name`- Je suis là .`qualname`en tant qu'objets, alors `firstlineno`Alors ...`linetable`et `exceptiontable`. Décoder manuellement l'objet du code du module du fichier 3.14 à partir de l' octet 17:

```text
00 00 00 00   argcount 0
00 00 00 00   posonlyargcount 0
00 00 00 00   kwonlyargcount 0
01 00 00 00   stacksize 1
00 00 00 00   flags 0
f3 22 00 00 00   type 's' with FLAG_REF, 34 bytes of co_code follow
80 00  52 06  74 00  52 01  17 00  74 01 ...
```

Les octets de code sont deux par instruction: `80 00`est `RESUME 0`- Je suis là .`52 06`est `LOAD_CONST 6`- Je suis là .`74 00`est `STORE_NAME 0`, et `17 00`est `MAKE_FUNCTION`La liste 3.14 des fixations imprime les mêmes cinq déclarations au niveau du module.`code`viennent les constantes, qui contiennent les quatre objets de code nichés, et après eux `names`Le numéro de la liste:`('LABELS', 'percent_change', 'grade', 'total', 'grade_label')`- Je suis désolé .

`Unmarshaller`dans `code/main.py`Il est écrit dans cette disposition et dans la version ancienne que 3.9 et 3.10 ont utilisée, où `nlocals`- Je suis là .`varnames`- Je suis là .`freevars`, et `cellvars`Les champs sont séparés.`CodeObject`avec les mêmes noms d'attributs que l'objet réel. `varnames`filtres `localsplusnames`par le `CO_FAST_LOCAL`Un peu de`localspluskinds`Pour ...`grade`dans chacun des quatre dossiers qu' il renvoie `('score', 'bucket')`- Je suis là .`firstlineno`10 et une non-voie `linetable`- Le test .`test_unmarshaller_agrees_with_the_running_interpreter`comparer chaque champ avec `marshal.loads()`Les trois autres fichiers sont lisibles par notre code et illisibles par notre code.`marshal`sous cet interprète.

## Quatre interprètes, quatre code de code pour la note

La dé-documentation contient un avertissement sur chaque version: le code octal est un détail d'implémentation de CPython et aucune garantie n'est faite qu'il reste le même.`python3.1N -m dis ledger.py`Il a été capturé pour chaque interprète.

Dans 3.11 `grade`commence comme ceci:

```text
 11           2 LOAD_FAST                0 (score)
              4 LOAD_CONST               1 (10)
              6 BINARY_OP                2 (//)
             10 STORE_FAST               1 (bucket)
 12          12 LOAD_FAST                1 (bucket)
             14 LOAD_CONST               2 (9)
             16 COMPARE_OP               5 (>=)
             22 POP_JUMP_FORWARD_IF_FALSE     2 (to 28)
```

`BINARY_OP`Les deux octets entre eux sont une entrée cache en ligne de PEP 659, l'interprète adaptatif spécialisé. `COMPARE_OP`Le 16 est suivi par le 22, donc il possède deux entrées cachées.`show_caches=True`. 3.12 remplace `LOAD_CONST`plus `RETURN_VALUE`avec `RETURN_CONST`, renommé le saut à `POP_JUMP_IF_FALSE`, et les psychiatriques .`grade`de 80 à 62 octets. 3,13 imprimés étiquettes `L1`à `L4`au lieu de compenser et donne l'indice de valeur`-C`pour les caches et `-O`pour les compensations. 3.14 supprime `RETURN_CONST`- encore une fois, ajoute-t-il.`LOAD_SMALL_INT`et `LOAD_FAST_BORROW`, et les inserts `NOT_TAKEN`après chaque saut conditionnel:

```text
 12         20       LOAD_FAST_BORROW         1 (bucket)
            22       LOAD_SMALL_INT           9
            24       COMPARE_OP             188 (bool(>=))
            26       CACHE                    0 (counter: 0)
            28       POP_JUMP_IF_FALSE        3 (to L1)
            30       CACHE                    0 (counter: 0)
            32       NOT_TAKEN
```

Le `-C -O`la liste montre 13 `CACHE`Les entrées à l'intérieur `grade`: cinq après `BINARY_OP`, un après l' autre .`COMPARE_OP`, un après l' autre .`POP_JUMP_IF_FALSE`Le dispositif d'adaptation montre à quoi servent les caches.`dis.dis(grade, adaptive=True)`des empreintes`RESUME_CHECK`et `COMPARE_OP_INT`où la liste simple avait `RESUME`et `COMPARE_OP`, et `total`Les spectacles`FOR_ITER_LIST`- Je suis là .`BINARY_OP_ADD_INT`, et `JUMP_BACKWARD_NO_JIT`Les formes spécialisées vivent dans la même`co_code`Les pyc ne les contiennent jamais, car la spécialisation se fait dans la mémoire.

C'est pourquoi les décompilateurs ont des plafonds de version. uncompyle6 3.9.3 indique qu'il accepte le code par octets de Python 1.0 à 3.8. décompyle3 3.9.3 lit 3.7 et 3.8. xdis 6.3.0, le démontant de version croisée derrière les deux, lit jusqu'à 3.13. pycdc a des tables de code optique jusqu'à `python_3_13.cpp`dans son fichier de construction, et sur une imprimante de code op inconnu `Unsupported opcode`PyLingual revendique chaque version publiée depuis 3.6 et utilise des interprètes de 3.8 à 3.15 pour recompiler et vérifier sa propre sortie. Pour un fichier 3.14 les trois premiers rejettent ou échouent, et seuls les deux derniers s'appliquent.

## Le dossier de classe garde un dictionnaire

Le JVMS Java SE 25, section 4.1, donne la mise en page en tant qu'une seule structure: `magic`- Je suis là .`minor_version`- Je suis là .`major_version`- Je suis là .`constant_pool_count`, la piscine constante,`access_flags`- Je suis là .`this_class`- Je suis là .`super_class`Les interfaces, les champs, les méthodes, les attributs.`0xCAFEBABE`Une machine virtuelle Java SE 25 accepte les versions majeures 45 à 69, et une version mineure de 65535 marque un fichier de fonctionnalité d'aperçu.`Ledger-g.class`sont`ca fe ba be 00 00 00 41 00 5e`: majeur 65 est Java 21, et `0x5e`= 94 est le nombre constant de poissons. `Ledger-jdk25.class`, construit par javac 25.0.2 à partir de la même source, commence `ca fe ba be 00 00 00 45`Le majeur 69.

La base constante est l'endroit où le fichier de classe garde son propre dictionnaire.`#13`dans le `-g`construire est un `Methodref`qui pointe à la classe `#8`et nom et type `#14`. Suivant les indices donne `Ledger.grade:(I)I`, ce qui est exactement ce que `javap -v`des empreintes après `//`- Le descripteur .`(I)I`est un`int`paramètre et un `int`Le résultat.`([I)I`sur`total`dit que le paramètre est un `int`Chaque méthode, champ et classe que le code touche a une telle entrée, et le vérificateur de la section 4.10 refuse un fichier dont les instructions ne sont pas d'accord avec eux.`ClassFile`classe en `code/main.py`lisent les 17 étiquettes constantes du tableau 4.4 et traitent un `Long`ou `Double`Il s'élève sur une balise inconnue et sur des octets après le dernier attribut.

Les attributs dépendent de la classe, de chaque champ, de chaque méthode et de la `Code`JVMS dit que la machine virtuelle doit ignorer les attributs qu'elle ne connaît pas, c'est pourquoi les tables de débogage peuvent être ajoutées ou supprimées sans changer un corps de méthode.`-g`génère toutes les informations de débogage, y compris les variables locales, que le numéro de ligne et le fichier source sont générés par défaut seulement, et que `-g:none`Les trois construisent une mesure qui:

| Build | Bytes | Constant pool count | `LineNumberTable` | `LocalVariableTable` | `SourceFile` | `grade` code bytes |
|---|---|---|---|---|---|---|
| `javac -g` | 1832 | 94 | 7 methods | 6 methods | yes | 50 |
| `javac` | 1581 | 82 | 7 methods | none | yes | 50 |
| `javac -g:none` | 1404 | 79 | none | none | no | 50 |

Les 50 octets de code de `grade`Les trois dossiers sont identiques, et le test le vérifie.`-g`ajout pour `grade`est un`LocalVariableTable`ligne: début 0, longueur 50, espace 0, nom `score`, décritrice `I`Sans elle , un décompilateur voit la fente de type 0 .`int`Il doit inventer un nom.`StackMapTable`reste dans chaque construction parce que le vérificateur en a besoin, et le `BootstrapMethods`l' attribut reste parce que `main`concatenant les chaînes à travers `invokedynamic`- Je suis désolé .

## Décodage des tables à la main

Le chapitre 6 du JVMS définit chaque instruction comme un octet opcode et les opérands qui nomment opcode. Les branches offsets sont des valeurs signées de 16 bits par rapport à l'adresse de l'opcode de branche, qui diffère de x86-64, où le déplacement est relatif à l'instruction suivante. `tableswitch`et `lookupswitch`commencez par 0 à 3 octets de remplissage de sorte que leurs opérands à 32 bits commencent à une adresse qui est un multiple de 4.

Les octets de `grade`à partir de l' ordinateur 4 en `Ledger-g.class`- Le numéro de la liste:

```text
aa                 tableswitch at pc 4
00 00 00           padding to pc 8
00 00 00 2c        default: 4 + 44 = 48
00 00 00 06        low 6
00 00 00 0a        high 10
00 00 00 2a        case 6:  4 + 42 = 46
00 00 00 28        case 7:  4 + 40 = 44
00 00 00 26        case 8:  4 + 38 = 42
00 00 00 24        case 9:  4 + 36 = 40
00 00 00 24        case 10: 4 + 36 = 40
```

36 octets, une instruction, et tout.`switch (score / 10)`La construction native de la même `switch`Dans la leçon 33, il fallait vérifier les limites,`lea`, une `movslq`, une`add`Les cibles sont dans l'instruction, et le fait qu'elles appartiennent ensemble.

`disassemble_jvm()`couvre tous les 202 opcodes définis avec leurs largeurs d'opérand, suit la règle de rembourrage et rend les opérands la façon `javap`Il le fait.`compare_with_javap()`Il analyse les prisonniers .`javap -c -p`texte, y compris le texte multilinéaire `tableswitch`Les quatre constructions sont d'accord sur 108 des 108 instructions.`test_branch_offsets_are_relative_to_the_branch_instruction`les chèques `ifne`à l'atterrissage sur le pc 1 le 6, `if_icmpge`à 13 et à 33`goto`À 30h, retour sur 10.

Radar2 6.2.4, construit à partir de la source pour ces leçons, lit le même fichier. `rabin2 -I`rapports `arch java`- Je suis là .`lang java 21`- Je suis là .`linenum true`, et `lsyms true`, qui est la lecture de la version et des deux tables de débogage. `pd 14 @ sym.Ledger.grade`imprime le `tableswitch`Il imprime aussi des traces.`Unable to parse code attribute: 2`Il est donc nécessaire de mettre en place des procédures de certification et de vérification des données.`javap`comme l'oracle. Les décompilateurs pour ce format sont matures par rapport à Python. CFR 0.152 est sous licence MIT. Procyon 0.6.0 et Vineflower 1.12.0 sont Apache-2.0, et Vineflower est un fourchette Fernflower avec support Java 21. Fernflower lui-même vit dans le référentiel de la communauté IntelliJ. JADX 1.5.6 couvre le format Android DEX, et JD-GUI a été publié pour la dernière fois en 2019. Harrand, Soto-Valero, Monperrus et Baudry en ont mesuré huit en 2020. Le meilleur produit syntaxiquement correct pour 84% des classes et sémantiquement équivalent pour 78%. Leur méta-décompilateur a récupéré 37,6% des classes qu'aucun outil ne gérait. Les types, les noms et les piles vérifiées rendent le format décompilable. Ils ne rendent pas un seul décompilateur complet.

## Un module de la chaux est une liste de sections

La spécification de base définit un module comme le préambule `00 61 73 6d 01 00 00 00`, la corde `\0asm`et la version 1, suivie de sections.`u32`Les ids 1 à 13 sont les sections connues, chacune au plus une fois et dans l'ordre. id 0 est une section personnalisée avec un nom, autorisé partout. tous les entiers sont LEB128, donc`e4 00`est de 100 et `80 08`C'est le 1024.`wasm-objdump -h`sur`ledger.wasm`listes de sept sections, et `WasmModule`trouve les mêmes sept avec les mêmes compensations de charge utile:

| Section | Payload | Count | What it holds for ledger |
|---|---|---|---|
| Type | 10 to 26 | 3 | `(i32) -> nil`, `(i32, i32) -> i32`, `(i32) -> i32` |
| Import | 28 to 39 | 1 | `env.log` with type 0 |
| Function | 41 to 46 | 4 | type index for each defined function |
| Memory | 48 to 51 | 1 | one page, no maximum |
| Export | 53 to 110 | 5 | `memory` and the four function names |
| Code | 113 to 271 | 4 | bodies of 23, 63, 44, and 23 bytes |
| Data | 273 to 290 | 1 | `F\0D\0C\0B\0A\0` at offset 1024 |

La division entre les sections Fonction et Code existe afin qu'un navigateur puisse commencer à compiler avant l'arrivée de l'ensemble du fichier.`grade`Le vecteur local est `01 7f`: un groupe d' un `i32`- C' est ça .`bucket`, avec son type et sans son nom.

Le flux de contrôle est structuré par construction.`block`- Je suis là .`loop`ou `if`Les étiquettes sont indexées par profondeur de nidification. `br_table`L'opérand est une branche indirecte qui indique une liste d'étiquettes.`grade`à l' échange de fichiers `0xa3`- Le numéro de la liste:

```text
0e                  br_table
0b                  11 labels follow
04 04 04 04 04 04   buckets 0 to 5: label 4, the outermost block, returns 0
03 02 01 00         buckets 6, 7, 8, 9: labels 3, 2, 1, 0
00                  bucket 10: label 0, shared with 9
04                  default label 4
```

Chaque étiquette est une profondeur.`block`, et le code après son `end`est `i32.const 4`suivie de `br 4`Le résultat est le bloc.`block`Les instructions devant la table, les onze étiquettes et les cinq `br`les instructions après l' ensemble `switch`, et un validateur rejette toute branche qui quitte la structure. `disassemble_wasm()`La profondeur est suivie, donc les tirages de liste comme `wasm-objdump`, couvre chaque code optique de un seul octet de `unreachable`à `i64.extend32_s`, et correspond aux captifs `wasm-objdump -d`sur 77 des 77 instructions pour les deux constructions.

Les noms sont la seule chose qu'un module simple laisse tomber.`wasm-objdump`Des empreintes`<grade>`Parce que la section Export carte que la chaîne à fonctionner l'index 2, et le lecteur fait la même chose.`wat2wasm --debug-names`ajoute une section personnalisée nommée `name`Le fichier est de 425 octets. Le lecteur trouve `{0: 'score', 1: 'bucket'}`pour la fonction 2 et `{0: 'values', 1: 'count', 2: 'sum', 3: 'i'}`pour la fonction 3. `wasm-strip`Le référentiel des conventions d'outils décrit deux autres porteurs.`.debug_*`Les sections personnalisées.`sourceMappingURL`la section personnalisée peut indiquer une carte source dont les numéros de colonne sont des dérivés en octets dans la liste binaire.

L'outillage a été déplacé en 2026.`wasm-decompile`, et la construction Homebrew sur cette machine a exactement les onze outils restants.`print`- Je suis là .`objdump`, et `demangle`. La version Binaryen_133 fournit `wasm-dis`Le radar 6.2.4 lit aussi le module.`rabin2 -I`dit `arch wasm`C' est le symbole de la`grade`commence à`0x8e`, où le vecteur local commence, alors que `wasm-objdump`étiquette la fonction à `0x8b`La liste utilise les mnemoniques plus anciens.`get_local`- Je suis là .`set_local`, et `brtable 11 4 4 ...`avec le nombre d'étiquettes imprimé. Les octets sont les mêmes dans les trois outils et les orthographes diffèrent, donc une comparaison doit nommer l'outil avec lequel il a été capturé. La recherche pour cette leçon n'a trouvé aucun module de processeur WebAssembly dans le répertoire de processeurs de Ghidra, et cette liste n'a pas été lue en entier. Le plugin ghidra-wasm v2.4.0 de tiers cible Ghidra 12.0. La spécification elle-même est un document vivant. La page du W3C pour `wasm-core-2`est un projet de recommandation de candidat du 13 octobre 2026 dont le résumé décrit la version 3.0, et le projet de rédaction porte la même date.

## Les métadonnées ECMA-335 sont la base de données du programme

Un assemblage .NET est un fichier PE. `LedgerNet.dll`commence par `MZ`, a trois sections nommées `.text`- Je suis là .`.rsrc`, et `.reloc`, et déclare machine `0x14c`Le répertoire de données 14 de l'en-tête optionnel, l'en-tête de temps d'exécution CLR, pointe vers RVA `0x2008`. L'ECMA-335 II.25.3.3 définit les 72 octets qui y sont:

```text
48 00 00 00   Cb 72
02 00 05 00   MajorRuntimeVersion 2, MinorRuntimeVersion 5
78 21 00 00   MetaData RVA 0x2178
84 07 00 00   MetaData size 1924
01 00 00 00   Flags: COMIMAGE_FLAGS_ILONLY
05 00 00 06   EntryPointToken 0x06000005, MethodDef row 5, which is Main
```

La racine de métadonnées à RVA `0x2178`commence par la signature `42 53 4a 42`, qui dit:`BSJB`, puis la version 1.1 et la chaîne `v4.0.30319`Cinq en-têtes de flux suivent:`#~`à l' équilibre `0x6c`Alors ...`#Strings`- Je suis là .`#US`- Je suis là .`#GUID`, et `#Blob`Il est également possible de détecter les effets de la pollution de l'air.`#~`Le flux commence par un `HeapSizes`octet, zéro ici, donc chaque chaîne, GUID, et index de la tache est de deux octets de large, et un 64 bits `Valid`masque qui donne le nom des tables présentes.

| Table | Id | Rows | Row size | What the rows say |
|---|---|---|---|---|
| Module | 0x00 | 1 | 10 | the module name and its GUID |
| TypeRef | 0x01 | 21 | 6 | `System.Object`, `System.Console`, 19 more |
| TypeDef | 0x02 | 4 | 14 | `<Module>`, `Ledger`, and two compiler helpers |
| Field | 0x04 | 2 | 6 | `Labels` and a static array initialiser |
| MethodDef | 0x06 | 6 | 14 | the five methods and `.cctor` |
| Param | 0x08 | 5 | 6 | `before`, `after`, `score`, `values`, `score` |
| MemberRef | 0x0a | 16 | 6 | 12 attribute constructors and the 4 methods that `Main` calls |
| CustomAttribute | 0x0c | 12 | 6 | attributes the compiler and the SDK add |
| StandAloneSig | 0x11 | 3 | 2 | the three local variable signatures |
| Assembly, AssemblyRef, ClassLayout, FieldRva, NestedClass | | 1, 2, 1, 1, 1 | | identity, references, layout |

Une taille de rangée n'est pas stockée nulle part. II.22 donne à chaque colonne un type, et II.24.2.6 donne la règle de taille.`HeapSizes`. Un index de table simple est de deux octets alors que cette table comporte moins de 65536 lignes. Un index codé pouvant indiquer n tables avec b tag bits est de deux octets alors que chacune de ces tables comporte moins de 2^(16-b) lignes. `CliAssembly`Le système.Reflection.Dépôt de métadonnées dans les appareils rapporte les mêmes tailles et les mêmes tableaux de compensation, ce qui est la façon dont le test prouve l'arithmétique.

III.1.9 définit le jeton: le byte haut est la table, les trois bytes bas sont la ligne basée sur 1. `0x06000002`est `MethodDef`la ligne 2. Cette ligne contient RVA `0x2060`, nom `Grade`, et une tache de signature .`00 01 08 08`: convention d'appel par défaut, un paramètre, retour `int32`, paramètre `int32`- C' est ...`ParamList`points à la ligne 3 de `Param`, dont le nom est `score`La table Param est la raison pour laquelle ILSpy peut imprimer`Grade(int score)`- Il a été démonté.`0x2060`le corps de la méthode commence par une tête de graisse:

```text
13 30         flags 0x13: fat, init locals; size 3 dwords
02 00         MaxStack 2
2d 00 00 00   CodeSize 45
01 00 00 11   LocalVarSigTok 0x11000001
```

`StandAloneSig`la ligne 1 est retenue `07 01 08`Le numéro de la liste:`LOCAL_SIG`, un local, `int32`. Le local a un type et aucun nom. ECMA-335 conserve les noms locaux dans le format de débogage d'interchange de Partition V, et le .NET SDK les écrit à un PDB. Ce dispositif a été construit avec `DebugType`Aucune, donc aucune table et aucun dossier ne porte le nom.`bucket`- Je suis là .`PercentChange`a une petite en-tête à la place: son premier octet `3a`est `0x0e << 2 | 2`, un corps de 14 bytes sans habitants et une pile par défaut de 8.

Les 45 octets de code de `Grade`décodez avec la table des codes optiques de la partie III. `switch`est opcode `0x45`suivie d'une `u32`Compte et beaucoup d' autres .`int32`objectifs relatifs à l'instruction après le tableau:

```text
IL_0000: 02           ldarg.0
IL_0001: 1f 0a        ldc.i4.s 10
IL_0003: 5b           div
IL_0004: 0a           stloc.0
IL_0005: 06           ldloc.0
IL_0006: 1c           ldc.i4.6
IL_0007: 59           sub
IL_0008: 45 05 00 00 00 08 00 00 00 06 00 00 00 04 00 00 00 02 00 00 00 02 00 00 00
                      switch (IL_0029, IL_0027, IL_0025, IL_0023, IL_0023)
IL_0021: 2b 08        br.s IL_002b
IL_0023: 1a 2a        ldc.i4.4, ret
```

Le compilateur a soustrait 6 avant le `switch`, donc le cas 6 est la première cible.`disassemble_cil()`rend les jetons à travers les tables: `ldsfld Labels`- Je suis là .`call Grade`- Je suis là .`call System.Console::WriteLine`, et `ldstr ' '`de la `#US`L'oracle de tout ça est`MetaDump`, un court programme C# dans les appareils construits sur `System.Reflection.Metadata`Il imprime le nombre de lignes de chaque table, la taille et l'offset, le RVA de chaque méthode, la signature, les noms des paramètres et les octets IL, et les 226 champs de `System.Reflection.Emit.OpCodes`- Je suis là .`compare_with_metadump()`Le tableau de l'heure d'exécution comporte 218 codes plus huit`prefix`La table de cours comporte 219 entrées car elle inclut:`no.`de III.2.2, que le tableau des émissions n'a jamais eu.

Les outils pour le format: ILSpy 11.0 (MIT, publié 2026-08-12) fonctionne sur macOS via sa mise en page de bureau Avalonia. `ilspycmd`11.1.0.9782 imprimées IL avec `-il`et sélectionne une méthode par token avec `-m 0x06000002`. dnSpyEx 6.6.0 continue dnSpy sur Windows seulement.`ildasm`Les données métadonnées sont également lues par radar2 6.2.4: `rabin2 -I`dit `arch cil`et `class PE32`, et `is`Liste `Ledger.Grade(int:score):int`Il n'a pas de démonteur CIL dans cette construction. Ce dispositif a été construit avec .NET SDK 8.0.205 parce que c'est le SDK installé sur la machine de construction. .NET 10 est en cours depuis 2025-11-11 et écrit les mêmes structures ECMA-335 .

## Ce que chaque conteneur conserve

Je vous ai mis`grade`Chaque rangée provient de`python3 code/main.py keeps`, qui fait passer les lecteurs sur les appareils:

| Container | Body bytes | Function name | Parameter names | Parameter and return types | Local names | Line numbers | Control flow |
|---|---|---|---|---|---|---|---|
| ELF object, clang -O0 | 118 | `grade` in `.symtab` | none | none | none | none, no `.debug_*` sections | 5 jumps and 1 indirect jump through a relocation table |
| CPython 3.14 pyc | 96 | `co_qualname` | `score` | none | `bucket` | `co_linetable`, first line 10 | 4 compare and `POP_JUMP_IF_FALSE` pairs |
| JVM class, `javac -g` | 50 | constant pool | `score` in `LocalVariableTable` | `(I)I` | none beyond the parameter | `LineNumberTable`, 6 rows | 1 `tableswitch` |
| WebAssembly | 63 | Export section | only with a name section | `(i32) -> i32` | only with a name section | none | 6 nested blocks, 1 `br_table`, 4 `br` |
| .NET assembly | 45 | `MethodDef` | `score` in `Param` | `static int32(int32)` | none, types only in `StandAloneSig` | none without a PDB | 1 `switch`, 1 `br.s` |

Le tableau explique les résultats de décompilation que vous verrez.`int total(const int *, int)`avec le `int total(long param_1, uint param_2)`Le code natif ne conserve aucun de ces colonnes. Un pyc conserve tous les noms et aucun type, et son flux de contrôle est des sauts dont la forme change à chaque sortie, de sorte que les décompilateurs Python se brisent par version et récupèrent les noms gratuitement.`-g`est en marche et exprime le`switch`Un module de Wasm maintient la structure par construction, donc `wasm2wat`Un assemblage .NET conserve la description la plus complète, car les tables de métadonnées sont normatives, et nécessite toujours un PDB pour le nom d'un local.

La table indique avec quoi le décompilateur a dû travailler, et quelles de ses sorties copient un fait stocké et qui en déduit une. Un nom de paramètre dans la sortie ILSpy est une copie. Un nom local dans la même sortie est une inférence à moins qu'un PDB ne soit présent.

## Faites-le

Exécutez la démo:

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats
python3 code/main.py
```

La sortie est un rapport JSON. Pour chaque pyc il donne les deux en-têtes, le fichier et `grade`Les données de base sont les données de chaque classe, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base, les données de base et les données de base.`javap`Pour chaque construction de Wasm, il donne des sections, des types, des fonctions, des contenus de la section nominale et les`wasm-objdump`Il donne pour l'assemblage l'en-tête CLI, les flux, le nombre de lignes de table, les méthodes et l'accord System.Reflection.Metadata. Il se termine par la table de conservation et la liste des interprètes trouvés sur `PATH`- Je suis désolé .

Lire `code/main.py`dans cet ordre:

1. `Reader`détient le curseur et les largeurs entières. `uleb()`et `sleb()`sont les lecteurs LEB128 dont Wasm et le .NET ont besoin.
2. `read_pyc_header()`- Je suis là .`Unmarshaller`, et `PycFile`Je lis un pyc.`parse_dis_text()`Il lit les prisonniers .`dis`les listes afin que la comparaison entre versions ne nécessite pas quatre interprètes au moment de l'essai.
3. `ClassFile`lisent le fichier de classe et `disassemble_jvm()`décode son code. `parse_javap_code()`et `compare_with_javap()`sont les vérifications oracle.
4. `WasmModule`Il lit les sections et `disassemble_wasm()`décode une expression avec sa profondeur. `compare_with_wasm_objdump()`C'est le contrôle de l'oracle.
5. `CliAssembly`marche PE, en-tête CLI, racine de métadonnées, flux et tables. `TABLE_SCHEMAS`et `CODED_INDEXES`sont II.22 et II.24.2.6 comme données. `disassemble_cil()`décode un corps et `compare_with_metadump()`Il le vérifie.
6. `NativeObject`Lire suffisamment ELF pour énumérer les symboles de fonction et prendre la leçon 33 comme texte. `what_each_keeps()`et `figure_data()`construire la comparaison.

Il y a cinq commandes de plus.`python3 code/main.py pyc FILE`- Je suis là .`class FILE`- Je suis là .`wasm FILE`, et `dll FILE`imprimer la structure et une liste de chaque fonction dans un fichier que vous leur donnez. `python3 code/main.py compare`compilations `ledger.py`avec chaque interprète sur .`PATH`et imprime ce que chacun a écrit.

## Utilisez-le

Utilisez les lecteurs dans cet ordre sur un conteneur que vous avez construit.

1. Lisez les premiers octets et nommez le format.`0d 0a`est un pyc, `ca fe ba be`un dossier de classe, `00 61 73 6d`un module, `4d 5a`avec l'annuaire de données 14 un ensemble de configuration.
2. Lisez l'en-tête et écrivez la version de producteur qu'il nomme. Pour un pyc qui est le mot magique, pour un fichier de classe la version principale, pour un assemblage les chaînes de version de temps d'exécution et de métadonnées.
3. Vérifiez chaque décompiler que vous prévoyez d'exécuter contre cette version avant de l'exécuter.
4. Exécutez le lecteur et l'outil du format sur le même fichier et comparez-les rangée par rangée. Un déséquilibre est un bug dans le lecteur ou un changement dans l'outil, et l'un ou l'autre doit être compris avant que la liste ne soit une preuve.
5. Remplissez la table de conservation pour la fonction qui vous intéresse, marquez chaque fait comme conservé ou perdu et nommez la table, l'attribut ou la section d'où il vient.
6. Il suffit de lire la sortie du décompilateur, de marquer chaque nom et de le taper comme une copie d'un fait conservé ou comme une inférence.

Gardez les fichiers texte oracle avec les conteneurs. Lorsque l'interprète, JDK, SDK ou wabt version change, reconstruire avec `build_fixtures.sh`et de recapture, parce qu'une liste modifiée est un fichier modifié.

## Laboratoire interactif

### Laboratoire A: lire l'en-tête, puis l'objet du code

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats/code
python3 -q
```

Je vais courir .

```python
from main import *
pyc = load_pyc("ledger.cpython-311.pyc")
pyc.header.to_dict()
pyc.module.names
pyc.functions()["grade"].varnames
len(pyc.functions()["grade"].code)
```

Écrivez le mot magique avant le premier appel.`ledger.cpython-314.pyc`et comparer les deux `code`Alors essayez.`import marshal; marshal.loads(pyc.data[16:])`sur le fichier 3.11 sous votre interprète et lire l'erreur.

### Laboratoire B: la chaîne de piscine constante

```python
cls = load_class("Ledger-g.class")
cls.constant(13)
cls.constant(14)
cls.describe_constant(13)
[row["text"] for row in jvm_listing(cls, "grade")]
```

Suivez`#13`à `#8`et `#14`, puis à la `Utf8`Les inscriptions, à la main avant d'appeler `describe_constant`- Alors chargez .`Ledger-gnone.class`et appeler`cls.method("grade").attribute_names()`Les deux.`Code`L'ensemble est le même, les attributs ne le sont pas.

### Laboratoire C: profondeur et étiquettes

```python
module = load_wasm("ledger.wasm")
[(row["depth"], row["text"]) for row in wasm_listing(module, "grade")]
module.function_name(2)
load_wasm("ledger-names.wasm").names["locals"][2]
```

Trouvez le `br_table`Résumez la profondeur de la ligne.`end`les rangées après elle et correspondent chacune à un numéro d'étiquette dans le tableau.`ledger.wat`et trouver le `block $six`Cette étiquette 3 noms.

### Laboratoire D: des jetons aux octets

```python
asm = load_assembly("LedgerNet.dll")
asm.method("Grade")
body = asm.method_body(asm.method("Grade")["RVA"])
body.fat, body.max_stack, body.code_size, hex(body.local_signature_token)
asm.token_text(0x11000001)
[row["text"] for row in cil_listing(asm, "GradeLabel")]
```

Compute le décalage de fichier de RVA `0x2060`de la `.text`en-tête de section à la main et comparer avec `asm.rva_to_offset(0x2060)`Alors regardez en haut .`asm.row(0x08, 3)`et trouver où le nom `score`Il vit.

## Laboratoire de pratique

Ajoutez une cinquième langue source à la comparaison, ou un sixième fait au tableau.

Les exigences:

- Écrivez une fonction avec une boucle et une variable locale dans une langue dont vous avez le compilateur, par exemple Kotlin à un fichier de classe ou Rust à un module Wasm. Commencez la sortie et la liste oracle avec la commande exacte en `build_fixtures.sh`- Je suis désolé .
- Faites en sorte que le lecteur correspondant soit d'accord avec l'oracle sur chaque instruction.
- Extension `what_each_keeps()`avec le nouveau conteneur, et avec un autre fait pour chaque conteneur: si le corps porte une table d'exception ou `try`région.
- Ajoutez le nouveau conteneur à `figure_data()`et régénérer `site/figures-re-43.js`- Je suis désolé .
- Écrivez un test qui échoue lorsqu'un lecteur rapporte un fait tel qu'il est conservé sans nommer la table, l'attribut ou la section qui le contient.

Le succès signifie`python3 code/main.py keeps`montre la nouvelle rangée avec des preuves pour chaque cellule, et chaque comparaison oracle est toujours d'accord.

## Leçon d'artisanat

Cette leçon comprend `outputs/skill-bytecode-and-managed-formats.md`. Utilisez-le pour transformer un conteneur de code octal en rapport d'en-tête, une liste vérifiée d'une fonction et une table de faits qui marque chaque nom, type et structure comme conservé ou perdu. Il nécessite un outil du format comme oracle, une vérification de version avant tout décompilateur et une déclaration de l'endroit où chaque fait perdu aurait vécu.

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La vérification doit prouver:

- Les quatre en-têtes pyc portent les mots magiques 3495, 3531, 3571 et 3627, les drapeaux 3, et un hash de 8 octets, et les fichiers timestamp portent des drapeaux 0, mtime 1791975565, et la taille 497
- chaque mot magique pour 3.11 et plus tard se trouve dans la gamme de la `2900 + 50n`la règle donne sa libération
- le démarcheur renvoie les mêmes noms, constantes, octets de code, noms de variables et table de lignes que `marshal.loads()`sur le fichier que l'interprète en cours d'exécution a écrit
- Les quatre fichiers conservent les noms des modules,`('score', 'bucket')`pour `grade`, première ligne 10, et `ledger.py`comme nom de fichier
- `grade.co_code`est 80, 62, 70 et 96 octets dans les quatre versions et les listes capturées ont 31, 26, 26 et 35 instructions
- La liste 3.11 a `POP_JUMP_FORWARD_IF_FALSE`, 3.12 et 3.13 ont été`RETURN_CONST`, et 3.14 a été .`LOAD_SMALL_INT`- Je suis là .`LOAD_FAST_BORROW`, et `NOT_TAKEN`
- le 3.14 `-C -O`la liste montre 13 entrées de cache dans `grade`, cinq d' eux après `BINARY_OP`, et les émissions de l' annonce adaptative `COMPARE_OP_INT`
- une tête sans `0d 0a`, une tête tronquée, et un corps qui n'est pas un objet de code lever `FormatError`
- une liste d'instructions n' existe que pour le pyc écrit par l' interprète en cours d' exécution et correspond à celui capturé `dis`dégagement
- les fichiers de classe ont des majors 65 ou 69, des tailles 1581, 1832, 1404 et 1581, et les nombres de la pile constante 82, 94, 79 et 82
- constantes `#13`résolue à `Method Ledger.grade:(I)I`et `#47`à `String F`
- le démonteur de la VVM est d' accord avec `javap -c -p`sur 108 des 108 instructions dans les quatre constructions
- `tableswitch`sur pc 4 est de 36 octets avec faible 6, haut 10, 48 par défaut et cibles 46, 44, 42, 40, 40
- Les trois niveaux de débogage changent le nombre d' attributs et laissent les 50 octets de code de `grade`identique
- `ifne`- Je suis là .`if_icmpge`, et `goto`les objectifs sont relatifs au code optique de la branche
- une mauvaise magie, des octets en retard et une augmentation indéfinie du code optique.`FormatError`
- Les sept sections de Wasm ont des compensations de charge utile `wasm-objdump -h`Les empreintes, et les noms construire ajoute un `name`section douanière à 293 à 425
- Les types, les `env.log`Les données de l'importation, les cinq exportations et le segment des données au décode 1024 comme écrit dans `ledger.wat`
- Le désassembleur de Wasm est d' accord avec `wasm-objdump -d`sur 77 des 77 instructions dans les deux constructions
- `grade`a 6 blocs en nichage, un `br_table`avec des étiquettes `4 4 4 4 4 4 3 2 1 0 0`et par défaut 4 à la profondeur 6, et un `i32`locaux
- Les noms de fonctionnalités proviennent des exportations sans section nom et des noms locaux uniquement avec elle
- LEB128 `e4 00`est de 100, `7f`est -1, `80 08`est 1024, et les types de blocs se rendent comme `wasm-objdump`
- une préamble erronée, version 2, et une augmentation inconnue du code optique `FormatError`
- l' assemblage est PE32 avec machine `0x14c`, en-tête CLI à RVA `0x2008`, temps de fonctionnement 2,5, `ILONLY`, et point d'entrée `0x06000005`
- la version des métadonnées est `v4.0.30319`Il y a cinq courants, et les 14 tableaux ont le nombre de rangées et les tailles de rangées que l'oracle rapporte
- le lecteur .NET est d' accord avec System.Reflection.Metadata sur 246 des 246 contrôles
- `Grade`a un paramètre `score`, signature `static int32(int32)`, une grosse tête avec `MaxStack`2 et 45 octets de code, et un `int32`locaux sans nom
- CIL `switch`cibles, `br.s`- Je suis là .`ldsfld`- Je suis là .`call`, et `ldstr`Les opérands se résolvent à travers les tables et les amas
- Le tableau des codes optiques CIL couvre tous les codes optiques en cours d' exécution et ne ajoute que `no.`
- le tableau de garde rapporte le nom du paramètre dans quatre conteneurs et dans aucun pour le code natif, et les types dans trois
- chaque ligne oracle dans la démo est n de n
- la figure interactive intègre les mêmes données `figure_data()`produit

## Mode de défaillance de la production

| Failure | What the weak tool reports | What the analysis must prove |
|---|---|---|
| Decompiler older than the interpreter | `Unsupported opcode`, output marked incomplete, or a claim of obfuscation | The magic word, the release it names, and the tool's documented ceiling |
| pyc read with the wrong interpreter | `marshal.loads` raises or returns garbage | Our unmarshaller reads names, constants, and sizes from any 3.9 to 3.15 file, and instructions come from the matching `dis` |
| Specialised opcodes mistaken for the file's content | `COMPARE_OP_INT` in a listing | A pyc holds the unspecialised form, specialisation happens in memory after warm-up |
| Local names reported for a default javac build | `var1`, `var2` presented as source names | `LocalVariableTable` is present only with `-g`, the 50 code bytes are identical either way |
| Branch target off by the instruction length | A JVM listing whose jumps land mid-instruction | Offsets relative to the branch opcode, checked against `javap` |
| Switch padding skipped | A `tableswitch` with a wrong default and shifted cases | Zero to three padding bytes to a 4-aligned pc, 36 bytes for five cases |
| Wasm names assumed stripped | "no symbols" | Exports name every exported function, locals need the `name` section, `wasm-strip` removes only that section |
| Three tools, three spellings | `get_local` versus `local.get`, a function at `0x8b` versus `0x8e` | Same bytes, and the listing that the comparison was captured with |
| Table row size guessed | Every row after the first table misread | Row sizes from `HeapSizes`, row counts, and coded index widths, equal to the oracle's |
| Local names expected in a release assembly | `V_0` presented as the source name | Locals have types in `StandAloneSig` and names only in a PDB that `DebugType` none did not write |
| Reader guesses an unknown opcode or type code | A plausible mnemonic | `FormatError` with the offset, and the oracle comparison stops there |
| Oracle captured from another build | Tests pass on stale text | `build_fixtures.sh` names every tool version and writes `versions.txt` and `sha256.txt` |

## Règlement d'exploitation

Lisez d'abord l'en-tête et nommez le producteur à qui il pointe. Lisez le contenant avec un lecteur qui refuse ce qu'il ne sait pas, et vérifiez sa liste par rapport à l'outil du format dans le même fichier. Puis dites, pour chaque nom et type dans la sortie d'un décompiler, que le contenant l'ait stocké ou que l'outil l'ait déduit.

## Pour en savoir plus

- [PEP 552: Deterministic pycs](https://peps.python.org/pep-0552/)
- [CPython `Include/internal/pycore_magic_number.h`](https://github.com/python/cpython/blob/main/Include/internal/pycore_magic_number.h)
- [Python `marshal` module documentation](https://docs.python.org/3/library/marshal.html)
- [CPython `Python/marshal.c`](https://github.com/python/cpython/blob/main/Python/marshal.c)
- [Python 3.14 `dis` module documentation](https://docs.python.org/3.14/library/dis.html)
- [PEP 659: Specializing Adaptive Interpreter](https://peps.python.org/pep-0659/)
- [PEP 626: Precise line numbers for debugging and other tools](https://peps.python.org/pep-0626/)
- [PEP 790: Python 3.15 Release Schedule](https://peps.python.org/pep-0790/)
- [uncompyle6 on PyPI](https://pypi.org/project/uncompyle6/)- Je suis là .[decompyle3 on PyPI](https://pypi.org/project/decompyle3/)- Je suis là .[xdis on PyPI](https://pypi.org/project/xdis/)- Je suis là .[pycdc on GitHub](https://github.com/zrax/pycdc)- Je suis là .[PyLingual on GitHub](https://github.com/syssec-utd/pylingual)
- [The Java Virtual Machine Specification, Java SE 25, Chapter 4: The class File Format](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-4.html)
- [The Java Virtual Machine Specification, Java SE 25, Chapter 6: The Java Virtual Machine Instruction Set](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-6.html)
- [javac manual, Java SE 25](https://docs.oracle.com/en/java/javase/25/docs/specs/man/javac.html)et [javap manual](https://docs.oracle.com/en/java/javase/25/docs/specs/man/javap.html)
- [CFR](https://www.benf.org/other/cfr/)- Je suis là .[Procyon](https://github.com/mstrobel/procyon)- Je suis là .[Vineflower](https://github.com/Vineflower/vineflower)- Je suis là .[Fernflower in intellij-community](https://github.com/JetBrains/intellij-community/tree/master/plugins/java-decompiler/engine)- Je suis là .[JADX](https://github.com/skylot/jadx)
- [Harrand, Soto-Valero, Monperrus, Baudry, "Java Decompiler Diversity and its Application to Meta-decompilation" (JSS 2020)](https://arxiv.org/abs/2005.11315)
- [ECMA-335, Common Language Infrastructure, 6th edition (June 2012)](https://ecma-international.org/publications-and-standards/standards/ecma-335/)
- [PE Format, Microsoft Learn](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format)
- [System.Reflection.Metadata.Ecma335.TableIndex](https://learn.microsoft.com/en-us/dotnet/api/system.reflection.metadata.ecma335.tableindex)
- [ILSpy on GitHub](https://github.com/icsharpcode/ILSpy)- Je suis là .[ilspycmd on NuGet](https://www.nuget.org/packages/ilspycmd)- Je suis là .[dnSpyEx on GitHub](https://github.com/dnSpyEx/dnSpy)
- [WebAssembly Core Specification: Binary Format, Modules](https://webassembly.github.io/spec/core/binary/modules.html)et [Values](https://webassembly.github.io/spec/core/binary/values.html)
- [WebAssembly Core Specification: Custom Sections (name section)](https://webassembly.github.io/spec/core/appendix/custom.html)
- [WebAssembly tool-conventions: Debugging](https://github.com/WebAssembly/tool-conventions/blob/main/Debugging.md)
- [wabt 1.0.42 release notes](https://github.com/WebAssembly/wabt/releases/tag/1.0.42)- Je suis là .[wasm-tools](https://github.com/bytecodealliance/wasm-tools)- Je suis là .[Binaryen](https://github.com/WebAssembly/binaryen)- Je suis là .[ghidra-wasm-plugin](https://github.com/nneonneo/ghidra-wasm-plugin)
- [radare2 on GitHub](https://github.com/radareorg/radare2), version 6.2.4 construite à partir de la source pour cette leçon
