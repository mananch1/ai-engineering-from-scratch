# Diffusion binaire et exécution symbolique: correspondance des fonctions et résolution des entrées

> Une correspondance entre deux fonctions est une revendication faite par une signature avec une qualité, et une entrée résolue est un modèle d'une condition de chemin; cette leçon construit les deux et les vérifie contre radiff2, Ghidra, angr et z3.

**Type:** Build
**Languages:** Python, C, x86-64 assembly, Java (one Ghidra script)
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), 34 (lifting to pseudocode), 36 (rebuild and prove equal)
**Time:** ~150 minutes

## Objectifs d'apprentissage

- Construisez cinq signatures de fonction à partir d'un objet et de sa liste: octets exacts, octets avec champs de relocalisation masqués, séquence mnemonique, forme de flux-graphe et nom du symbole.
- Exécutez un match avide en scène dans le style BinDiff, où une signature ne correspond que lorsqu'elle est unique des deux côtés, et enregistrez l'étape et la qualité derrière chaque match.
- Reproduire le cas où deux fonctions identiques en octets ne parviennent pas à obtenir un corrélateur en octets sur les images chargées et le transmettre après le masquage, avec l'arithmétique de relocalisation qui explique les quatre octets modifiés.
- Reproduire les ratios de similitude de radiff2 à partir de la distance de Levenshtein et expliquer pourquoi un ratio n'apparaît que lorsque la comparaison utilise les 82 octets de radar2 atteints.
- Enumérez les chemins d'une fonction sur une petite IR, portez une condition par chemin et résolvez chaque feuille pour les entrées avec une recherche limitée.
- Mesurer l'explosion de la voie sur une boucle avec une limite symbolique et comparer les nombres, les classes et les modèles avec angr 10.0.3 et z3 5.1.0.

## Le problème

Vous avez reconstruit .`ledger.c`à`-O2`et a demandé à un agent si quelque chose avait changé depuis le`-O1`L'agent a fait un diff et a rapporté que`grade`Il a ensuite dépensé son budget pour expliquer un changement qui n'existe pas.`grade`Les images comparées après le relocalisation diffèrent.`.rodata`est situé à 0x70 octets plus loin dans le `-O2`programme, donc un déplacement dans `grade`Il a augmenté de 0x70.

Le même après-midi , vous m' avez demandé quel score .`grade`Le code est "90". C'est une réponse. La fonction renvoie 4 pour chaque score de 90 à 109, et la question n'avait pas de limite, donc la réponse de l'agent était un échantillon où vous avez besoin d'un ensemble.

Les deux défaillances proviennent du même écart. Une correspondance et une entrée résolue sont chacune une revendication avec une méthode derrière elle. Cette leçon construit les deux méthodes dans la bibliothèque standard Python et exécute les outils de production à côté d'eux. Là où les outils et le code conviennent, vous avez deux dérivées. Là où ils ne sont pas d'accord, vous avez une adresse, un octet ou un lien à regarder.

```figure
re-44-matches-and-paths
```

## Une signature de fonction est une revendication de qualité

Flake a cadré la différence binaire en 2004 comme la construction d'un isomorphisme entre les ensembles de fonctions de deux exécutables similaires. Dullien et Rolles l'ont étendu en 2005 à des blocs et des instructions de base. La page des concepts de BinDiff décrit la stratégie qui est issue de ce travail. Compute un attribut pour chaque fonction des deux côtés. Une valeur qui apparaît exactement une fois de chaque côté est un match. Une valeur qui apparaît plus d'une fois est ambiguë et est perçue avec l'attribut suivant à l'intérieur de cette classe d'équivalence. Une valeur sans contrepartie reste inégalée. Après chaque passage, les appelants et les appelés de nouveaux matchs forment de petits ensembles où l'attribut suivant est susceptible d'être unique, et le processus se répète avec l'attribut suivant.

Chaque algorithme de correspondance est caractérisé par une qualité. Les pages évaluent le hash de correspondance des octets de fonction brute et le hash de nom comme très bon, l'indice MD du graphique de flux et le parallèle de la signature primaire (chaque mnemonique un petit prime, produit sur la fonction) comme bon, les références de chaîne comme moyen et le nombre de boucles ou la séquence d'adresses comme pauvre. La confiance d'une correspondance est la qualité moyenne des algorithmes qui l'ont produite, pondérée par un sigmoïde. La similitude de fonction est d'environ 25% des bords du graphique de flux correspondant, 15% des blocs de base correspondant, 10% des instructions correspondantes et 50% de la différence dans l'indice MD du graphique de flux, multiplié par la confiance.

Cette leçon conserve cinq de ces attributs et la seule règle qui compte: unique des deux côtés ou pas de correspondance. Il laisse tomber l'index MD, parce que les pages de concepts le nomment sans formule, et utilise une séquence de degrés simples à sa place. Les étiquettes de qualité restent, parce qu'un correspondance est seulement aussi bonne que le fait le plus faible qui l'a produit.

## Cinq signatures d'un seul enregistrement

Le différent donne deux choses par objet.`.rela.*`Les archives.`llvm-objdump -d`La liste donne des limites d'instruction et des mnemoniques, et la leçon 33 a prouvé un décodeur égal à cette liste sur chaque instruction de ces objets. `function_record()`Vérifie que les octets de chaque instruction répertoriée sont égaux aux octets de l'objet à cette adresse et augmente autrement.

Prenez .`grade`dans `ledger-x86_64-O1.o`. Le symbole dit de décompte 0x10 et de taille 46, donc le record couvre 0x10 à 0x3e et exclut les deux bytes `nop`qui aligne la fonction suivante.

La signature de bytes exactes est la SHA-256 des 46 bytes. La signature de bytes masquées zéro chaque champ de relocalisation d'abord. Il y a un champ: le record`R_X86_64_PC32 .rodata+0x10`à l'écart de 0x36, quatre octets de largeur, le déplacement de l'`leaq (%rip), %rcx`Dans un objet RELA ces quatre octets tiennent déjà zéro, donc les deux signatures sont d'accord sur chaque fonction de chaque objet ici.

La signature mnemonique hashes la séquence de 14 mnemoniques:

```text
movslq imulq movq sarq shrq addl addl xorl cmpl ja movl leaq movl retq
```

La signature de forme du graphique de flux provient des blocs de base et des bords taillés de la leçon 33, limités aux blocs auxquels l'entrée peut atteindre. `grade`à`-O1`Il a trois blocs et trois bords.`ja 0x3d`Avec un bord pris et un bord tombé, le bloc 0x31 passe à 0x3d, et 0x3d revient.`3b 3e [0/2,1/1,2/0]`Pour ...`grade`à`-O0`Le dossier résolve le saut indirect à travers les cinq `.rela.rodata`Les entrées sont exactement comme la leçon 33, ce qui donne huit blocs et onze bords avec quatre `table`Les bords.

La signature du nom est le nom du symbole, c'est le fait le plus fort quand il existe et rien quand la construction est dépouillée.

Un champ supplémentaire s'allonge pour le boîtier de patch plus tard.`lea`réinstallation dans `.rodata`, une vérification des limites, et une charge indexée,`referenced_table()`La table est lue comme l' a fait la leçon 36:`cmpl $0x4`avant `ja`donne cinq entrées et la charge `movl (%rcx,%rax,4), %eax`donne quatre octets chacun.`grade`à`-O1`la table à `.rodata+0x14`Il lit`1, 2, 3, 4, 4`- Je suis désolé .

## Le coiffeur avide

`match_functions()`Il regroupe les fonctions encore inégalées de chaque côté par la signature de cette étape et accepte chaque valeur qui a exactement une fonction de chaque côté.

```python
for stage in order:
    for signature, candidates in left_groups.items():
        partners = right_groups.get(signature, [])
        if len(candidates) != 1 or len(partners) != 1:
            continue
        a, b = candidates[0], partners[0]
        agreeing = [s.id for s in STAGES if a.signatures()[s.id] == b.signatures()[s.id]]
        matches.append(Match(a, b, stage, agreeing, radare2_ratio(a.raw, b.raw), data_equal))
```

L'ordre des leçons passe d'abord par les quatre étapes structurelles et les noms par la suite, de sorte que les étapes structurelles sont testées et les noms les classent ensuite.`-O1`contre `-O2`- Le numéro de la liste:

| Function | Stage | Agreeing signatures | Similarity |
|---|---|---|---|
| `percent_change` | exact bytes | all five | 1.000 |
| `grade` | exact bytes | all five | 1.000 |
| `total` | name | name only | 0.171 |
| `grade_label` | exact bytes | all five | 1.000 |

`total`Le clang vectorize la boucle avec SSE2 et maintient une boucle restante escalar. Sa forme est passée de cinq blocs et cinq bords à dix blocs et treize bords. Rien de structural n'a survécu, et seul le symbole relie les deux.

Pour `-O0`contre `-O1`le même ordre produit une correspondance erronée:

| Function at -O0 | Matched at -O1 | Stage | Names agree |
|---|---|---|---|
| `percent_change` | `grade_label` | flow-graph shape | no |
| `grade` | `grade` | name | yes |
| `total` | `total` | name | yes |
| `grade_label` | unmatched | | |

`percent_change`à`-O0`est un diamant à quatre blocs: une entrée qui se ramifie sur `before == 0`La séquence de degrés est de`0/2, 1/1, 1/1, 2/0`- Je suis là .`grade_label`à`-O1`Il a`grade`La forme est unique des deux côtés et la règle l'accepte. L'étiquette de qualité moyenne est l'avertissement, et le nom de l'étape, exécuté après, contredit.`grade_label`à droite, donc la vraie `percent_change`et `grade_label`Les paires restent inégalées.

L'ordre BinDiff exécute la correspondance de noms comme deuxième étape où les noms réels existent, et `BINDIFF_ORDER`Dans le code, il fait la même chose.`-O0`Les fonctions correspondent à leurs `-O1`Les deux commandes sont conservées dans la démo, de sorte que l'échec est visible.`if`et `else`qui rejoint avant le retour.

## Des octets identiques, des opérations réaffectées

Ghidra applique des relocations lorsqu'il importe un objet. Son corrélateur en octets compare donc les images chargées et les champs de relocalisation masqués pour cette raison. `load_image()`Il reproduit l'importation. Il passe les sections allouées dans l'ordre des en-têtes, aligne chacune sur sa `sh_addralign`, et les détermine à partir de 0x100000.`S + A - P`écrit sous forme de valeur signée de 32 bits pour `R_X86_64_PC32`et `R_X86_64_PLT32`- Je suis là .`S + A`en 64 bits pour `R_X86_64_64`- Je suis désolé .

La mise en page est conforme à Ghidra 12.1.4 sur chaque section démarrée dans les deux programmes, que la suite de test vérifie contre la sortie de suivi de version capturée:

| Section | `-O1` program | `-O2` program |
|---|---|---|
| `.text` | 0x100000, 159 bytes | 0x100000, 271 bytes |
| `.rodata` | 0x1000a0 | 0x100110 |
| `.rodata.str1.1` | 0x1000f0 | 0x100160 |

`.text`est 112 octets plus long à `-O2`, la croissance de `total`Je suis un homme .`.rodata`commence par 0x70 octets plus loin. `grade`Il est à 0x100010 dans les deux programmes.`leaq`à l'offset 0x33 a son champ de déplacement à 0x36 et se termine à 0x3a.`.rodata+0x14`- Je suis désolé .

Dans le `-O1`image: `0x1000a0 + 0x14 - 0x10003a = 0x7a = 122`Dans le...`-O2`image: `0x100110 + 0x14 - 0x10003a = 0xea = 234`Les octets de champ sont `7a 00 00 00`dans une seule image et `ea 00 00 00`Un octet de 46 diffère, le ratio de Levenshtein est de 0,978, et l'étape des octets exacts échoue.`grade_label`Il est déplacé par le même 0x70 que sa table, donc ses déplacements sont restés égaux et il passe des octets exacts dans les deux vues.

Ghidra rapporte la même scission.`-O1`contre `-O2`, Exact fonction par défaut de parcours trouvé `percent_change`et `grade_label`, Exact Function Instructions Match a trouvé ces deux plus `grade`La suite de tests affirme les deux équations.

réa 6.5.0 `compare_functions`Le texte fournisseur de défiles de ligne. sur la dimension d'assemblage il attache la limitation " normalisation de relocalisation est indisponible. " Le chargement et le masque dans cette leçon fournissent cette normalisation.

## Quelles mesures radiff2

Radar2 6.2.4 navires `radiff2`Avec `-AC`Il analyse les fichiers et imprime une rangée par fonction: la taille et l'adresse gauche, MATCH ou UNMATCH, un ratio, et la taille et l'adresse droite.

| Pair | `percent_change` | `grade` | `total` | `grade_label` |
|---|---|---|---|---|
| `-O0` vs `-O1` | UNMATCH 0.130435 | UNMATCH 0.426829 | UNMATCH 0.106061 | UNMATCH 0.269841 |
| `-O1` vs `-O2` | MATCH 1.000000 | MATCH 1.000000 | UNMATCH 0.171429 | MATCH 1.000000 |
| `-O1` vs `-O1` | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 |

Le rapport dans la source de radar2 est `1 - distance / max(len_a, len_b)`, où la distance est la distance de Levenshtein entre les deux empreintes digitales de la fonction et une empreinte digitale est la concaténation des blocs de base de la fonction dans l'ordre d'adresse.`radare2_ratio()`dans `code/main.py`Il implique la même formule sur les octets de fonction brute et reproduit 0,130435, 0,106061, 0,269841, et 0,171429 à six décimales.

Il ne reproduit pas 0,426829 pour `grade`à`-O0`. Sur les 118 octets de la fonction, le ratio est de 0,296610.`118`Il est plus petit, sans déplacement appliqué, il ne peut pas résoudre la table de saut, de sorte que son analyse atteint le bloc d'entrée, le bloc qui se termine en`jmpq *%rax`La distance de Levenshtein entre ces 82 octets et les 46 octets est de`-O1`est de 47, et `1 - 47/82 = 0.426829`Avec `-e bin.relocs.apply=true`Radar2 trouve les 118 octets et imprime 0,288136, ce qui équivaut au rapport de la leçon sur les images relocalisées, où les quatre octets de déplacement diffèrent des champs zéro de l'objet.

`-O`Il est possible de supprimer les empreintes digitales en octets opcode et de réduire les opérands à zéro.`-O0`contre `-O1`Les ratios de la valeur de l'émission de données augmentent à 0,239130, 0,512195, 0,303030 et 0,523810, et `total`à`-O1`contre `-O2`Les distances de fichiers entiers existent aussi: `-s`(Myers) rapporte une similitude de 0,956 et une distance de 162 entre les deux objets.`-ss`Ces chiffres comprennent des en-têtes, des tables de symboles et des sections de relocalisation et ne disent rien sur les fonctions.

## Corrélateurs de suivi de la version Ghidra

L'outil de suivi de version de Ghidra correspond à un programme source à un programme de destination avec des corrélateurs. Les corrélateurs de fonction sont le Match de bytes de fonction exacte, le Match d'instructions de fonction exacte (opérands masqués), le Match de mémoires de fonction exacte et le Match d'instructions de fonction dupliquée (1:N et N:M). Les corrélateurs de données sont exacts, duplicés et correspondants de données similaires. Les corrélateurs de symboles correspondent exactement aux noms, en tant que duplicates ou par similitude. Les corrélateurs de référence utilisent des correspondances acceptées pour en trouver plus. Les fonctions doivent être définies dans le programme de destination et la longueur minimale de la fonction est définie par défaut à 10 octets. Auto Version Tracking exécute le nom exact du symbole, les données exactes, les octets exactes de la fonction, les instructions exactes de la fonction, les méméniques exactes de la fonction, les instructions de la fonction dupliquée, puis un corrélateur de référence, et crée des correspondances implicites pour les acceptés. Un corrélateur BSim a été ajouté à la section 11.2.

La leçon a capturé les corrélateurs sans tête. `capture_ghidra_vt.sh`Les trois objets sont importés dans un projet et sont exécutés `ListVersionTrackingMatches.java`, un court script qui s' étend`AbstractGhidraVersionTrackingScript`, ouvre les programmes source et destination par chemin, exécute les six corrélateurs non-référentiels dans l'ordre de suivi de la version automatique, et imprime une ligne par bloc, fonction, corrélateur et correspondance.

| Correlator | `-O0` to `-O1` | `-O1` to `-O2` |
|---|---|---|
| Exact Symbol Name Match | 4 functions | 4 functions, 5 data items (`labels.rel`, `.L.str.1` to `.L.str.4`) |
| Exact Data Match | 0 | 0 |
| Exact Function Bytes Match | 0 | 2 (`percent_change`, `grade_label`) |
| Exact Function Instructions Match | 0 | 3 (adds `grade`) |
| Exact Function Mnemonics Match | 0 | 3 |
| Duplicate Function Instructions Match | 0 | 0 |

Chaque fonction a changé entre `-O0`et `-O1`La page d'aide décrit les corrélateurs exacts comme la similitude 1.0 et la confiance 1.0, de sorte que l'échelle de la confiance imprimée est une question ouverte que cette leçon enregistre et ne résoud pas.

Les différences de programme, dans Tools, est une chose différente. Il compare deux programmes avec le même espace d'adresse sur les octets, les étiquettes, les unités de code, les références, les commentaires et les fonctions à des adresses égales.

## Diaphore et Ninja binaire, à partir de leur documentation

Diaphora est un plugin IDA Pro, publié pour la première fois en 2015 et à la version 3.4.2 à partir de 2026-08-18. Il est AGPL-3.0 depuis 2.0 et a besoin d'IDAPython, donc il fonctionne dans IDA Home ou IDA Pro et non dans IDA Free. Il exporte chaque base de données à SQLite et exécute ses heuristiques sous forme de requêtes SQL. Chaque heuristique a un nom, une catégorie de Best, Partial ou Non fiable, un type de ratio et des balises facultatives telles que SAME_CPU ou SLOW. Les noms dans la source actuelle se lisent comme les étapes de cette leçon avec plus d'entre eux: hash de bytes, Same address and mnemonics, Same cleaned assembly, Mnemonics small-primes-product, Same nodes, edges, loops and strongly connected components, topological sort hash, and pseudo-code fuzzy hashs that need the Hex-Rays decompiler. Ses notes de sortie montrent l'historique de l'accord: les heuristiques lentes ne fonctionnent que sous environ 2000 fonctions, et "Same graph" est passé à la catégorie peu fiable. Ce mouvement est le diamant de quatre blocs de cette leçon, vu à l'échelle.

Binary Ninja 6.0, sorti en 2026-09-03, a ajouté la similitude binaire comme cadre avec des fournisseurs et des résolveurs. Il envoie BinDiff en tant que fournisseur structurel et WARP pour des correspondances de fonction exactes qui tolèrent des opérands relocalisables, et son guide dit d'utiliser WARP lorsque les fonctions correspondent à part des instructions relocalisables et BinDiff lorsqu'elles restent structurellement liées après la recompilation. La fonctionnalité est dans l'édition Ultimate seulement. Binary Ninja n'a pas de gestionnaire de paquets installé ici et n'a pas été exécuté, donc ce paragraphe provient de sa documentation et rien d'autre.

BinDiff lui-même est open source depuis 2023-09-25 sous Apache-2.0, avec la dernière version v8. Il a besoin d'un démonteur et du plugin BinExport, qui prend en charge IDA Pro 9.4 ou ultérieur, Binary Ninja et Ghidra. Le chemin libre pour un apprenant est Ghidra plus BinExport plus la ligne de commande BinDiff. La leçon ne l'a pas exécuté, et son Practice Lab vous le demande.

## Le même code, une constante différente

La leçon 36 a permis de maintenir une réorganisation de la candidature .`grade`dont la table comporte 3 comme dernière entrée, où l'original comporte 4. `grade_case10_wrong.o`a les mêmes 46 octets de code que `grade`à`-O1`Le matcher les associe à des octets exacts avec chaque signature d'accord.`1, 2, 3, 4, 4`par un seul déménagement et `1, 2, 3, 4, 3`par l'autre et les rapports `dataEqual: false`Les trois autres .`-O1`Les fonctions sont inégalées à gauche, car l'objet candidat possède une fonction.

La fonction différente qui s'arrête à bytes ne dit rien changé. Le match exact de données de Ghidra et le match de référence de chaîne de BinDiff existent pour la même raison. La version de la leçon est petite: pour chaque paire correspondante, comparez les bytes où chaque fonction se déplace, avec le compte des limites et la largeur de la charge.

## Exécution symbolique: chemins, conditions, résolutions

King décrit l'exécution symbolique en 1976. Les entrées deviennent des symboles. Chaque déclaration met à jour un magasin symbolique. Chaque branche ajoute une condition au chemin, et un résolveur décide si le chemin est réalisable et produit une entrée concrète qui le suit. Une fonction avec des branches devient un arbre de chemins, et une feuille est un retour avec une condition attachée.

Baldoni, Coppa, D'Elia, Demetrescu et Finocchi ont enquêté sur le terrain en 2018. Leur section 5 nomme l'explosion de chemin comme le coût central: un exécuteur peut forger à chaque branche, de sorte que le nombre d'états peut croître de manière exponentielle dans le nombre de branches, et une boucle dont le bord dépend d'une entrée symbolique donne un ensemble illimité de chemins. Les atténuations qu'ils énumèrent sont la taille des chemins irréalisables en résolvant à chaque branche, les résumés des fonctions et des boucles, la fusion d'état et la sous-approximation. L'exécution concolique exécute le programme de manière concrète et symbolique à la fois, enregistre les conditions le long de la course concrète, en négate une et demande au résolveur la prochaine entrée. DART (2005), CUTE (2005) et SAGE (2008) sont les systèmes nommés.

KLEE (Cadar, Dunbar, Engler, OSDI 2008) est une machine virtuelle symbolique sur bitcode LLVM, donc son entrée est `clang -emit-llvm`Il expédie un modèle d'environnement POSIX et une bibliothèque de lecture. Sur les 89 programmes de base GNU, il atteint une couverture de ligne de 84,5% contre 67,7% pour les tests des développeurs et a trouvé 56 bugs graves. KLEE 3.2 a été publié en 2025-12-23 sous la licence NCSA, avec LLVM 16 recommandé.

Angr travaille sur les binaires. `angr.Project`possède un chargeur CLE qui cartographient l'objet et applique des délocalisations, un élévateur qui transforme un bloc en VEX IR (`block.vex`) avec un code pipeline comme alternative, et une usine pour les États.`SimState`Il contient des registres, une mémoire et un résolveur.`SimulationManager`étapes de stockage des États nommés `active`- Je suis là .`deadended`- Je suis là .`found`- Je suis là .`avoided`, et `errored`, et `explore(find=..., avoid=...)`Les moteurs fonctionnent par étape dans l'ordre: défaillance, syscall, crochet, unicorne pour les données concrètes, et VEX comme le retrait.`from angr import claripy`. angr 10.0.3 a été téléchargé le 2026-10-10, est BSD-2-Clause, et nécessite Python 3.12 ou ultérieur. Z3 5.1.0, publié 2026-08-16 sous MIT, est le résolveur derrière angr, Triton et KLEE backend optionnel.

## Un petit exécuteur sur la leçon 34 IR

L' exécuteur en `code/main.py`utilise une petite IR avec les formes d'expression de la leçon 34: `Assign`- Je suis là .`Branch`- Je suis là .`Jump`- Je suis là .`Return`, plus le `Switch`Le terminateur que le laboratoire de pratique de la leçon 34 a demandé.`Const`- Je suis là .`Var`- Je suis là .`Bin`avec une largeur, `Cast`- Je suis là .`Load`à partir d'une table en béton nommée, et `Not`. Les quatre fonctions de fixation sont soulevées à la main au niveau atteint par la leçon 34 après simplification, où le `0x66666667`Le multiplication est devenu `score / 10`, et chaque bloc IR énumère les adresses d'instruction d'où il vient. `check_ir_against_listing()`affirme que les adresses couvrent la liste de la fonction exactement une fois.

`grade`à`-O0`dans cette IR:

```text
0x30: local_8 = score; local_10 = score / 10 - 6; if (local_10 >u 4) goto 0x9a else 0x62
0x62: switch local_10 -> 0x91, 0x88, 0x7f, 0x76, 0x76
0x76: local_4 = 4; goto 0xa1        0x7f: local_4 = 3; goto 0xa1
0x88: local_4 = 2; goto 0xa1        0x91: local_4 = 1; goto 0xa1
0x9a: local_4 = 0                    0xa1: return local_4
```

Les cibles de l' échange proviennent de `jump_table_targets()`, la même arithmétique de relocalisation que la leçon 33, donc l'IR lit le fichier.

`execute()`Un état a le bloc actuel, la trace de blocs, la liste des conditions et un environnement qui cartographient chaque variable à une expression sur les entrées. `Assign`Il remplace l'environnement dans la source et stocke le résultat. `Branch`Il est donc nécessaire de faire une analyse de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, puis de la situation, de la situation, puis de la situation, de la façon dont le solveur ne peut pas satisfaire, de sorte que le solveur ne peut pas satisfaire la partie, c'est-à-dire de la façon dont l'équipe décrit l'enquête. `Switch`Les entrées de la table sont regroupées par cible et les fourches une fois par cible distincte avec la dissolution des équités d'entrée, ce qui est la façon dont Angr traite également une cible de saut symbolique. `Return`Une trace plus longue que la limite de marche marque le chemin tronqué et l'arrête.

Suivez le chemin de `return 4`L'entrée attribue `local_10 = score / 10 - 6`- L' état de la branche`local_10 >u 4`est faux sur ce chemin, donc la liste des conditions est valable `!(score / 10 - 6 >u 4)`. Les entrées 3 et 4 du commutateur pointent toutes les deux à 0x76, donc la fourchette à 0x76 ajoute `score / 10 - 6 == 3 || score / 10 - 6 == 4`. Bloc 0x76 attribue `local_4 = 4`, passe à 0xa1, et le remplacement de retour à `4`La feuille est la paire de cette condition et de cette valeur.

Le résolveur est `BoundedSearch`. Il contient un domaine fini par entrée, énumère le produit des domaines et évalue chaque contrainte avec la sémantique C: enveloppe de 32 bits, division de tronc, comparaisons signées et non signées, et changements arithmétiques ou logiques. Une division par zéro ou un indice de table en dehors de la table fait que cette assignation échoue à la contrainte.`score`dans les -200 à 1200 les cinq voies de `grade`Le coût de l'évaluation est de 44 460 évaluations et quelques millisecondes.

| Leaf | Returns | Condition | Scores in the bound |
|---|---|---|---|
| 0x76 | 4 | `!(q >u 4) && (q == 3 \|\| q == 4)` | 90 to 109, 20 inputs |
| 0x7f | 3 | `!(q >u 4) && q == 2` | 80 to 89, 10 inputs |
| 0x88 | 2 | `!(q >u 4) && q == 1` | 70 to 79, 10 inputs |
| 0x91 | 1 | `!(q >u 4) && q == 0` | 60 to 69, 10 inputs |
| 0x9a | 0 | `q >u 4` | 1,351 inputs, the rest of the bound |

`q`est pour `score / 10 - 6`Les mêmes classes sortent de`grade`à`-O1`, où l'IR a deux chemins, le chemin de la table retournant `lookup_14[q]`et le chemin par défaut renvoie 0, et la question devient une contrainte sur la valeur retournée, parce qu'il n'y a pas de bloc de cas à atteindre.`lookup_14`est lu à partir de l'objet `.rodata`à 0x14.

La limite est la limite de la méthode. Une recherche limitée est complète à l'intérieur de son domaine et silencieuse à l'extérieur. Dans les limites de -200 à 1200 aucun score de 60 à 109 ne renvoie 0 et aucun score de 110 ou plus ne renvoie une valeur non nulle. z3 prouve les deux pour tous les scores 2^32, et la section suivante montre où les deux méthodes renvoient des témoins différents.

## Étapes de la route mesurées

`total`à`-O1`est la boucle de la leçon 33: un test sur `count`, un corps de boucle qui ajoute `values[rdx]`et des incréments `rdx`, et une `jne`retourner au corps pendant que `rdx != rcx`Dans le IR`rcx`est `zext64(count)`et `rdx`Chaque passage à travers le bord arrière des fourches sur`rdx != count`avec `rdx`une constante, donc le chemin pour `count = c`Il dirige le corps.`c`Les temps et les sorties.`count`limité à 0 à 2, l'exécuteur trouve 3 chemins. Avec 0 à 4 il trouve 5, et avec 0 à 8 il trouve 9. Chaque chemin renvoie la somme triangulaire de son compte: (0, 0), (1, 1), (2, 3), jusqu'à (8, 36). angr rapporte 3, 5 et 9 états morts et les mêmes neuf paires de compte et de somme à tous les trois niveaux d'optimisation, y compris `-O2`, dont la boucle vectorielle a besoin d'un nombre d'au moins 8 et ne change donc jamais le nombre de chemins dans cette plage.

Sans limite, l'exécuteur ne termine jamais, car chaque compte d'itérations est un chemin plus faisable.`count`Le rapport donne ces trois nombres. le coût de résolution augmente aussi: 33 évaluations à la ligne 2, 115 à 4, 531 à 8, et 6 900 pour la course plafonnée.

## angr et z3 sur les mêmes questions

`capture_angr.py`CLE a chargé chaque objet relocalisable avec le backend ELF à 0x400000, appliqué 13 relocalisations pour`-O0`et 8 pour les autres, et résolu les quatre symboles.`grade`avec un argument symbolique de 64 bits, `run()`Il a été conclu avec 5 États en voie de disparition à`-O0`et 2 à `-O1`et `-O2`, en 0,475, 0,408 et 0,318 secondes.`eax == v`Pour chaque état mortel et demandant au solveur le minimum et le maximum signés, il a donné 60 à 69, 70 à 79, 80 à 89, et 90 à 109 pour `v`Les deux contrôles croisés n'ont pas été remplis.

Le formulaire de la section "Rach-a-block" a utilisé CFGFast sur `grade`à`-O0`, qui énumère les huit blocs de la leçon 33 trouvés, alors `explore(find=0x400076)`Il a trouvé un état avec trois encore actifs en 0,47 secondes, avec un minimum de 90 et un maximum de 109 sur l'argument et les échantillons 91, 93, 94, 97 et 101.

Pour `percent_change`, deux États en voie de disparition avec les contraintes `before[31:0] != 0`et `before[31:0] == 0`- Demandez à l' état non zéro`eax == 50`Il a donné`before = 21055744, after = 332315210`à`-O0`- Je suis là .`40566804, 834296877`à`-O1`, et `-24484867, 993934694`à`-O2`La recherche limitée dans -16 à 16 réponses`before = -10, after = -15`. z3, a posé la même question sur la liste de la leçon 34, réponses `before = -25092599, after = 305712160`. La suite de tests évalue les cinq paires avec l'évaluateur IR et donne 50 résultats chacun. Un modèle prouve qu'une entrée existe.

`capture_z3.py`modèles de la `-O1`liste des `grade`dans les vecteurs de bits de 32 bits avec le multiplicateur, les deux shifts, et la table, et pose quatre questions supplémentaires.`switch`Pour un score de 32 bits: non fixé en 0,068 secondes, donc la fonction compilée est égale à la source sur toutes les entrées 2^32.`score / 10`La même différence restreinte à 0 à 109: non satisfaite, c'est pourquoi la grille de la leçon 36 ne pouvait pas la voir.`percent_change`- Je suis là .`before == 3`avec le résultat -133 est placé avec `after = 1073741823`, la cellule de débordement de la leçon 36, et la recherche limitée trouve un deuxième témoin à l'intérieur de son domaine,`after = -1`Où ?`(-1 - 3) * 100 / 3`Il est réduit à -133 sans débordement.

## Faites-le

Exécutez la démo:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
```

La sortie est un rapport JSON. `diffs`contient les correspondances de fichier objet et d' image chargée pour `-O0`contre `-O1`et `-O1`contre `-O2`, plus le `-O0`une paire dans l'ordre BinDiff. `toolComparison`met la phase de la leçon à côté du verdict et du ratio radiff2 et des corrélateurs Ghidra pour chaque fonction. `patch`est le `grade_case10_wrong.o`- Dans le cas.`radare2Ratios`reproduit les numéros radiff2, `gradeRelocationCase`montre la disposition et les deux déplacements, et `symbolic`Il tient les chemins, les classes et les explosions.

Lire `code/main.py`dans cet ordre:

1. `ElfObject`Il lit des sections, des symboles et `.rela.*`Les dossiers.`parse_listing()`Il lit le `llvm-objdump`Je vous ai envoyé un SMS.
2. `jump_table_targets()`et `build_cfg()`donner des blocs accessibles et des bords taillés. `FunctionRecord.signatures()`construit les cinq signatures.
3. `load_image()`définit les sections allouées et s' applique `S + A - P`et `S + A`- Je suis désolé .
4. `match_functions()`est le match de mise en scène. `levenshtein()`et `radare2_ratio()`Donner la similitude.
5. `parse_radiff2()`et `parse_ghidra_vt()`lire les résultats de l'outil capturé pour la comparaison.
6. `evaluate()`- Je suis là .`substitute()`, et `BoundedSearch`sont la sémantique des expressions et le résolveur.
7. `execute()`énumère les chemins. `solve_returns()`demande à chaque feuille un ensemble d'entrées.
8. `grade_ir_o0()`- Je suis là .`grade_ir_o1()`- Je suis là .`percent_change_ir_o1()`, et `total_ir_o1()`sont les ascenseurs avec leurs adresses d'instruction.

Il y a trois commandes de plus.`python3 code/main.py diff A.o B.o`Diffère entre deux objets dont les listes sont situées à côté d'eux, et `--image`diffère de leurs images chargées. `python3 code/main.py paths grade O0`imprime les chemins et les classes d'une fonction, `paths total 8`la boucle avec une limite. `python3 code/main.py figure-data`imprime les données que la figure interactive intègre.

## Utilisez-le

Utilisez le diff et l'exécuteur dans cet ordre sur deux versions de votre propre programme.

1. Capturez une liste pour chaque objet et laissez `function_record()`Vérifiez les octets, et réparez la liste avant tout.
2. Différez d'abord les fichiers objet. Lisez l'étape de chaque correspondance et traitez les correspondances de forme du graphique de flux comme des suggestions jusqu'à ce qu'une deuxième signature soit d'accord.
3. Différer les images chargées. Une fonction qui passe des octets exacts aux octets masqués se déplace par rapport à ses données, et l'arithmétique de déplacement nomme la section qui a déménagé.
4. Pour chaque correspondance, comparez les tables référencées. Le même code avec différentes constantes est une fonction modifiée.
5. Exécutez l'étape des noms et comparez. Lorsque la construction a des symboles, exécutez l'ordre BinDiff et utilisez le résultat structurel uniquement pour classer les étapes.
6. Quand un outil rapporte un rapport, découvrez quels octets y sont entrés avant de le comparer avec le vôtre.
7. Pour la fonction dont vous voulez des entrées, écrivez l'IR avec une adresse par instruction, exécutez `execute()`avec une limite sur chaque compteur de boucle symbolique, et résoudre chaque feuille.
8. Exécutez la même requête à travers angr ou z3 lorsqu'ils sont disponibles et comparez les limites de la classe.

## Laboratoire interactif

### Laboratoire A: cinq signatures à la main

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution/code
python3 -q
```

Je vais courir .

```python
from main import *
grade = {r.name: r for r in fixture_records("ledger-x86_64-O1.o")}["grade"]
grade.raw.hex(" ")
grade.masked_bytes() == grade.raw
grade.mnemonics
grade.cfg_shape()
grade.signatures()
```

Écrivez les quatre octets à l' offset 0x26 de la fonction avant d' imprimer `grade.raw`Alors continuez les mêmes lignes .`grade`de `ledger-x86_64-O0.o`et compter le `table`les bords dans `grade.edges`- Je suis désolé .

### Laboratoire B: regardez la relocalisation briser le hash byte

```python
case = grade_relocation_case()
case["layout"]
case["displacements"], case["rodataShift"]
case["objectBytesEqual"], case["imageBytesEqual"], case["maskedImageBytesEqual"]
```

Compte `0x100110 + 0x14 - 0x10003a`sur papier avant de lire .`displacements`- Alors changez de base en`load_image()`à 0x400000 et confirmer que les déplacements ne changent pas, car les deux opérands se déplacent ensemble.

### Laboratoire C: la fausse correspondance et l'ordre qui l'empêche

```python
report = diff_fixtures("ledger-x86_64-O1.o", "ledger-x86_64-O2.o")
[(m.left.name, m.right.name, m.stage) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o")
[(m.left.name, m.right.name, m.stage, m.names_agree) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o", stages=BINDIFF_ORDER)
[(m.left.name, m.right.name, m.stage) for m in report.matches]
```

Imprimé `records(...)["percent_change"].cfg_shape()`pour `-O0`et `grade_label`pour `-O1`Puis écrivez une sixième étape qui hashes les mnemoniques triés, donc l'ordre n'a plus d'importance, et voyez si elle les sépare.

### Laboratoire D: les chemins et les limites

```python
report = grade_paths("O0")
[(p["trace"], p["condition"], p["returns"]) for p in report["paths"]]
report["returns"]["4"]
total_paths(2), total_paths(4), total_paths(8)
total_paths(64, step_limit=24)["truncated"]
```

Le changement`SCORE_DOMAIN`à `range(0, 110)`et de réécrire `grade_paths("O0")`La classe pour 0 se réduit à 60 entrées et les deux contrôles croisés sont toujours valides.`range(90, 100)`et lire quelles feuilles n'ont pas de modèle du tout.

## Laboratoire de pratique

Remplissez BinDiff sur les fixtures et comparez-le avec les trois résultats que vous avez déjà.

Les exigences:

- Installez l'extension BinExport dans Ghidra 12.1.4 et exportez les trois programmes de registre.
- Installez la ligne de commande BinDiff 8 et produisez la `-O0`contre `-O1`et `-O1`contre `-O2`les résultats avec l'option de similitude uniquement.
- Capturez la similitude et la confiance par fonction dans `code/fixtures/`avec les commandes exactes dans `build_fixtures.sh`- Je suis désolé .
- Extension `tool_comparison()`avec un `bindiff`colonne et ajouter un test qui compare l'ensemble correspondant de BinDiff sur `-O1`contre `-O2`avec le jeu de données masquées de la leçon.
- Écrivez un paragraphe qui donne le nom de l' algorithme utilisé par BinDiff `grade`et si elle a comparé des images chargées.

Le succès signifie que la démo imprime quatre colonnes d'outils par fonction et que le nouveau test indique une égalité ou une différence expliquée.

## Leçon d'artisanat

Cette leçon comprend `outputs/skill-binary-diffing-and-symbolic-execution.md`. Utilisez-le pour transformer deux constructions et une fonction en une différence en étapes avec une qualité déclarée par match, une explication de chaque fonction inégalée ou modifiée, et un rapport de chemin avec un ensemble d'entrée résolu par feuille et la limite qui l'a produite.

## Vérifiez

Exécutez la démo et la suite:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La vérification doit prouver:

- Les tableaux de symboles donnent les quatre fonctions avec des tailles 46, 118, 66, 39 à `-O0`, 16, 46, 31, 63 à `-O1`, et 16, 46, 140, 63 à `-O2`
- le `.rela.text`les documents de l'`-O1`objet décrire comme `llvm-objdump -r`les imprime
- chaque instruction répertoriée se trouve dans sa portée symbolique et ses octets sont égaux à l'objet, et un octet modifié augmente
- `total`à`-O1`Il a cinq blocs, cinq bords et une boucle autonome, et sa chaîne de forme est déterministe
- `grade`à`-O0`récupère cinq cibles de saut de table `.rela.rodata`et il a huit blocs et onze bords
- sans table,`grade`à`-O0`atteint 82 octets en quatre blocs
- `percent_change`à`-O0`et `grade_label`à`-O1`ont la même forme de quatre blocs
- les hashes exacts et masqués sont d'accord sur chaque fonction d'un objet relocalisable
- `-O1`et `-O2`partager des octets pour trois fonctions, `-O0`et `-O1`aucune part
- le tableau de référence de `grade`Il lit`1, 2, 3, 4, 4`et les lectures du candidat `1, 2, 3, 4, 3`
- les emplacements du chargement `.text`- Je suis là .`.rodata`, et `.rodata.str1.1`où Ghidra les a placés dans les deux programmes
- `R_X86_64_PC32`est écrit comme `S + A - P`et donne un déplacement de 122, et `R_X86_64_64`les entrées dans `.data`recevoir des adresses de chaîne
- `grade`Les octets sont d'accord dans les objets et diffèrent dans les images par un décalage de 0x70, et le masquage rétablit la correspondance
- `-O1`contre `-O2`correspond à trois fonctions sur des octets exacts et `total`par nom; la vue d' image correspond `grade`par octets masqués ayant une similitude de 1 à 1/46
- l' ordre de cours produit une correspondance de forme de flux-graphe fausse sur `-O0`contre `-O1`, et l' ordre BinDiff correspond à tous les quatre par leur nom
- le candidat au patch correspond aux octets et échoue aux données référencées
- Le ratio Levenshtein reproduit 0,130435, 0,106061, 0,269841, 0,171429, 0,426829 sur 82 octets, et les ratios relocalisés-bytes radiff2 imprimés avec des relocalisations appliquées
- L'ensemble MATCH de radiff2 est égal à l'étape des octets exacts, et les corrélateurs de octets et d'instructions de Ghidra sont égaux aux ensembles exacts et masqués de la vue d'image
- Ghidra n' a trouvé que des noms sur`-O0`contre `-O1`, avec une similitude de 1,0 et une confiance de 10,0 sur chaque match
- l'IR manuellement levée couvre chaque instruction énumérée une fois, et l'évaluateur suit la sémantique C
- `grade`Il y a cinq chemins à la`-O0`avec les conditions de commutation et deux à `-O1`avec les mêmes classes
- les classes limites 60 à 69, 70 à 79, 80 à 89, 90 à 109 angr's égaux aux trois niveaux et z3, avec 5 et 2 états délimités et la plage de portée de bloc 90 à 109
- les deux contrôles croisés n'ont pas de modèle dans la limite, en angr ou en z3, et z3 prouve que la liste est égale au commutateur source
- chaque fois`percent_change`Le modèle de la recherche délimité, angr et z3 évalue à la valeur demandée
- `total`donne 3, 5, 9 chemins avec les sumes triangulaires, la même que angr, et le plafond de marche retourne 22 et tronque 2
- Les données figuratives contiennent les deux vues, les numéros de relocalisation et cinq feuilles avec leurs ensembles d'entrées, et le module de site intègre les mêmes données

## Mode de défaillance de la production

| Failure | What the weak tool reports | What the analysis must prove |
|---|---|---|
| Byte hash on a loaded image | "grade modified" between two builds with identical code | The field offset, the two displacements, and the section shift; the masked hash matches |
| Masked hash on a relocatable object | "masking made no difference" | The fields were already zero; the relocation records and the referenced constants carry the change |
| Flow-graph shape alone | A confident pair of unrelated functions | A second signature agrees, or the names contradict and the match is dropped |
| Names before structure on a stripped build | Nothing matches | The structural stages with their qualities, and the unmatched list |
| A ratio compared across tools | "radiff2 says 0.43 and the lesson says 0.30" | Which bytes each side fingerprinted; 82 bytes without relocations, 118 with |
| Same code, different constant | "exact match" | The table read through the relocation and the bounds check |
| One model reported as the input | "the score is 90" | The class 90 to 109 with its count, and the bound it was found in |
| Bounded search reported as unsat | "no input gives -133" | Unsat inside the bound only; z3 finds 1073741823 outside it |
| Symbolic loop without a bound | A timeout | The bound or the cap, the paths returned, truncated, and never started |
| Closed tool described as run | Numbers with no capture | A capture file with the command and version, or a sentence saying documentation only |

## Règlement d'exploitation

Partagez une signature à la fois, acceptez uniquement des valeurs uniques des deux côtés et gardez la scène et sa qualité avec la correspondance. Différez les objets et les images chargées et expliquez chaque fonction qui change de scène entre les deux. Enumérez les chemins avec un résolveur à chaque branche, liez chaque boucle symbolique et rapportez un ensemble d'entrée avec sa limite. Un modèle unique sans limite est un échantillon.

## Pour en savoir plus

- [BinDiff concepts: matching strategy, algorithms, and quality ratings](https://github.com/google/bindiff/blob/main/docs/concepts.md)
- [BinDiff repository and the v8 open-source release](https://github.com/google/bindiff/releases/tag/v8)
- [BinExport README: IDA Pro, Binary Ninja, and Ghidra exporters](https://github.com/google/binexport)
- [Flake, "Structural Comparison of Executable Objects" (DIMVA 2004) and Dullien, Rolles, "Graph-based comparison of Executable Objects" (SSTIC 2005), shipped in the BinDiff repository](https://github.com/google/bindiff/tree/main/docs/papers)
- [Diaphora README and heuristics source](https://github.com/joxeankoret/diaphora)
- [Ghidra Version Tracking correlators help page](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/VersionTracking/src/main/help/help/topics/VersionTrackingPlugin/VT_Correlators.html)
- [Ghidra Auto Version Tracking help page](https://www.ghidradocs.com/11.0_PUBLIC/help/VersionTracking/help/topics/VersionTrackingPlugin/VT_AutoVT.html)
- [Ghidra 12.1.4 release](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Binary Ninja 6.0 "Krypton" release notes](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)et [Binary Similarity user guide](https://docs.binary.ninja/guide/similarity.html)
- [radare2 book: radiff2](https://book.rada.re/tools/radiff2/intro.html)et [radare2 6.2.4 source](https://github.com/radareorg/radare2/releases/tag/6.2.4)
- [King, "Symbolic execution and program testing" (CACM 1976)](https://dl.acm.org/doi/10.1145/360248.360252)
- [Baldoni et al., "A Survey of Symbolic Execution Techniques" (ACM Computing Surveys 2018)](https://dl.acm.org/doi/10.1145/3182657)
- [Cadar, Sen, "Symbolic execution for software testing: three decades later" (CACM 2013)](https://dl.acm.org/doi/10.1145/2408776.2408795)
- [Cadar, Dunbar, Engler, "KLEE: Unassisted and Automatic Generation of High-Coverage Tests for Complex Systems Programs" (OSDI 2008)](https://www.usenix.org/legacy/events/osdi08/tech/full_papers/cadar/cadar.pdf)et [KLEE 3.2 release](https://github.com/klee/klee/releases/tag/v3.2)
- [Shoshitaishvili et al., "SoK: (State of) The Art of War: Offensive Techniques in Binary Analysis" (IEEE S&P 2016)](https://oaklandsok.github.io/papers/shoshitaishvili2016.pdf)
- [angr documentation: loading, simulation managers, and the simulation pipeline](https://docs.angr.io/en/latest/core-concepts/toplevel.html)et [angr 10.0.3 on PyPI](https://pypi.org/project/angr/)
- [Z3 5.1.0 release](https://github.com/Z3Prover/z3/releases/tag/z3-5.1.0)
- [Triton dynamic binary analysis library](https://github.com/JonathanSalwan/Triton)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Table 4.9: Relocation Types](https://gitlab.com/x86-psABIs/x86-64-ABI)
