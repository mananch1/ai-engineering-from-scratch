# Analyse dynamique de votre propre programme: débogage, traces et confirmation des revendications statiques

> Une atteinte à un point de rupture est une observation sur une course d'une construction. Cette leçon construit le harnais qui transforme ces observations en verdicts confirmés, réfugiés et non observés sur les affirmations des leçons statiques faites.

**Type:** Build
**Languages:** Python, C, arm64 assembly, lldb command language
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 35 (evidence, confidence, and unknowns)
**Time:** ~130 minutes

## Objectifs d'apprentissage

- Expliquez ce qu'un débogageur a besoin du système d'exploitation sur Linux (`ptrace`) et sur macOS (portes de tâches de la machine et ports d'exception), et pourquoi une coque à sable ne peut pas lancer un débogage.
- Définir les points de rupture logiciels par nom et par adresse, définir un point de surveillance matériel et lire les limites que le processeur impose à ces deux.
- Drive lldb en mode lot à partir de Python avec un crochet stop qui enregistre les registres, la mémoire et les variables du cadre à chaque arrêt, et analyse le texte dans une chronologie.
- Vérifiez les revendications statiques des leçons 32, 33, 34 et 41 contre cette chronologie: les registres d'arguments, les compensations de champs structurels, quel bord une branche conditionnelle a pris, et quels appels ont eu lieu.
- - Je vais faire le même plan .`-O2`construire, lire des points de rupture qui résolvent et ne touchent jamais, et dire ce qui fait et ne prouve pas.
- Enregistrez et chargez un fichier de base, et nommez ce qu'un noyau de style pile laisse de côté.
- Mettez en carte les ensembles de commande gdb et lldb et placez `strace`- Je suis là .`ltrace`- Je suis là .`bpftrace`- Je suis là .`dtrace`, et `rr`sur le bon système d'exploitation avec leurs versions et licences.
- Indiquez les faits de signature de code macOS qui permettent à lldb de déboguer un binaire que vous avez construit vous-même, et les deux conditions dans lesquelles il refuse.

## Le problème

Leçon 41 .`struct account`de DWARF et a dit `balance`La leçon 33 construit le graphe de contrôle-flu de`classify`et a dit que la première branche va à la`return -1`La leçon 32 dit que les deux premiers arguments arrivent dans`x0`et `x1`Chacune de ces phrases était dérivée de bytes.

Un agent lit le même programme et rapporte "les appels principaux"`bump`- Je suis là .`classify`, et `sum_balances`" La table des symboles répertorie les trois, de sorte que la phrase semble sûre.`-O2`Le processus imprime ses quatre lignes et sorties.`hit count = 0`- Je suis désolé .

L'agent n'avait pas tort sur la source, il avait tort sur la construction à propos de laquelle il a été interrogé.`sum_balances(rows, 3)`La seule façon de savoir si cette machine est binaire est de la faire passer sous un débogageur et de regarder.

Cette leçon construit un harnais qui fait la recherche.`account.c`Le classement a trois résultats. `confirmed`signifie un arrêt montrant que l'affirmation est vraie pour cette course. `refuted`signifie un arrêt montré faux.`not-observed`La leçon 35 stocke chaque verdict comme un résultat.`observed`enregistrement avec autorité `controlled-replay`- Je suis désolé .

```figure
re-42-run-timeline
```

## Un débogageur est trois services système et une table

Un débogageur fait quatre choses à un processus dont il ne possède pas le code: il arrête le processus, lit et écrit ses registres, lit et écrit sa mémoire, et il lui est dit quand le processus s'arrête à nouveau. Le noyau fournit chacun d'eux comme service, et les services diffèrent selon le système d'exploitation.

Sur Linux , le service est `ptrace(2)`La page manuelle indique qu'elle " fournit un moyen par lequel un processus (le traceur) peut observer et contrôler l'exécution d'un autre processus ". `PTRACE_TRACEME`Laissez un enfant annoncer que son parent le traquera,`PTRACE_ATTACH`prend un processus non lié, `PTRACE_PEEKDATA`Il y a un mot à l'adresse de l'adré dans la mémoire du tracé.`PTRACE_GETREGS`Il lit les registres et `PTRACE_SINGLESTEP`Les limites du module Yama sont jointes: `/proc/sys/kernel/yama/ptrace_scope`en défaut de 1 signifie "le processus cible doit être un descendant de l'appelant", 2 signifie que seul un administrateur peut joindre, et 3 signifie que personne ne peut.

Sur macOS `ptrace`C'est une interface plus petite.`PT_TRACE_ME`- Je suis là .`PT_ATTACHEXC`- Je suis là .`PT_STEP`- Je suis là .`PT_CONTINUE`, et `PT_KILL`, et dit que avec `PT_ATTACHEXC`"Les signaux de l'enfant vont au parent comme des exceptions Mach".`ptrace`Je ne suis pas du tout.`debugserver`Les appels`task_for_pid()`pour obtenir un port de tâche Mach pour le processus, alors `task_set_exception_ports()`Les points de rupture arrivent comme des messages d'exception.`taskgated`décide qui peut appeler `task_for_pid`- Je suis désolé .

Vous pouvez voir le coût de cette conception dans une transcription que cette leçon garde. La coquille de la session se déroule sous une boîte à sable de la ceinture de siège. Le même plan qui produit onze arrêts en dehors de la boîte à sable produit ceci à l'intérieur:

```text
(lldb) run
error: process exited with status -1 (attach failed ((os/kern) invalid address))
```

Le processus a été créé et ne peut pas être adopté. L'erreur concerne les droits du débogageur, et il ne dit rien sur le programme.

Le débogageur se souvient de quel octet d'instruction il a remplacé à quelle adresse, de sorte qu'il peut remettre l'original lorsque vous continuez, et il compte les coups par emplacement. `breakpoint list`Il imprime la table et le harnais la lit à la fin de chaque course.

## Un byte ou un mot remplacé

Un point de rupture logicielle est une instruction sur laquelle le processeur est piégé, écrit sur les premiers octets de l'instruction que vous voulez arrêter.`INT3`, code op`CC`, d'un octet de longueur, que le manuel d'Intel décrit comme " destiné à appeler le gestionnaire d'exception de débogage ".`BRK #imm16`, une instruction de quatre octets dont l'exception "enregistre l'exception dans ESR_ELx, en utilisant la valeur CE 0x3C".`BRK`l'instruction elle-même, ce qui explique pourquoi un arrêt rapporte l'adresse de l'instruction remplacée et non celle qui la suit.

Lldb définit les points de rupture par nom, par fichier et ligne, ou par adresse.

```text
(lldb) breakpoint set -n bump
Breakpoint 1: where = account-O0`bump + 12 at account.c:13:19, address = 0x00000001000004bc
```

`bump + 12`C'est la quatrième instruction.`sub sp, sp, #0x10`- Je suis là .`str x0, [sp, #0x8]`, et `str w1, [sp, #0x4]`, la configuration du cadre que la table de lignes DWARF attribue à la ligne d'ouverture de la fonction.`x0`et `x1`Je suis toujours en train de tenir les arguments, parce que le prologue ne les a copiés que.

Par adresse, l'Ildb doit savoir à quel module appartient l'adresse.`breakpoint set -a 0x100000504`avant l'existence du processus est une adresse de charge sans section derrière elle, et dans un premier test il est resté `unresolved`tout au long de la course avec `hit count = 0`La forme utilisée par le harnais donne le nom du module:

```text
(lldb) breakpoint set -a 0x100000504 -s account-O0
Breakpoint 4: address = 0x0000000100000504
```

Avec `-s`Le numéro est une adresse de fichier à l'intérieur `account-O0`, lldb ajoute la diapositive de charge, et après la course, l'emplacement est lu `resolved, hit count = 2`. La liste de leçon 33 donne des adresses de fichiers, donc c'est le formulaire qui relie une liste statique à un arrêt dynamique.

### Décodez la branche à la main

La première branche conditionnelle en `classify`à`-O0`C' est le mot .`36f800a8`à`0x1000004f0`, qui `llvm-objdump`des empreintes comme `tbz w8, #0x1f, 0x100000504`- Décodez-le à partir du code Arm A64 de `TBZ`- Je suis désolé .

1. Le 31 est `b5`, le plus haut de la partie.
2. Les bits 30 à 25 sont `011011`, le groupe test-et-branche.`TBZ`et non pas`TBNZ`- Je suis désolé .
3. Les bits 23 à 19 sont `b40`, le petit de cinq bits du nombre de bits: `11111`- 31 avec`b5`= 0 le bit testé est 31, le bit de signalisation d'une valeur de 32 bits.
4. Les bits 18 à 5 sont `imm14`Le numéro de la liste:`00000000000101`= 5. Le décalage de branche est `imm14 * 4`- 20 = `0x14`, donc la cible est `0x1000004f0 + 0x14 = 0x100000504`- Je suis désolé .
5. Les bits 4 à 0 sont `Rt`Le numéro de la liste:`01000`= 8, donc le registre est `w8`- Je suis désolé .

`w8`Une instruction a été chargée plus tôt par `ldr w8, [x8, #0x4]`, qui est `a->balance`à l' offset 4. `TBZ`La branche est prise lorsque le bit de signe est clair, ce qui est `balance >= 0`La source a écrit:`if (a->balance < 0) return -1;`, le compilateur a donc inversé la condition: le bord pris saute le `return -1`Le bloc, et la chute à travers à `0x1000004f4`L'attestation statique que porte le harnais est donc une attestation de bord avec un prédicat: prise lorsque`a->balance >= 0`L'ascenseur de la deuxième branche fonctionne de la même manière.`subs w8, w8, #0xc8`calculs `balance - 200`et `b.le 0x100000524`est pris quand c'est au maximum zéro, ce qui est `balance <= 200`- Je suis désolé .

## Les points de garde sont un matériel et peu

Un point de surveillance arrête le processus lorsque la mémoire d'une adresse est accédée. Il ne peut pas être une instruction remplacée, car n'importe quelle instruction n'importe où pourrait faire l'accès. Les processeurs portent donc un petit ensemble de comparateurs d'adresses.`DR0`à travers `DR3`détient "les adresses de quatre points de rupture maximum", et `DR7`Il permet de définir chaque option et de définir sa condition (exécuter, écrire ou lire-écrire) et sa longueur (1, 2, 4 ou 8 octets).`DBGWVR<n>_EL1`, qui "dont l'adresse virtuelle des données utilisée pour l'ajustement des points de surveillance" et `DBGWCR<n>_EL1`Le manuel de l'Arm dit: "Une mise en œuvre peut inclure entre 2 et 64 points de surveillance".

L' Ildb rapporte le numéro qui lui a été donné:

```text
(lldb) watchpoint set variable a->balance
Watchpoint created: Watchpoint 1: addr = 0x16fdfe00c size = 4 state = enabled type = m
(lldb) watchpoint list
Number of supported hardware watchpoints: 4
```

Quatre sur cette machine.`0x16fdfe00c`est `x0 + 4`: le pointeur structurel à la`bump`stop plus le décalage DWARF de `balance`- Je suis là .`size = 4`est le `int`- Je suis là .`type = m`Lorsque les fentes sont épuisées, lldb refuse le point de surveillance suivant, et parce qu'une région non alignée ou non dotée d'une puissance de deux peut prendre deux fentes, le nombre de points de surveillance que vous pouvez définir peut être inférieur à quatre.

Les points de surveillance de données rapportent après l'accès. Le manuel Intel dit "une instruction qui écrit la mémoire surécrit les données originales avant que l'exception de débogage générée par un point de rupture de données ne soit générée", et lldb garde l'ancienne valeur dans son propre cache pour imprimer les deux:

```text
Watchpoint 1 hit:
old value: 250
new value: 255
* thread #1, queue = 'com.apple.main-thread', stop reason = watchpoint 1
    frame #0: 0x00000001000004d0 account-O0`bump(a=0x000000016fdfe008, by=5) at account.c:14:12
```

L' arrêt est à `0x1000004d0`, l' instruction après `str w8, [x9, #0x4]`à`0x1000004cc`Une chose à savoir avant d'analyser la sortie.`watchpoint set`, empreintes de papier`Watchpoint 1 hit:`avec une valeur ancienne vide et `new value: 250`C'est la première lecture, et le harnais ignore un coup sans valeur ancienne.

## Passer, arrêter les crochets et faire des lots

La grammaire de commande de l'IDB est `<noun> <verb> [-options] [arguments]`Les verbes de marche sont`thread step-in`(le secteur de l'énergie)`s`), `thread step-over`(le secteur de l'énergie)`n`), `thread step-inst`(le secteur de l'énergie)`si`), `thread step-inst-over`(le secteur de l'énergie)`ni`), et `thread step-out`(le secteur de l'énergie)`finish``thread until 12`Il court à la ligne.`thread return <expr>`Le harnais ne marche pas, il s'arrête à des adresses prévues et laisse un crochet d'arrêt faire la lecture:

```text
target stop-hook add -o "register read x0 x1 pc" -o "memory read -s4 -fx -c24 $x0" -o "frame variable -P 1"
```

`-o`Les commandes sont régulières à chaque arrêt.`register read`nom des deux registres d'arguments et du compteur de programme. `memory read -s4 -fx -c24 $x0`imprime 24 mots en hex à partir de l' adresse en `x0`, qui est de 96 octets, trois `struct account`Les rangées.`frame variable -P 1`imprime les arguments et les locaux et élargit les pointeurs à un niveau, donc `a`montre ses cinq champs.

Le mode de lot est `lldb -b`Le manuel dit:`--batch`- Il fait les commandes de `-s`- Je suis là .`-S`- Je suis là .`-o`, et `-O`et puis cesse, et si une commande s'arrête "en raison d'un signal ou d'un crash, le débogageur reviendra à la demande interactive". Le harnais écrit le plan dans un fichier et le passe avec `-s`- Un seul point est important:`settings set interpreter.stop-command-source-on-error false`Sans elle, le premier.`continue`après la sortie du processus, le dossier entier est terminé, et `breakpoint list`Avec elle, des empreintes de l'IDB.`error: Process must be launched.`et continue.

Deux faits sur la forme de sortie déterminent le fonctionnement du parseur.`Process N stopped`bloc avec son `frame #0`Le parseur collecte donc les lignes de registre, de mémoire et de variables dans une pièce pendante et attache la pièce à la suivante `frame #0`Une pièce dont le compteur de programme est dans`dyld`à`_dyld_start`est l'arrêt d'entrée avant l'exécution du programme, et non `frame #0`Deuxièmement, un point de rupture avec`--auto-continue`La première version de ce harnais utilisait un continu automatique et n'enregistrait rien.`continue`les commandes au lieu.

## Le plan vient de la liste

`Plan.commands()`construit le fichier de commande à partir de `llvm-objdump -d`Le texte du binary il sera exécuté.`account-O0`- Le numéro de la liste:

```text
breakpoint set -n bump
breakpoint set -n classify
breakpoint set -n sum_balances
breakpoint set -a 0x100000504 -s account-O0
breakpoint set -a 0x1000004f4 -s account-O0
breakpoint set -a 0x100000524 -s account-O0
breakpoint set -a 0x100000514 -s account-O0
target stop-hook add -o "register read x0 x1 pc" -o "memory read -s4 -fx -c24 $x0" -o "frame variable -P 1"
run
watchpoint set variable a->balance
watchpoint list
continue
...
breakpoint list
watchpoint list -v
```

Les trois points de rupture de nom sont les entrées de fonction.`Listing.conditional_branches()`trouve dans `classify`Le numéro de la liste:`tbz`à`0x1000004f0`et `b.le`à`0x100000510`Le point de surveillance est fixé au premier arrêt, ce que le plan prévoit être `bump`- 16 .`continue`Le budget n'est pas une connaissance du programme, c'est un plafond.

Les revendications sont construites à partir des mêmes entrées par `claims_for()`- Le numéro de la liste:

| Id | Kind | Claim | Source |
|---|---|---|---|
| `arg-registers` | argument-register | at every entry the first arguments arrive in `x0` and `x1` | lesson 32 and AAPCS64 |
| `field-offsets` | field-offset | `struct account` is 32 bytes with `kind` at 0, `balance` at 4, `code` at 8, `rate` at 16, `label` at 24 | lesson 41 and the DWARF dump |
| `branch-1` | branch-taken | `tbz` at `0x1000004f0` goes to `0x100000504` when `a->balance >= 0`, otherwise to `0x1000004f4` | lesson 33 edges and the lesson 34 lift |
| `branch-2` | branch-taken | `b.le` at `0x100000510` goes to `0x100000524` when `a->balance <= 200`, otherwise to `0x100000514` | lesson 33 edges and the lesson 34 lift |
| `calls` | call-edge | `main` calls `bump` once, `classify` three times, `sum_balances` once | source text and `nm` |
| `store-once` | store | `bump` writes `a->balance` once and the new value is the old value plus `by` | lesson 34 pseudocode |
| `stride` | struct-stride | `list[i].balance` sits at `list + 32*i + 4` | lesson 41 and the psABI aggregate rule |
| `output` | behaviour | the program prints `class 0 1 -1` and `sum 315` | source arithmetic |

Les compensations de champ proviennent de `struct_layout()`, qui dit:`DW_TAG_structure_type`avec `DW_AT_byte_size (0x20)`et les cinq `DW_TAG_member`les entrées avec leur `DW_AT_data_member_location`Le plan ne tape jamais le numéro 4.

## La course à -0: onze arrêts

Le premier arrêt est le`bump`Le crochet a imprimé ceci avant le bloc d'arrêt:

```text
      x0 = 0x000000016fdfe008
      x1 = 0x0000000000000005
      pc = 0x00000001000004bc  account-O0`bump + 12 at account.c:13:19
0x16fdfe008: 0x00000073 0x000000fa 0x00000009 0x00000000
0x16fdfe018: 0x00000000 0x3ff40000 0x00000719 0x00000001
0x16fdfe028: 0x0000006c 0xffffffd8 0x00000003 0x00000000
0x16fdfe038: 0x00000000 0x40000000 0x00000721 0x00000001
(account *) a = 0x000000016fdfe008 {
  kind = 's'
  balance = 250
  code = 9
  rate = 1.25
  label = 0x0000000100000719 "savings"
}
(int) by = 5
```

Lisez les huit mots de la première rangée contre les dérivés DWARF.`0x73`, qui est `'s'`, le `kind`Le mot 1 à l'offset 4 est `0xfa`= 250 ,`balance`Le mot 2 à l' offset 8 est `0x9`- Je suis là .`code`, une `short`Le mot 3 à l'offset 12 est le rembourrage avant le 8 octets aligné`double`Les mots 4 et 5 au 16 sont `0x3ff4000000000000`Les mots 6 et 7 au complément 24 sont `0x100000719`, le `label`point dans `__TEXT`Les 8 prochains mots commencent par`0x6c`- Je suis là.`'l'`et `0xffffffd8`= -40: troisième rangée, `loan`, 32 octets plus tard. chaque nombre est d'accord avec`frame variable`Je suis un homme .`_check_layout()`enregistrements `field-offsets`comme `confirmed`à cet arrêt, et il le fait à nouveau à chacun des onze arrêts où`x0`Il est doté d'un pointeur structural.

`x0`égale `a`et `x1`égale `by`Je suis un homme .`arg-registers`Le harnais ne compare que les arguments déclarés.`classify`Arrêtez`x1`Il lit toujours .`0x5`, un reste de `bump`, et `classify`Il est donc impossible de comparer les arguments de l'auteur avec ceux de l'auteur.

Le second arrêt est le point de contrôle, ancienne valeur 250, nouvelle valeur 255, `by`= 5. Un coup dans toute la course, donc `store-once`Puis trois activations de`classify`, chacun avec son arrêt d'entrée et un ou deux arrêt de bord:

| Stop | Address | Stop reason | What it tests |
|---|---|---|---|
| 2 | `0x1000004e8` | breakpoint 2.1 | entry, `a->balance` = 100, `x0` = `a` |
| 3 | `0x100000504` | breakpoint 4.1 | `tbz` taken, 100 >= 0, `branch-1` confirmed |
| 4 | `0x100000524` | breakpoint 6.1 | `b.le` taken, 100 <= 200, `branch-2` confirmed |
| 5 | `0x1000004e8` | breakpoint 2.1 | entry, `a->balance` = 255 |
| 6 | `0x100000504` | breakpoint 4.1 | `tbz` taken, 255 >= 0 |
| 7 | `0x100000514` | breakpoint 7.1 | `b.le` fell through, 255 > 200 |
| 8 | `0x1000004e8` | breakpoint 2.1 | entry, `a->balance` = -40 |
| 9 | `0x1000004f4` | breakpoint 5.1 | `tbz` fell through, -40 < 0 |
| 10 | `0x100000544` | breakpoint 3.1 | `sum_balances` entry, `x1` = 3 |

`_check_branches()`Les deux parties sont les mêmes que les autres parties de l'opération.`branch-1`confirmé trois fois,`branch-2`Deux fois. l'arrêt 5 montre le solde comme 255, qui est la valeur que le point de surveillance a vu écrite.

Au 10e arrêt , le crochet a lu 24 mots de`list`- Je suis là .`_check_stride()`Il prend le mot 1, le mot 9 et le mot 17, qui sont des compensations 4, 36, et 68: 100, 255, et -40. Il compare chacun avec l'équilibre `classify`Il a vu à la même adresse, les a sumés à 315, et a comparé la somme avec la somme de l'argent.`sum 315`L'exécution du processus est alors effectuée avec un statut de 0 et `breakpoint list`Il montre des nombres de 1, 3, 1 pour les noms et 2, 1, 1, 1 pour les quatre bords.`confirmed`, avec 25 records derrière eux, 23 d'entre eux liés à un arrêt et deux à la course finale.

## La course à -O2: Zéro arrêt

La même source à `-O2`C'est un programme différent.`llvm-objdump -d`de `main`Il y en a quatre .`bl`Les instructions, tout à `printf`Au lieu des appels qu' il a`mov w8, #-0x1`- Je suis là .`mov w8, #0x1`, un zéro à travers `xzr`, et `mov w8, #0x13b`Le compilateur a calculé chaque résultat au moment de la compilation.`nm`Liste de listes`_bump`- Je suis là .`_classify`, et `_sum_balances`, car ce sont des fonctions externes et les copies hors ligne doivent exister pour que d'autres objets puissent être appelés.`DW_TAG_subprogram`entrées pour les quatre fonctions et zéro `DW_TAG_inlined_subroutine`Les entrées, donc même les informations de débogage ne enregistrent pas l'inline.

Le plan de cette construction a trois points de rupture de nom et aucun point de rupture de bord. `classify`à`-O2`est de six instructions, `ldr`- Je suis là .`cmp`- Je suis là .`cset`- Je suis là .`cmp`- Je suis là .`csinv`- Je suis là .`ret`La limite est de la partie supérieure de la ligne de démarrage.`-O0`Les listes n'ont pas d'adresse dans ce binary, et le harnais ne les transporte pas.

La course:

```text
(lldb) breakpoint set -n bump
Breakpoint 1: where = account-O2`bump at account.c:13:16, address = 0x0000000100000460
(lldb) run
sizeof 32 align 8
offsets 0 4 8 16 24
class 0 1 -1
sum 315
Process 48444 exited with status = 0 (0x00000000)
(lldb) watchpoint set variable a->balance
error: Command requires a process which is currently stopped.
(lldb) breakpoint list
1: name = 'bump', locations = 1, resolved = 1, hit count = 0
```

Chaque point de rupture résolu.`BRK`Le nombre de morts est nul parce qu'aucune information n'a jamais été trouvée.

| Claim | Verdict | Record |
|---|---|---|
| `calls` | refuted | after a full run with exit status 0, entry stops bump 0, classify 0, sum_balances 0 |
| `output` | confirmed | the four lines and exit status 0 |
| `arg-registers` | not-observed | no stop tested it, hit counts all 0 |
| `field-offsets` | not-observed | no stop tested it |
| `store-once` | not-observed | no stop in bump, so the watchpoint was never set |
| `stride` | not-observed | no stop in sum_balances |

Il a dit que le gouvernement avait fait une réfutation.`main`appelle les trois fonctions de ce programme. Pour cette construction et cette course, il ne le fait pas, et le harnais a vérifié que la course a été terminée, parce qu'un nombre de coups de zéro avant la sortie signifierait seulement que la course n'avait pas encore atteint l'appel. La demande ne dit pas que les fonctions sont un code mort, et la course ne dit pas non plus: elles sont exportées et un autre objet pourrait les appeler. Les quatre autres allégations ne sont ni confirmées ni réfutées. Le struct a encore`balance`dans cette construction, et le `-O2`DWARF le dit, mais pas d'arrêt dans cette course de lecture de mémoire pour le montrer. `not-observed`La leçon 35 garde l'inconnu ouvert jusqu'à ce qu'une course l'observe.

## Les déchets de base

Un fichier de base est la mémoire du processus et les registres écrits sur le disque, de sorte que le débogageur peut lire un programme arrêté après la disparition du programme.`core(5)`"Un petit nombre de signaux qui provoquent une fin anormale d'un processus provoquent également l'écriture d'un enregistrement de l'état de base du processus sur le disque",`/cores/core.pid`"à condition que le processus terminé ait une autorisation d'écriture dans le répertoire, et que le répertoire existait".`sysctl`rapports `kern.coredump: 1`et `kern.corefile: /cores/core.%P`- Je suis là .`/cores`existe avec mode `drwxrwxr-t root admin`, et la coquille `ulimit -c`est 0. Aucun noyau de noyau n'est écrit avant d'augmenter la limite avec `ulimit -c unlimited`, ou à l'échelle du système avec `launchctl limit core unlimited`- Je suis désolé .

L'Ildb écrit un noyau sur demande sans accident.`bump`point de rupture:

```text
(lldb) process save-core --style stack account-O0-stack.core
Saving 49152 bytes of data for memory region at 0x16fdf4000

Modified-memory or stack-memory only corefile created.  This corefile may 
not show library/framework/app binaries on a different system, or when 
those binaries have been updated/modified. Copies are not included
in this corefile.  Use --style full to include all process memory.
```

Le fichier était de 53 248 octets.

```text
(lldb) target create "./account-O0" --core "account-O0-stack.core"
(lldb) bt
* thread #1, stop reason = ESR_EC_BRK_AARCH64 (fault address: 0x0)
  * frame #0: 0x00000001000004bc account-O0`bump(a=0x000000016fdfe008, by=5) at account.c:13:19
    frame #1: 0x0000000100000640 account-O0`main at account.c:40:5
    frame #2: 0x000000019a949d54 dyld`start + 7184
(lldb) frame variable *a
(account) *a = (kind = 's', balance = 250, code = 9, rate = 1.25, label = "")
(lldb) frame variable a->label
(const char *) a->label = 0x0000000100000719 ""
```

Les champs de rétractation et de nombres entiers sont intacts, parce qu'ils vivent sur la pile que le noyau a sauvegardée. `label`Les empreintes sont une chaîne vide avec son pointeur intact.`0x100000719`est correct; les octets qu' il pointe vers sont les littéraux `"savings"`dans le `__TEXT`Le pointer est une observation. la chaîne n'est pas disponible, et lldb ne l'invente pas. `--style full`Il comprend toute la mémoire de processus, et un noyau complet de ce programme de 34 KB sur la machine de recherche était de 2,1 Go, parce que le processus cartographiera le cache partagé.`ESR_EC_BRK_AARCH64`: le noyau a enregistré le `BRK`exception que le point de rupture a soulevé.

Sur Linux , le noyau écrit le noyau selon `/proc/sys/kernel/core_pattern`, un modèle avec `%p`pour le PID, `%e`pour le nom de commande, `%s`pour le signal, et `%t`pour le temps, et un leader `|`Les distributions système utilisent ce tuyau pour`systemd-coredump`, et `coredumpctl list`- Je suis là .`info`- Je suis là .`dump`, et `debug`récupérer ce qu'il a stocké.`gdb prog core`et `lldb -c core prog`ouvrez les résultats.

## Gdb et lldb côte à côte

Les deux débogageurs font le même travail avec des mots différents. `COMMAND_MAP`dans le harnais contient 35 lignes, imprimées par `python3 code/main.py commands`- Les mêmes que cette leçon:

| Task | gdb | lldb | lldb alias |
|---|---|---|---|
| break at a function | `break bump` | `breakpoint set -n bump` | `b bump` |
| break at an address in one module | `break *0x1000004f0` | `breakpoint set -a 0x1000004f0 -s account-O0` | |
| run commands at every stop | `define hook-stop` ... `end` | `target stop-hook add -o '...'` | |
| list breakpoints with hit counts | `info breakpoints` | `breakpoint list` | `br l` |
| watch a variable for writes | `watch a->balance` | `watchpoint set variable a->balance` | `wa s v a->balance` |
| list watchpoints | `info watchpoints` | `watchpoint list -v` | `wa l -v` |
| read one register | `p $x0` | `register read x0` | |
| read memory as words | `x/8xw 0x16fdfe008` | `memory read -s4 -fx -c8 0x16fdfe008` | `x -s4 -fx -c8 0x16fdfe008` |
| show arguments | `info args` | `frame variable --no-locals` | |
| step one instruction | `stepi` | `thread step-inst` | `si` |
| run until the function returns | `finish` | `thread step-out` | `finish` |
| write a core file | `gcore core` | `process save-core --style stack core` | |
| open a core file | `gdb account-O0 core` | `lldb -c core account-O0` | |

L'Ildb's `p`est un alias de `dwim-print`, qui imprime une variable simple à travers le `frame variable`chemin et tombe à nouveau à `expression`Pour tout le reste.`v`est `frame variable`Je préfère .`frame variable`dans un harnais: il lit les emplacements DWARF et ne réalise jamais le code dans le débogage.

Les versions et licences sont disponibles le 10-14 du 2026 à 2026 . lldb porte le numéro de version LLVM, et la version actuelle de llvm.org est LLVM 23.1.3 du 10-06 du 2026 au 2026 .`lldb`Les outils de la ligne de commandement de cette machine .`lldb-1703.0.236.21`Le GDB 18.1 a été publié le 2026-09-25 sous GPL-3.0 ou ultérieur.`.gdb_index`, AArch64 `FPMR`, et Python `gdb.Corefile`GDB n'a pas de support natif pour les processus de silicium d'Apple jusqu'à 18.1: une série de 15 patches "Aarch64-Darwin support initial" a été publiée en octobre 2025, et en février 2026 un entretenu a défini la route pour le débarquer par l'attribution du droit d'auteur et les patches BFD en premier.`gdb`La formule a des bouteilles arm64 et est fabriquée avec un faux .`--target=x86_64-apple-darwin20`, et la requête de retrait qui les a ajoutés dit "il n'est pas possible de déboguer les binaires aarch64-darwin via GDB". Tout déboguer non Apple sur macOS a également besoin d'un certificat auto-signé et le `com.apple.security.cs.debugger`droit ou `task_for_pid`s'échoue avec "(os/kern) failure (0x5)". la place de gdb dans ce cours est Linux, où Debian stable envoie gdb 16.3 pour arm64, et où `rr`Il vit.

## Le programme demande au noyau

Un débogageur montre l'état du programme en arrêt. Un traceur montre les effets du programme sur le système sans l'arrêter longtemps. Les outils sont par système d'exploitation.

| Tool | OS | Version | Licence | What it records |
|---|---|---|---|---|
| `strace` | Linux | 7.2, 2026-08-18 | LGPL-2.1-or-later | system calls with arguments and return values |
| `ltrace` | Linux | 0.7.3 last tarball, git snapshots in Debian | GPL-2.0 | dynamic library calls |
| `bpftrace` | Linux | v0.27.0, 2026-09-10 | Apache-2.0 | kernel and user probes through eBPF |
| `dtrace` and `dtruss` | macOS | `Sun D 1.19` in macOS 26.2 | CDDL, from Sun's OpenSolaris | D probes; `dtruss` prints system calls |

`dtruss`Le manuel indique que "puisque cela utilise DTrace, seuls les utilisateurs dotés de droits de racine peuvent exécuter cette commande".`dtrace(1)`ajoute que "avec les paramètres par défaut de Protection de l'intégrité du système, les programmes D ne sont pas autorisés à accéder aux valeurs d'adresses du noyau ou aux contenus de la mémoire du noyau" et que des actions destructives ne sont pas autorisées sous SIP. `sudo dtruss ./account-O0`La machine n'a pas été exécutée dans cette leçon, parce que le harnais fonctionne sans racine, et la leçon enregistre que comme non vérifié.`write`pour chacun `printf`"Line" est une question de trace, "a-t-il fait`x1`Restez à 5`bump`" est une question de débogage.

## Enregistrement et lecture

`rr`enregistrer une course une fois et la réécrire déterministiquement sous gdb, avec exécution inverse: définir un point de contrôle sur la valeur que vous aimez et `reverse-cont`Il court en arrière vers l'écriture. Le projet se décrit comme " Vous enregistrez une défaillance une fois, puis débogagez l'enregistrement, déterministiquement, autant de fois que vous le souhaitez. " La version actuelle est 5.9.0 de 2025-02-13, sous la licence MIT avec un bloc BSD-2-Clause. C'est seulement Linux. Il a besoin de compteurs de performance matérielle pour compter les branches, il fonctionne donc sur Intel Nehalem et plus tard, certains cœurs AMD Zen et "certaines microarchitectures AArch64 (par exemple. ARM Neoverse N1 ou la série M de silicone d'Apple) " sous Linux. À l'intérieur d'une machine virtuelle, les comptoirs doivent être virtualisés, et le wiki du projet dit "Aucune autre virtualisation sur Mac ne le fait non plus", donc Docker et Colima sur cette machine ne peuvent pas l'exécuter. Pernosco est le service commercial basé sur des enregistrements.

Gdb a son propre enregistreur logiciel. `record full`enregistrer "un journal de l'exécution du processus" avec une limite par défaut de 200 000 instructions, et `record btrace`Il utilise le suivi des branches Intel et " n'enregistre pas de données ". Aucun n'est disponible en lldb. Sur cette machine macOS, la transcription enregistrée du harnais est l'enregistrement: le même plan sur le même binary produit les mêmes arrêts dans le même ordre, et le fichier teste cette ordre.

## Ce que prouve la course

Une mise en œuvre prouve une entrée sur une construction. La norme C définit un comportement indéfini comme "comportement, sur l'utilisation d'une construction de programme non portable ou erronée ou de données erronées, pour lesquelles ce document n'impose aucune exigence", donc un programme qui dépasse un document signé `int`peut légitimement imprimer des résultats différents à `-O0`et `-O2`La leçon 36 a été suivie par le`percent_change`Les cellules de débordement.`-fsanitize=signed-integer-overflow -fsanitize-trap=undefined`Le débordement "exécute une instruction de piège", qui atterrit dans l'ILDB comme un arrêt avec une raison.

Le code optimisé déplace également ce que vous voulez regarder. Le manuel de GDB dit "l'optimisateur a réorganisé votre code; le débogageur vous montre ce qui est vraiment là" et qu'une variable optimisée hors d'existence est une "GDB ne voit jamais".`<optimized out>`, et sa source le sépare de `<unavailable>`: optimisé " n'existe pas réellement dans le programme ", indisponible " existe dans le programme (au niveau de la machine), mais GDB n'a aucun moyen de l'obtenir.`<variable not available>`Un point de surveillance sur un local qui vit dans un registre ne fonctionne jamais, car un registre n'est pas de mémoire.`-O2`Le plus grave est que la fonction entière a été déformée et le silence d'un point de surveillance qui n'a jamais été mis en place ne prouve rien.

Le harnais écrit donc trois types d'enregistrements et jamais un quatrième.`confirmed`Un arrêt qui l' a testé et qui n' est pas d' accord est`refuted`Une affirmation que personne ne peut tester est`not-observed`Le livre de la leçon 35 accepte les deux premiers comme étant`observed`enregistrements avec autorité `controlled-replay`, et garde le troisième comme un inconnu ouvert.

## Déboguer uniquement vos propres constructions sur macOS

Sur le silicone d'Apple, les liens sont pour vous.`ld(1)`dit de `-adhoc_codesign`: "Le code par défaut pour les binaires Apple Silicon est de coder ad hoc". `codesign -dvvv account-O0`sur les empreintes de l'appareil:

```text
Format=Mach-O thin (arm64)
CodeDirectory v=20400 size=387 flags=0x20002(adhoc,linker-signed) hashes=9+0 location=embedded
Signature=adhoc
TeamIdentifier=not set
```

`0x20002`est `CS_ADHOC`(le secteur de l'énergie)`0x2`) plus `CS_LINKER_SIGNED`(le secteur de l'énergie)`0x20000`), "Automatisé signé par le liant" dans l'en-tête de signing de code d'Apple. `codesign -d --entitlements -`Il n'imprime rien après le nom exécutable: le binaire n'a pas de droits. Lldb l'a lancé, a frappé chaque point de rupture, a mis en place un point de surveillance matériel, et a écrit un noyau, avec `DevToolsSecurity -status`le signalement "Le mode développeur est actuellement désactivé" et la protection de l'intégrité du système activée.

Deux droits contrôlent qui peut déboguer quoi, et ils sont assis sur des côtés opposés. `com.apple.security.cs.debugger`est pour le débogageur: "Une valeur Boolean qui indique si l'application est un débogageur et peut s'accorder à d'autres processus ou obtenir des ports de tâches".`debugserver`Il est sous sa forme privée et `codesign -d --entitlements -`sur les listes de copies des outils de ligne de commande `com.apple.private.cs.debugger`- Je suis là .`com.apple.private.set-exception-port`- Je suis là .`com.apple.private.thread-set-state`, et `com.apple.private.logging.diagnostic`- Je suis là .`com.apple.security.get-task-allow`Xcode le injecte dans des constructions de débogage, le débarrasse de l'exportation, et la notariation rejette un binaire qui le possède encore. Les ingénieurs d'Apple énoncent la règle du refus: le port de tâche est refusé lorsque la cible a le temps d'exécution durci activé et n'a pas`get-task-allow`Une plaine .`clang`Si vous signez un accord avec le constructeur, il ne sera pas durci et ne déclenchera aucune condition.`codesign --options runtime`, résignez-le avec `codesign -s - -f --entitlements get-task-allow.plist`avant de débogage.

Trois faits supplémentaires ferment la section. Premièrement, vous pouvez voir le dialogue "Developer Tools Access doit prendre le contrôle d'un autre processus pour continuer le débogage" la première fois dans une session de connexion; `sudo DevToolsSecurity -enable`supprime le mot de passe supplémentaire pour les membres de la `admin`ou `_developer`Les groupes. Deuxièmement, vous ne désactivez pas SIP pour déboguer votre propre code. Apple dit " Vous n'avez pas besoin de désactiver SIP pour exécuter et déboguer des applications à partir de Xcode ", et le seul composant SIP qui touche déboguer est celui `csrutil enable --without debug`Troisièmement, la boîte à sable de la ceinture de siège bloque cette session.`task_for_pid`entièrement, et le fichier `sandbox-O0.lldb.txt`Une course qui ne peut pas commencer est un fait environnemental, et le harnais le rapporte comme tel.

Tout ce qui est dans cette leçon s'applique aux programmes que vous avez compilés. Les mêmes commandes seraient attachées à un autre processus seulement si son propriétaire a opté pour, et rien ici n'enseigne comment changer cela.

## Faites-le

Exécutez la démo:

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program
python3 code/main.py
```

La sortie est un rapport JSON sur les transcriptions enregistrées. Pour chaque construction, il énumère les points d'interruption avec leur résolution et leur nombre de coups, le nombre de points de surveillance matériels, chaque arrêt avec ses registres et arguments, l'état de sortie, la sortie du programme, les erreurs imprimées, et les revendications avec leurs verdicts et les enregistrements de preuves derrière eux. Il ajoute le tour de fond, les résultats de l'API Python, la panne de la boîte à sable et les outils externes qui sont absents sur cette machine.

Forme attendue:

```json
{
  "program": "account.c",
  "variants": {
    "O0": {
      "binary": "account-O0",
      "breakpoints": {"1": {"spec": "name=bump", "address": "0x1000004bc", "resolved": true, "hits": 1}, "...": "..."},
      "hardwareWatchpoints": 4,
      "watchpoints": {"1": {"address": "0x16fdfe00c", "size": 4, "type": "m", "hits": [[250, 255]], "hitCount": 1}},
      "stops": [{"index": 0, "pc": "0x1000004bc", "function": "bump", "kind": "entry", "registers": {"x0": "0x16fdfe008", "x1": "0x5"}, "...": "..."}],
      "exitStatus": 0,
      "claims": [{"id": "arg-registers", "verdict": "confirmed", "confirmed": 5, "refuted": 0, "not-observed": 0, "...": "..."}],
      "evidence": [{"claim": "arg-registers", "verdict": "confirmed", "stop": 0, "observation": "bump: x0 = 0x16fdfe008 and a = 0x000000016fdfe008, x1 = 0x5 and by = 5"}]
    },
    "O2": {"stops": [], "claims": [{"id": "calls", "verdict": "refuted", "...": "..."}], "...": "..."}
  },
  "core": {"savedAt": {"function": "bump", "pc": "0x1000004bc"}, "labelString": "", "...": "..."},
  "types": {"python": "3.9.6", "size": 32, "...": "..."},
  "sandbox": {"stops": 0, "errors": ["process exited with status -1 (attach failed ((os/kern) invalid address))"]},
  "absentTools": ["bpftrace", "gdb", "llvm-objdump", "ltrace", "rr", "strace"]
}
```

Lire `code/main.py`dans cet ordre:

1. `Listing`analyse le `llvm-objdump -d`Les textes sont transférés en fonctions et instructions, et `Function.conditional_branches()`Il trouve les branches avec leurs cibles et leurs traces.
2. `struct_layout()`lisent la structure et les entrées des membres de la `llvm-dwarfdump`Je vous ai envoyé un SMS.
3. `lift_condition()`transforme une branche conditionnelle en un prédicat, la leçon 34 étape pour `tbz`- Je suis là .`cbz`, et les formulaires de comparaison et de branche.
4. `Plan.commands()`écrit le fichier de commande lldb de la liste, et `Plan.lldb_command()`est l'invocation de lot.
5. `parse_transcript()`marche la sortie dans l'ordre et construit `Stop`les enregistrements contenant des registres, la mémoire, les variables et les valeurs de point de garde, plus le tableau des points de rupture et l'état de sortie.
6. `claims_for()`Les États membres déclarent les huit revendications, et `evaluate()`les notes les arrêtent par arrêt avec le `_check_*`fonctionnalités.
7. `rollup()`réduit les dossiers à un verdict par réclamation, le réfuté ayant la priorité sur le confirmé.
8. `capture()`et `capture_extras()`a produit les appareils, et `live()`répète toute la boucle sur une nouvelle construction lorsque `clang`et `lldb`Je peux courir.

Il y a quatre commandements.`python3 code/main.py plan O0`imprime le fichier de commande pour une construction. `python3 code/main.py commands`imprime la table de la GDB et de la LLDB. `python3 code/main.py tools`rapports sur les outils externes mis en œuvre `PATH`- Je suis là .`python3 code/main.py live O0`compilations `account.c`Dans un répertoire temporaire, il met en place des plans à partir de sa nouvelle liste, exécute lldb et classe les réclamations contre cette course.

## Utilisez-le

Utilisez le harnais sur un programme de votre propre ordre.

1. Construisez-le avec `-g`Gardez le binaire, son dSYM ou son DWARF, et le `llvm-objdump -d`- Ils ont envoyé des textos ensemble.
2. Écrivez les revendications avant de courir. Chacune nomme une fonction, un registre ou une adresse ou un décalage, une valeur ou un bord attendu, et la leçon ou le document d'où elle vient. Une revendication que vous écrivez après avoir vu l'arrêt décrit l'arrêt et ne teste rien.
3. Générez le plan à partir de la liste et lisez-le. Chaque adresse doit être tracée à une ligne dans la liste. Si un point de rupture n'a pas une telle ligne, enlevez-le.
4. Exécutez une fois en mode lot et gardez la transcription avec la ligne de commande et la version de débogage dans son en-tête. La transcription est la preuve. Les verdicts sont dérivés d'elle, et n'importe qui peut les dériver à nouveau.
5. Je vous ai lu toutes les notes.`not-observed`Les raisons vous indiquent quelles revendications ont besoin d'une autre entrée, d'une autre construction ou d'un autre point de rupture.
6. Exécutez la construction optimisée avec les mêmes demandes. Attendez-vous à des points de rupture résolus avec zéro hits où le compilateur a été inscrit, et gardez ces demandes ouvertes.
7. Donnez les dossiers confirmés et réfutés dans le livre de la leçon 35 comme `observed`, l' autorité `controlled-replay`Laissez les non-observés comme inconnus avec les points de rupture.

## Laboratoire interactif

### Laboratoire A: lire un arrêt à la main

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program/code
python3 -q
```

Je vais courir .

```python
from main import *
t = load_transcript("session-O0.lldb.txt")
s = t.stops[0]
hex(s.pc), s.function, s.reason, s.registers
[hex(w) for w in s.memory[:8]]
s.struct_pointer().fields
```

Avant la deuxième ligne, écrivez quel mot vous attendez de contenir 250 et quelles deux mots contenir 1.25.`struct.unpack("<d", struct.pack("<II", s.memory[4], s.memory[5]))`et comparer. Alors regardez`t.stops[1].watch_hit`et `t.stops[1].pc`, et trouver cette adresse dans `account-O0.objdump.txt`: c'est l'instruction après le magasin.

### Laboratoire B: branche et son successeur

```python
plan = make_plan("O0")
plan.branches()
classify = plan.listing.function("classify")
[lift_condition(classify, b, "a->balance")[0] for b in plan.branches()]
[(hex(s.pc), s.struct_pointer().field_int("balance")) for s in t.stops_in("classify")]
```

Parallèlement chaque arrêt de bord à son arrêt d'entrée et décider, avant de regarder les preuves, quel successeur le prédicat prédit.`[e.to_dict() for e in evaluate(plan, t, struct_layout(open("fixtures/account-O0.dwarfdump.txt").read(), "account")) if e.claim.startswith("branch")]`et comparer.

### Le labo C: la course qui ne s'est jamais arrêtée

```python
o2 = load_transcript("session-O2.lldb.txt")
len(o2.stops), o2.exit_status, o2.hits()
o2.errors[:3]
make_plan("O2").branches()
rollup(evaluate(make_plan("O2"), o2, struct_layout(open("fixtures/account-O2.dwarfdump.txt").read(), "account")))
```

Expliquez- lui chacun .`not-observed`Dans vos propres mots, alors ouvrez.`account-O2.objdump.txt`et trouver les quatre constantes dans `main`Ça a remplacé les appels.`DW_TAG_inlined_subroutine`dans `account-O2.dwarfdump.txt`avec `count_tags()`- Je suis désolé .

### Laboratoire D: défaire les preuves

```python
text = open("fixtures/session-O0.lldb.txt").read().replace("x1 = 0x0000000000000005", "x1 = 0x0000000000000006", 1)
broken = parse_transcript(text)
[e.to_dict() for e in evaluate(plan, broken, struct_layout(open("fixtures/account-O0.dwarfdump.txt").read(), "account")) if e.claim == "arg-registers"][:2]
```

Un registre édité tourne le premier `arg-registers`enregistrement à `refuted`et le roulement avec elle, tandis que les quatre autres entrées restent `confirmed`- Alors supprimez le `old value: 250`ligne avant l' arrêt de la garde et de la surveillance `store-once`tomber à`not-observed`Un élève qui ne peut pas être fait échouer n'est pas un élève.

Si l' Ildb peut lancer un processus dans votre coquille, terminez avec `python3 main.py live O0`et `python3 main.py live O2`Les adresses de la pile seront différentes des fixations.

## Laboratoire de pratique

Prolongez le harnais avec une réclamation concernant la boucle dans `sum_balances`- Je suis désolé .

Les exigences:

- Trouvez la boucle à l' arrière de la`sum_balances`dans le `-O0`liste avec `conditional_branches()`et ajouter un point de rupture à l'en-tête de la boucle.
- Ajouter une revendication que l' en-tête exécute `count + 1`temps pour `count`= 3, qui est la leçon 33 de l'arrière bord plus le test de sortie.
- Étendre le crochet d'arrêt ou le plan de sorte que le crochet lit la fente de pile du compteur de boucle à chaque arrêt d'en-tête, et écrire un `_check_loop()`qui enregistre la séquence des contre-valeurs.
- Récupérez la transcription avec `python3 main.py capture O0`Le test du plan échouera jusqu'à ce que le fichier de commande engagé correspond au nouveau plan, qui est le but de ce test.
- Ajoutez un test qui modifie une valeur de contre-indication dans la transcription et affirme les virages de la demande `refuted`- Je suis désolé .

Le succès signifie que le rapport montre les en-têtes de l'en-tête dans l'ordre de leurs contre-valeurs, et que les`-O2`La course conserve la nouvelle revendication `not-observed`avec la raison que la boucle a été vectorisée ou pliée.

## Leçon d'artisanat

Cette leçon comprend `outputs/skill-dynamic-analysis-of-your-own-program.md`. Utilisez-le pour transformer un programme que vous avez construit, sa liste et une liste de revendications statiques en une transcription de débogage enregistrée et un verdict par revendication, avec l'arrêt qui a produit chaque verdict et la raison de chaque revendication qui est restée non observée.

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La suite fonctionne sans Ildb. Elle lit les transcriptions capturées une fois dans`code/fixtures/`La vérification doit prouver:

- Le parseur de liste trouve les quatre fonctions avec leurs rangées et les deux branches conditionnelles de `classify`à`-O0`avec cible et dépassage
- l' ascenseur de `tbz w8, #0x1f`est `a->balance >= 0`et le lifting de `subs`plus `b.le`est `a->balance <= 200`
- `main`à`-O0`Les appels`bump`- Je suis là .`classify`trois fois, et `sum_balances`, et `main`à`-O2`- Il n' y a que des appels .`printf`
- x86-64 `j<cc>`Les mémoires sont conditionnelles et `jmp`et `callq`ne sont pas
- le dépôt DWARF produit un structure de 32 octets avec des membres à 0, 4, 8, 16 et 24
- le `-O2`DWARF dispose de quatre sous-programmes et de zéro entrée en sous-routine en ligne, et `nm`encore énumère les trois fonctions
- le plan équivaut au fichier de commande engagé pour les deux constructions, et l'en-tête de transcription enregistre la commande lldb qui l'a exécuté
- le `-O0`Transcript a onze arrêts dans les fonctions attendues, état de sortie 0, et les quatre lignes de sortie
- la sortie du crochet est fixée à l'arrêt qui la suit, avec `x1`= 5, la mémoire à partir de `x0`, et la chaîne d' étiquette
- le tableau des points de rupture contient la résolution et les nombres de coups 1, 3, 1, 2, 1, 1, 1
- le point de contrôle est de 4 octets de type modifier à `x0 + 4`, frappé une fois avec la valeur ancienne 250 et la valeur nouvelle 255, et quatre machines à sous matérielles sont signalés
- le `-O2`La transcription a zéro arrêt, trois points de rupture résolus avec zéro accès, aucun point de surveillance et une erreur par non utilisé `continue`
- le lancement en boîte à sable échoue avant tout arrêt avec l'erreur de fixation
- Le centre de l' itinéraire fait le tracé et `balance`= 250 et perd la chaîne derrière `label`
- L'API Python lit les cinq champs, la taille 32, et la fonction varie sans processus
- une transcription écrite à la main x86-64 analyse avec `rdi`- Je suis là .`rsi`, et `rip`
- chaque réclamation est confirmée à `-O0`avec des enregistrements 5, 11, 3, 2, 1, 1, 1, 1
- la preuve de la branche nomme la valeur, la condition et le bord
- la preuve de l'offset de champ décode les mots à travers la mise en page DWARF
- Les preuves de stockage et de marche portent la valeur ancienne et la valeur nouvelle et les équilibres de trois rangées
- le `-O2`run réfute les bordures d'appel et laisse les autres affirmations non observées
- une valeur de registre modifiée est refusé, un point de contrôle supprimé n'est pas observé et la réfutation gagne le rallumage
- la carte de commande contient des formulaires gdb et lldb pour chaque ligne, et la table de matériel nomme quatre registres de débogage x86
- la démo liste les outils absents au lieu de l'échec, et le mode en direct rapporte l'outil manquant par nom
- Les données figuratives proviennent des deux sessions et le module du site intègre les mêmes données

## Mode de défaillance de la production

| Failure | What the weak report says | What the harness must prove |
|---|---|---|
| Resolved breakpoint with zero hits | "the function runs" or "the function is dead" | a full run with exit status, hit counts, and the verdict `refuted` for the call edge only |
| Register read at the wrong stop | "`x1` = 5 inside `classify`" | compare only declared arguments at the entry stop |
| Struct read without offsets | a column of hex words | every word mapped to a `DW_AT_data_member_location` and compared with `frame variable` |
| Branch graded from the mnemonic | "`tbz` means taken when negative" | the lifted predicate from the encoding and the stop at the successor |
| Watchpoint on a register-resident local | "no write happened" | the watchpoint was set on an address and was hit, or the record is `not-observed` |
| Auto-continue breakpoints | a clean transcript with no stops | plain breakpoints, a stop hook, and a continue budget |
| Address breakpoint without a module | `unresolved, hit count = 0` | `-s MODULE` so the listing address resolves to a load address |
| First command error ends the batch | no `breakpoint list` at the end | `interpreter.stop-command-source-on-error false` |
| Stack-style core read as full memory | `label = ""` reported as the program's value | the core style, the pointer, and the segment it points into |
| Sandboxed shell | "lldb is broken" | `attach failed ((os/kern) invalid address)` recorded as an environment fact |
| Hardened target without `get-task-allow` | "macOS forbids debugging" | the two conditions from Apple's rule, and a re-sign of your own binary |
| Transcript discarded after grading | verdicts with no source | the transcript file, its command line, and the debugger version |

## Règlement d'exploitation

Planifiez les arrêts de la liste, exécutez une fois en mode lot, gardez la transcription et notez chaque réclamation statique à chaque arrêt qui peut le tester. Écrivez confirmé ou refusé uniquement à partir d'un arrêt, écrivez non observé avec le nombre de coups et la raison, et ne laissez jamais un point de rupture résolu intervenir pour une course. Débogagez uniquement les constructions que vous avez faites.

## Pour en savoir plus

- [LLDB Tutorial: command grammar, breakpoints, watchpoints, stop hooks](https://lldb.llvm.org/use/tutorial.html)
- [lldb(1) manual page: --batch, --one-line, --source, --core](https://lldb.llvm.org/man/lldb.html)
- [GDB to LLDB Command Map](https://lldb.llvm.org/use/map.html)
- [LLDB Python Reference: SBTarget, SBFrame, SBValue, and the embedded interpreter](https://lldb.llvm.org/use/python-reference.html)
- [LLDB Caveats: the Python build lldb links against](https://lldb.llvm.org/resources/caveats.html)
- [LLVM 23.1.3 release](https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.3)
- [GDB 18.1 release announcement, 2026-09-25](https://sourceware.org/pipermail/gdb-announce/2026/000150.html)
- [GDB manual: Process Record and Replay](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Process-Record-and-Replay.html)
- [GDB manual: Debugging Optimized Code](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Optimized-Code.html)
- [GDB wiki: Permissions on Darwin, code signing a debugger](https://sourceware.org/gdb/wiki/PermissionsDarwin)
- [gdb-patches: Initial AArch64-Darwin support series, October 2025](https://sourceware.org/pipermail/gdb-patches/2025-October/222196.html)
- [Homebrew gdb formula pull request adding arm64 bottles](https://github.com/Homebrew/homebrew-core/pull/209753)
- [ptrace(2), Linux manual page](https://man7.org/linux/man-pages/man2/ptrace.2.html)
- [lldb debugserver MachTask.mm: task_for_pid and task_set_exception_ports](https://github.com/llvm/llvm-project/blob/main/lldb/tools/debugserver/source/MacOSX/MachTask.mm)
- [Intel SDM Volume 2: INT n, INTO, INT3, INT1](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Intel SDM Volume 3: debug registers DR0 to DR7 (Debug, Branch Profile, TSC, and Resource Monitoring Features)](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Arm A64 ISA: BRK, Breakpoint instruction](https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/BRK--Breakpoint-instruction-)
- [Arm A64 ISA: TBZ, Test bit and branch if zero](https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/TBZ--Test-bit-and-branch-if-zero-)
- [Arm Architecture Reference Manual, D2.9 Watchpoint exceptions](https://developer.arm.com/documentation/ddi0487/latest/)
- [Apple ld(1): -adhoc_codesign](https://keith.github.io/xcode-man-pages/ld.1.html)
- [Apple ld64 LinkEdit.hpp: CS_LINKER_SIGNED](https://github.com/apple-oss-distributions/ld64/blob/59a99ab60399c5e6c49e6945a9e1049c42b71135/src/ld/LinkEdit.hpp)
- [Apple: Debugging Tool Entitlement, com.apple.security.cs.debugger](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.security.cs.debugger)
- [Apple: Hardened Runtime](https://developer.apple.com/documentation/security/hardened-runtime)
- [Apple: Resolving common notarization issues, the get-task-allow entitlement](https://developer.apple.com/documentation/security/resolving-common-notarization-issues)
- [Apple: Disabling and Enabling System Integrity Protection](https://developer.apple.com/documentation/security/disabling-and-enabling-system-integrity-protection)
- [Apple Developer Forums: debugserver may not get the taskport of a hardened process](https://developer.apple.com/forums/thread/127544)
- [Apple DevToolsSecurity(8)](https://keith.github.io/xcode-man-pages/DevToolsSecurity.8.html)
- [Apple core(5)](https://keith.github.io/xcode-man-pages/core.5.html)
- [Apple Technical Note TN2124: Mac OS X Debugging Magic, core dumps](https://developer.apple.com/library/archive/technotes/tn2124/_index.html)
- [lldb `process save-core` help text](https://github.com/llvm/llvm-project/pull/143126)
- [core(5), Linux manual page](https://man7.org/linux/man-pages/man5/core.5.html)
- [coredumpctl(1)](https://man7.org/linux/man-pages/man1/coredumpctl.1.html)
- [Apple dtrace(1)](https://keith.github.io/xcode-man-pages/dtrace.1.html)
- [Apple dtruss(1m)](https://keith.github.io/xcode-man-pages/dtruss.1m.html)
- [strace releases](https://github.com/strace/strace/releases)
- [ltrace](https://ltrace.org/)
- [bpftrace CHANGELOG](https://github.com/bpftrace/bpftrace/blob/master/CHANGELOG.md)
- [rr project](https://rr-project.org/)
- [rr 5.9.0 release](https://github.com/rr-debugger/rr/releases/tag/5.9.0)
- [rr wiki: Building And Installing, hardware and VM requirements](https://github.com/rr-debugger/rr/wiki/Building-And-Installing)
- [Clang UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
- [ISO/IEC 9899:2024 working draft N3220, clause 3.5 undefined behavior](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3220.pdf)
- [DWARF Debugging Information Format Version 5](https://dwarfstd.org/doc/DWARF5.pdf)
- [Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64)](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement](https://gitlab.com/x86-psABIs/x86-64-ABI)
