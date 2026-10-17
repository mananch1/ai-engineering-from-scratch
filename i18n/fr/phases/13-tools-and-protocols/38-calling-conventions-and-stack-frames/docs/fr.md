# Appels à conventions et à des cadres de pile: système V, Windows x64 et AArch64

> Un démontage montre quel registre une fonction lit. Seul l'ABI dit ce que ces registres signifient, et l'ABI est choisi par l'en-tête de l'objet, donc cette leçon lit d'abord l'en-tête et vérifie ensuite chaque registre contre la règle qui l'y a mis.

**Type:** Build
**Languages:** Python, C, x86-64 and AArch64 assembly
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~140 minutes

## Objectifs d'apprentissage

- Choisissez la convention d'appel de l'en-tête d'objet: ELF x86-64 est le système V psABI, COFF AMD64 est Microsoft x64, ELF AArch64 est AAPCS64, Mach-O arm64 est la variante d'Apple.
- Décodez le prologue d'une fonction octet par octet pour chaque ABI et placez chaque registre enregistré, l'adresse de retour et le pointeur de cadre à son décalage du pointeur de pile d'entrée.
- Inférer les registres de paramètres d'une fonction des registres qu'elle lit avant de les écrire, et les espaces d'arguments de pile des charges au-dessus du pointeur de pile d'entrée.
- Indiquez les séquences de registre indépendantes du système V à partir des fentes de position de Microsoft x64, et à la fois des compteurs NGRN et NSRN de l'AAPCS64.
- Vérifiez chaque inférence contre la signature source, la règle dans la spécification, et les données de décomposition du compilateur dans `.pdata`- Je suis là .`.xdata`, et `.eh_frame`- Je suis désolé .
- Comparer comment Ghidra, IDA, Binary Ninja, radar2, rizin et angr nom et récupérer les mêmes faits, et enregistrer où leur sortie ne convient pas à la source.

## Le problème

Vous avez compilé`convention.c`pour Linux x86-64 à `-O2`Il a demandé à Radar2 6.2.4 les arguments de`eight_args`Il a répondu avec sept:`rdi`- Je suis là .`rsi`- Je suis là .`rdx`- Je suis là .`r8`- Je suis là .`r9`- Je suis là .`rsp+0x8`, et `rsp+0x10`La source C a huit paramètres.`%rcx`Exactement une fois, en `leaq (%rdx,%rcx), %rax`, et la récupération variable de l'outil n'a pas compté un registre qui apparaît uniquement à l'intérieur d'une adresse.

Vous avez ensuite ouvert la version Windows du même fichier dans rizin 0.9.1.`amd64`et trouvé`r8`- Je suis là .`rcx`- Je suis là .`rdx`L'objet est COFF, donc la convention est Microsoft x64, le quatrième argument de registre est `r9`Les deux outils manquent de différents registres dans la même source.

Une liste est une séquence d'instructions. Une convention d'appel est un contrat qui ne se trouve pas dans la liste: quelles registres contiennent des arguments, que l'appelant doit conserver, où se trouvent l'adresse de retour et les arguments de pile, et ce que l'appelant doit laisser au-dessus du pointeur de pile. Quatre documents définissent les contrats qui comptent ici. Le système V AMD64 psABI (version 1.0, construction du 12 mars 2025) les sections 3.2.1 à 3.2.3. La convention d'appel Microsoft x64, l'utilisation de la pile et les pages prologues et épilog (date de page de 2018-12-17 à 2025-03-19). Le RAPCS64 du bras publie les sections 6.1 à 6.9 du 2025Q4 (23 janvier 2026). Les notes de plateforme arm64 d'Apple. Cette leçon lit ces règles, les applique à un fichier C compilé de trois façons, et construit l'analyseur qui transforme une liste en une liste vérifiée de paramètres de localisation, enregistrements enregistrés, et de cadres de compensation.

```figure
re-38-three-frames
```

## Un titre choisit le manuel de règles

`convention.c`a douze fonctions: `leaf_add`- Je suis là .`eight_args`- Je suis là .`mixed`- Je suis là .`by_value`- Je suis là .`big_ret`- Je suis là .`sum_varargs`- Je suis là .`keeps`, et cinq appelants de `extern`Les structures utilisent des fonctions de même forme.`long long`, parce que `long`Il est de 4 octets sur la cible MSVC et 8 octets sur les autres, et une structure qui change de taille change de convention.`-target x86_64-unknown-linux-gnu`- Je suis là .`x86_64-pc-windows-msvc`, et `aarch64-unknown-linux-gnu`à`-O0`et `-O2`, et `arm64-apple-macos`à`-O2`. Aucun en-tête Linux ou Windows n'est installé, donc le fichier déclare ce dont il a besoin et inclut uniquement le propre clang `stdarg.h`et `stddef.h`- Je suis désolé .

`describe_object()`dans `code/main.py`Il lit le numéro magique de 4 bytes et un champ de machine et nomme le contrat:

| Header | Field | Value | Rulebook |
|---|---|---|---|
| `7f 45 4c 46` ELF | `e_machine` at offset 0x12 | 62 | System V AMD64 psABI |
| `64 86` COFF | `Machine` at offset 0 | 0x8664 | Microsoft x64 |
| `7f 45 4c 46` ELF | `e_machine` | 183 | AAPCS64 |
| `cf fa ed fe` Mach-O 64 | `cputype` at offset 4 | 0x0100000c | AAPCS64 with Apple differences |

Les trois contrats sont en désaccord sur presque toutes les lignes du tableau suivant.

| Rule | System V psABI | Microsoft x64 | AAPCS64 |
|---|---|---|---|
| Integer arguments | `%rdi %rsi %rdx %rcx %r8 %r9`, next free register | `RCX RDX R8 R9`, by position | `x0` to `x7`, by NGRN |
| Floating-point arguments | `%xmm0` to `%xmm7`, next free register | `XMM0` to `XMM3`, by position | `v0` to `v7`, by NSRN |
| Integer result | `%rax`, then `%rdx` | `RAX` | `x0`, then `x1` |
| Floating-point result | `%xmm0`, then `%xmm1` | `XMM0` | `v0` |
| Result too large for registers | address passed in `%rdi` "as if it were the first argument", returned in `%rax` | pointer in `RCX`, arguments shifted right, returned in `RAX` | address in `x8`, `x0` to `x7` untouched |
| Callee-saved | `%rbp %rbx %r12` to `%r15` | `RBX RBP RDI RSI RSP R12` to `R15`, `XMM6` to `XMM15` | `r19` to `r29`, `SP`, low 64 bits of `v8` to `v15` |
| First stack argument on entry | `8(%rsp)` | `40(%rsp)`, above a 32-byte shadow store | `0(sp)` |
| Below the stack pointer | 128-byte red zone | "considered volatile" | no red zone (Apple: 128 bytes) |
| Alignment at the call | `(%rsp + 8)` multiple of 16 on entry | 16-byte aligned outside prolog and epilog | `SP mod 16 = 0` |
| Variadic marker | `%al` holds an upper bound on vector registers used | floats duplicated in the integer register | none in registers; Apple uses 8-byte stack slots |

## Système V: deux séquences de registre indépendantes et une zone rouge

psABI 3.2.3 classe chaque octogène de chaque argument en tant qu'INTEGER, SSE ou MEMORY et attribue ensuite des registres à partir de deux listes qui ne s'interagissent pas. "INTEGER" prend "le prochain registre disponible de la séquence %rdi, %rsi, %rdx, %rcx, %r8 et %r9". SSE prend `%xmm0`à `%xmm7`. "S'il n'y a pas de registres disponibles pour un argument de huit octets, l'ensemble de l'argument est passé sur la pile", et les arguments de mémoire "sont poussés sur la pile dans l'ordre inversé (de droite à gauche).

`eight_args(a, b, c, d, e, f, g, h)`à`-O2`est de 27 octets:

```text
      10: 48 01 fe                     	addq	%rdi, %rsi
      13: 48 8d 04 0a                  	leaq	(%rdx,%rcx), %rax
      17: 48 01 f0                     	addq	%rsi, %rax
      1a: 4c 01 c0                     	addq	%r8, %rax
      1d: 4c 01 c8                     	addq	%r9, %rax
      20: 48 03 44 24 08               	addq	0x8(%rsp), %rax
      25: 48 03 44 24 10               	addq	0x10(%rsp), %rax
      2a: c3                           	retq
```

Six caisses, puis deux charges à la`8(%rsp)`et `16(%rsp)`. La figure 3.3 du psABI met "argument mémoire 8byte 0" à `16(%rbp)`quand ?`%rbp`est la base du cadre, qui est `8(%rsp)`à l'entrée avant toute poussée. `g`est à 8, `h`À 16 ans.

`mixed(int i, double d, long long l, float f)`Les deux listes sont indiquées.`%edi`et `%rsi`Les deux premières et deuxièmes enregistrements de l'INTEGER.`%xmm0`et `%xmm1`Les premières instructions sont:`cvtsi2sd %edi, %xmm2`, et `%xmm2`est un registre de rayures que la fonction écrit avant de lire.`cvtsi2sd`Comme un lecteur invente un troisième argument de point flottant.`mixed`comme `(double arg1, double arg2, double arg3)`dans `xmm0`- Je suis là .`xmm1`- Je suis là .`xmm2`- Je suis désolé .

`by_value(struct pair p)`passe un struct de 16 octets comme deux octets intégrés en `%rdi`et `%rsi`, et le renvoie de la même manière: `leaq 0x1(%rdi), %rax; leaq -0x1(%rsi), %rdx; retq`- Je suis là .`big_ret(long long seed)`renvoie 32 octets, qui est la mémoire de classe, et le psABI dit "l'appelant fournit de l'espace pour la valeur de retour et passe l'adresse de ce stockage en %rdi comme si c'était le premier argument". `seed`se déplace vers `%rsi`, et la fonction commence par `movq %rdi, %rax`parce que "sur le retour %rax contiendra l'adresse qui a été passée par l'appelant en %rdi".

La zone rouge est l'autre signature du système V. psABI 3.2.2: "La zone de 128 octets au-delà de l'emplacement indiqué par %rsp est considérée comme réservée et ne peut pas être modifiée par des manipulateurs de signaux ou d'interruptions".`-O0`- Je suis là .`eight_args`est `pushq %rbp; movq %rsp, %rbp`et puis six magasins à `-0x8(%rbp)`à travers `-0x30(%rbp)`sans aucun`subq`Les magasins sont à 48 bytes de la pointeur.`-O2`construire `sum_varargs`répartit 88 octets avec `subq $0x58, %rsp`et écrit ensuite la zone d' épargne du registre à `-0x58(%rsp)`à la`-0x80(%rsp)`: 128 octets en dessous du pointeur de pile, toute la zone rouge.`-mno-red-zone`et `leaf_add`gagne un `pushq %rax`et une `addq $0x8, %rsp`pour tenir ses 8 bytes de locaux au-dessus du pointeur de pile.

Les appels variadiques ajoutent un registre. "%al est utilisé comme argument caché pour spécifier le nombre de registres vectoriels utilisés", une limite supérieure de 0 à 8. `call_varargs`Il est en train de le mettre avec `xorl %eax, %eax`à`-O2`et `movb $0x0, %al`à`-O0`- Je suis là .`sum_varargs`Il le lit d' abord: `testb %al, %al; je`Il saute huit .`movaps %xmmN`enregistrement lorsque aucun registre vectoriel n'a été utilisé.

## Microsoft x64: Les positions, un magasin d'ombre et aucune zone rouge

Les règles de Microsoft sont positionnelles. " Les arguments entiers sont passés dans les registres RCX, RDX, R8 et R9. Les arguments de point flottant sont passés dans XMM0L, XMM1L, XMM2L et XMM3L. " et " Toutes les arguments de point flottant et de double précision dans les quatre premiers paramètres sont passés dans XMM0 - XMM3, selon la position. " Le même `mixed`arrive comme `%ecx`- Je suis là .`%xmm1`- Je suis là .`%r8`- Je suis là .`%xmm3`: la fente 0 est un int, donc `RCX`La fente 1 est double, donc `XMM1`La fente 3 est flottante, donc `XMM3`- Je suis là .`XMM0`et `XMM2`Restez vide.

L'appelant doit également laisser de la place. "L'appelant doit toujours allouer suffisamment d'espace pour stocker quatre paramètres de registre, même si l'appel ne prend pas autant de paramètres". Les pages l'appellent le magasin d'ombre ou les adresses d'accueil, et les quatre fentes sont "propriété effective de la fonction appelée". À l'entrée `0(%rsp)`est l'adresse de retour, `8(%rsp)`à `39(%rsp)`Les quatre maisons sont les premières, et la première étape de l'argument est à`40(%rsp)`- Je suis là .`eight_args`à`-O2`Il lit`0x28(%rsp)`- Je suis là .`0x30(%rsp)`- Je suis là .`0x38(%rsp)`- Je suis là .`0x40(%rsp)`: 40, 48, 56, 64.

Le côté de l'appelant montre l'arithmétique. `call8`à`-O2`est `subq $0x48, %rsp`et puis des magasins de 5, 6, 7, 8 à `0x20(%rsp)`à `0x38(%rsp)`avant `movl $1, %ecx`72 octets est 32 pour le magasin d'ombre, 32 pour quatre arguments de pile, et 8 pour que l'adresse de retour soit sur une limite de 16 octets. `call_varargs`n' a que des arguments enregistrés et continue de répartir `subq $0x28, %rsp`: 32 pour le magasin d'ombres et 8 pour l'alignement. `keeps`pousse trois registres non volatils (24 octets) et puis `subq $0x20, %rsp`, donc que `ext`trouve son stock d'ombre de 32 octets sous l'adresse de retour.

Les structures suivent leur propre règle. " Tout argument qui ne se trouve pas dans 8 octets, ou qui n'est pas 1, 2, 4 ou 8 octets, doit être passé par référence. " et " les arguments de 16 octets sont passés par référence. " `by_value`Il est donc indiqué que la copie est en `%rdx`, après un indicateur de résultat caché dans `%rcx`: "l'appelant doit allouer la mémoire pour la valeur de retour et lui passer un pointeur comme premier argument. Les arguments restants sont ensuite déplacés un argument à droite". `call_pair`construit la copie à `0x20(%rsp)`avec `movaps`, passe `leaq 0x20(%rsp), %rdx`, passe le tampon de résultat avec `leaq 0x38(%rsp), %rcx`, et il lit `.b`de retour de `0x40(%rsp)`- Je suis désolé .

Il n'y a pas de zone rouge. "Toute mémoire au-delà de l'adresse actuelle de RSP est considérée comme volatile: le système d'exploitation ou un débogageur peut surécrire cette mémoire lors d'une session de débogage de l'utilisateur ou d'un gestionnaire d'interruption".`-O0` `leaf_add`Il a besoin de 8 octets de locaux et les obtient avec `pushq %rax`à l'entrée et `popq %rcx`avant `retq`, deux instructions en un seul octet au lieu d' une `subq`et une `addq`. La page d'utilisation de la pile définit la feuille qui peut faire ceci: "Une fonction de feuille est celle qui ne nécessite pas une entrée de table de fonctions. Elle ne peut pas apporter de modifications à des registres non volatils, y compris RSP".

Variadic appelle à la maison leurs registres. " Pour les fonctions vararg ou non prototypées, toutes les valeurs de point flottant doivent être dupliquées dans le registre général correspondant. " `sum_varargs`à`-O2`magasins `%rdx`- Je suis là .`%r8`- Je suis là .`%r9`à `0x18(%rsp)`- Je suis là .`0x20(%rsp)`- Je suis là .`0x28(%rsp)`après une .`pushq %rax`, qui est l'entrée des compensations 16, 24, 32: les logements des paramètres du deuxième, troisième et quatrième registre.`va_list`est `leaq 0x18(%rsp), %rax`, la maison de `%rdx`, le premier argument anonyme.

## AAPCS64, et ce que Apple change

AAPCS64 6.8.2 marche une liste de règles avec deux compteurs. C.9: un argument intégral "est copié aux bits les moins significatifs dans x[NGRN]" et les incréments NGRN. C.1 fait la même chose pour les arguments de point flottant avec NSRN. C.12: un composite de au plus `8 - NGRN`Les deux mots sont "copiés dans des registres consécutifs à usage général". C.13 définit "Le NGRN est réglé sur 8" une fois qu'un argument est passé à la pile, de sorte que rien ne retourne ensuite aux registres. `eight_args`utilisations `x0`à `x7`et ne touche jamais la pile.`mixed`est `w0`- Je suis là .`d0`- Je suis là .`x1`- Je suis là .`s1`- Je suis là .`by_value`est `x0`et `x1`- Dans et hors.

6.9 dit qu'un résultat qui serait passé dans les registres "est retourné dans les mêmes registres", sinon "L'adresse du bloc de mémoire doit être passée comme argument supplémentaire à la fonction dans x8". `big_ret`écrit `str x0, [x8]`- Je suis là .`stur q0, [x8, #0x8]`- Je suis là .`str x9, [x8, #0x18]`et ne écrit jamais .`x0`- Je suis là .`call_big`ensembles `mov x8, sp`, et `seed`Il reste là .`x0`, parce que 6.9 ajoute "il n'y a pas d'exigence pour le callee de préserver la valeur stockée dans x8" et `x8`n'est pas l'un des registres des arguments.

6.4.5.2: à une interface publique "SP mod 16 = 0". La pile est en pleine descente et "Aucun fil n'est autorisé à accéder (pour lire ou écrire) à la région inactive" en dessous de celle-ci, de sorte que le Linux `-O0` `leaf_add`émet`sub sp, sp, #0x10`pour 8 octets de locaux où le système V utilisait la zone rouge.

6.4.6 définit l'enregistrement du cadre: "un enregistrement du cadre de deux valeurs de 64 bits sur la pile... Le double mot adressé le plus bas doit indiquer l'enregistrement du cadre précédent et le double mot adressé le plus haut doit contenir la valeur passée dans LR lors de l'entrée dans la fonction actuelle. " et "L'emplacement du registre du cadre dans un cadre de pile n'est pas spécifié".`keeps`à`-O2`sur Linux met le record en bas avec `stp x29, x30, [sp, #-0x30]!; mov x29, sp`- La même fonction à`-O0`Il est au sommet avec `sub sp, sp, #0x20; stp x29, x30, [sp, #0x10]; add x29, sp, #0x10`- Je suis désolé .

Les notes arm64 d'Apple conservent les registres et l'alignement et modifient le cadre, la zone rouge et les appels variadiques. "Le registre du pointeur de cadre (x29) doit toujours aborder un enregistrement de cadre valide. Certaines fonctions, telles que les fonctions de feuille ou les appels de queue, peuvent choisir de ne pas créer une entrée dans cette liste. "`keeps`économiser`x22, x21, x20, x19`Il est le premier et met le record en haut: `stp x29, x30, [sp, #0x20]; add x29, sp, #0x20`"La zone rouge ARM64 est composée des 128 octets situés juste sous le pointeur de pile", et le clang est toujours émis `sub sp, sp, #0x10`pour le `-O0`Les arguments variadiques sont attribués "au nombre approprié de fentes de pile de 8 octets" et "le type va_list est un alias pour char*". `call_varargs`dans les magasins Apple 10, 20, 30 à `[sp]`- Je suis là .`[sp, #8]`- Je suis là .`[sp, #0x10]`et passe seulement `w0`; la mise en place de Linux passe `w1`- Je suis là .`w2`- Je suis là .`w3`et des appels à l' écoute avec `b extva`- La pomme .`sum_varargs`Il lit`add x8, sp, #0x10`, le pointeur de pile de l'appelant, comme son `va_list`; le Linux `sum_varargs`économiser`x1`à `x7`et `q0`à `q7`dans un cadre de 224 octets et initiale les champs de structure `__gr_offs`à -56 et `__vr_offs`à -128 avec un `mov x8, #-0x38; movk x8, #0xff80, lsl #32`- Je suis désolé .

Les symboles diffèrent aussi. Mach-O préfixe chaque symbole C avec `_`, donc la liste montre `_keeps`et la première fonction apparaît sous le label privé `ltmp0`L'analyseur renomme les étiquettes dans la table des symboles.

## Décodez un prologue à la main, trois fois

`keeps(int n)`Les appels`ext(i)`dans une boucle et continue `n`- Je suis là .`i`, et `acc`Chaque ABI doit enregistrer les registres.`-O2`construire pour chaque cible.

Système V, 4 octets:

1. `55`est `push r64`avec le registre dans les trois bits inférieurs: `50 + 5`- Je suis là .`%rbp`- La profondeur 8,`%rbp`enregistré à l'entrée de compensation -8.
2. `41 56`Le numéro de la liste:`41`est REX avec `B`l'ensemble, `56`est `50 + 6`. REX.B est étendu de 6 à 14: `pushq %r14`à -16.
3. `53`est `50 + 3`Le numéro de la liste:`pushq %rbx`à -24.

Trois pousses sont 24 octets. l'adresse de retour ajoute 8, et 32 est un multiple de 16, donc le `callq ext`à l'intérieur de la boucle est alignée sans un `subq`Il n' y en a pas .`movq %rsp, %rbp`Le corps utilise des couches de couleur.`%rbp`comme le registre sauvé par les appels qui détient `acc`(le secteur de l'énergie)`xorl %ebp, %ebp`- Je suis là .`addl %eax, %ebp`L'épilogue est le miroir:`5b 41 5e 5d c3`- Je suis désolé .

Microsoft x64, 7 octets:

1. `56`est `pushq %rsi`à -8.`57`est `pushq %rdi`à -16.`53`est `pushq %rbx`Les trois sont non volatiles ici, et`RDI`et `RSI`étaient des registres d'arguments du système V.
2. `48 83 ec 20`: REX.W, code op `83 /5`avec un 8 bits immédiat, ModRM `ec`est `11 101 100`Le numéro de la liste:`subq $0x20, %rsp`- La profondeur 56.

56 plus l'adresse de retour est 64, aligné.`ext`La page d'épilogue ne permet qu'une "ajout de RSP, constante ou lea RSP, constante [FPReg], suivie d'une série de fenêtres de registre zéro ou plus de 8 octets et d'un retour ou d'un jmp", et les octets sont `48 83 c4 20 5b 5f 5e c3`- Je suis désolé .

AArch64, 16 octets en quatre mots, petit enjeu:

1. `a9bd7bfd`est `stp`avec un pré-index.`10`sélectionnez les références 64 bits, bits 25:23 `011`sélectionnez pré-index, bit 22 est 0 pour stocker. imm7 à bits 21:15 est `1111010`Rt2 à 14:10 est 30, Rn à 9:5 est 31 (`sp`), Rt à 4:0 est 29. `stp x29, x30, [sp, #-0x30]!`: le pointeur de pile tombe à 48, `x29`passe à -48 et `x30`à moins de 40.
2. `f9000bf5`est `str`avec un offset non signé de 12 bits égalé par 8: imm12 à 21:10 est 2, Rn est `sp`, Rt est 21.`str x21, [sp, #0x10]`- Le décalage d'entrée - 32.
3. `910003fd`est `add`immédiatement: imm12 est 0, Rn est `sp`, Rd est 29.`mov x29, sp`: le pointeur de cadre pointe à -48 au record.
4. `a9024ff4`est `stp`avec un décalage signé: imm7 est 4, étalé à 32. Rt2 est 19, Rt est 20. `stp x20, x19, [sp, #0x20]`Le numéro de la liste:`x20`à -16, `x19`à -8.

Le cadre est de 48 octets: `x29`à 0, `x30`à 8 heures.`x21`à 16, 8 octets de rembourrage à 24, `x20`à 32, `x19`Le mot le plus bas du registre indique le record précédent et le mot suivant détient LR, comme l'exige 6.4.6. `cmp w0, #0x1`Le programmeur le déplace dans le prologue et l'analyseur tolère une comparaison ou une instruction arithmétique entre les opérations de pile.

## Le compilateur a écrit deux fois les mêmes faits

Un prologue parse est une dérivée. Le compilateur en émet une seconde pour le déwind, et les deux doivent être d'accord.

Sur Windows , la deuxième dérivée est `.pdata`et `.xdata`- Construisez le`-O2`objet sans `-fno-asynchronous-unwind-tables`et `llvm-objdump -h`Les spectacles`.xdata`de 0x34 octets et `.pdata`de 0x48 octets. 0x48 est 72, six `RUNTIME_FUNCTION`entrées de 12 octets avec trois `IMAGE_REL_AMD64_ADDR32NB`Les six sont des délocalisations.`sum_varargs`- Je suis là .`keeps`- Je suis là .`call8`- Je suis là .`call_varargs`- Je suis là .`call_pair`, et `call_big`. Les six autres fonctions sont des feuilles, et la page de traitement des exceptions dit "Si la recherche ne trouve pas une entrée de table de fonction, le code est supposé faire partie d'une fonction de feuille, et RSP adresse directement le pointeur de retour".

Le `.xdata`octets pour `keeps`sont`01 07 04 00 07 32 03 30 02 70 01 60`- Je suis là .`UNWIND_INFO`commence par Version en 3 bits et Flag en 5 bits: version 1, pas de flags. SizeOfProlog est 7, CountOfCodes est 4, FrameRegister et FrameOffset sont 0.`UNWIND_CODE`est un décalage dans le prologue et un octet de code op et d'information. `07 32`: à l'offset 7, op 2 est `UWOP_ALLOC_SMALL`, info 3 signifie `3 * 8 + 8 = 32`- Les octets.`03 30`: op 0 est `UWOP_PUSH_NONVOL`, le registre 3 est `rbx`- Je suis là .`02 70`Le numéro de la liste:`rdi`- Je suis là .`01 60`Le numéro de la liste:`rsi`. Les codes sont stockés dans l'ordre inverse d'exécution, et ils donnent le même nom à trois pousses et la même allocation de 32 octets que les octets `56 57 53 48 83 ec 20`- Le .`sum_varargs`l' entrée est `01 01 01 00 01 02 00 00`: un prologue de 1 octet avec un code, `UWOP_ALLOC_SMALL`C'est comme ça que le détoxificateur voit.`pushq %rax`: une allocation, car `%rax`Il est volatil et rien n'est sauvé.

Sur Linux , la deuxième dérivée est `.eh_frame`, qui est défini par la section 4.2.4 du Règlement et codifié dans la section 6.4 du Règlement DWARF 5. `llvm-dwarfdump --eh-frame`sur l' objet relocalisable imprime un FDE par fonction et résolve le `R_X86_64_PC32`Les délocalisations, les rangées sont utilisables.`keeps`- Le numéro de la liste:

```text
  DW_CFA_def_cfa_offset: +16
  DW_CFA_def_cfa_offset: +24
  DW_CFA_def_cfa_offset: +32
  DW_CFA_offset: RBX -32
  DW_CFA_offset: R14 -24
  DW_CFA_offset: RBP -16
```

Le CFA est le pointeur de pile avant l'appel, 8 au-dessus du pointeur de pile d'entrée, donc `RBP -16`est compensé par l'entrée -8, `R14 -24`est -16,`RBX -32`est -24, et le plus profond CFA de 32 est le cadre de 24 octets plus l'adresse de retour.`keeps`dit `DW_CFA_def_cfa: W29 +48`avec `W29 -48`- Je suis là .`W30 -40`- Je suis là .`W21 -32`- Je suis là .`W20 -16`- Je suis là .`W19 -8`. Là, le CFA est le pointeur de la pile d'entrée lui-même, donc les compensations sont les compensations de cadres sans conversion, et le CFA est défini à partir de `x29`pendant que le record de cadre est en direct. `eh_frame_cross_check()`Il est nécessaire de faire une comparaison pour les 12 FDE des deux objets.`call8`sous le système V pousse deux arguments sortants au milieu du corps, de sorte que son décalage CFA le plus profond est 32 tandis que son cadre de prologue est 8, et le contrôle compare le pointeur de pile le plus profond que le corps atteint, 24, plus 8.

## Comment les outils appellent ça

Chaque outil de ce domaine a une structure de données pour la convention d'appel et un passe qui en attribue une à une fonction. Les noms diffèrent et les lacunes diffèrent. Les outils fermés sont décrits à partir de leur documentation; les ouverts ont été exécutés sur les fixtures et leur sortie est capturée à côté des objets.

**Ghidra 12.1.4 (Apache-2.0).**Les règles sont dans les fichiers de spécifications du compilateur.`.cspec`, à côté du langage SLEIGH. `x86-64-gcc.cspec`déclare une `<default_proto>`nommé `__stdcall`, "Dérivé du supplément de processeur d'architecture AMD64 d'interface binaire d'application système V avril 2016", avec `<pentry>`Liste de `XMM0_Qa`à `XMM7_Qa`et `RDI, RSI, RDX, RCX, R8, R9`, une entrée de pile à l' offset 8, a `join_dual_class`règle pour les types jusqu'à 16 octets, sorties `XMM0_Qa, XMM1_Qa, RAX, RDX`, et `hidden_return`Pour le reste.`x86-64-win.cspec`déclare `__fastcall`avec quatre `<group>`éléments qui s'accouplent `XMM0_Qa`avec `RCX`- Je suis là .`XMM1_Qa`avec `RDX`, et ainsi de suite, qui est la règle de position, une entrée de pile à l'offset 40, `convert_to_ptr`pour les traits de 3, 5, 6 ou 7 octets avec la mention "seuls les traits de taille 8,16,32,64 bits peuvent être emballés pour enregistrer", `unaffected`liste `RBX RBP RDI RSI RSP R12`à `R15`et `XMM6`à `XMM15`, et une `localrange`de la pile 8 à 39, qui est le magasin d'ombres. `AARCH64.cspec`déclare `__cdecl`dont la première entrée est `x8`avec `storage="hiddenret"`Alors ...`q0`à `q7`et `x0`à `x7`- Je suis là .`convert_to_ptr`pour les textes de 17 octets ou plus, et `killedbycall`y compris `x8`avec le commentaire "réjiste de localisation des résultats indirects, qui ne se reflète pas dans la liste de pentry". L'étape finale du décompilant est nommée "Prototipe de la fonction d'établissement"; les analystes sont "ID de convention d'appel" et "ID de paramètre de décompilant"; l'option de programme "Évaluation de prototype" choisit le modèle par défaut.

**IDA Pro and IDA Free (Hex-Rays, commercial; current release 9.4, build 9.4.260714).**Le type de fonction est une déclaration C définie par SetType, et "IDA utilisera les informations de type pour commenter le démontage avec les informations sur les arguments de fonction. " Les emplacements que aucune convention standard ne produit sont écrits dans la convention de l'IDA: `int __usercall func@<ebx>(int x, int y@<esi>)`met le résultat en `EBX`et `y`dans `ESI`, et `@<^12.4>`nomme un emplacement de pile par décalage et taille. `__userpurge`C'est la même chose avec le nettoyage de la pile Callee.`__spoils`pour le jeu de clobes, `__hidden`- Je suis là .`__return_ptr`, et `__struct_ptr`Pour les arguments implicites, cette leçon se récupère manuellement.`__usercall`IDA Free ne prend en charge que x86 et x86-64.

**Binary Ninja 6.0 "Krypton" (Vector 35, commercial, with a Free edition that decompiles x86, x86_64, ARMv7, and AArch64).**Une .`Platform`"contient toutes les informations relatives à l'environnement d'exécution du binaire, principalement les conventions d'appel utilisées, le système d'exploitation et l'architecture", et la liste comprend `linux-x86_64`- Je suis là .`windows-x86_64`- Je suis là .`linux-aarch64`, et `mac-aarch64`, chacun avec un `default_calling_convention`- Je suis un .`CallingConvention`Il a`int_arg_regs`- Je suis là .`float_arg_regs`- Je suis là .`callee_saved_regs`- Je suis là .`caller_saved_regs`- Je suis là .`int_return_reg`- Je suis là .`high_int_return_reg`- Je suis là .`float_return_reg`, et deux drapeaux qui codent les différences dans cette leçon directement: `arg_regs_share_index`"Lorsque True, l'argument Nth consomme la rançon Nth des listes de registre entier et flottant, quel que soit son type", et `stack_reserved_for_arg_regs`, "l'espace d'ombre/domicile utilisé par la convention d'appel Windows x64". `eligible_for_heuristics`- Je suis là .`required_arg_regs`, et `required_clobbered_regs`Les types de plateforme sont les en-têtes C sous `types/platform/`, un par nom de plateforme.

**radare2 6.2.4 (LGPL-3.0, run here).** `afcl`répertorie les conventions que le plugin architecture connaît: `amd64 amd64syscall dlang golang golang_abi0 ms p9 reg swift`pour x86-64 et `arm64 dlang golang p9 reg swift`pour AArch64. `afc`rapporte la convention de la fonction actuelle, `afcr`des empreintes`cc.amd64.arg0: rdi`à travers `arg5: r9`et `ret0: rax`, et les profils sont des fichiers SDB nommés `cc-ARCH-BITS.sdb`- Le radar 2 est choisi .`amd64`pour l' objet ELF et `ms`pour l'objet COFF, qui est la fraction correcte. `afv`Liste des arguments et variables, et `afs`Il imprime une signature sur le Linux.`-O2`objet `afv`J' ai trouvé sept arguments pour `eight_args`et a été abandonné .`rcx`; sur l' objet Windows qu' il a trouvé `rcx, rdx, r8`et quatre fentes de pile et tombé `r9`Les deux registres manquants ne sont affichés qu' à l' intérieur d' une`lea`- Je suis en train de .`mixed`Il a rapporté `double arg1 @ xmm0, arg2 @ xmm1, arg3 @ xmm2`Il y a des problèmes avec les logiciels Linux et pas de problèmes avec Windows.`x0`à `x7`pour `eight_args`, et pour `big_ret`Il est répertorié `x0`et `double arg2 @ d1`où la fonction est `x8`et charge une constante dans `q1`- Chaque fois .`afs`La signature commence par `void`, parce que le type de retour n'est pas déduit.`-O0`, où chaque argument est stocké dans le cadre,`afv`J' ai trouvé les huit arguments de `eight_args`avec `rbp+0x10`et `rbp+0x18`Pour la paire de pile.

**rizin 0.9.1 (LGPL-3.0, run here) and Cutter 2.5.0.** `afcl`Liste `amd64 amd64syscall ms reg swift`- Je suis là .`afc`retourné `amd64`pour l'objet COFF, le profil Microsoft existe et n'a pas été sélectionné dans le format. `afi`Il a énuméré sept arguments pour `eight_args`sur Linux, encore une fois sans`rcx`, et sur Windows `r8, rcx, rdx`avec trois fentes de pile, sans `r9`et sans la fente à `0x40`- Je suis là .`afv`n' existe pas; la commande de liste est `afvl`Cutter est le front graphique sur Rizin.

**angr 10.0.3 (BSD-2, run here).** `CFGFast`suivie de `CompleteCallingConventions(recover_variables=True)`ensembles `function.calling_convention`à `SimCCSystemVAMD64`- Je suis là .`SimCCMicrosoftAMD64`ou `SimCCAArch64`, en format binaire, et `function.prototype`Il a récupéré les huit arguments de la`eight_args`sur les trois cibles, le pointeur caché de `big_ret`sur x86-64 comme `(BOT*, long long) -> BOT*`, la structure de référence parallèle des fenêtres `by_value`comme `(BOT*, BOT*) -> BOT*`, et le système V varié `sum_varargs`comme `(long long, ...)`Ses espaces sont différents des radars:`mixed`Je suis revenu comme`(long long, long long, double, float)`Dans le système V et AArch64, les registres de droite sont dans l'ordre incorrect, car la liste entière est imprimée avant la liste flottante; l'AArch64 `big_ret`Je suis revenu comme`(long long) -> double`, donc le `x8`La convention a été ratée et les magasins vectoriels ont été lus en conséquence; et l' AArch64 `sum_varargs`Je suis revenu avec 16 paramètres, l'ensemble du registre est en place.

**Others.**Hopper 6.6.0 est un démantèleur et décompilateur commercial pour macOS et Linux; sa gestion de convention n'a pas été exécutée ni documentée pour cette leçon.`frame info`et `register read`L'Explorateur de compilation montre les trois listes côte à côte à la source, ce qui est le moyen le plus rapide de confirmer une règle avant de la chercher en octets.

| Concept | Ghidra | IDA | Binary Ninja | radare2 / rizin | angr | this lesson |
|---|---|---|---|---|---|---|
| The convention object | `prototype` in a `.cspec` | calling convention keyword in the function type | `CallingConvention` on a `Platform` | `afc` name, `cc-ARCH-BITS.sdb` | `SimCC` subclass | `expected_locations(abi, name)` |
| Default per target | `<default_proto>` per cspec | compiler setting | `Platform.default_calling_convention` | `afc=`, `anal.cc` | by format | `describe_object()` |
| Argument registers | `<pentry>` list, `<group>` for positional | `@<reg>` in `__usercall` | `int_arg_regs`, `float_arg_regs`, `arg_regs_share_index` | `cc.NAME.argN` | `ARG_REGS`, `FP_ARG_REGS` | `INT_ARG_REGS`, `FLOAT_ARG_REGS` |
| Shadow store | `localrange` stack 8 to 39 | implicit | `stack_reserved_for_arg_regs` | `cc.ms.argn=stack` | `STACKARG_SP_DIFF` | `STACK_ARG_BASE[WIN64] = 40` |
| Hidden result pointer | `hidden_return`, `storage="hiddenret"` | `__return_ptr`, `__hidden` | return type too large for `int_return_reg` | not modelled | `BOT*` first argument | `HIDDEN_RETURN_REG` |
| Callee-saved | `<unaffected>` | `__spoils` lists the opposite | `callee_saved_regs` | profile | `CALLEE_CLEANUP` and arch | `CALLEE_SAVED` |
| Assigning one to a function | "Call Convention ID" analyzer, "Establish Function's Prototype" | user or type library | heuristic detection with `required_arg_regs` | `afc`, `afva`, `aaft` | `CallingConventionAnalysis` | read-before-write plus the rulebook |

## Faites-le

Exécutez la démo:

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames
python3 code/main.py
```

La sortie est un rapport JSON. Pour chacun des sept objets, il donne le format, la machine et l'ABI, et pour chacune des douze fonctions, il donne les octets du prologue, les registres enregistrés avec des compensations d'entrée, la taille du cadre, le pointeur du cadre, les registres des paramètres, les compensations d'arguments de pile, les déversements, le registre de retour ou le pointeur caché, les profondeurs d'appel, et les registres sortants et les magasins des appelants. Chaque fonction se termine par ses vérifications: la règle, la valeur attendue de la signature, la valeur observée des octets et si elles sont d'accord. Le rapport se termine par les deux contrôles croisés et les captures de l'outil. Sur les 84 analyses de fonction, 489 contrôles sont effectués et les 489 sont d'accord.

Forme attendue:

```json
{
  "objects": {
    "convention-x86_64-windows-O2": {
      "format": "COFF", "machine": "AMD64", "abi": "Microsoft x64", "checks": "73/73 agree",
      "functions": {
        "keeps": {
          "prologue": ["56  pushq %rsi", "57  pushq %rdi", "53  pushq %rbx", "48 83 ec 20  subq $0x20, %rsp"],
          "frameBytes": 56,
          "savedRegisters": [{"register": "rsi", "entryOffset": -8}, {"register": "rdi", "entryOffset": -16}, {"register": "rbx", "entryOffset": -24}],
          "parameterRegisters": ["rcx"],
          "returnRegisters": ["rax"],
          "calls": [{"callee": "ext", "depth": 56}],
          "checks": [{"rule": "Microsoft x64: the caller must always allocate ...", "expected": "at least 32 bytes ...", "observed": [56], "agree": true}, "..."]
        }
      }
    }
  },
  "sehCrossCheck": ["..."],
  "ehFrameCrossCheck": {"...": "..."}
}
```

Lire `code/main.py`dans cet ordre:

1. `describe_object()`Lire l'en-tête ELF, COFF ou Mach-O et le tableau des symboles et renvoyer l'ABI et la liste des fonctions.
2. `parse_listing()`Il divise le `llvm-objdump -d -r`Le texte est divisé en fonctions, conserve la cible de relocalisation de chaque appel et saut, et démarque les commentaires avec le marqueur de la bonne syntaxe: `#`pour AT&T, `//`et `;`pour AArch64, où `#`Il y a une situation immédiate.
3. `effect_x86()`et `effect_arm()`Transformer une instruction en lecture, adresse lit, écrit, chargement, magasins et delta de pointeur de pile. L'opérand AT&T dernier est la destination; le premier opérand AArch64 est la destination sauf dans les magasins, compares et branches. `lea`L'adresse est lue et rien ne chargé.`mov`- Je suis là .`cvtsi2sd`- Je suis là .`setcc`, et l' idiome zéro .`xor r, r`Un appel écrit l'ensemble du jeu volatil.`sp`avant l'accès et après une charge post-indexée.
4. `analyze_function()`Il marche le corps une fois jusqu' à la première.`ret`Il suit la profondeur du pointeur de pile en dessous de sa valeur d'entrée, donc une mémoire opère`disp(%rsp)`est à l' entrée compensée `disp - depth`et `disp(%rbp)`est à `frame_pointer + disp`. Il enregistre les registres de classe d'arguments qui sont lus avant qu'ils ne soient écrits, charge au premier argument de la pile ou au-dessus de celui de l'ABI, stocke les registres d'arguments dans le cadre, accède au-dessous du pointeur de pile actuel et à la profondeur à chaque appel.
5. `_return_registers()`Les données sont enregistrées dans un registre de résultats lorsque leur dernier accès avant la sortie est une écriture.`rax`vivant dans `big_ret`dans Microsoft x64, où `movq %rcx, %rax`est suivi par les magasins à travers `(%rax)`- Je suis désolé .
6. `expected_locations()`Applique les trois manuels de règles à une signature C: les deux listes de système V avec des classes de huit octets, les fentes positionnelles Microsoft avec des structures de référence parallèle et le compteur AAPCS64 avec `x8`pour les résultats supérieurs à 16 octets.
7. `run_checks()`compare les deux et ajoute les règles de cadre: les registres enregistrés par appel doivent être enregistrés, `(rsp + 8) % 16 == 0`ou `sp % 16 == 0`à chaque appel, 32 octets d'ombre à chaque appel Microsoft, aucun accès en dessous du pointeur de pile sous Microsoft et AAPCS64 et au plus 128 octets sous System V, et un épilogue qui reflète le prologue avec le pointeur de pile à sa valeur d'entrée.
8. `seh_cross_check()`et `eh_frame_cross_check()`lire les données de détournement des objets construits sans `-fno-asynchronous-unwind-tables`et comparer avec la prologue parse.

Il y a trois commandes de plus.`python3 code/main.py analyze convention-x86_64-windows-O2 keeps`imprime une fonction. `python3 code/main.py object FILE`Il imprime le verdict de l'en-tête.`python3 code/main.py seh`imprime le décodé `UNWIND_INFO`de chaque fonction qui en a une.

## Utilisez-le

Utilisez l'analyseur dans cet ordre sur un nouvel objet que vous avez construit.

1. On court .`python3 code/main.py object FILE.o`Si le chef et votre commande de construction ne sont pas d'accord, arrêtez, l'un d'eux est faux.
2. Capture `llvm-objdump -d -r FILE.o`Lisez les octets du prologue et les registres enregistrés avant tout autre, et comparez la taille du cadre avec les pousses et les `sub`- Je suis désolé .
3. Lisez les registres de paramètres et les compensations de la pile contre votre signature source. Un registre que la signature n'explique pas est un pointeur caché, un marqueur variadique ou un registre de rayures que l'analystes ont confondu avec un argument. Le dernier cas est un bug dans la table d'effets.
4. Chacun nomme la règle sur laquelle il repose, donc la première chose à vérifier est la règle, puis les octets, puis l'analyseur.
5. Lorsque l'objet a des données de détournement, effectuez la vérification croisée.
6. Lorsque vous avez installé un autre outil, saisissez sa sortie avec la commande exacte et comparez.

## Laboratoire interactif

### Laboratoire A: une instruction, trois effets

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames/code
python3 -q
```

Je vais courir .

```python
from main import *
effect(Instruction(0, "48 8d 04 0a", "leaq", "(%rdx,%rcx), %rax"), SYSV)
effect(Instruction(0, "f2 0f 2a d7", "cvtsi2sd", "%edi, %xmm2"), SYSV)
effect(Instruction(0, "a9bd7bfd", "stp", "x29, x30, [sp, #-0x30]!"), AAPCS64)
```

Écrivez les lectures, les lectures d'adresse, les écritures et le delta de pile que vous attendez avant chaque appel.`[sp, #-0x30]!`à `[sp], #0x30`avec `ldp`et regardez le delta bouger de `sp_delta`à `post_sp_delta`- Je suis désolé .

### Laboratoire B: le cadre des réserves

```python
info, listing = load_fixture("convention-aarch64-linux-O2")
keeps = analyze_function("keeps", listing["keeps"], info.abi)
keeps.saved
keeps.frame_pointer
[c.to_dict() for c in frame_cells(keeps)]
```

Comparez les compensations avec le décodeur manuel ci-dessus.`convention-arm64-macos-O2`et faire fonctionner les mêmes trois lignes; le record se déplace à -16 et quatre registres sont assis sous elle.`convention-x86_64-linux-O2-fp`et lire `saved`pour `keeps`: le pointeur de cadre est de retour, et aussi est un `pushq %rax`dont le seul travail est de faire`(rsp + 8)`Un multiple de 16 à l'appel.

### Laboratoire C: lecture avant écriture sur mélange

```python
info, listing = load_fixture("convention-x86_64-linux-O2")
mixed = analyze_function("mixed", listing["mixed"], info.abi)
mixed.param_registers
mixed.return_registers
expected_locations(SYSV, "mixed")["parameters"]
expected_locations(WIN64, "mixed")["parameters"]
```

La liste du système V est `rdi, xmm0, rsi, xmm1`et la liste de Microsoft est `rcx, xmm1, r8, xmm3`Maintenant, chargez l'objet Windows et exécutez les trois premières lignes à nouveau. Ensuite, ouvrez la capture radar2 pour l'objet Linux et trouvez`[afv mixed]`Trouvez l'instruction qui explique le troisième.

### Lab D: deuxième avis du découlant

```python
for row in seh_cross_check():
    print(row["function"], row["sizeOfProlog"], row["unwindPushes"], row["unwindAllocBytes"], row["agree"])
for row in eh_frame_cross_check("convention-x86_64-linux-O2-eh"):
    print(row["function"], row["cfiSaved"], row["cfiMaxOffset"], row["deepestStack"], row["agree"])
```

Trouvez le `pushq %rax`de `sum_varargs`Dans le tableau SEH, expliquez pourquoi le code de détournement l'enregistre comme une allocation.`call8`dans le `.eh_frame`Je vais vous expliquer le 32.

## Laboratoire de pratique

Ajoutez une quatrième cible et un quatrième manuel de règles.

Les exigences:

- Compile `convention.c`avec `-target x86_64-apple-macos -O2`et saisir sa liste.`describe_object()`Pour lire le Mach-O `cputype`0x01000007 et le cartographier au système V psABI, que Apple suit sur x86-64, avec le `_`préfixe du symbole.
- Vérifiez que chaque fonction est conforme aux contrôles du système V. Enregistrez la seule différence que vous trouvez dans les prologues: Darwin conserve le pointeur de cadre à chaque niveau d'optimisation.
- Ajouter un `-fstack-protector-strong`Élargir la numérotation du prologue pour que le chargement canarien de `%fs:0x28`et le magasin en dessous des locaux est enregistré comme un `canary`cellule, et la comparaison précédente `ret`comme le chèque qui garde l'épilogue.
- Ajoutez un test qui échoue quand un magasin de canaries est compté comme un argument déversé.
- Écrivez le cadre de `with_canary`- le code de style à la main avant d'exécuter l'analyseur, et gardez le dessin à côté du test.

Le succès signifie`python3 code/main.py`rapporte le nouvel objet avec chaque vérification d'accord, et les cellules de cadre de la fonction protégée montrent le canary entre les locaux et les registres enregistrés.

## Leçon d'artisanat

Cette leçon comprend `outputs/skill-calling-conventions-and-stack-frames.md`. Utilisez-le pour transformer un objet que vous avez construit, sa liste et les signatures C en emplacements de paramètres, une mise en page de cadre, un registre de retour et une vérification de chaque inférence contre la règle sur laquelle elle repose et contre les données de détournement du compilateur. Il nécessite d'abord le verdict ABI de l'en-tête, interdit de lire `cvtsi2sd`ou `xor r, r`Les destinations sont les mêmes, et se terminent par le premier désaccord.

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La vérification doit prouver:

- les en-têtes ELF, COFF et Mach-O des sept objets nommés System V, Microsoft x64, AAPCS64 et Apple arm64
- chaque tableau de symboles de chaque objet énumère les mêmes douze fonctions, avec le Mach-O `_`préfixe et le `ltmp0`étiquette résolue
- le parseur de l' annonce garde `ext`comme cible d'appel et maintient AArch64 immédiate comme `#-0x30`intact
- `leaq (%rdx,%rcx), %rax`lisent deux registres d' adresses et ne chargent rien; `xorl %eax, %eax`et `cvtsi2sd`sont uniquement écrits; un pré-indexé `stp`- Je suis là .`sp`avant le magasin et après l' indexation `ldp`après la charge
- `keeps`à`-O2`est `55 41 56 53`sur le système V avec un cadre de 24 octets et sans pointeur de cadre, `56 57 53 48 83 ec 20`sur Microsoft x64 avec un cadre de 56 octets, et `a9bd7bfd`premier sur AArch64 avec le record de -48
- La construction d'Apple met le record à -16 au-dessus de quatre registres enregistrés
- chaque fonction a un indicateur de cadre à `-O0`sur Linux et aucun sur `-O2`, et `-fno-omit-frame-pointer`le ramène avec une poussée d'alignement
- le `-O0`feuille utilise 8 octets de zone rouge dans le système V, `pushq %rax`sous Microsoft x64, et `sub sp, sp, #0x10`selon le règlement AAPCS64; `-mno-red-zone`supprime l'utilisation de la zone rouge
- Chaque épisode reflète son prologue et le pointeur de pile est de retour à l'entrée
- `eight_args`lisent six registres et les fentes 8 et 16, quatre registres et les fentes 40 à 64, ou `x0`à `x7`, à tous les deux niveaux d' optimisation
- `mixed`est `rdi, xmm0, rsi, xmm1`dans le cadre du système V et `rcx, xmm1, r8, xmm3`sous Microsoft x64, avec `xmm0`en conséquence sur les deux
- Le structure de 16 octets se déplace dans deux registres sous le système V et AAPCS64 et par référence derrière un pointeur caché sous Microsoft x64
- La structure de 32 octets retourne à travers `rdi`- Je suis là .`rcx`ou `x8`
- La variade de la callee est `%al`, les maisons `rdx, r8, r9`à 16, 24, 32, économise `x1`à `x7`et `q0`à `q7`, ou lis le pointeur de pile de l'appelant sur Apple
- Les appelants définissent les registres attendus, stockent 7 et 8 à 0 et 8 ou 5 à 8 à 32 à 56, zéro `%al`pour l'appel variadique, et passez l'adresse du tampon de résultat dans `rdi`- Je suis là .`rcx`ou `x8`
- Les 489 contrôles sont conformes à leurs règles.
- Les six .`UNWIND_INFO`Les entrées correspondent à la prologue parse et les feuilles n'ont pas
- Les 24 FDE correspondent aux compensations enregistrées et au pointeur de pile le plus profond
- L'outil capture les radars enregistrés.`amd64`et `ms`Le rizin.`amd64`sur l'objet COFF, le manquant `rcx`, les trois doubles, et les prototypes d'Angr
- les données figuratives intégrées dans le module de site sont égales à la sortie de l'analyseur

## Mode de défaillance de la production

| Failure | What the weak tool reports | What the analysis must prove |
|---|---|---|
| COFF object read as System V | `rdi` and `rsi` as arguments that the function never reads | The header says 0x8664; the convention is Microsoft x64 |
| Register read only inside a `lea` | Seven arguments for an eight-argument function | Base and index registers are reads |
| `cvtsi2sd` destination counted as a read | A third double argument in `xmm2` | The first write to `xmm2` is a write; `mixed` has two SSE arguments |
| Positional slots read as sequences | `xmm0` named as the double argument on Windows | Slot 1 is `XMM1`, slot 3 is `XMM3` |
| `push rbp` taken as a frame pointer | A frame pointer in an `-O2` function that uses `rbp` as an accumulator | Only `mov %rsp, %rbp` after the push creates one |
| Red zone measured from the entry stack pointer | 216 bytes "below the stack pointer" | 128 bytes below the current stack pointer after an 88-byte `sub` |
| Shadow store treated as locals | Stack arguments at 8 to 32 | Microsoft stack arguments start at 40 |
| `lea` treated as a load | A stack argument at `0x60(%rsp)` in a variadic callee | `lea` computes `overflow_arg_area`; it loads nothing |
| Hidden pointer read as a normal argument | `big_ret(void *, long long)` | The pointer is stored through and copied to `rax`, `rax`, or stays in `x8` |
| Deepest CFA compared with the prologue frame | A mismatch on a caller that pushes outgoing arguments | Compare with the deepest stack pointer the body reaches |
| Convention asserted from the ISA | `amd64` for a COFF object | The format picks the rulebook, then the ISA picks the registers |
| Prototype trusted from one tool | Three doubles, or sixteen variadic parameters | The source signature or a second derivation must agree |

## Règlement d'exploitation

Lisez l'en-tête avant la première instruction et nommez le manuel de règles. Placez chaque registre enregistré et chaque argument à un décalage du pointeur de pile d'entrée, et vérifiez-le contre la règle qui l'a mis là-bas et contre les données de décalage que le compilateur a écrites pour le même cadre. Traitez le prototype d'un outil comme une revendication jusqu'à ce que les octets ou la source le confirment.

## Pour en savoir plus

- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Version 1.0 (sections 3.2.1 to 3.2.3, 4.2.4)](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Microsoft x64 calling convention](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention)
- [Microsoft x64 stack usage](https://learn.microsoft.com/en-us/cpp/build/stack-usage)
- [Microsoft x64 prolog and epilog](https://learn.microsoft.com/en-us/cpp/build/prolog-and-epilog)
- [Microsoft x64 exception handling: RUNTIME_FUNCTION, UNWIND_INFO, UNWIND_CODE](https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64)
- [Arm Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64), release 2025Q4](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [Apple: Writing ARM64 code for Apple platforms](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)
- [DWARF Debugging Information Format Version 5, section 6.4 Call Frame Information](https://dwarfstd.org/doc/DWARF5.pdf)
- [Clang driver frame-pointer defaults, `useFramePointerForTargetByDefault` in `CommonArgs.cpp`](https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/ToolChains/CommonArgs.cpp)
- [Clang command line reference: `-fomit-frame-pointer`, `-momit-leaf-frame-pointer`](https://clang.llvm.org/docs/ClangCommandLineReference.html)
- [Fedora 38 change: frame pointers by default](https://fedoraproject.org/wiki/Changes/fno-omit-frame-pointer)
- [Ubuntu 24.04: performance engineering with frame pointers by default](https://ubuntu.com/blog/ubuntu-performance-engineering-with-frame-pointers-by-default)
- [Ghidra compiler specification: `x86-64-gcc.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/x86/data/languages/x86-64-gcc.cspec)
- [Ghidra compiler specification: `x86-64-win.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/x86/data/languages/x86-64-win.cspec)
- [Ghidra compiler specification: `AARCH64.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/AARCH64/data/languages/AARCH64.cspec)
- [Hex-Rays: SetType and the `__usercall` convention](https://docs.hex-rays.com/ida-actions/settype)
- [Hex-Rays: Decompile as call](https://docs.hex-rays.com/ida-actions/hx_userdefinedcall)
- [Binary Ninja API 6.0: `callingconvention` module](https://api.binary.ninja/binaryninja.callingconvention-module.html)
- [Binary Ninja API 6.0: `platform` module](https://api.binary.ninja/binaryninja.platform-module.html)
- [Binary Ninja user documentation: platform types](https://docs.binary.ninja/guide/types/platformtypes.html)
- [The Official Radare2 Book: calling conventions](https://book.rada.re/analysis/calling_conventions.html)
- [The Official Radare2 Book: code analysis](https://book.rada.re/analysis/code_analysis.html)
- [angr documentation: decompiler analysis passes, including calling convention recovery](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [radare2 6.2.4 release](https://github.com/radareorg/radare2/releases/tag/6.2.4)
- [rizin 0.9.1 release](https://github.com/rizinorg/rizin/releases/tag/v0.9.1)
