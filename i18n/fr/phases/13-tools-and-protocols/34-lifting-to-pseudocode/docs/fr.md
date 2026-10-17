# Leur mise en place est un processus de mise en œuvre de la stratégie de référencement.

> La sortie de décompilateur est une revendication avec une dérivation cachée. Lorsque vous construisez la dérivation vous-même, chaque ligne de cette sortie devient une chose que vous pouvez vérifier.

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~150 minutes

## Objectifs d'apprentissage

- L'ascenseur décode les instructions x86-64 dans une petite IR avec des drapeaux explicites, des charges, des magasins, des succursales, des appels et des retours.
- Transformer les espaces de cadre en variables locales et les registres d'arguments du système V en paramètres.
- Construisez un SSA minimal avec des fonctions phi sur les frontières de domination, puis remplacez, simplifiez et supprimez le code mort.
- Récupérer `if`- Je suis là .`if`avec `else`- Je suis là .`while`, et `do while`du graphique de contrôle des flux avec des post-dominateurs et des bords arrière.
- Expliquez le multiplication par `0x66666667`comme une division par 10 et prouver l'identité.
- Lisez le pseudocode Ghidra pour les mêmes fonctions ligne par ligne et nommez l'instruction, ou la devinette, derrière chaque différence.

## Le problème

Vous avez demandé à Rea le pseudocode de `percent_change`dans `ledger-x86_64-O1.o`Ghidra 12.1.4 a répondu en dix secondes:

```c
undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)

{
  long lVar1;
  undefined1 auVar2 [16];
  undefined1 auVar3 [16];
  
  if (param_1 != 0) {
    lVar1 = (long)((param_2 - param_1) * 100);
    auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
    auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
    return auVar3;
  }
  auVar2._8_8_ = 0;
  auVar2._0_8_ = param_3;
  return auVar2 << 0x40;
}
```

La source est de quatre lignes.`int`valeurs et rendements un `int`. Ghidra rapporte une valeur de retour de 16 octets et un troisième paramètre que la source ne dispose pas. Les seize octets de code machine sont corrects, et Ghidra a décodé chacun d'eux correctement. La signature est une devinette, et rien dans le texte ne le marque comme un.

réa renvoie ce texte comme le `procedure_pseudo_code`La limite est "Pseudocode est la sortie du décompilateur Ghidra, pas source originale et pas texte équivalent à la sortie Hopper".`compare_functions`L'outil diffère du pseudocode en tant que lignes de texte. Son propre enregistrement de dimension dit "Assembly is opaque provider text" et que la normalisation de relocation est indisponible.

Le premier résultat de la décompilée était`FUN_00406400(void)`Je vous appelle .`__ftol()`L'interface de l'instruction affichait une entrée de pile et une expression x87.

Cette leçon construit la dérivation que le décompilateur cache. Vous commencez par les 56 instructions que la leçon 33 décode, les relevez à un IR, renommez en SSA, simplifiez et structurez. Ensuite, vous placez votre pseudocode à côté de celui de Ghidra pour`percent_change`- Je suis là .`grade`, et `total`Les tests compilent votre sortie avec le clang et le font contre la source sur une grille d'entrée.

```figure
lifting-stages-percent-change
```

## La fixation et l'oracle

`code/fixtures/ledger.c`est le fichier partagé de cette partie du cours:

```c
int percent_change(int before, int after) {
    if (before == 0) return 0;
    return (after - before) * 100 / before;
}

int grade(int score) {
    switch (score / 10) {
    case 10:
    case 9: return 4;
    case 8: return 3;
    case 7: return 2;
    case 6: return 1;
    default: return 0;
    }
}

int total(const int *values, int count) {
    int sum = 0;
    for (int i = 0; i < count; i++) sum += values[i];
    return sum;
}
```

`grade_label`Les trois fonctions ci-dessus sont celles que cette leçon teste.

L' objet a été construit avec `clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c ledger.c`- Le .`-O0`l'objet est placé à côté pour la section de la fente de pile. `code/fixtures/README.md`Il contient les deux commandes, le SHA-256 de chaque objet, et la commande réa qui a produit les fixations Ghidra.

L' oracle du décodeur est `llvm-objdump -d -r`Le texte pour les deux objets est engagé comme`ledger-x86_64-O1.objdump.txt`et `ledger-x86_64-O0.objdump.txt`Un test analyse ces fichiers et compare chaque instruction par adresse, par octets et par texte AT&T. Cela couvre les 56 instructions à l'adresse `-O1`et chaque instruction à `-O0`- Quand ?`llvm-objdump`est installé sur le chemin fixé, un deuxième test le fait vivre et compare à nouveau.

Je vous en prie .`percent_change`Comme l' oracle l' imprime:

```text
0000000000000000 <percent_change>:
       0: 85 ff                        	testl	%edi, %edi
       2: 74 09                        	je	0xd <percent_change+0xd>
       4: 29 fe                        	subl	%edi, %esi
       6: 6b c6 64                     	imull	$0x64, %esi, %eax
       9: 99                           	cltd
       a: f7 ff                        	idivl	%edi
       c: c3                           	retq
       d: 31 c0                        	xorl	%eax, %eax
       f: c3                           	retq
```

## Étape 1: Bytes à l'instruction

La leçon 33 a construit le décodeur. Cette leçon garde une petite copie de sorte que le code fonctionne par lui-même. La copie couvre exactement les opcodes qui ont été émis pour`ledger.c`à`-O0`et `-O1`Pour tout ce qui soulevait .`DecodeError`avec l'adresse et le octet opcode.

Le décodeur suit l'ordre des octets du SDM Intel, volume 2, section 2.1. Les préfixes viennent en premier, puis le code op, puis ModR/M, puis SIB, puis le déplacement, puis le immédiat.`0x40`à `0x4F`, qui doit être directement devant le code op. REX.W sélectionne un opérand de 64 bits, REX.R étend le `reg`dans le champ, REX.X élargit l'indice SIB et REX.B élargit `r/m`La section 2.1.3 définit les trois champs ModR/M `mod`- Je suis là .`reg`, et `r/m`, et les trois champs de SIB `scale`- Je suis là .`index`, et `base`En mode 64 bits,`mod = 00`avec `r/m = 101`déplaceur de 32 bits par rapport à l'instruction suivante (section 2.2.1.6).

| Opcode bytes | Form | Fixture example |
|---|---|---|
| `01 29 31 39 85 89` | `op Ev, Gv` | `29 fe` `subl %edi, %esi` |
| `03 2b 3b 8b 8d` | `op Gv, Ev` | `03 04 97` `addl (%rdi,%rdx,4), %eax` |
| `63` with REX.W | `movsxd Gv, Ev` | `48 63 c7` `movslq %edi, %rax` |
| `69` and `6b` | `imul Gv, Ev, imm` | `48 69 c0 67 66 66 66` |
| `83 /0 /5 /7` | `add sub cmp Ev, imm8` | `83 c1 fa` `addl $-0x6, %ecx` |
| `c1 /5 /7` | `shr sar Ev, imm8` | `48 c1 f9 22` `sarq $0x22, %rcx` |
| `f7 /7` | `idiv Ev` | `f7 ff` `idivl %edi` |
| `ff /0 /4` | `inc Ev`, `jmp Ev` | `48 ff c2` `incq %rdx` |
| `70` to `7f`, `eb`, `e8` | `jcc rel8`, `jmp rel8`, `call rel32` | `74 09` `je 0xd` |
| `99`, `c3`, `90`, `0f 1f` | `cltd`, `retq`, `nop` | `66 0f 1f 44 00 00` `nopw (%rax,%rax)` |

Le `leaq (%rip), %rcx`à`grade+0x23`est de 7 octets avec un déplacement zéro.`R_X86_64_PC32.rodata+0x10`à l' équilibre `0x36`Le soulèverain le résolve à l'étape 3.

## Étape 2: Blocs et dominateurs

Un bloc de base est une séquence d'énoncés avec une entrée en haut et une sortie en bas.`build_blocks()`collecte les leaders: l'entrée de la fonction, chaque cible de saut à l'intérieur de la fonction, et chaque adresse après un saut ou un retour. Chaque bloc obtient des successeurs de sa dernière instruction.`jmp`Il y en a un.`ret`Il n'en a pas, et un bloc qui se termine par une instruction ordinaire passe au prochain leader.

`total`à`-O1`Il est composé de cinq blocs:

| Block | Instructions | Successors |
|---|---|---|
| `0x40` | `testl %esi, %esi` then `jle 0x5c` | `0x5c`, `0x44` |
| `0x44` | `movl %esi, %ecx`, `xorl %edx, %edx`, `xorl %eax, %eax`, `nopw` | `0x50` |
| `0x50` | `addl (%rdi,%rdx,4), %eax`, `incq %rdx`, `cmpq %rdx, %rcx`, `jne 0x50` | `0x50`, `0x5b` |
| `0x5b` | `retq` | exit |
| `0x5c` | `xorl %eax, %eax` then `retq` | exit |

`immediate_dominators()`L'algorithme de l'interface est l'algorithme itératif de Cooper, Harvey et Kennedy (2001). Il passe les blocs dans un ordre post-revers et intersecte les dominateurs des prédécesseurs traités jusqu'à ce que rien ne change.`total`Il est écrit que`0x44`dominent`0x50`Il dit aussi que`0x50`Il se domine à travers le bord arrière, c'est ainsi que la phase 8 trouve la boucle.

La même fonction fonctionne deux fois plus. `dominance_frontiers()`utilise pour l' ASS à l' étape 5. `post_dominators()`Il le fait sur le graphique inversé avec un nœud de sortie virtuel pour trouver des joints à l'étape 8.

## Étape 3: Levé à une petite IR

L' IR dispose de six types de déclarations: `Assign`- Je suis là .`Store`- Je suis là .`Branch`- Je suis là .`Jump`- Je suis là .`IndirectJump`, et `Return`. Les expressions sont constantes, variables, opérations binaires avec une largeur, casts (`sext`- Je suis là .`zext`- Je suis là .`trunc`Il existe cinq formes supplémentaires qui seront ensuite simplifiées:`Flags`- Je suis là .`Cond`- Je suis là .`Concat`- Je suis là .`SignFill`, et `Symbolic`- Je suis désolé .

Les registres sont des variables nommées par leur famille de 64 bits.`%eax`est une `Assign`à `rax`avec une largeur de 32 et une lecture de `%eax`est une lecture de `rax`En mode 64 bits, une écriture 32 bits s'étend à zéro dans le registre complet, de sorte qu'une lecture ultérieure de 64 bits de la même valeur est une `zext`Le lecteur ne connaît pas la définition de la portée, il enregistre donc la largeur de lecture et permet à SSA d'insérer le moule à l'étape 5.

| Instruction | IR statements |
|---|---|
| `subl %edi, %esi` | `flags = cmp(esi, edi)` then `esi = esi - edi` |
| `testl %edi, %edi` | `flags = test(edi, edi)` |
| `je 0xd` | `if je(flags) goto 0xd else goto 0x4` |
| `xorl %eax, %eax` | `eax = 0` |
| `cltd` | `edx = signfill(eax)` |
| `idivl %edi` | `dividend = edx:eax`, `eax = dividend / edi`, `edx = dividend % edi` |
| `movslq %edi, %rax` | `rax = sext64(edi)` |
| `sarq $0x22, %rcx` | `rcx = rcx >>s 34` |
| `shrq $0x3f, %rax` | `rax = rax >>u 63` |
| `addl (%rdi,%rdx,4), %eax` | `flags = add(eax, load32(rdi + rdx * 4))` then `eax = eax + load32(rdi + rdx * 4)` |
| `leaq (%rip), %rcx` with its relocation | `rcx = &.rodata+0x14` |
| `pushq %rbp`, `movq %rsp, %rbp`, `subq $0x10, %rsp`, `popq %rbp` | nothing, frame bookkeeping |
| `retq` | `return rax` |

### Les drapeaux sont une variable

Chaque instruction qui définit les codes de condition écrit une variable nommée `flags`, et chaque saut conditionnel le lit.`cmp a, b`écrit `cmp(a, b)`, une `test a, b`écrit `test(a, b)`, et `sub`écrit la même chose `cmp`La forme comme effet secondaire.`jcc`devient `Branch(Cond(cc, flags), target, fallthrough)`Le code de condition décide de l'opérateur de comparaison lorsque les deux se rencontrent à l'étape 6:

| Code | After `cmp(a, b)` | After `test(a, a)` |
|---|---|---|
| `je` | `a == b` | `a == 0` |
| `jne` | `a != b` | `a != 0` |
| `jl`, `jge`, `jle`, `jg` | signed `<`, `>=`, `<=`, `>` | signed compare with 0 |
| `jb`, `jae`, `jbe`, `ja` | unsigned `<`, `>=`, `<=`, `>` | unsigned compare with 0 |

`xorl %eax, %eax`est soulevé directement à `eax = 0`Les drapeaux qu' il dépose sont écrits comme:`add`Les drapeaux de style qui ne sautent pas dans le fichier.`xor`L' idiome est important dans `grade`, où Ghidra l'a perdu.

### L'équipement de transport

`cltd`remplir`%edx`avec le signe de `%eax`- Je suis là .`idivl %edi`divise la paire de 64 bits `%edx:%eax`par `%edi`, met le quotient dans `%eax`, et le reste en `%edx`Le soulèvent écrit la paire dans un nom temporaire`dividend`La phrase restante est ensuite la vieille.`%eax`Dans le SSA, la moitié supérieure apparaît comme`signfill`Le simplificateur réduit la paire à la moitié inférieure, et la division devient une 32-bit `/`- Je suis désolé .

### Les emplacements à -O0

À `-O0`Clang garde tous les locaux dans le cadre.`percent_change`magasins `%edi`à `-0x8(%rbp)`et `%esi`à `-0xc(%rbp)`Le soulevant les surveille.`pushq %rbp`suivie de `movq %rsp, %rbp`Après ce prologue, une mémoire avec base.`%rbp`, sans indice, et un déplacement devient une variable nommée `local_<hex>`- Je suis là .`-0x8(%rbp)`est `local_8`- Je suis là .`-0xc(%rbp)`est `local_c`- Je suis désolé .

Les instructions de cadre elles-mêmes ne produisent pas d'IR, et ne le font pas non plus `subq`et `addq`sur`%rsp`Tout autre opéra de mémoire reste un`Load`ou `Store`- Je suis désolé .

### Relocations

Le `leaq (%rip), %rcx`dans `grade`Le déplacement à l'offset`0x36`dit `R_X86_64_PC32.rodata+0x10`Le lien écrira`S + A - P`Dans le champ.`S`est l'adresse de `.rodata`- Je suis là .`A`est l' additif `0x10`, et `P`L'instruction compute ensuite`next_ip + disp = (P + 4) + S + A - P = S + A + 4`- Je suis désolé .

Le lecteur de calcul utilise la même arithmétique sans lien.`0x10 + 4 = 0x14`- L' IR dit`rcx = &.rodata+0x14`, et la table là-bas tient les cinq`int`valeurs `1, 2, 3, 4, 4`- Je suis désolé .

## Étape 4: Parametres de la convention d'appel

Le système V AMD64 psABI, section 3.2.3, passe les arguments de classe INTEGER en `%rdi`- Je suis là .`%rsi`- Je suis là .`%rdx`- Je suis là .`%rcx`- Je suis là .`%r8`- Je suis là .`%r9`Le résultat est renvoyé en`%rax`et puis ...`%rdx`La section 3.2.1 précise que `%rbp`- Je suis là .`%rbx`, et `%r12`à travers `%r15`Tout le reste appartient à l'appelant.

`collect_params()`Il est vrai qu'une règle est appliquée: un registre d'arguments qui est lu avant toute écriture dans la fonction est un paramètre.`%rsi`et jamais`%rdi`Il a encore deux paramètres.`percent_change`Il lit`%edi`et `%esi`, donc il a .`param_1`et `param_2`- Je suis là .`grade`Il ne lit que `%edi`- Je suis là .`total`Il lit`%rdi`comme base de 64 bits et `%esi`comme un compte de 32 bits.

Maintenant, c'est Ghidra qui est là.`param_3`Il y a une explication.`idivl`écrit `%rdx`, et le psABI permet un retour à 128 bits INTEGER en `%rax:%rdx`Dans un fichier objet il n'y a pas d'appelant pour dire combien large est la valeur de retour.

Sur le chemin de zéro,`xorl %eax, %eax`Il écrit seulement .`%eax`Ghidra a lu .`%rdx`Il y a une entrée de la fonction, et cette entrée est devenue `param_3`La signature est une devinette sur les parties que les instructions laissent ouvertes.

## Étape 5: SSA, Un nom par définition

Après avoir levé,`grade`écrit `rax`quatre fois et `rcx`- Je l'ai fait trois fois. Cytron, Ferrante, Rosen, Wegman et Zadeck (1991) définissent la forme de l'affectation statique unique. Chaque variable est attribuée exactement une fois, et les fonctions phi aux points de jonction sélectionnent parmi les définitions qui les atteignent. Leur construction a deux étapes. La section 5.1 place les fonctions phi à l'aide de frontières de domination, et la section 5.2 renommée chaque définition et utilisation avec une marche sur l'arbre dominateur.`insert_phis()`et `rename()`Suivez ces deux sections.

L' inscription de l' ASS `total`montre ce que produit la construction:

```text
block 0x44 -> 0x50
  ecx_1 = esi_0
  edx_1 = 0
  eax_1 = 0
block 0x50 -> 0x50, 0x5b
  eax_2 = phi(0x44: eax_1, 0x50: eax_3)
  rdx_2 = phi(0x44: edx_1, 0x50: rdx_3)
  flags_2 = phi(0x44: flags_1, 0x50: flags_5)
  flags_3 = add(eax_2, load32(rdi_0 + rdx_2 * 4))
  eax_3 = eax_2 + load32(rdi_0 + rdx_2 * 4)
  flags_4 = inc(rdx_2, 1)
  rdx_3 = rdx_2 + 1
  flags_5 = cmp(zext64(ecx_1), rdx_3)
  if jne(flags_5) goto 0x50 else goto 0x5b
```

La version 0 d'un registre est sa valeur à l'entrée. `esi_0`et `rdi_0`Les paramètres sont les paramètres.`0x50`Il a deux prédécesseurs, le bloc de configuration et lui-même, donc chaque variable avec une définition dans la boucle obtient un phi là.`flags_2`Le SSA le place dans le cas où il serait mort.`flags`Il a des définitions dans les deux prédécesseurs, et la phase 6 le supprime parce que rien ne le lit.

La modification de nomme répare également les largeurs. `edx_1 = 0`est une définition de 32 bits et `rdx_3 = rdx_2 + 1`est de 64 bits, donc le phi `rdx_2`La lecture de la lettre est plus large.`%rcx`dans `cmpq %rdx, %rcx`est de 64 bits alors que sa définition `ecx_1 = esi_0`est de 32 bits, donc l'utilisation devient `zext64(ecx_1)`Chaque personnage du pseudocode final vient de cette règle.

La partie "light" est la sortie de l'ASS. `coalesce_webs()`Le système de calcul de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la

## Étape 6: Simplifier, remplacer, supprimer le code mort

`optimize()`Le simplificateur réécrit les expressions vers le bas avec ces règles:

| Before | After |
|---|---|
| `Cond(je, cmp(a, b))` | `a == b`, and the same for the other nine codes |
| `Cond(jle, test(a, a))` | `a <= 0` |
| `x + (-6)` | `x - 6` |
| `trunc32(zext64(x))` where `x` is 32-bit | `x` |
| `trunc32(a + b)` on 64-bit operands | `trunc32(a) + trunc32(b)` |
| `(signfill(x): x) / d` | `x / d` at width 32 |
| `load32(base + idx * 4)` | `base[idx]` |
| `trunc32(p >>s 34) + trunc32(p >>u 63)` with `p = sext64(x) * 0x66666667` | `x / 10` |

La substitution avant déplace une définition dans son seul usage. La règle a trois conditions. L'utilisation est dans le même bloc. L'utilisation est une déclaration et non un phi. Lorsque la définition contient une charge, aucun magasin ou appel ne se trouve entre la définition et l'utilisation. Une constante ou une adresse de section est substituée n'importe où sauf dans un phi, car un argument phi doit rester une variable pour le web.

Le code mort supprime une assignation ou un phi dont la destination n'est pas utilisée et dont l'expression n'a pas d'appel.`percent_change`le reste `edx_2 = dividend_1 % edi_0`Il est mort, donc il va en premier.`edx_1 = signfill(eax_1)`Il est donc remplacé par `dividend_1 = edx_1:eax_1`- Je suis désolé .

La règle de la paire réduit alors`signfill(eax_1):eax_1`à `eax_1`, et la division devient `eax_1 / edi_0`- Maintenant .`eax_1 = esi_1 * 100`a une seule utilisation et `esi_1 = esi_0 - edi_0`Les deux sont utilisés dans la déclaration de retour.`flags`Les définitions ont été remplacées par les conditions de branche et disparaissent.

```c
return (param_2 - param_1) * 100 / param_1;
```

### Le 0x66666667 est multiplié par

`grade`Divise le score par 10 et le clang émet non `idiv`Il émet .`imulq $0x66666667, %rax, %rax`Alors ...`sarq $0x22, %rcx`Dans une copie, alors `shrq $0x3f, %rax`Le compilateur choisit un multiplicateur.`m`et un quart .`s`Pour chaque`x`dans la gamme, `(x * m) >> s`doit être égale `x / d`- Je suis désolé .

Je vous en prie .`m = 0x66666667 = 1717986919`et `s = 34`Les calculs sont corrects:`2^34 / 10 = 1717986918.4`, et le plafond est `1717986919`L' erreur est`m * 10 - 2^34 = 6`Pour que l' identité soit conservée sur tous les 32 bits .`x`, les temps d' erreur `2^31`Il doit rester en dessous.`2^34`, donc l'erreur doit être au maximum 8.`(x * m) >> 34`égale `floor(x / 10)`pour chaque non négatif `int`- Je suis désolé .

Pour négatif `x`Le plancher tourne à l'écart de zéro tandis que le C tourne vers zéro. `shrq $0x3f`extrait le bit de signalisation du produit, qui est 1 exactement quand `x`L'ajout corrige l'arrondissement.

`magic_divisor(m, s)`Il teste une paire de candidats.`d`comme le quotient entier de `2^s`par `m`et essaie`d`et `d + 1`Il accepte un`d`seulement lorsque `m`est égal au plafond de `2^s / d`et l'erreur de la liaison est maintenue.`s = 33`donne`d = 5`, ce qui est exact, parce que `0x66666667`est aussi le plafond de `2^33 / 5`- Un test vérifie l' identité contre`c_div(x, 10)`Les valeurs de l'indice de valeur sont les valeurs de l'indice de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de valeur de .`int`La portée.

Avec reconnaissance,`grade`Il lit`ecx_1 = param_1 / 10 - 6;`Avec la reconnaissance désactivée, la démo imprime la forme brute:

```c
rax_1 = (long)param_1 * 0x66666667;
ecx_1 = (int)(rax_1 >> 34) + (int)((unsigned long)rax_1 >> 63) - 6;
```

Les deux compiler et les deux calculer les mêmes valeurs. Gardez le commutateur. Lorsque vous lisez le code d'un compilateur que vous n'avez pas étudié, la forme brute est la plus honnête.

## Étape 7: Les types d'utilisation

Le code machine a des largeurs et aucun type. `infer_types()`Il est possible de faire des tests de la taille de la toile, en utilisant les données de la toile.`int`, et un web 64 bits est `long`Un web qui est la base d' un accès à un élément`base[idx]`avec des éléments de 4 octets est `int *`Le type pointer se répand sur les copies et les phis, donc `local_8 = param_1`à`-O0`fait les deux `int *`. Le type de retour est la largeur de la valeur dans le `return`Les déclarations.

Le résultat pour `total`est `int total(int *param_1, int param_2)`Ghidra a écrit:`int total(long param_1,uint param_2)`Le soulevant a utilisé la charge à l'échelle comme indice.`*(int *)(param_1 + uVar2 * 4)`- Je ne vois pas non plus .`const`, et aucun ne peut le dire .`int *`de `unsigned *`- Je suis désolé .

Lee, Avgerinos et Brumley (TIE, NDSS 2011) récupèrent le type de cadre en tant que résolution de contraintes sur un réseau. Le résultat est une limite supérieure et une limite inférieure au lieu d'une hypothèse. Noonan, Loginov et Cok (Retypd, PLDI 2016) ajoutent polymorphisme et récursion.`const`Cette leçon utilise les deux règles les moins chères de cette ligne de travail et nomme ce qu'ils ne peuvent pas voir.

L'arithmétique non signée est conservée dans les expressions et non dans les types de variables. `ja`après `cmpl $0x4, %ecx`des empreintes comme `(unsigned int)ecx_1 > 4`, et `shrq`des empreintes comme `(unsigned long)rax_1 >> 63`. Les variables restent signées de sorte que le lecteur voit la valeur du registre et le seul endroit où l'instruction la traite comme non signée.

## Étape 8: Structuration à partir du graphique

Cifuentes (1994), chapitre 6, correspond aux régions du graphique par rapport aux formes d'un langage de haut niveau. Les boucles sont les premières (6.6.1), puis les conditionnels bidirectionnels (6.6.2), puis les conditionnels n-directionnels (6.6.3). Yakdan, Eschweiler, Gerhards-Padilla et Smith (NDSS 2015) prennent l'autre route. Ils dérivent la condition dans laquelle chaque nœud exécute et construisent la structure à partir de ces conditions, sans formes fixes. Leur motivation est une mesure. Hex-Rays sur un échantillon produit 1,571 `goto`Les déclarations de 49 514 lignes, une pour 32 lignes, et leur structuration n'ont produit aucune des 1 530 fonctions de base.

`structure()`Il faut deux faits à partir de l'étape 2.`Branch`La boucle naturelle est l'ensemble de blocs qui atteignent la source de la borne arrière sans passer l'en-tête.

Pour une`Branch`avec joint `j`- Le numéro de la liste:

- Si vous`j`est la cible de saut, émettez `if (!cond) { fallthrough region }`et continuer à `j`- Je suis désolé .
- Si vous`j`est la chute, émet`if (cond) { target region }`et continuer à `j`- Je suis désolé .
- Si vous`j`est la sortie de fonction, les deux bras reviennent.`if (cond) {... }`et l'autre bras après lui, sans un `else`C' est le retour précoce .`percent_change`et `total`- Je suis désolé .
- Sinon émettez`if (cond) { target } else { fallthrough }`et continuer à `j`- C' est ça .`percent_change`à`-O0`, où les deux armes se stockent `local_4`et nous nous retrouverons à l'épilogue.

Pour un en-tête de boucle:

- Si l' en-tête se termine par un `Branch`avec un successeur en dehors de la boucle, émettez `while (cond) { body }`- Je suis là .`total`à`-O0`a cette forme: l'en-tête compare `local_14`avec `local_c`et les feuilles en .`jge`- Je suis désolé .
- Si l'en-tête est également le bloc avec le bord arrière, ou le verrou sort à la place, émettez `do { body } while (cond);`- Je suis là .`total`à`-O1`a cette forme: bloc `0x50`ajoute, augmente, compare et saute à elle-même.

Un `IndirectJump`finit un bloc sans successeur. `grade`à`-O0`est une vraie table de saut,`jmpq *%rax`à travers cinq compensations relatives à `.rodata+0x0`La leçon 33 résout ces cibles.`0x76`- Je suis là .`0x7f`- Je suis là .`0x88`, et `0x91`Les résultats de la recherche ont été obtenus en raison de la présence de la présence de la même substance dans les cellules.`switch`- Je suis désolé .

### Que fait -O1 au commutateur ?

À `-O1`Clang n' a pas gardé une table de saut pour `grade`- Il a calculé .`score / 10 - 6`Il a ensuite chargé la valeur de retour d'un tableau de 5 entrées de `int`à`.rodata+0x14`La comparaison non signée est une instruction qui couvre les deux.`score / 10 < 6`et `score / 10 > 10`Une différence négative est une valeur non signée importante, elle échoue donc au même contrôle.

La table tient .`1, 2, 3, 4, 4`Pour les quotients 6 à 10, le `switch`avec cinq `return`Les déclarations sont devenues `if`Le soulevageur émet exactement ça, et nomme la table.`lookup_14`après son décalage de section.

La longueur de la table est une heuristique. `table_values()`Les lectures de la table commencent à la suivante `.rodata`compenser que tout `.text`Les références de relocation ou à la fin de la section.`0x28`, la table de `grade_label`Je suis un homme .`lookup_14`Il a`(0x28 - 0x14) / 4 = 5`Les entrées. la comparaison non signée contre 4 est d'accord. Ghidra ne sait pas la longueur non plus.`*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`, une adresse en octets plus un décalage, et laissez les cinq valeurs pour le lecteur de regarder vers le haut.

## Avec Ghidra

La démo imprime chaque paire.`code/fixtures/ghidra/*.json`Les preuves montrent que Rea est revenu le 10-14 du 2026 .`npx -y rea-agents@6.5.0 decompile ledger-x86_64-O1.o <function> --provider ghidra --format json`Chaque disque porte`confidence: observed`et `authority: shipped-artifact`Les revendications contenues dans le texte sont dérivées, et les dossiers ne disent pas lesquelles.

### pourcentage_changement

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
int percent_change(int param_1, int param_2)   | undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)
{                                              | {
  if (param_1 == 0) {                          |   long lVar1;
    return 0;                                  |   undefined1 auVar2 [16];
  }                                            |   undefined1 auVar3 [16];
  return (param_2 - param_1) * 100 / param_1;  |   if (param_1 != 0) {
}                                              |     lVar1 = (long)((param_2 - param_1) * 100);
                                               |     auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
                                               |     auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
                                               |     return auVar3;
                                               |   }
                                               |   auVar2._8_8_ = 0;
                                               |   auVar2._0_8_ = param_3;
                                               |   return auVar2 << 0x40;
                                               | }
```

| Difference | Instruction or guess |
|---|---|
| Return type `undefined1 [16]` against `int` | Guess. `idivl` writes `%rax` and `%rdx`, and no caller fixes the return width. Both registers were kept. |
| `param_3` | Guess that follows from the first. On the zero path only `%eax` is written, so `%rdx` is read as an input. |
| `lVar1 / (long)param_1 & 0xffffffff` against `/ param_1` | Instruction. `cltd` plus `idivl` is a 64-bit division of a sign-extended 32-bit value, truncated to 32 bits. Both texts describe it. Ours reduced the pair with the `signfill` rule. |
| `lVar1 % (long)param_1` kept | Follows from the return type. The remainder in `%rdx` is dead in our analysis and alive in Ghidra's. |
| `if (param_1 != 0)` with the zero case after it | Instruction order. `je 0xd` jumps to the zero return. Ghidra put the computed arm first, we put the smaller arm first. Both are correct. |
| `return auVar2 << 0x40` | Guess. A 16-byte value with `param_3` in the low half, shifted into the high half, is how the decompiler expresses `%rax = 0` with `%rdx` unknown. |

La version réparée en `code/fixtures/ghidra/ledger-x86_64-O1.ghidra-repaired.c`La fonction de l'indicateur est la même que celle de la source de l'indicateur.

### de qualité

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
static const int lookup_14[5] = {1, 2, 3, 4, 4}; | uint grade(int param_1)
int grade(int param_1)                         | {
{                                              |   uint uVar1;
  int eax_1;                                   |   uint uVar2;
  int ecx_1;                                   |   
                                               |   uVar2 = param_1 / 10 - 6;
  ecx_1 = param_1 / 10 - 6;                    |   uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);
  eax_1 = 0;                                   |   if (uVar2 < 5) {
  if ((unsigned int)ecx_1 <= 4) {              |     uVar1 = *(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4);
    eax_1 = lookup_14[(unsigned long)(unsigned int)ecx_1]; |   }
  }                                            |   return uVar1;
  return eax_1;                                | }
}                                              |
```

| Difference | Instruction or guess |
|---|---|
| `param_1 / 10 - 6` in both | Instruction. Both recognized the `0x66666667` idiom. |
| `uVar1 = -(param_1 >> 0x1f) ^ (uint)(... >> 0x3f)` against `eax_1 = 0` | Instruction `xorl %eax, %eax` at `grade+0x1a`. The register is zero. Ghidra expressed the `xor` as the sign of `param_1` exclusive-or the sign bit of the product, two spellings of the same bit, and did not cancel them. The value is 0 for every `int`, and the test proves it over 141 scores. A reader who trusts the text believes the default depends on the score. |
| `uVar2 < 5` against `(unsigned int)ecx_1 <= 4` | Instruction `cmpl $0x4, %ecx` then `ja`. Same condition. Ghidra typed `uVar2` as `uint`, we cast at the use. |
| `*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)` against `lookup_14[...]` | Instruction plus relocation. Ghidra loaded the object with `.rodata` at `0x001000a0`, so `0x001000b4` is `.rodata + 0x14`. It applied the relocation and kept a byte address. We named the table and read its five values. |
| `uint` return against `int` | Guess on both sides. The instructions only say 32 bits. |

### le total

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
int total(int *param_1, int param_2)           | int total(long param_1,uint param_2)
{                                              | {
  int eax_1;                                   |   int iVar1;
  int ecx_1;                                   |   ulong uVar2;
  long rdx_1;                                  |   
                                               |   if (0 < (int)param_2) {
  if (param_2 <= 0) {                          |     uVar2 = 0;
    return 0;                                  |     iVar1 = 0;
  }                                            |     do {
  ecx_1 = param_2;                             |       iVar1 = iVar1 + *(int *)(param_1 + uVar2 * 4);
  rdx_1 = 0;                                   |       uVar2 = uVar2 + 1;
  eax_1 = 0;                                   |     } while (param_2 != uVar2);
  do {                                         |     return iVar1;
    eax_1 = eax_1 + param_1[rdx_1];            |   }
    rdx_1 = rdx_1 + 1;                         |   return 0;
  } while ((unsigned long)(unsigned int)ecx_1 != rdx_1); | }
  return eax_1;                                |
}                                              |
```

| Difference | Instruction or guess |
|---|---|
| `int *param_1` against `long param_1` | Inference on our side from `addl (%rdi,%rdx,4), %eax`. Ghidra kept the width. |
| `uint param_2` and `(int)param_2` against `int param_2` | Ghidra chose unsigned because `cmpq %rdx, %rcx` compares the zero-extended count. We kept `int` and cast at the compare. |
| `do { } while` in both | Instruction `jne 0x50` from the block to itself. The compiler rotated the loop. The `-O0` object keeps the `while`. |
| `ecx_1 = param_2` kept against `param_2 != uVar2` | Our substitution rule stays inside one block. The copy `movl %esi, %ecx` is in the setup block and the compare is in the loop, so the variable stays. Ghidra propagated it. |
| `(unsigned long)(unsigned int)ecx_1` | Instruction. `movl %esi, %ecx` zero-extends, and the 64-bit compare reads the full register. |

Le test de comportement compile le texte de Ghidra pour `grade`et `total`Il y a deux autres.`typedef`Les lignes et la définition de `DAT_001000b4`à partir des vingt octets à `.rodata+0x14`Les deux fonctions correspondent à la source sur la grille.

## Ce que la recherche dit sur la lecture du pseudo-code

LLM4Decompile (EMNLP 2024) a mesuré Ghidra seul à 20,12% de réexécutabilité sur HumanEval-Decompile. Environ une fonction sur cinq a recompilé et passé les affirmations qui accompagnent la référence. En raffinant la production de Ghidra avec leur modèle 6.7B, il est passé à 52,74%. DecompileBench (Findings of ACL 2025) a utilisé six décompilateurs industriels et six approches LLM sur 23 400 fonctions provenant de 130 programmes. Les résultats du LLM sont meilleurs et ont une précision de fonctionnalité inférieure de 52,2%.

Liu, Raff et Micinski (arXiv 2609.05370, septembre 2026) expliquent le fossé. Ils ont utilisé 300 fonctions de bibliothèque GitHub et 287 fonctions basées sur CVE. Le modèle de raffinement le plus fort a augmenté le taux de construction de Ghidra de 75 à 90%, tandis que le taux de correspondance comportementale est tombé de 74 à 62%. Les candidats qui ont passé tous les tests dans la référence ont toujours divergé sur les entrées fuzzed. Cela s'est produit dans 4,9% des cas dans huit systèmes, et jusqu'à 13% pour un système. Jusqu'à un dixième des vulnérabilités révélées ont disparu du code raffiné.

Les champs, les types, les calles et les gardes inventés ont remplacé les inconnus visibles que le décompilateur avait laissés dans le texte.

C' est la raison pour laquelle le lifting imprime .`(unsigned long)(unsigned int)ecx_1`Il imprime une note au lieu d'une lettre.`switch`pour le saut non résolu, et `lookup_14`Au lieu d'une étiquette, un modèle invente.`code/main.py`- Je suis Ghidra.`undefined1 [16]`C'est la plus belle des modèles.`int percent_change(int, int)`Le même modèle produira une signature pure et erronée sur la fonction suivante.

## Faites-le

Exécutez la démo:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
```

La démo décode les quatre fonctions de la`-O1`l' objet et compare chaque instruction avec l' engagement `llvm-objdump`Pour `percent_change`- Je suis là .`grade`, et `total`Il imprime l'IR avant le SSA, la liste du SSA avec des fonctions phi, et le pseudocode à côté du texte Ghidra.`grade`avec la reconnaissance de division désactivée, le `-O0`versions de `percent_change`- Je suis là .`total`, et `grade_label`, et un résumé:

```json
{
  "functions": ["grade", "grade_label", "percent_change", "total"],
  "instructions": 56,
  "blocks": {"percent_change": 3, "grade": 3, "total": 5, "grade_label": 4},
  "phis": {"percent_change": 0, "grade": 1, "total": 2, "grade_label": 1},
  "objdumpDifferences": 0
}
```

Lire `code/main.py`dans cet ordre:

1. `read_object()`Lire les en-têtes de section, le tableau des symboles et le `.rela`les sections d'un objet ELF64 relocalisable.
2. `decode_one()`et `render_att()`Décodez une instruction et imprimez-la comme ça .`llvm-objdump`- Il le fait.
3. `build_blocks()`- Je suis là .`immediate_dominators()`, et `dominance_frontiers()`construire les faits graphiques.
4. `lift_insn()`transforme une instruction en déclarations IR et résout les délocalisations.
5. `insert_phis()`et `rename()`construire SSA et insérer des lancers de largeur.
6. `make_simplifier()`- Je suis là .`substitute_once()`, et `remove_dead_code()`courir à l' intérieur `optimize()`- Je suis désolé .
7. `collect_params()`- Je suis là .`coalesce_webs()`, et `infer_types()`nommer et taper les réseaux.
8. `structure()`avec `post_dominators()`et `natural_loops()`émet le corps C.
9. `lift_function()`connecte les étapes et `lift_all()`Levez un objet entier avec des signatures de callee.

## Utilisez-le

Pour lever une fonction de votre propre objet, construisez-la avec `clang -target x86_64-unknown-linux-gnu -c`et appeler`lift_all(read_object(path))`Quand le décodeur monte`DecodeError`, lire le octet opcode du message et le trouver dans la carte SDM opcode. Puis ajouter une entrée aux tables en haut du fichier. Lorsque le souverain soulève sur un mnemonique, ajouter une règle à `lift_insn()`- lorsque le constructeur émet un`goto`, imprimez le graphique avec `IrPrinter`Chaque addition est petite et chaque addition est vérifiée contre`llvm-objdump`ou de la claque par les tests existants.

Lorsque Ghidra est installé, exécutez ré sur la même fonction et placez les deux textes côte à côte. Remplissez la table de différence avant de croire l'un ou l'autre texte. Une différence avec une instruction derrière elle est un choix de présentation. Une différence sans instruction derrière elle est une devinette, et la devinette peut être des deux côtés.

## Laboratoire interactif

### Laboratoire A: une fonction à travers cinq étapes

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode/code
python3 -q
```

Je vais courir .

```python
from main import *
obj = read_object(FIXTURES / "ledger-x86_64-O1.o")
lifted = lift_function(obj, "percent_change")
for insn in lifted.decoded: print(f"0x{insn.address:x}: {insn.raw.hex(' ')}  {insn.att()}")
print(lifted.ir_text)
print(lifted.ssa_text)
print(lifted.pseudocode)
```

Suivez`%esi`de `subl %edi, %esi`à `esi_1 = esi_0 - edi_0`à `(param_2 - param_1)`- Le byte .`fe`est le ModR/M de la `subl`, avec `mod = 11`- Je suis là .`reg = edi`, et `r/m = esi`- Je suis désolé .

### Laboratoire B: le langage de division

```python
print(lift_function(obj, "grade", recognize_division=False).pseudocode)
print(magic_divisor(0x66666667, 34), magic_divisor(0x66666667, 33), magic_divisor(0x66666667, 31))
```

La première ligne imprime le multiplicateur brut et les déplacements.`10 5 None`. Découvrez pourquoi le même multiplicateur se divise par 5 avec un décalage en moins, et pourquoi le décalage 31 ne respecte pas la limite d'erreur.

### Laboratoire C: un nouvel objet arrête le décodeur

Construisez le dispositif à `-O2`et le soulevez:

```bash
clang -target x86_64-unknown-linux-gnu -O2 -fno-asynchronous-unwind-tables -c fixtures/ledger.c -o "$TMPDIR/ledger-O2.o"
python3 -c 'from main import *; from pathlib import Path; import os; lift_all(read_object(Path(os.environ["TMPDIR"]) / "ledger-O2.o"))'
```

Le décodeur s' arrête à `0x56`avec `opcode 81 is outside the fixture subset`- Opcode .`81`est le formulaire de groupe 1 avec un instant de 32 bits.`andl $0x7ffffff8, %edx`, et clang l' a utilisé dans la boucle vectoriée de `total`- Ajoutez-le à côté .`0x83`dans `decode_one()`Le sous-ensemble augmente une entrée à la fois, et chaque entrée est vérifiée contre `llvm-objdump`- Je suis désolé .

### Laboratoire D: valeur par défaut de Ghidra en classe

```python
print(ghidra_pseudocode("grade"))
for score in (-25, 0, 59, 60, 100, 115):
    sign = -(score >> 31)
    product_sign = ((score * 0x66666667) & ((1 << 64) - 1)) >> 63
    print(score, sign ^ product_sign)
```

Chaque rangée imprime 0, Ghidra's `uVar1`l'expression est correcte et illisible, et l'instruction `xorl %eax, %eax`Il le dit en deux octets.

## Laboratoire de pratique

Ajouter `switch`rétablissement pour `grade`à`-O0`- Je suis désolé .

Les exigences:

- Lisez les cinq délocalisations en `.rela.rodata`de `ledger-x86_64-O0.o`, chacun `R_X86_64_PC32.text+offset`, et calculer les cinq cibles de l' affaire à partir du tableau à `.rodata+0x0`- Je suis désolé .
- Ajouter un `Switch`terminateur à l'IR avec l'expression index, la liste cible et le défaut de l'indice `ja`avant le chargement de la table.
- Donnez au bloc de commutateur un successeur par cible pour que les dominateurs, les post-dominateurs et les SSA continuent à travailler.
- Émettez`switch (local_10) { case 0:... }`avec un seul `case`par cible et l'épilogue partagé après la fusion.
- Gardez le billet pour tout saut indirect dont vous ne pouvez pas attacher la table.
- Compiler le résultat avec le clang et le comparer avec la source sur la grille de scores dans les tests.

Le succès signifie`grade`à`-O0`et `grade`à`-O1`produire la même sortie sur le réseau, avec un `switch`et non`goto`- Je suis désolé .

## Artéfacts

Cette leçon produit `outputs/skill-lifting-to-pseudocode.md`. Utilisez-le pour lever une fonction avec une dérivation pour chaque ligne et pour remplir la table de différence contre le texte d'un décompilateur.

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La suite comporte 25 tests.

- le décodeur est d'accord avec le compromis `llvm-objdump`texte sur chaque instruction des deux objets, et avec un live `llvm-objdump`lorsqu'il est installé
- REX, ModR/M, SIB et le décalage de champ de déplacement de `leaq (%rip), %rcx`sont décodés comme le décrit le MDD
- un opcode en dehors du sous-ensemble soulève `DecodeError`au lieu de deviner
- Le lecteur ELF énumère les quatre fonctions et les trois `.text`délocalisations
- `total`Il y a cinq blocs.`0x44`dominent`0x50`, et `0x50`est un en-tête de boucle avec lui-même comme le corps
- `%rdi`et `%rsi`devenir `param_1`et `param_2`, et `param_1`dans `total`est `int *`
- `test`et `cmp`avec `je`- Je suis là .`jle`- Je suis là .`ja`, et `jge`devenir `==`- Je suis là .`<=`, non signé `>`, et `<`
- SSA place deux fonctions Phi en direct à l' en-tête de boucle de `total`et aucun dans `percent_change`
- le pseudocode de `percent_change`- Je suis là .`grade`, et `total`est égal au texte attendu
- `grade`Il lit`param_1 / 10 - 6`avec reconnaissance sur et le brut multiplier avec reconnaissance hors
- `0x66666667`avec le changement 34 divisé par 10, avec le changement 33 divisé par 5, et l'identité est maintenue sur la grille
- les emplacements de pile à `-O0`devenir `local_4`- Je suis là .`local_8`- Je suis là .`local_c`, et le pointeur de cadre n' apparaît jamais
- la table à `.rodata+0x14`Il lit`1, 2, 3, 4, 4`par le transfert
- le saut indirect dans `grade`à`-O0`est déclaré avec ses quatre blocs non relevés
- Les appareils Ghidra contiennent le retour de 16 bytes, le `param_3`, le non-simplifié `xor`, et le `long param_1`
- Les six unités émises sont compilées avec `clang -std=c11 -Wall -Werror`pour `x86_64-unknown-linux-gnu`
- les fonctions soulevées, le texte Ghidra réparé et la source impriment les mêmes 232 lignes sur la grille d'entrée

## Mode de défaillance de la production

| Failure | What the weak reading reports | What the derivation must show |
|---|---|---|
| Decompiler widens the return value | "returns a 16-byte struct" | `idivl` writes `%rdx`, and no caller fixes the width |
| Decompiler adds a parameter | "takes three arguments" | The register is read on one path only because the return width was guessed |
| Idiom goes unrecognized | "multiplies by a magic constant" | The multiplier is the ceiling of `2^s / d` and the error bound holds |
| Idiom is recognized and a `xor` is lost | "default depends on the input" | `xorl %eax, %eax` writes zero, two spellings of one bit cancel |
| Pointer typed as integer | "takes a long" | A scaled 4-byte load from the register is pointer evidence |
| Table length unknown | "reads past the table" | The next referenced offset or the section end bounds it, and the range check agrees |
| Loop shape misread | "while loop" | The header has no exit, the latch jumps back, so it is `do while` |
| Indirect jump guessed | "switch with five cases" | The targets come from `.rela.rodata`, or the jump is reported as unresolved |
| Substitution across a store | "one expression" | A loaded value moved past a store or a call changes meaning |
| Refinement model edits the text | "compiles and looks right" | Build rate up, behaviour match down, invented guards replaced visible unknowns |
| Decoder guesses an unknown opcode | "disassembly looks fine" | `DecodeError` with the address and the byte, then one table entry checked against `llvm-objdump` |

## Règlement d'exploitation

Traitez le pseudocode comme une dérivation que vous n'avez pas vue. Reproduisez-le à partir des instructions avec un élévateur que vous pouvez lire. Placez les deux textes côte à côte, et écrivez l'instruction ou la devinette derrière chaque ligne qui diffère. Compilez les deux et courez les deux contre la source lorsque vous en avez. Une ligne sans instruction derrière elle reste une devinette peu importe la propreté de sa lecture.

## Pour en savoir plus

- [Intel SDM Volume 2, chapter 2, Instruction Format](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [System V AMD64 psABI, section 3.2, Function Calling Sequence](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Cytron et al., Efficiently Computing Static Single Assignment Form and the Control Dependence Graph, TOPLAS 1991](https://doi.org/10.1145/115372.115320)
- [Cooper, Harvey, Kennedy, A Simple, Fast Dominance Algorithm, 2001](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf)
- [Cifuentes, Reverse Compilation Techniques, PhD thesis, QUT 1994](https://eprints.qut.edu.au/36820/)
- [Yakdan et al., No More Gotos, NDSS 2015](https://www.ndss-symposium.org/ndss2015/ndss-2015-programme/no-more-gotos-decompilation-using-pattern-independent-control-flow-structuring-and-semantics/)
- [Granlund, Montgomery, Division by Invariant Integers using Multiplication, PLDI 1994](https://doi.org/10.1145/178243.178249)
- [Lee, Avgerinos, Brumley, TIE, NDSS 2011](https://www.ndss-symposium.org/ndss2011/tie-principled-reverse-engineering-of-types-in-binary-programs/)
- [Noonan, Loginov, Cok, Polymorphic Type Inference for Machine Code, PLDI 2016](https://arxiv.org/abs/1603.05495)
- [LLM4Decompile, EMNLP 2024](https://arxiv.org/abs/2403.05286)
- [DecompileBench, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1194/)
- [Liu, Raff, Micinski, When LLM Decompilers Recompile More and Preserve Less, 2026](https://arxiv.org/abs/2609.05370)
- [rea 6.5.0 Ghidra provider notes](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/ghidra-provider.md)
- réa 6.5.0 source à la balise `3b79af2`que cette leçon décrit: `src/domain/functionComparisonDimensions.ts`- Je suis là .`website/evidence/dx-ball-sound-pan.md`- Je suis désolé .
