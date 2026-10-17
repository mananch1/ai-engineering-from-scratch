# Idiomes de compilateur: ce que l'optimisation fait à votre code et comment le lire

> Un compilateur d'optimisation remplace les constructions de source par des formes d'instruction moins chères. Chaque forme a une dérivation, et cette leçon construit le reconnaisseur qui exécute la dérivation vers l'arrière et la vérifie contre la source.

**Type:** Build
**Languages:** Python, C, x86-64 assembly, AArch64 assembly
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode), Phase 13 · 38 (calling conventions)
**Time:** ~150 minutes

## Objectifs d'apprentissage

- Dérivez le nombre magique et déplacez pour la division par une constante avec la limite Granlund-Montgomery, et récupérez le diviseur des constantes émises par le clang.
- Nombre des cinq façons dont LLVM réduit un `switch`et lire le nombre d'entrées, l'emplacement de la table et les entrées de l'objet.
- Lisez une boucle vectoriée sous forme de voies, d'accumulateurs, d'éléments par itération, de masque de triple-compte, de réduction et de boucle restante.
- Reconnaître les appels de queue, les appels en ligne, les sélections sans branches et les boucles qui sont devenues `memset`ou `memcpy`- Je suis désolé .
- Rapportez chaque langage avec ses instructions de preuve et sa forme source décodée, aux x86-64 et à AArch64, à `-O0`à travers `-O3`- Je suis désolé .
- Comparez le résultat avec ce que Ghidra, Angr, radar2, rizin et lldb montrent pour les mêmes octets.

## Le problème

Vous avez compilé`idioms.c`à`-O2`et a demandé à un agent ce que `div7u`L'agent a lu`imulq $0x24924925, %rax, %rax`et a répondu que la fonction se multiplie par 613566757, se déplace et ajoute.`x / 7`- Je suis désolé .

Le même objet contient un `switch`Avec sept cas qui n'ont pas de saut du tout, un `switch`avec cinq affaires qui est une table de saut, un `switch`avec deux cas qui est un `sete`et une `cmovne`, et une boucle à travers `n`des entiers qui exécutent huit entiers par itération en deux `xmm`Il est donc possible de faire une analyse de la situation de l'entreprise.`-O0`la même source produite `divl`, des tables de saut à sept entrées, et une boucle escalare.`-O1`Le commutateur à sept cas est devenu un tableau de valeurs et la boucle vectorielle n'existait pas encore.

Ces formes sont des idiomes. Un compilateur émet chacune d'elles à partir d'une règle que vous pouvez lire dans sa source, et chacune peut être décodée à la construction d'où elle vient. Le décoding est arithmétique, et l'arithmétique peut être vérifiée.`0x66666667`Cette leçon construit le reconnaisseur pour tous, le traite sur 18 versions de notre code, et vérifie les 373 rapports contre la source C.

```figure
re-39-idioms-by-level
```

## Un idiome est une forme avec une dérivation

Un idiome est une séquence d'instructions que le compilateur émet pour une construction de source. Trois choses le rendent reconnaissable. Il a une forme fixe, les opérands portent des constantes que la dérivation prédit, et la forme diffère de ce qu'un programmeur écrirait à la main.

Le reconnaisseur en `code/main.py`Les travaux sur le texte de `llvm-objdump -d -r`Il analyse chaque fonction en instructions avec adresse, octets, mnemonique, opérands et la ligne de relocalisation qui la suit. Il lit le fichier objet pour deux choses seulement: les octets de`.rodata`, où les tableaux de valeurs sont présents, et le `.rela.rodata`les entrées, où les cibles de la table de saut vivent avant de se connecter.

Chaque détection est une`Idiom`enregistrement avec un `kind`, un décodeur `source`forme, une liste `evidence`adresses, une `detail`Le dictionnaire et un`verified`Le drapeau provient de la source C.`parse_source()`Il lit`idioms.c`, trouve chaque fonction, et extrait les diviseurs après `/`et `%`, les multiplicateurs après `*`, le `switch`Les cas et leurs valeurs, les`? :`Les opérateurs, les `for`Un langage de division est vérifié lorsque son diviseur est dans la liste de division de la fonction. Une table de recherche est vérifiée lorsque ses entrées correspondent aux valeurs du cas, jusqu'à un facteur constant. Un appel de queue est vérifié lorsque le call est appelé dans la source. Le rapport de démo se termine par un `unverified`la liste, et la suite de tests affirme que la liste est vide pour les 18 constructions.

Le reconnaisseur exécute un petit évaluateur symbolique sur les blocs de base de chaque fonction, dans l'ordre de liste, avec les noms des paramètres de la source dans les registres d'arguments des conventions du système V et AAPCS64 de la leçon 38.`-O0`Le code se lit à travers ses déversements. À chaque entrée de bloc les états des prédécesseurs sont fusionnés, et où ils diffèrent un nom qu'une idiome antérieure attribuée gagne sur une constante.`cmovll %edx, %eax`dans `digest`Il est écrit:`sum % 10 < 5 ? sum % 10 : 5`: l' idiome de la boucle nommé l' accumulateur `sum`, l' idiome de division nomme son quotient `sum / 10`, et le modulo idiome nommé son résultat `sum % 10`La source écrit:`rem > 5 ? 5 : rem`Les deux expressions donnent le plus petit de `rem`et 5, et ils sont d'accord à `rem == 5`- Je suis désolé .

## Divisions par une constante

Le rapport Granlund et Montgomery, "Division par intégrales invariables à l'aide de la multiplication" (PLDI 1994), remplace `n / d`Avec un multiplicateur et un changement.`s`et un multiplicateur `m = ceil(2^s / d)`- Alors ...`floor(n * m / 2^s)`égale `floor(n / d)`pour chaque`n`dans la plage lorsque l'erreur `e = m * d - 2^s`est assez petit. Pour un non signé 32 bits`n`La limite est `e <= 2^(s - 32)`Pour les signataires .`n`avec `|n| < 2^31`La limite est `e <= 2^(s - 31)`, et une correction de signage ajoute 1 quand `n`est négatif, parce que le sol tourne dans le mauvais sens pour les quotients négatifs.

`magic_numbers(d, bits, signed)`dans `main.py`Les recherches `s`La recherche reproduit chaque clang constant émis dans les appareils.

### 10 non signé à la main

`2^35 / 10 = 3435973836.8`Le plafond est`3435973837 = 0xCCCCCCCD`L' erreur est`0xCCCCCCCD * 10 - 2^35 = 2`, et le bord à `s = 35`est `2^3 = 8`À la .`s = 34`l'erreur serait 6 contre une limite de 4, donc 35 est le plus petit changement.`n = 123`Le numéro de la liste:`123 * 3435973837 = 422624781951`, et `422624781951 >> 35 = 12`- Je suis désolé .

Clang émet exactement ça pour `div10u`à`-O2`sur x86-64:

```text
       0: 89 f9                        	movl	%edi, %ecx
       2: b8 cd cc cc cc               	movl	$0xcccccccd, %eax
       7: 48 0f af c1                  	imulq	%rcx, %rax
       b: 48 c1 e8 23                  	shrq	$0x23, %rax
```

`0x23`Il est 35.`movl %edi, %ecx`Le reconnaisseur lit le multiplicateur de la variable `movl $imm`qui nourrit le double-opérand `imulq`, lit le quart, et appelle `divisor_from_magic(0xCCCCCCCD, 35)`Cette fonction essaie .`2^35 // m = 10`et 11, et accepte 10 parce que `magic_numbers(10)`donne le même multiplicateur et recule.

Sur AArch64 la même fonction est `mov w8, #0xcccd; movk w8, #0xcccc, lsl #16; umull x8, w0, w8; lsr x0, x8, #35`La constante arrive en deux halves de 16 bits, le multiplicateur est `umull`Le reconnaisseur rassemble les demi-parties et rapporte le même divisor.

### 7 non signé à la main, avec la correction additive

`2^35 / 7 = 4908534052.57`Le plafond est`4908534053 = 0x124924925`, qui a besoin de 33 bits. L'erreur est 3 contre une limite de 8, donc le changement est 35, et aucun changement plus petit ne fonctionne. Un multiplicateur de 32 bits ne peut pas contenir le multiplicateur, donc le compilateur se multiplie par les 32 bits bas`0x24924925`et ajoute le dividende dans une séquence fixe:

```text
      32: 48 69 c0 25 49 92 24         	imulq	$0x24924925, %rax, %rax
      39: 48 c1 e8 20                  	shrq	$0x20, %rax
      3d: 29 c7                        	subl	%eax, %edi
      3f: d1 ef                        	shrl	%edi
      41: 01 f8                        	addl	%edi, %eax
      43: c1 e8 02                     	shrl	$0x2, %eax
```

Lisez comme `t = (n * 0x24924925) >> 32`Alors ...`q = (((n - t) >> 1) + t) >> 2`Le changement total est`32 + 1 + 2 = 35`Avec `n = 100`Le numéro de la liste:`t = 14`- Je suis là .`(100 - 14) >> 1 = 43`- Je suis là .`43 + 14 = 57`- Je suis là .`57 >> 2 = 14`Le reconnaisseur détecte le `sub`- Je suis là .`shr`- Je suis là .`add`- Je suis là .`shr`la queue après un changement de 32 bits, ajoute `2^32`retour au multiplicateur, et récupère 7. `evaluate_division()`Il est utilisé pour la séquence exacte et le test le vérifie contre`n // 7`Le formulaire AArch64 est `sub w9, w0, w8; add w8, w8, w9, lsr #1; lsr w0, w8, #2`, une instruction plus courte parce que le déplacé add est une instruction.

### Signature 3 à la main

`2^32 / 3 = 1431655765.33`, plafond `1431655766 = 0x55555556`, erreur 2, et la liaison signée à `s = 32`est `2^1 = 2`Le bouton est fermement fermé, donc le quart est 32 et clang utilise`shrq $0x20`Avec `n = -7`: le produit est `-10021590362`, le changement arithmétique par 32 donne `-3`, le bit de sign est réglé, donc la fixation ajoute 1 et le résultat est `-2`, qui est `-7 / 3`dans C. Les appareils signés `div10s`et `div3s`Ils portent tous les deux la réparation comme `shrq $0x3f`sur une copie du produit suivie de `addl`- Je suis désolé .

### Ce que les constantes ne sont pas

Le multiplicateur seul ne nomme aucun divisor. `0x66666667`Le même multiplicateur avec le changement 33 est 5, comme l'a montré la leçon 34.`clangMatchesAlgorithm`Les pouvoirs de deux ne font jamais usage du multiplicateur.`magic_numbers()`les rejette, parce que le compilateur émet un changement avec un biais de signe à la place, et `mod8s`Dans la section suivante, on voit cette forme.

### L'idiome appartient au sélecteur d'instructions

À `-O0`- Je suis là .`div10u`est `movl $0xa, %ecx; xorl %edx, %edx; divl %ecx`- Je suis là .`div10s`est `cltd; idivl %ecx`Dans le même objet,`digest`divise sa somme de boucle par 10 avec `imulq $0x66666667`Un drapeau explique la différence.`idioms-x86_64-nofastisel-O0.o`a été construit avec `-O0 -mllvm -fast-isel=false`, et dans chaque division est le multiplicateur, y compris `div10u`avec `movl $0xcccccccd, %ecx; imulq %rcx, %rax; shrq $0x23`Le partage matériel est le choix du sélecteur d'instructions rapide à `-O0`. Le multiplicateur est la diminution de SelectionDAG, ce qui est ce que LLVM `UnsignedDivisionByConstantInfo`et `SignedDivisionByConstantInfo`Les produits de la`digest`Le bloc qui tient la division tient aussi un`switch`Le reconnaisseur rapporte les deux formulaires,`hardware idivl`et `magic multiply`, et la figure montre le mélange `-O0`fonction.

## Modulo suit le quotient

`x % d`est `x - (x / d) * d`Le compilateur calcule le quotient avec l'idiome ci-dessus, le multiplie et le soustrait.`addl %ecx, %ecx; leal (%rcx,%rcx,4), %ecx`, qui est `2 * q * 5`Sur AArch64, il y a une instruction.`msub w0, w8, w9, w0`, qui compute `w0 - w8 * w9`avec `w9 = 10`. Le reconnaisseur suit le registre des quotients à travers des copies, suit le facteur accumulé à travers `add`- Je suis là .`lea`- Je suis là .`shl`, et `imul`, et accepte une`sub`lorsque le facteur est égal au divisor.

Lorsque le même quotient donne aussi une autre expression, les deux expressions restent dans le rapport.`-O1`le compilateur compute `sum / 10`Une fois dans la vie .`digest`et l' utilise pour le `switch`Il est même combiné le quotient 2 fois,`leal (%rax,%rsi), %edi`pour le modulo et `addl %esi, %eax`Le reconnaisseur rapporte une division et un module et marque le quotient comme partagé.`switch`l'indice est illisible.

Le modulo signé par une puissance de deux a sa propre forme, car le quotient n'est jamais calculé:

```text
      92: 8d 48 07                     	leal	0x7(%rax), %ecx
      95: 85 ff                        	testl	%edi, %edi
      97: 0f 49 cf                     	cmovnsl	%edi, %ecx
      9a: 83 e1 f8                     	andl	$-0x8, %ecx
      9d: 29 c8                        	subl	%ecx, %eax
```

Pour un négatif `x`Le code s'écarte de 7 avant de masquer, de sorte que la tranchée tourne vers zéro comme C le demande.`x`ronde à un multiple de 8, et la soustraction laisse `x % 8`AArch64 le fait avec `negs w8, w0; and w9, w0, #0x7; and w8, w8, #0x7; csneg w0, w9, w8, mi`: masquer les deux `x`et `-x`Ghidra appelle sa règle pour cela.`RuleSignMod2nOpt`, avec le bref "Convertir les formulaires INT_SREM: `(V + (sign >> (64-n)) & (2^n-1)) - (sign >> (64-n)) => V s% 2^n`" " Je suis là.

À `-O0`AArch64 écrit modulo comme `sdiv w9, w8, w10; mul w9, w9, w10; subs w0, w8, w9`Le reconnaisseur accepte également ce formulaire et rapporte`hardware sdiv + mul + subs`- Je suis désolé .

## Cinq façons de réduire le bouton

La LVM décide de la forme d' une`switch`Dans deux endroits.`SimplifyCFG::simplifySwitchLookup`Il est à `-O1`et de plus en plus et fait tourner un interrupteur dont tous les cas produisent des constantes dans une table de valeurs, quand il y a au moins trois cas et que le but peut construire des tables de recherche.`SingleValueKind`- Je suis là .`LinearMapKind`- Je suis là .`BitMapKind`, et `ArrayKind`- Le .`-O0`Le pipeline ne passe jamais par ce passage.`SwitchLowering::findJumpTables`dans la sélection des instructions, qui crée une table de saut lorsque le cluster a au moins `min-jump-table-entries`les entrées, 4 par défaut, et suffisamment denses: `NumCases * 100 >= Range * MinDensity`avec `jump-table-density`10 pour les fonctions normales et 40 sous `-Os`Ce qui reste devient une recherche binaire sur les constantes de cas, avec `splitWorkItem`Les appareils ont une fonction par résultat.

| Function | Source | `-O0` x86-64 | `-O2` x86-64 | `-O2` AArch64 |
|---|---|---|---|---|
| `linear_switch` | cases 0 to 6 return 10 to 70 | jump table, 7 entries | linear map `10*k + 10`, default -1 | `madd` and `csinv` |
| `table_switch` | cases 0 to 6 return 10 to 60 then 71 | jump table, 7 entries | lookup table at `.rodata+0x28` | lookup table at `.rodata+0x0` |
| `call_switch` | cases 0 to 4 call one of five pointers | jump table, 5 entries | jump table, 5 entries | comparison tree, pivot 1 |
| `sparse_switch` | cases 1, 100, 10000, 1000000, 100000000 | compare chain | comparison tree, pivot 9999 | comparison tree, pivot 9999 |
| `two_cases` | cases 3 and 4 return 7 and 9 | compare chain | select chain `k != 3 ? (k == 4)*9 : 7` | two `csel` |

### La table de recherche, lue à partir de l'objet

`table_switch`à`-O2`- Le numéro de la liste:

```text
      d0: b8 ff ff ff ff               	movl	$0xffffffff, %eax
      d5: 83 ff 06                     	cmpl	$0x6, %edi
      d8: 77 0c                        	ja	0xe6 <table_switch+0x16>
      da: 89 f8                        	movl	%edi, %eax
      dc: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		00000000000000df:  R_X86_64_PC32	.rodata+0x24
      e3: 8b 04 81                     	movl	(%rcx,%rax,4), %eax
      e6: c3                           	retq
```

Les limites sont vérifiées .`cmpl $6; ja`Le recensement est le plus important.`0x24`plus les 4 octets du champ relatif à la RIP met la table à `.rodata+0x28`Les octets sont`0a 00 00 00 14 00 00 00 ...`, donc les entrées sont `[10, 20, 30, 40, 50, 60, 71]`, et ils sont égaux aux valeurs de cas source.`.Lswitch.table.table_switch`dans le `.s`fichier, avec un commentaire `@switch.table.table_switch`Il n'y a pas de saut, donc le graphique de contrôle-flux a trois blocs.

Le formulaire AArch64 met l' adresse du tableau avec `adrp x8`et `add x8, x8, #0x0`, chacun portant sa propre relocalisation à `.rodata`, et chargés de `ldr w0, [x8, w0, uxtw #2]`Le reconnaisseur lit l' addendum de la`add`déménagement.

### La carte linéaire, sans table du tout

`linear_switch`Retour `10 * (k + 1)`pour `k`dans 0 à 6. `simplifySwitchLookup`détecte la relation affine et émet l'arithmétique:

```text
      b0: 83 ff 07                     	cmpl	$0x7, %edi
      b3: 8d 04 bf                     	leal	(%rdi,%rdi,4), %eax
      b6: 8d 4c 00 0a                  	leal	0xa(%rax,%rax), %ecx
      ba: b8 ff ff ff ff               	movl	$0xffffffff, %eax
      bf: 0f 42 c1                     	cmovbl	%ecx, %eax
```

L' évaluateur lit `leal (%rdi,%rdi,4)`comme `5*k`- Je suis là .`leal 0xa(%rax,%rax)`comme `10*k + 10`, et le `cmovb`après `cmpl $7`comme `k < 7 ? 10*k + 10 : -1`Le reconnaisseur vérifie ensuite la formule contre chaque cas source et marque l'idiome vérifié.`madd w8, w0, w8, w9`et `csinv w0, w8, wzr, lo`Où ?`csinv`avec `wzr`rendements `-1`sur le chemin par défaut.

### La table de saut, lue à partir de délocalisations

`call_switch`à`-O2`est la leçon de forme 33 récupéré en `grade`à`-O0`Le numéro de la liste:`cmpl $4; ja`, une `leaq`à `.rodata-0x4`, une `movslq`d'une entrée de 4 octets, `addq`, et `jmpq *%rax`Les cinq entrées sont des délocalisations en`.rela.rodata`, et `target = addend - 4k`donne`0x10c, 0x107, 0x10e, 0x110, 0x109`, cinq blocs distincts qui sautent chacun à travers un pointeur de fonction.`dispatch`Le fixture étalonne ceci à 16 cas: 16 entrées de relocalisation sur x86-64, et sur AArch64 une table comprimée de 16 octets uniques lues avec `ldrb w11, [x9, x8]`, étalé par 4 avec `add x10, x10, x11, lsl #2`, et basé sur un`adr`Le reconnaisseur décode`[0x24, 0x74, 0x54, ...]`du tableau en octets et des rapports `compressed: true`- Je suis désolé .

### Le même interrupteur à quatre niveaux

`digest`Il tient un interrupteur allumé .`sum / 10`Avec les cas 6 à 10 et les valeurs 1, 2, 4, 8, 8, puis multipliez le résultat par 16. sur x86-64 il est une table de saut à `-O0`, une table de recherche à `-O1`avec des entrées `[16, 32, 64, 128, 128]`, et une table de saut à nouveau à `-O2`et `-O3`dont quatre blocs de coffre sont stockés `0x10`- Je suis là .`0x20`- Je suis là .`0x40`- Je suis là .`0x80`- De la .`-O1`Le taux de multiplication par 16 a augmenté.`-O0`Les cas stockent 1, 2, 4 et 8, et la multiplication reste après le commutateur.`-O1`Tableau de recherche le reconnaisseur compare les entrées avec les valeurs de source, trouve un ratio constant et rapporte `foldedFactor: 16`Sur l' arche 64 , le`-O2`forme est un arbre de comparaison avec un pivot à 8, tests d'égalité pour 6, 7 et 8, et un test de plage `sub w9, w9, #9; cmp w9, #2; b.hs`Le reconnaisseur rapporte que les cas 9 et 10 sont couverts par deux instructions.`rangeTests: [[9, 10]]`- Je suis désolé .

Deux drapeaux déplacent ces frontières.`-fno-jump-tables`tourne `table_switch`- Je suis là .`call_switch`, et le `digest`Pour la fonction de sept cas, il faut passer à des arbres de comparaison sur x86-64, avec des pivots 2 et 4.`-mllvm -min-jump-table-entries=4`feuilles`call_switch`En effet, le nombre d'appels indirects ne désigne pas un nombre de cas de conversion.`idioms-arm64-jt4-O2.o`Des enregistrements qui fonctionnent.

Le CCG met en œuvre les deux décisions `tree-switch-conversion.cc`, avec les cours `jump_table_cluster`- Je suis là .`bit_test_cluster`- Je suis là .`switch_decision_tree`, et `switch_conversion`- C' est ...`-ftree-switch-conversion`est en train de`-O2`- Je suis là .`-fjump-tables`et `-fbit-tests`à`-O1`, et le seuil de saut de table vient de `case-values-threshold`Il n'y a pas de GCC installé sur cette machine, donc ces noms proviennent de la source et du manuel du GCC, et le compiler Explorer est là où vous les vérifiez.

## Boucles: vectoriées, déroulées ou remplacées

Le vectorificateur de boucle de LLVM est allumé à `-O2`- Je suis là .`-O3`, et `-Os`, et de l' éteindre à `-O0`et `-O1`, une règle datant de LLVM 3.4 en janvier 2014. `sum_n`à`-O2`montre chaque partie de la forme:

```text
     1f6: 83 fe 07                     	cmpl	$0x7, %esi
     1f9: 77 09                        	ja	0x204 <sum_n+0x14>
     206: 81 e2 f8 ff ff 7f            	andl	$0x7ffffff8, %edx
     230: f3 0f 6f 14 37               	movdqu	(%rdi,%rsi), %xmm2
     235: 66 0f fe c2                  	paddd	%xmm2, %xmm0
     239: f3 0f 6f 54 37 10            	movdqu	0x10(%rdi,%rsi), %xmm2
     23f: 66 0f fe ca                  	paddd	%xmm2, %xmm1
     243: 48 83 c6 20                  	addq	$0x20, %rsi
     24a: 75 e4                        	jne	0x230 <sum_n+0x40>
     24c: 66 0f fe c8                  	paddd	%xmm0, %xmm1
     250: 66 0f 70 c1 ee               	pshufd	$0xee, %xmm1, %xmm0
     262: 66 0f 7e c8                  	movd	%xmm1, %eax
     270: 03 04 97                     	addl	(%rdi,%rdx,4), %eax
     279: 75 f5                        	jne	0x270 <sum_n+0x80>
```

Le garde .`cmpl $7; ja`Envoie le nombre de déplacements inférieur à 8 à la boucle scalaire.`andl $0x7ffffff8`Les rounds du voyage comptent jusqu'à un multiple de 8. Le corps charge 32 octets par itération en deux.`xmm`Les accumulateurs, quatre voies de 32 bits chacune, donc le facteur de vectorisation est 4 et le facteur d'interleaving est 2.`paddd`, deux `pshufd`, deux autres .`paddd`, et `movd`réduire les huit voies à un entier.`0x270`Le reconnaisseur trouve la boucle depuis son bord arrière, voit les registres SIMD, lit le nombre de voies de `paddd`, compte des accumulateurs distincts, divise l'étape `0x20`par 4, et des rapports `4 lanes x 2 accumulators = 8 elements per iteration`, avec le masque, la garde, la réduction et le reste de la boucle comme champs.`ldp q2, q3`, deux `add v.4s`- Je suis là .`addv s0, v0.4s`, et `fmov w8, s0`- Je suis désolé .

Un nombre fixe de déplacements supprime la boucle. `sum4`est un`movdqu`, la réduction et `movd`, sans bord arrière. `poly8`ne peut pas vectorier, parce que chaque itération dépend de l'accumulatrice précédente, donc le compilateur le déroulera huit fois: huit charges à des compensations de 0 à `0x1c`de `%rdi`, suivie de 5 et d' un soustrait qui compte `acc * 31`. Le reconnaisseur compte les charges à des décompenses consécutives à partir d' une base et rapporte `loop unrolled by 8, no back edge`Sur l' AArch64 , les mêmes charges arrivent que quatre .`ldp`- Les deux.

Éteignez le vectorificateur avec `-fno-vectorize -fno-slp-vectorize`et `sum_n`devient une boucle écalier déroulée par 8, avec huit `addl k(%rdi,%rdx,4), %eax`et `addq $8`, plus une boucle restante. `sum4`Le reconnaisseur rapporte`unrolled loop`avec `copies: 8`et `indexStep: 8`pour la première fois et `copies: 4`Pour la seconde fois.

Leçon 33 a trouvé deux bords arrière dans `total`à`-O2`Cette leçon les nomme: un corps vectoriel et un reste escalier de la même boucle source.`is_loop()`Il y a une question de sécurité.`digest`sur AArch64 contient des branches arrière vers un bloc par défaut partagé qui ne sont pas des boucles, et la première version du reconnaisseur les a comptées comme des boucles vectorielles.

### Les boucles qui deviennent des appels de bibliothèque

`LoopIdiomRecognize`transforme une boucle qui stocke zéro dans chaque octet en un appel à `memset`, à partir de `-O1`- Je suis en train de monter.`fill`à`-O2`est `testl %esi, %esi; jle; movl %esi, %edx; xorl %esi, %esi; jmp memset@PLT`L' appel est un saut de queue avec le déménagement`R_X86_64_PLT32 memset-0x4`, et la fonction est `retq`Le reconnaisseur rapporte que le système est en voie de disparition.`store loop recognized as memset(...)`et le vérifie contre la boucle source, parce que la source ne nomme jamais `memset`- Je suis désolé .

`copy16`Les appels`memcpy`avec une constante 16, et le backend x86 l'agrandit en ligne comme `movups (%rsi), %xmm0; movups %xmm0, (%rdi)`, dans sa limite de 8 magasins par`memcpy`À la .`-O0`le même appel est deux 8 octets `movq`avec `-fno-builtin`l'appel reste un appel, `jmp memcpy@PLT`avec `movl $0x10, %edx`, et `fill`conserve sa boucle, que le vectorificateur transforme ensuite en une boucle de deux 16 bytes `movups`Les rapports du reconnaisseur `vectorized store loop: 2 x 16-byte stores = 32 bytes per iteration`Pour cette construction.

## Des appels qui disparaissent ou deviennent des sauts

`tail_caller`Retour `tail_target(x + 1)`À la .`-O0`C' est vrai .`addl $0x1, %edi; callq tail_target@PLT; addq $0x10, %rsp; popq %rbp; retq`À la .`-O2`C' est vrai .`incl %edi; jmp tail_target@PLT`Le cadre de l'appelant est parti et l'appelant revient à l'appelant.`b tail_target`avec `R_AARCH64_JUMP26`Le reconnaisseur traite un`jmp`ou `b`La descente récursive de la leçon 33 doit la traiter comme une sortie de fonction, car la cible est en dehors de la fonction.

`twice_sum`Les appels`leaf_add`À la .`-O0`L'appel est là.`callq leaf_add@PLT`- De la .`-O1`le corps de `leaf_add`- Je suis là .`leal (%rdi,%rsi), %eax`, apparaît à l' intérieur `twice_sum`Il n'y a plus d'appel.`-O0`Clang marque chaque fonction `optnone`et `noinline`Le reconnaisseur sait par la source que`twice_sum`Les appels`leaf_add`, ne trouve pas d'appel ou de saut de queue à elle, et cherche ensuite la forme du corps de l'appelant dans l'appelant: le mnemonique et la source opérands avec la destination enlevée, donc `add w8, w1, w0`dans les correspondances d' appel d' AArch64 `add w0, w1, w0`Une correspondance est la preuve.

`dispatch`combine les deux: une table de saut à 16 entrées dont chaque cas est`movl %esi, %edi; jmp fN@PLT`Le reconnaisseur rapporte un interrupteur et 16 appels de retard, tous vérifiés par rapport aux 16 appels de la source.

## Les branches qui sont devenues des données se déplacent

`clamp_max`est `movl %esi, %eax; cmpl %esi, %edi; cmovgl %edi, %eax`Les noms des évaluateurs`%edi`comme `a`et `%esi`comme `b`, enregistre la comparaison et lit la`cmovg`comme `a > b ? a : b`, qui est le texte source. `abs_int`est `movl %edi, %eax; negl %eax; cmovsl %edi, %eax`et se lit comme `-x < 0 ? x : -x`; AArch64 l' écrit comme `cmp w0, #0; cneg w0, w0, mi`et se lit comme `x < 0 ? -x : x`- Je suis là .`is_pos`est `xorl %eax, %eax; testl %edi, %edi; setg %al`, qui se lit comme `(x > 0)`- Ce sont ...`-fif-conversion`Dans les termes du CCG et `cmov`La sélection dans les LLVM.`X86CmovConversion`Passe peut tourner un `cmov`Il est possible de revenir à des branches quand il juge que rentable, c'est pourquoi la même source peut montrer l'une ou l'autre forme dans une boucle.

À `-O0`Aucune d'entre elles n'existe. `clamp_max`est une comparaison, une `jle`Le reconnaisseur ne rapporte rien à la poste.`-O0`et `branchless select`de `-O1`, qui est l'un des faits que le chiffre rend visible.

`mul9`est un`leal (%rdi,%rdi,8), %eax`, et les rapports du reconnaisseur `strength reduction`avec `x * 9`, parce que la source se multiplie par 9. Il ne le fait que lorsque la source a le facteur; a `lea`qui équivaut à un index de table n'est pas une multiplication.

## Deux caractéristiques durcis qui ressemblent à des idiomes

Le protecteur de pile est une fonctionnalité de compilateur, et il laisse une forme qu'un agent peut autrement lire comme logique de programme. `canary.c`compilé avec `-fstack-protector-strong`pour x86-64 Linux donne:

```text
       7: 64 48 8b 04 25 28 00 00 00   	movq	%fs:0x28, %rax
      10: 48 89 44 24 40               	movq	%rax, 0x40(%rsp)
      26: 64 48 8b 0c 25 28 00 00 00   	movq	%fs:0x28, %rcx
      2f: 48 3b 4c 24 40               	cmpq	0x40(%rsp), %rcx
      34: 75 05                        	jne	0x3b <with_canary+0x3b>
      3b: e8 00 00 00 00               	callq	0x40 <plain_leaf>
		000000000000003c:  R_X86_64_PLT32	__stack_chk_fail-0x4
```

`%fs:0x28`est `stack_guard`dans le glibc's `tcbhead_t`, à byte offset 40, et à LLVM `getIRStackGuard`Le prologue copie la garde au-dessus du tampon, l'épilogue compare et le chemin d'échec appelle`__stack_chk_fail`Sur AArch64 Linux , la garde vient du monde entier .`__stack_chk_guard`par le GOT avec `adrp`et `ldr`, et l' échec est `bl __stack_chk_fail`Le reconnaisseur rapporte la source de garde, la fente, la comparaison et l'appel d'échec.`plain_leaf`, qui n'a pas de filet, n'a pas de canary sous`strong`, qui protège les fonctions avec des matrices,`alloca`Les notes de recherche enregistrent également le formulaire MSVC sous`/GS`, un biscuit mélangé avec `%rsp`et vérifiée par `__security_check_cookie`; il n'existe aucun connecteur MSVC sur cette machine, de sorte que cette forme est décrite uniquement à partir de la sortie de l'assemblage du compilateur.

`-fcf-protection=full`- Il met`endbr64`, octets `f3 0f 1e fa`Il s'agit d'une piste d'atterrissage pour les branches indirectes sur les processeurs avec contrôle de débit, et il s'exécute comme un`nop`Les modèles de démarrage de fonction x86-64 de Ghidra incluent le`endbr64`variantes préfixes de `push rbp; mov rbp, rsp`Le reconnaisseur le rapporte comme:`cet landing pad`, donc une fonction qui commence par quatre octets inhabituels ne doit pas être confondue avec des données.

## Ce que les décompilateurs déploient

Chaque décompilateur de la leçon 37 utilise certaines de ces dérivées comme règles de réécriture.`ruleaction.cc`- Je suis là .`RuleDivOpt`est "Convertir INT_MULT et déplacer les formulaires en INT_DIV ou INT_SDIV", avec les formulaires `sub( (zext(V)*c), d) >> e => V / (2^n/(c-1)) where n = d*8 + e`et la variante signée avec `sext`et `s>>`- Je suis là .`RuleModOpt`est "Simplifier les expressions qui optimisent INT_REM et INT_SREM". `RuleDivTermAdd`gère le complément, `RuleSignDiv2`tourne `(V + -1*(V s>> 31)) s>> 1`dans `V s/ 2`, et `RulePositiveDiv`La leçon 37 a capturé Ghidra 12.1.4 sur la leçon 33's `grade`à`-O1`, et la sortie est `uVar2 = param_1 / 10 - 6;`La division est détachée.

L'entrée de ces règles est visible sans Java.`code/fixtures/idioms-x86_64-O2.lifts.json`détient le code P brut pour `div10s`: 112 entrées pour sept instructions, dont 7 sont les instructions `IMARK`les marqueurs qui ouvrent chaque instruction, parmi eux `RAX = RAX * unique[95a00:8]`après `unique[95a00:8] = sext(0x66666667)`- Je suis là .`RCX = RCX >> unique[d1a00:4]`, et `RAX = RAX s>> unique[c5c00:4]`- Le .`shrq`Il est chargé de réaliser 41 opérations et de`sarq`Il y en a 40, et la plupart calculent les drapeaux.`RuleDivOpt`correspond au multiplicateur et aux deux changements dans ce flux, et l'élimination du code mort supprime les drapeaux que rien ne lit.

Angr 10.0.3 a été lancé sur notre .`-O2`objet, et `idioms-x86_64-O2.angr.json`Il détient sa sortie de décompiler pour 20 fonctions. Il replie le formulaire non signé: `div10u`est `return a0 / 10;`Il ne plie pas le formulaire signé: `div10s`est `v1 = a0 * 1717986919; return ((long long)(v1) >> 0x22) + (v1 >> 63);`et `mod10s`est `a0 - (...) * 2 * 5`Pour ...`div7u`Il imprime .`(a0 / 7 + ((a0 * 6 / 7 & 0xffffffff) >> 1 & 0xffffffff) & 0xffffffff) >> 2`, où les termes de fixation survivent comme l'arithmétique.`mul9`à `a0 * 9`- Je suis là .`linear_switch`à `(a0 < 7 ? a0 * 10 + 10 : 0xffffffff)`- Je suis là .`clamp_max`à `((int)a1 < (int)a0 ? a0 : a1)`- Je suis là .`is_pos`à `a0 > 0`, et `fill`à `return memset(a0, 0, a1);`- Il imprime .`table_switch`comme une charge .`*(int *)(4 * a0 + 4195400)`sans les valeurs, `sum_n`avec `_mm_add_epi32`et `_mm_shuffle_epi32`Les données de base`tail_caller`comme `tail_target(); return;`Le débat a été perdu.`digest`La table de saut devient `goto *((void *)(...))`avec l'arithmétique de la table écrite.

Les radares2 6.2.4 et rizine 0.9.1, intégrés à partir de la source dans le cache de recherche, ont été exécutés sur le même objet.`pdc`des empreintes`div10s`comme `rax = rax * 0x66666667; rcx >>>= 0x3f; rax >>= 0x22; eax += ecx`, pseudo-C au niveau du registre sans pliage, et `table_switch`comme `eax = dword [rcx + rax*4]`avec un `RELOC 32 .rodata`commentaire sur le `lea`. rizin n' a pas de décompilateur intégré; son `aoi`imprime le RzIL de la `imul`Lldb-1703.0.236.21 charge l'objet ELF et son`disassemble -n div10s`Il imprime les mêmes sept instructions que `llvm-objdump`Capstone 5.0.7 décode les mêmes octets et rapporte des groupes tels que `ret`et `mode64`Le code de décoder est un code de décodeur, et un langage n'est jamais dans la sortie d'un décodeur. IDA Pro et Binary Ninja n'ont pas été exécutés ici.

L'Explorateur de compilation est l'outil pour la direction avancée.`idioms.c`, choisissez une entrée de compilateur telle que "x86-64 clang 18.1.0" ou "ARM64 GCC 14.2.0", et définissez les drapeaux. Le service répertorie environ 1.050 entrées de compilateur C et dispose d'une API: `GET /api/compilers/c`renvoie les identifiants et `POST /api/compiler/<id>/compile`L'article ACM Queue de Matt Godbolt de 2019 traverse la division, le décompte de la population, les idiomes conditionnels enchaînés et de somme avec cet outil. Utilisez-le pour la sortie du GCC, que cette machine ne peut pas produire, et comparez les seuils de la table de saut du GCC et `-O2`vectorisation par défaut du CCG 12 avec la sortie de clang dans les fixtures.

## Faites-le

Exécutez la démo:

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization
python3 code/main.py
```

La sortie est un rapport JSON. Pour chacune des 18 constructions, il énumère chaque fonction et ses idiomes, avec type, forme source, adresses de preuve, vérification et les champs de détails.`-O2`contre `-O3`La comparaison des deux architectures et de la`unverified`La liste.

Forme attendue:

```json
{
  "builds": {
    "idioms-x86_64-O2": {
      "arch": "x86_64", "level": "O2", "flags": "x86_64",
      "functions": {
        "div7u": [{"kind": "division", "source": "x / 7", "evidence": ["0x30", "0x32", "0x39", "0x3d", "0x3f", "0x41", "0x43"], "verified": true, "form": "magic multiply with add fix-up", "magic": "0x24924925", "effectiveMagic": "0x124924925", "shift": 35, "...": "..."}],
        "table_switch": [{"kind": "switch", "source": "switch (k) as lookup table", "table": ".rodata+0x28", "entries": 7, "values": [10, 20, 30, 40, 50, 60, 71], "...": "..."}],
        "sum_n": [{"kind": "vector loop", "lanes": 4, "accumulators": 2, "elementsPerIteration": 8, "...": "..."}]
      }
    }
  },
  "kindMatrix": {"idioms-x86_64-O2": {"division": 5, "modulo": 3, "switch": 6, "vector loop": 3, "...": "..."}},
  "sameCodeAtO2AndO3": {"x86_64": {"identical": true}, "arm64": {"identical": false, "differences": ["..."]}},
  "unverified": []
}
```

Lire `code/main.py`dans cet ordre:

1. `parse_listing()`tourne le `llvm-objdump -d -r`Les deux architectures doivent être intégrées à des fonctions, à des instructions et à des rélocalisations jointes.
2. `ElfFile`Il lit des sections, des symboles et `.rela.*`les entrées, et `words()`Il lit les entrées de table.
3. `magic_numbers()`- Je suis là .`divisor_from_magic()`, et `evaluate_division()`sont la dérivation Granlund-Montgomery vers l'avant, vers l'arrière, et comme la séquence d'instructions.
4. `parse_source()`extrait les constructions des besoins de vérification du dossier C.
5. `Evaluator`est le bloc par bloc symbolique qui enregistrent les noms et les emplacements.
6. `find_divisions()`et `find_modulos()`correspondent au multiplication, au changement, au fixement, au multiplication et au pouvoir de deux formes.
7. `find_switches()`classe les tables de saut, les tables de recherche, les cartes linéaires, sélectionne les chaînes, compare les arbres et compare les chaînes.
8. `find_vector_loops()`- Je suis là .`find_unrolled_loops()`- Je suis là .`find_tail_calls()`- Je suis là .`find_inlined_calls()`- Je suis là .`find_selects()`- Je suis là .`find_memory_builtins()`- Je suis là .`find_stack_protector()`- Je suis là .`find_cet()`, et `find_strength_reduction()`Couvre le reste.
9. `analyze_function()`les organise dans l'ordre, en introduisant le nom du résultat de chaque idiome dans le prochain passe d'évaluation.

Il y a trois commandes de plus.`python3 code/main.py scan FILE.objdump.txt [FUNCTION]`Il utilise le reconnaisseur sur une liste, avec l'objet et le fichier C correspondant à côté. `python3 code/main.py magic DIVISOR [signed]`imprime le multiplicateur, le déplacement, l'erreur et si l'identité tient les valeurs de bord. `python3 code/main.py figure-data`imprime les données que la figure intègre.

## Utilisez-le

Utilisez le reconnaisseur dans cet ordre sur un nouvel objet que vous avez construit.

1. Capture `llvm-objdump -d -r`Le reconnaisseur lit l'architecture de l'en-tête de liste.
2. On court .`scan`sur la liste et lire le `unverified`Chacun est soit une construction que le parseur source n'a pas vue, soit une forme que le reconnaisseur a mal lue, et les deux méritent d'être examinés.
3. Pour chaque division, lisez le`algorithm`Le champ.`clangMatchesAlgorithm`Si c'est faux, le compilateur a eu une raison que vous ne connaissez pas encore; écrivez les constantes.
4. Pour chaque commutateur, comparez le `lowering`Une table qui devient un arbre, ou un arbre qui devient une table, marque un seuil que vous pouvez nommer à partir de la source LLVM.
5. Pour chaque boucle, lisez `elementsPerIteration`et `scalarRemainderLoop`, puis confirmez que les deux doses de leçon 33 rapports sont une boucle source.
6. Exécutez un décompilateur sur le même objet, citez son texte à côté de chaque idiome, et enregistrez les autres.

## Laboratoire interactif

### Laboratoire A: déduire un nombre magique, puis le trouver

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization/code
python3 -q
```

Je vais courir .

```python
from main import *
magic_numbers(10), magic_numbers(10, signed=True)
magic_numbers(7), magic_numbers(7, signed=True)
divisor_from_magic(0x66666667, 34, signed=True), divisor_from_magic(0x66666667, 33, signed=True)
```

Compte `ceil(2^35 / 10)`sur le papier avant le premier appel.`unsigned div13(unsigned x) { return x / 13; }`à une copie de `idioms.c`, construire à `-O2`avec le commandement en `build_fixtures.sh`, et vérifier la constante dans la liste contre `magic_numbers(13)`- Je suis désolé .

### Laboratoire B: la table et l'arbre

```python
listing = load_listing("idioms-x86_64-O2.objdump.txt")
elf = load_elf("idioms-x86_64-O2.o")
sources = load_source("idioms.c")
[i.to_dict() for i in analyze_function(listing, "table_switch", elf, sources)]
[i.to_dict() for i in analyze_function(listing, "sparse_switch", elf, sources)]
elf.words(".rodata", 0x28, 7)
```

Le changement`71`Retour à `70`dans une copie de `table_switch`, reconstruire à`-O2`, et regarder la table de recherche devient une carte linéaire. Puis changer une valeur de cas à 1000 et regarder rester une`ArrayKind`La table.

### Laboratoire C: la boucle à trois réglages

```python
for name in ("idioms-x86_64-O1", "idioms-x86_64-O2", "idioms-x86_64-novec-O2"):
    print(name, [i.source for i in analyze_function(load_listing(name + ".objdump.txt"), "sum_n", load_elf(name + ".o"), sources)])
```

Dessinez le graphique de contrôle des flux de `sum_n`à`-O2`à la main, marquer le garde, le masque, le corps vectoriel, la réduction et le reste de la boucle, et comparer avec le `detail`les champs.

### Laboratoire D: le quotient partagé

```python
for name in ("idioms-x86_64-O1", "idioms-arm64-O1"):
    print([(i.kind, i.source, i.detail.get("sharedQuotient")) for i in analyze_function(load_listing(name + ".objdump.txt"), "digest", load_elf(name + ".o"), sources)])
```

Lisez le `-O1`liste des `digest`et trouver les deux instructions qui combinent le quotient avec son signe fixation sur x86-64.

## Laboratoire de pratique

Ajoutez des trucs à la lumière et étendons le reconnaisseur.

Les exigences:

- Écrivez`popcount`- Je suis là .`ctz`, et `rotl`Les fonctionnalités de la construction du CCG et de la rotation de la position de la`-O2`avec et sans`-mpopcnt -mbmi`, et engager les objets avec leurs listes.
- Reconnaissez le`0x55555555`- Je suis là .`0x33333333`- Je suis là .`0x0f0f0f0f`- Je suis là .`0x01010101`séquence de dénombrement de la population qui `TargetLowering::expandCTPOP`émet et le rapporte comme `popcount`avec les quatre constantes comme preuve.
- Reconnaître`rep bsfl`comme le `tzcnt`- ou...`bsf`la codage et `roll %cl`comme le rotation, et décodez les deux à la construction.
- Ajoutez les nouveaux types à `parse_source()`la vérification, donc une `popcount`L' idiome n' est vérifié que lorsque la source appelle `__builtin_popcount`- Je suis désolé .
- Ajoutez un test par idiome qui lit les constantes de la liste des fichiers.

Le succès signifie`python3 code/main.py`rapporte les trois nouveaux types avec `verified: true`Dans les deux constructions, et la suite rapporte toujours un vide `unverified`La liste.

## Leçon d'artisanat

Cette leçon comprend `outputs/skill-compiler-idioms-and-optimization.md`. Utilisez-le pour transformer une liste, un objet et sa source en un rapport de langue vérifié avec une comparaison de décompilés. Il nécessite des adresses de preuve pour chaque revendication, la dérivée du nombre magique pour chaque division, le type de réduction et la source d'entrée pour chaque commutateur, et une ligne de décompilés citée pour chaque langue.

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La vérification doit prouver:

- Le parseur de liste lit 25 fonctions de la `-O2`Objet, conserve les octets, et attache `R_X86_64_PC32 .rodata+0x24`à la `leaq`dans `table_switch`
- le lecteur ELF retourne `[10, 20, 30, 40, 50, 60, 71]`de `.rodata+0x28`et dix délocalisations en `.rela.rodata`
- `magic_numbers`Retour `0xcccccccd`avec la troisième,`0x66666667`avec 34, `0x124924925`avec 35 et le formulaire additif, `0x55555556`avec 32, et `0xaaaaaaab`avec 33
- l'identité tient les valeurs de bord de 32 bits pour chaque divisor de 2 à 64 qui n'est pas une puissance de deux, signée et non signée
- `divisor_from_magic`récupère 10, 5, 7 et 3 de leurs paires et rejette une mauvaise paire
- le calcul de la séquence de fixation additive `n // 7`exactement
- Les quatre divisions x86-64 indiquent le multiplicateur droit, le changement, le diviseur et la forme, et les quatre divisions AArch64 décodent `mov`et `movk`
- `-O0`utilisations `divl`et `idivl`- Je suis là .`-O0 -fast-isel=false`utilise le multiplicateur, et `digest`à`-O0`mélange les deux
- le modulo est reconnu comme un retour multiplicateur,`msub`, puissance à deux, `sdiv`plus `mul`plus `subs`, et le matériel `idivl`
- Un quotient partagé maintient à la fois la division et le modulo à `-O1`
- Les cinq réductions sont classées en `-O2`sur les deux architectures, avec la formule, les valeurs, les objectifs, les pivots et l'expression sélectionnée
- `-O0`Les commutateurs sont des tables de saut sur x86-64 et comparer les chaînes sur AArch64
- le `digest`Le commutateur est une table de saut, puis une table de recherche avec le facteur 16 plié, puis une table de saut avec quatre cibles distinctes sur x86-64, et un arbre de comparaison avec le test d'intervalle 9 à 10 sur AArch64
- `-fno-jump-tables`produit des arbres de comparaison, et le cas 16 `dispatch`lisent ses objectifs à partir de délocalisations sur x86-64 et d'un tableau en octets comprimés sur AArch64
- le boucle vectorielle rapporte 4 voies, 2 accumulateurs, 8 éléments par itération, le masque, la garde, la réduction et le boucle restant
- Les boucles à trajet fixe sont entièrement vectoriées ou entièrement déroulées, et `-fno-vectorize`déroulées à 8
- `-O1`n'a pas de vectorisation, `-O2`et `-O3`sont identiques sur x86-64, et sur AArch64 ils diffèrent seulement en `call_switch`Où ?`-O3`Il contient 17 instructions et `-O2`a 18
- `-fno-builtin`transforme la boucle de magasin en boucle de magasin vectoriée et garde `memcpy`un appel
- Les appels de la queue, le `memset`le remplacement, la copie en ligne de 16 octets, et le `-O0`les copies à deux mouvements sont reconnues
- La callee en ligne est trouvée à partir de `-O1`par sa forme corporelle
- sélectionne le décode à `a > b ? a : b`- Je suis là .`-x < 0 ? x : -x`- Je suis là .`(x > 0)`, et `sum % 10 < 5 ? sum % 10 : 5`
- le rapport de protection de pile `%fs:0x28`, la fente, et `__stack_chk_fail`, et `endbr64`est déclaré sur chaque fonction de la structure du CET
- le parseur source lit les paramètres du pointeur de fonction, les cas de commutateur, les appels et les noms de l'accumulateur
- chaque idiome dans chaque construction est vérifié et les matrices de type correspondent
- la figure interactive intègre les mêmes données que l'analyse produit

## Mode de défaillance de la production

| Failure | What the weak tool reports | What the analysis must prove |
|---|---|---|
| Multiply by a large odd constant | "multiplies by 1717986919" | The constant is `ceil(2^s / d)` for one `d`, the error satisfies the bound, and the shift is the smallest valid one |
| Add fix-up after a 32-bit shift | Five arithmetic instructions | The multiplier is 33 bits wide and the tail is the add form of the same method |
| Same multiplier, different shift | The same divisor | The shift selects the divisor: 34 is 10, 33 is 5 |
| Power-of-two signed modulo | "adds 7, masks, subtracts" | The bias rounds toward zero and the result is `x % 8` |
| Indexed load from `.rodata` | "a jump table" | A lookup table of values, with entries read from the bytes at addend plus 4 |
| Affine arithmetic and a `cmov` | "computes 10k + 10" | A switch lowered as a linear map, verified against every source case |
| Compare chain with ordered splits | "a sequence of ifs" | A binary search over the case constants with the pivots named |
| Backward branch to a default block | "a loop" | The target leaves the region with an unconditional jump; it is a shared tail |
| Two back edges in one function | "two loops" | One source loop: the vector body and its scalar remainder |
| `jmp memset@PLT` with no `ret` | "the function has no return" | A tail call that returns through the callee, replacing a store loop |
| Callee missing from the call graph | "never called" | Inlined: the callee's body shape appears in the caller |
| `%fs:0x28` load and compare | "reads thread-local data" | The stack protector canary, with the fail path to `__stack_chk_fail` |
| Four odd bytes at function start | "data before the function" | `endbr64`, the CET landing pad |
| Decompiler shows raw arithmetic | "the tool is wrong" | The tool did not apply a fold; the derivation gives the source form and the quoted output records the gap |

## Règlement d'exploitation

Partagez la forme, faites revenir la dérivée et écrivez les constantes qui la prouvent. Rapportez chaque idiome avec ses adresses de preuve et sa forme source décodée, vérifiez chacun contre la source que vous avez compilée, et citons ce que chaque décompilateur a imprimé pour les mêmes octets.

## Pour en savoir plus

- [Granlund, Montgomery, "Division by Invariant Integers using Multiplication" (PLDI 1994)](https://gmplib.org/~tege/divcnst-pldi94.pdf)
- [Warren, Hacker's Delight, 2nd edition, Chapter 10: Integer Division By Constants](https://www.informit.com/store/hackers-delight-9780321842688)
- [Lemire, Kaser, Kurz, "Faster Remainder by Direct Computation" (2019)](https://arxiv.org/abs/1902.01961)
- [libdivide, header-only division by runtime constants](https://github.com/ridiculousfish/libdivide)
- [LLVM `DivisionByConstantInfo.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/DivisionByConstantInfo.cpp)
- [LLVM `SwitchLoweringUtils.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SwitchLoweringUtils.cpp)et [`TargetLoweringBase.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/TargetLoweringBase.cpp)
- [LLVM `SimplifyCFG.cpp`, `simplifySwitchLookup`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/Utils/SimplifyCFG.cpp)
- [LLVM Auto-Vectorization documentation](https://llvm.org/docs/Vectorizers.html)et [LLVM 3.4 release notes](https://releases.llvm.org/3.4/docs/ReleaseNotes.html)
- [LLVM `LoopIdiomRecognize.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/Scalar/LoopIdiomRecognize.cpp)
- [LLVM `X86CmovConversion.cpp`](https://github.com/llvm/llvm-project/blob/release/20.x/llvm/lib/Target/X86/X86CmovConversion.cpp)
- [LLVM `StackProtector.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/StackProtector.cpp)et [`X86ISelLoweringCall.cpp`, `getIRStackGuard`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLoweringCall.cpp)
- [glibc `sysdeps/x86_64/nptl/tls.h`, `tcbhead_t`](https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/x86_64/nptl/tls.h)
- [GCC `tree-switch-conversion.cc`](https://github.com/gcc-mirror/gcc/blob/master/gcc/tree-switch-conversion.cc)- Je suis là .[GCC Optimize Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html), et [GCC 12 release notes](https://gcc.gnu.org/gcc-12/changes.html)
- [Ghidra decompiler `ruleaction.cc`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/decompile/cpp/ruleaction.cc)
- [angr decompiler documentation](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [radare2 book: decompilers](https://book.rada.re/arch/decompile.html)
- [Compiler Explorer API](https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/API.md)
- [Godbolt, "Optimizations in C++ Compilers: A practical journey", ACM Queue 17(5), 2019](https://queue.acm.org/detail.cfm?id=3372264)
- [Microsoft, `/GS` (Buffer Security Check)](https://learn.microsoft.com/en-us/cpp/build/reference/gs-buffer-security-check)
