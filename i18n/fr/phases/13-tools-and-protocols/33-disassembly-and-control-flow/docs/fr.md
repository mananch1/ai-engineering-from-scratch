# Déassemblage et contrôle du flux: de bytes à un graphique que vous pouvez vérifier

> Une liste de démontage est une affirmation sur les octets qui sont des instructions, et cette leçon construit le décodeur et le graphique qui vérifient la affirmation.

**Type:** Build
**Languages:** Python, C, x86-64 assembly
**Prerequisites:** Phase 13 · 32 (ELF from scratch)
**Time:** ~150 minutes

## Objectifs d'apprentissage

- Décoder le sous-ensemble d'instructions x86-64 dans les fiches du registre à partir des règles du volume 2 du chapitre 2 de l'Intel SDM.
- Prouvez le décodeur contre `llvm-objdump -d`sur toute instruction de toute construction.
- Exécutez une balayage linéaire et une descente récursive sur les mêmes octets et expliquez où chacun se trompe.
- Coupez les instructions en blocs de base et construisez un graphique de contrôle-flu avec des bords tapés.
- Compute les dominateurs immédiats avec l'algorithme itératif de Cooper, Harvey et Kennedy, et trouve des boucles à partir des bords arrière.
- Récupérer les cibles d'un saut indirect des entrées de relocalisation qui pointent à sa table.

## Le problème

Vous avez compilé`ledger.c`à`-O0`et a demandé à un agent ce que `grade`L'agent a appelé un démonteur, lu la liste et a répondu que la fonction se termine en`jmpq *%rax`Il a ensuite écrit un pseudocode avec un seul`return 0`- Je suis désolé .

Vous ouvrez le même objet en .`llvm-objdump -d`Et voir quatre blocs après le saut qui stocke 4, 3, 2 et 1. La liste les avait tout le temps. Rien dans la liste ne dit comment l'exécution les atteint, donc l'outil les a traités comme code mort.

Une liste répond à une seule question: qui fait des octets de quelle instruction. Elle ne dit pas quelles instructions sont exécutées, dans quel ordre ou d'où. Pour les répondre, vous avez besoin de trois structures supplémentaires: les octets que le flux de contrôle peut atteindre, les blocs de base et les bords entre eux.

réa 6.5.0 n'a pas de démonteur propre. Pour une analyse approfondie, son README vous demande d'apporter Ghidra, Hopper ou IDA. Pour l'inspection hors ligne, rea peut également appeler un objdump GNU ou un Rizin que vous installez. Son graphique d'appels s'arrête sur des cibles non résolues: "Les appels/appels ne contiennent que des fonctions résolues.... Le flux informatique sans cible reste inconnu". Cette leçon construit la partie qui produit ces faits, de sorte que vous pouvez les juger quand un outil les rapporte.

```figure
re-33-sweep-vs-descent
```

## Le format d'instruction a une règle par champ

Le volume 2 de l'Intel SDM, la figure 2-1 donne l'ordre en octets de chaque instruction. D'abord viennent des préfixes optionnels, puis un à trois octets de code optique, puis un octet ModRM optionnel et un octet SIB optionnel. Un déplacement de 0, 1, 2 ou 4 octets suit, puis un déplacement immédiat de 0, 1, 2 ou 4 octets. L'ensemble de l'instruction est de 15 octets au maximum (SDM 2.3.11). Le volume 3 de l'APM AMD, chapitre 1, sections 1.1 à 1.7 décrivent le même format en moins de pages et constituent une bonne vérification croisée.

Chaque champ décide de ce que signifie le champ suivant. Un décodeur est donc une chaîne de petites décisions, et chaque décision a une phrase dans le manuel derrière elle.

### Les préfixes se composent de quatre groupes

Le SDM 2.1.1 répertorie les préfixes hérités.`F0`verrouillage et les préfixes de répétition `F2`et `F3`.Le groupe 2 détient les six segments de surpoids (`2E`est `%cs`Le groupe 3 est `66`Le groupe 4 est le groupe 4`67`, la taille de l'adresse est annulée.

Le manuel indique qu'un préfixe de chaque groupe peut être utilisé.`ledger-x86_64-O0.o`à l' équilibre `0xf2`Il lit`66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00`, cinq exemplaires de `66`dans une rangée. `llvm-objdump`Il l' imprime comme `nopw %cs:(%rax,%rax)`Le décodeur recueille chaque préfixe en octets qu'il voit et demande plus tard lequel en compte.

`66`Pour les instructions SSE, il s'agit d'un préfixe obligatoire qui sélectionne le code optique, donc `66 0f ef c0`est `pxor %xmm0, %xmm0`et pas un 16 bits quoi que ce soit.`F3 0f 6f`est `movdqu`Le décodeur recherche d'abord la paire de préfixes obligatoires et la seconde octet opcode.`66`C'est une opéra à 16 bits.

### REX ajoute quatre bits et doit être placé à côté du code op

L' SDM 2.2.1 définit REX comme un octet de `40`à `4F`avec la forme `0100WRXB`- Je suis là .`W`sélectionne un opérand 64 bits. `R`est étendu au champ de réglementation ModRM. `X`élargit le champ de l'indice SIB. `B`est étendu sur le champ ModRM r/m, le champ de base SIB ou le registre dans le octet opcode (tableau 2-4). REX doit apparaître juste avant le octet opcode ou les octets d'échappement, et un REX suivi d'un préfixe hérité est ignoré.

Dans `48 89 e5`le byte `48`est REX avec `W`le code optique `89`est `MOV r/m, r`, et ModRM `e5`sélectionne `%rsp`dans `%rbp`. Retirez le REX et les mêmes octets sont `movl %esp, %ebp`- Je suis désolé .

REX change également les noms des registres en octets.`%spl`- Je suis là .`%bpl`- Je suis là .`%sil`, et `%dil`Au lieu de `%ah`- Je suis là .`%ch`- Je suis là .`%dh`, et `%bh`Le décodeur garde deux tables pour cette raison.

### Les codes optiques sont répartis en deux cartes

Un code op d'un octet est le cas courant. `0F`s'échappe à la carte de deux octets, et `0F 38`ou `0F 3A`Les codes optiques utilisent le champ ModRM reg comme trois autres bits opcode, écrits `/digit`dans le manuel. `83 /7`est `cmp`avec un 8 bits immédiat, `83 /5`est `sub`, et `FF /4`est l' indirect `jmp`- Je suis désolé .

Le décodeur en `code/main.py`Il couvre les opcodes des quatre objets fixes plus quelques voisins.`mov`- Je suis là .`lea`- Je suis là .`movslq`- Je suis là .`movzbl`- Je suis là .`imul`, les changements,`test`, et `idiv`Il continue avec `push`- Je suis là .`pop`- Je suis là .`call`- Je suis là .`ret`, les seize sauts conditionnels, le multi-byte `nop`, et cinq codes optiques SSE2.`DecodeError`Un décodeur qui devinera est pire qu'un décodeur qui s'arrête.

### ModRM et SIB sélectionnent les opérands

Le SDM 2.1.3 divise le ModRM en `mod`(Bits 7:6), `reg`- Le nombre de personnes concernées`r/m`Il est possible de le faire.`mod = 11`signifie que r/m est un registre. `mod = 00`- Je suis là .`01`, et `10`mémoire moyenne sans déplacement, un déplacement de 8 bits ou un déplacement de 32 bits. `r/m = 100`signifie qu'un byte SIB suit.`scale`(7, 6), `index`- Le nombre de personnes concernées`base`(2:0), et un indice de `100`signifie qu'il n'y a pas de registre d'index.

Deux lignes spéciales dans le tableau 2-2 sont en mode 64 bits. `mod = 00`avec `r/m = 101`- Ce n' est pas vrai .`(%rbp)`. Il s'agit d'un déplacement de 32 bits par rapport à l'instruction suivante (SDM 2.2.1.6). `mod = 00`avec une base SIB de `101`est un déplacement de 32 bits sans registre de base.

Prenez .`8d 44 08 fa`de `grade`à`-O0`Le code optique .`8D`est `lea`- ModRM `44`est `01 000 100`: un déplacement de 8 bits, rég `%eax`, et une SIB suit.`08`est `00 001 000`: échelle 1, indice `%rcx`, base `%rax`- Le byte de déplacement`fa`Le texte de AT&T est `leal -0x6(%rax,%rcx), %eax`, qui compute `score / 10 - 6`, l'indice de commutation.

### Le déplacement est où l'éditeur de lien écrit

`48 8d 0d 00 00 00 00`est `leaq (%rip), %rcx`Les quatre octets zéro sont le champ de déplacement d'un opérand relatif à RIP. Dans un objet relocalisable, ils sont des placeholders.`.rela.text`à l' équilibre `0x69`dit `R_X86_64_PC32.rodata-0x4`. Le psABI (tableau 4.9) définit `R_X86_64_PC32`comme `S + A - P`: l'adresse du symbole plus l'additif moins l'adresse du champ.

Le processeur ajoute le champ à l'adresse de la prochaine instruction, de sorte que la cible finale est `S + A + (end - P)`Pour ça .`lea`le champ est les quatre derniers octets, donc `end - P`est 4 et la cible est `.rodata + (-4) + 4 =.rodata + 0x0`- l'addendum de -4 compense ces quatre octets à l'avance.`.rodata+0x10`sur le `-O1` `lea`les moyens `.rodata+0x14`- Je suis désolé .

Le décodeur enregistre le décalage du champ de déplacement à l'intérieur de chaque instruction. `resolve_reference()`Les résultats de l'opérand sont les suivants:`leaq (%rip), %rcx`Dans la table à`.rodata+0x0`" " Je suis là.

### Les instantanés sont de 32 bits au maximum, à une exception exception.

Le SDM 2.2.1.5 indique que les instantanés restent au plus 32 bits en mode 64 bits et sont étendus par signe.`MOV r64, imm64`(le secteur de l'énergie)`B8+r`avec `REX.W`), que l'AT&T appelle la syntaxe `movabsq`Le décodeur lit les 8 bits immédiats comme signé pour l' arithmétique (`83 c1 fa`est `addl $-0x6, %ecx`) et non signés pour les compteurs de changements et les sélecteurs de mélange (`pshufd $0xee`Ça correspond à quoi ?`llvm-objdump`Des empreintes.

À proximité des branches,`push`, et `pop`par défaut sur une opérand 64 bits et ignorer `66`(MDD 2.2.1.7) C'est pourquoi le décodeur les nomme `pushq`- Je suis là .`popq`- Je suis là .`callq`, et `retq`sans lire un préfixe de taille opérande.

## Décodez une instruction à la main

Avant de faire confiance au code, décodez.`48 69 c0 67 66 66 66`de `grade`sur le papier.

1. `48`Il est là .`40`à `4F`, donc c' est REX avec `W = 1`La taille de l'opéra est de 64 bits.
2. `69`est `IMUL r, r/m, imm32`- Je suis désolé .
3. `c0`est ModRM `11 000 000`: formulaire de registre, rég `%rax`, r/m `%rax`- Je suis désolé .
4. Les quatre bytes suivantes sont les plus immédiates, les plus petites:`0x66666667`- Je suis désolé .
5. L'ordre AT&T est immédiat, source, destination: `imulq $0x66666667, %rax, %rax`- Je suis désolé .

C'est 7 octets, et `llvm-objdump`La constante est la réciproque qui permet à clang de se diviser par 10 avec un multiplicateur et le `sarq $0x22`Vous rencontrerez cette paire à nouveau dans la leçon 34, où le soulèverain la renverra en`score / 10`- Je suis désolé .

Le décodeur fait ces cinq étapes en `Decoder.decode()`- Je suis là .`_State.modrm()`, et `_State.finish()`Lisez-les dans cet ordre et correspondrez chaque branche à la section SDM nommée dans sa chaîne de documents.

## L'analyse linéaire

Schwarz, Debray et Andrews décrivent le balayage linéaire dans la section 3.1 de "Discontribing of Executable Code Revisited" (WCRE 2002). Commencez à la première octet de la section de code, décodez une instruction, continuez à la octet après elle, et arrêtez à la fin. `objdump`fonctionne de cette façon, et aussi fait `llvm-objdump`- Je suis désolé .

```python
def linear_sweep(code: bytes, start: int = 0, end: int | None = None) -> SweepResult:
    position = start
    while position < end:
        try:
            instruction = decode(code[:end], position)
        except DecodeError:
            undecodable.append((position, code[position]))
            position += 1
            continue
        instructions.append(instruction)
        position = instruction.end
```

La méthode a un mode d'échec, et la figure 1 du papier le montre. Les octets qui sont des données sont décodés en tant qu'instructions, et le décodeur manque ensuite les limites réelles des instructions pendant un certain temps. Sur x86, le décodeur se ré-synchronise généralement dans quelques bytes, ce qui rend l'erreur difficile à voir. La détection ne se fait que sur un code opcal invalide, et la conclusion du journal est que ces algorithmes "échouent silencieusement".

Le balayage est aussi l'oracle de cette leçon.`llvm-objdump -d`sur chaque instruction de chaque pièce, et `llvm-objdump`Le texte capturé vit dans`code/fixtures/*.objdump.txt`. Le test analyse chaque ligne en adresses, octets et texte, puis laisse tomber les commentaires après `#`Les quatre objets contiennent 86, 56, 84 et 11 instructions, 237 au total, et le décodeur correspond à tous.

## Descent récursif

La deuxième méthode est appelée traversation récursive. Commencez à un point d'entrée connu et décodez jusqu'à un transfert de contrôle. Pour un saut direct, continuez à la cible. Pour une branche conditionnelle, continuez à la cible et à l'instruction suivante, et pour un appel, continuez après, car le callee revient. Pour un retour, arrêtez. Gardez un ensemble d'adresses visitées afin que la marche se termine.

```python
if instruction.flow == "branch":
    worklist.append(instruction.target)
    address = instruction.end
elif instruction.flow == "indirect-jump":
    table = resolver(instruction, instructions) if resolver else None
    if table is None:
        unresolved.append(instruction)
        break
```

Le mode de défaillance est le contraire. Un saut indirect n'a pas de cible dans ses octets. La marche s'arrête là, et chaque bloc accessible uniquement par le saut reste inconnu. La figure 2 du papier montre un deuxième danger: un ensemble de cibles dérivé de relocalisations qui pointent vers le milieu d'une instruction.`main.py`enregistrer ce cas comme un conflit et arrêter ce chemin, de sorte que la chevauchement est visible au lieu de décoder silencieusement deux fois.

Andriesse et d'autres ont mesuré neuf désassemblaires sur 981 binaires générés par le compilateur (USENIX Security 2016). Les données en ligne et le code superposé étaient plus rares dans la sortie réelle du compilateur que la littérature a supposé, et le hard primitif était la récupération des limites de fonction. Gardez à l'esprit les deux résultats. Le piège dans la section suivante est rare dans la sortie du compilateur.`grade`C'est commun.

## Un lieu où la balayage est mauvaise et la descente correcte

`code/fixtures/sweep-trap.s`est de 24 octets d'assemblage écrit à la main avec une table de quatre octets à l'intérieur du corps de la fonction.`movzbl (%rcx,%rax), %eax`et saute dessus.

```asm
lookup:
        movslq  %edi, %rax
        leaq    .Ltable(%rip), %rcx
        movzbl  (%rcx,%rax), %eax
        jmp     .Ldone
.Ltable:
        .byte   0x31, 0xc0, 0x90, 0x48
.Ldone:
        addl    $1, %eax
        retq
```

Construisez-le avec `clang -target x86_64-unknown-linux-gnu -c sweep-trap.s`Les quatre octets de la table ont été choisis pour décoder comme des instructions plausibles.`31 c0`est `xorl %eax, %eax`- Je suis là .`90`est `nop`- Je suis là .`48`est un préfixe REX, donc le balayage le rejoint avec les trois octets du réel `addl $1, %eax`et des empreintes`addq $0x1, %rax`Les prisonniers .`llvm-objdump`la sortie montre exactement que:

```text
       e: eb 04                        	jmp	0x14 <lookup+0x14>
      10: 31 c0                        	xorl	%eax, %eax
      12: 90                           	nop
      13: 48 83 c0 01                  	addq	$0x1, %rax
      17: c3                           	retq
```

Rien dans cette liste ne semble cassé.`retq`La liste est toujours erronée de trois façons.`0xe`cibles `0x14`, et aucune instruction dans la liste commence à `0x14`- L' instruction qui fonctionne à`0x14`est un 32-bit `addl`, et la liste montre une 64 bits `addq`Les quatre octets de la table apparaissent comme code.

Descent récursif de la`lookup`Le symbole décode six instructions, suit le saut à `0x14`, et des rapports `addl $0x1, %eax`Il ne revendique aucun des quatre bytes de table et aucun des huit bytes de remplissage avant .`plus_one`- Je suis là .`trap_report()`dans `main.py`imprime la comparaison:

```json
"linearSweep": {
  "instructions": 9,
  "atJumpTarget": {"starts": "0x13", "text": "addq\t$0x1, %rax", "startsAtTarget": false},
  "claimedDataBytes": 4,
  "claimedPaddingBytes": 8
},
"recursiveDescent": {
  "instructions": 6,
  "atJumpTarget": "addl\t$0x1, %eax",
  "claimedDataBytes": 0,
  "claimedPaddingBytes": 0
}
```

Le test `test_sweep_and_objdump_share_the_same_mistake`affirme que `llvm-objdump`imprimé `addq $0x1, %rax`à`0x13`L'oracle est une balayage linéaire et hérite de sa tache aveugle.

## Blocs de base

Allen définit le bloc de base dans "Control Flow Analysis" (1970). Il s'agit d'une séquence d'instructions qui n'est entrée que lors de sa première instruction et laissée que lors de sa dernière. Yakdan et d'autres le réaffirment pour la décompilation dans la section II.A.2 de "No More Gotos" (NDSS 2015), avec le nœud d'en-tête comme entrée de fonction.

La construction est une analyse à deux passes sur les instructions décodées, triées par adresse. Le premier passe recueille les leaders. Un leader est l'entrée, une branche ou une cible de saut, une cible de table ou l'instruction après tout transfert de contrôle. Le deuxième passe démarre un nouveau bloc à chaque leader et à chaque écart dans la séquence d'adresses.

```python
for instruction in self.instructions:
    if instruction.address in leaders or current is None or current.end != instruction.address:
        current = Block(instruction.address, [])
        self.blocks[instruction.address] = current
    current.instructions.append(instruction)
```

Les bords suivent la dernière instruction de chaque bloc.`jump`Une branche conditionnelle a deux, marquée`taken`et `fallthrough`Une instruction simple à la fin d'un bloc passe à l'autre bloc, ce qui se produit lorsque l'instruction suivante est une cible branche.`table`, ou aucun, et le bloc est enregistré en `unresolved`- Je suis désolé .

Deux autres faits sont découverts dans le même scan.`grade`à`-O0`Un bloc dont l'adresse de départ tombe dans une instruction de l'inscription d'une autre méthode est le cas de chevauchement du piège.

## Le graphique de contrôle des flux du total

`total`à`-O1`est de 31 octets et 13 instructions:

```text
      40: 85 f6                        	testl	%esi, %esi
      42: 7e 18                        	jle	0x5c <total+0x1c>
      44: 89 f1                        	movl	%esi, %ecx
      46: 31 d2                        	xorl	%edx, %edx
      48: 31 c0                        	xorl	%eax, %eax
      4a: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)
      50: 03 04 97                     	addl	(%rdi,%rdx,4), %eax
      53: 48 ff c2                     	incq	%rdx
      56: 48 39 d1                     	cmpq	%rdx, %rcx
      59: 75 f5                        	jne	0x50 <total+0x10>
      5b: c3                           	retq
      5c: 31 c0                        	xorl	%eax, %eax
      5e: c3                           	retq
```

Les dirigeants sont`0x40`(entrée), `0x44`(après le `jle`), `0x50`(objectif de la `jne`), `0x5b`(après le `jne`), et `0x5c`(objectif de la `jle`Cinq blocs, cinq bords:

| From | To | Kind | Why |
|---|---|---|---|
| `0x40` | `0x44` | fallthrough | `jle` not taken, count is positive |
| `0x40` | `0x5c` | taken | count is zero or negative, return 0 |
| `0x44` | `0x50` | fallthrough | the `nopw` is alignment inside the block, execution passes through it |
| `0x50` | `0x50` | taken | `jne 0x50`, the loop |
| `0x50` | `0x5b` | fallthrough | loop exit, return the sum |

Le six-byte `nopw`à`0x4a`est de rembourrage de sorte que la boucle se dirige vers `0x50`Il est accessible, donc il appartient au bloc`0x44`Le 1 byte .`nop`à`0x5f`après la dernière .`retq`La coupe sur la fonction fait un sixième bloc pour elle, et `unreachable()`Il l'a fait.

À `-O2`La même fonction est de 140 octets avec deux boucles.`0x80`ajoute huit valeurs par passe avec `paddd`, et le reste escalier court à `0xc0`Le CFG a deux bords arrière.`0x80 -> 0x80`et `0xc0 -> 0xc0`, et la suite de tests vérifie les deux.

## Les dominateurs avec l'algorithme itératif

Un bloc .`d`domine un bloc `b`Quand chaque chemin de l'entrée à `b`passe par là `d`Le dominateur immédiat de`b`est le dominateur de `b`C' est le plus proche de `b`La notion remonte à Prosser (1959) à Allen (1970), et c'est la structure que la leçon 34 doit mettre en place `if`et `while`- C'est bien vrai.

Cooper, Harvey et Kennedy ont écrit "Un algorithme de domination simple et rapide" (2001). Ils soutiennent que la formulation du flux de données, exécutée en ordre postérieur inverse avec une représentation compacte, bat Lengauer-Tarjan (1979) sur les graphiques sous environ mille nœuds. Leur article dit que son but est "de ne pas présenter un nouvel algorithme". Il montre que les algorithmes simples peuvent gagner en pratique et sont de bons outils d'enseignement. Une fonction CFG n'a presque jamais mille nœuds.

```python
def intersect(first: int, second: int) -> int:
    while first != second:
        while number[first] > number[second]:
            first = idom[first]
        while number[second] > number[first]:
            second = idom[second]
    return first

changed = True
while changed:
    changed = False
    for start in order[1:]:
        processed = [p for p in self.predecessors(start) if p in idom]
        candidate = processed[0]
        for other in processed[1:]:
            candidate = intersect(other, candidate)
        if idom.get(start) != candidate:
            idom[start] = candidate
            changed = True
```

`number`détient la position post-ordre inverse de chaque bloc, avec l'entrée à 0. `intersect`Le papier écrit la comparaison avec des nombres post-ordre et`<`Ce code utilise des numéros de post-ordre inversés, donc la comparaison est `>`Tout le reste suit le pseudocode du journal.

Pour `total`à`-O1`Le résultat est `idom(0x44) = 0x40`- Je suis là .`idom(0x50) = 0x44`- Je suis là .`idom(0x5b) = 0x50`, et `idom(0x5c) = 0x40`- Le bord .`0x50 -> 0x50`est un côté arrière parce que sa cible domine sa source, et un côté arrière marque une boucle (Allen 1970). `back_edges()`Le test.`test_dominators_on_a_hand_built_diamond_with_a_loop`exécute le même code sur une séquence de 16 octets avec un point de jonction, donc `intersect`Il voit deux prédécesseurs.

## La table de saut en classe à -O0

À `-O0`Clang compile le `switch`dans `grade`à l'idiome de la table de saut indépendante de la position:

```text
      5d: 83 e8 04                     	subl	$0x4, %eax
      60: 77 38                        	ja	0x9a <grade+0x6a>
      62: 48 8b 45 f0                  	movq	-0x10(%rbp), %rax
      66: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		0000000000000069:  R_X86_64_PC32	.rodata-0x4
      6d: 48 63 04 81                  	movslq	(%rcx,%rax,4), %rax
      71: 48 01 c8                     	addq	%rcx, %rax
      74: ff e0                        	jmpq	*%rax
```

Lisez-le comme quatre faits.`subl $0x4`suivie de `ja`est une vérification des limites: les indices supérieurs à 4 vont au cas par défaut à `0x9a`La table comporte donc cinq entrées.`leaq`avec son déplacement met l' adresse du tableau `%rcx`, et la section précédente a montré que l' adresse est `.rodata+0x0`- Le .`movslq`charge une entrée signée de 32 bits étalée par 4.`addq`ajoute la base de la table, de sorte que chaque entrée est la distance entre la table et son bloc de case.

Dans le fichier objet, chaque octet de table est zéro.`.rela.rodata`- Le numéro de la liste:

```text
RELOCATION RECORDS FOR [.rodata]:
OFFSET           TYPE                     VALUE
0000000000000000 R_X86_64_PC32            .text+0x91
0000000000000004 R_X86_64_PC32            .text+0x8c
0000000000000008 R_X86_64_PC32            .text+0x87
000000000000000c R_X86_64_PC32            .text+0x82
0000000000000010 R_X86_64_PC32            .text+0x86
```

Appliquer `S + A - P`à l'entrée `k`à l' équilibre `4k`La valeur stockée est `.text + A - (.rodata + 4k)`, et le code ajoute `.rodata + 0`Il est donc la cible.`.text + A - 4k`. Entrée 0 est `0x91 - 0 = 0x91`, l' entrée 1 est `0x8c - 4 = 0x88`, et l' entrée 2 est `0x87 - 8 = 0x7f`Les entrées 3 et 4 sont `0x82 - 12 = 0x76`et `0x86 - 16 = 0x76`- cas 6, 7, 8, 9 et 10 de la`score / 10`Allez aux blocs qui stockent 1, 2, 3, 4 et 4. Cela correspond à la ligne source C pour la ligne, y compris la case partagée pour 9 et 10.

`recover_table()`Il fait cette arithmétique.`jump_table_resolver()`correspond à l'idiome à l' arrière de la `jmpq`Il trouve le`addq`qui a fait le registre, le `movslq`qui l'a chargé, le `leaq`qui a fait la base et la limite `ja`. La descente pousse ensuite les quatre cibles distinctes et se poursuit. Un exécutable lié a des valeurs finales dans la table et aucune relocalisation. La même fonction lit ensuite les entrées signées à 32 bits et ajoute l'adresse de la table, et `evidence`les données relatives aux objectifs.

Les chiffres pour `grade`à`-O0`, à partir de `python3 code/main.py cfg code/fixtures/ledger-x86_64-O0.o grade`- Le numéro de la liste:

| Method | Instructions | Blocks | Edges | Bytes | What is wrong |
|---|---|---|---|---|---|
| linear sweep | 34 | 9 | 7 | 118 code + 10 padding | 5 unreachable blocks, no edge into the four case blocks, padding decoded as `nopw` |
| recursive descent | 25 | 3 | 2 | 82 code | 36 code bytes never reached, `jmpq *%rax` at `0x74` unresolved |
| descent with the table | 33 | 8 | 11 | 118 code | nothing |

Avec la table, l'arbre dominateur est terminé.`idom`de chaque bloc de cas est `0x62`, le bloc avec le saut.`idom`de `0xa1`est l' entrée `0x30`, parce que la voie par défaut de `0x9a`atteint `0xa1`sans passer .`0x62`- Je suis désolé .

## Ce que l'optimisateur a fait pour évaluer à -O1

Vous verrez .`grade`Les octets disent quelque chose de plus précis.`-O1`et `-O2`Il n' y a pas de saut indirect dans `grade`à tout prix:

```text
      2c: 83 f9 04                     	cmpl	$0x4, %ecx
      2f: 77 0c                        	ja	0x3d <grade+0x2d>
      31: 89 c8                        	movl	%ecx, %eax
      33: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		0000000000000036:  R_X86_64_PC32	.rodata+0x10
      3a: 8b 04 81                     	movl	(%rcx,%rax,4), %eax
      3d: c3                           	retq
```

L' additif de relocalisation `0x10`Plus le 4 habituel met la table à `.rodata+0x14`La charge est simple .`movl`, et les octets sont là `01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 04 00 00 00`. Le compilateur a remplacé les cinq blocs de cas par une table des cinq valeurs de retour. Il y a une relocalisation, une table et une vérification des limites. Le flux de contrôle est de trois blocs et trois bords avec une branche conditionnelle. `find_value_tables()`rapporte que `value-table`avec des entrées `[1, 2, 3, 4, 4]`et `evidence: "bytes"`- Je suis désolé .

`grade_label`à`-O2`Il est plus important que tout.`labels`array de cinq pointeurs de chaîne est devenu une table de cinq 32 bits de compensation à `.rodata+0x0`Chaque entrée porte un`R_X86_64_PC32`la réinstallation contre la section de la chaîne, et la fonction ajoute l'adresse de la table à l'entrée qu'elle charge. `recover_table()`Il lit ces cinq délocalisations et les retours de compensation `0, 2, 4, 6, 8`dans `.rodata.str1.1`, qui tiennent`F`- Je suis là .`D`- Je suis là .`C`- Je suis là .`B`, et `A`. La même arithmétique qui récupère une table de saut récupère une table de données relatives, car les deux sont `S + A - P`à un décalage connu.

La leçon de ces deux constructions est de nommer ce que les octets montrent. Une table de saut change le graphique de contrôle-flux. Une table de valeurs change une charge. Appeler les deux "le commutateur" cache la différence que la leçon 34 doit reconstruire.

## Que vous donnent réa et Ghidra ici

ré 6.5.0 avec le fournisseur Ghidra 12.1.4 renvoie le texte de liste d'une fonction à partir de `procedure_assembly`comme une seule corde. `inspect_native_instruction`renvoie les octets, la longueur, le texte décodeur, les jetons d'opérand, les références typées et le flux d'une instruction, le retour étant classé à partir du code p. `xrefs`renvoie les adresses de source uniques triées de Ghidra `ReferenceManager.getReferencesTo`- Je suis là .`analyze_function`porte le CFG et `body`de la fonction. `procedure_info`donne à l'organisme des plages inclusives et met en garde que "les lacunes et les coulisses partagées ne doivent pas être traitées comme des octets détenus".

Chacun de ces faits est la descente récursive de Ghidra plus ses analystes, sérialisés.`instruction_flow_evidence`- Je suis là .`function_references`, et `jump_table_evidence`Un résultat nomme donc les règles qui l'ont produit. Ce qu'il ne peut pas faire, c'est montrer son travail. Quand Ghidra ne trouve pas la table, le flux est "inconnu", et les blocs de cas sont manquants pour les appelants, les appelés et le CFG.

Vous avez le travail.`python3 code/main.py cfg`Si vous avez une dérivée indépendante, vous avez une adresse spécifique à regarder, qui est ce que la leçon 35 enregistrera comme inconnu.

Capstone 6.0.0-Alpha10 (2026-07-21) fixe hors limites lit et écrit sous CVE-2026-55893 et CVE-2026-55894, et 5.0.9 (2026-05-28) backporté CVE fixes. Un décodeur lit des octets qu'il n'a pas écrit. La limite de 15 octets, la vérification de troncation en `_Cursor.byte()`, et le refus de deviner un code opcode inconnu protège `main.py`de l'entrée pour laquelle il n'a pas été construit.

## Faites-le

Exécutez la démo:

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/main.py
```

La sortie est un rapport JSON. Pour chaque objet fixe, il donne l'accord oracle. Pour chaque fonction, il donne des instructions, des blocs et des comptes de bord selon les trois méthodes. Il énumère également des blocs inaccessibles, des sauts indirects non résolus, des bordures arrière, des dominateurs immédiats et toutes les tables.

Forme attendue:

```json
{
  "fixtures": {
    "ledger-x86_64-O0.o": {
      "oracle": "86/86 instructions agree with llvm-objdump",
      "functions": {
        "grade": {
          "range": "0x30-0xa6",
          "paddingBytes": 10,
          "linearSweep": {"instructions": 34, "blocks": 9, "edges": 7, "unreachableBlocks": ["0x76", "0x7f", "0x88", "0x91", "0xa6"], "...": "..."},
          "recursiveDescent": {"instructions": 25, "blocks": 3, "edges": 2, "unresolvedIndirectJumps": ["0x74"], "...": "..."},
          "recursiveDescentWithTables": {"instructions": 33, "blocks": 8, "edges": 11, "...": "..."},
          "tables": [{"kind": "jump-table", "table": ".rodata+0x0", "count": 5, "entries": ["0x91", "0x88", "0x7f", "0x76", "0x76"], "evidence": "relocation", "...": "..."}]
        }
      }
    }
  },
  "sweepTrap": {"...": "..."}
}
```

Lire `code/main.py`dans cet ordre:

1. `ElfObject`lisent les en-têtes de section, les symboles et `.rela.*`les entrées, et `symbolize()`Nommer une adresse pour la route `llvm-objdump`- Il le fait.
2. `Decoder.decode()`lisent les préfixes et REX, puis les envoient sur la carte opcode d'un ou deux octets.
3. `_State.modrm()`lisent le ModRM, le SIB et le déplacement, et enregistrent le déplacement compensatoire du champ.
4. `resolve_reference()`est appliquée `S + A - P`à la relocalisation sur un lieu de déplacement ou sur un terrain immédiat.
5. `linear_sweep()`et `recursive_descent()`Les deux méthodes sont les deux méthodes, chacune avec son échec enregistré.
6. `recover_table()`et `jump_table_resolver()`lire une table par ses déplacements ou ses octets.
7. `ControlFlowGraph`construit des leaders, des blocs et des bords taillés, et `dominators()`est l'algorithme itératif.
8. `analyze_function()`fonctionne chaque méthode sur une seule fonction et `compare_with_oracle()`Vérifie le décodeur.

Il y a deux autres commandes.`python3 code/main.py disasm FILE.o`imprime une liste avec la cible de relocalisation après chaque instruction qui en a une. `python3 code/main.py cfg FILE.o FUNCTION`imprime l'analyse d'une fonction.

## Utilisez-le

Utilisez les trois méthodes dans cet ordre sur un nouvel objet.

1. Faites la comparaison oracle d'abord.`llvm-objdump`sur une instruction simple a un bug, et chaque structure ultérieure l'hérite.
2. Exécutez la descente récursive de chaque symbole de fonction et lisez `unresolved`Chaque entrée est un saut ou appel indirect dont vous n'avez pas de cibles.
3. Pour chaque saut non résolu, recherchez l'idiome de table et le déplacement qui le pointe vers. Si l'idiome ne correspond pas, lisez les octets avant le saut et écrivez ce que vous voyez.
4. Comparer le balayage et le descendant octet par octet. Les octets que seuls les balayage revendiquent sont des rembourrages ou des données. Les octets que ni revendiquent sont inaccessibles ou appartiennent à une fonction que vous n'avez pas trouvé.
5. Construisez le CFG à partir de la descente avec des tables récupérées et lisez les bords arrière.`while`ou `for`- Je suis désolé .

Gardez les fichiers texte oracle avec les objets. Lorsque vous passez à une nouvelle version de compilateur, recapturez-les, car une liste modifiée est un fichier modifié.

## Laboratoire interactif

### Laboratoire A: décode sur papier, puis vérifie

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow/code
python3 -q
```

Je vais courir .

```python
from main import *
decode(bytes.fromhex("48 8d 0d 00 00 00 00"), 0).text
decode(bytes.fromhex("8d 44 08 fa"), 0).text
decode(bytes.fromhex("66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00"), 0).length
```

Écrivez ce que vous attendez avant chaque appel.`48`Puis essayez de voir la taille de l'opéra.`decode(bytes.fromhex("0f 0a"), 0)`et lire le `DecodeError`- Je suis désolé .

### Laboratoire B: le déménagement sur un lit

```python
elf = load_fixture("ledger-x86_64-O0.o")
code = elf.section(".text").data
lea = decode(code, 0x66)
lea.displacement_offset
resolve_reference(elf, lea)
```

La référence dit:`.rodata+0x0`- Changez l' adresse en `0x116`dans `grade_label`et lire la référence dans `.data`- Alors décodez le `callq`à`0x10e`Le champ immédiat est en train de porter`R_X86_64_PLT32 grade-0x4`, et la cible résolue est `0x30`, qui est `grade`- Je suis désolé .

### Laboratoire C: observez les deux méthodes ne s'accordent pas

```python
trap = load_fixture("sweep-trap.o")
text = trap.section(".text").data
[i.text for i in linear_sweep(text, 0, 0x20).instructions]
[i.text for i in recursive_descent(text, [0], 0x20).sorted_instructions()]
```

Comptez les instructions de chaque liste et trouvez celle qui commence par `0x13`Alors changez le dernier octet de la table en `sweep-trap.s`de `0x48`à `0xe8`Le balayage avale maintenant le corps.`addl`et le `retq`dans une `callq`La fonction semble ne pas avoir de retour.

### Laboratoire D: dominateurs à la main

```python
analysis = analyze_function(load_fixture("ledger-x86_64-O1.o"), "total")
analysis.cfg.edges
analysis.cfg.reverse_postorder()
analysis.cfg.dominators()
analysis.cfg.back_edges()
```

Dessinez les cinq blocs sur papier et calculez les dominateurs avant de regarder le résultat.`grade`à`-O0`et expliquer pourquoi .`idom(0xa1)`est `0x30`et non pas`0x62`- Je suis désolé .

## Laboratoire de pratique

Ajouter un site d'appel indirect aux appareils et étendre l'analyse.

Les exigences:

- Écrivez une fonction C qui appelle à travers un pointeur de fonction choisi à partir d'un petit tableau.`-O0`et `-O1`et engager les objets avec leurs `llvm-objdump`Je vous ai envoyé un SMS.
- Faites en sorte que le décodeur soit conforme à l'oracle sur les deux objets.
- Extension `recursive_descent()`Ainsi, un appel indirect enregistre le site d'appel en `calls`avec `None`comme cible et continue après elle.
- Écrivez un résolveur qui lit l'ensemble du pointeur de fonction à travers ses délocalisations et rapporte les appelés possibles.
- Ajoutez le jeu de callee au rapport de fonction comme `indirectCalls`avec le type de preuve pour chaque cible.
- Écrivez un test qui échoue lorsque le résolveur renvoie une cible qui n'est pas un symbole de fonction.

Le succès signifie`python3 code/main.py cfg`nomme chaque appel possible avec le déplacement qui le prouve, et la comparaison oracle continue de passer sur chaque objet.

## Leçon d'artisanat

Cette leçon comprend `outputs/skill-disassembly-and-control-flow.md`. Utilisez-le pour transformer un objet et un nom de fonction en une liste vérifiée, des blocs, des bords taillés, des dominateurs et tous les transferts non résolus. Il nécessite une comparaison oracle, les deux méthodes de démontage, la comptabilité des octets, la preuve de relocalisation pour chaque table récupérée, et une déclaration de ce qui est resté inconnu.

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/main.py
python3 code/tests/test_main.py
```

La vérification doit prouver:

- le décodeur est d' accord avec `llvm-objdump -d`sur les 237 instructions des quatre objets de fixation
- REX.W sélectionne des opérands 64 bits et REX.R et REX.B étendent les champs du registre
- ModRM et SIB rendent l'opérand de mémoire AT&T, y compris l'échelle, l'index et le déplacement signé
- un déplacement relatif au RIP est enregistré avec son champ compensé à l'intérieur de l'instruction
- préfixes hérités répétés et un décode de segment de suppression comme l'oracle
- une obligation `66`Le préfixe sélectionne le code optique SSE et laisse les registres à 32 bits
- Les 8 bits immédiats sont sign-extendus pour l'arithmétique et non signés pour le nombre de coups
- une instruction de plus de 15 octets, une instruction tronquée et une augmentation inconnue du code optique `DecodeError`
- `R_X86_64_PC32`sur une`lea`résolue à la table et `R_X86_64_PLT32`sur une`call`résolve à la callee
- Le balayage décode la table de piège comme code et n'a aucune instruction sur la cible de saut
- La descente ne prétend jamais la table ou le rembourrage et lit `addl $0x1, %eax`à la cible
- `llvm-objdump`fait la même erreur que le balayage sur le piège
- une entrée dans une instruction est enregistrée comme un conflit
- un saut indirect non résolu arrête la descente et est indiqué
- `total`à`-O1`Il a cinq blocs, cinq bords taillés, un côté arrière, et les dominateurs immédiats attendus
- un diamant fabriqué à la main avec une boucle donne aux dominateurs l'algorithme définit
- `total`à`-O2`a deux boucles
- `grade`à`-O0`produit une table de saut à cinq entrées à partir de `.rela.rodata`avec des objectifs `0x91, 0x88, 0x7f, 0x76, 0x76`
- sans la table .`grade`à`-O0`Il a trois blocs et un saut non résolu
- `grade`à`-O1`et `-O2`est une table de valeurs `[1, 2, 3, 4, 4]`sans saut indirect
- `grade_label`à`-O2`lisent une table de chaînes relative à travers cinq délocalisations
- la figure interactive intègre les mêmes données que l'analyse produit

## Mode de défaillance de la production

| Failure | What the weak tool reports | What the analysis must prove |
|---|---|---|
| Data inside `.text` | A plausible listing with every byte decoded | The descent claims no data byte and the jump target starts an instruction |
| Padding after the last return | A `nopw` instruction at the end of the function | The bytes are unreachable from the entry |
| Indirect jump through a table | "jmp rax" and four dead blocks | Table address from the `lea` relocation, entry count from the bounds check, targets from `.rela.rodata` |
| Switch turned into a value table | "a jump table with a relocation" | A `movl` load from `.rodata+0x14`, three blocks, no indirect transfer |
| Call with a zero displacement | `callq 0x113`, the next instruction | `R_X86_64_PLT32 grade-0x4` resolves to `0x30` |
| Decoder guesses an unknown opcode | A mnemonic that looks right | `DecodeError` with the address, and the sweep records the byte |
| Instruction longer than 15 bytes | A very long "instruction" | The cursor raises before reading byte 16 |
| Branch into the middle of an instruction | Two overlapping listings | A conflict record with both addresses |
| Loop reported as straight-line code | No `while` in the pseudocode | A back edge whose target dominates its source |
| Oracle captured from a different compiler | Tests pass on stale text | The fixture README names the exact compiler and command |

## Règlement d'exploitation

Décodez contre un oracle, contrôlez le flux de navigation à partir d'entrées connues et expliquez chaque octet comme revendiqué, rembourrage, données ou inaccessible.

## Pour en savoir plus

- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2, Chapter 2: Instruction Format](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [AMD64 Architecture Programmer's Manual Volume 3, Chapter 1: Instruction Encoding](https://docs.amd.com/v/u/en-US/24594_3.38_APM_Vol3_PUB)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Table 4.9: Relocation Types](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Schwarz, Debray, Andrews, "Disassembly of Executable Code Revisited" (WCRE 2002)](https://www2.cs.arizona.edu/~debray/Publications/disasm.pdf)
- [Andriesse et al., "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse)
- [Allen, "Control Flow Analysis" (1970)](https://doi.org/10.1145/800028.808479)
- [Cooper, Harvey, Kennedy, "A Simple, Fast Dominance Algorithm" (2001)](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf)
- [Lengauer, Tarjan, "A fast algorithm for finding dominators in a flowgraph" (1979)](https://doi.org/10.1145/357062.357071)
- [Capstone 6.0.0-Alpha10 release notes](https://github.com/capstone-engine/capstone/releases/tag/6.0.0-Alpha10)
- réa 6.5.0 source à la balise `3b79af2`que cette leçon décrit: `README.md`- Je suis là .`docs/ghidra-provider.md`- Je suis là .`docs/ghidra-dos.md`- Je suis là .`docs/reverse-engineering-tools.md`- Je suis là .`bridge/ghidra/ReaGhidraBridge.java`- Je suis là .`src/contracts/officialToolContracts.ts`- Je suis là .`src/ghidra/GhidraAnalysisProfile.ts`- Je suis désolé .
