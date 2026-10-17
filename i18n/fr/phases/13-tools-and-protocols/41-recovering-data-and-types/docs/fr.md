# Récupération de données et de types: Structures, Arrays, Objets C++ et DWARF

> Un décompilateur imprime `a->balance`lorsqu' il dispose d' une table de type et `*(int *)(param_1 + 4)`Cette leçon construit la table à partir des accès, lit la table réelle du DWARF 5 et montre où les deux sont d'accord.

**Type:** Build
**Languages:** Python, C, C++, x86-64 assembly
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~140 minutes

## Objectifs d'apprentissage

- Lisez un accès de membre structure en tant que registre de base plus déplacement, et un accès de tableau en tant que étape de base plus temps d'index, à partir des règles d'adressage Intel SDM.
- Appliquer manuellement les règles d' alignement du système V à `struct account`et expliquer chaque octet de remplissage.
- Construisez un passe de récupération de type qui transforme les accès dans une liste en compensations de champs, largesses, types, étapes et tailles, avec la preuve pour chacun.
- Écrivez un lecteur DWARF 5 pour `.debug_info`- Je suis là .`.debug_abbrev`- Je suis là .`.debug_str_offsets`, et `.debug_addr`d'un objet relocalisable et de le vérifier contre `llvm-dwarfdump`- Je suis désolé .
- Comparez la mise en page récupérée avec la mise en page DWARF et nommez chaque différence comme étant une preuve manquante, un accès fusionné ou un type non prouvable.
- Lisez un vtable Itanium C++, son typeinfo et ses noms déformés à partir de délocalisations, et résolvez un appel virtuel à un slot.

## Le problème

Vous avez compilé`account.c`sans`-g`et a demandé à un agent ce que `bump`L'agent a envoyé l'objet à Ghidra 12.1.4 et a récupéré ceci:

```c
void bump(long param_1,int param_2)
{
  *(int *)(param_1 + 4) = param_2 + *(int *)(param_1 + 4);
  return;
}
```

La même source compilée avec `-g`donne ceci du même décompilateur:

```c
int bump(account *a,int by)
{
  a->balance = by + a->balance;
  return a->balance;
}
```

Les instructions sont identiques dans les deux objets. La différence est une table de type. Avec la table, le décompilateur sait que le premier paramètre pointe vers une structure de 32 octets dont le champ à l'offset 4 est un 4 octets `int`nommé `balance`Sans elle , le décompilateur sait seulement que quelque chose de 8 octets de large est arrivé .`%rdi`et que 4 octets à `+4`ont été lues et écrites.

Tout ce qui est dans cette table a une source.`addl 0x4(%rax), %ecx`La largeur 4 vient de la .`l`Le nom vient d'une`DW_TAG_member`entrée dans `.debug_info`Cette leçon comporte deux parties: un pass qui récupère ce que les instructions prouvent et un lecteur qui décode ce que le compilateur a enregistré.

```figure
re-41-struct-recovery
```

## Un accès à un membre est un déplacement

L'adresse effective est définie à l'article 3.7.5 du volume 1 du MDD Intel comme `Offset = Base + (Index * Scale) + Displacement`, avec une échelle de 1, 2, 4 ou 8 et un déplacement de 8, 16 ou 32 bits.`RIP + Displacement`. Un accès à un membre de structure utilise le premier formulaire sans index: le registre de base contient l'adresse de l'objet et le déplacement est le décalage de membre.

Prenez .`bump`à`-O0`- La liste de `llvm-objdump -d -r`Il est écrit:

```text
       4: 48 89 7d f8                  	movq	%rdi, -0x8(%rbp)
       e: 48 8b 45 f8                  	movq	-0x8(%rbp), %rax
      12: 03 48 04                     	addl	0x4(%rax), %ecx
      15: 89 48 04                     	movl	%ecx, 0x4(%rax)
```

Décodez`03 48 04`avec les règles de la leçon 33. `03`est `ADD r32, r/m32`- ModRM `48`est `01 001 000`Le numéro de la liste:`mod = 01`désigne un opérand de mémoire à déplacement de 8 bits, `reg = 001`est `%ecx`- Je suis là .`r/m = 000`est `%rax`Le byte de déplacement est `04`L' instruction est de 4 octets à `%rax + 4`et les ajoute à `%ecx`- Le .`l`le suffixe qui `llvm-objdump`Les empreintes sont de la taille de l'opéra, donc le champ est de 4 octets de large.`%rax`est venu de la fente de cadre à `-0x8(%rbp)`, et cette fente a été reçue .`%rdi`, le premier argument entier de la convention d'appel du système V. Trois faits suivent: le premier paramètre est un pointeur, l'objet qu'il pointe a un champ de 4 bytes à l'offset 4, et `bump`Il lit et écrit ce champ.

Rien dans ces octets ne dit`int`- Je suis là .`balance`ou `struct account`Les 28 octets restants de la structure sont invisibles.`bump`- Je suis désolé .

## Les règles de mise en page du système V

Le x86-64 psABI, section 3.1.2, donne trois règles pour les agrégats. " Les structures et les syndicats prennent l'alignement de leur composant le plus strictement aligné. Chaque membre est assigné à l'offset le plus bas disponible avec l'alignement approprié. La taille d'un objet est toujours un multiple de l'alignement de l'objet. " La figure 3.1 du même document donne les tailles et alignements scalaires: `char`1 et 1, `short`2 et 2, `int`4 et 4, `double`8 et 8, n'importe quel pointer 8 et 8.

Appliquer les règles à la structure en`account.c`- Le numéro de la liste:

```c
struct account {
    char kind;
    int balance;
    short code;
    double rate;
    const char *label;
};
```

| Member | Size | Alignment | Lowest offset that fits | Padding before |
|---|---|---|---|---|
| `kind` | 1 | 1 | 0 | 0 |
| `balance` | 4 | 4 | 4 | 3 (bytes 1 to 3) |
| `code` | 2 | 2 | 8 | 0 |
| `rate` | 8 | 8 | 16 | 6 (bytes 10 to 15) |
| `label` | 8 | 8 | 24 | 0 |

Le dernier membre se termine à 32. L'alignement le plus strict est 8, et 32 est un multiple de 8, de sorte que la taille est 32 sans rembourrage de queue. Neuf octets des 32 sont rembourrés, et le psABI dit "Le contenu de tout rembourrage est indéfini".

L'ordre des membres détermine la taille.`double, pointer, int, short, char`Les remplissages compensent 0, 8, 16, 20, 22 et se terminent à 23, ce qui fait 24 rondes. `sysv_layout()`dans `code/main.py`Il est possible de trouver des solutions qui permettent de mettre en œuvre les trois règles et de produire les deux résultats.`ctypes`Le module applique les mêmes règles par l'intermédiaire de la convention du compilateur hôte, et `ctypes.sizeof`- Je suis là .`ctypes.alignment`, et le `.offset`de chaque champ donne 32, 8 et `0, 4, 8, 16, 24`Le programme lui-même imprime la troisième copie lorsque vous la construisez et l'exécutez sur cette machine:

```text
sizeof 32 align 8
kind 0 balance 4 code 8 rate 16 label 24
sum 315 tag -3 projected 318.75 label_length 7
```

Cette construction native est arm64. L'Arm AAPCS64 énonce les mêmes règles agrégées dans sa section 5.10.1, et le document arm64 d'Apple conserve ces cinq tailles scalaires, de sorte que les chiffres concordent entre les deux cibles.`-target x86_64-unknown-linux-gnu`pour que le lecteur ELF de la leçon 33 et le lecteur DWARF ci-dessous travaillent sur eux.

## Lisez le reste de la structure à la main

`bump`Les quatre autres fonctions sont en `account.c`Les adresses ci-dessous sont de `account-x86_64-O0-g.objdump.txt`- Je suis désolé .

`sum_balances`indice une gamme de structures:

```text
      51: 48 8b 45 f8                  	movq	-0x8(%rbp), %rax
      55: 48 63 4d ec                  	movslq	-0x14(%rbp), %rcx
      59: 48 c1 e1 05                  	shlq	$0x5, %rcx
      5d: 48 01 c8                     	addq	%rcx, %rax
      60: 8b 40 04                     	movl	0x4(%rax), %eax
```

`%rax`est `list`- Je suis là .`%rcx`est `i`d'une portée de 64 bits, et `shlq $0x5`multiplié par 32.`addq`- Je suis là .`%rax`est `list + 32 * i`, et la charge est de 4 octets à l' offset 4 de l' élément `i`. Le nombre de déplacements est le seul endroit dans cet objet où la taille de la structure apparaît comme un nombre.`0x49`à la `jmp 0x49`à`0x72`- Je suis désolé .

`tag`Il y a deux champs étroits:

```text
      8c: 0f be 00                     	movsbl	(%rax), %eax
      98: 0f bf 48 08                  	movswl	0x8(%rax), %ecx
```

`movsbl`charge un octet à l'offset 0 et le sign-extension, de sorte que le champ à 0 est un entier de 1 octet signé. `movswl`charge deux octets à l'offset 8 avec extension de signe, donc le champ à 8 est un entier signé de 2 octets.`-O1`Le premier accès devient `cmpb $0x6c, (%rdi)`, qui dit toujours 1 byte à l'offset 0 et ne dit rien sur le signe.

`projected`Il y a un flot:

```text
      cc: f2 0f 2a 40 04               	cvtsi2sdl	0x4(%rax), %xmm0
      d5: f2 0f 59 40 10               	mulsd	0x10(%rax), %xmm0
```

`cvtsi2sdl`convertisse un nombre entier signé de 32 bits de la mémoire en un double, ce qui confirme que l'offset 4 contient un nombre entier signé de 4 bits. `mulsd`multiplié par le double de la mémoire, donc le compensateur 16 a une valeur de point flottant de 8 bytes.

`label_length`lis un pointeur et le suit ensuite:

```text
      f3: 48 8b 40 18                  	movq	0x18(%rax), %rax
      f7: 48 63 4d f4                  	movslq	-0xc(%rbp), %rcx
      fb: 80 3c 08 00                  	cmpb	$0x0, (%rax,%rcx)
```

Le `movq`Il est 8 octets à l'offset 24.`cmpb`La valeur utilisée comme adresse est un pointeur.`cmpb`lisent 1 octet à l' index `%rcx`de l'intérieur, et à`-O1`le pointeur est avancé avec `leaq 0x1(%rcx), %rcx`dans la boucle, donc les éléments pointés vers sont à 1 octet de l'autre.

Si on met les cinq fonctions ensemble, le structure est `{ i8 at 0, i32 at 4, i16 at 8, f64 at 16, pointer at 24 }`Les écarts de 1 à 3 et de 10 à 15 sont exactement le rembourrage prévu par les règles de psABI.`const char *`, le type déclaré du octet signé à 0, qui est orthographié par la source `char`, et que les cinq fonctions reçoivent le même type.`main`, qui passe un objet de cadre à tous les cinq.`-O0`les sites d'appels rendent visible: `leaq -0x70(%rbp), %rdi`suivie de `addq $0x20, %rdi`avant `callq bump`, et `addq $0x40, %rdi`avant `callq tag`Les compensations sont 0, `0x20`, et `0x40`sont trois éléments d'un même tableau, et le `memcpy`de `0x60`les octets qui le remplissent sont 96 octets, ce qui est trois éléments de 32.

## Le passe

`trace_function()`dans `code/main.py`Il est utilisé pour traverser les instructions d'une fonction dans l'ordre d'adresse et conserve un petit état abstrait.`-0x8(%rbp)`Chaque détenteur a une des trois valeurs.`Pointer`est une racine plus un décalage constant plus un ensemble de pas, où un pas enregistre qu'un multiple inconnu de ces nombreux octets a été ajouté.`Index`est un nombre entier d'une valeur inconnue qui est un multiple d'une échelle.

L' État commence par la tenue des six registres de l' argumentation du système V `Pointer("param0", 0)`à travers `Pointer("param5", 0)`Alors chaque instruction s' applique à une règle:

- Une .`movq`Une écriture à 32 bits ou plus étroite dans un registre en fait une copie.`Index`avec l'échelle 1.
- Une .`leaq`avec une base de pointeur produit un pointeur avec le déplacement ajouté, et avec un registre d'index il ajoute `scale * index_scale`- Je suis en train de faire des progrès.`leaq`dont la base est `%rbp`ou `%rsp`nomme un objet de cadre, qui est comment `main`passe `rows`- Je suis désolé .
- `shlq $n`sur une`Index`multiplié par `2^n`, et `imulq $k`multiplié par `k`- Je suis désolé .
- `addq`d'un pointeur et d'un index conserve le pointeur et ajoute l'échelle de l'indice aux pas. `addq $k`à un pointeur, le décalage est déplacé et est également enregistré comme un élément de ce registre.
- Un appel rend chaque registre enregistré par l'appelant inconnu, et les registres d'arguments contenant des pointeurs sont enregistrés comme un site d'appel.
- Tout opérand de mémoire dont le registre de base contient un pointeur devient un `Access`avec la racine et le décalage du pointeur, la largeur et la sorte du mnemonique, et un drapeau de magasin.

La largeur et la gentillesse viennent du seul mnemonique.`access_width()`carte de la `b`- Je suis là .`w`- Je suis là .`l`, et `q`les suffixes pour 1, 2, 4 et 8 octets, `movsbl`et `movswl`à des charges étroites signées, `movzbl`à l'absence de signature, le `sd`et `ss`instructions de la SSE scalaire pour les flottes de 8 octets et de 4 octets, `cvtsi2sdl`à une source entière de 4 octets signée, et à la source de 16 octets `movaps`famille à un vecteur. `leaq`et le `nop`Les formulaires ne sont pas des accessoires.

```python
if memory.base in FRAME_REGISTERS:
    return Pointer(f"frame({memory.base}{format_signed(memory.displacement) if memory.displacement else ''})", 0)
base = env.get(memory.base)
if not isinstance(base, Pointer):
    return Unknown
strides = set(base.strides)
if memory.index is not None:
    index_value = env.get(memory.index)
    strides.add((index_value.scale if isinstance(index_value, Index) else 1) * memory.scale)
return Pointer(base.root, base.offset + memory.displacement, frozenset(strides), base.origin)
```

Une charge de 8 octets à travers un pointeur produit un nouveau pointeur dont les noms racinaires du champ d'où il vient, tels que `param0->0x18`. Lorsque cette valeur est utilisée plus tard comme base, ou avancée avec `leaq`C'est la règle qui transforme le décalage 24 de "8 octets" en "pointer".

### Quatre types de preuves de démarches

`infer_layout()`fusionne les accès de chaque fonction témoin et recueille le pas de quatre sources, chacune avec l'instruction qui l'a produit.

| Kind | Where it comes from | Build that shows it |
|---|---|---|
| `scale` | an `Index` with scale above 1 reached a pointer, from `shlq $0x5` or `imulq` | `-O0`, `shlq $0x5, %rcx` at `0x59` |
| `loop-step` | a 64-bit increment of the base or index register inside a loop that also contains the access | `-O1`, `addq $0x20, %rdx` at `0x24` |
| `offset-delta` | same-width accesses through one base register inside one loop, at offsets a constant apart | `-O2`, eight loads 32 apart |
| `call-site` | the offsets at which one frame object is passed to the witness functions | `-O0`, `main` passes `rows`, `rows+0x20`, `rows+0x40` |

Une boucle est une boucle vers l' arrière .`jmp`La valeur de l'augmentation est calculée en fonction de la valeur de l'augmentation ou de la branche conditionnelle, et un increment ne compte que lorsque la plage de boucle contient à la fois l'augmentation et l'accès.`addl $0x1, %eax`qui avance `i`à`-O0`, parce que c' est un add-on de 32 bits et parce que l' accès ne passe pas par `%eax`La taille est la taille lorsque la mesure est au moins aussi grande que la fin du dernier champ.

### La boucle vectorisée à -O2

À `-O2`Clang vectories `sum_balances`Le corps de la boucle de`0x50`à `0xaa`Il y a huit éléments par passe:

```text
      3b: 48 8d 87 e4 00 00 00         	leaq	0xe4(%rdi), %rax
      50: 66 0f 6e 50 80               	movd	-0x80(%rax), %xmm2
      55: 66 0f 6e 98 60 ff ff ff      	movd	-0xa0(%rax), %xmm3
      61: 66 0f 6e 90 40 ff ff ff      	movd	-0xc0(%rax), %xmm2
      69: 66 0f 6e a0 20 ff ff ff      	movd	-0xe0(%rax), %xmm4
      7d: 66 0f 6e 10                  	movd	(%rax), %xmm2
      81: 66 0f 6e 58 e0               	movd	-0x20(%rax), %xmm3
      8a: 66 0f 6e 50 c0               	movd	-0x40(%rax), %xmm2
      8f: 66 0f 6e 60 a0               	movd	-0x60(%rax), %xmm4
      a0: 48 05 00 01 00 00            	addq	$0x100, %rax
      aa: 75 a4                        	jne	0x50 <sum_balances+0x40>
```

`%rax`commence à`list + 0xe4`Les huit déplacements placent les charges à des compensations.`0x04, 0x24, 0x44, 0x64, 0x84, 0xa4, 0xc4, 0xe4`de `list`, qui sont `4, 36, 68, 100, 132, 164, 196, 228`- L' augmentation est`0x100`Un passe qui enregistre les compensations en les voyant rapporte huit champs et un structure d'au moins 232 octets.

`fold_unrolled()`Dans une boucle, prenez les mêmes accès de largeur à travers un registre de base.`d`Si la boucle est de l'étape`k * d`et il y en a exactement.`k`Les compensations distinctes, la boucle a été déroulée `k`temps, et chaque accès est le même élément d'accès à l'offset `(offset - lowest) mod d`- Je vous en prie .`256 = 8 * 32`avec huit décalages, donc toutes les huit charges se plient sur le décalage 4, et les données de décalage enregistrent`offset-delta 32`- Je suis là .`unrolled 32`, et `loop-step 256`La boucle de résidu scalaire à `0xe0`ajoute `scale 32`de `shlq $0x5, %rdx`et `loop-step 32`de `addq $0x20, %rdx`Le plus grand divisor commun des cinq valeurs est 32.

La figure a un bouton qui débranche le pliage.`-O2`l'état est ce que voit un lecteur qui fait confiance au déplacement sans la boucle.

## DWARF 5 Des bytes

Le `-g`construire `account.c`contient huit sections de débogage. `.debug_info`est de 493 octets, `.debug_abbrev`179, `.debug_str`- Je suis là.`.debug_str_offsets`124, `.debug_addr`112, `.debug_line`312, `.debug_line_str`12, et `.debug_frame`264. L'introduction des listes de normes DWARF 5 `.debug_str_offsets`- Je suis là .`.debug_addr`, et `.debug_line_str`Comme nouveau dans la version 5. Le lecteur dans `code/main.py`Il faut les cinq premiers.

### L'en-tête de l'unité

La section 7.5.1.1 de la norme donne l'en-tête complet de l'unité de compilation: `unit_length`- Je suis là .`version`- Je suis là .`unit_type`- Je suis là .`address_size`, et `debug_abbrev_offset`Les 12 premiers octets de `.debug_info`sont:

```text
e9 01 00 00   unit_length = 0x1e9 = 489
05 00         version = 5
01            unit_type = DW_UT_compile
08            address_size = 8
00 00 00 00   debug_abbrev_offset = 0
```

Le `unit_type`la longueur compte les octets après elle-même, de sorte que l'unité suivante commencerait à `4 + 489 = 493`, la fin de la section. `llvm-dwarfdump --debug-info`imprime les mêmes cinq valeurs sur son `Compile Unit:`la ligne, et les premiers tests analysent cette ligne et comparent.

### Abréviations

La section 2.1 indique que chaque entrée d'information de débogage "se compose d'une balise d'identification et d'une série d'attributs".`.debug_abbrev`(section 7.5.3) qui donne le nom de la balise, indique si les enfants suivent, et énumère chaque attribut avec sa forme. Cet objet a 14 déclarations.

```text
[12] DW_TAG_structure_type  DW_CHILDREN_yes
     DW_AT_name              DW_FORM_strx1
     DW_AT_byte_size         DW_FORM_data1
     DW_AT_decl_file         DW_FORM_data1
     DW_AT_decl_line         DW_FORM_data1
[13] DW_TAG_member           DW_CHILDREN_no
     DW_AT_name              DW_FORM_strx1
     DW_AT_type              DW_FORM_ref4
     DW_AT_decl_file         DW_FORM_data1
     DW_AT_decl_line         DW_FORM_data1
     DW_AT_data_member_location  DW_FORM_data1
```

`parse_abbreviations()`Il est écrit par code ULEB128, tag ULEB128, un enfant octet, puis des paires d'attributs et de formes jusqu'à un `0, 0`La section 7.5.6 donne les codes de formulaire: `data1`est un octet, `ref4`est un décalage de 4 bytes par rapport à l'unité, `strx1`est un indice de 1 octet dans la table des compensations de chaîne, `sec_offset`et `data4`sont 4 octets, `addrx`est un indice ULEB128 dans le tableau d'adresses, `exprloc`est une longueur ULEB128 et que de nombreux octets, et `flag_present`n'a pas de bytes du tout.

### Décoder le membre de la balance à la main

L' entrée de structure pour `account`commence à l' offset `0x195`avec les octets `0c 15 20 00 03`Le code abréviation est 12, la structure.`DW_AT_name`est l' indice de chaîne `0x15`- Je suis là .`DW_AT_byte_size`est `0x20`- 32 ?`DW_AT_decl_file`est 0, et `DW_AT_decl_line`est 3, qui est la ligne de `struct account {`dans la source.

La deuxième entrée de membres commence à `0x1a3`avec `0d 10 88 01 00 00 00 05 04`- Le numéro de la liste:

1. `0d`est l'abréviation 13, un membre.
2. `10`est `DW_AT_name`comme `strx1`: index de chaîne 16.
3. `88 01 00 00`est `DW_AT_type`comme `ref4`: compensation `0x188`à partir du départ de l'unité, qui est le `DW_TAG_base_type`entrée nommée `int`- Je suis désolé .
4. `00`est `DW_AT_decl_file`et `05`est `DW_AT_decl_line`: ligne 5 de `account.c`- Je suis désolé .
5. `04`est `DW_AT_data_member_location`: compensation 4.

La section 5.7.6 indique qu'une entrée membre "peut avoir soit un attribut DW_AT_data_member_location soit un attribut DW_AT_data_bit_offset" et qu'une constante entière "est le décalage en octets à partir du début de l'entité contenant". Ce octet est le même nombre que le déplacement en `addl 0x4(%rax), %ecx`, et `compare_layouts()`est la fonction qui met les deux côtés.

### Les chaînes et les adresses passent par deux tables

L' index de chaîne 16 n' est pas un décalage vers `.debug_str`Le lecteur prend`DW_AT_str_offsets_base`de l'entrée unitaire, qui est 8, et lit l'entrée de 4 bytes à `8 + 4 * 16`dans `.debug_str_offsets`Cette entrée contient le décompte de`"balance"`dans `.debug_str`- La même indirection s'applique aux adresses.`sum_balances`commence par les octets `07 07`: abréviation 7, sous-programme, puis `DW_AT_low_pc`comme `addrx`Index 7. L'entrée de 8 octets à `8 + 8 * 7`dans `.debug_addr`est `0x30`, l' adresse de la fonction dans `.text`Les six premières entrées de ce tableau sont les adresses des six lignes littérales.

Dans un objet relocalisable, aucune table ne contient ses valeurs finales.`.debug_str_offsets`est zéro dans le dossier, et `.rela.debug_str_offsets`détient 29 `R_X86_64_32`Les délocalisations contre `.debug_str`avec le réel compensation dans l'addendum. `.debug_addr`Il a 13 `R_X86_64_64`Les délocalisations contre `.text`et `.rodata.str1.1`- Je suis là .`.debug_info`a quatre délocalisations propres: l'abréviation compensation, `DW_AT_str_offsets_base`comme `.debug_str_offsets+0x8`- Je suis là .`DW_AT_stmt_list`, et `DW_AT_addr_base`comme `.debug_addr+0x8`- Je suis là .`ElfObject.relocated()`est appliquée `S + A`Pour les types 32 et 64 bits avant que le lecteur analyse une section. Un symbole de section a une valeur 0 dans un fichier objet, de sorte que le champ reçoit l'addend. `llvm-dwarfdump`Il est donc possible de modifier le code de référence pour les données de référence.

### Types, fonctions et lieux

`DwarfInfo.type_name()`suivant `DW_AT_type`Les références.`label`points de membre à un `DW_TAG_pointer_type`, qui indique une`DW_TAG_const_type`, qui indique que le `char`type de base, le nom se compose donc de `const char *`- Je suis là .`type_size()`Retour `DW_AT_byte_size`pour les types de base et les structures, 8 pour un pointeur, et compter les temps de taille des éléments pour un tableau. `type_kind()`réduit le`DW_AT_encoding`de type de base à `int`avec un signe, ou `float`, et un type de pointeur à `pointer`, qui est le vocabulaire utilisé par la comparaison.

`DwarfInfo.functions()`Il est 6 .`DW_TAG_subprogram`Les entrées. `bump`Il s' en va de `0x0`à `0x21`, qui est `DW_AT_low_pc`plus le `DW_AT_high_pc`Sa base de cadre est `DW_OP_reg6 RBP`, et son paramètre `a`a l' expression de localisation `91 78`Le numéro de la liste:`DW_OP_fbreg`avec la valeur SLEB128 `0x78`, qui est -8. C'est la fente du cadre `-0x8(%rbp)`dans lequel la liste est stockée `%rdi`à l' adresse `0x4`Les informations de débogage et les instructions décrivent la même fente de deux côtés.

## Comparons les deux conceptions

`compare_layouts()`Il est possible de trouver un champ déduit à la même distance.`no-evidence`Un champ de taille différente est`size-mismatch`Un membre de pointeur dont le champ n' a jamais été dérivé est`pointer-unproven`. Un type qui ne partage pas l'avis, tel qu'un flot où les instructions utilisent l'arithmétique des nombres entiers, est `kind-mismatch`Les champs inférieurs qui commencent à l'intérieur d' un membre ou à l' intérieur du rembourrage sont `unexpected`La taille compare le pas avec `DW_AT_byte_size`- Je suis désolé .

Les chiffres des trois constructions, de `python3 code/main.py`- Le numéro de la liste:

| Build | Accesses through the first parameter | Fields | Stride evidence | Members matched | Size |
|---|---|---|---|---|---|
| `-O0` | 11 | 5 | `scale 32`, `call-site 32`, `call-site 64` | 5 of 5 | 32, match |
| `-O1` | 9 | 5 | `loop-step 32` | 5 of 5 | 32, match |
| `-O2` | 17 | 5 after folding, 12 before | `offset-delta 32`, `unrolled 32`, `loop-step 256`, `scale 32`, `loop-step 32` | 5 of 5 | 32, match |

Les espaces déduits sont des octets de 1 à 3 et de 10 à 15 dans chaque construction, et ce sont exactement les octets de remplissage de la mise en page DWARF. Chaque champ déduit se trouve à un multiple de sa propre taille, ce qui est ce que les règles psABI exigent, et `natural_alignment_ok()`Ça, ça, ça.

Ajoutez les fonctions une à la fois et l'image change.`bump`seule la disposition est un champ à l'offset 4, la taille est inconnue, et la limite inférieure est 8. `sum_balances`ajoute le pas. `tag`ajoute les compensations 0 et 8. `projected`ajoute le flot à 16. `label_length`ajoute le pointeur à 24 et prouve qu'il s'agit d'un pointeur. `main`ajoute les sites d'appels à `-O0`et ne ajoute rien à `-O1`et `-O2`La figure passe par cet ordre pour chaque build.

Trois choses restent hors de portée.`const`Le type pointee de l'appareil est un type de pointee.`label`est seulement "éléments 1 octet de l'autre" de la `leaq 0x1(%rcx), %rcx`dans la boucle de chaîne, ce qui est cohérent avec `char`et avec `unsigned char`et `_Bool`À la .`-O1`Le signe du octet à l'offset 0 est inconnu, car `cmpb $0x6c, (%rdi)`Les rapports de passage `int8`là et `i8`à`-O0`Où ?`movsbl`Les faits sont les faits de l'étude 35 enregistrés comme inconnus.

## Objets C++

`shapes.cpp`définit `Shape`avec un destructeur virtuel, un pur virtuel `area()`, et une virtuelle `name()`, et `Rect`, qui dérive de `Shape`et ajoute deux `int`L'ABI Itanium C++ régit comment clang les déploie sur x86-64 Linux.

### Nom déformé

L'article 5.1.2 de l'ABI déclare qu'"un nom est déformé par le préfixe `_Z`L'objet a 19 symboles de ce type.`_ZNK4Rect4areaEv`Il est écrit:`_Z`Alors ...`N`pour un nom encastré, `K`pour une fonction de membre const, `4Rect`et `4area`comme noms de source à longueur préfixée, `E`pour fermer le nom de l'emplacement, et `v`pour une liste de paramètres vide. `llvm-cxxfilt`des empreintes`Rect::area() const`- Je suis là .`_Z8describeRK5Shapei`est `describe(Shape const&, int)`Le numéro de la liste:`R`est une référence, `K`est const, `5Shape`est la classe, `i`est `int`Les noms spéciaux de la section 5.1.4 sont `_ZTV`pour une table de travail, `_ZTI`pour typeinfo, et `_ZTS`Pour la chaîne de noms typeinfo.`C1`- Je suis là .`C2`- Je suis là .`D0`- Je suis là .`D1`, et `D2`Je suis un homme .`_ZN4RectD0Ev`et `_ZN4RectD2Ev`à la fois imprimé comme `Rect::~Rect()`: le destructeur de suppression et le destructeur d'objet de base partagent un nom de source.

`Demangler`dans `code/main.py`couvre ce sous-ensemble plus les codes de type de construction de la section 5.1.5.2 et le `S_`Les résultats de l'essai sont comparés aux résultats obtenus.`llvm-cxxfilt --no-strip-underscore`Le drapeau est important sur macOS, où l'outil enlève un sous-titrage principal par défaut et ne reconnaît pas ensuite `Z8describe...`- Je suis désolé .

### Le vtable de ses délocalisations

La section 2.5.2 de l'ABI énumère les composants d'une table virtuelle dans l'ordre: les compensations d'appels virtuels, les compensations de base virtuelles, les compensations vers le haut, le pointeur typeinfo, puis les pointeurs de fonction virtuelle. Le point d'adresse, la valeur de vptr d'un objet, est l'adresse du premier pointeur de fonction (section 2.6.2). `Rect`n'a pas de bases virtuelles, donc sa table est compensée par le haut, typeinfo et les fonctions.

Le symbole`_ZTV4Rect`est de 0x30 octets, six espaces de 8 octets. Dans l'objet chaque octet est de zéro.`.rela.data.rel.ro._ZTV4Rect`- Le numéro de la liste:

```text
0000000000000008 R_X86_64_64              _ZTI4Rect
0000000000000010 R_X86_64_64              _ZN4RectD2Ev
0000000000000018 R_X86_64_64              _ZN4RectD0Ev
0000000000000020 R_X86_64_64              _ZNK4Rect4areaEv
0000000000000028 R_X86_64_64              _ZNK4Rect4nameEv
```

La fente 0 maintient le décalage vers le haut, ce qui est 0 et ne nécessite aucun déménagement.`+16`De là, la fente 0 est le destructeur complet de l'objet, la fente 1 est la destruction de la suppression, la fente 2`area`, point 3 `name`- Je suis là .`read_vtable()`Il lit la taille du symbole, les délocalisations dans sa plage et le zéro à l'écart 0. `Shape`La table a la même forme que `__cxa_pure_virtual`dans la fente 2, qui est la fonction de temps d'exécution qu'une pure appel virtuel atteint.

### RTTI

La section 2.9.4 décrit `std::type_info`comme un vptr suivi d'un pointeur vers le nom du type mangled, et énumère les classes dérivées. `_ZTI4Rect`a trois délocalisations: `_ZTVN10__cxxabiv120__si_class_type_infoE+0x10`- Je suis là .`_ZTS4Rect`, et `_ZTI5Shape`La première est la table de`__si_class_type_info`, la classe utilisée par l'ABI pour un type avec une base non virtuelle publique unique à l'offset zéro, et le `+0x10`est le point d'adresse de cette table.`4Rect`, qui est le nom du type manglé, et `name()`sur une`type_info`Le troisième est la base.`_ZTI5Shape`utilisations `__class_type_info`et n'a pas de pointeur de base. `read_typeinfo()`Retourne ces trois faits.

### L'appel virtuel

`describe`à`-O0`fait un appel virtuel:

```text
       f: 48 8b 7d f8                  	movq	-0x8(%rbp), %rdi
      13: 48 8b 07                     	movq	(%rdi), %rax
      16: ff 50 10                     	callq	*0x10(%rax)
```

La première .`movq`Le second lit 8 octets à l'offset 0 de l'objet, qui est le vptr, parce que l'ABI le place en premier dans une classe avec des fonctions virtuelles.`callq`passe par là .`0x10`Le rapport de l'Agence de sécurité a été publié en décembre.`0x10 / 8`est la tranche 2.`find_virtual_calls()`correspond au modèle et rapporte les deux candidats: `Rect::area() const`de `_ZTV4Rect`et `__cxa_pure_virtual`de `_ZTV5Shape`Lequel de ces deux sort dépend de l'objet, c'est pourquoi l'appel est indirect.

Le DWARF enregistre la même position.`DW_TAG_subprogram`les entrées pour `area`dans les deux classes de transport`DW_AT_vtable_elem_location (DW_OP_constu 0x2)`- Je suis là .`name`Les destructeurs ont la fente 0.`Rect`est une `DW_TAG_structure_type`de 16 octets avec un `DW_TAG_inheritance`entrée pour `Shape`à l' offset 0 et membres `w`à 8 heures et `h`À 12 heures.`Shape`est de 8 octets avec un membre artificiel `_vptr$Shape`. Deux dérivées indépendantes donnent la place 2 pour `area`: le tableau de relocalisation et les entrées de débogage.

À `-O1`L'objet n'a pas de vtable du tout. `main`constructions `Rect`comme un local, donc le compilateur résolu `r.name()`et `describe(r, 2)`La table n'est donc pas émise. `describe`contient encore `movq (%rdi), %rax`et `callq *0x10(%rax)`, parce que son paramètre peut être n' importe .`Shape`Une table vtable manquante est un fait sur un fichier objet, et le test `test_the_optimised_object_keeps_the_call_pattern_but_drops_the_vtables`Vérifie les deux parties.

Le Microsoft ABI diffère dans chacun de ces détails.`?`et encoder la convention d'appel, donc `void test(void)`Compilée en C++ `__cdecl`est `?test@@ZAXXZ`Le localisateur RTTI est situé à `vfptr[-1]`comme une `_RTTICompleteObjectLocator`, et les symboles sont `??_7`pour un vftable et `??_R4`La documentation de Microsoft dit qu'il n'y a pas de normes pour le nommage en C++ entre les fournisseurs de compilateurs.

## Comment les outils présentent les types récupérés

Chaque ligne ci-dessous qui donne un nom a été produite sur cette machine et capturée dans `code/fixtures/tools/`Les lignes pour les outils qui ne sont pas installés ici décrivent leur documentation et le disent.

| Tool | Version, licence | Where types live | On our fixtures |
|---|---|---|---|
| Ghidra | 12.1.4, 2026-09-21, Apache-2.0 | Data Type Manager with built-in, user-defined, and derived types; `.gdt` archives; types used by a program are copied into it | With DWARF: `/DWARF/account.c/account size=32 align=8` and `a->balance = by + a->balance;`. Without: `*(int *)(param_1 + 4)` with `long param_1`, `tag(char *param_1)`, and `*(char **)(param_1 + 0x18)` |
| IDA Pro | 9.4, 2026-07-13, commercial; IDA Free is non-commercial with a cloud decompiler for x86 only | Local Types window, `.til` type libraries, DWARF and PDB import into a `dwarf` or `pdb` folder | Not installed. Documentation only |
| Binary Ninja | 6.0, 2026-09-03, commercial with a Free edition | Type editing, header import and export, type archives, debug info from DWARF and PDB, type libraries, platform types | Not installed. Documentation only |
| radare2 | 6.2.4, 2026-10-05, LGPLv3 | `ts`, `tsc`, `afvd` | `ts` and `tsc account` printed nothing for the relocatable object with `bin.dbginfo=true`; `pdf` shows `mov dword [rax + 4], ecx` with source line comments from DWARF |
| rizin | 0.9.1, 2026-06-29, LGPL core | `ts`, `tsc`, `afvd` | `tsc account` printed `struct account { char kind; int balance; short code; double rate; const char *label; };` from DWARF; the build without `-g` lists only its bundled header types |
| lldb | lldb-1703.0.236.21 | `image lookup -t` | `byte-size = 32` and the full `compiler_type` text from the ELF object |
| gdb | not installed here | `ptype /o` prints "the offsets and sizes of fields in a struct" with holes marked | Documentation only |
| pyelftools | 0.33, public domain | Python API over `.debug_info` | A 43-line script prints the same five members and `unit version 5` |
| llvm-dwarfdump | Apple LLVM 17.0.0 | `--debug-info`, `--debug-abbrev`, `--debug-str-offsets` | The oracle for every reader test |

La sortie de Ghidra sans DWARF montre les mêmes deux règles en fonction.`char *`à la valeur de `tag`parce que la fonction compare un octet à travers elle, et `char **`sur le terrain à `0x18`Parce que`label_length`déference la valeur chargée et compare les octets. Ce sont les mêmes deux règles que le passage de cette leçon: largeur de l'instruction, pointeur de la dériverence. La documentation du décompilateur de Ghidra indique que les varnodes de code p "n'ont pas nécessairement un type de données associé à eux" et que le décompilateur "attribue finalement un type de données formel" par propagation. L'option "Inférer des pointeurs constants" étend la même propagation aux constantes. Ghidra 12.1 a ajouté la récupération de bitfield au décompiler, ce que les fixtures de cette leçon n'exercent pas.

La table faite à la main est l'autre moitié de l'histoire.`td`Tout le monde vous laisse taper le struct de la mise en page récupérée et regarder le décompiler rediriger.

réa 6.5.0 avec son fournisseur Ghidra renvoie le texte du décompilateur en tant qu'enregistrement. Tout ce qui est dans la ligne Ghidra s'applique à elle, et l'enregistrement porte les noms de type sans la preuve qui les a produits.`python3 code/main.py infer`sur la même liste vous donne cette preuve.

## Recherche de type d'inférence

Le passage ici est la moitié de la génération de contraintes d'un système d'inférence de type, limité à un struct atteint à partir d'un paramètre.

- TIE (Lee, Avgerinos, Brumley, NDSS 2011) a cadré la reconstruction de type comme génération de contraintes et résolution sur une IR binaire, a mesuré la conservatrice et la précision, et comparé avec un décompilateur commercial sur 87 programmes de base.
- Retypd (Noonan, Loginov, Cok, PLDI 2016) a ajouté des sous-types, des polymorphismes et des types récursifs, et a rapporté une reconstruction du pointeur `const`Une mise en œuvre open source existe sous le nom de GrammaTech.
- OSPREY (Zhang et autres, IEEE S&P 2021) traite chaque emplacement de mémoire comme une variable aléatoire et infère les types les plus probables.
- DIRTY (Chen et autres, USENIX Security 2022) et ReSym (Xie et autres, CCS 2024) produisent des décompiles post-processus avec des modèles apprentis, et ReSym rapporte une précision de 72,9% sur les mises en page complètes de structures.
- RecStruct (Chen et autres, USENIX Security 2026) récupère les structs nichés sans décompiler, avec une exécution symbolique insensible au flux et une union-trouver sur les fentes de pile.

La structure de pile est la première chose que cette leçon passe.`-O1`- Je suis là .`main`Il le garde .`rows`sur la pile et le code écrit en ligne`0x24(%rsp)`- Je suis là .`0x40(%rsp)`- Je suis là .`0x48(%rsp)`- Je suis là .`0x30(%rsp)`, et `0x38(%rsp)`Ces modules 32 sont 4, 0, 8, 16 et 24: le même structure, sans paramètre de semence.

## Faites-le

Exécutez la démo:

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types
python3 code/main.py
```

La sortie est un rapport JSON avec cinq parties. `dwarf`détient l'en-tête de l'unité, les structures et les fonctions de l'unité `-O0`objet.`oracles`contient la disposition des règles psABI, le `ctypes`L'élaboration de la mise en page, la mise en page DWARF, la mise en œuvre native et un `agree`Le drapeau.`builds`Il contient les accès, la disposition déduite et la comparaison pour chacun des trois objets. `cxx`contient les noms démêlés, les deux vtables, les deux enregistrements typeinfo, l'appel virtuel et les fentes DWARF. `tools`liste des sorties de l'outil capturées.

Forme attendue:

```json
{
  "dwarf": {"unit": {"unitLength": 489, "version": 5, "unitType": "DW_UT_compile", "addressSize": 8, "abbrevOffset": 0}, "...": "..."},
  "oracles": {"agree": true, "psabiRules": {"size": 32, "align": 8, "...": "..."}, "...": "..."},
  "builds": {
    "O0": {"layout": {"stride": 32, "size": 32, "gaps": [1, 2, 3, 10, 11, 12, 13, 14, 15], "...": "..."}, "comparison": {"matches": 5, "members": 5, "...": "..."}},
    "O1": {"...": "..."},
    "O2": {"...": "..."}
  },
  "cxx": {"virtualCalls": [{"slot": 2, "candidates": {"Shape": "__cxa_pure_virtual", "Rect": "_ZNK4Rect4areaEv"}, "...": "..."}], "...": "..."}
}
```

Lire `code/main.py`dans cet ordre:

1. `ElfObject`Il lit des sections, des symboles et des délocalisations, et `relocated()`est appliquée `S + A`à une section de débogage.
2. `parse_abbreviations()`et `DwarfInfo._parse_unit()`Lisez l'en-tête, la table des abréviations et les entrées, puis résolvez `strx`et `addrx`à travers les deux tables.
3. `DwarfInfo.struct_layouts()`et `functions()`transformer les entrées en `StructLayout`et `FunctionInfo`les documents, avec `type_name()`- Je suis là .`type_size()`, et `type_kind()`- Je suis désolé .
4. `parse_listing()`- Je suis là .`parse_memory()`, et `access_width()`Lire le `llvm-objdump`Je vous ai envoyé un SMS.
5. `trace_function()`est la machine de l'état abstrait, et `fold_unrolled()`et `infer_layout()`transformer les accès en un plan.
6. `compare_layouts()`produit les lignes de statut, et `sysv_layout()`et `ctypes_layout()`sont les deux autres oracles.
7. `Demangler`- Je suis là .`read_vtable()`- Je suis là .`read_typeinfo()`, et `find_virtual_calls()`couvrir l'objet C++.

Il y a quatre commandements.`python3 code/main.py dwarf FILE.o`imprime les structures et les fonctions de tout objet relocalisable DWARF 5. `python3 code/main.py infer LISTING.txt [FILE.o]`Il utilise le passe dans une liste et compare avec un objet son DWARF. `python3 code/main.py demangle NAME...`et `python3 code/main.py vtable FILE.o`couvrir la moitié de C++.

## Utilisez-le

Sur un nouvel objet que vous avez construit vous-même:

1. Capture `llvm-objdump -d -r`Le passe lit ce texte.
2. On court .`python3 code/main.py infer`Lire la ligne témoins: elle nomme les fonctions dont le premier paramètre a été utilisé comme base de mémoire. Si une fonction manque, son paramètre n'a jamais atteint un opérand de mémoire, ou il a atteint un par un chemin que le passage ne suit pas.
3. Lisez chaque champ avec son nombre d'accès.
4. Lisez les preuves de la marche.`call-site`ou `scale`L'entrée est forte.`loop-step`L'entrée sans autre type est faible, car une boucle déroulée rapporte l'étape de plusieurs éléments.`unrolled`Il apparaît, vérifiez le pli à la main.
5. Si une`-g`construire existe, faire la comparaison et lire chaque ligne qui n'est pas `match`- Je suis un .`no-evidence`la ligne nomme un membre sans fonction touché.`unexpected`nom de rangée à un accès que le passe ne pouvait pas placer.
6. Tapez la disposition dans le gestionnaire de type de votre outil comme un struct, et enregistrer quels champs, noms et types sont des devisions.

Gardez les trois oracles à portée de main.`ctypes`, et DWARF ne sont pas d'accord sur une mise en page que vous avez écrite, vous avez trouvé un struct chargé, un bit-field, ou une cible avec des règles différentes.

## Laboratoire interactif

### Laboratoire A: décodez l'entrée membre

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types/code
python3 -q
```

Je vais courir .

```python
from main import *
elf = load_object("account-x86_64-O0-g.o")
dwarf = DwarfInfo(elf)
dwarf.header()
elf.section(".debug_info").data[0x1a3:0x1ac].hex(" ")
dwarf.dies[0x1a3].attributes
dwarf.type_name(dwarf.type_of(dwarf.dies[0x1a3]))
```

Décodez les neuf octets sur papier avec l'abréviation 13 avant de regarder le dictionnaire.`elf.section(".debug_str_offsets").data[8:16].hex()`, et celui qui a été déplacé,`elf.relocated(".debug_str_offsets")[8:16].hex()`, et expliquer la différence.

### Laboratoire B: tracer une fonction

```python
listing = load_listing("account-x86_64-O0-g.objdump.txt")
trace = trace_function("label_length", listing["label_length"])
[(hex(a.address), a.root, a.offset, a.width, a.kind) for a in trace.accesses]
trace.pointer_uses
trace_function("sum_balances", listing["sum_balances"]).loops
```

Avant d'exécuter, écrivez quel accès prouve que le décalage 24 est un pointeur.`-O1`la liste et trouver les `leaq 0x1(%rcx), %rcx`dans `trace.increments`- Je suis désolé .

### Laboratoire C: plier la boucle vectoriée

```python
o2 = load_listing("account-x86_64-O2-g.objdump.txt")
trace = trace_function("sum_balances", o2["sum_balances"])
sorted({a.offset for a in trace.accesses})
folded, evidence = fold_unrolled(trace, trace.accesses)
sorted({a.offset for a in folded}), [(e.kind, e.value, e.text) for e in evidence]
```

Alors , courez .`analyze_build("O2", fold=False)["comparison"]`et compter le `unexpected`Les lignes. Modifiez la règle de pliage pour que l'étape de la boucle soit exactement`k * d`avec `k`au moins 2, et confirmer que le `-O1`boucle, avec un accès et l'étape 32, est laissé seul.

### Laboratoire D: le tableau et les noms

```python
shapes = load_object("shapes-x86_64-O0-g.o")
read_vtable(shapes, "_ZTV4Rect")
read_typeinfo(shapes, "_ZTI4Rect")
demangle("_ZNK4Rect4areaEv"), demangle("_Z8describeRK5Shapei")
find_virtual_calls(load_listing("shapes-x86_64-O0-g.objdump.txt"), [read_vtable(shapes, "_ZTV5Shape"), read_vtable(shapes, "_ZTV4Rect")])
```

Calculez la fente à partir de `callq *0x10(%rax)`sur papier, puis confirmez contre `DwarfInfo(shapes).struct("Rect").virtual_methods`- Alors chargez .`shapes-x86_64-O1-g.o`et essayez`read_vtable`Je vais lire le livre.`ElfError`et expliquer pourquoi le symbole est absent.

## Laboratoire de pratique

Récupérer la structure de pile en `main`à`-O1`- Je suis désolé .

Les exigences:

- Extension `trace_function()`Ainsi, une région de la fente de cadre peut être une racine.`movaps`Les magasins qui se remplissent `0x0(%rsp)`à `0x5f(%rsp)`de `.data.rel.ro`, qui est l' initialisateur de 96 octets de `rows`- Je suis désolé .
- Attribuer les accès en ligne `0x24(%rsp)`- Je suis là .`0x40(%rsp)`- Je suis là .`0x48(%rsp)`- Je suis là .`0x30(%rsp)`, et `0x38(%rsp)`Je suis en train de faire une demande de répartition de la part de la Commission.
- Dériver l' étape 32 de la boucle à `0x120`à `0x12c`Où ?`%r14`commence à 4 et avance à `0x20`jusqu' à`0x64`, et montrer que la boucle est liée `0x64`est trois éléments plus le champ de compensation.
- Comparez le résultat avec la mise en page DWARF et atteignez cinq correspondances, et enregistrer les types restent non prouvés.
- Ajoutez un test qui échoue lorsque la racine du cadre est attribuée à un paramètre.

Le succès signifie`python3 code/main.py infer`sur le `-O1`Liste de référencement `main`comme un témoin avec une racine de cadre, et la comparaison montre cinq correspondances de `main`Je suis seule.

## Leçon d'artisanat

Cette leçon comprend `outputs/skill-recovering-data-and-types.md`. Utilisez-le pour transformer une liste et une configuration de débogage optionnelle en une mise en page structurelle avec des preuves par champ, un pas avec ses sources, une comparaison avec DWARF et une liste de ce qui est resté inconnu. Il nécessite la table d'accès, la preuve des pas, la vérification du pli, l'oracle DWARF lorsqu'il existe, et une déclaration de noms et de qualifications qu'aucune instruction ne peut fournir.

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La vérification doit prouver:

- L'en-tête de l'unité, les 14 abréviations et les cinq composantes de l'offset sont d'accord avec `llvm-dwarfdump`
- Les noms arrivent par les délocalisés `.debug_str_offsets`table et de l'unité `str_offsets_base`et `addr_base`sont 8
- six fonctions portent les gammes, `DW_OP_reg6 RBP`, et `DW_OP_fbreg`les emplacements que le dépôt affiche et leurs adresses basses équivaut à la fonction de mise en liste démarre
- un objet sans `.debug_info`Le décodeur LEB128 est utilisé.`0x78`comme -8
- `pyelftools`et `lldb`imprimer la même disposition que le lecteur
- les règles de la PSABI, `ctypes`, DWARF, et la course native donner des compensations `0, 4, 8, 16, 24`et la taille 32, et le struct réorganisé est de 24 octets
- Le parseur de liste lit 13 instructions dans `bump`, attaches `R_X86_64_PLT32 bump-0x4`à l' appel à `0x194`, et trouve le saut en arrière à `0x72`
- Opérands de mémoire, largeurs, types et signes décodés du texte AT&T et du mnemonique
- `bump`à`-O0`lis son pointeur à l'arrière de la fente du cadre et fait trois accès à l'offset 4
- Le pas vient de `shlq $0x5`à`-O0`, à partir de `addq $0x20, %rdx`à`-O1`, et à partir de huit charges pliées par un pas de 256 à`-O2`
- `label_length`prouve le pointeur à l'offset 24 en le déférencant
- `main`passe un objet de cadre à des décalages 0, `0x20`, et `0x40`à`-O0`et ne appelle que `printf`à`-O1`
- chaque construction correspond à cinq des cinq membres avec taille 32 et les espaces égaux au rembourrage DWARF
- sans plier le `-O2`construire des rapports sept champs passé par byte 32 et aucune taille
- Le démangeur reproduit les 19 `llvm-cxxfilt`les chaînes
- les deux tables vtables, les deux enregistrements typeinfo, et l'appel virtuel à `0x16`résoudre à la fente 2, et DWARF est d'accord
- le `-O1`L'objet C++ conserve le modèle d'appel et n'a pas de vtable
- Les captures de Ghidra, rizin et radar2 montrent la structure avec DWARF et les compensations brutes sans elle
- la figure interactive intègre les mêmes données que l'analyse produit

## Mode de défaillance de la production

| Failure | What the weak tool reports | What the analysis must prove |
|---|---|---|
| Unrolled loop | Eight fields 32 bytes apart and a 232-byte struct | Loop step 256 equals 8 times the offset delta 32, so one field at offset 4 |
| Stride read from one increment | `sizeof` 256 from `addq $0x100` | The greatest common divisor of every stride source, with the scalar tail's `addq $0x20` |
| 8-byte field called a pointer | `void *` at offset 24 | A later access uses the loaded value as a base, or the kind stays `int64` |
| Pointer called an integer | `long` at offset 24 | The dereference at `0xfb`, recorded as pointer proof |
| Narrow compare | `char` with a sign | `cmpb` carries no sign; only `movsbl` or `movzbl` does |
| Five functions assumed to share a type | One struct with no evidence of aliasing | Call sites in `main` that pass one frame object, or an explicit analyst assertion |
| Strings read from `.debug_str` by index | Garbage names from a relocatable object | `str_offsets_base` plus the relocated `.debug_str_offsets` entry |
| Addresses read as numbers | `low_pc` of 0 for every function | `addrx` through the relocated `.debug_addr` table |
| DWARF 4 object | A reader that misparses the header | Version check on the second header field; 4 has no `unit_type` byte |
| Missing vtable | "the class has no virtual functions" | The `-O1` object inlined every call; the `-O0` object has the table and its relocations |
| Vtable slot read from the wrong base | Off-by-two slot numbers | The address point is 16 bytes past the symbol, after offset-to-top and typeinfo |
| Mangled name stripped | `Z8describeRK5Shapei` left as is | `--no-strip-underscore` on macOS, or a demangler that checks for `_Z` |
| Struct typed from one tool's guess | `char *param_1` taken as the field type | The width and the dereference, written as evidence, and the name marked as a guess |

## Règlement d'exploitation

Récupérer un champ d'une accès et un pas d'une boucle, et joindre l'instruction à chacun. Faites des boucles déroulées avant de faire confiance à un décalage passé par la marche. Lisez DWARF à travers ses abréviations et ses tableaux déplacés, jamais par indexation dans une section de chaîne. Comparez champ par champ et écrivez chaque membre sans preuve, chaque pointeur non prouvé et chaque nom comme inconnu. Dans un objet C++, lisez la table v à partir de ses délocalisations et la fente à partir de l'appel, puis utilisez DWARF pour confirmer la fente que vous lisez.

## Pour en savoir plus

- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, section 3.1.2: Data Representation](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64), section 5.10: Composite Types](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [Writing ARM64 code for Apple platforms](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)
- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 1, section 3.7.5: Specifying an Offset](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [DWARF Debugging Information Format Version 5, sections 2.1, 5.7.6, 7.5.1.1, 7.5.3, 7.5.6](https://dwarfstd.org/doc/DWARF5.pdf)
- [Itanium C++ ABI, sections 2.5 (Virtual Table Layout), 2.9 (Run-Time Type Information), 5.1 (External Names)](https://itanium-cxx-abi.github.io/cxx-abi/abi.html)
- [Microsoft Learn: Decorated names](https://learn.microsoft.com/en-us/cpp/build/reference/decorated-names?view=msvc-170)et [Name Decoration](https://learn.microsoft.com/en-us/cpp/error-messages/tool-errors/name-decoration?view=msvc-170)
- [llvm-dwarfdump command guide](https://llvm.org/docs/CommandGuide/llvm-dwarfdump.html)et [llvm-cxxfilt command guide](https://llvm.org/docs/CommandGuide/llvm-cxxfilt.html)
- [Python ctypes: Structure layout, sizeof, alignment](https://docs.python.org/3/library/ctypes.html)
- [Ghidra Data Type Manager](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Base/src/main/help/help/topics/DataTypeManagerPlugin/data_type_manager_description.htm)- Je suis là .[Decompiler Concepts](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/main/help/help/topics/DecompilePlugin/DecompilerConcepts.html)- Je suis là .[Decompiler Options](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/main/help/help/topics/DecompilePlugin/DecompilerOptions.html), et [Ghidra 12.1.4 release](https://github.com/NationalSecurityAgency/ghidra/releases/tag/Ghidra_12.1.4_build)
- [IDA 9.1 user guide: Types](https://docs.hex-rays.com/9.1/user-guide/types)- Je suis là .[IDA 9.4 release post](https://hex-rays.com/blog/ida-9.4-release-a-new-dyld-shared-cache-swift-analysis-new-teams-add-on-and-more-), et [IDA Free](https://hex-rays.com/ida-free)
- [Binary Ninja user guide: Types](https://docs.binary.ninja/guide/types/)et [Binary Ninja 6.0 Krypton](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)
- [radare2 6.2.4 release](https://github.com/radareorg/radare2/releases/tag/6.2.4)et [rizin v0.9.1 release](https://github.com/rizinorg/rizin/releases/tag/v0.9.1)
- [GDB manual: Examining the Symbol Table, ptype /o](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Symbols.html)
- [pyelftools on PyPI](https://pypi.org/project/pyelftools/)
- [Lee, Avgerinos, Brumley, "TIE: Principled Reverse Engineering of Types in Binary Programs" (NDSS 2011)](https://www.ndss-symposium.org/ndss2011/tie-principled-reverse-engineering-of-types-in-binary-programs/)
- [Noonan, Loginov, Cok, "Polymorphic Type Inference for Machine Code" (PLDI 2016)](https://arxiv.org/abs/1603.05495)et [the retypd repository](https://github.com/GrammaTech/retypd)
- [Zhang et al., "OSPREY: Recovery of Variable and Data Structure via Probabilistic Analysis for Stripped Binary" (IEEE S&P 2021)](https://ieeexplore.ieee.org/document/9519451/)
- [Chen et al., "Augmenting Decompiler Output with Learned Variable Names and Types" (USENIX Security 2022)](https://www.usenix.org/conference/usenixsecurity22/presentation/chen-qibin)
- [Xie et al., "ReSym: Harnessing LLMs to Recover Variable and Data Structure Symbols from Stripped Binaries" (CCS 2024)](https://dl.acm.org/doi/10.1145/3658644.3670340)
- [Chen et al., "RecStruct" (USENIX Security 2026)](https://www.usenix.org/conference/usenixsecurity26/presentation/chen-yuxin)
