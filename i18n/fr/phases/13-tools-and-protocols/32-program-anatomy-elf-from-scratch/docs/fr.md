# Anatomie du programme: un lecteur ELF à partir de zéro

> Un programme compilé est un ensemble de tables qui pointent les unes vers les autres. Un lecteur que vous avez écrit donne une sortie que vous pouvez vérifier octet par octet.

**Type:** Build
**Languages:** Python (stdlib), C (fixture only)
**Prerequisites:** Phase 0 · 01 (Dev Environment) with clang installed, Python 3.10 or later, hexadecimal arithmetic
**Time:** ~120 minutes

## Objectifs d'apprentissage

- Indiquez la règle d'autorisation pour chaque cible de cette piste et les quatre limites de 17 U.S.C. 1201 ((f).
- Lisez l'en-tête ELF, le tableau d'en-tête section et les deux tables de chaînes avec `struct.unpack`- Je suis désolé .
- Décodez`.symtab`les entrées et `.rela`des enregistrements avec les noms du gABI et du x86-64 psABI.
- Retournez les octets exacts d'une fonction nommée et cartez chaque fichier décalé à la structure qui la possède.
- Lisez les en-têtes de programme et cartez une adresse virtuelle à un fichier compensé à travers `PT_LOAD`- Je suis désolé .
- Lisez les commandes de charge d'un objet Mach-O et nommez l'idée ELF qui correspond à chacun.

## Avant de commencer: autorisation et loi

Chaque cible des leçons 32 à 36 est un programme que vous avez écrit et compilé.`code/fixtures/ledger.c`Vous possédez la source, la commande du compilateur et les fichiers objet qu'il a produits. Il n'y a pas de licence à lire, pas de protection à passer, et pas de tiers dont vous ouvrez le code. La piste ne couvre jamais les protections logicielles, les vérifications de licence ou les programmes que vous ne possédez pas.

La règle existe parce que la loi traite le démantèlement comme une copie. Dans *Sega Enterprises Ltd. v. Acccolade, Inc.*, 977 F.2d 1510 (9e Cir. 1992), la Cour a statué que les copies intermédiaires faites lors du démontage sont des copies en vertu de la Loi sur le droit d'auteur. Il a ensuite conclu que "lorsque le démontage est le seul moyen d'accéder aux idées et aux éléments fonctionnels incarnés dans un programme informatique protégé par le droit d'auteur et qu'il existe une raison légitime de demander un tel accès, le démontage est un usage équitable de l'œuvre protégée par le droit d'auteur, en tant que question de droit". 

Le deuxième cas est * Sony Computer Entertainment, Inc. c. Connectix Corp. *, 203 F.3d 596 (9e Cir. 2000). Connectix a copié le BIOS de PlayStation alors qu'il construisait un émulateur, et le tribunal a jugé que ces copies étaient utilisées équitablement.

Deux lois fixent les limites. 17 U.S.C. 1201(f) permet à une personne "qui a obtenu légalement le droit d'utiliser une copie d'un programme informatique" de contourner un contrôle d'accès "pour identifier et analyser les éléments du programme nécessaires à l'interopérabilité d'un programme informatique créé indépendamment avec d'autres programmes". Quatre limites: une copie légale, l'objectif d'interopérabilité, des éléments qui n'étaient pas déjà "pratiquement disponibles" et une analyse qui n'est pas contraire aux lois. La section 1201 ((f) est une exception à la seule règle de l'évasion. La question de la copie reste à utiliser équitablement.

Dans l'Union européenne, la directive 2009/24/CE article 5 ((3) couvre l'étude. Une personne ayant le droit d'utiliser une copie peut "observer, étudier ou tester le fonctionnement du programme afin de déterminer les idées et les principes qui sous-tendent tout élément du programme". Le droit s'applique lors du chargement, de l'affichage, de l'exécution, de la transmission ou du stockage du programme.

L'article 6 autorise la décompilation sans l'autorisation du titulaire du droit lorsqu'il est "indispensable d'obtenir les informations nécessaires pour atteindre l'interopérabilité d'un programme informatique créé indépendamment". Les conditions correspondent à 1201 ((f). Un titulaire de licence effectue les actes, les informations n'étaient pas déjà facilement disponibles et les actes restent dans les parties nécessaires. L'article 6(2) interdit l'utilisation du résultat pour tout autre objectif ou pour un programme qui est substantiellement similaire en termes d'expression. L'article 8 rend nulles les conditions contractuelles contraires à l'article 6 ou à l'article 5*3).

| Source | What it allows | Conditions |
|---|---|---|
| 17 U.S.C. 1201(f) | Circumventing an access control to analyze a program | Lawful copy, interoperability purpose, information not readily available, no infringement |
| Directive 2009/24/EC Art. 5(3) | Observing, studying, and testing a running program | A right to use the copy, acts within normal loading and running |
| Directive 2009/24/EC Art. 6 | Decompilation | Licensee, interoperability purpose, information not readily available, confined to the needed parts |
| *Sega v. Accolade* | Disassembly as fair use | The only way to reach the ideas and functional elements, a legitimate reason |
| *Sony v. Connectix* | Intermediate copies during reverse engineering | A non-infringing final product |

Ces notes proviennent des statuts et des avis, et ne constituent pas un avis juridique. Lorsque vous n'avez pas de but à poursuivre, obtenez une autorisation écrite avant d'ouvrir le dossier et gardez l'autorisation avec vos preuves.

## Le problème

Vous avez compilé`ledger.c`et a couru`llvm-objdump -t`Les listes de sortie `percent_change`à la valeur `0000000000000000`avec taille `0000000000000010`Quelles sont les 16 octets du fichier de 1768 octets qui sont cette fonction ?`R_X86_64_PC32 .rodata+0x10`à l' équilibre `0x36`, et les quatre octets à ce point dans le fichier sont zéro.`0x10`quand la table commence à `0x14`- Je suis désolé .

ré ré répond aux questions de mise en page avec `inspect_binary_layout`, et cet outil n' existe que sur Linux x64 avec pwntools 4.15.0 derrière `REA_PWNTOOLS_PYTHON`- Sur ce Mac, il est absent.`address_to_file_offset`Il faut une session Ghidra et une copie complète des octets du fichier original avec un SHA-256 correspondant. `inspect_artifact`Il n'ouvre jamais les tables à l'intérieur d'un fichier objet. Vous écrivez donc le lecteur vous-même dans environ 950 lignes de Python avec `struct.unpack`Les tests comparent chaque table avec`llvm-objdump`texte capturé une fois dans les appareils.

```figure
elf-byte-map
```

## Le fichier est des tables qui pointent vers des tables

Le système V gABI, chapitre 4, figure 4-1, présente deux vues d'un seul fichier. La vue de liaison dispose d'une en-tête ELF, d'une table optionnelle d'en-tête de programme, de sections et d'une table requise d'en-tête de section. La vue d'exécution dispose d'une en-tête ELF, d'une table d'en-tête de programme requise, de segments et d'une table d'en-tête de section optionnelle. Le gABI ajoute une phrase qui régit l'ensemble du lecteur: "Seul l'en-tête ELF a une position fixe dans le fichier". Chaque autre structure est trouvée par un décalage stocké dans une autre structure.

Le gABI fixe également les types de scalaires et dit que "ELF n'utilise pas de champs de bits". `Elf64_Addr`et `Elf64_Off`sont 8 octets, `Elf64_Half`est de 2, `Elf64_Word`est de 4, et `Elf64_Xword`Le x86-64 psABI, section 4.1.1, exige `ELFCLASS64`- Je suis là .`ELFDATA2LSB`- Je suis là .`e_machine` `EM_X86_64`(62), et `e_flags`0. Moins de largeur fixe moyenne `struct.unpack`avec un `<`Le préfixe lit tous les enregistrements en un seul appel.

Les 64 premiers octets de `ledger-x86_64-O1.o`- Le numéro de la liste:

```text
00000000: 7f45 4c46 0201 0100 0000 0000 0000 0000  .ELF............
00000010: 0100 3e00 0100 0000 0000 0000 0000 0000  ..>.............
00000020: 0000 0000 0000 0000 2804 0000 0000 0000  ........(.......
00000030: 0000 0000 4000 0000 0000 4000 0b00 0100  ....@.....@.....
```

Lisez-les contre la page "ELF Header" de GABI. `e_ident`est le 16 octets `7f 45 4c 46`(la magie), classe 2 (64 bits), données 1 (petit-endia), version 1, OS ABI 0 (Système V), et rembourrage.

Alors ...`e_type` `0x0001`est `ET_REL`, un objet relocalisable. `e_machine` `0x003e`est de 62, `EM_X86_64`- Je suis là .`e_entry`est zéro parce qu'un objet n'a pas de point d'entrée. `e_phoff`est zéro parce qu'un objet n'a pas d'en-têtes de programme. `e_shoff`est `0x428`, le fichier de la table d'en-tête de section.`e_ehsize` `0x40`- Je suis là .`e_phentsize`0, `e_phnum`0, `e_shentsize` `0x40`- Je suis là .`e_shnum` `0x0b`11 titres), et `e_shstrndx` `0x0001`- Je suis désolé .

Le lecteur transforme cette ligne en un appel:

```python
fields = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
```

La chaîne de format est la figure gABI lue de gauche à droite: deux demi-parts, un mot, trois valeurs de 8 octets, un mot et six demi-parts. Le lecteur refuse un fichier dont la magie, la classe ou l'ordre de octets diffère, et il refuse `e_shentsize`ou `e_phentsize`Les valeurs qui ne sont pas 64 et 56 sont de mauvaise largeur rendrait chaque compensation ultérieure incorrecte, donc le chèque est en haut.

## Titres de section et tables à deux cordes

Chaque en-tête de section est une `Elf64_Shdr`de 64 octets (figure 4 à 8): `sh_name`- Je suis là .`sh_type`- Je suis là .`sh_flags`- Je suis là .`sh_addr`- Je suis là .`sh_offset`- Je suis là .`sh_size`- Je suis là .`sh_link`- Je suis là .`sh_info`- Je suis là .`sh_addralign`, et `sh_entsize`La chaîne de formatage est `<IIQQQQIIQQ`- Je suis là .`sh_name`est un décalage dans la section dont l'indice est `e_shstrndx`, et le nom se termine à la première zéro octet.`e_shstrndx`est 1, et clang nommé cette section `.strtab`, donc une table sert à la fois les noms de sections et les noms de symboles.

Le lecteur imprime les 11 en-têtes de l'objet -O1:

```text
Idx Name             Type         Flags   Offset     Size       Addr Link Info Align
  0                  NULL                    0x0      0x0        0x0    0    0     0
  1 .strtab          STRTAB                0x368     0xbe        0x0    0    0     1
  2 .text            PROGBITS     AX        0x40     0x9f        0x0    0    0    16
  3 .rela.text       RELA         I        0x2a8     0x48        0x0   10    2     8
  4 .rodata          PROGBITS     A         0xe0     0x50        0x0    0    0     8
  5 .rela.rodata     RELA         I        0x2f0     0x78        0x0   10    4     8
  6 .rodata.str1.1   PROGBITS     AMS      0x130      0xa        0x0    0    0     1
  7 .comment         PROGBITS     MS       0x13a     0x2f        0x0    0    0     1
  8 .note.GNU-stack  PROGBITS              0x169      0x0        0x0    0    0     1
  9 .llvm_addrsig    LLVM_ADDRSIG E        0x368      0x0        0x0   10    0     1
 10 .symtab          SYMTAB                0x170    0x138        0x0    1    9     8
```

L'index 0 est l'en-tête nul réservé. `.text`contient 159 octets de code à l' écart de fichier `0x40`, juste après l'en-tête, avec des drapeaux `A`(alloué en mémoire) et `X`(exécutable). `.rodata`contient 80 octets de données uniquement lisibles. `.rodata.str1.1`Il contient les cinq chaînes d' étiquette `F\0D\0C\0B\0A\0`, 10 octets, avec des drapeaux `M`(fusion) et `S`Je suis en train de vous dire:

`.comment`détient le timbre de compilateur de 47 octets `\0Apple clang version 17.0.0 (clang-1700.6.3.2)\0`- Je suis là .`.llvm_addrsig`a le type `0x6fff4c03`, dans la gamme spécifique au système d'exploitation `SHT_LOOS`à `SHT_HIOS`que les réserves de la gABI et celles de la LLVM `ELF.h`Son drapeau.`E`est `SHF_EXCLUDE`, donc un lien le laisse tomber.

Le `sh_link`et `sh_info`Les colonnes sont les indicateurs entre les tables. `.rela.text`Il a`sh_info`2 et `sh_link`10: il patche la section 2 et prend ses symboles de la section 10. `.symtab`Il a`sh_link`1, la table des chaînes pour les noms des symboles, et `sh_info`9, l'indice du premier symbole non local. `sh_addr`est zéro dans chaque ligne parce qu'un objet n'a pas encore d'adresses.

Deux règles gABI permettent au lecteur de dessiner une carte complète du fichier. Les sections " sont contiguës et ne se chevauchent pas ", et un fichier " peut contenir un espace inactif ". Le lecteur trie chaque structure par compensation et remplit les trous:

```text
     0x0 +64    header   ELF header
    0x40 +159   section  .text
    0xdf +1     gap      inactive space
    0xe0 +80    section  .rodata
   0x130 +10    section  .rodata.str1.1
   0x13a +47    section  .comment
   0x169 +7     gap      inactive space
   0x170 +312   section  .symtab
   0x2a8 +72    section  .rela.text
   0x2f0 +120   section  .rela.rodata
   0x368 +190   section  .strtab
   0x426 +2     gap      inactive space
   0x428 +704   table    section header table
```

Les trois écarts sont de 10 octets, et chacun est alignement. `.rodata`Il a`sh_addralign`8, donc ça commence à `0xe0`plutôt que `0xdf`- Je suis là .`.symtab`a alignement 8, donc sept octets restent inutilisés après `.comment`La table d'en-tête de section elle-même est alignée sur 8.

Section de type `SHT_NOBITS`Il n'y a rien dans cet objet qui ait ce type.`.bss`dans un programme lié, et un lecteur qui a confiance `sh_size`Il y a des mots qui passent par là.

## Symbole: liant, type, section, valeur, taille

Un `Elf64_Sym`est de 24 octets (figure 4 à 18): `st_name`- Je suis là .`st_info`- Je suis là .`st_other`- Je suis là .`st_shndx`- Je suis là .`st_value`- Je suis là .`st_size`, format `<IBBHQQ`- Je suis là .`st_info`Le gABI nomme les liaisons `STB_LOCAL`0, `STB_GLOBAL`1 , `STB_WEAK`2, et les types `STT_NOTYPE`0, `STT_OBJECT`1 , `STT_FUNC`2 , `STT_SECTION`3 , `STT_FILE`Les deux bits bas de `st_other`maintenir la visibilité, `STV_DEFAULT`0 ici tout le long.

```text
Idx            Value   Size Bind   Type    Section          Name
  0              0x0      0 LOCAL  NOTYPE  *UND*
  1              0x0      0 LOCAL  FILE    *ABS*            ledger.c
  2              0x0     20 LOCAL  OBJECT  .rodata          labels.rel
  3              0x0      0 LOCAL  SECTION .rodata          .rodata
  4              0x2      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.1
  5              0x4      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.2
  6              0x6      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.3
  7              0x8      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.4
  8              0x0      0 LOCAL  SECTION .rodata.str1.1   .rodata.str1.1
  9              0x0     16 GLOBAL FUNC    .text            percent_change
 10             0x10     46 GLOBAL FUNC    .text            grade
 11             0x40     31 GLOBAL FUNC    .text            total
 12             0x60     63 GLOBAL FUNC    .text            grade_label
```

L'indice 0 est le symbole réservé à l'indifférence. `st_shndx`nomme la section qui définit le symbole, avec trois valeurs réservées: `SHN_UNDEF`0, `SHN_ABS` `0xfff1`pour le nom du fichier, et `SHN_COMMON` `0xfff2`Les symboles de section ont un nom vide, donc `llvm-objdump`La règle gABI selon laquelle "tous les symboles avec STB_LOCAL avant les symboles faibles et globaux" apparaissent comme les neuf premières lignes, et `.symtab`enregistré 9 en `sh_info`Pour cette raison.

Dans un objet relocalisable `st_value`est un décalage à l'intérieur de la section du symbole, parce que `sh_addr`est zéro. Le fichier de compensation d'une fonction est donc `sh_offset + st_value - sh_addr`, et sa longueur est `st_size`- Le numéro de la liste:

```python
section = elf.sections[symbol.st_shndx]
file_offset = section.sh_offset + (symbol.st_value - section.sh_addr)
code = elf.data[file_offset : file_offset + symbol.st_size]
```

Pour `percent_change`C' est ça .`0x40 + 0`, et les 16 octets sont `85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3`Les tailles comptent autant que les débuts.`grade`se termine à `.text`décompensation `0x3e`Les deux octets .`66 90`qui suivent sont un `nop`le compilateur a ajouté de sorte que `total`commence à`0x40`Les deux octets ne sont pas des symboles.`st_size`La leçon 36 les masque avant de comparer une reconstruction.

## Relocations: ce que signifient les zéro octets

Un `Elf64_Rela`est de 24 octets (figure 4-23): `r_offset`- Je suis là .`r_info`, et une signature `r_addend`, format `<QQq`Le GABI se divise .`r_info`avec deux macros: `ELF64_R_SYM(i)`est `i >> 32`et `ELF64_R_TYPE(i)`est `i & 0xffffffff`Dans un dossier relocalisable`r_offset`est un décalage à l'intérieur de la section nommée par le tableau `sh_info`Les noms et formules des types proviennent du x86-64 psABI, tableau 4.9.`S`est la valeur du symbole, `A`l'additif, `P`l'endroit où le patch est effectué, et `L`l'entrée PLT. `R_X86_64_64`est `S + A`- Je suis là .`R_X86_64_PC32`est `S + A - P`, et `R_X86_64_PLT32`est `L + A - P`- Je suis désolé .

L' objet -O1 a huit enregistrements:

```text
  .rela.text       0x36 R_X86_64_PC32          .rodata+0x10
  .rela.text       0x84 R_X86_64_PC32          .rodata+0x24
  .rela.text       0x93 R_X86_64_PC32          .rodata-0x4
  .rela.rodata      0x0 R_X86_64_PC32          .rodata.str1.1
  .rela.rodata      0x4 R_X86_64_PC32          .L.str.1+0x4
  .rela.rodata      0x8 R_X86_64_PC32          .L.str.2+0x8
  .rela.rodata      0xc R_X86_64_PC32          .L.str.3+0xc
  .rela.rodata     0x10 R_X86_64_PC32          .L.str.4+0x10
```

Détruisez le premier.`0x36`Il est à l' intérieur .`grade`, qui s' étend sur `0x10`à `0x3e`- L' instruction à`0x33`est `48 8d 0d 00 00 00 00`, une `lea`d'une adresse relative à la RIP, et son déplacement de quatre octets se situe à `0x36`. Le processeur ajoute le déplacement à l'adresse de l'instruction suivante, `0x3a`- Je suis désolé .

Le compilateur veut que le résultat soit `.rodata + 0x14`, où une table à cinq entrées de 32 bits `1, 2, 3, 4, 4`La valeur stockée doit donc être`(.rodata + 0x14) - 0x3a`La formule`S + A - P`avec `P = 0x36`donne`.rodata + A - 0x36`- Régler les deux égaux donne`A = 0x14 - 4 = 0x10`L'addend contient le décalage cible moins les quatre octets entre le déplacement et la fin de l'instruction.

Ce tableau est comment clang a compilé le `switch`dans `grade`Le code compute`score / 10`avec un multiplicateur par `0x66666667`Les données de référence sont les données de la table de référence, les données de référence sont les données de référence.`.rodata`Plus un add-on, et seule la construction arm64 de la même source le nomme `l_switch.table.grade`Le deuxième record, à`0x84`dans `grade_label`, points dans une deuxième table de valeurs de 8 octets à `.rodata + 0x28`- Le troisième, à`0x93`, points à `.rodata + 0x0`, le début de `labels.rel`- Je suis désolé .

Les cinq .`.rela.rodata`remplir les dossiers `labels.rel`Chaque entrée est compensée .`4i`et obtient le supplément `4i`Je suis un homme .`S + A - P`réduit à l'adresse de la chaîne moins la base de la table. `grade_label`est lu dans une entrée avec `movslq`Le premier enregistrement nomme le symbole de la section`.rodata.str1.1`avec l' addendum 0 parce que la chaîne `F`est le premier octet de cette section.`.L.str.1`à `.L.str.4`, les étiquettes locales du compilateur pour `D`- Je suis là .`C`- Je suis là .`B`, et `A`- Je suis désolé .

La même source compile à trois tables différentes à trois niveaux d'optimisation. Le lecteur montre la différence sans démontageur:

| Build | Where `labels` lives | How `grade` selects | How `grade_label` reaches `grade` |
|---|---|---|---|
| -O0 | `.data`, five 8-byte pointers, five `R_X86_64_64` records | A jump table in `.rodata`: five `R_X86_64_PC32` records pointing into `.text` | A call with `R_X86_64_PLT32 grade-0x4` |
| -O1 | `.rodata` as `labels.rel`, five 32-bit relative entries | A value table at `.rodata+0x14`, one `R_X86_64_PC32` | Inlined, no call |
| -O2 | Same as -O1 | Same as -O1 | Inlined, `total` grows to 140 bytes |

À -0 le `switch`est une vraie table de saut dont les cinq entrées pointent vers `.text`à`+0x91`- Je suis là .`+0x8c`- Je suis là .`+0x87`- Je suis là .`+0x82`, et `+0x86`- L' appel à`grade`Il porte la même chose.`-0x4`- l'addend pour la même raison que le`lea`L'optimisateur a vu que chaque cas renvoie une constante et a remplacé les sauts par un tableau de résultats.

## Titres de programmes: le point de vue du chargateur

Un objet relocalisable n'a pas d'en-tête de programme, donc `e_phnum`Un objet exécutable ou partagé les a, et le chargement ne lit rien d'autre.`Elf64_Phdr`est de 56 octets (graphique 5-1 de l'ABI): `p_type`- Je suis là .`p_flags`- Je suis là .`p_offset`- Je suis là .`p_vaddr`- Je suis là .`p_paddr`- Je suis là .`p_filesz`- Je suis là .`p_memsz`- Je suis là .`p_align`, format `<IIQQQQQQ`Le GABI fixe trois règles pour`PT_LOAD`Ils apparaissent en ascendant.`p_vaddr`l'ordre et une`p_memsz`plus grande que `p_filesz`signifie zéro remplissage. `p_vaddr`est congruent à `p_offset`modulo `p_align`- Je suis désolé .

Chaque cible de cette piste est un programme que vous avez construit, et aucun connecteur sur ce Mac ne produit un exécutable Linux à partir de `ledger.o`- Alors , le dispositif .`tiny-x86_64`vient de `code/fixtures/build_tiny_elf.py`, qui écrit 736 octets avec `struct.pack`Le fichier contient une en-tête ELF, deux en-têtes de programme, un 36-byte `_start`, un 3 bytes `.rodata`, une table de symboles et six titres de section. `_start`écrit `ok\n`Il est utilisé pour le fichier descriptor 1 et sort avec le statut 42.

```text
Program headers:
  PT_LOAD          r-x offset=0x0 vaddr=0x400000 filesz=0x2e0 memsz=0x2e0 align=0x1000
  PT_GNU_STACK     rw- offset=0x0 vaddr=0x0 filesz=0x0 memsz=0x0 align=0x10

entry 0x4000b0 is file offset 0xb0 through PT_LOAD
```

Le single`PT_LOAD`Il repère le fichier de 736 octets en `0x400000`Le lecteur cartographiera une adresse dans un fichier compensé en trouvant le `PT_LOAD`qui le contient et l'informatique `p_offset + (address - p_vaddr)`Pour le point d'entrée qui est`0 + (0x4000b0 - 0x400000) = 0xb0`, le début de `.text`- Je suis désolé .

C'est le calcul derrière réa.`address_to_file_offset`Pour une cible Ghidra, cet outil revient.`{address, file_offset}`Le fournisseur de Hopper revient aussi .`provider_file_offset`et `image_base_file_offset`L'outil a besoin d'une copie originale complète du dossier et le SHA-256 de cette copie doit correspondre à l'objectif.`inspect_binary_layout`ajoute une ligne à garder: "Une adresse liée n'est pas une adresse en temps d'exécution". Une charge exécutable indépendante de la position à une base choisie par le noyau, et le fichier de compensation est la seule coordonnée stable.

Les en-têtes de programme expliquent également pourquoi un exécutable dérivé est toujours en cours d'exécution. Un lien peut supprimer entièrement la table d'en-tête de section, et `e_shoff`Un outil qui passe par des sections ne voit rien.

## Les mêmes idées à Mach-O

Apple ne publie aucune spécification de prose pour Mach-O. Les en-têtes dans la source XNU sont les spécifications: `EXTERNAL_HEADERS/mach-o/loader.h`- Je suis là .`nlist.h`, et `reloc.h`- Le dispositif .`ledger-arm64-O1.o`est la même `ledger.c`construit avec `clang -target arm64-apple-macos -O1 -fno-asynchronous-unwind-tables -c`- Je suis désolé .

Une .`mach_header_64`est de 32 octets: `magic`- Je suis là .`cputype`- Je suis là .`cpusubtype`- Je suis là .`filetype`- Je suis là .`ncmds`- Je suis là .`sizeofcmds`- Je suis là .`flags`- Je suis là .`reserved`La magie est`0xfeedfacf`- Je suis là .`MH_MAGIC_64`- Je suis là .`cputype`est `0x0100000c`- Je suis là .`CPU_TYPE_ARM64`- Je suis là .`filetype`est de 1, `MH_OBJECT`- Je suis là .`ncmds`est de 5 et `sizeofcmds`est 616: les commandes de chargement occupent les 616 octets après l'en-tête, et chacune commence par `{cmd, cmdsize}`Le drapeau .`0x2000`est `MH_SUBSECTIONS_VIA_SYMBOLS`- Je suis désolé .

```text
Load commands:
  0 LC_SEGMENT_64                cmdsize=472 at 0x20
  1 LC_BUILD_VERSION             cmdsize=24 at 0x1f8
  2 LC_LINKER_OPTIMIZATION_HINT  cmdsize=16 at 0x210
  3 LC_SYMTAB                    cmdsize=24 at 0x220
  4 LC_DYSYMTAB                  cmdsize=80 at 0x238
```

`LC_SEGMENT_64`(0x19) est la réponse Mach-O aux deux tables ELF.`segment_command_64`Il a`segname`- Je suis là .`vmaddr`- Je suis là .`vmsize`- Je suis là .`fileoff`- Je suis là .`filesize`- Je suis là .`maxprot`- Je suis là .`initprot`- Je suis là .`nsects`, et `flags`- Alors ...`nsects`enregistrements de type `section_64`, 80 octets chacun, suivez la même commande.`loader.h`commentaire sur `MH_OBJECT`Les sections sont "dans un seul segment non nommé sans encombrement de segment".`fileoff`- Je suis en train de vous dire.`filesize`456, et cinq sections.

```text
  Idx Section                    Addr   Size Offset Reloff Nreloc Type
    1 __TEXT,__text               0x0   0xd0    648   1104      6 S_REGULAR
    2 __DATA,__const             0xd0   0x28    856   1152      5 S_REGULAR
    3 __TEXT,__cstring           0xf8    0xa    896      0      0 S_CSTRING_LITERALS
    4 __TEXT,__const            0x108   0x40    912      0      0 S_REGULAR
    5 __LD,__compact_unwind     0x148   0x80    976   1192      4 S_REGULAR
```

Chaque section nomme son segment parent en `segname`, donc deux sections appelées`__const`sont distinctes parce qu'une est `__DATA,__const`et l' autre `__TEXT,__const`Les délocalisations sont par section à travers `reloff`et `nreloc`, où ELF a une séparation `.rela.X`section par cible.

Chacun d' eux .`relocation_info`est de 8 octets.`r_address`Un champ de 32 bits suit, rempli comme `r_symbolnum`(à 24 bits),`r_pcrel`(1) `r_length`(2) `r_extern`(1) et `r_type`Les six`__text`Les enregistrements sont trois paires de `ARM64_RELOC_PAGE21`et `ARM64_RELOC_PAGEOFF12`Un bras64`adrp`plus `add`fait ce que l' un x86-64 `lea`Il faut donc deux dossiers pour chaque référence.`__DATA,__const`Les enregistrements sont `ARM64_RELOC_UNSIGNED`avec `r_length`3, donc chacun patche un pointeur absolu de 8 bytes.

À -O1 la construction arm64 est maintenue `labels`comme des pointeurs simples où la construction x86-64 a fait une table relative.

`LC_SYMTAB`(0x2) tient `symoff`1240 ,`nsyms`17 ans.`stroff`1512, `strsize`176 et 1512 plus 176 est 1688, la taille du fichier.`nlist_64`est de 16 octets: `n_strx`- Je suis là .`n_type`- Je suis là .`n_sect`- Je suis là .`n_desc`- Je suis là .`n_value`- Je suis là .`n_type`C'est un peu un champ.`N_TYPE`(masque)`0x0e`) égale à `N_SECT`(le secteur de l'énergie)`0xe`) désigne la section dont l' ordinal à base de 1 est `n_sect`- Je suis là .`N_EXT`(le secteur de l'énergie)`0x01`) est un symbole externe.

`nlist_64`Il n'y a pas de champ de taille.`_percent_change`le lecteur prend l'adresse du symbole suivant dans la même section, `_grade`à`0x18`. Les symboles C portent un avant`_`ici, donc la fonction est `_percent_change`- Je suis désolé .

`LC_DYSYMTAB`(0xb) enregistre la même règle locale-première que l' ELF `sh_info`Le numéro de la liste:`ilocalsym`0, `nlocalsym`13 ans.`iextdefsym`13 ans.`nextdefsym`4e.`LC_BUILD_VERSION`(0x32) enregistrements `platform`1 (macOS), `minos`26.0 et `sdk`26,2 comme emballé `XXXX.YY.ZZ`La commande est la raison pour laquelle le fichier arm64 est engagé en octets.

Le lecteur `compare_formats`met les deux fichiers côte à côte:

| Idea | ELF, `ledger-x86_64-O1.o` | Mach-O, `ledger-arm64-O1.o` |
|---|---|---|
| File identity | `e_ident` `7f 45 4c 46`, class 64, data LSB | `magic` `0xfeedfacf` |
| Machine | `e_machine` 62, `EM_X86_64` | `cputype` `0x100000c`, `CPU_TYPE_ARM64` |
| File kind | `e_type` 1, `ET_REL` | `filetype` 1, `MH_OBJECT` |
| Where the tables are | `e_shoff` `0x428`, 11 section headers | 5 load commands, 616 bytes after the header |
| The code | `.text` at `0x40`, size `0x9f`, flags `AX` | `__TEXT,__text` at 648, size `0xd0`, in the unnamed segment |
| Symbol names | `.symtab` (section 10), names in `.strtab` through `sh_link` 1 | `LC_SYMTAB` `symoff` 1240, `nsyms` 17, names at `stroff` 1512 |
| Symbol size | `st_size`: `percent_change` is 16 bytes | No field: next symbol minus this one, 24 bytes |
| Relocations | `.rela.text`, 3 records, `R_X86_64_PC32` | `__text` `reloff` 1104, 6 records, `PAGE21` and `PAGEOFF12` |
| Function bytes | `percent_change`: 16 bytes at `0x40` | `_percent_change`: 24 bytes at `0x288` |

réa lit Mach-O à travers `inspect_macho`, qui se déroule `otool -h -l`- Je suis là .`nm -gjU`- Je suis là .`dyld_info -imports -exports`, et `dwarfdump --uuid`à travers `xcrun`Votre lecteur n'en a pas besoin et il lit la même chose.`symoff`et `stroff`Ça ...`otool -l`Les empreintes.`open_binary`L'outil de réa détecte ELF, MZ, PE et Mach-O mince ou universel à partir des octets magiques.`e_machine`C'est les 20 premiers octets de ce que vous avez construit ici. Pour un fichier sans en-tête, vous donnez`format: raw-binary`Le fournisseur de Ghidra refuse une binaire universelle et demande une tranche éclairée.`0xcafebabe`- Je suis désolé .

## Faites-le

Exécutez le lecteur bibliothécaire standard:

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/main.py
```

La démo est là .`ledger-x86_64-O1.o`et imprime son en-tête, sections, symboles, relocations, et la carte de décalage de fichier. Il imprime ensuite les quatre chaînes de fonction en octets, lit `ledger-arm64-O1.o`, et imprime la table de comparaison.`tiny-x86_64`- Les lignes attendues:

```text
percent_change: 16 bytes at file offset 0x40: 85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3
grade: 46 bytes at file offset 0x50: 48 63 c7 48 69 c0 67 66 66 66 ...
entry 0x4000b0 is file offset 0xb0 through PT_LOAD
```

Passer un chemin pour lire un fichier: `python3 code/main.py code/fixtures/ledger-x86_64-O0.o`Le lecteur choisit ELF ou Mach-O parmi les quatre premiers octets.

Lire `code/main.py`dans cet ordre:

1. `ElfFile._read_header()`Vérifie la magie, la classe, l'ordre des octets et les tailles d'entrée, puis déballe l'en-tête.
2. `ElfFile._read_sections()`et `read_cstring()`Lisez la table d' en-tête de section et résolvez les noms à travers `e_shstrndx`- Je suis désolé .
3. `ElfFile._read_symbols()`Les divisions`st_info`, résolve les noms à travers `sh_link`, et garde lisibles les indices de sections réservées.
4. `ElfFile._read_relocations()`Il marche à chaque fois .`SHT_RELA`section, partage `r_info`, et les types de noms du tableau 4.9.
5. `ElfFile.function_span()`et `function_bytes()`transformer un symbole en une gamme de fichiers et les octets qui y sont.
6. `ElfFile.file_offset_map()`classe chaque structure par compensation et remplit l'espace inactif.
7. `ElfFile._read_program_headers()`et `address_to_file_offset()`- Les exécutables.
8. `MachOFile`Il lit l'en-tête, les commandes de charge,`LC_SEGMENT_64`avec ses sections, `LC_SYMTAB`- Je suis là .`LC_BUILD_VERSION`, et les délocalisations par section.
9. `compare_formats()`nomme la même idée dans les deux fichiers avec les valeurs lues de chacun.

## Utilisez-le

Importer le lecteur des leçons ultérieures avec `from main import ElfFile`après avoir ajouté la leçon `code`répertoire à `sys.path`La leçon 33 demande que vous le fassiez .`function_bytes`et `relocations_for(".text")`La leçon 35 enregistre sa sortie comme preuve observée. La leçon 36 lui demande de masquer les compensations de relocalisation avant une comparaison en octets.

Compilez votre propre C en un objet avec le même `clang -target x86_64-unknown-linux-gnu -c`commander et exécuter le lecteur dessus. Ajouter un nom de relocation seulement lorsque le tableau 4.9 x86-64 psABI le répertorie. Ajouter un type de section seulement lorsque le gABI ou la chaîne d'outils producteur le documentent. Lorsque le lecteur rencontre une valeur pour laquelle il n'a pas de nom, il imprime le nombre en hexadecimal plutôt que de deviner.

## Laboratoire interactif

### Laboratoire A: l'en-tête à la main

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch/code
python3 -q
```

Je vais courir .

```python
import struct
from main import ElfFile, FIXTURES
data = (FIXTURES / "ledger-x86_64-O1.o").read_bytes()
data[:4], data[4], data[5]
struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
elf = ElfFile(data)
elf.header.e_shoff, elf.header.e_shnum, elf.sections[elf.header.e_shstrndx].name
```

Les tupiers montrent`e_type`1 , `e_machine`62, et `e_shoff`1064, qui est `0x428`La dernière ligne montre que la section 1`.strtab`, détient les noms des sections.

### Laboratoire B: le tableau de valeur derrière le commutateur

```python
rodata = elf.section_bytes(elf.section(".rodata"))
[int.from_bytes(rodata[i : i + 4], "little") for i in range(0x14, 0x28, 4)]
[int.from_bytes(rodata[i : i + 8], "little") for i in range(0x28, 0x50, 8)]
[r for r in elf.relocations_for(".text") if r.symbol == ".rodata"]
```

Les deux tables sont lues .`1, 2, 3, 4, 4`- Vérifiez leur résistance .`ledger.c`Le numéro de la liste:`grade(65)`calculs `65 / 10 - 6 = 0`et lit l'entrée 0, qui est 1, la valeur de `case 6`- Je suis là .`grade(100)`L'entrée 4 est 4, la valeur partagée par `case 10`et `case 9`- Je suis désolé .

### Laboratoire C: trois constructions d'une seule source

```python
for level in ("O0", "O1", "O2"):
    obj = ElfFile.from_path(FIXTURES / f"ledger-x86_64-{level}.o")
    print(level, [s.name for s in obj.sections if s.name in (".data", ".rodata")])
    print(level, sorted({r.type_name for r in obj.relocations}))
    print(level, {s.name: s.st_size for s in obj.symbols if s.type == "FUNC"})
```

L' objet -O0 a une `.data`Les objets -O1 et -O2 n'ont pas de`.data`et un type. `total`est de 66 octets à -O0, 31 à -O1 et 140 à -O2.

### Laboratoire D: la même fonction dans Mach-O

```python
from main import MachOFile
macho = MachOFile.from_path(FIXTURES / "ledger-arm64-O1.o")
[(c.name, c.cmdsize) for c in macho.load_commands]
macho.function_span("_percent_change")
macho.function_bytes("_percent_change")[:4].hex()
```

La durée est de 24 octets à l'offset de fichier 648. Les quatre premiers octets sont `a0000034`, la forme de l' instruction en petit-en-tête `0x340000a0`, qui `llvm-objdump -d`des empreintes comme `cbz w0, 0x14`C' est ça .`before == 0`La même test que le code de la construction x86-64`85 ff 74 09`- Je suis désolé .

## Laboratoire de pratique

Extension du lecteur à un programme lié. sur une machine x86-64 Linux, construire `clang -O1 -shared -fPIC code/fixtures/ledger.c -o libledger.so`et `clang -O1 code/fixtures/ledger.c -o ledger_main`avec un `main`vous ajoutez, puis lancez le lecteur sur les deux.

Les exigences:

- Lire `.dynsym`avec le même `Elf64_Sym`Le plan, depuis `SHT_DYNSYM`est 11 et le record est inchangé.
- Lire `.dynamic`comme `Elf64_Dyn`enregistrements de `d_tag`et `d_val`, et nom `DT_NEEDED`- Je suis là .`DT_STRTAB`- Je suis là .`DT_SYMTAB`- Je suis là .`DT_RELA`, et `DT_JMPREL`à partir de la page "Séction dynamique" du gABI.
- Lisez les symboles non définis avec `st_shndx` `SHN_UNDEF`Les importations et les`R_X86_64_JUMP_SLOT`enregistrements dans `.rela.plt`comme les sites d'appels qui les atteignent.
- - Je vais vous faire une carte .`PT_LOAD`segment aux sections qu'il contient, en utilisant la règle gABI selon laquelle "un segment de fichier objet contient une ou plusieurs sections".
- Prouvez avec `address_to_file_offset`que les octets de chaque fonction définie dans l'objet partagé sont les octets `function_bytes`Il revient.
- Capture `llvm-objdump -h -t -r -p`et `llvm-objdump -T`texte pour les deux fichiers dans `code/fixtures/`et comparer le lecteur à eux dans des tests.

Le succès signifie que la liste d'importation, la carte de segment et chaque gamme de fonctions sont en accord avec la sortie de l'outil capturée, sans exécution d'outil pendant les tests.

## Leçon d'artisanat

Cette leçon produit `outputs/skill-program-anatomy-elf-from-scratch.md`. Utilisez-le avec Claude, Cursor, Codex, OpenClaw, Hermes ou tout agent qui lit SKILL.md. La compétence transforme un objet compilé ou exécutable que l'utilisateur a intégré dans un rapport de mise en page. Le rapport donne la gamme de fichiers de chaque structure, symbole et relocalisation, vérifiée contre la sortie de l'outil capturé.

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La suite prouve:

- Chaque section, nom, taille, adresse et type correspondent à la capture.`llvm-objdump -h`texte pour les trois objets x86-64 et le minuscule exécutable
- chaque valeur, liaison, type, section, taille et nom du symbole correspondent à ce qui a été capturé `llvm-objdump -t`texte
- Chaque délocalisation de compensation, nom de type, symbole et addendum correspondent à ceux capturés `llvm-objdump -r`texte, y compris `R_X86_64_PLT32`et `R_X86_64_64`à -O0
- les deux chefs de programme de `tiny-x86_64`- Je ne suis pas d'accord avec les prisonniers .`llvm-objdump -p`le texte, et les objets n'ont pas
- `percent_change`est de 16 octets à l' offset de fichier `0x40`avec les octets attendus, et chaque fonction se trouve à l'intérieur `.text`
- la carte de compensation de fichiers couvre chaque octet de chaque fixture une fois, avec les trois espaces d'alignement de l'objet -O1 à leur place
- le point d'entrée de `tiny-x86_64`cartes à déposer `0xb0`à travers `PT_LOAD`, et un objet relocalisable refuse la cartographie d' adresse
- l'en-tête Mach-O, chaque commande de charge, le segment, les cinq sections,`LC_SYMTAB`, et `LC_BUILD_VERSION`- Je ne suis pas d'accord avec les prisonniers .`--private-headers`texte
- Les symboles Mach-O et les 15 déplacements correspondent aux captifs `llvm-objdump`le texte et `_percent_change`est de 24 octets à 648
- une mauvaise magie, une classe de 32 bits, un fichier big-endian, une table tronquée, un binaire universel, et un fichier source C tout augmenter.`FormatError`
- `build_tiny_elf.py`reproduit les engagements `tiny-x86_64`octet par octet
- le module de chiffres `site/figures-re-32.js`contient les octets exacts de `ledger-x86_64-O1.o`

## Mode de défaillance de la production

| Failure | What the weak reading reports | What the reader must prove |
|---|---|---|
| `st_value` used as a file offset | "`percent_change` is at 0" | Offset is `sh_offset + st_value - sh_addr`, here `0x40` |
| Padding counted as code | "`grade` is 48 bytes" | `st_size` is 46 and the `66 90` after it belongs to no symbol |
| Section names read from the wrong table | Garbage names or an exception | `e_shstrndx` names the table, here section 1 |
| `sh_size` trusted for `SHT_NOBITS` | A read past the end of the file | `.bss` has no file bytes and the map skips it |
| Zero bytes read as the operand | "`lea 0(%rip)`" | The `.rela.text` record at `0x36` names `.rodata+0x10` |
| Addend read as the target offset | "the table starts at `0x10`" | `S + A - P` with the instruction end gives `0x14` |
| Sections walked on a stripped executable | "the file has no code" | `PT_LOAD` maps every address the loader uses |
| A linked address treated as a runtime address | A wrong offset under a random base | The file offset from `PT_LOAD` is the stable coordinate |
| A Mach-O section looked up by `sectname` alone | Two `__const` sections collapse into one | `segname,sectname` is the name |
| A size field expected in `nlist_64` | An exception or zero bytes | The next symbol in the same section ends the function |
| A universal binary read as one Mach-O | "magic `0xcafebabe` is unknown" | Refuse it and ask for a thinned slice |
| The arm64 object rebuilt on another Mac | "the fixture changed" | `LC_BUILD_VERSION` records the SDK, so the committed bytes are the fixture |

## Règlement d'exploitation

Lisez l'en-tête en qui vous avez confiance, suivez les dérivés qu'il contient et nommez chaque champ de la spécification qui le définit. Gardez l'outil de sortie que vous avez comparé comme un fichier engagé, de sorte que la comparaison fonctionne sans l'outil. Quand une valeur n'a pas de nom dans la spécification, imprimez le numéro. Le lecteur que vous pouvez vérifier octet par octet est celui sur lequel chaque leçon ultérieure s'appuie.

## Pour en savoir plus

- [System V gABI, chapter 4: Object Files](https://www.sco.com/developers/gabi/latest/ch4.intro.html)et le [ELF Header](https://www.sco.com/developers/gabi/latest/ch4.eheader.html)- Je suis là .[Sections](https://www.sco.com/developers/gabi/latest/ch4.sheader.html)- Je suis là .[Symbol Table](https://www.sco.com/developers/gabi/latest/ch4.symtab.html), et [Relocation](https://www.sco.com/developers/gabi/latest/ch4.reloc.html)pages
- [System V gABI, chapter 5: Program Header](https://www.sco.com/developers/gabi/latest/ch5.pheader.html)et [Dynamic Linking](https://www.sco.com/developers/gabi/latest/ch5.dynamic.html)
- [System V ABI, AMD64 Architecture Processor Supplement](https://gitlab.com/x86-psABIs/x86-64-ABI), Tableau 4.9 Types de déménagement
- [Apple XNU `loader.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h)- Je suis là .[`nlist.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/nlist.h), et [`reloc.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/reloc.h)
- [17 U.S.C. 1201](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title17-section1201&num=0&edition=prelim)et [Directive 2009/24/EC](https://eur-lex.europa.eu/eli/dir/2009/24/oj/eng)
- [Sega v. Accolade, 977 F.2d 1510](https://law.resource.org/pub/us/case/reporter/F2/977/977.F2d.1510.92-15655.html)et [Sony v. Connectix, 203 F.3d 596](https://law.resource.org/pub/us/case/reporter/F3/203/203.F3d.596.99-15852.html)
- [rea 6.5.0 `docs/binary-diagnostics.md`](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/binary-diagnostics.md)et [`docs/ghidra-provider.md`](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/ghidra-provider.md)
