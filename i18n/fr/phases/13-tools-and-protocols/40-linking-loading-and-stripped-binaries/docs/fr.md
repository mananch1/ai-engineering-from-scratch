# Les données binaires liées, chargées et dérivées: PLT, GOT, importations, fixups et identification des fonctions

> Un programme lié appelle des fonctions qu'il ne contient pas. Cette leçon lit les tables que l'éditeur de liens a écrites pour ces appels dans trois formats, suit un appel à la fente que le chargement remplit, et trouve les fonctions d'une construction après que ses noms ont disparu.

**Type:** Build
**Languages:** Python, C, x86-64 and arm64 assembly
**Prerequisites:** Phase 13 · 32 (ELF from scratch), 33 (disassembly and control flow), 38 (calling conventions)
**Time:** ~160 minutes

## Objectifs d'apprentissage

- Décidez à partir d'un enregistrement de relocalisation et des tableaux de symboles de ce que l'éditeur de lien en fait: une distance fixe, une entrée PLT avec un point GOT ou une relocalisation dynamique.
- Lisez la section dynamique, le PLT et le GOT d'un ELF exécutable indépendant de position et suivez un appel à travers eux avec les formules psABI.
- Lire un objet COFF (20 octets en-tête de fichier, 40 octets en-têtes de section, 18 octets symboles, 10 octets relocations) et les annuaires d'importation, d'exportation et de relocalisation de base d'une image PE32+.
- Décodez une chaîne de fixation enchaînée Mach-O mot par mot et partagez chaque mot contre`dyld_info`- Je suis désolé .
- Déclarer ce que `strip`supprime et ce qui survit dans chaque format, et exécute le programme dépouillé pour prouver qu'il fonctionne toujours.
- Récupérer la fonction démarre d' une construction arm64 dépouillée avec `LC_FUNCTION_STARTS`, les prologues scannés, et les cibles d'appel, marquer chaque méthode contre la table des symboles du jumeau non dérivé, et nommer les fonctions avec une signature en forme de FLIRT.

## Le problème

Vous avez fait le lien .`main.c`contre `libledger.dylib`et a demandé à un agent quelles fonctions `main`Les appels à `0x1000005a0`L' agent a démonté l' instruction et lu .`bl 0x100000694`Cette adresse contient 12 octets:`adrp x16`- Je suis là .`ldr x16, [x16, #0x38]`- Je suis là .`br x16`L' agent a suivi la charge jusqu' à`0x100004038`, lis les huit octets `00 00 00 00 00 00 10 80`, et a signalé un appel à travers un pointeur corrompu dans l'espace du noyau.

Puis vous avez dépouillé la construction statique du même programme et demandé combien de fonctions il a.`nm`, a vu un nom,`__mh_execute_header`, et a répondu à un.

Les deux réponses lisent le fichier comme si c'était le processus en cours.`0x100004038`est un fixup enchaîné. bit 63 dit bind, 12 bits disent où est le prochain fixup, et les bits bas disent quelle importation à lier. dyld remplace le mot par l'adresse de`_puts`avant `main`Le fichier dérivé a perdu sa table de symboles et est conservé.`LC_FUNCTION_STARTS`, qui énumère les cinq fonctions qui commencent en neuf octets.

Cette leçon construit cinq lecteurs en Python bibliothèque standard. Un lecteur COFF pour nos propres objets et un lecteur PE32+ pour les annuaires d'importation, d'exportation et de relocalisation de base. Un lecteur Mach-O pour les commandes de chargement, les étiquettes, la table des symboles indirects, la chaîne de fixation enchaînée et le tri d'exportation, vérifié mot pour mot contre `dyld_info`. Un lecteur ELF pour la section dynamique, le PLT et le GOT, plus un planificateur qui indique ce qu'un lien PIE de nos objets créerait. Un chercheur de fonction pour une construction dérivé avec quatre méthodes et un score pour chacune. Un constructeur de signature en forme de FLIRT, avec des octets de relocalisation à cartes sauvages et un CRC16 de la queue.

Les objets ELF et COFF ont été compilés ici avec`clang -target`, et aucun éditeur de lien ELF ou PE n'est installé, de sorte que les deux images liées dans ces formats ont été écrites octet par octet à partir des spécifications par deux scripts Python dans le dossier fixations. `llvm-objdump`Ils n'ont pas été exécutés, et le texte dit ça là où ça compte.

```figure
re-40-call-through-three-formats
```

## Ce que fait l'éditeur de liens avec un déménagement

Leçon 32: Les dossiers de déménagement et leçon 33 appliquée `S + A - P`à une `lea`Un enregistrement nomme un endroit .`P`, un type, un symbole `S`, et un additif `A`. Le tableau 4.9 du x86-64 psABI donne une formule par type, avec `L`pour le lieu de l'entrée PLT du symbole, `G`pour son GOT compensation, `GOT`pour l'adresse GOT, et `B`Trois résultats couvrent nos objets.

L'éditeur de lien écrit la valeur dans le champ et l'enregistrement est perdu. `R_X86_64_PC32`La distance entre deux octets d'une image ne change jamais, quelle que soit l'adresse de charge.

L'éditeur de lien fait une entrée PLT et une fente GOT. `R_X86_64_PLT32`L'appel est reçu.`L + A - P`La distance entre les deux points, l'entrée passe par la fente.`R_X86_64_JUMP_SLOT`enregistrement dans `.rela.plt`indique au lien dynamique de remplir la fente.

L'éditeur de lien conserve un relocalisation pour le chargement. `R_X86_64_64`est une adresse absolue de huit octets, et une exécutable indépendante de position ne connaît pas son adresse avant qu'elle ne se charge.`R_X86_64_RELATIVE`avec formule `B + A`dans `.rela.dyn`- Je suis désolé .

`main-x86_64-O1.o`Il a 18 records: 14 en `.text`, 2 en `.data.rel.ro`, et 2 en `.rodata`- Je suis là .`plan_dynamic_link()`dans `code/main.py`les trois règles s'appliquent à eux en même temps que `ledger-x86_64-O1.o`- Le numéro de la liste:

| Record | Count | Decision | Why |
|---|---|---|---|
| `R_X86_64_PC32` to `.rodata`, `.L.str`, `fns`, the table in `.rodata` | 16 | in place | the symbol is defined in the inputs |
| `R_X86_64_PLT32` to `percent_change`, `grade`, `total`, `grade_label` | 4 | direct, no PLT | defined in `ledger-x86_64-O1.o`, so the link editor reduces PLT32 to PC32 |
| `R_X86_64_PLT32` to `puts` twice and `printf` twice | 4 | PLT entry and GOT slot | undefined, provided by a shared library |
| `R_X86_64_64` to `grade` in `fns[]` | 2 | `R_X86_64_RELATIVE` | an absolute field in a PIE |

Deux importations donnent une`.plt`de 48 octets (PLT0 plus deux entrées de 16), un `.got.plt`de 40 octets (trois espaces réservés plus deux), et un `.rela.plt`Les deux enregistrements relatifs donnent une`.rela.dyn`Ces chiffres proviennent des spécifications appliquées à nos dossiers.`tiny-dyn-x86_64`met en œuvre la partie PLT du présent plan pour les mêmes deux importations et ses dimensions correspondent au plan.

Le compilateur écrit `PLT32`Dans les deux cas, le rédacteur en chef de l'éditeur de liens décide.`PLT32`Dans un objet, il est donc écrit "un appel", et ne dit rien sur l'existence d'une entrée PLT.`R_X86_64_GOTPCRELX`et `R_X86_64_REX_GOTPCRELX`(le secteur de l'énergie)`G + GOT + A - P`) demander une place GOT sans PLT, qui `-fno-plt`produit et que l'éditeur de lien peut relaxer à une direct `lea`ou `call`lorsque le symbole est local (psABI Appendice B). `R_X86_64_32`et `R_X86_64_32S`Il n'y a pas de lien entre les deux objets.

## La section dynamique, le PLT et le GOT

Le chapitre 5 du gABI indique qu' un exécutable qui participe à des liaisons dynamiques a un `PT_INTERP`en-tête de programme, qui doit précéder tout segment chargable, et un `PT_DYNAMIC`, qui couvre le `.dynamic`La section est une série de `Elf64_Dyn`paires, une étiquette signée et une valeur, terminée par `DT_NULL`- Je suis désolé .

`tiny-dyn-x86_64`est de 5 184 octets écrits par `build_tiny_dyn_elf.py`La section dynamique de la gABI et de la psABI contient douze paires à la`0x1000`- Le numéro de la liste:

| Tag | Number | Value | Meaning |
|---|---|---|---|
| `DT_NEEDED` | 1 | `libc.so.6` | a string table offset, `0xb`, the library to load |
| `DT_HASH` | 4 | `0x1a0` | the SysV hash table |
| `DT_STRTAB` | 5 | `0x188` | `.dynstr` |
| `DT_SYMTAB` | 6 | `0x140` | `.dynsym`, 24 bytes per entry |
| `DT_STRSZ` | 10 | `0x15` | 21 bytes of strings |
| `DT_SYMENT` | 11 | `0x18` | 24 |
| `DT_PLTGOT` | 3 | `0x10c0` | `.got.plt` |
| `DT_PLTRELSZ` | 2 | `0x30` | two 24-byte records |
| `DT_PLTREL` | 20 | 7 | the records are `DT_RELA` |
| `DT_JMPREL` | 23 | `0x1b8` | `.rela.plt` |
| `DT_FLAGS_1` | `0x6ffffffb` | `0x08000000` | `DF_1_PIE`, a GNU tag |
| `DT_NULL` | 0 | 0 | end |

Le GABI dit pourquoi .`DT_JMPREL`est séparé: il "contient l'adresse des entrées de relocalisation associées uniquement au tableau de liaison de procédure. La séparation de ces entrées de relocalisation permet au lien dynamique de les ignorer lors de l'initialisation du processus, si la liaison paresseuse est activée". `DF_BIND_NOW`dans `DT_FLAGS`- Je suis là .`DF_1_NOW`dans `DT_FLAGS_1`ou `LD_BIND_NOW`dans l'environnement, les liens paresseux sont désactivés et le liant dynamique les traite au démarrage.

### PLT0 et l'entrée pour les couches

La section 5.2 de psABI donne les deux formes de la figure 5.2. Voici les octets de notre fichier:

```text
     1f0: ff 35 d2 0e 00 00    pushq  0xed2(%rip)      # 0x10c8 = GOT+8
     1f6: ff 25 d4 0e 00 00    jmpq   *0xed4(%rip)     # 0x10d0 = GOT+16
     1fc: 0f 1f 40 00          nopl   (%rax)

     200: ff 25 d2 0e 00 00    jmpq   *0xed2(%rip)     # 0x10d8 = GOT+24, the slot for puts
     206: 68 00 00 00 00       pushq  $0x0             # index 0 into DT_JMPREL
     20b: e9 e0 ff ff ff       jmp    0x1f0            # PLT0
```

Faites l'arithmétique une fois. Le déplacement relatif à la RIP est relatif à l'instruction suivante. `0x1f6 + 0xed2 = 0x10c8`, qui est `.got.plt + 8`- Je suis là .`0x1fc + 0xed4 = 0x10d0`, qui est `.got.plt + 16`- Je suis là .`0x206 + 0xed2 = 0x10d8`, la quatrième fente.`0x210 - 0x20 = 0x1f0`Chaque entrée est de 16 octets, donc l'entrée pour `exit`commence à`0x210`, pousse`$0x1`, et il lit `0x10e0`- Je suis désolé .

Le GOT se réserve trois espaces. psABI 5.2: l'entrée zéro contient l'adresse de `_DYNAMIC`, et "les entrées 1 et 2 dans la table globale de compensation sont également réservées".`.got.plt`Il lit`0x1000, 0, 0, 0x206, 0x216`. glibc remplit la zone 1 avec son `struct link_map`et la fente 2 avec `_dl_runtime_resolve`La fente 3 tient.`0x206`Dans le dossier, qui est le `pushq`à l' intérieur`puts@plt`. psABI: la zone "initialement ... contient l'adresse de l'instruction de poussée suivante".

Donc le premier appel est lancé .`call 0x200`- Je suis là .`jmpq *0x10d8`Les terres sur`pushq $0`- Je suis là .`jmp 0x1f0`pousse GOT+8 et saute à travers GOT+16 dans le résolveur, qui lit enregistrement 0 de `DT_JMPREL`- Je suis là .`R_X86_64_JUMP_SLOT puts`- Je suis pas sûr .`puts`En haut .`libc.so.6`, écrit son adresse dans `0x10d8`Le second appel est lancé.`call`- Je suis là .`jmpq *0x10d8`Drepper écrit que "les frais généraux pour tous les autres que le premier appel sont donc 'uniquement' un seul saut indirect".`%r11`explique un choix de registre dans la leçon 38: c'est un registre à gratter, donc "le code dans le PLT n'a pas besoin de déverser de registre".

Le site d' appel en `.text`Il lit`e8 d4 ff ff ff`à`0x227`, et `0x22c - 0x2c = 0x200`- Je suis là .`llvm-objdump`des empreintes`callq 0x200 <puts@plt>`pour elle et nomme l'entrée de la `.rela.plt`- Je ne sais pas.`plt_entries()`Nomme aussi.

### Ce que le fichier ne contient pas

Le dossier n' a pas de`PT_GNU_RELRO`. Jakub Jelinek a proposé à RELRO de la liste des services binaires en 2004-01-06: déplacer les sections écrites uniquement lors du déménagement au début du segment écrit, afin que le lien dynamique puisse `mprotect`Le script par défaut de GNU ld met`.init_array`- Je suis là .`.fini_array`- Je suis là .`.data.rel.ro`- Je suis là .`.dynamic`, et `.got`à l'intérieur, et ne garde que les trois fentes réservées de `.got.plt`24 octets, car les autres fentes sont écrites avec laxité.`-z now`ensembles `DF_BIND_NOW`et déplace le tout `.got.plt`à l'intérieur, plein RELRO, parce que rien n'est écrit après le démarrage.`_dl_protect_relro`Les appels`mprotect(start, end - start, PROT_READ)`sur la gamme alignée sur la page. `-fno-plt`passe un pas plus loin du côté de l' compilateur: les appels passent `jmpq *name@GOTPCREL(%rip)`Le code de la communication est un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, un code de communication, etc.

Le fichier n'a pas non plus de sections de version.`libc.so.6`ajoute `.gnu.version`Un .`Elfxx_Half`par`.dynsym`l'entrée et `.gnu.version_r`, une `Verneed`pour chaque bibliothèque avec `Vernaux`enregistrements pour chaque nom de version. `printf@GLIBC_2.2.5`moyens: l'entrée de `printf`dans `.gnu.version`détient un indice, le `Vernaux`dont `vna_other`est égal à l'indice de`vna_name`"GLIBC_2.2.5", dans le champ `Verneed`dont `vn_file`est "libc.so.6". L'indice 0 est local, 1 est global et bit 15 (`VERSYM_HIDDEN`- Je suis là .`0x8000`) cache une version de l'éditeur de lien.

La table de hachage est un SysV `DT_HASH`avec un seau et une chaîne de trois entrées, 24 octets.`DT_HASH`ou `DT_GNU_HASH`Ce fichier n'exporte rien et n'a pas été exécuté ici, donc si glibc accepte une table à bouquet unique n'est pas vérifié.

### .dynsym, .symtab et strip

Le GABI fait`.dynsym`et `.dynstr`sections allouées et `.symtab`et `.strtab`Le lien dynamique lit la première paire et jamais la seconde. " La table de symboles normale n'est pas utilisée en temps d'exécution et il n'est donc pas nécessaire de la dépouiller d'un binary.

GNU `strip --strip-all`dégage `.symtab`- Je suis là .`.strtab`- Je suis là .`.debug_*`, et `.comment`Chaque section allouée reste:`.interp`- Je suis là .`.dynsym`- Je suis là .`.dynstr`- Je suis là .`.gnu.hash`- Je suis là .`.gnu.version*`- Je suis là .`.rela.dyn`- Je suis là .`.rela.plt`- Je suis là .`.plt`- Je suis là .`.text`- Je suis là .`.rodata`- Je suis là .`.eh_frame`- Je suis là .`.init_array`- Je suis là .`.dynamic`- Je suis là .`.got`- Je suis là .`.got.plt`- Je suis là .`.data`- Je suis là .`.bss`Les chefs de sections restent à moins d' ajouter`--strip-section-headers`(articles 2.41) ou de course `llvm-strip --strip-sections`- Je suis là .`--strip-debug`Il garde les symboles.`--only-keep-debug`écrit le fichier de débogage qui `--add-gnu-debuglink`Les noms de celui qui a été dépouillé, avec un CRC32 dans `.gnu_debuglink`, et `ld --build-id`écrit `.note.gnu.build-id`pour qu'un débogageur puisse les combiner.

## L'objet de la COFF

`clang -target x86_64-pc-windows-msvc -O1 -fno-asynchronous-unwind-tables -c`produit `main-x86_64.obj`et `ledger-x86_64.obj`. La page PE Format définit l'objet: une en-tête de fichier de 20 octets, sans en-tête optionnel, une table de section de 40 octets, une table de symbole de 18 octets d'enregistrements et des enregistrements de relocalisation de 10 octets par section.

`CoffObject`dans `code/main.py`L'en-tête du fichier dit machine.`0x8664`, 10 sections, 37 enregistrements de symboles,`SizeOfOptionalHeader`0. Les noms des tables de sections `.text`- Je suis là .`.data`- Je suis là .`.bss`, six sections appelées`.rdata`, et `/16`Un nom qui commence par`/`est un décompte décimal dans la table de chaînes, qui commence par sa propre taille de quatre bytes; `/16`résolue à `.llvm_addrsig`, c' est le nom `llvm-objdump -h`Des empreintes.

Les 37 enregistrements comprennent des enregistrements auxiliaires.Un symbole de section (classe de stockage 3) contient un symbole contenant la longueur de la section, le nombre de délocalisations, une somme de contrôle et une sélection COMDAT.`.rdata`Les sections ont la sélection 2, "tout", et contiennent un caractère littéral de chacune des chaînes sous un nom déformé MSVC: `??_C@_04JDDJEAHN@even?$AA@`Le rédacteur de liens garde une copie de chaque.`.file`avec le registre auxiliaire "main.c". `llvm-objdump -t`imprime ces éléments comme `[ 0]`- Je suis là .`AUX scnlen ...`- Je suis là .`[ 2]`, et ainsi de suite, en ignorant les indices auxiliaires, et le test compare nos indices et les paires de noms avec ce texte.

Les délocalisations: 14 `IMAGE_REL_AMD64_REL32`dans `.text`et 4 `IMAGE_REL_AMD64_ADDR64`dans `.rdata`- Je suis là .`REL32`est "l'adresse relative de 32 bits de la byte suivante le déplacement".`R_X86_64_PC32`avec le `-4`intégré: le dossier ELF contient un addendum `-4`parce que le dossier a un champ additif et que le dossier COFF n'en a pas. `ledger-x86_64.obj`Il y en a trois .`REL32`à`0x36`- Je suis là .`0x84`, et `0x93`, et la leçon 32 `ledger-x86_64-O1.o`Il y en a trois.`R_X86_64_PC32`Le code est le même et les formats ne diffèrent que par l'écriture des quatre. `ADDR64`est une adresse absolue de huit octets, le `labels`table de pointeurs de chaîne, et il devient une relocalisation de base dans l'image, comme `R_X86_64_64`devient `R_X86_64_RELATIVE`- Je suis désolé .

| | ELF `Elf64_Rela` | COFF relocation | Mach-O `relocation_info` |
|---|---|---|---|
| record size | 24 bytes | 10 bytes | 8 bytes |
| fields | offset, type and symbol in `r_info`, addend | offset, symbol index, type | address, then a 32-bit word with symbolnum:24, pcrel:1, length:2, extern:1, type:4 |
| addend | explicit | none, lives in the field | none, or an `ARM64_RELOC_ADDEND` record before the real one |
| the call to `puts` | `R_X86_64_PLT32 puts-0x4` | `IMAGE_REL_AMD64_REL32 puts` | `ARM64_RELOC_BRANCH26 _puts` |

## L'image de l'EP: importations, exportations et délocalisations de base

`tiny-x86_64.exe`est de 2 560 octets écrits par `build_tiny_pe.py`Il a le même travail que le petit ELF:`puts`et `exit`de `msvcrt.dll`Il exporte également un nom et une relocalisation de base, de sorte que les trois annuaires dont cette leçon a besoin sont tous présents. `pefile`2024.8.26 et `llvm-objdump -p`Les deux le lisent, et les deux captures sont des fixations.

La série MS-DOS commence par`MZ`et à compensation `0x3c`Il tient`e_lfanew`- Je suis là .`0x40`À la .`0x40`la signature `PE\0\0`- Puis l' en-tête du fichier COFF: machine `0x8664`, 4 sections, `SizeOfOptionalHeader`240, caractéristiques `0x22`L'en-tête optionnel commence par le `Magic`champ, `0x20b`, ce qui signifie PE32+. `AddressOfEntryPoint`est `0x1000`- Je suis là .`ImageBase`est `0x140000000`- Je suis là .`SectionAlignment` `0x1000`- Je suis là .`FileAlignment` `0x200`- Je suis là .`SizeOfImage` `0x5000`- Je suis là .`Subsystem`3 (console). `DllCharacteristics` `0x8160`décode à `HIGH_ENTROPY_VA 0x0020`- Je suis là .`DYNAMIC_BASE 0x0040`- Je suis là .`NX_COMPAT 0x0100`, et `TERMINAL_SERVER_AWARE 0x8000`- 16 répertoires de données de RVA et de taille.`0x2080`taille `0x4d`, Importé à `0x2000`taille `0x28`, Relocation de base à `0x4000`taille `0xc`, IAT à`0x2040`taille `0x18`- Je suis désolé .

Les adresses d'une image sont RVAs, des compensations de la base de charge. La table de section les cartographient en des compensations de fichiers: `.text`RVA `0x1000`dans le dossier `0x200`- Je suis là .`.rdata`RVA `0x2000`à`0x400`- Je suis là .`.data`RVA `0x3000`à`0x600`- Je suis là .`.reloc`RVA `0x4000`à`0x800`- Je suis là .`PeImage.rva_to_offset()`C'est cette table.

### Le répertoire des importations

À RVA `0x2000`Il y a une liste de descripteurs d'importation de 20 octets terminés par un descripteur zéro.`0x2028`, timestamp 0, chaîne d'expéditeur 0, nom `0x2068`, Tableau d' adresses d' importation `0x2040`. Le nom RVA pointe vers "msvcrt.dll". Le tableau de recherche contient des entrées de 64 bits terminées par zéro. Le jeu de bits 63 signifie importation par ordinal, avec l'ordinal en bits 15 à 0.`0x2058`(indice 0, "puts") et `0x2060`(indice 0, "sortie").

La spécification indique que l'IAT est "identique au contenu de la table de recherche d'importation jusqu'à ce que l'image soit liée".`0x2040`possède les mêmes deux RVAs, et le test vérifie que les 24 octets à `0x2028`égal aux 24 octets à `0x2040`. " Pendant la liaison, les entrées du tableau d'adresses d'importation sont écrasées avec les adresses de 32 bits (pour PE32) ou de 64 bits (pour PE32+) des symboles qui sont importés. " Le chargement Windows le fait avant l'exécution du point d'entrée. Il n'y a pas de liaison paresseuse et pas de tronçon. Le code atteint directement la fente:

```text
    100b: ff 15 2f 10 00 00    call qword ptr [rip + 0x102f]   # 0x1011 + 0x102f = 0x2040, the IAT slot of puts
    1013: ff 15 2f 10 00 00    call qword ptr [rip + 0x102f]   # 0x1019 + 0x102f = 0x2048, the IAT slot of exit
```

`FF /2`Les deux instructions ont les mêmes octets parce que chacune se trouve huit octets après l'autre et lit une fente huit octets après l'autre. `iat_calls()`analyse la section de code pour `ff 15`, ajoute six et le déplacement, et nomme la fente du répertoire d'importation.

### Le répertoire des exportations

À RVA `0x2080`le tableau de l'annuaire d'exportation de 40 octets: drapeaux, timestamp, version, nom RVA indiquant "tiny-x86_64.exe", base ordinaire 1, entrée de table d'adresse 1, point de nom.`0x20a8`détient une RVA, `0x1000`. Le tableau de référence des noms d' exportation à `0x20ac`contient un RVA à la chaîne "tiny_entry".`0x20b0`possède un indice impartial de 16 bits, 0. Nom `k`cartes pour adresser l'entrée `ordinals[k]`, et l' ordinal exporté est `ordinals[k] + ordinal base`. La spécification ajoute une règle que nous n' utilisons pas: une entrée d'adresse qui pointe à l'intérieur de la section d'exportation est une chaîne d'expéditeur telle que `MYDLL.expfunc`, et le nom du tableau de pointeur est "ordonné léxagiquement pour permettre des recherches binaires".

### Relocations de bases

`.reloc`Chaque bloc commence par une page RVA et une taille de bloc, puis des entrées de 16 bits avec le type dans les quatre bits supérieurs et une page compensée dans la douzième inférieure.`0x3000`, taille 12, entrée `0xa000`, type 10, `IMAGE_REL_BASED_DIR64`, compensation 0, et une entrée de rembourrage de type 0, `IMAGE_REL_BASED_ABSOLUTE`Les huit octets à RVA`0x3000`- Je vous en prie .`0x1400020d0`, la base d'image plus le RVA de la chaîne " liée à la main ". Si le chargement place l'image ailleurs, il ajoute la différence à ce champ. La spécification: " Les fichiers d'image ne contiennent pas de relocalisations COFF ... Une image contient des informations de relocalisation sous la forme de relocalisations de base dans la section .reloc (sauf si l'image a l'attribut IMAGE_FILE_RELOCS_STRIPPED). " `/DYNAMICBASE`est activé par défaut dans le lien.exe actuel, donc les images portent `.reloc`et `DYNAMIC_BASE`- Je suis désolé .

Une image PE n'a pas de table de symboles à déraciner.`PointerToSymbolTable`"devaient être zéro pour une image parce que les informations de débogage COFF sont dépassées", et les noms vivent dans un PDB séparé. Ce qui survit dans chaque image est le répertoire d'importation, le répertoire d'exportation, et, pour le code de 64 bits, `.pdata`, dont `RUNTIME_FUNCTION`Les entrées nomment le début et la fin de chaque fonction qui alloue la pile ou appelle une autre fonction.

## L'image Mach-O: commandes de chargement, couches et tableau de symboles indirects

`ledger_main`est l'exécutable arm64 lié. `otool -h`Imprimez 19 commandes de chargement et des drapeaux `0x00200085`- Je suis là .`MH_NOUNDEFS 0x1`- Je suis là .`MH_DYLDLINK 0x4`- Je suis là .`MH_TWOLEVEL 0x80`, et `MH_PIE 0x200000`Les 19 commandes sont quatre.`LC_SEGMENT_64`(le secteur de l'énergie)`__PAGEZERO`- Je suis là .`__TEXT`- Je suis là .`__DATA_CONST`- Je suis là .`__LINKEDIT`), deux `LC_LOAD_DYLIB`(le secteur de l'énergie)`@rpath/libledger.dylib`Alors ...`/usr/lib/libSystem.B.dylib`), et un pour chacun `LC_DYLD_CHAINED_FIXUPS`- Je suis là .`LC_DYLD_EXPORTS_TRIE`- Je suis là .`LC_SYMTAB`- Je suis là .`LC_DYSYMTAB`- Je suis là .`LC_LOAD_DYLINKER`- Je suis là .`LC_UUID`- Je suis là .`LC_BUILD_VERSION`- Je suis là .`LC_SOURCE_VERSION`- Je suis là .`LC_MAIN`- Je suis là .`LC_RPATH`- Je suis là .`LC_FUNCTION_STARTS`- Je suis là .`LC_DATA_IN_CODE`, et `LC_CODE_SIGNATURE`- Je suis là .`MachoFile.command_names()`renvoie la même liste dans l'ordre suivant `otool -l`- Je suis désolé .

`__TEXT`commence à`0x100000000`- Je suis là .`__text`est de 256 octets à `0x100000558`- Je suis là .`main`Et rien d'autre, parce que les quatre fonctions du registre vivent dans le dylib.`__stubs`est de 84 octets à `0x100000658`, sept épaules de douze octets.`__DATA_CONST`commence à`0x100004000`avec `__got`, 64 octets, huit fentes de pointeurs de type section `S_NON_LAZY_SYMBOL_POINTERS`, et `__const`, 32 octets, le `fns[]`et `names[]`Tableaux de `main.c`- Je suis désolé .

### Un bâton à la main

Le bâton pour `_puts`est à `0x100000694`- Le numéro de la liste:

```text
    30 00 00 90    adrp x16, 0x100004000
    10 1e 40 f9    ldr  x16, [x16, #0x38]
    00 02 1f d6    br   x16
```

`0x90000030`est `adrp`Le registre est le plus bas de cinq bits, 16.`immlo`est de 30 bits à 29, 0. `immhi`est de 23 bits à 5, 1. La valeur est `immhi << 2 | immlo = 4`, dans des pages de 4 096 octets, donc la page cible est la page de l'instruction, `0x100000000`, plus `0x4000`- Je suis là .`0xf9401e10`est `ldr`avec un décalage de 12 bits non signé égalé par 8: bits 21 à 10 sont 7, fois 8 est `0x38`, donc la charge est de lire `0x100004038`- Je suis là .`0xd61f0200`est `br x16`Le bâton de l'arm64 dans la source de l'ID64 est exactement la même instruction à trois.`StubAtom`- Je suis désolé .

Le nom de la fente vient de la table des symboles indirects. `LC_DYSYMTAB`Les points à 15 indices dans la table des symboles.`reserved1`où commence sa série d'indices et dans `reserved2`la taille de chaque tronc (12).`reserved1`et huit octets par fente. `__stubs`commence à l' indice 0 et `__got`à l'indice 7, alors cliquez sur les cartes 5 pour l'entrée de la table des symboles 10, `_puts`, et la tranche 7 de `__got`cartes de la même entrée. `otool -Iv`Il imprime les deux courants, et `indirect_table()`reproduit chaque rangée.

### Espace de noms à deux niveaux

`nm -m`des empreintes`(undefined) external _puts (from libSystem)`et `(undefined) external _grade (from libledger)`La bibliothèque provient du symbole de l'enregistrement lui-même.`nlist.h`définit `GET_LIBRARY_ORDINAL(n_desc)`comme bits 15 à 8 de `n_desc`, 1 pour la première .`LC_LOAD_DYLIB`, 2 pour le second, `0xfe`pour la recherche dynamique, `0xff`pour le plus performant. `ld(1)`: " Avec -two_levelnamespace (par défaut), le lien ne recherche que les dylibs de la ligne de commande pour les symboles et les enregistrements dans lesquels ils ont été trouvés. " `nm_lines()`rend les douze disques de `n_type`- Je suis là .`n_sect`- Je suis là .`n_desc`, et `n_value`, et l'essai compare sa ligne de sortie par ligne avec celle capturée `nm -m`- Rencontre avec `-Wl,-flat_namespace`et lire `otool -h`Pour voir le `MH_TWOLEVEL`Un peu disparaître.

## Fixups enchaînés, octet par octet

`LC_DYLD_CHAINED_FIXUPS`est une `linkedit_data_command`Le numéro de la liste:`dataoff 0x8000`- Je suis là .`datasize 208`Les 208 octets suivent .`fixup-chains.h`- Je suis désolé .

L' en-tête est composé de sept champs de 32 bits: `fixups_version 0`- Je suis là .`starts_offset 32`- Je suis là .`imports_offset 80`- Je suis là .`symbols_offset 112`- Je suis là .`imports_count 8`- Je suis là .`imports_format 1`(le secteur de l'énergie)`DYLD_CHAINED_IMPORT`), `symbols_format 0`(non comprimé). `llvm-objdump --macho --chained-fixups`Il imprime ces sept valeurs et le test les compare.

À l'offset 32, `dyld_chained_starts_in_image`Le numéro de la liste:`seg_count 4`et quatre compensations, une par segment, zéro pour les segments sans fixations.`__DATA_CONST`Il y en a un.`dyld_chained_starts_in_segment`Il lit`size 24`- Je suis là .`page_size 0x4000`- Je suis là .`pointer_format 6`(le secteur de l'énergie)`DYLD_CHAINED_PTR_64_OFFSET`, "objectif est vm compensé"), `segment_offset 0x4000`- Je suis là .`max_valid_pointer 0`- Je suis là .`page_count 1`, et une .`page_start`de 0. `DYLD_CHAINED_PTR_START_NONE`- Je suis là .`0xffff`, signifierait une page sans problèmes.

La chaîne commence à l' offset de fichier `0x4000 + 0`Chaque mot de 64 bits contient deux choses: ce qu'il faut écrire et où se trouve le mot suivant.

```text
  dyld_chained_ptr_64_bind    ordinal:24  addend:8  reserved:19  next:12  bind:1 = 1
  dyld_chained_ptr_64_rebase  target:36   high8:8   reserved:7   next:12  bind:1 = 0
```

Lisez le premier mot.`0x8010000000000000`- Le 63 est réglé, un lien.`next`est de 62 à 51 bits.`0x002`Le ordinal est 0, lisez le douzième mot à `0x4058`- Je suis là .`0x00000000000006db`- Un peu de 63 francs, un rebase.`next`est 0, la fin de la chaîne.`0x6db`, un vm de compensation, et la base d'image `0x100000000`Ça le rend .`0x1000006db`, la chaîne "odd" dans `__cstring`Le mot qui l'a précédé.`0x00100000000006d6`, les rébases `names[0]`à "même".

La marche se déroule en douze mots: huit se lient en `__got`avec des ordinaires de 0 à 7, deux liaisons en `__const`avec ordinal 0 (les deux entrées de `fns[]`point à `_grade`), et les deux contrebases de `names[]`- Je suis là .`chained_fixups()`Les résultats sont ensuite comparés avec les douze.`(offset, raw)`couples avec `dyld_info -fixup_chain_details`, qui imprime `raw: 0x8010000000000000 bind: (next: 002, bindOrdinal: 0x000000)`Pour la première fois.

Le tableau des importations à l' offset 80 contient huit entrées de 32 bits: `lib_ordinal:8`- Je suis là .`weak_import:1`- Je suis là .`name_offset:23`, avec des noms à l' offset 112.`_grade`- Je suis là .`_grade_label`- Je suis là .`_percent_change`- Je suis là .`_total`Les livres de la bibliothèque 1 et 4 à 7 sont `___stack_chk_fail`- Je suis là .`___stack_chk_guard`- Je suis là .`_printf`- Je suis là .`_puts`de la bibliothèque 2. `fixup_rows()`joins les deux tables dans les lignes `dyld_info -fixups`des empreintes, telles que `__DATA_CONST __got 0x100004038 bind libSystem/_puts`, et le test compare les douze.

Donc le "pointer corrompu" à`0x100004038`est importé 7. dyld marche cette chaîne une fois, à la charge, et écrit l'adresse de `_puts`Dans la fente. La session WWDC22 d'Apple l'a dit ainsi: "dans l'emplacement du pointeur de 64 bits dans DATA, certains bits contiennent le décalage vers l'emplacement de fixation suivant. Il y a aussi un petit message indiquant si la fixation est un bind ou un rebase. " Xcode 13 a fait du format le format par défaut pour un objectif de déploiement de macOS 12 ou iOS 15, et Xcode 13.3 l'a étendu à macOS 11 et iOS 13.4. Il n'y a pas de liaison paresseuse dans ce format. Chaque fente est fermée avant .`main`Il court.

### Le plan qu'il a remplacé

`ledger_main_nochain`est le même lien avec `-Wl,-no_fixup_chains`- Il a été .`LC_DYLD_INFO_ONLY`Il est également composé de trois sections dont le fichier enchaîné manque:`__TEXT,__stub_helper`(108 octets), une `__got`de seulement 16 octets de détention `___stack_chk_guard`et `dyld_stub_binder`, et `__DATA,__la_symbol_ptr`(56 octets, sept emplacements). `dyld_info -fixups`Liste de sept `lazy-bind`Les rangées et les sept basses qui pointent chaque fente paresseuse à son propre trampoline dans`__stub_helper`Le trampoline pousse un index et saute à`dyld_stub_binder`, qui résolve le nom et écrit la fente.`__la_symbol_ptr`est `.got.plt`- Je suis là .`__stub_helper`est PLT0 et le `pushq $index`, et `dyld_stub_binder`est `_dl_runtime_resolve`- Je suis là .`code/main.py`rapporte les compensations et les sections de ce fichier et ne décode pas les codes de liaison; le laboratoire de pratique en demande la demande.

### Les exportations

`LC_DYLD_EXPORTS_TRIE`chaque nœud est une taille de terminal uleb128, puis, si elle est non zéro, des drapeaux uleb128 et un offset uleb128, puis un octet de nombre d'enfants et, pour chaque enfant, un bord de chaîne et un offset de nœud uleb128. `exports_trie()`Il marche.`libledger.dylib`Il revient .`_percent_change 0x410`- Je suis là .`_grade 0x428`- Je suis là .`_total 0x464`- Je suis là .`_grade_label 0x498`, les quatre compensations `dyld_info -exports`Sur l'exécutable, il est retourné.`__mh_execute_header`- Je suis là .`_fns`, et `_main`- Je suis désolé .

## Quelle bande laisse

Les pommes de terre`strip(1)`dit que sans options sur un exécutable dynamiquement lié il se comporte comme `-u -r`: sauvegarder les symboles non définis et les symboles référencés dynamiquement. `-x`supprime les symboles locaux. `-N`supprime la table de symboles et la table de chaînes entières. " Pour les bibliothèques partagées dynamiques, le niveau maximal de dérippage est généralement -x ", car " Seuls les symboles globaux sont utilisés par le processus de liaison dynamique ".

Observé sur nos appareils.`strip ledger_main`Il a pris .`LC_SYMTAB`de 12 à 10 enregistrements et supprimé `_main`- Je suis là .`_fns`, et `_names`- les huit importations non définies restent, car il faut que leurs noms soient liés, et`__mh_execute_header`reste parce qu' il est `REFERENCED_DYNAMICALLY`Le dixième record est un coup de couteau .`N_OPT`nommé `radr://5614542`, que les outils d'Apple écrivent pour que la table ne soit jamais vide; `nm`Le programme dérivé imprime les mêmes trois lignes que l'original et sort 0. `LC_FUNCTION_STARTS`Il n'a qu'un début.`main`- Je suis désolé .

`strip libledger.dylib`sans aucune option, il est allé plus loin que d'habitude.`-x`Le numéro de la liste:`LC_SYMTAB`a 0 enregistrements et `nm`Le trie d'exportation énumère toujours les quatre fonctions aux mêmes compensations, parce que dyld se lie à travers le trie et jamais à travers.`LC_SYMTAB`- Je suis là .`LC_FUNCTION_STARTS`Le laboratoire C a encore une liste de quatre démarches.`main.c`contre le dylib dépouillé pour montrer que l'éditeur de lien est également servi par le tri.

| What | ELF after `--strip-all` | PE image | Mach-O after `strip` |
|---|---|---|---|
| static symbol table | gone | never present in an image | local and defined names gone, undefined imports stay |
| names the loader binds | `.dynsym` and `.dynstr` stay | import directory stays | imports stay in `LC_SYMTAB` and in the chained imports table |
| exported names | `.dynsym` stays | export directory stays | exports trie stays |
| function starts | `.eh_frame` FDEs, unless built without unwind tables | `.pdata` for non-leaf functions | `LC_FUNCTION_STARTS` stays |
| debug information | `.debug_*` gone | in the PDB, never in the image | in the dSYM, never in the image |
| section or load command headers | stay unless `--strip-section-headers` | stay | stay |

## Trouver des fonctions sans symbole

`static-arm64-O2`liens `main-arm64-O2.o`et `ledger-arm64-O2.o`en un exécutable avec cinq fonctions en `__text`- Je suis là .`nm -n`La vérité est la vérité:`_main`à`0x100000500`(244 octets), `_percent_change`à`0x1000005f4`(24), `_grade`à`0x10000060c`60), `_total`à`0x100000648`(248), `_grade_label`à`0x100000740`(72). `static-arm64-O2.stripped`n' a pas`N_SECT`symbole à l' intérieur `__text`- Je suis là .`FunctionFinder`Il y a quatre méthodes et `score()`Il compare chaque liste à la vérité.

### LC_FUNCTION_STARTS

Les commandes sont à neuf octets en `__LINKEDIT`Le numéro de la liste:`80 0a f4 01 18 3c f8 01 00`Le format est uleb128 deltas, le premier de la série `__TEXT`adresse. `80 0a`est `0x500`, donc le premier départ est `0x100000500`- Je suis là .`f4 01`est `0xf4`Je suis un homme .`0x1000005f4`- Je suis là .`18`donne`0x10000060c`- Je suis là .`3c`donne`0x100000648`- Je suis là .`f8 01`donne`0x100000740`, et `00`5 sur 5, pas de faux positifs, à la fin de la liste.`-O0`et à`-O2`L'éditeur de lien écrit cette table pour le détoxificateur et le reporter de crash, et `strip`Ghidra le lit dans son analyseur "La fonction Mach-O démarre".`dyld_info -function_starts`Il l'imprime et LIEF l'expose comme `function_starts`- Je suis désolé .

### Scans du prologue

Les modèles de départ de Ghidra AARCH64 commencent par `fd 7b bf a9`, qui est `stp x29, x30, [sp, #-0x10]!`Sur nos dossiers , il trouve 0 sur 5 à tous les niveaux .`sub sp, sp, #N`D' abord et ensuite .`stp x29, x30, [sp, #M]`Le formulaire de pré-indexation ne se produit jamais.`sub sp, sp, #imm`- Je suis là .`(word & 0xff0003ff) == 0xd10003ff`, trouve 5 sur 5 à `-O0`, où chaque fonction a un cadre, et 1 de 5 à `-O2`Où ?`percent_change`- Je suis là .`grade`- Je suis là .`total`, et `grade_label`sont des fonctions de feuilles sans cadre. `percent_change`commence par `cbz w0, ...`Un prologue de scan rapporte ce que le compilateur a choisi d'émettre, et la leçon 38 a montré combien souvent c'est rien.

Aucun scan n'a produit un faux positif ici, parce que le programme est petit.`55`de `push rbp`Il a touché d'autres instructions, c'est pourquoi les fichiers de modèle x86-64 de Ghidra se couplent en un post-pattern comme `55 48 89 e5`avec un modèle préalable tel que `ret`ou `nop`qui doit mettre fin à la fonction précédente.

### Appels à destination

Chaque .`bl`dans `__text`nomme une fonction démarrer, parce qu'une `bl`dans le milieu d'une fonction est quelque chose qu'un compilateur n'émite pas pour C. Le codage est `0x94000000`avec un commentaire de 26 bits signé.`bl`à`0x1000005a0`Il lit`3d 00 00 94`Le numéro de la liste:`0x3d`est 61, fois 4 est `0xf4`, donc la cible est `0x100000694`- Dans .`ledger_main`C' est le point de départ pour `_puts`Dans la construction statique,`call_targets()`Il recueille les cibles à l' intérieur .`__text`, dépose ceux qui arrivent .`__stubs`, et ajoute le `LC_MAIN`L'entrée, parce que rien ne vous appelle`main`Nucleus (Andriesse, Slowinska, Bos, 2017) construit sa détection de fonctions sur cette idée de code généré par le compilateur, et Andriesse et d'autres ont mesuré en 2016 que les limites de fonctions sont la partie la plus difficile du démontage, plus difficile que les limites d'instruction.

## Une signature en forme de flûte

Hex-Rays décrit FLIRT dans sa documentation: un modèle est "les 32 premiers octets d'une fonction où tous les octets variants sont marqués", les octets variants proviennent de fixups et de références externes, les modèles vivent dans un arbre, et au-delà du modèle "nous calculons le CRC16 des octets à partir de la position 33 jusqu'au premier octet variante" et stockons cette longueur. Deux fonctions ayant le même motif et CRC sont séparées par un byte de queue à une position où elles diffèrent, puis par les noms qu'elles font référence. Une collision que rien ne sépare est " résolue par l'intelligence naturelle ". Les fonctions de moins de quatre octets sans références externes sont exclues, et "le fichier de signature ne contient aucun octet des bibliothèques originales, à l'exception des noms des fonctions".

`signatures_from_object()`construit que de `ledger-arm64-O2.o`et `main-arm64-O2.o`La façon dont FLAIR la construit à partir des objets d'une bibliothèque.`relocation_info`l'enregistrement couvre un mot de quatre octets, et ces octets deviennent une variante.`.pat`la disposition de la ligne, documentée dans le parseur de rizin, est le motif, la longueur du CRC, la longueur du CRC16, la longueur de la fonction, `:0000`et le nom, et `^offset name`pour chaque nom de référence:

```text
A00000342800004B890C8052087D091B000DC01AC0035FD6 00 0000 0018 :0000 _percent_change
E8CC8C52C8CCAC72087C289B09FD7FD308FD62930801090B081900511F110071 04 9AAF 003C :0000 _grade ^0000 l_switch.table.grade ^0001 l_switch.table.grade
E8CC8C52C8CCAC72087C289B09FD7FD308FD62930801090B081900511F110071 04 9AAF 0048 :0000 _grade_label ^0000 l_switch.table.grade_label ^0001 l_switch.table.grade_label ^0002 _labels ^0003 _labels
```

`_percent_change`est de 24 octets, donc son schéma est la fonction entière et la longueur CRC est de 0. `_grade`est de 60 octets avec sa première relocalisation à 36 octets, donc le CRC couvre quatre octets.`.pat`Parser utilisée: polynôme `0x8408`, valeur initiale `0xffff`La valeur de la chaîne ASCII "123456789" est `0x6e90`, et les broches de test que.

`match_signatures()`teste chaque signature à chaque fonction de démarrage de la construction dérivé, le modèle d'abord, puis CRC.`-O0`Les cinq fonctions ont tous leurs noms et rien ne se heurte.`-O2`Les deux lignes se heurtent.`grade_label`en ligne`grade`, les deux fonctions partagent leurs 36 premiers octets et les quatre octets CRC.`0x10000060c`et à`0x100000740`La queue les sépare à 44 bytes .`_grade`Il a`20`et `_grade_label`Il a`28`, qui est `ldr w0, [x9, w8, uxtw #2]`contre `ldr x8, [x9, w8, uxtw #3]`Le rapport rapporte les collisions et le byte qui les a décidées.

Le test `test_object_bytes_equal_linked_bytes_outside_relocated_words`vérifie la prémisse: chaque octet non variable de chaque fonction objet équivaut au octet au même décalage dans l'exécutable lié, de sorte que l'éditeur de lien a modifié uniquement les mots déplacés. `_total`n'a pas de relocalisation et aucune variante en octets du tout.

### Ce que les outils font avec la même idée

L' IDA Pro 9.4 envoie des signatures FLIRT par processeur et le `sigmake`outil pour les construire; IDA Free couvre seulement x86 et x86-64 et est destiné à une utilisation non commerciale. Lumina est un serveur qui met des métadonnées de fonction par md5 de la fonction et saute des fonctions sous 32 octets. Ghidra 12.1.4 (Apache-2.0) dispose d'un ID de fonction: un hash complet sur les modes de mnemonics et d'adressage qui est " robuste contre les changements dus au lien " et un hash spécifique qui ajoute des opérands constants, notés par instruction correspondante, avec des fonctions parent et enfant comme coupe-liens; ses bases de données expédiées couvrent uniquement les bibliothèques Visual Studio, et vous remplissez votre propre à partir de vos propres binaires. Binary Ninja 6.0 Krypton (commercial, avec une édition gratuite) utilise WARP (Apache-2.0): la fonction GUID est une UUIDv5 sur les GUID de bloc de base, chacune une UUIDv5 sur les octets de bloc après avoir zéro "toutes instructions relocables" et la chute de NOP, avec appelé, appelant, et les fonctions adjacentes comme contraintes. BinDiff 8 (Apache-2.0) correspond aux fonctions entre deux binaires avec un ordre fixe de étapes, de la correspondance de hachage à travers les indices MD du graphique de flux à l'appel de correspondance de séquences, et pèse la similitude des bords, des blocs, des instructions et de l'indice MD du graphique de flux. Diaphora 3.4.3 (AGPL-3.0, IDA seulement) ajoute le produit des petits nombres premiers, où chaque carte mnemonique correspond à un nombre premier et une fonction est le produit, une empreinte digitale multiset sans ordre. Le radar2 6.2.4 (LGPL-3.0) est signé (`zg`- Je suis là .`z/`- Je suis là .`zb`) et se lit FLIRT `.sig`fichiers avec `zfs`; rizine 0.9.1 (LGPL-3.0) crée `.pat`ou `.sig`à partir d' une liste binaire ouverte avec `Fc`et s' applique à l' un avec `Fs`- Je suis désolé .

L'IDA et le Binary Ninja sont fermés et sont décrits ici à partir de leur documentation.`clang`- Je suis là .`ld`- Je suis là .`llvm-objdump`- Je suis là .`otool`- Je suis là .`nm`- Je suis là .`strip`, et `dyld_info`L'analyse de l'analyse de la base de données de l'analyse de données de la base de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de données de données de l'analyse de données de données de l'analyse de données de l'analyse de données de l'analyse de données de données de l'analyse de données de données de l'analyse de données de données de l'analyse de données de données de l'analyse de données de données de l'analyse de données de données de l'analyse de données de données de l'analyse de données de données de l'analyse de données de l'analyse de données de données de l'analyse de données de l'analyse de données de l'analyse de l'analyse de données de l'analyse de données de l'analyse de données de l'analyse de l'analyse de données de l'analyse de données de l'analyse de l'analyse de données de l'analyse de l'analyse de l'analyse de données de l'analyse de l'analyse de données de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de données de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de l'analyse de la base.`ledger_main`, les mêmes douze corrections. les outils de peinture lisent uniquement les fichiers ELF. le profil refusé`main-x86_64.obj`avec `PEFormatError: 'DOS Header magic not found.'`et lire la petite image.

## Faites-le

Exécutez la démo:

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries
python3 code/main.py
```

La sortie est un rapport JSON avec cinq parties. `elf`détient les chiffres de délocalisation des deux objets, le plan de liaison et la section dynamique, PLT et GOT de `tiny-dyn-x86_64`- Je suis là .`coff`détient les sections, les délocalisations, les fonctions et les noms indéfinis des deux objets. `pe`détient les titres, les annuaires, les importations, les exportations, les délocalisations de bases et les appels d'appels de l'IAT `tiny-x86_64.exe`- Je suis là .`macho`Il tient`ledger_main`Avec ses étiquettes, ses fixages enchaînés, ses rangées de fixations et ses exportations, l'héritage de la mise en page de `ledger_main_nochain`, le dylib, et les deux rapports de strip.`identification`Il y a la vérité, les quatre méthodes avec les scores, et les signatures avec leurs collisions, à `-O0`et `-O2`- Je suis désolé .

Forme attendue:

```json
{
  "elf": {"linkPlan": {"pltSymbols": ["puts", "printf"], "sizes": {".plt": 48, ".got.plt": 40, ".rela.plt": 48, ".rela.dyn": 48}, "...": "..."}, "tiny-dyn-x86_64": {"...": "..."}},
  "coff": {"main-x86_64.obj": {"...": "..."}},
  "pe": {"imports": [{"dll": "msvcrt.dll", "iat": "0x2040", "...": "..."}], "...": "..."},
  "macho": {"ledger_main": {"flags": "0x200085", "fixupRows": ["__DATA_CONST __got 0x100004000 bind libledger/_grade", "..."], "...": "..."}, "strip": {"...": "..."}},
  "identification": {"O2": {"methods": {"functionStarts": {"recall": "5/5"}, "prologueSub": {"recall": "1/5"}, "...": "..."}, "correctNames": 5}}
}
```

Lire `code/main.py`dans cet ordre:

1. `ElfFile`Les titres, les sections, les tables de symboles et les délocalisations sont lus, et `dynamic_entries()`- Je suis là .`jump_slots()`- Je suis là .`plt_entries()`, et `plt0()`Lisez l'image liée.
2. `plan_dynamic_link()`s'applique au tableau 4.9 et à la section 5.2 des dossiers de déménagement des objets relocalisables.
3. `CoffObject`lisent l'en-tête du fichier, la table de section, les symboles de 18 octets avec leurs enregistrements auxiliaires et les délocalisations de 10 octets.
4. `PeImage`Lisez le code MS-DOS, la signature PE, l'en-tête facultatif, les annuaires de données, les sections, ainsi que les annuaires d'importation, d'exportation et de délocalisation de base, et `iat_calls()`décode `ff 15`- Je suis désolé .
5. `MachoFile`lisent les commandes de chargement, les segments, `nlist`les enregistrements, la table des symboles indirects, les coups, les fixations enchaînées, les essais d'exportation et les délocalisations d'objets.
6. `FunctionFinder`Il dirige les quatre trouvailles de départ et `score()`comparer chacun à la vérité.
7. `signatures_from_object()`et `match_signatures()`construire et appliquer les signatures en forme de FLIRT.
8. Le `parse_*`Les fonctions lisent le texte de l'outil capturé par rapport aux tests.

Plus de commandes: `python3 code/main.py elf FILE`- Je suis là .`coff FILE`- Je suis là .`pe FILE`, et `macho FILE`imprimer un rapport. `python3 code/main.py plan OBJECT...`imprime le plan de liaison pour tout ensemble d'objets ELF. `python3 code/main.py functions STRIPPED UNSTRIPPED OBJECT...`Il marque les découvreurs et les signatures sur une autre paire de vos propres constructions.

## Utilisez-le

Utilisez les lecteurs dans cet ordre sur un programme que vous avez construit.

1. Lisez les en-têtes et nommez le format et le type: objet déplaçable, exécutable ou bibliothèque partagée.
2. Liste des importations avec leurs fournisseurs: `DT_NEEDED`et `.dynsym`, l'annuaire des importations ou le tableau des importations en chaîne et `LC_LOAD_DYLIB`Chaque appel à un nom sur cette liste laisse l'image.
3. Suivez un appel à la main à son emplacement. Écrivez les octets d'appel, les octets de stub ou PLT, l'adresse du emplacement, le mot dans le fichier et ce que le chargement y écrit. Si le mot dans le fichier ressemble à de la poubelle, décodez-le comme un correctif avant de l'appeler déchet.
4. Comparer la table des symboles avec les exportations et la fonction démarre.`strip`- Les noms que le débogageur a besoin ne le font pas.
5. Dans une construction dérivé, exécutez les trouvailles que vous pouvez exécuter et les marquer contre le jumeau non dérivé pendant que vous en avez encore un.`LC_FUNCTION_STARTS`- Je suis là .`.pdata`ou `.eh_frame`, sur les modèles de octets, et utiliser des cibles d'appel pour combler les lacunes.
6. Construisez des signatures à partir de vos propres objets et appliquez-les. Enregistrez chaque collision et le octet qui la divise. Un nom que deux signatures ont revendiqué est un inconnu jusqu'à ce que la queue ou un nom de référence décide.

Gardez le texte de l'outil capturé avec les fichiers. Lorsque la chaîne d'outils change, récupérez, car une liste modifiée est un fichier modifié.

## Laboratoire interactif

### Laboratoire A: le PLT à la main

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries/code
python3 -q
```

Je vais courir .

```python
from main import *
elf = load_elf("tiny-dyn-x86_64")
[(e.tag_name, e.text) for e in elf.dynamic_entries()]
elf.plt0()
[(e.symbol, hex(e.address), hex(e.slot), hex(e.slot_initial)) for e in elf.plt_entries()]
[hex(w) for w in elf.got_plt_words()]
```

Avant chaque appel, calculer la réponse à partir des octets dans le texte ci-dessus.`IMPORTS`dans `build_tiny_dyn_elf.py`à trois noms, reconstruire, et prédire les nouvelles tailles de `.plt`- Je suis là .`.got.plt`, et `.rela.plt`avant de les lire.

### Laboratoire B: un réparateur enchaîné

```python
exe = load_macho("ledger_main")
stub = next(s for s in exe.stubs() if s.symbol == "_puts")
stub.lines()
raw = int.from_bytes(exe.read(stub.slot, 8), "little")
hex(raw), raw >> 63, (raw >> 51) & 0xfff, raw & 0xffffff
[(i.index, i.name, i.library) for i in exe.chained_fixups()["imports"]]
```

Décodez d'abord le mot de liaison sur papier: le bit de liaison, le `next`Lire le mot 8 octets plus loin et confirmer que c'est la prochaine correction.`0x100004058`et expliquer pourquoi il est `next`est de zéro.

### Laboratoire C: dépouille et relie

Dans le répertoire des appareils, connectez un nouveau programme à la dilib entièrement dépouillée:

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries/code/fixtures
clang -target arm64-apple-macos -O1 main.c -L. -lledger.stripped -Wl,-rpath,@executable_path -o /tmp/relinked
nm /tmp/relinked | grep ' U '
dyld_info -exports libledger.stripped.dylib
```

Le lien réussit avec une bibliothèque dont `LC_SYMTAB`est vide, parce que l'éditeur de lien lit le test d'exportation.`python3 ../main.py macho libledger.stripped.dylib`et comparer `symtab`- Je suis là .`exports`, et `functionStarts`- Alors dépouillez la construction statique avec`-N`Au lieu de ça, regardez ce que je vais faire.`dyld_info -function_starts`Il reste des empreintes.

### Laboratoire D: la collision

```python
from main import *
stripped = load_macho("static-arm64-O2.stripped")
sigs = signatures_from_object(load_macho("ledger-arm64-O2.o"))
[s.pat_line() for s in sigs]
starts = FunctionFinder(stripped).function_starts()
match_signatures(sigs, stripped, starts)
```

Lisez les deux dossiers de collision, puis construisez les signatures de`ledger-arm64-O0.o`, les faire comparer à `static-arm64-O0.stripped`, et expliquer à partir de `.pat`Les lignes de collision ne sont pas utilisées pour les collisions`-O0`- Alors raccourcissez .`PATTERN_LENGTH`à 16 et compter combien de fonctions se heurtent.

## Laboratoire de pratique

Décodez les tables de liaison antérieures et ajoutez le détecteur de PE.

Les exigences:

- Décodez le `LC_DYLD_INFO_ONLY`rébaser, lier et plier paresseux lier les flux de code optique de `ledger_main_nochain`dans des rangées avec segment, section, adresse, type et cible, et les rendre égaux aux 21 rangées dans la capture `dyld_info -fixups`Je vous ai envoyé un SMS.
- Ajouter un `PT_GNU_RELRO`en-tête du programme et `DT_FLAGS`avec `DF_BIND_NOW`à `build_tiny_dyn_elf.py`- Je suis là .`.got.plt`dans la plage couverte, et faire `llvm-objdump -p`Dans les fixations README, indiquez quelle partie de RELRO complet est reproduite et quelle partie seulement glibc peut effectuer.
- Écrivez une`.pdata`lecteur pour une image PE32+ et un chercheur de fonctions qui utilise `RUNTIME_FUNCTION`Commencez les adresses.`clang -target x86_64-pc-windows-msvc -c`sans`-fno-asynchronous-unwind-tables`, lire son `.pdata`les délocalisations et indiquer quelles de ses fonctions sont les fonctions de feuilles que le tableau omet.
- Écrivez un test pour chaque lecteur qui compare avec le texte de l'outil capturé.

Le succès signifie`python3 code/main.py macho ledger_main_nochain`imprime les mêmes 21 lignes de fixation que `dyld_info`Les empreintes, le minuscule ELF porte une`PT_GNU_RELRO`Ça ...`llvm-objdump`Les listes et les`.pdata`le chercheur nomme les fonctions qu'il peut et énumère les fonctions de feuille qu'il ne peut pas.

## Leçon d'artisanat

Cette leçon comprend `outputs/skill-linking-loading-and-stripped-binaries.md`. Utilisez-le pour transformer un programme que vous avez construit en une liste de ses importations avec leurs fournisseurs, un appel tracé à l'emplacement que remplit le chargement, une déclaration de ce que `strip`Il faut le jumeau non dérivé comme vérité de base, une liste d'outils capturés pour chaque revendication, et un enregistrement de chaque collision.

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries
python3 code/main.py
python3 code/tests/test_main.py
```

La vérification doit prouver:

- la section dynamique de `tiny-dyn-x86_64`les correspondances `llvm-objdump -p`- le nombre de points de référence`DT_FLAGS_1`porte `DF_1_PIE`
- Les deux .`R_X86_64_JUMP_SLOT`Les enregistrements correspondent `llvm-objdump -R`et nom `puts`et `exit`
- PLT0 pousse GOT+8 et saute à travers GOT+16, chaque entrée lit sa fente et pousse son indice, et chaque fente contient son propre adresse `pushq`
- Les sections et les en-têtes du programme de la petite rencontre ELF `llvm-objdump -h -p`
- Les dossiers de délocalisation des deux objets ELF correspondent `llvm-objdump -r`
- Le plan de lien trouve deux symboles PLT, quatre appels directs, deux déplacements relatifs, et les tailles de la petite image
- le tableau de la section COFF, 37 enregistrements symboliques avec enregistrements auxiliaires et correspondance des délocalisations `llvm-objdump -h -t -r`, et `/16`résolue à `.llvm_addrsig`
- Les trois COFF `REL32`enregistrements de `ledger`S' asseoir au troisième ELF `PC32`des compensations
- les en-têtes PE, `DllCharacteristics`, et les annuaires de données correspondent `llvm-objdump -p`
- le répertoire des importations, les indications/noms et la correspondance IAT `llvm-objdump`et `pefile`, et le TPI est égal à la table de recherche avant la liaison
- l'annuaire des exportations et le `DIR64`décode de délocalisation de base, et le pointeur délocalisé nomme la chaîne "liée à la main"
- `ff 15`à RVA `0x100b`résolve à l' emplacement IAT de `puts`
- Les drapeaux Mach-O, les noms des 19 commandes de chargement, et les deux bibliothèques correspondent.`otool`
- Les sept étiquettes sont décodées en huit.`__got`les machines à sous et les correspondances de table des symboles indirects `otool -Iv`
- Les fixups enchaînés correspondent à l' en-tête `llvm-objdump --macho --chained-fixups`
- La chaîne de marche reproduit les douze mots bruts de `dyld_info -fixup_chain_details`
- les rangées de fixation et le tableau des importations correspondent `dyld_info -fixups`et `-imports`
- Les exportations sont en équilibre.`dyld_info -exports`pour le dylib et pour sa copie dépouillée
- `nm_lines()`égale `nm -m`pour l'exécutable et sa copie dépouillée
- `strip`supprimé `_main`- Je suis là .`_fns`, et `_names`Il a gardé neuf noms et un .`radr://5614542`coupe, et le programme dépouillé imprime la même sortie
- la mise en page héritée a `LC_DYLD_INFO_ONLY`, sept pointeurs paresseux, et des basses dans `__stub_helper`
- Les déplacements d' objets arm64 correspondent `llvm-objdump -r`
- `LC_FUNCTION_STARTS`récupère cinq sur cinq à tous les niveaux et égale `nm`et `dyld_info -function_starts`
- le `sub sp`La scan trouve 5 sur 5 à la`-O0`et un de cinq à `-O2`, et le `stp`l'analyse du cadre ne trouve aucun
- `bl`les objectifs plus les `LC_MAIN`entrée récupérer cinq sur cinq
- Les signatures nomment les cinq fonctions à ces deux niveaux et enregistrent les`grade`collision divisée par byte 44
- le CRC16 de "123456789" est `0x6e90`, et les huit octets sous la paire de charge GOT de `main`sont variantes
- chaque octet d'objet non variable est égal au octet lié
- la démo et les données figuratives rapportent les mêmes adresses, et le module du site intègre les mêmes données dans ES5

## Mode de défaillance de la production

| Failure | What the weak tool reports | What the analysis must prove |
|---|---|---|
| A `bl` or `call` lands in a stub | "calls an unnamed function at 0x694" | the stub bytes, the slot it loads, and the import the slot binds |
| A GOT word read from the file | "corrupt pointer 0x8010000000000007" | bit 63 is bind, `next` is 2, ordinal 7 is `_puts` from library 2 |
| `PLT32` in an object | "every call goes through the PLT" | defined symbols are reduced to a direct distance, only undefined ones get an entry |
| Partial RELRO read as full | "GOT is read-only" | `.got.plt` slots after the three reserved ones sit outside `PT_GNU_RELRO` without `-z now` |
| COFF `REL32` compared with ELF `PC32` | "the Windows build has different relocations" | same offsets, the `-4` lives in the ELF addend and in the COFF field |
| IAT read before load | "imports point at strings" | the IAT equals the lookup table until the loader binds, and the strings are hint/name entries |
| A stripped dylib | "no symbols, nothing exported" | the exports trie and `LC_FUNCTION_STARTS` list every function |
| `nm` on a stripped executable | "one function" | `LC_FUNCTION_STARTS` lists five starts, nine bytes long |
| A prologue scan at `-O2` | "one function" | four leaf functions have no frame, call targets and the starts table find them |
| Two functions share 36 bytes | "grade_label is grade" | the tail byte at 44 and the lengths 60 and 72 split the collision |
| A tool run on another machine's build | the numbers in this text | recapture every listing from your own fixtures |

## Règlement d'exploitation

Suivez chaque appel externe depuis le site d'appel jusqu'à l'emplacement rempli par le chargement, et décodez le mot dans cet emplacement tel que le format le définit avant de l'appeler une adresse. Ne laissez pas tomber avant d'avoir enregistré ce dont le chargement a encore besoin. La fonction de récupération commence à partir des tables que l'éditeur de liens a écrites avant de rechercher des octets, de scorer chaque rechercheur contre un jumeau avec des symboles et d'enregistrer chaque collision de signature avec le octet qui le divise.

## Pour en savoir plus

- [System V gABI, chapter 5: Program Loading and Dynamic Linking](https://www.sco.com/developers/gabi/latest/ch5.dynamic.html)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement: section 5.2 and Table 4.9](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Drepper, "How To Write Shared Libraries", version 4.1.2](https://akkadia.org/drepper/dsohowto.pdf)
- [Drepper, "ELF Symbol Versioning"](https://akkadia.org/drepper/symbol-versioning)
- [Jelinek, "Little hardening DSOs/executables against exploits", binutils list, 2004-01-06](https://www.sourceware.org/ml/binutils/2004-01/msg00070.html)
- [Taylor, "Linker relro", 2008](https://www.airs.com/blog/archives/189)
- [binutils commit bd7ab16b: PLT32 for calls, 2018-02-13](https://gnu.googlesource.com/binutils-gdb/+/bd7ab16b4537788ad53521c45469a1bdae84ad4a)et [LLVM D43383](https://reviews.llvm.org/D43383)
- [GNU Binutils: strip](https://sourceware.org/binutils/docs/binutils/strip.html)et [llvm-objcopy](https://llvm.org/docs/CommandGuide/llvm-objcopy.html)
- [Microsoft, PE Format](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format)
- [CALL instruction reference, FF /2](https://www.felixcloutier.com/x86/call)
- [Apple, mach-o/loader.h](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h)et [mach-o/nlist.h](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/nlist.h)
- [Apple dyld, mach-o/fixup-chains.h](https://github.com/apple-oss-distributions/dyld/blob/main/include/mach-o/fixup-chains.h)
- [Apple ld64, stub_arm64.hpp](https://github.com/apple-oss-distributions/ld64/blob/main/src/ld/passes/stubs/stub_arm64.hpp)
- [Xcode 13 release notes: chained fixups](https://developer.apple.com/documentation/xcode-release-notes/xcode-13-release-notes)et [WWDC22 session 110362, "Link fast: Improve build and launch times"](https://developer.apple.com/videos/play/wwdc2022/110362/)
- [strip(1)](https://keith.github.io/xcode-man-pages/strip.1.html)- Je suis là .[ld(1)](https://keith.github.io/xcode-man-pages/ld.1.html), et [dyld(1)](https://keith.github.io/xcode-man-pages/dyld.1.html)
- [Hex-Rays, IDA F.L.I.R.T. Technology: In-Depth](https://docs.hex-rays.com/core/flirt/concepts/ida-f.l.i.r.t.-technology-in-depth)et [Lumina overview](https://docs.hex-rays.com/add-ons/lumina/overview)
- [Ghidra Function ID help](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/FunctionID/src/main/help/help/topics/FunctionID/FunctionID.html)et [AARCH64 function start patterns](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/AARCH64/data/patterns/AARCH64_LE_patterns.xml)
- [Vector 35, WARP](https://github.com/Vector35/warp)et [Binary Ninja WARP guide](https://docs.binary.ninja/guide/warp.html)
- [BinDiff manual](https://www.zynamics.com/bindiff/manual/)et [BinDiff source](https://github.com/google/bindiff)
- [Diaphora](https://github.com/joxeankoret/diaphora)
- [rizin pat.c, the .pat parser](https://github.com/rizinorg/rizin/blob/dev/librz/sign/pat.c)- Je suis là .[radare2 flirt.c](https://github.com/radareorg/radare2/blob/master/libr/anal/flirt.c), et [radare2 book: Zignatures](https://book.rada.re/signatures/zignatures.html)
- [Andriesse et al., "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse)et [Andriesse, Slowinska, Bos, "Compiler-Agnostic Function Detection in Binaries" (EuroS&P 2017)](https://danielandriesse.com/papers/eurosp-2017.pdf)
- [Bao et al., ByteWeight (USENIX Security 2014)](https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/bao)et [Shin, Song, Moazzezi, "Recognizing Functions in Binaries with Neural Networks" (USENIX Security 2015)](https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/shin)
- [LIEF changelog](https://lief.re/doc/latest/changelog.html)- Je suis là .[pyelftools](https://github.com/eliben/pyelftools), et [pefile](https://github.com/erocarrera/pefile)
