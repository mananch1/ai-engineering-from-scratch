# Le paysage de l'ingénierie inverse: histoire, familles d'outils et construction d'un décompilant

> Cinq outils lisent le même objet et donnent quatre adresses, trois blocs comptent, et deux réponses pour un interrupteur, et cette leçon construit le harnais qui montre lequel est lequel.

**Type:** Build
**Languages:** Python, C, x86-64 assembly, Java (one Ghidra script)
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~150 minutes

## Objectifs d'apprentissage

- Placez les principaux outils du champ sur une chronologie datée de 1973 à 2026 et nommez la famille à laquelle chacun appartient.
- Nombre les huit étapes d'un décompilateur et le terme que Ghidra, Ninja binaire, Hex-Rays, radar2, rizin et angr utilisent pour chaque étape.
- Écrivez une instruction en ESIL, RzIL, P-code et VEX et expliquez laquelle d'entre elles écrit des drapeaux avec impatience et qui stocke un morceau.
- Réduire les adresses de cinq outils à un `.text`compenser et prouver la réduction sur une fonction.
- Construisez un harnais de comparaison qui lit les sorties de l'outil capturé et rapporte l'accord sur les démarres d'instructions, les blocs, les bords et les démarres de fonction.
- Nom, à partir de preuves capturées, où chaque outil invente du code, manque du code ou s'arrête à un saut indirect.

## Le problème

Vous demandez à un agent de résumer ce que trois outils d' ingénierie inverse disent à propos de .`grade`dans `ledger-x86_64-O1.o`L'agent exécute chaque outil, lit trois listes et rapporte que la fonction est à `0x08000050`dans un seul outil, `0x400010`dans une autre, et `0x100010`Il est donc possible de trouver des solutions qui permettent de déterminer la fonction de l'outil.

Les outils sont d'accord. L'objet est relocable et n'a pas d'adresse de chargement, donc chaque outil en choisit une. Soustraire l'adresse donnée par chaque outil `.text`et chaque réponse est `0x10`L'agent a comparé les adresses brutes parce que rien ne lui a dit ce que l'adresse signifiait dans chaque outil.

Plus tard , le même agent dit:`grade`dans le `-O0`construire et de rapports 3, 8, 8 et 9 blocs de base à partir de quatre outils.`jmp rax`Deux outils lisent la table de saut à travers ses déplacements. Un outil divise un bloc au saut pour placer une étiquette. Un nombre de blocs sans raison derrière lui est un nombre sans revendication.

Cette leçon est la carte dont vous avez besoin avant de faire confiance à ces chiffres. Les leçons 32 à 36 ont construit un décodeur, un élévateur, un enregistrement de preuves et une reconstruction. La seconde moitié de la piste les place à côté des outils utilisés. Pour cela, vous avez besoin de l'histoire du champ, des familles dans lesquelles les outils se trouvent, du pipeline dans lequel chaque décompilateur fonctionne, et d'un harnais qui compare ce que les outils disent sur les objets que vous avez compilés vous-même.

```figure
re-37-field-map
```

## Cinquante-trois ans dans des étapes datées

Le champ est plus ancien que la plupart de ses outils.

### 1973 à 1991: débogageurs d'abord, puis démontageurs

Les premières thèses de décompilation sont de 1973. Hollander à Stanford et Housel à Purdue les ont écrites, et les deux sont connues aujourd'hui par des citations.`adb`Le débogageur a été livré avec UNIX V7 en 1979.`DEBUG`Le système de décomposition de la base de données de Microsoft, avec un désassembleur 8086, est livré dans PC-DOS 1.00 en août 1981.

Les démantèlements de lots sont venus ensuite. Sourcer, de V Communications en 1988, a exécuté plusieurs passes sur un programme DOS et a écrit une liste commentée.`objdump`Ilfak Guilfanov a écrit la première ligne de l'IDA en janvier 1991 et a publié l'IDA 0.1 en tant que shareware le 20 mai 1991.

### 1991 à 2009: démontants interactifs et les premiers décompilateurs

Cristina Cifuentes a terminé "Techniques de compilation inverse" à l'Université de QUT en juillet 1994.`dcc`En 1995, elle a écrit un article sur la conception de la technologie de compilation, qui a porté i80286 DOS à C, et son article de 1995 avec Gough a établi les phases que tous les décompilateurs ultérieurs ont gardées.

IDA est devenu un produit commercial chez DataRescue en 1996, a acquis un SDK de plugins en 1999 et a déménagé à Hex-Rays, fondé en 2005. Le Hex-Rays Decompiler 1.0 a été expédié le 17 septembre 2007. La même année, Mike Van Emmerik a présenté "Static Single Assignment for Decompilation", la thèse qui a placé SSA au centre du flux de données des décompilateurs. La "comparation structurelle des objets exécutables" (2004) de Halvar Flake a commencé la famille différente, et BinDiff est entré en vente en août 2004.

Radare a été lancé en février 2006 comme éditeur hex de forensique avec des compensations de 64 bits. Radare2, la réécriture modulaire sous LGPL-3.0, a été publiée pour la première fois le 7 mars 2009.

### 2009 à 2019: cadres et représentations intermédiaires

LLDB a été annoncé comme un sous-projet LLVM le 8 juin 2010. BAP, la plateforme d'analyse binaire avec son langage intermédiaire BIL, est apparue au CAV 2011. Google a acquis zynamics, les fabricants de BinDiff, le 1er mars 2011. Hopper est apparu sur le Mac App Store vers la fin de 2011. Matt Godbolt a lancé Compiler Explorer en 2012 afin qu'un programmeur C++ puisse voir l'assemblage produit par un compilateur.

Angr a reçu sa première ligne de code en août 2013. Le document Phoenix (Schwartz, Lee, Woo, Brumley, USENIX Security 2013) a montré une structuration sémantique conservant avec 40 `goto`Les données de la société DREAM (Yakdan et autres, NDSS 2015) ont été supprimées.`goto`les déclarations structurées indépendamment des modèles. Vector 35 a été fondé le 14 janvier 2015, et Binary Ninja est entré en vente le 23 juillet 2016 avec son IL de bas niveau et une API Python. Angr a été publié en 2015 et a été classé troisième au DARPA Cyber Grand Challenge en août 2016. BinDiff est devenu gratuit en mars 2016. Le coupeur 1.0 a donné à radar2 une interface graphique Qt en décembre 2017. En 2018, Guilfanov a présenté "Decompiler internals: microcode" à Recon, la première description publique de l'IR Hex-Rays.

### 2019 à 2026: décompilateurs ouverts et tour de la recherche

Ghidra 9.0 a été publié par la NSA lors de la Conférence RSA le 5 mars 2019 sous Apache-2.0, avec source complète sur GitHub un mois plus tard. En angr 8.19.2.4, un décompilateur préliminaire a été ajouté la même année. Binary Ninja 2.0 a expédié son décompilateur IL de haut niveau le 20 mai 2020. Rizin a été annoncé comme une fourchette de radar2 le 8 décembre 2020, et l'équipe de Cutter s'y est rendue. IDA Freeware 7.6 a ajouté un décompilateur x64 en nuage en mai 2021.

BinDiff 8 a été lancé sous Apache-2.0 le 25 septembre 2023. SAILR (Basque et autres, USENIX Security 2024) est devenu le structurateur par défaut d'angr et a soutenu que seulement faux `goto`Les déclarations doivent être supprimées, car Linux 6.1 en a 3.754 dans sa source.`idalib`Les documents de décompilateur appris vont de SLaDe (2023) à LLM4Decompile (2024) à DecompileBench (2025) et deux études de 2026 sur la lisibilité assistée par un agent et sur les décompilateurs qui recompilent plus et conservent moins.

Les versions de cette leçon sont les actuelles du 14 octobre 2026: Ghidra 12.1.4 (21 septembre 2026), radar2 6.2.4 (5 octobre 2026), rizin 0.9.1 (29 juin 2026), et angr 10.0.3 (10 octobre 2026).

## Les familles des outils

Un outil appartient à une famille par ce qu'il produit.

### Les démontants

Un balayage linéaire décode du premier octet d'une section de code à la dernière.`objdump -d`et `llvm-objdump -d`fonctionne de cette façon, et aucun ne construit un graphique de contrôle-flux.`objdump`Les binutils 2.47 (26 juillet 2026) sont GPL-3.0 ou ultérieur. LLVM 23.1.3 est Apache-2.0 avec l'exception LLVM.`llvm-objdump`comme "Apple LLVM version 17.0.0", et c'est la copie de cette leçon capturée.

Un désassembleur interactif effectue une descente récursive à partir d'entrées connues, ajoute des heuristiques pour les démarres de fonction et conserve une base de données.`Disassembler`Plus des analystes d'analyse automatique), Ninja binaire (descendance récursive, puis balayage linéaire sur des régions inconnues), Hopper, radar2 et rizin (`aa`- Je suis là .`aaa`- Je suis là .`af`Les données de l'analyse de données ont été analysées par l'analyse de données de type "User", avec Cutter et iaito comme GUIs, tous dans cette famille. Andriesse et d'autres (USENIX Security 2016) ont mesuré neuf d'entre eux sur des binaires générés par le compilateur.

### Des découpageurs

Décompilateurs commerciaux: le décompilateur Hex-Rays (douze familles cibles de x86 à RISC-V), le HLIL de Binary Ninja avec son rendu Pseudo C et la vue pseudo-code de Hopper. Décompilateurs ouverts: le décompilateur Ghidra, un moteur C++ sous Apache-2.0 qui r2ghidra et rz-ghidra réutilisent comme plugin et qui pypcode réutilise comme élévateur; le décompilateur angr sur AIL avec SAILR structurant sous BSD-2-Clause; r2dec et jsdec en JavaScript; Boomerang et `dcc`Le radar2 a aussi un système intégré.`pdc`Rizin n'a pas de décompilateur intégré.`pdc`ne se trouve pas dans la rizine 0.9.1, et `pdg`Il a besoin du plugin RZ-Gidra.

### Débogageurs

GDB (1986) avec le GEF et pwndbg avant-termes, LLDB (2010), WinDbg avec Débogage de Voyage Temporal (2017), OllyDbg (congelé en 2014), x64dbg (GPL-3.0, 2014), le Débogageur Ghidra avec ses connecteurs Python, le Débogageur Binary Ninja, et les débogageurs intégrés de radar2 et rizin.

### Cadres d'analyse binaire

angr (VEX IR, exécution symbolique, `CFGFast`et `CFGEmulated`Le système de calcul de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur est de la valeur de la valeur de la valeur de la valeur est de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur de la valeur`import binaryninja`, et l' IDA à travers `idalib`- Je suis désolé .

### Bibliothèques de format et de moteurs

LIEF 1.0.0 (12 juillet 2026, Apache-2.0) parsèse les formats ELF, PE, Mach-O, COFF et Android. CLE, "CLE Loads Everything", est le chargement de l'angr. Capstone 5.0.9 (28 mai 2026, BSD-3-Clause) décode; assemblages Keystone 0.9.2 (GPL-2.0); émulations Unicorn 2.1.4 (GPL-2.0). radare2 6.2.4 bundles Capstone 5, et rizin 0.9.1 fournisseurs Capstone, Zydis, et tree-sitter.

### Outils de diffage et affichage côté compilateur

BinDiff 8 (Apache-2.0) avec son plugin BinExport 12 pour IDA Pro, Binary Ninja et Ghidra; Diaphora 3.4.3 (11 octobre 2026, AGPL-3.0) comme plugin IDA; `radiff2`et `rz-diff`.L'Explorateur de compilateur (BSD-2-Clause) montre l'assemblage qu'un compilateur émet pour un fragment source, qui est la direction de l'avant de tout le reste de cette liste.

### Les versions, les licences et ce que cette leçon a été

| Tool | Version read on 2026-10-14 | Licence | Headless entry | Ran here |
|---|---|---|---|---|
| Ghidra | 12.1.4 | Apache-2.0 | `analyzeHeadless`, `pyghidra.start()` | yes, on all four objects |
| radare2 | 6.2.4 | LGPL-3.0 | `r2 -qc '<cmds>' file`, `rabin2` | yes |
| rizin | 0.9.1 | LGPL-3.0 | `rizin -qc '<cmds>' file`, `rz-bin` | yes |
| angr | 10.0.3 | BSD-2-Clause | `angr.Project` in Python | yes |
| llvm-objdump | Apple LLVM 17.0.0 | Apache-2.0 with LLVM exception | `llvm-objdump -d -r` | yes |
| Binary Ninja | 6.0 "Krypton" | commercial; Free edition for non-commercial use | `binaryninja.load()` with a Commercial, Ultimate, or headless licence | no, documentation only |
| IDA Pro | 9.4 | commercial subscription; IDA Free for non-commercial x86 and x86-64 | `idat`, `ida -A -S`, `idalib` | no, documentation only |
| Hopper | 6.6.0 | commercial, perpetual | none documented | no, documentation only |
| Cutter | 2.5.0 | GPL-3.0 | GUI over rizin | no |

L'édition gratuite de Binary Ninja décompie x86, x86-64, ARMv7 et AArch64 seulement et n'a pas d'API.`idalib`Chaque déclaration sur ces deux outils dans cette leçon provient de leur documentation et de leurs notes de sortie, et la figure les marque comme non exécutés.

## Le pipeline de décompilation, étape par étape

Cifuentes a nommé les phases en 1994: analyse de syntaxe, analyse sémantique, génération de code intermédiaire, génération de graphes de flux de contrôle, analyse de flux de données, analyse de flux de contrôle et génération de code. Elle les a divisés en une partie avant dépendante de la machine, un milieu indépendant de la machine et du langage, et une partie arrière dépendante du langage. Chaque décompilateur conserve cette forme et renomme les pièces. Le tableau ci-dessous donne les huit étapes utilisées dans cette leçon et le nom donné par chaque outil. La figure montre le même tableau avec les colonnes d'outils sélectionnables.

| Stage | Ghidra | Binary Ninja | Hex-Rays / IDA | radare2 | rizin | angr |
|---|---|---|---|---|---|---|
| Loader | `ElfLoader`, `MachoLoader`, `PeLoader`; `.ldefs` picks the language | `BinaryView`, `BinaryViewType` | loader module (`LDSC`) | RBin (`rabin2`) | RzBin (`rz-bin`) | CLE (`cle.Loader`, `main_object`) |
| Disassembly | `Disassembler`; SLEIGH `.slaspec` compiled to `.sla`; analyzers "Disassemble Entry Points", "Function Start Search" | architecture plugin; Recursive Descent, then linear sweep phases | processor module (`ana.cpp`, `emu.cpp`, `out.cpp`); autoanalysis queues; FLIRT | RArch plugin to `RAnalOp`; `aa`, `aaa`, `af` | RzAnalysis plugin; `aa`, `aaa`, `af` | Capstone blocks; `CFGFast` function start search |
| CFG recovery | "Generate Basic Blocks and the CFG"; `BlockGraph`; "Decompiler Switch Analysis" | `Function`, `BasicBlock`; "Control Flow Graph (CFG) Analysis" phase | `mbl_graph_t`, ready at `MMAT_LOCOPT` | `afb`, `agf`, `aab` | `afb`, `agf`, `aab` (Nucleus) | `CFGFast`, `CFGEmulated`; `kb.functions`; IRSB |
| Lifting | "Generate Raw P-code"; Varnodes in `ram`, `register`, `const`, `unique` | Lifted IL, LLIL, Mapped MLIL, MLIL | microcode (`minsn_t`, `mop_t`); `MMAT_GENERATED` | ESIL (`asm.esil`, `ae`) | RzIL (`aoi`, `aez`) | VEX (`pyvex.lift`, IRSB), then AIL |
| SSA | `Heritage`, `ActionHeritage`; `MULTIEQUAL`, `INDIRECT` | LLIL SSA, MLIL SSA, HLIL SSA; `HLIL_VAR_PHI` | no named SSA; use-def data flow in `MMAT_GLBOPT1` to `3` | none; ESIL is built for emulation | none; RzIL is built for emulation | `_stage_transform_to_ssa_level0` and `level1`; `SeqNodeDephication` |
| Data flow and types | `ActionDeadCode`, `ActionInferTypes`, the `Rule` pool; prototype from `.cspec` | MLIL constant propagation, variables and types; `analyzeReturns` | "Transform microcode"; `MMAT_CALLS`, `MMAT_GLBOPT`, `MMAT_LVARS`; TIL | `afv`, `afc`, `aaft`; ESIL emulation | `afv`, `afc`, `aft` | `CallingConventionAnalysis`, `VariableRecoveryFast`, `Typehoon` |
| Structuring | `CollapseStructure`; `BlockIf`, `BlockWhileDo`, `BlockSwitch`, `BlockGoto` | `HLIL_IF`, `HLIL_WHILE`, `HLIL_SWITCH`; AST at `root` | "Generate ctree"; structural analysis; goto regions | `pdc`; `pdd` (r2dec); `pdg` (r2ghidra) | `pdg` (rz-ghidra); jsdec | `RegionIdentifier`; `RecursiveStructurer` with `SAILRStructurer` |
| Emission | "Emit Final C Tokens"; `PrintC`, `ActionNameVars`, `ActionSetCasts` | HLIL rendering; Pseudo C, Pseudo Rust, Pseudo Python | "Beautify ctree; Print ctree"; `cfunc_t` | text of `pdc`, `pdd`, `pdg` | `pdg` text | `CStructuredCodeGenerator`; `codegen.text` |

### Chargement

Le chargement analyse le conteneur, cartographiant les sections et lisant les symboles et les délocalisations. Dans un objet relocalisable, rien n'a encore d'adresse, donc le chargement choisit un.`0x08000000`, ce qui met `.text`à`0x08000040`parce que la section commence à l' offset de fichier `0x40`- Les lieux de l' école de classe .`.text`à`0x400000`- Je suis Ghidra.`ElfLoader`Il est placé à `0x100000`- Je suis là .`llvm-objdump`Il imprime des compensations de section et aucune base du tout.

Les chargeurs diffèrent également en matière de relocalisation. Ghidra les traite à l'importation et CLE les applique. radar2 imprime un indice lorsqu'il ouvre l'objet: les relocalisations n'ont pas été appliquées, et `bin.relocs.apply=true`Ce qu'un chargement fait avec les délocalisations décide de ce que chaque étape ultérieure peut voir, et vous constaterez l'effet dans le `-O0`Sautez la table en bas.

### Démantèlement

La leçon 33 a construit les deux méthodes.`llvm-objdump`Il fait descendre la fonction recursive et ajoute une recherche pour le démarrage de la fonction. La documentation de Binary Ninja nomme les phases dans l'ordre: Descendement récursif, analyse de cible d'appel, analyse CFG comme "démantèlement linéaire des régions inconnues" et analyse d'appel de queue. Ghidra gère ses analyzers par priorité, avec "Déassembler les points d'entrée" et "Cercher le démarrage de la fonction" tôt. L'IDA garde des files d'attente d'analyse automatique, chacune avec une priorité, et des signatures FLIRT pour le code de la bibliothèque. - Je suis en colère.`CFGFast`Les fonctions de l'angle angr sont les fonctions de l'angle angr.

### Rétablissement du CFG

Chaque outil coupe les blocs de la même manière que la leçon 33, avec une différence que vous mesurerez.`call`L'IRSB est un bloc d'entrée unique, de sortie multiple et un appel est une sortie.`grade_label`à`-O0`, qui appelle`grade`Les deux blocs et les trois autres rapports 1.

### Le levage

Le code de référence est un code de référence qui est utilisé pour la traduction de la langue de niveau supérieur. Les IR diffèrent dans ce qu'ils sont pour. Le code P et le code VEX ont été construits pour l'analyse et la décompilation.`MMAT_GENERATED`à `MMAT_LVARS`. Binary Ninja se déplace vers Lifted IL, puis replie les drapeaux en conditions dans LLIL, et MLIL "transforme les registres et les accès à la mémoire en variables".

### SSA

La thèse de Van Emmerik énumère ce que SSA achète un décompilateur: propagation des expressions, paramètre et identification de retour sans supposition ABI, analyse de conservation, analyse de type comme problème de flux de données rare, et analyse des sauts indirects et des appels.`Heritage`et ajoute deux opérations qui existent uniquement dans le graphique,`MULTIEQUAL`pour le phi et`INDIRECT`Les effets secondaires. le Ninja binaire expose une forme de SSA à chaque niveau de IL. angr a deux étapes de SSA à l'intérieur`Clinic`. Hex-Rays ne nomme pas de formulaire SSA et utilise le flux de données de définition dans ses passes d'optimisation globale.

### Flux et types de données

La boucle de simplification principale de Ghidra fonctionne "Éliminer le code mort", "Propager les types locaux" et "Exécuter la réécriture à terme" jusqu'à ce que rien ne change, avec la réécriture effectuée par un pool de `Rule`Les rayons Hex transforment le microcode, détectent les arguments d'appel à l'adresse `MMAT_CALLS`, et attribue des variables locales à `MMAT_LVARS`. Angr court `CallingConventionAnalysis`- Je suis là .`VariableRecoveryFast`, et `Typehoon`La leçon 41 concerne cette étape.

### La structure

Le Ghidra `CollapseStructure`Il est utilisé pour rechercher des sous-graphes qui correspondent à une structure, les effondre en un seul nœud, et quand il est coincé enlève un bord et le marque non structuré.`if`- Je suis là .`while`- Je suis là .`do`, et `switch`L'angr identifie les régions à entrée unique et à sortie unique et les structure avec `SAILRStructurer`par défaut, avec `PhoenixStructurer`et `DreamStructurer`La leçon 34 a été écrite par un structuriste à la main.

### Émissions

Ghidra émet des jetons C à travers `PrintC`avec une imprimante Oppen après`ActionNameVars`et `ActionSetCasts`Les rayons Hex-Rays embellir et imprimer l'arbre.`codegen.text`. des empreintes radar2`pdc`- Je suis là .`pdd`ou `pdg`- Je suis désolé .

## Une instruction dans quatre IR

Prenez .`48 83 ec 10`- Je suis là .`subq $0x10, %rsp`- Décodez-le sur papier avec les règles de la leçon 33.

1. `48`est REX avec `W = 1`L'opérand est de 64 bits.
2. `83`est le groupe-1 opcode arithmétique avec un 8 bits immédiat.
3. `ec`est ModRM `11 101 100`- Je suis là .`mod = 11`est le formulaire de registre. `reg = 101`est `/5`, qui est `sub`- Je suis là .`r/m = 100`est `%rsp`- Je suis désolé .
4. `10`est immédiat, 16.

Quatre élévateurs ont reçu ces quatre octets.

ESIL de radar2 6.2.4 est une chaîne séparée par la virgule de 30 jetons:

```text
16,0x8000000000000000,-,!,16,rsp,-=,63,$o,^,of,:=,63,$s,sf,:=,$z,zf,:=,$p,pf,:=,64,$b,cf,:=,4,$b,af,:=
```

C'est une machine à pile.`16,rsp,-=`chaque drapeau est ensuite écrit avec un opérateur interne: `$o`débordement,`$s`le signe, `$z`- Le taux de débit`$p`parité, `$b`Je vous en prête.

Le RzIL de rizine 0.9.1 est une expression S avec 14 `set`effets:

```text
(seq (set op1 (var rsp)) (set op2 (bv 64 0x10)) (set sub (- (var op1) (var op2))) (set rsp (var sub))
     (set _result (var sub)) (set pf (! (lsb (let _val (cast 8 false (var _result)) ...))))
     (set zf (is_zero (var _result))) (set sf (msb (var _result))) ... (set cf (|| ...)) (set of (|| ...)) (set af (|| ...)))
```

Chaque drapeau est une expression booléenne sur des bitvecteurs.`let`- Je suis désolé .

Code P de la spécification SLEIGH `x86:LE:64:default`par le code pip 4.0.1 est de neuf opérations après la`IMARK`- Le numéro de la liste:

```text
CF = RSP < 0x10
OF = sborrow(RSP, 0x10)
RSP = RSP - 0x10
SF = RSP s< 0x0
ZF = RSP == 0x0
unique[58300:8] = RSP & 0xff
unique[58400:1] = popcount(unique[58300:8])
unique[58500:1] = unique[58400:1] & 0x1
PF = unique[58500:1] == 0x0
```

Enregistrez les transferts avec les transférés temporaires dans le `unique`Les drapeaux sont calculés à l'instruction, avec impatience, et`CF`et `OF`sont calculées avant que la soustraction ne soit écrasée `RSP`- Je suis désolé .

Le VEX de pyvex 10.0.3 est de sept déclarations après le `IMark`- Le numéro de la liste:

```text
t2 = GET:I64(offset=48)
t0 = Sub64(t2,0x0000000000000010)
PUT(offset=144) = 0x0000000000000008
PUT(offset=152) = t2
PUT(offset=160) = 0x0000000000000010
PUT(offset=168) = 0x0000000000000000
PUT(offset=48) = t0
```

Aucun drapeau n'apparaît. Les noms VEX sont enregistrés par leur décalage dans l'état invité, et le décalage 48 est `rsp`Les compensations 144 à 168 sont `cc_op`- Je suis là .`cc_dep1`- Je suis là .`cc_dep2`, et `cc_ndep`Le lecteur stocke le code d'opération 8, qui est la soustraction de 64 bits, et les deux opérands.`pcode.writesFlags`est vrai et `vex.computesFlags`est faux dans le rapport de harnais, et pourquoi VEX est le plus court des quatre.

Les mêmes 4 ascenseurs .`48 63 c7`- Je suis là .`movslq %edi, %rax`, la première instruction de `grade`à`-O1`, donne 5 jetons ESIL (`32,edi,~,rax,=`), 1 RzIL `set`, 1 opération de code P (`RAX = sext(EDI)`Aucun élévateur n'écrit un drapeau, car l'instruction n'en a pas.`(register, 0x0, 8) INT_SEXT (register, 0x38, 4)`: le registre décompte 0 est `RAX`, compensation `0x38`est `RDI`, et les tailles sont 8 et 4 octets.

`83 f9 04`- Je suis là .`cmpl $0x4, %ecx`, depuis le contrôle des limites `grade`, donne 30 jetons ESIL, 13 RzIL `set`Les effets, 10 opérations de code P et 7 déclarations VEX avec `cc_op`7, la soustraction à 32 bits.`%ecx`Les deux largeurs de soustraction que vous rencontrerez dans VEX sont 7 pour 32 bits et 8 pour 64 bits, et le traducteur que vous construisez porte cette table.

## Quatre bases d'adresse pour une fonction

Avant que le harnais puisse comparer quoi que ce soit, il doit être d'accord avec lui-même sur l'emplacement d'une fonction.`grade`dans `ledger-x86_64-O1.o`- Je suis désolé .

| Tool | Address of `.text` | Address of `grade` | Offset |
|---|---|---|---|
| `llvm-objdump` | `0x0` | `0x10` | `0x10 - 0x0 = 0x10` |
| radare2, rizin | `0x08000040` | `0x08000050` | `0x08000050 - 0x08000040 = 0x10` |
| angr | `0x400000` | `0x400010` | `0x400010 - 0x400000 = 0x10` |
| Ghidra | `0x100000` | `0x100010` | `0x100010 - 0x100000 = 0x10` |

La table des symboles dit:`grade`est à `.text + 0x10`Chaque outil est d'accord une fois la base retirée.`iSj`dans le radare2 et le rizin, le`.text`l'entrée de la section dans la capture de l'angr, le `.text`Le bloc de mémoire dans la capture de Ghidra, et zéro pour le balayage.`test_every_tool_places_grade_at_offset_0x10_from_its_own_base`- Ça répare ce fait.

## Où les outils sont d'accord et où ils se divisent

Le harnais dans `code/main.py`Il lit les captures, réduit chaque adresse et compare les chiffres qu'il rapporte, et les tests confirment chacun d'eux.

### Le cas de l'accord: registre à -O1

Chaque outil rapporte la même instruction qui commence dans chaque fonction: 9, 14, 13 et 18. Les quatre constructeurs de graphes rapportent 3, 3, 5 et 4 blocs et 2, 3, 5 et 4 bords, et les ensembles de bords sont identiques, paire par paire, sur radar2, rizin, angr et Ghidra.`total`sont ceux que la leçon 33 a dessinés à la main.

La seule différence est la compression.`66 90`après `grade`- Je suis là .`90`après `total`) sont décodés par le balayage, qui n'a aucune raison d'arrêter, et par l'angr, qui crée `sub_40003e`et `sub_40005f`Radar2, Rizin et Ghidra ne prétendent pas à l'un d'eux.

### Le cas vectorié: registre à -O2

`total`Il s'agit de 41 instructions en 10 blocs avec 13 bords, dans chacun des quatre constructeurs de graphiques.

### Tableau de saut: note à -O0

Ici, les outils se divisent, et la division est la leçon.

| Tool | Blocks | Edges | What the capture says |
|---|---|---|---|
| radare2 | 3 | 2 | the block at `0x62` has no successor; bytes `0x76` to `0x9a` are never reached; 8 instruction starts missing |
| rizin | 8 | 11 | the block at `0x62` carries a `switch_op` with 5 cases and 4 distinct targets |
| angr | 8 | 7 | the case blocks exist, and the transfer at `0x62` goes to an unresolvable target node |
| Ghidra | 9 | 12 | `switchD` label at `0x74` splits the block; four `COMPUTED_JUMP` flows to `0x76`, `0x7f`, `0x88`, `0x91` |

Les quatre ont décodé les mêmes 33 instructions quand ils les ont atteintes.`jmp rax`Rizin et Ghidra ont trouvé les quatre cibles distinctes que la leçon 33 a lues dans les cinq délocalisations en`.rela.rodata`. Angr répertorie les blocs de l' intérieur`grade`et n' a toujours pas de bord de la barrière de saut vers eux; sa capture enregistre le transfert à `0x62`comme conduisant à un nœud cible insoluble, et son décompilateur imprimé un calcul `goto`. Ghidra a un bloc et un bord plus que la rizine pour le même flux de contrôle parce que son analyse de commutateur a placé une étiquette sur le `jmp rax`et le modèle de bloc démarre un nouveau bloc sur une étiquette.

Le graphique de leçon-33 avait 8 blocs et 11 bords. rizin correspond exactement. Le point est de voir que 9 et 11 et 7 et 2 ne sont pas quatre mesures de code trouvées. Ce sont quatre réponses à la distance qu'un outil a suivie un saut indirect et où il a choisi de couper.

### Données en code: piégeage

`lookup`est de 24 octets avec les 4 octets de la table `31 c0 90 48`à`0x10`Le harnais énumère ses différences par rapport à la majorité:`0x10`- Je suis là .`0x12`, et `0x13`, et un début manquant à `0x14`C' est ça .`addq $0x1, %rax`Radar2, Rizin et Ghidra rapport 6 démarre, 2 blocs, 1 bord, et quitte `0x10`à `0x14`Unclaimed. Angr rapporte les mêmes 2 blocs et 1 bord pour `lookup`et puis invente .`sub_400010`sur la table des octets: 2 instructions dans un bloc de 4 octets qui se termine à `0x14`, où le deuxième bloc de `lookup`commence, donc le byte REX à `0x13`La dernière découverte de l'harnais pour cet objet est qu'aucun constructeur de graphiques ne décode.`0x10`à `0x14`, donc ces octets sont à l'intérieur d'une fonction et sont des données.

### Inventé et manqué, par outil

Dans les quatre objets, le harnais nomme 9 fonctions que Angr invente sans symbole, chacune à l'emplacement ou à la table de piège.`grade`à`-O0`Il nomme 9 intervalles que le balayage linéaire décode et aucun constructeur de graphiques ne traite comme code: les huit écarts de remplissage plus la table de piège.

## Trois décompilateurs en classe

Trois des outils capturés émettent du texte C-type pour `grade`à`-O1`La source est `switch (score / 10)`avec cinq cas, compilés à partir d'une vérification des limites et d'un tableau des valeurs de rendement.

Ghidra 12.1.4:

```c
uint grade(int param_1)
{
  uint uVar1;
  uint uVar2;

  uVar2 = param_1 / 10 - 6;
  uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);
  if (uVar2 < 5) {
    uVar1 = *(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4);
  }
  return uVar1;
}
```

angr 10.0.3:

```c
extern unsigned int g_4000b4[4];

unsigned int grade(int a0)
{
    unsigned int v1;  // ecx
    unsigned int v2;  // eax

    v1 = (unsigned int)((long long)(a0 * 1717986919) >> 0x22) + (unsigned int)(a0 * 1717986919 >> 63) - 6;
    v2 = 0;
    if (v1 <= 4)
        v2 = g_4000b4[v1];
    return v2;
}
```

Radar2 6.2.4, `pdc`- Le numéro de la liste:

```c
void sym.grade (int64_t arg1) {
    rax = edi
    rax = rax * 0x66666667
    rcx = rax
    rcx >>= 0x22
    rax >>>= 0x3f
    ecx += eax
    ecx += 0xfffffffa
    eax = 0
    v = ecx - 4
    if (((unsigned) v) <= 0) {
        eax = ecx
        rcx = rip
        eax = dword [rcx + rax*4]
    }
    return
}
```

Lisez les trois contre le pipeline.`0x66666667`et le changement de 34 en retour en `param_1 / 10`Angr a soulevé les mêmes octets et a laissé l'arithmétique comme`a0 * 1717986919 >> 0x22`- Les radars.`pdc`est l'émulation ESIL plus une pseudo-syntaxe: noms de registre, aucune variable, aucun type, et les limites vérifiées imprimées comme `<= 0`Ghidra a tapé la charge de la table.`DAT_001000b4`; angr déclaré `g_4000b4[4]`En effet, ni Ghidra ni Angr n'ont pu récupérer le nombre de cas, car à`-O1`Il n'y a plus de commutateur dans les octets.`-O0`, où la table de saut existe, Ghidra imprimé `switch(param_1 / 10)`avec des cas 6 à 10 et par défaut, et angr imprimé un calcul `goto`- Je suis désolé .

Rizin n'a pas de décompilateur et les enregistrements de capture`pdc`Le HLIL de Binary Ninja et le Ctrl Hex-Rays sont les deux sorties que cette leçon ne pouvait pas capturer.

## Faites-le

Exécutez la démo:

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape
python3 code/main.py
```

La sortie est un rapport JSON. Il s'ouvre avec les versions des outils enregistrées au moment de la capture et avec les outils présents sur votre machine, qui peuvent tous être absents. Ensuite, pour chacun des quatre objets, il donne les extensions du symbole, les espaces de rembourrage, la base de chaque outil utilisé, la fonction démarre chaque outil trouvé ou inventé, et pour chaque fonction l'accord d'instruction-démarrage, les comptes de blocs, les comptes de bord, et les bords de chaque outil manquent. Il se termine par la comptabilité des octets par outil, les notes propres des outils, une liste des résultats en phrases simples et la section IR: la sortie du traducteur pour les trois instructions à côté des comptes mesurés sur les ascenseurs capturés.

Forme attendue:

```json
{
  "captureTools": {"llvm-objdump": "Apple LLVM version 17.0.0", "radare2": "radare2 6.2.4 ...", "rizin": "rizin 0.9.1 ...", "angr": "10.0.3", "ghidra": "12.1.4"},
  "localTools": {"radare2": "absent", "...": "..."},
  "fixtures": {
    "ledger-x86_64-O0.o": {
      "bases": {"llvm-objdump": "0x0", "radare2": "0x8000040", "angr": "0x400000", "ghidra": "0x100000", "...": "..."},
      "functions": {"grade": {"blocks": {"radare2": 3, "rizin": 8, "angr": 8, "ghidra": 9}, "edges": {"radare2": 2, "rizin": 11, "angr": 7, "ghidra": 12}, "...": "..."}},
      "findings": ["radare2 never reaches bytes 0x76-0x9a inside grade that the other graph builders decode", "..."]
    }
  },
  "ir": {"translator": [{"text": "subq $0x10, %rsp", "flags": {"pcodeLike": "eager", "vexLike": "thunk"}, "...": "..."}], "observed": {"...": "..."}}
}
```

Lire `code/main.py`dans cet ordre:

1. `read_text_symbols()`lisent les titres de la section et le `STT_FUNC`Symbole de `.text`La table des symboles est la seule vérité de base que le harnais a, et`TextSection.gaps()`Il en tire le rembourrage.
2. `parse_objdump()`- Je suis là .`parse_r2_family()`- Je suis là .`parse_angr()`, et `parse_ghidra()`Chaque capture est transformée en une`ToolView`: une base, les instructions en compensation de la longueur, et une `FunctionView`Chaque parseur connaît le vocabulaire d'un outil et rien d'autre.
3. `majority()`et `ranges()`Les deux aides sont toutes deux comparaisons.
4. `function_report()`Comparer les instructions commencent par la majorité des constructeurs de graphiques, puis les blocs comptent, puis les ensembles de bord, et énumère ce qui manque à chaque outil.
5. `byte_report()`compte pour chaque octet comme code à l'intérieur d'un symbole, rembourrage, non réclamé, inventé ou manqué, et nomme les octets à l'intérieur des symboles qu'aucun constructeur de graphiques décode.
6. `findings()`Il les transforme en phrases.`compare()`réunit un rapport par objet.
7. `decode_small()`- Je suis là .`to_pcode_like()`, et `to_vex_like()`Les traducteurs sont les trois codants en forme de registre, un style passionné, un style thunk.
8. `observed_ir()`Mesure les ascenseurs capturés, et `figure_data()`Il réunit la chronologie, le tableau de scène et les preuves que montre la figure.

Il y a trois commandes de plus.`python3 code/main.py compare FIXTURE.o`imprime le rapport d'un objet. `python3 code/main.py ir 4883ec10`Il imprime les deux formulaires du traducteur pour une instruction. `python3 code/main.py figure-data`imprime les données intégrées à la figure.

Les captures ont été faites une fois par `code/fixtures/capture_tools.py`, qui enregistre chaque commande et version dans `captures.json`Si vous avez les outils, exécutez le script de capture pour rafraîchir les fichiers et lisez les fixations README pour l'environnement dont elle a besoin.

## Utilisez-le

Utilisez le harnais sur un nouvel objet dans cet ordre.

1. Vous devez créer l'objet vous-même et garder la commande du compilateur. Lisez sa table de symboles avant d'ouvrir un outil, afin de savoir combien de fonctions il y a et où est le rembourrage.
2. Capturez chaque outil une fois en mode sans tête. Conservez la sortie brute et la commande exacte à côté de l'objet. Un désaccord que vous ne pouvez pas lire à nouveau n'est pas une découverte.
3. Vérifiez la base. Choisissez une fonction et confirmez que chaque outil la place à la même compensation après que vous avez soustrait sa `.text`Si ce n'est pas le cas, arrêtez-vous et découvrez ce que l'outil a chargé.
4. Si les graphistes ne sont pas d'accord ici, le reste de la comparaison concerne le décoding et non l'analyse.
5. Comparer les blocs et les bords, et lire les notes de chaque outil avant de juger un compte.`switch_op`, une `COMPUTED_JUMP`, ou un nœud cible insoluble explique la plupart des différences.
6. Compte pour les octets. Padding que l'outil prétend est une heuristique en cours de travail. Symbole octets qu'un outil laisse sont une marche arrêtée. Symbole octets que aucun constructeur de graphique prétend sont des données.
7. Une description tirée de la documentation est utile et est un type de preuve différent d'une capture.

## Laboratoire interactif

### Laboratoire A: une fonction, quatre bases

De la `code`répertoire, ouverture Python:

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape/code
python3 -q
```

Je vais courir .

```python
from main import *
views = load_views("ledger-x86_64-O1.o")
{tool: hex(view.base) for tool, view in views.items()}
{tool: hex(view.functions["grade"].start + view.base) for tool, view in views.items()}
```

Écrivez les quatre adresses brutes avant d'exécuter la deuxième ligne et les vérifier contre la table dans cette leçon.`grade`avec `grade_label`et calculer à la main où il est assis dans Ghidra.

### Laboratoire B: la table de saut de quatre côtés

```python
report = compare("ledger-x86_64-O0.o")
report["functions"]["grade"]["blocks"]
report["functions"]["grade"]["edgesMissingPerTool"]["radare2"]
report["notes"]["rizin"]
report["notes"]["ghidra"]
report["bytes"]["radare2"]["misses"]
```

Comptez les bordures du radar2 manquants et correspondent chacune à une ligne de la`-O0`Lisez la note de Ghidra et trouvez le texte suivant:`switchD`L'étiquette dans la capture de Ghidra.

### Laboratoire C: des drapeaux impatients contre un morceau

```python
translate("48 83 ec 10")["pcodeLike"]
translate("48 83 ec 10")["vexLike"]
translate("83 f9 04")["vexLike"]
{row["name"]: (row["pcode"]["ops"], row["vex"]["statements"]) for row in observed_ir()["instructions"]}
```

Trouvez le `cc_op`constantes dans les deux sorties et expliquer pourquoi l'un est 8 et l'autre 7.`translate("48 01 c8")`et lire le `TranslateError`- Je suis désolé .

### Laboratoire D: le piège sous cinq outils

```python
report = compare("sweep-trap.o")
report["functions"]["lookup"]["instructionStarts"]
report["bytes"]["dataInsideSymbols"]
report["functionStarts"]["angr"]
```

Décodez les octets `31 c0 90 48 83 c0 01`sur le papier comme un balayage, puis comme une descente après la`jmp`- Partagez vos deux annonces avec les autres et les rapports de harnais commencent à manquer.

## Laboratoire de pratique

Ajoutez une cinquième capture et étendre le harnais.

Les exigences:

- Choisissez un outil que cette leçon n'a pas capturé et que vous pouvez exécuter sur votre machine: GNU `objdump`, LIEF avec Capstone, une session Binary Ninja Free exportée à la main, ou Ghidra avec un ensemble d'analyseur différent.`captures.json`- Je suis désolé .
- Écrivez un analyseur qui transforme la capture en un `ToolView`avec la base de l'outil, les instructions, les fonctions, les blocs où l'outil les a, et les bords où il les a.
- Ajouter l' outil à `TOOLS`et, si elle construit un graphique, à `DESCENT_TOOLS`- Exécutez la démo et lisez ce qui a changé dans les ensembles majoritaires.
- Écrivez un test qui affirme que le bloc de l'outil compte `ledger-x86_64-O1.o`et un test qui affirme ce qu'il fait avec la table des pièges.
- Écrivez un paragraphe pour la leçon qui dit, dans les mots de l'outil, ce qu'il appelle le chargement, le démontageur et le CFG.

Le succès signifie`python3 code/main.py`rapporte le nouvel outil en `captureTools`, chaque fonction dans `ledger-x86_64-O1.o`Il le garde .`allDescentToolsAgree`Les résultats de l'enquête ont été`sweep-trap.o`Dites-moi ce que le nouvel outil a fait avec les octets `0x10`à `0x14`- Je suis désolé .

## Leçon d'artisanat

Cette leçon comprend `outputs/skill-the-reverse-engineering-landscape.md`. Utilisez-le pour exécuter plusieurs outils sur un objet que vous avez construit, réduire leurs adresses à une base et rendre compte, par fonction, où ils s'accordent sur les démarres d'instructions, les blocs et les bords, et qui octroie chaque outil invente ou manque. Il nécessite des captures avec des commandes et des versions enregistrées, une table de symboles lue du fichier, et une déclaration des outils décrits uniquement à partir de la documentation.

## Vérifiez

Exécutez la démo et la suite déterministe:

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La vérification doit prouver:

- Le lecteur ELF trouve quatre symboles de fonction dans `ledger-x86_64-O1.o`avec des tailles 16, 46, 31 et 63 et deux espaces de rembourrage
- les captures de balayage contenant les instructions 86, 56, 84 et 11 pour les quatre objets
- Radar2 et rizin rapportent les mêmes instructions, blocs et bords sur `ledger-x86_64-O1.o`, et seul le radar2 l' a fait .`pdc`texte
- Marqueurs de l'angre`sub_40003e`et `sub_40005f`comme fonctions sans symbole
- Les blocs de Ghidra pour `total`sont les cinq blocs de la leçon 33 et ses flux d'appel ne sont pas des bords de graphique
- chaque outil place `grade`à l' équilibre `0x10`après l'enlèvement de sa base
- chaque outil est d' accord sur chaque instruction commence dans chaque fonction de `ledger-x86_64-O1.o`
- Les quatre constructeurs de graphes rapportent 3, 3, 5, 4 blocs et 2, 3, 5, 4 bords là-bas, avec des ensembles de bords identiques
- `total`à`-O2`Il a 10 blocs et 13 bords dans chaque constructeur de graphes
- `grade`à`-O0`a 3, 8, 8 et 9 blocs et 2, 11, 7 et 12 bords en radare2, rizin, angr et Ghidra, avec les raisons dans les notes de chaque outil
- `grade_label`à`-O0`est un bloc partout sauf angr, qui se termine un bloc à l'appel
- angr invente une fonction par écart de rembourrage plus une sur la table de piège, et aucun autre outil n'invente ou ne manque une fonction
- Les gains de balayage commencent `0x10`- Je suis là .`0x12`- Je suis là .`0x13`et perd .`0x14`sur le piège, et aucun graphiste ne décode `0x10`à `0x14`
- Les résultats donnent le nom de la plage manquée dans le radar2 et de chaque fonction inventée dans l'angr
- le manifeste enregistre une version et une commande pour les 21 fichiers de capture
- Décompiler trois outils `grade`, et seul Ghidra pliera l' idiome de division
- Le traducteur écrit CF, OF, SF, ZF, PF avec enthousiasme dans un style et une `cc_op`L'autre, ne rédige pas de registre pour `cmp`, et pas de drapeau pour `movslq`
- le traducteur refuse les formulaires de mémoire, les autres opcodes, les bits REX supplémentaires et les longueurs incorrectes
- le traducteur `cc_op`Les chiffres 8 et 7 sont égaux à ceux du VEX capturé
- les ascenseurs capturés donnent des opérations de code P 9, 1, 10 et 7, 4, 7 déclarations VEX pour les trois instructions
- La ligne de temps s'étend de 1973 à 2026 dans l'ordre, le pipeline a huit étapes et six outils avec un terme dans chaque cellule, et les deux outils fermés sont marqués comme non exécutés
- la figure interactive intègre les mêmes données que le harnais produit

## Mode de défaillance de la production

| Failure | What the weak report says | What the harness must prove |
|---|---|---|
| Raw addresses from different tools | "the tools disagree about where grade is" | one offset after each base is removed, checked on one function |
| Block count as a measure of code found | "Ghidra found the most code" | the same instruction starts, with the extra block explained by a label |
| One tool as the oracle | "objdump shows the table bytes are instructions" | the sweep is a linear sweep; three graph builders leave the bytes unclaimed |
| Padding functions counted as discoveries | "angr found seven more functions" | every one sits in a padding gap or on data, and none has a symbol |
| A stopped walk reported as dead code | "the four case blocks are unreachable" | 36 bytes missed by one tool and reached by three |
| Edge counts compared without the notes | "rizin and Ghidra disagree on grade" | the same four computed targets, one extra split at a label |
| A decompiler's arithmetic read as the program | "the function multiplies by 1717986919" | the idiom from lesson 34, folded by one tool and left by another |
| A closed tool described as if run | "Binary Ninja reports five blocks" | the figure and the table mark it as documentation only |
| Captures without commands | "we ran radare2" | the manifest with version, command, and date for every file |
| An IR with no flags read as a lossy lift | "VEX dropped the flags" | the thunk in `cc_op`, `cc_dep1`, `cc_dep2`, evaluated at the branch |

## Règlement d'exploitation

Capturez chaque outil une fois avec sa commande et sa version, réduisez chaque adresse à une base et comparez les instructions commençant avant les blocs et les blocs avant les bords. Comptez chaque octet comme réclamé, rembourrage, données ou manqué. Lisez la note d'un outil avant de juger son nombre, et dites quels outils vous ne lisez que.

## Pour en savoir plus

- [Cifuentes, "Reverse Compilation Techniques", PhD thesis, QUT (1994)](https://eprints.qut.edu.au/36820/)
- [Cifuentes and Gough, "Decompilation of Binary Programs", Software: Practice and Experience (1995)](https://doi.org/10.1002/spe.4380250706)
- [Van Emmerik, "Static Single Assignment for Decompilation", PhD thesis, University of Queensland (2007)](https://espace.library.uq.edu.au/view/UQ:158682)
- [Schwartz, Lee, Woo, Brumley, "Native x86 Decompilation Using Semantics-Preserving Structural Analysis and Iterative Control-Flow Structuring" (USENIX Security 2013)](https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/schwartz)
- [Yakdan et al., "No More Gotos" (NDSS 2015)](https://www.ndss-symposium.org/ndss2015/ndss-2015-programme/no-more-gotos-decompilation-using-pattern-independent-control-flow-structuring-and-semantics/)
- [Basque et al., "Ahoy SAILR! There is No Need to DREAM of C" (USENIX Security 2024)](https://www.usenix.org/conference/usenixsecurity24/presentation/basque)
- [Andriesse et al., "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse)
- [Flake, "Structural Comparison of Executable Objects" (DIMVA 2004)](https://dl.gi.de/items/7f573931-cb51-49ef-a12c-0452a9fb0ea4)
- [Maranzano and Bourne, "A Tutorial Introduction to ADB", Bell Labs (1977)](http://squoze.net/UNIX/v7/files/doc/18_adb.pdf)
- [Stallman, KTH lecture on GDB entering distribution (30 October 1986)](https://www.gnu.org/philosophy/stallman-kth.html)
- [Hex-Rays, "IDA: celebrating 30 years of binary analysis innovation"](https://hex-rays.com/blog/ida-celebrating-30-years-of-binary-analysis-innovation)
- [Guilfanov, "Decompiler internals: microcode", Recon Brussels 2018 (slides)](https://www.recon.cx/2018/brussels/resources/slides/RECON-BRX-2018-Decompiler-internals-microcode.pdf)
- [Hex-Rays, microcode view and maturity levels](https://docs.hex-rays.com/core/decompiler/how-tos/microcode-view)
- [Hex-Rays, IDA 9.4 release notes](https://docs.hex-rays.com/release-notes/9_4)
- [Hex-Rays, IDA Free](https://hex-rays.com/ida-free)
- [Ghidra releases (12.1.4)](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Ghidra P-code reference](https://github.com/NationalSecurityAgency/ghidra/blob/master/GhidraDocs/languages/html/pcoderef.html)
- [Ghidra decompiler source, main work flow and Action classes](https://github.com/NationalSecurityAgency/ghidra/tree/master/Ghidra/Features/Decompiler/src/decompile/cpp)
- [Ghidra analyzeHeadless README](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/RuntimeScripts/support/analyzeHeadlessREADME.md)
- [Binary Ninja, BNIL overview](https://docs.binary.ninja/dev/bnil-overview.html)
- [Binary Ninja, "Architecture Agnostic Function Detection in Binaries" (2017)](https://binary.ninja/2017/11/06/architecture-agnostic-function-detection-in-binaries.html)
- [Binary Ninja, batch processing and headless licences](https://docs.binary.ninja/dev/batch.html)
- [Binary Ninja 6.0 "Krypton" release post](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)
- [Binary Ninja Free](https://binary.ninja/free/)
- [radare2 book, ESIL](https://book.rada.re/emulation/esil.html)
- [radare2 book, decompilers](https://book.rada.re/arch/decompile.html)
- [radare2 releases (6.2.4)](https://github.com/radareorg/radare2/releases)
- [rizin, RzIL design notes](https://github.com/rizinorg/rizin/blob/dev/doc/rzil.md)
- [rizin releases (0.9.1)](https://github.com/rizinorg/rizin/releases)
- [Announcing Rizin (8 December 2020)](https://rizin.re/posts/announcing-rizin/)
- [Cutter releases (2.5.0)](https://github.com/rizinorg/cutter/releases)
- [angr documentation, CFG recovery](https://docs.angr.io/en/latest/analyses/cfg.html)
- [angr documentation, the decompiler and its analysis passes](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [angr documentation, intermediate representation (VEX)](https://docs.angr.io/en/latest/advanced-topics/ir.html)
- [angr on PyPI (10.0.3)](https://pypi.org/project/angr/)
- [pypcode on PyPI](https://pypi.org/project/pypcode/)
- [llvm-objdump command guide](https://llvm.org/docs/CommandGuide/llvm-objdump.html)
- [GNU objdump manual](https://sourceware.org/binutils/docs/binutils/objdump.html)
- [LIEF documentation](https://lief.re/doc/stable/index.html)
- [Capstone releases](https://github.com/capstone-engine/capstone/releases)
- [BinDiff releases (v8, open source)](https://github.com/google/bindiff/releases/tag/v8)
- [Diaphora](https://github.com/joxeankoret/diaphora)
- [Hopper Disassembler downloads](https://www.hopperapp.com/download.html)
- [Compiler Explorer](https://github.com/compiler-explorer/compiler-explorer)
- [LLDB announced as an LLVM subproject (2010)](https://blog.llvm.org/2010/06/new-lldb-debugger.html)
- [Boomerang decompiler](https://boomerang.sourceforge.net/)
- [Tan et al., "LLM4Decompile" (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.203/)
- [Gao et al., "DecompileBench" (Findings of ACL 2025)](https://aclanthology.org/2025.findings-acl.1194/)
- [Liu, Raff, Micinski, "When LLM Decompilers Recompile More and Preserve Less" (2026)](https://arxiv.org/abs/2609.05370)
