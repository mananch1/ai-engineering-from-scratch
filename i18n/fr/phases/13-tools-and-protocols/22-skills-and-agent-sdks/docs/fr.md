# Les compétences des agents: contrat portable et limite de temps d'exécution

> Une compétence n'est pas une longue demande avec un meilleur nom de fichier. C'est un paquet découvertable d'instructions, de ressources et d'assistants exécutables qui entre dans le contexte d'un agent par un contrat de fonctionnement.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 01 (The Tool Interface), Phase 13 · 05 (Tool Schema Design)
**Time:** ~90 minutes

## Objectifs d'apprentissage

- Définir une compétence d'agent sans la confondre avec une requête, des instructions de référentiel, un outil, un crochet, un subagent ou un plugin.
- Lisez le portable `SKILL.md`contractant et le séparant des prolongations spécifiques à la durée d'exécution.
- Expliquer la découverte, la sélection, l'activation, le chargement des ressources, l'utilisation des outils et la vérification comme étapes distinctes du cycle de vie.
- Valider un ensemble de compétences avant qu'un temps de course ne le place dans le catalogue d'un agent.
- Choisissez entre une compétence, un outil MCP, un crochet, un sous-bout ou un code ordinaire pour une tâche concrète.

## Dix minutes de premier succès

Faites ceci avant la longue explication. Vous allez créer une petite compétence, installer
le groupe complet de critiques dans un hôte agent réel, l'invoquer, vérifier le
Il est donc possible de le supprimer, ce qui prouve le cycle de vie avec un résultat observable.

### Pré-vol pour le laboratoire hôte réel

Le point de contrôle de l'hôte réel nécessite Node.js, `npx`Python 3, un sélectionné
l'hébergeur compétent et écrire l'accès au projet ou à la portée utilisateur que vous choisissez
Vérifiez les commandes locales d'abord:

```bash
node --version
npx --version
python3 --version
```

Décidez de l'hôte et de la portée que vous allez utiliser avant l'installation.
l'exigence n'est pas disponible, lire cette leçon sur le site ou continuer avec
Ce qui est bien, c'est que le contrat est un peu plus long.
ne prouve pas la découverte de l'hôte, l'invocation, l'exécution de scripts en paquets, ou
Débranchez le comportement.

### 1. Commencez dans un répertoire de travail vide

Exécutez ces commandes dans n'importe quel répertoire parent où vous continuez à apprendre à travailler:

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

La commande finale ne doit pas imprimer quoi que ce soit.
le répertoire vide, donc la revue a une limite claire.

Créez un répertoire pour votre première compétence:

```bash
mkdir -p my-first-skill
```

Créer`my-first-skill/SKILL.md`avec ce contenu:

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

Vérifiez que vous avez créé le fichier dans le répertoire prévu:

```bash
test -f my-first-skill/SKILL.md
```

Aucun code de sortie et de sortie 0 signifie que le fichier existe.

### 2. Installez le paquet complet de l'examen

Restez à l' intérieur .`agent-skills-first-run`et courir:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

Choisissez l'hôte et la portée de l'agent que vous utilisez.
`skill-contract-reviewer`et la destination qu'il a écrite. `--full-depth`est
Il est nécessaire parce que la compétence de cette leçon est un ensemble de références, un
Le script, et un actif.

Réglage `SKILL_ROOT`à l'annuaire absolu indiqué par l'installateur.
être l' annuaire contenant les installations `SKILL.md`, pas la source de leçon
répertoire et non l'espace de travail actuel:

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

Si la session de l'agent était déjà ouverte, démarrez une nouvelle session ou utilisez la session de l'hôte
Ne supposez pas que chaque hôte recharge son catalogue.

### 3. Envoyez-le explicitement

Dans l'agent installé, avec `agent-skills-first-run`comme le travail
répertoire, utilisez la syntaxe prise en charge par cet hôte:

| Host | Explicit invocation |
|---|---|
| Codex | `skill-contract-reviewer`, or choose it from `/skills`, then provide the review request |
| Claude Code | `/skill-contract-reviewer` followed by the review request |
| Portable fallback | `Use skill-contract-reviewer to review the target package.` |

Utilisez les valeurs absolues imprimées pour `SKILL_ROOT`et `TARGET_ROOT`dans le
Exiger de l'hôte pour les étendre avant l'exécution et montrer l' exact
commande résolue, pas commande dépendante du répertoire de travail du processus:

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

La commande résolue doit avoir la forme suivante, sans place-holders restants:

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

Un résultat réussi a les trois propriétés suivantes:

1. L' hôte trouve .`skill-contract-reviewer`par son nom.
2. L'examen du contrat de forfait est lu et le validateur en paquets est exécuté.
3. La réponse contient un rapport de validation sans erreur structurelle pour les
   l'échantillon, plus une sélection primitive justifiée.

Les preuves de l'exécution doivent également indiquer le chemin du script, le chemin de cible, le cwd, l'exactitude
Un rapport fluide sans ces champs ne peut être utilisé pour
prouver que le script de compagnon installé fonctionnait.

Si l'hôte indique que la compétence n'est pas disponible, vérifiez l'installation
la destination, le scan ou le redémarrage une fois, et réessayer la demande explicite.
réécrire la description des compétences pour cacher une défaillance de l'installation.

### 4. Sélection implicite de l'épreuve

Commencez un nouveau tour d' agent et entrez la même tâche sans nommer la compétence:

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

Si l'hôte expose des compétences sélectionnées, enregistrez si elle a choisi
`skill-contract-reviewer`Si l'hôte n'expose pas le routage, marquer implicitement
L'invocation explicite est le retrait portable.

### 5. Nettoie-toi !

Supprimer uniquement le paquet de réviseur installé:

```bash
npx skills remove skill-contract-reviewer
```

Sélectionnez le même hôte et la même portée utilisés lors de l'installation.
La première session, une demande explicite de`skill-contract-reviewer`devrait déclarer que
Il est indisponible.`my-first-skill`pour les cours ultérieurs, ou enlever le
Le répertoire du laboratoire après avoir terminé la piste.

## Le problème

Supposons que votre équipe dispose d'un flux de travail de sortie fiable. Il trouve les changements fusionnés, vérifie les notes de migration, met à jour le journal des changements, exécute une commande d'emballage et produit une liste de contrôle de révision.

Le prompt n'a pas d'identité stable, aucune règle de découverte, aucune limite de ressources, aucune forme de paquet testable et aucune réponse aux questions de base: qui peut l'invoquer? quand le modèle doit-il le sélectionner? quels scripts peut-il exécuter? quels fichiers sont fiables? ce qui survit lorsque le contexte est compacté?

L'erreur opposée est de considérer chaque instruction réutilisable comme une compétence.`SKILL.md`produit un répertoire qui semble portable tout en dépendant du comportement indocumenté d'un hôte.

La première tâche d'ingénierie est de classer, de décider de l'artefact avant de décider comment l'emballer.

## Le concept

### Les compétences codent les connaissances procédurales

Une compétence d' agent est un répertoire dont le point d'entrée est `SKILL.md`. Le fichier d'entrée contient la matrice YAML suivie des instructions Markdown.

```figure
skill-package-anatomy
```

Le répertoire, pas seulement le fichier Markdown, est l'unité déployable.`SKILL.md`avec des références manquantes est un paquet cassé même si sa matière première est analysée.

### Les abstractions voisines

| Artifact | Primary job | Loaded or run when | What it should not impersonate |
|---|---|---|---|
| Prompt | Shape one model interaction | Included by an application or user | A versioned package with resources |
| Repository instructions | Explain one codebase's standing rules | A coding runtime enters that scope | A reusable task workflow |
| Agent skill | Supply reusable procedural knowledge | Explicit or implicit activation | A hard authorization boundary |
| MCP tool | Expose a typed remote capability | The model or application calls it | A detailed operating procedure |
| Hook | Run deterministic logic on an event | The declared event occurs | Probabilistic model routing |
| Subagent | Delegate work with separate context and state | An orchestrator creates or calls it | A static instruction bundle |
| Plugin | Distribute a larger runtime extension | The host installs or enables it | The portable skill contract itself |
| Learned skill library | Store behavior discovered through experience | A policy retrieves a prior program or trajectory | A standards-based `SKILL.md` package |

Un serveur MCP peut exposer le registre de la libération. Un crochet peut interdire les pousses directes. Un subagent peut vérifier indépendamment le candidat. Ces pièces se composent parce qu'elles conservent différentes responsabilités.

### Le mot "habilité" désigne deux idées différentes

Les systèmes de recherche appellent parfois un programme appris, une trajectoire réussie ou un fragment de politique spécifique à l'environnement une compétence. Un agent peut créer ces artefacts lors de l'exploration, les récupérer par similitude de tâche, les exécuter et réviser la bibliothèque à partir de commentaires.

Une compétence d'agent dans cette mini-track est différente. Il s'agit d'un paquet d'auteur avec un contrat déclaré du système de fichiers, des métadonnées de catalogue, une divulgation progressive, une invocation médiée par le temps d'exécution et des outils contrôlés par l'hôte.

| Dimension | Agent Skill package | Learned skill library |
|---|---|---|
| Primary unit | `SKILL.md` directory | Program, policy, trajectory, or memory record |
| Creation | Authored, generated, or curated | Usually discovered from environment experience |
| Selection | Catalog description plus runtime policy | Retrieval or policy over task state |
| Execution | Model follows instructions and calls host tools | Environment runs a stored behavior or code artifact |
| Portability | Package contract can cross compatible hosts | Often tied to one environment and action space |
| Evaluation | Routing, artifact, safety, and host compatibility | Reward, success rate, transfer, and library growth |

Les deux idées contiennent des compétences réutilisables.

### Le noyau portable

La spécification Agent Skills exige deux champs de première matière:

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name`est l'identifiant stable. il doit satisfaire aux règles de nommage du spécification et correspondre au répertoire parent. `description`Il doit indiquer ce que fait la compétence et quand elle s'applique.

Les champs portables optionnels sont:

| Field | Purpose | Portability note |
|---|---|---|
| `license` | State the terms for the package | Core specification |
| `compatibility` | State environmental requirements | Core specification |
| `metadata` | Carry string-valued extension data | Core specification |
| `allowed-tools` | Suggest pre-approved tools | Experimental; host support varies |

Le corps Markdown détient les instructions opérationnelles. Il doit définir le flux de travail, les points de décision, le comportement d'échec et les chemins directs vers les ressources de support.

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### Les extensions de temps d'exécution sont une deuxième couche

Certains hôtes acceptent des configurations de frontmatter ou de compagne supplémentaires.

| Behavior | Example host extension | Portable core? |
|---|---|:---:|
| Hide a skill from model routing while keeping direct user invocation | `disable-model-invocation` | No |
| Hide a skill from the user's command menu while allowing model routing | `user-invocable` | No |
| Show argument help in a command menu | `argument-hint` | No |
| Run the skill in delegated context | `context`, `agent` | No |
| Pin model or reasoning settings | `model`, `effort` | No |
| Register lifecycle automation | `hooks` | No |
| Disable implicit invocation in Codex | `agents/openai.yaml` policy | No |

Traitez chaque extension comme un adaptateur. Gardez le flux de travail principal valide sans lui, documenter le retrait et tester l'hôte qui le consomme. Un temps d'exécution peut ignorer un champ inconnu, le rejeter ou le préserver sans mettre en œuvre le comportement.

### La première matière est des métadonnées exécutables

Les métadonnées modifient le comportement du système avant que le corps des compétences ne soit lu.

- Un malformé .`name`peut faire échouer la découverte.
- Une vague .`description`peut rediriger les mauvaises demandes.
- Un drapeau humain peut retirer l'habileté du catalogue du modèle.
- Une allocation d'outils peut modifier si un hôte demande la permission.
- Un réglage contextuel peut déplacer l'exécution dans une session d'agent séparée.

Examinez la matière première comme le code de configuration, validez-la, la modifiez et incluez son comportement dans les évaluations.

### Le cycle de vie des compétences

```figure
skill-runtime-lifecycle
```

Chaque flèche est une limite avec ses propres modes d'échec.

1. **Discovery**trouve des paquets possibles dans des emplacements configurés.
2. **Validation**rejette les emballages malformés ou dangereux avant la publication du catalogue.
3. **Cataloging**- Il est un peu plus long.`name`et `description`- Pas le tout.
4. **Selection**décide si la compétence est pertinente.
5. **Activation**charge le corps dans un contexte visible du modèle.
6. **Disclosure**ne lit les références ou les actifs que lorsqu'une succursale en a besoin.
7. **Execution**utilise des outils d'hébergement en vertu des règles d'autorisation et d'isolement de l'hébergeur.
8. **Verification**contrôle l'artefact produit indépendamment de la demande du modèle.

L'effondrement de ces étapes provoque de mauvais modèles mentaux. Une compétence découverte n'est pas active. Une compétence active n'est pas autorisée à faire tout ce qu'elle décrit. Un appel à l'outil autorisé n'est pas la preuve que le résultat est correct.

### Les compétences et les outils sont orthogonaux

Le MCP répond: "Quelles capacités peut-elle demander à cette application, et quels sont leurs schémas?" Une compétence répond: "Comment un agent devrait-il aborder cette classe de tâches?"

```figure
skill-tool-orthogonality
```

La compétence peut nommer un outil, mais l'hôte possède le réel registre de capacités. Si l'outil est absent, la compétence doit indiquer clairement un défaut ou une défaillance.

### Les compétences et les instructions de référentiel sont des domaines différents

Les instructions de référentiel décrivent l'environnement dans lequel vous vous trouvez déjà: commandes, conventions, fichiers générés et limites.

Lorsque les deux s'appliquent, la requête utilisateur active et les règles du référentiel limitent la compétence.

### Les compétences ne s'importent pas les unes les autres

Une compétence peut diriger l'agent à invoquer une autre, mais ce n'est pas une importation au niveau de la langue.

Écrire les dépendances intermédiaires de compétences comme des bords de flux de travail observables:

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

Cela rend la dépendance testable et donne à l'hôte la possibilité de faire respecter la politique.

## Faites-le

`code/main.py`Il est utilisé pour la mise en œuvre d'un petit validateur orienté vers les normes et d'un sélecteur d'artefact.

Le validateur expose:

- `parse_frontmatter(text)`pour séparer les métadonnées du corps.
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())`pour vérifier les champs requis, les noms, les extensions inconnues, la présence du corps et les limites de portabilité.
- `ValidationIssue`et `SkillReport`pour retourner des preuves structurées au lieu d'un booléen opaque.
- `FrontmatterSyntaxError`pour les données qui ne peuvent être interprétées en toute sécurité.

Le choix est dévoilé .`TaskShape`et `select_primitives(task)`Il traite les besoins d'une tâche vers un code ordinaire, des instructions de référentiel, une compétence, un crochet, un subagent ou un outil MCP.

- Il conduit le labo.

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Ce bloc de commande nécessite un clone local et doit commencer de n'importe où à l'intérieur
Ce clone aussi .`git rev-parse --show-toplevel`peut résoudre la racine du référentiel.

La démo imprime JSON pour une compétence portable valide, une compétence étendue pour l'hôte, un paquet invalide et plusieurs décisions de forme de tâche.

### Les questions relatives à l'ordre de validation

Valider les faits structurels bon marché avant de réglementer le contenu plus profond:

```figure
skill-validation-order
```

Cet ordre empêche les erreurs secondaires d'occulter la première invariante cassée.

## Utilisez-le

Avant d'écrire une compétence, remplissez cette carte de décision:

| Question | If yes | Likely primitive |
|---|---|---|
| Does this need reusable model judgment across several steps? | The procedure is stable but decisions vary | Skill |
| Must this happen every time an event fires? | Missing one execution is unacceptable | Hook or application code |
| Does the model need an external capability with typed inputs? | The operation lives outside model context | Tool or MCP server |
| Does the work need isolated context, state, or ownership? | A separate worker returns a bounded result | Subagent |
| Is this guidance specific to one repository? | It describes local commands and constraints | Repository instructions |
| Is one interaction enough? | No package lifecycle is needed | Prompt |

De nombreux flux de travail de production utilisent plus d'une ligne. La carte empêche un artefact de prétendre fournir chaque propriété.

## La faire partir

Cette leçon produit la`skill-contract-reviewer`le paquet sous `outputs/`Il contient:

- un portable `SKILL.md`qui examine un ensemble de compétences proposé;
- les listes de contrôle de référence pour le contrat portable et la sélection primitive;
- un script de validation déterministe;
- les dispositifs de forme de tâche couvrant les instructions, les compétences, les outils, les crochets, le code ordinaire et les sous-accessoires.

Installez le paquet complet, pas seulement son fichier d'entrée:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

L'installateur du cours rapporte chaque compétence de la phase 13 copiée et écrit
`/tmp/aiefs-skills/manifest.json`Cette destination propre vérifie la forme de l' emballage;
la boucle de premier succès ci-dessus contrôle la découverte et l'invocation dans un hôte réel.

Les leçons suivantes approfondissent chaque étape du cycle de vie. La leçon 24 construit la découverte et la divulgation progressive. La leçon 25 construit la politique d'invocation et le routage. La leçon 26 sépare les autorisations du sandboxing. La leçon 27 transforme l'ensemble du package en un artefact de libération évalué.

## Exercices

1. Classifiez cinq flux de travail de votre propre équipe en utilisant `TaskShape`Défendre chaque cas où vous choisissez plus d'un primitif.
2. Ajoutez des tests de limite prouvant qu' une valeur de 500 caractères `compatibility`une valeur de 501 caractères est échouée en raison d'une erreur de spécification.
3. Ajouter une extension de temps d'exécution à la liste d'allowl. Écrire un test prouvant que le même fichier est toujours distinguable d'une compétence portable seulement.
4. Divisez une requête de 400 lignes en `SKILL.md`Un modèle de sortie, un contrat de script, un modèle de sortie, et un type d'information.
5. Conceptez une réponse d'échec pour une compétence qui fait référence à un outil MCP indisponible. Ne remplacez pas silencieusement un outil par des autorisations plus larges.
6. Revoir une compétence existante et étiqueter chaque phrase comme le routage, la procédure, la politique, le pointeur de référence ou le contrat de sortie.

## Les termes clés

| Term | What people say | What it actually means |
|---|---|---|
| Agent skill | "A saved prompt" | A discoverable directory of procedural instructions and optional resources |
| Portable core | "Fields every runtime shares" | The contract defined by the Agent Skills specification |
| Runtime extension | "Extra frontmatter" | Host-specific configuration whose behavior requires a compatible adapter |
| Activation | "The skill ran" | The skill body entered model-visible context; execution may come later |
| Skill dependency | "Import another skill" | A runtime-mediated invocation edge with availability and policy checks |
| Tool contract | "A function schema" | Inputs, outputs, permissions, side effects, errors, and evidence for a capability |

## Pour en savoir plus

- [Agent Skills specification](https://agentskills.io/specification)pour le répertoire portable et le contrat de première main.
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices)pour la portée, les instructions et l'organisation des ressources.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)pour le comportement actuel de découverte et d'invocation du Codex.
- [Claude Code skills](https://code.claude.com/docs/en/skills)pour l'invocation, l'argumentation, l'outil et les extensions de contexte délégué d'une seule période de fonctionnement.
