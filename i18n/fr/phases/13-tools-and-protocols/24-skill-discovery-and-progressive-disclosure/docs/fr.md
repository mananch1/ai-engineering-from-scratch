# Découverte des compétences et révélation progressive

> Une compétence devient utile avant que son corps ne soit chargé. Son nom et sa description gagnent une place dans le catalogue; ses fichiers plus profonds ne gagnent de contexte que lorsque la tâche les atteint.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22 (Agent Skills: Portable Contract and Runtime Boundary)
**Time:** ~105 minutes

## Objectifs d'apprentissage

- Construire un pipeline de découverte du système de fichiers qui sépare la portée, la validation, la politique de collision et la publication du catalogue.
- Expliquez les trois niveaux de divulgation: les métadonnées du catalogue, les instructions actives et les ressources spécifiques à la tâche.
- Références de conception afin qu'un agent puisse atteindre directement les détails requis sans charger l'ensemble du colis.
- L'espace de catalogue budgétaire indépendamment du contexte des compétences actives.
- Rejeter le trajet et le symlink échapper quand une compétence lit ses propres ressources.

## Le problème

Votre agent a 200 compétences installées.`SKILL.md`Le fichier de référence, le script et le modèle au début de la session enterraient la tâche actuelle dans une procédure non liée.

Le compromis habituel est un catalogue: montrer au modèle une identité compacte et une description de routage pour chaque compétence admissible, puis charger le corps entier seulement après sélection.

Premièrement, la découverte n'est pas seulement une recherche de fichiers récursive. Les compétences peuvent exister dans les domaines de projet, d'utilisateur, d'administrateur, de plugin ou intégrés. Deux paquets peuvent partager un nom. Un symlink peut pointer hors de la racine de confiance. Un package malformé peut consommer de l'espace de catalogue ou devenir impossible à invoquer.

Deuxièmement, la divulgation progressive peut devenir une confusion progressive.`SKILL.md`Si chaque guide pointe vers trois fichiers supplémentaires, le chargement devient une marche graphique illimitée.

Un bon temps de fonctionnement rend la découverte déterministe et la divulgation intentionnelle.

## Le concept

### La découverte est un pipeline de compilateur

Traitez le système de fichiers comme une entrée source. Ne publiez pas les chemins bruts directement sur le modèle.

```figure
skill-discovery-pipeline
```

Chaque étape doit produire des données structurées et des défaillances structurées.

- Quelles racines ont été recherchées ?
- Quels candidats ont été trouvés ?
- Quels candidats ont été rejetés, et pourquoi?
- Quel paquet a gagné une collision ?
- Quelles entrées ont été raccourcies ou omises en raison du budget?

Sans cette preuve, "le modèle n'a pas utilisé ma compétence" est presque impossible à diagnostiquer.

### La politique de mise en œuvre des règles de mise en œuvre

La spécification portable définit un ensemble de compétences, pas un chemin d'installation universel ou un ordre de priorité.

Un temps d'exécution générique pourrait utiliser ces champs de champs:

| Scope | Example root | Intended ownership |
|---|---|---|
| Workspace | `<repo>/.agents/skills/` | Project maintainers |
| User | `<user-data>/skills/` | One developer |
| Administrator | `<system>/skills/` | Machine or organization policy |
| Plugin | A signed plugin bundle | Plugin publisher and installer |
| Built-in | Runtime package | Runtime vendor |

En août 2026, le Codex documente la découverte du projet de `$CWD/.agents/skills`Il prend en charge les annuaires de compétences symboliques. Les noms dupliqués peuvent apparaître plutôt que de fusionner. Ce sont des comportements Codex, pas des exigences de `SKILL.md`; vérifier le courant [Codex skill documentation](https://learn.chatgpt.com/docs/build-skills)quand vous écrivez un adaptateur.

Le laboratoire de cours utilise un rang entier explicite pour chaque classe.`Scope`Donc le même ensemble de candidats résoud toujours de la même manière.

### Les collisions ont besoin d' une identité au-delà de `name`

Deux paquets nommés `release-readiness`L'un peut être un écart de l'espace de travail et l'autre un défaut d'utilisateur.

```json
{
  "name": "release-readiness",
  "description": "Inspect a release candidate for this repository.",
  "scope": "workspace",
  "source": "/repo/.agents/skills/release-readiness",
  "selected": true
}
```

Les politiques communes en matière de collision comprennent:

| Policy | Benefit | Risk |
|---|---|---|
| Keep every candidate | Nothing is hidden | The model sees ambiguous names |
| Highest-precedence scope wins | Simple invocation | A local package can shadow a trusted one |
| Reject duplicates | No silent shadowing | Legitimate overrides stop working |
| Qualify names by source | Explicit identity | User-facing names become longer |

Choisissez une politique pour l'hôte. Conservez les candidats rejetés ou ombragés dans le diagnostic même lorsqu'ils sont absents du catalogue de modèles.

### Trois niveaux de divulgation

Les compétences des agents décrivent les étapes de chargement.

```figure
skill-disclosure-levels
```

#### Niveau 1: métadonnées de catalogue

Le modèle a besoin de suffisamment d'informations pour distinguer la compétence des voisins.

Une description utile comporte deux clauses:

```yaml
description: Validate a release candidate and produce a readiness report. Use when the user asks whether a version, tag, or package is ready to publish.
```

La première clause indique la capacité, la seconde la limite de déclenchement, la leçon 25 évalue cette limite avec des indications positives et quasi-misses.

#### Niveau 2: instructions actives

Après activation, l'organisme doit fonctionner comme une carte et une procédure.`SKILL.md`C'est un signal de conception, pas une cible à remplir.

Le corps doit contenir:

- la limite de tâches;
- le flux de travail par défaut;
- conditions de succursale;
- des références directes à des fichiers plus profonds;
- les contrats d'outils et de scénarios;
- l'échec et le comportement d'arrêt;
- la production attendue et sa vérification.

Ne pas déplacer le flux de travail central dans une référence pour réduire le fichier d'entrée.

#### Niveau 3: ressources de soutien

Les références fournissent des données ou des proses. Les scripts fournissent des calculs déterministes. Les actifs sont copiés, remplis ou transformés en produits de livraison plutôt que traités comme des instructions.

| Directory | Model reads it? | Model executes it? | Typical content |
|---|:---:|:---:|---|
| `references/` | Yes, when needed | No | schemas, policies, domain guides |
| `scripts/` | May inspect it | Through a permitted tool | validators, converters, collectors |
| `assets/` | Only if useful | No | templates, fixtures, images, starter files |

Ces noms sont des conventions, pas des capacités magiques.

### Les références spécifiques aux branches dépassent les dépôts de sujets

Écrivez le fichier d'entrée comme une carte de décision:

```markdown
## Choose the path

- For a Python package, read `references/python-release.md`.
- For a container image, read `references/container-release.md`.
- For a documentation-only release, read `references/docs-release.md`.
- If the release combines artifact types, read only the guides for those artifacts.
```

Cela donne à chaque référence une condition de charge observable.`references/`Pour plus " ne le fait pas.

Gardez le graphique de référence peu profond.`SKILL.md`Un seul saut rend la facilité de partage des données testable et réduit les chances qu'une contrainte nécessaire n'entre jamais dans le contexte.

```figure
skill-reference-map
```

### Le budget du catalogue et le contexte actif sont des budgets différents

Je vous laisse .`c_i`être le coût du catalogue sérialisé de compétences `i`- Je suis là .`B_c`le budget du catalogue, `b_j`le coût du corps actif, et `r_k`les ressources effectivement chargées.

```text
catalog_cost = sum(c_i for every published skill)
active_cost = sum(b_j for every activated skill) + sum(r_k for every disclosed resource)
```

La réduction d'un budget ne réduit pas automatiquement l'autre. Des descriptions courtes peuvent économiser de l'espace du catalogue alors qu'un corps de 900 lignes activé continue de submerger la tâche.

Le Codex évalue actuellement la liste des compétences initiales à 2% du contexte
la fenêtre lorsque la taille de la fenêtre contextuelle est connue.
la chute ne se produit que lorsque cette taille est inconnue; il ne s'agit pas d'un deuxième plafond combiné avec
La règle des 2%: lorsque le catalogue dépasse le budget applicable,
Les descriptions peuvent être raccourcies ou omises.
La politique du codex, pas une propriété de la norme Agent Skills.

### Les voies de ressources sont une limite de confiance

Une compétence ne doit lire que des fichiers à l'intérieur de son paquet.

```text
references/../../../../.ssh/config
references/external-link -> /private/company-secrets
```

Résolvez la racine du paquet et le candidat avec la sémantique du système de fichiers, rejetez les entrées absolues et vérifiez que le candidat résolu reste sous la racine résolue. Déterminez si les liens symboliques sont autorisés avant la découverte. Si cela est permis, vérifiez la cible résolue à chaque fois.

```figure
skill-resource-containment
```

La contention de chemin ne établit pas la confiance du contenu. Une référence valide dans le package peut toujours contenir des instructions malveillantes.

### La charge doit être observable

Enregistrer les événements de divulgation sans enregistrer les secrets:

```json
{
  "event": "skill.resource.loaded",
  "skill": "release-readiness",
  "resource": "references/python-release.md",
  "reason": "candidate contains pyproject.toml",
  "bytes": 2840
}
```

La raison en fait un choix de contexte en preuve révisable.

## Faites-le

`code/main.py`construit un moteur déterministe de découverte et de divulgation.

La surface de découverte comprend:

- `Scope`pour les métadonnées de source et de priorité;
- `SkillCandidate`pour un candidat non validé au système de fichiers;
- `discover_scope(scope)`pour énumérer les répertoires de compétences immédiates;
- `resolve_collisions(candidates, precedence)`d'appliquer une politique déclarée;
- `CatalogEntry`et `build_catalog(...)`publier des métadonnées limitées;
- `CatalogBudget`pour rendre compte des entrées sérialisées sans prétendre que les caractères sont des jetons universels.

La surface de divulgation comprend:

- `load_skill_body(entry, ...)`pour l'activation de niveau 2;
- `validate_reference(skill_dir, reference)`pour la détention des sentiers;
- `load_reference(...)`pour les lectures de niveau 3 délimitées.

- Il conduit le labo.

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Ce bloc nécessite un clone local et résolve la racine du référentiel à partir de n'importe quelle
l'annuaire de travail à l'intérieur de ce clone.

La démo crée des champs temporaires de projet et d'utilisateur, insère une collision, construit un catalogue avec un budget délibérément réduit, active une compétence et tente à la fois une lecture de référence valide et une échappatoire de traversée.

### Pourquoi la découverte est superficielle

`discover_scope`vérifie les annuaires d'enfants immédiats pour `SKILL.md`Il ne traite pas chaque nid récursivement .`SKILL.md`Les données de l'installation sont fournies par le système de gestion de la sécurité et de la sécurité des véhicules.

### Pourquoi le laboratoire ne analyse pas les YAML arbitraires

Le laboratoire prend en charge la matière première scalaire nécessaire à son catalogue. Un temps de production devrait utiliser un parseur YAML sûr avec un schéma explicite, des limites de taille et désactivé la construction d'objets personnalisés. "Stdlib-only" est une contrainte d'enseignement, pas une autorisation pour inventer un dialecte partiel YAML en silence.

## Utilisez-le

Appliquer cette liste de contrôle à tout adaptateur de découverte:

1. Listez chaque racine configurée et qui peut y écrire.
2. Indiquer si des paquets liés sont autorisés.
3. Valider le nom du colis, le nom du répertoire, les métadonnées requises et la taille du corps d'entrée.
4. Préserver la source et la portée de l'identité interne.
5. Déclarer et tester le comportement de nom dupliqué.
6. Mesurer le catalogue sérialisé exact envoyé au modèle.
7. Enregistrer la raison pour laquelle un corps ou une ressource a été chargé.
8. Gardez les lectures de ressources à l'intérieur de la racine du paquet résolu.
9. Échec clair lorsqu'un fichier de référence est manquant.
10. Reconstruire le catalogue lorsque les installations ou les politiques changent.

## La faire partir

Cette leçon produit la`skill-catalog-builder`Il analyse les racines ordonnées explicitement, rejette les fichiers d'entrée symboliques et les incompatibilités entre les répertoires de noms, résout les collisions entre les domaines, rejette les duplicates de priorité égale et intègre les métadonnées sélectionnées dans les budgets d'entrée déclarés, la description et les caractères sérialisés.

Son rapport JSON contient des entrées sélectionnées, des candidats omis, des entrées omises, des erreurs de validation, des préférences et une utilisation budgétaire.

## Exercices

1. Ajoutez un champ de fonctionnement du plugin et placez-le entre l'utilisateur et la priorité intégrée.
2. Modifier la politique de collision de la priorité la plus élevée à des noms qualifiés.
3. Ajoutez une limite de taille en octets à `load_reference`Testez un fichier exactement à la limite et un octet au-dessus.
4. Créez deux descriptions qui semblent presque identiques et réécrivez-les pour que les limites de la démarche ne se chevauchent pas.
5. Ajoutez un manifeste contenant des hachages pour chaque référence et script. Détectez une ressource modifiée avant de la charger.
6. L'instrument de démonstration pour signaler les nombres de octets de niveau 1, de niveau 2 et de niveau 3 séparément.

## Les termes clés

| Term | What people say | What it actually means |
|---|---|---|
| Skill discovery | "Find every SKILL.md" | Search configured scopes, validate packages, attach provenance, and apply policy |
| Skill catalog | "The list of installed skills" | Compact model-visible routing metadata for eligible packages |
| Collision policy | "Which duplicate wins" | A declared rule for same-name candidates from different sources |
| Progressive disclosure | "Lazy loading" | Staged context admission from catalog to body to branch-specific resources |
| Reference graph | "Files linked by the skill" | The reachable resource structure and its load conditions |
| Path containment | "Stay in the folder" | Verify resolved resource targets remain inside the resolved package root |

## Pour en savoir plus

- [Agent Skills specification](https://agentskills.io/specification)pour la forme de l'emballage et les niveaux de divulgation progressifs.
- [Optimizing skill descriptions](https://agentskills.io/skill-creation/optimizing-descriptions)pour les métadonnées de routage de catalogue.
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices)pour les références directes et la taille du fichier d'entrée.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)pour les champs de découverte et les limites de catalogue du Codex actuels.
