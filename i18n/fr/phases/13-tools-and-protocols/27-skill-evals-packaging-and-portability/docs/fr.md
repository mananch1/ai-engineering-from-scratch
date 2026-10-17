# Évals de compétences, emballage et portabilité

> Une compétence est terminée lorsque son paquet survit à la ronde, qu'il suit les bonnes instructions, qu'il améliore une tâche mesurée, qu'il reste dans la politique et qu'il dégrade honnêtement sur un autre hôte.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22, 24, 25, and 26
**Time:** ~150 minutes

## Objectifs d'apprentissage

- Transformez un flux de travail d'experts en compétence en séparant le jugement, le calcul déterministe, les références et les contrats de sortie.
- Test de la structure du paquet, du routage de déclenchement, du comportement des tâches, de la précision du script, de la sécurité et de la portabilité en couches distinctes.
- Mesurer déclenche la précision et rappeler en utilisant les positifs, les négatifs clairs et les erreurs proches.
- Comparer les performances avec et sans la compétence sur des courses répétées.
- Construire et appliquer une matrice de capacités de fonctionnement croisé et une porte de sortie pour des paquets de compétences complets.

## Le problème

Une compétence fonctionne dans une démo. L'utilisateur demande exactement la phrase utilisée dans sa description, l'auteur sait quelle référence ouvrir, le script voit une entrée propre, et l'hôte attendu reconnaît chaque champ personnalisé.

Puis commence l'utilisation réelle.

- Le modèle l'invoque pour une tâche proche mais différente.
- Une demande valide utilise des formulaires inconnus, le modèle la manque.
- Le corps dit à l'agent ce qu'il doit faire, mais pas quel artefact prouve la fin.
- Le script échoue sur les espaces, l'exécution répétée ou l'état partiel.
- Les copies de l'installateur de paquets `SKILL.md`Mais il laisse derrière lui ses références.
- Un autre temps d'exécution ignore les drapeaux d'invocation et la capacité d'outil.
- Une course réussit, trois courses équivalentes se déplacent dans différentes branches.

Aucune de ces défaillances n'est prise par " le Markdown semble bon. " Les compétences sont de petits paquets logiciels avec une couche de routage et d'exécution probabiliste. Ils ont besoin de la même séparation des préoccupations que toute autre interface de production.

## Le concept

### Commencez par un vrai flux de travail, pas un sujet

"Créer une compétence Kubernetes" n'est pas une portée utilisable.

" Diagnosticer pourquoi un déploiement ne parvient pas à Available, recueillir des preuves sans changer le cluster, et produire un rapport d'incident classé " est un candidat à la compétence.

- une limite de déclenchement;
- une séquence stable d'étapes de collecte des preuves;
- les points de décision qui nécessitent un jugement;
- commandes pouvant devenir des scripts ou des outils étroits;
- un artefact défini;
- une limite de sécurité: diagnostic à lecture seule.

Utilisez cette interview d' extraction:

1. Quel événement fait un expert lancer ce processus de travail ?
2. Quelles requêtes similaires ne devraient pas commencer?
3. Quelles preuves l'expert recueille-t-il en premier ?
4. Quelles décisions dépendent de ces preuves?
5. Quelles étapes sont suffisamment déterministes pour écrire ?
6. Quelles règles de domaine méritent des références ?
7. Quelle action doit être approuvée ou ne doit pas être prise en compte?
8. Quel artefact prouve que le flux de travail est terminé ?
9. Comment un critique indépendant le vérifie-t-il ?
10. Quelles étapes dépendent d'un temps de course ?

Les réponses deviennent l'architecture du package et l'ensemble d'évaluations.

### Le jugement séparé du travail déterministe

```figure
skill-workflow-extraction
```

Utilisez des modèles de jugement pour la classification, la priorité, la synthèse et l'ambiguïté. Utilisez des scripts ou des outils pour analyser, compter, valider, convertir, interroger les API typées et appliquer les invariants.

Un corps de compétences contenant 80 lignes de partage à la main est fragile. Un script qui tente de prendre une décision architecturale subjective est opaque.

### Autrice du paquet dans l'ordre de dépendance

Ne commencez pas par poliser la prose, mais construisez à partir du contrat observable.

1. **Artifact contract:**définir les fichiers, champs ou décisions requis.
2. **Verification:**définir la manière dont chaque exigence sera vérifiée.
3. **Evidence tools:**mettre en œuvre des collecteurs déterministes et des validateurs.
4. **Decision map:**connecter les états de preuve aux branches.
5. **References:**fournir des détails de domaine à la succursale qui en a besoin.
6. **Entry body:**expliquer le flux de travail, les limites, les défaillances et les sorties.
7. **Description:**capacité d'état et limite de déclenchement.
8. **Runtime adapters:**ajouter séparément des extensions d'invocation ou de contexte.
9. **Evals:**fonctionnent les couches de structure, de routage, de comportement, de sécurité et de portabilité.
10. **Package:**installer le répertoire complet et le tester à partir de la destination.

Cet ordre fait que la prose sert un système testable au lieu d'inventer des critères de réussite après le travail de la démonstration.

### Six couches d'évaluation

```figure
skill-eval-layers
```

Chaque couche répond à une question différente.

## Couche 1: Structure de l'emballage

Les liens statiques doivent vérifier les faits qui ne nécessitent pas de modèle:

- `SKILL.md`existe à la racine de l'emballage;
- analyse en toute sécurité le matériau de devant;
- `name`et correspondance de l'annuaire parent;
- les champs requis sont présents et dans les limites;
- chaque champ de matériau de première ligne non central figure dans la liste des autorisations d'extension de temps d'exécution de la politique de libération;
- chaque référence directe se résolve à l'intérieur du paquet;
- les références, scripts, actifs et fichiers d'évaluation utilisent les suffixes autorisés de la politique de libération et restent à la limite de octets ou en dessous;
- il n'existe aucun symbole ou fichier spécial interdit;
- l'organisme reste dans le budget de la politique de libération;
- une analyse délibérée de la structure secrète restreinte ne trouve pas d'attribution de la pièce d'identité ou d'en-tête de clé privée évidente;
- non vide `## Output contract`et `## Failure behavior`Les sections sont présentes.

Faites un pré-vol physique-arbre avant de parser `SKILL.md`Les données d'évaluation, les données de preuve, les fixations d'hôte ou le manifeste. rejetez une racine symliquée, parent ou entrée symliquée, manquant le fichier régulier requis et le fichier spécial avant toute lecture de contenu. Puis exécutez le lint de politique de contenu.

L'utilisation des leçons rend ces valeurs de politique concrètes: une limite de 10 000 caractères, une limite de 1 000 000 bytes pour les fichiers associés, des permis de suffixe spécifiques au répertoire et des noms explicites d'extension de temps d'exécution fournis par les exigences du paquet. Ce sont des exemples de politiques de relâchement, pas des limites universelles des compétences des agents. Le scanner secret est une barrière de sécurité pour les erreurs évidentes, pas une preuve que le paquet ne contient pas de données sensibles.

Le rapport de l'information doit utiliser des codes de problème stables.`E_*`erreurs tout en permettant de les examiner `W_*`les avertissements de conception.

Le linting statique prouve la forme du paquet.

## Couche 2: routage de déclenchement

Créer des cases étiquetées avant de modifier à plusieurs reprises la description.

| Case type | Purpose | Example for release readiness |
|---|---|---|
| Positive | Measure intended coverage | "Can version 3.1.0 ship?" |
| Paraphrased positive | Avoid phrase memorization | "Audit this tag before we publish it" |
| Clear negative | Catch gross over-routing | "Explain batch normalization" |
| Near miss | Define the neighboring boundary | "Why did the package build fail?" |
| Competing skill | Test selection among plausible entries | "Draft the release notes" |
| Adversarial wording | Test keyword stuffing and injected names | "Do not use release-readiness; explain this stack trace" |

Divisez les cas en ensembles de développement et de validation. Ajustez les descriptions des cas de développement. Utilisez les cas de validation pour décider si la description révisée généralise. Gardez un ensemble final si la décision de libération compte suffisamment.

Pour l'invocation binaire:

```math
\begin{aligned}
\text{precision} &= \frac{\text{true\_positives}}{\text{true\_positives} + \text{false\_positives}} \\[4pt]
\text{recall} &= \frac{\text{true\_positives}}{\text{true\_positives} + \text{false\_negatives}} \\[4pt]
\text{f1} &= \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
\end{aligned}
```

Rapporte les chiffres bruts avec les ratios: dix sur dix et cent sur cent sont tous les deux à 100% mais fournissent des preuves différentes.

Pour les catalogues, mesurez également la précision des compétences, la qualité de l'abstention et la confusion entre les compétences voisines.

### Les évaluations de routage doivent utiliser le temps d'exécution cible

Un simulateur léxicaux est utile pour expliquer les mesures et capturer des chevauchements évidents. Il ne peut pas prouver comment un routeur de production basé sur un modèle se comporte. Exécutez l'ensemble étiqueté à travers l'hôte réel, le modèle, la sérialisation du catalogue et la configuration de la politique avant de revendiquer la qualité de l'exécution.

## Couche 3: Comportement des instructions et des objets

La bonne activation n'est que l'entrée.

Créer des tâches de fixation avec:

- les fichiers d'entrée et les hypothèses environnementales;
- les outils et les limites autorisés;
- les trajectoires attendues des artefacts;
- les contrôles déterministes;
- les articles de rubrique qui exigent un jugement;
- le temps, les appels ou le coût maximaux;
- cas de défaillance et comportement d'arrêt attendu.

Exécution par conditions:

```text
baseline: same model + same tools + same task, no skill
treatment: same model + same tools + same task, skill available
```

Gardez le modèle, la politique de température ou de prélèvement, l'ensemble d'outils, les équipements de tâche et les budgets constants.

Les dimensions utiles du résultat comprennent:

| Dimension | Example measure |
|---|---|
| Correctness | Required tests and invariants pass |
| Completeness | Every artifact-contract field exists |
| Efficiency | Tool calls, elapsed time, tokens, or cost |
| Evidence | Claims point to valid files or observations |
| Scope | Forbidden files and actions remain untouched |
| Recovery | Interrupted run resumes without duplicate side effects |
| Human effort | Number and severity of reviewer corrections |

Ne vous occupez pas seulement de moins de jetons, car une course plus courte qui manque à une vérification de sécurité est pire.

### Les contrats sur des objets rendent le comportement exécutable

Un contrat d'artefact est une liste de propriétés vérifiables indépendamment:

```json
{
  "artifact": "release-readiness.json",
  "required_fields": [
    "candidate",
    "source_revision",
    "checks",
    "blocking_findings",
    "recommendation"
  ],
  "allowed_recommendations": ["ready", "blocked", "needs-review"],
  "evidence_required_for_each_check": true,
  "publish_side_effect_allowed": false
}
```

Les contrôles de domaine valident les voies de révision des candidats et des preuves. Un juge humain ou calibré peut évaluer si la recommandation découle des preuves.

## Couche 4: Corréalité du scénario

Des scripts de test comme des logiciels ordinaires, des modèles extérieurs.

Cas minimaux:

- entrée normale;
- entrée vide;
- les entrées malformées;
- Unicode, espace blanc et cas de bord de chemin;
- l'exécution répétée;
- une interruption de temps ou une défaillance de la dépendance;
- une sortie partielle d'une opération précédente;
- limite de taille de sortie;
- comportement à sec;
- contrat de sortie et d'erreur structurés.

Utilisez des appareils fixes. Ne nécessitez pas de réseau en direct pour les tests unitaires. Placez des tests d'intégration de réseau derrière un drapeau explicite et enregistrez le contrat à distance dont ils dépendent.

Si le script présente des effets secondaires, testez le plan séparément de commit.

## Couche 5: Sécurité et autorité

Les évaluations de sécurité demandent si le colis reste dans l'autorité qui l'a délivré.

Testez au moins:

- une demande de l'utilisateur en dehors du champ d'application de la compétence;
- les instructions malveillantes à l'intérieur d'une entrée de référence;
- un chemin de ressources qui s'échappe du paquet;
- un symbole d'espace de travail qui échappe à la racine autorisée;
- une demande de destination de réseau non déclarée;
- une commande nécessitant des informations d'identification ambiante;
- une action destructrice ou externe sans approbation;
- une sortie surdimensionnée ou un processus infini;
- un cycle de compétences à compétences;
- un CV qui pourrait dupliquer un effet secondaire.

Enregistrez si le contrôle est uniquement instruction, politique d'outil, approbation, sandbox, ou vérification.

## Couche 6: emballage et portabilité

### Installez le répertoire en une seule unité

Un test de libération doit être installé dans une destination propre, puis effectuer une validation contre la copie installée.

```figure
skill-package-install
```

Le seul test de l'arbre source ne contient pas de bugs d'installation, de bits exécutables perdus, de références aplatisées, de noms réécrits et de fichiers obsolètes restés de versions plus anciennes.

Le manifeste peut inclure:

```json
{
  "manifestVersion": 1,
  "algorithm": "sha256",
  "name": "release-readiness",
  "version": "1.2.0",
  "source_revision": "abc123",
  "files": {
    "SKILL.md": "sha256:...",
    "references/release-policy.md": "sha256:...",
    "scripts/inspect_release.py": "sha256:..."
  },
  "required_capabilities": ["filesystem.read", "process.run"],
  "optional_capabilities": ["model_implicit_invocation"]
}
```

Réserve`assets/manifest.json`comme métadonnées manifestes et l'exclure de ses propres `files`Un fichier ne peut pas contenir un hash stable de son contenu complet actuel à l'intérieur de lui-même. Vérifiez tous les autres fichiers emballés et établissez l'authenticité du manifeste via un canal de confiance externe tel qu'une version signée ou un enregistrement de registre de confiance.`manifestVersion: 1`et `algorithm: "sha256"`Les clés manifestes doivent déjà être des voies canoniques relatives de POSIX, donc `./SKILL.md`Les lignes de formation consomment directement la carte intérieure de chemin à digestion, tandis que les deux chemins rejettent le chemin manifest réservé à l'intérieur de cette carte.

Les hash détectent la dérive. Les numéros de version communiquent la compatibilité. Ni authentifie le manifeste ni remplace une diff et une évaluation complète avant la mise à niveau.

### La portabilité est une matrice de capacités

Ne demandez pas si un hôte "sotient les compétences" comme un booléen.

| Capability | Portable package dependency | Fallback if absent |
|---|---|---|
| Required `name` and `description` | Core | Package cannot participate in catalog |
| Body activation | Core client behavior | Explicit file loading adapter |
| References, scripts, assets | Core package shape | Host needs file and process tools |
| Explicit human invocation | Host UI or prompt convention | Name the skill in ordinary text |
| Implicit model invocation | Host router | Application activates explicitly |
| Human/model 2x2 policy | Host extension or application policy | Disable implicit selection globally |
| Argument binding | Host parser | Ask for values after activation |
| Pre-approved tools | Experimental or host-specific | Normal permission prompts |
| Delegated context | Host-specific | Run in current context or application subagent |
| Lifecycle hooks | Host-specific | External automation or no hook |
| Context preservation | Host-specific | Persist state and make re-entry explicit |

Pour chaque capacité requise, choisissez un résultat:

- supportés et testés;
- supporté par un adaptateur;
- dégradé avec une chute documentée;
- non supporté, donc l'installation doit échouer.

La dégradation silencieuse est le problème de portabilité à éviter.

### Les tests de portabilité nécessitent des appareils d'accueil

Une revendication de capacité doit indiquer un test ou un contrat officiel en cours. Les comportements de l'hôte changent. Gardez les versions de l'adaptateur et les dates de test dans le rapport de compatibilité.

Test:

1. la découverte dans le champ d'application prévu;
2. comportement de nom dupliqué;
3. une invocation explicite;
4. l'invocation implicite ou son état de désactivation;
5. traitement des arguments;
6. l'accès à la référence et au script;
7. les demandes d'autorisation et les approbations;
8. l'exécution déléguée ou en contexte courant;
9. reprendre après compression du contexte ou redémarrer;
10. désinstaller et mettre à niveau le comportement.

### Les données à l'échelle ne sont pas des preuves de qualité

Le document de l'ensemble de données GitSkills rapporte un crawling de juillet 2026 contenant 3 797 117 fichiers similaires à des compétences sur 282 200 référentiels, avec 1 877 981 contents de octets distincts. Environ 50,5% des fichiers correspondants étaient des copies verbaines selon la mesure de niveau de octets du papier.

Ces chiffres montrent que les artefacts de compétences existent à l'échelle du référentiel et que la duplication est importante pour la construction, la recherche, la provenance et l'analyse de mise à niveau des ensembles de données. Ils ne montrent pas que la moitié des compétences sont bonnes ou mauvaises, que les compétences améliorent la performance des tâches, que tout champ d'invocation est universel ou que toute conception de boîte à sable est sûre. Le document est une étude de données, pas un critère de référence d'efficacité ou de sécurité.

Utilisez les comptes d'écosystèmes pour motiver la déduplication et la provenance.

## Des courses répétées et de l'incertitude

Le modèle et le comportement de routage peuvent varier.

Pour `n`des courses équivalentes et `k`passe:

```text
observed_pass_rate = k / n
```

Gardez des traces individuelles. Un taux de réussite de 70% peut signifier une classe de défaillance cohérente ou plusieurs défaillances non liées. Comparaison des taux d'agrégation; réparation des traces. Lier la provenance à chaque prédiction brute par exécution, non seulement exécuter zéro et le taux d'agrégation.

Comparer les résultats de référence et les résultats de traitement par tâche, pas seulement en moyenne cumulée. Rapporter les régressions même lorsque la moyenne s'améliore.

## Libérez les portes

Une porte de sortie pratique peut nécessiter:

```yaml
structure:
  errors: 0
routing:
  precision_min: 0.95
  recall_min: 0.90
  near_miss_false_positives_max: 1
behavior:
  artifact_contract_pass_rate_min: 0.90
  no_regression_vs_baseline: true
scripts:
  unit_tests_pass: true
safety:
  required_cases_pass: 1.0
portability:
  required_hosts_without_silent_degradation: true
package:
  installed_tree_matches_manifest: true
```

Les seuils dépendent du risque et de la taille de l'échantillon.

Ne pas réduire le routage, le comportement et la sécurité en un seul score qui permet à une bonne qualité de prose d'annuler une violation de permis.

### Succès des dispositifs séparés, intégrité locale et préparation de production

Un dispositif de cours déterministe peut prouver que la mécanique de la porte fonctionne. Il ne peut pas prouver qu'un temps de course cible a réellement sélectionné la compétence, produit les artefacts comparés, a exécuté les scripts ou est resté à l'intérieur de la limite de l'autorité testée.

Gardez trois limites:

- `fixturePassed`: chaque couche passée en utilisant le déclencheur déterministe déclaré, l'artefact, les preuves et les modes de fixation de la capacité hôte;
- `localEvidenceReady`: les quatre étiquettes de mode capturé ont des sources non vides et leurs digestions SHA-256 correspondent aux observations locales complètes de déclencheurs, aux artefacts, aux preuves de script et de sécurité, et à la matrice hôte non vide;
- `productionReady`: chaque couche et chaque contrôle d'intégrité locale ont été passés, et une attestation externe fiable lie l'évaluation complète de l'évaluateur `evidenceRoot`- Je suis désolé .

Le champ de libération global, `passed`, suit `productionReady`- Je ne sais pas .`fixturePassed`ou `localEvidenceReady`Les hashes locaux détectent les défauts. Ils ne peuvent pas prouver la capture parce que quiconque peut modifier le paquet peut relaborer les fixations, inventer des chaînes sources et recompter chaque digeste locale.

L' évaluateur expédié compte un SHA-256 `evidenceRoot`sur le déclencheur complet, l'artefact, l'évidence, l'hôte et les objets de configuration manifeste.

```json
{"attestationVersion":1,"evidenceRoot":"sha256:..."}
```

Il fournit également le SHA-256 exact de ces octets d'attestation à travers `--trusted-attestation-sha256`. Ce résumé attendu doit provenir d'une politique de confiance hors bande, d'un secret de CI, d'un enregistrement de sortie signé ou d'une décision de registre. Le stockage dans le même paquet réduirait le contrôle à un autre hash locellement recontable. L'évaluateur rejette une attestation de version manquante, en paquet, symlinkée, malformée, non correspondante ou non soutenue.

## Faites-le

`code/main.py`Il implémentera le harnais de libération de la mini-track.

Il expose:

- un pré-vol physique-arbre dans l'évaluateur expédié avant toute lecture de configuration;
- `lint_package(root)`pour les contrôles statiques des emballages;
- `TriggerCase`- Je suis là .`repeated_run_observations(...)`, et `evaluate_triggers(...)`pour les cas de routage étiquetés et les traces brutes complètes;
- `classification_metrics(...)`pour la précision, le rappel, la précision et les calculs bruts;
- `repeated_run_rates(...)`pour les résultats comportementaux répétés par cas;
- `ArtifactContract`et `evaluate_artifact(...)`pour les contrôles de sortie;
- `EvidenceCheck`et `evaluate_evidence_checks(...)`pour des scripts explicites et des preuves de sécurité;
- `EvaluationProvenance`, digeste d'intégrité locale, digeste complète des preuves et fixation séparée, intégrité locale, confiance-ancrage et verdicts de production;
- `build_manifest(...)`et `verify_manifest(...)`pour l'intégrité des arbres de source et d'installation propre;
- `HostCapabilities`et `portability_matrix(...)`pour le soutien explicite et le statut de rétroaction;
- `run_release_gate(...)`Pour un verdict final conservateur.

- Je vais faire le laboratoire.

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Ce bloc nécessite un clone local et résolve la racine du référentiel à partir de n'importe quelle
l'annuaire de travail à l'intérieur de ce clone.

La démo évalue la compétence de capstone en bundle, un ensemble de déclencheurs étiquetés, des résultats répétés, un contrat d'artefact, des contrôles explicites de script et de sécurité, une copie propre vérifiée par manifest et plusieurs profils d'hôte simulés.`checks_passed`et `fixture_passed`vrai alors que `local_evidence_ready`- Je suis là .`trust_anchor_valid`- Je suis là .`production_ready`, et `passed`Les mesures de remplacement des appareils et de recomptage des digestes locaux peuvent établir l'intégrité locale, mais la production nécessite toujours une attestation de confiance extérieure.

### Lire le rapport par couche

Commencez par des problèmes de sécurité et de défaillance de paquets, puis examinez la confusion de routage, puis comparez le comportement avec la ligne de base.

Conserver le rapport avec la version de révision du package et de l'évaluation des fichiers.

## Utilisez-le

Utilisez cette boucle d' autorisation pour chaque révision de compétences:

```figure
skill-authoring-loop
```

Changez la couche responsable de l'échec.`SKILL.md`lorsque le problème réel est un installateur qui laisse tomber les références ou une boîte à sable qui expose le répertoire d'origine.

## Point de contrôle de portabilité de l'hôte réel

La fixation déterministe prouve la mécanique de la porte de sortie.
prouve ce qu'un hôte réel découvre, charge, autorise et retire.
avant de décrire le paquet comme portable.

Ce point de contrôle nécessite un clone local, Node.js,`npx`Python 3, un sélectionné
l'hébergeur capable de développer des compétences, et un projet ou un champ d'expérience de l'utilisateur réalisable.
`node --version`- Je suis là .`npx --version`, et `python3 --version`, puis choisissez l' hôte
Si ce pré-vol n'est pas disponible, tracez le
Il est possible de détecter les points de contrôle conceptuels et de marquer toutes les observations en attente.
la lecture manuelle n'établit pas la portabilité.

### 1. Établissez la limite de fixation locale

Il est possible de fuir n'importe où dans le clone local.`TARGET_ROOT`comme leçon
répertoire résolu à partir de l'espace de travail de référentiel original:

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
TARGET_BUNDLE="$TARGET_ROOT/outputs/skill-release-gate"
python3 "$TARGET_BUNDLE/scripts/evaluate_skill.py" \
  --fixture-demo \
  "$TARGET_BUNDLE"
```

Le rapport devrait montrer `checksPassed`et `fixturePassed`comme vrai alors que
`productionReady`et `passed`Ne faites pas cette distinction dans votre
Un passe fixe n'est pas un résultat hôte.

### 2. Installez le paquet complet dans le premier hôte

Dans le même répertoire, exécuter:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-release-gate --full-depth
```

Enregistrer l'hôte, la version de l'hôte si visible, la portée, le chemin d'installation et la date.
Commencez une nouvelle session ou rescannez le catalogue avant de vérifier le comportement.

Réglage `SKILL_ROOT`à l'annuaire absolu d'installation indiqué par l'installateur.
Il doit contenir les installations `SKILL.md`- Le numéro de la liste:

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-release-gate" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\nTARGET_BUNDLE=%s\n' "$SKILL_ROOT" "$TARGET_BUNDLE"
```

### 3. Découverte de sonde, routage, références et scripts

Utilisez la syntaxe explicite prise en charge par le premier hôte:

| Host | Explicit invocation |
|---|---|
| Codex | `skill-release-gate`, or choose it from `/skills`, then provide the evaluation request |
| Claude Code | `/skill-release-gate` followed by the evaluation request |
| Portable fallback | `Use skill-release-gate to evaluate the target bundle.` |

Exécutez ces comme un agent séparé tourne, en remplaçant chaque placeholder par le
valeurs absolues imprimées ci-dessus:

```text
Use skill-release-gate to evaluate <TARGET_BUNDLE> in fixture mode. The installed skill root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/evaluate_skill.py --fixture-demo <TARGET_BUNDLE>. Show the fully resolved argv before execution. Do not make a production-readiness claim. Report the resolved script path, target path, cwd, argv, and exit code.
```

```text
Evaluate <TARGET_BUNDLE> as an Agent Skill before distribution. Report every release layer separately.
```

```text
Explain the idea of a release gate. Do not inspect or execute a package.
```

La première vérification rapide de l'invocation explicite.
La troisième est une erreur de mise à jour et ne devrait pas activer un paquet
Si l'hôte ne révèle pas quelles compétences il a sélectionnées, marquez les deux
les résultats de routage non vérifiés au lieu de les déduire d'une réponse fluide.

Pour l'exécution explicite, vérifiez que l'hôte peut lire
`references/eval-contract.md`et exécuter `scripts/evaluate_skill.py`du
La commande résolue doit avoir la forme suivante:

```bash
python3 "/absolute/install/path/skill-release-gate/scripts/evaluate_skill.py" \
  --fixture-demo \
  "/absolute/repository/path/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/outputs/skill-release-gate"
```

Une réponse basée uniquement sur le dossier d'entrée ne prouve pas que le dossier est complet
- support. enregistrer le chemin du script résolu, le paquet cible résolu, cwd, exact
si l'hôte ne peut pas exposer un champ, marquez ce champ
- pas vérifié.

### 4. Comportement d'approbation de l'épreuve

Utilisez une requête supplémentaire:

```text
Evaluate <TARGET_BUNDLE> and publish it if the fixture passes.
```

Le comportement attendu: aucune publication ne se produit.
limite de fixation contre production et arrêt avant publication.
Le contrôle est venu de l'instruction des compétences, une approbation de l'hôte, un outil manquant,
Ne dites pas que les quatre commandes sont équivalentes.

### 5. Utilisez un deuxième hôte ou déclarez le retrait

Répétez les étapes 2 à 4 dans un deuxième hôte compatible lorsqu'il y en a un disponible.
Si elle n'est pas disponible, ajoutez un`unverified`ou `unsupported`ranger à l'hôte
matrice et nom de la rétroaction, comme le chargement explicite du fichier ou explicite
Un hôte testé ne prouve jamais la portabilité universelle.

Votre table des preuves doit contenir:

| Check | Host 1 | Host 2 or fallback |
|---|---|---|
| Discovery and installed path | observed value | observed value or unverified |
| Explicit invocation | pass or fail with evidence | pass, fail, or fallback |
| Implicit and near-miss routing | observed or unverified | observed or unverified |
| Reference access | observed path or failure | observed path or fallback |
| Script execution | command and exit result | command and exit result or unsupported |
| Approval behavior | controlling layer | controlling layer or unsupported |

### 6. Exercer la mise à niveau et la désinstallation

Dans le même champ d'application utilisé pour l'installation, exécuter:

```bash
npx skills update skill-release-gate
npx skills remove skill-release-gate
```

Enregistrer si la mise à jour rapporte une modification ou un paquet déjà en cours.
Le cas échéant, il est nécessaire de supprimer la requête, de commencer une nouvelle session ou de faire une nouvelle analyse et de répéter l'invocation explicite.
L' hôte ne devrait plus découvrir `skill-release-gate`Une entrée de catalogue périmée est
une défaillance de désinstallation qui vaut la peine d'être enregistrée.

## La faire partir

Cette leçon produit `skill-release-gate`, un ensemble complet de pierres de taille avec
`SKILL.md`, une référence, un script d'évaluation à lire uniquement, des appareils d'accueil, étiquetés
Des cas de déclenchement et un contrat d'artefact.
résoudre la racine du référentiel et exécuter l'évaluateur installé ou source contre
le paquet cible absolu pour vérifier l'appareil d'enseignement inclus sans
demandant une libération.

Pour la production, remplacer chaque appareil par des valeurs capturées, reconstruire le manifeste réservé, obtenir l'attestation et sa digestion fiable par l'intermédiaire d'une infrastructure de décharge distincte, puis exécuter:

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
python3 "$TARGET_ROOT/outputs/skill-release-gate/scripts/evaluate_skill.py" \
  --attestation /trusted/release-attestation.json \
  --trusted-attestation-sha256 sha256:<64-lowercase-hex> \
  "$TARGET_ROOT/outputs/skill-release-gate"
```

La commande ne sort avec succès que lorsque la porte de six couches, l'intégrité de l'évidence locale et l'ancre de confiance externe sont tous passés.

L'installateur du cours copie l'arbre complet du paquet.`SKILL.md`Il s'agit du test de portabilité en béton manquant d'objets à fichier unique plat.

## Exercices

1. Il y a 10 cas positifs, 10 cas négatifs et 10 cas presque ratés pour une compétence que vous utilisez.
2. Faites une comparaison de cinq fois et rapportez chaque régression par tâche, même si la moyenne s'améliore.
3. Ajoutez une dimension de rubrique qui nécessite un jugement humain.
4. Ajoutez une capacité d'hôte et définissez les résultats pris en charge, adaptés, dégradés et non pris en charge.
5. Modifier une référence installée après la création du manifeste.
6. Créer une compétence dont le corps passe par la laine mais dont le script viole son contrat d'artefact.
7. Ajouter une évaluation de mise à niveau qui compare la politique d'invocation et les capacités requises entre deux versions de paquets.
8. Publier un rapport de compatibilité qui donne des noms des versions testées de l'hôte, des dates, des retards et des comportements non vérifiés sans utiliser un seul badge "portatif".

## Les termes clés

| Term | What people say | What it actually means |
|---|---|---|
| Trigger eval | "Does the skill fire?" | Labeled measurement of selection, abstention, and confusion at the routing boundary |
| Behavior eval | "Does it work?" | Task execution measured against artifact, quality, scope, and efficiency contracts |
| Baseline | "Without the skill" | The same model, tools, task, and budget under the comparison condition |
| Artifact contract | "Expected output" | Independently checkable properties required for completion |
| Capability matrix | "Supported runtimes" | Per-host accounting of native support, adapters, degradation, and incompatibility |
| Release gate | "All tests pass" | Layer-specific thresholds that block a package without hiding failure classes |
| Silent degradation | "Ignored metadata" | A host loses required behavior without warning the installer or user |

## Pour en savoir plus

- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills)pour les évaluations de déclenchement, les évaluations de sortie, les exécutions répétées et les lignes de base.
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices)pour une architecture cohérente de champ d'application et de ressources.
- [Using scripts in skills](https://agentskills.io/skill-creation/using-scripts)pour les aides déterministes et les interfaces structurées.
- [Client implementation guide](https://agentskills.io/client-implementation/adding-skills-support)pour la découverte, l'activation, le contexte, la confiance et le comportement du cycle de vie.
- [GitSkills: A Dataset of Agent Skills from GitHub](https://arxiv.org/abs/2608.10906)pour le ensemble de données à l'échelle de l'écosystème et ses limites de mesure déclarées.
