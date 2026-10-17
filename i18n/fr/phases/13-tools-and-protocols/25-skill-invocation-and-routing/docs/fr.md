# Apprendre à utiliser des compétences et à les orienter

> Une bonne description aide le modèle à choisir; une bonne politique décide si ce choix est autorisé.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 24 (Skill Discovery and Progressive Disclosure)
**Time:** ~105 minutes

## Objectifs d'apprentissage

- Distinguer entre l'invocation explicite de l'utilisateur, l'invocation implicite du modèle, l'invocation de l'application et l'invocation de compétences.
- Modéliser la visibilité humaine et l'éligibilité comme dimensions politiques indépendantes.
- Écrivez des descriptions de routage avec des déclencheurs positifs et des limites de proximité.
- Séparation de l'éligibilité, de la sélection, de l'activation, de la liaison des arguments et de l'exécution dans les traces et les tests.
- Adapter les champs d'invocation spécifiques à l'heure d'exécution sans les présenter comme une matrice de front portable.

## Le problème

Vous installez une`database-migration`La compétence. L'utilisateur peut l'exécuter par nom, mais le modèle voit également sa description et la sélectionne lorsque quelqu'un pose une question de base de données générale. La compétence propose ensuite un changement de schéma pour une tâche qui ne nécessite qu'une explication.

Vous ajoutez`user-invocable: false`Dans un autre temps d'exécution, ce champ est ignoré.`disable-model-invocation: true`Dans le temps de fonctionnement qui le comprend, l'utilisateur peut encore l'invoquer explicitement.

Il n'y a rien de mal avec les noms de champs. Le modèle est faux. "Utilisateur peut le voir", "modèle peut le sélectionner", "application peut le prélouer", et "outils à l'intérieur de celui-ci peuvent exécuter" sont des faits distincts.`invocable`Je ne peux pas les exprimer.

Le routage a un deuxième mode d'échec. Si les descriptions sont vagues, plusieurs compétences deviennent plausibles. Si les descriptions sont remplies de mots clés, des tâches non liées les déclenchent. Le catalogue est une interface probabiliste: assez compacte pour s'adapter, assez spécifique pour rouvrir.

## Le concept

### Cinq canaux peuvent déclencher le cycle de vie

| Actor | Invocation shape | Typical use | Main risk |
|---|---|---|---|
| Human user | Names a skill in the UI or prompt | Deliberate workflow selection | User expects availability or authority the host does not grant |
| Model or autonomous agent | Selects a catalog entry from task context | Automatic expert procedure | False-positive routing |
| Application | Activates or preloads a skill through runtime code | Fixed product workflow | Hidden coupling to one host |
| Another skill or subagent | Requests an exact skill as a workflow dependency | Composition | Cycles, missing dependency, or context bleed |
| Evaluation harness | Activates an exact skill under a fixed scenario | Repeatable measurement | Tests the skill while accidentally bypassing the production policy under study |

La spécification portable Agent Skills définit le package. Elle ne normalise pas une interface utilisateur universelle de commandes de slash, un flag implicit de routage, une API d'application ou un cycle de vie de subagent.

### Les cinq étapes d'invocation

```figure
skill-invocation-stages
```

Utilisez ces mots précisément:

- **Eligible**La politique permet à cet acteur de demander la compétence.
- **Selected**signifie que l'utilisateur l'a nommé ou qu'un routeur l'a jugé pertinent.
- **Activated**signifie que ses instructions sont entrées dans le contexte de travail.
- **Executing**signifie que l'agent a commencé à travailler sur le modèle ou l'outil conformément à ces instructions.
- **Completed**signifie que la sortie a été contrôlée indépendamment.

Une trace qui n' enregistre que .`skill_used=true`cache la limite où un échec s'est produit.

### L'invocation humaine et le modèle forment une matrice 2x2

| Human can invoke | Model can invoke | Mode | Suitable examples |
|:---:|:---:|---|---|
| Yes | Yes | Shared | Code explanation, test planning, documentation review |
| Yes | No | Human-only | Publish preparation, billing export, destructive cleanup plan |
| No | Yes | Model-only | Internal style guide, domain reference, automatic support procedure |
| No | No | Disabled or application-only | Staged rollout, deprecated package, programmatic preload |

La matrice est un modèle de politique, pas YAML standard.

Un hôte actuel utilise `disable-model-invocation: true`pour la ligne uniquement humaine et `user-invocable: false`Pour la ligne modèle seulement. Le défaut est les deux. Un autre hôte utilise `agents/openai.yaml`avec `allow_implicit_invocation: false`Il est possible que les hôtes inconnus les ignorent.

Le détail confus est important:`user-invocable: false`ne signifie pas "le modèle ne peut pas utiliser ceci". Il supprime l'invocation directe de l'utilisateur dans l'hôte qui le définit. `disable-model-invocation: true`Il supprime la sélection initiée par le modèle tout en conservant un accès explicite à l'utilisateur.

### L' invocation explicite est d'abord identité

Une invocation explicite fournit directement l'identité:

```text
/release-readiness v2.4.0
```

ou

```text
release-readiness check v2.4.0 without publishing
```

Document d' interfaces du Codex actuel `/skills`Pour les demandes d'invocation explicite, il est nécessaire de choisir et de désigner des compétences simples.`/skill-name`La syntaxe exacte, la visibilité du menu, les règles de citation et l'expansion des variables appartiennent à l'hôte.

Une demande explicite passe toujours la politique. La nomination d'une compétence ne doit pas contourner les autorisations manquantes, les contraintes de l'espace de travail, les portes d'approbation ou l'isolement en temps d'exécution.

### L'invocation implicite est la description-première

Pour le routage implicite, le modèle voit initialement les métadonnées du catalogue plutôt que le corps complet.

Faible:

```yaml
description: Helps with releases.
```

- sur-large:

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

Réservé:

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

La version limitée contient:

1. **Capability:**inspecter un candidat préparé.
2. **Output:**rapport de préparation.
3. **Positive boundary:**demande si un artefact de libération est prêt.
4. **Negative boundary:**Les constructions et les développements ordinaires sont hors de portée.

Les limites négatives sont utiles lorsque deux compétences proches partagent le vocabulaire.

### Le routage est une classification avec une option d'abstention

Pour une compétence .`s`et demande `x`, imaginez un score du routeur:

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

Le score exact peut être une décision de LLM plutôt que d'arithmétique.

```figure
skill-routing-abstention
```

Pour les compétences à fort impact, le routage implicite peut être inapproprié même avec une description forte.Utilisez une politique uniquement humaine lorsque le coût d'un faux positif dépasse la commodité de la sélection automatique.

### L'éligibilité doit précéder le classement

Ne marquez pas chaque compétence découverte, choisissez la meilleure et vérifiez ensuite la politique d'une compétence.

Utilisez cette commande pour l'itinéraire implicite:

1. Filtre découvert les compétences de l'acteur demandeur et l'adaptateur hôte actif.
2. N'évaluez que les candidats admissibles.
3. Sélectionnez le match admissible le plus fort si il élimine les règles de seuil et d'ambiguïté.
4. S'abstenir lorsqu'aucun candidat n'est admissible ou qu'aucun score admissible n'est suffisamment élevé.

Supposons que`incident-triage`les résultats `0.80`mais son extension hôte désactive l'invocation du modèle. `incident-review`les résultats `0.55`Le routeur doit évaluer`incident-review`Il ne devrait pas choisir.`incident-triage`Ne pas le dire et arrête.

Cette ordonnance empêche également les changements de politique de modifier la signification d'un score de pertinence.

### Les évaluations de routage nécessitent des échecs proches

Les cas positifs prouvent le rappel:

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

Les négatifs clairs prouvent une précision fondamentale:

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

Les défaillances proches exposent la qualité limite:

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

Les actions de la quasi-mère .`package`et `build`Un ensemble de routings composé uniquement de positifs évidents et de négatifs non liés surestimera la qualité.

### Les arguments ont trois représentations

Un argument d'invocation traverse plusieurs limites:

```figure
skill-argument-boundaries
```

À chaque limite, préservez l'intention sans traiter le texte comme du code.

- Le parseur hôte décide de la syntaxe et de la citation des commandes.
- La compétence reçoit du texte lié ou des variables selon les règles de l'hôte.
- Les instructions valident les valeurs requises et les défauts.
- Un appel à outils convertit les valeurs en un schéma typé et les révalides.

Ne pas interpolier les arguments bruts dans des commandes de shell. préférer un script invoqué avec un vecteur d'argument ou un outil MCP typé.

### L'invocation d'une demande est une orchestration explicite

Un produit peut activer une compétence parce que son flux de travail connaît déjà le type de tâche.`pull-request-risk-review`après que l'utilisateur ait appuyé sur Review.

Cela élimine l'incertitude de routage mais crée une dépendance à l'API de l'exécution.

```figure
skill-host-adapter
```

La compétence doit rester compréhensible lorsqu'elle est ouverte par un client différent.

### L'invocation des compétences est un avantage similaire à un outil

Supposons que`release-readiness`Il demande`security-change-review`lorsque les fichiers de dépendance ont changé.

L'appelant doit fournir:

- l'identité des compétences cibles;
- une tâche et un artefact délimités;
- le contrat de réponse prévu;
- la raison de l'invocation;
- une chute si elle n'est pas disponible;
- une règle de profondeur maximale ou de cycle.

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

La deuxième compétence n'est pas collée à la première. L'hôte décide comment l'activer et si elle partage le contexte, s'exécute dans une fourchette ou revient à travers un résultat d'outil.

### Le cycle de vie du contexte est spécifique à l'hôte

Après activation, le corps de compétences peut rester dans la conversation, être résumé pendant la compression ou exécuter dans un contexte délégué.

Ne rédigez pas une compétence qui dépend d'une hypothèse invisible de durée de vie.

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## Faites-le

`code/main.py`Il implémentera la politique et le routage en tant qu'adaptateurs distincts.

Le modèle comprend:

- `Actor`pour les appelants humains, modèles, agents autonomes, applications, compétences et harnais;
- `SkillMetadata`pour l'identité de routage;
- `InvocationPolicy`pour la matrice humaine/modèle;
- `InvocationRequest`et `InvocationDecision`pour les entrées et les résultats traçables;
- `CorePolicyAdapter`pour un comportement portable sans extensions hôte;
- `ExtensionPolicyAdapter`pour les champs de temps de course reconnus;
- `build_invocation_matrix(policy)`pour la vue 2x2;
- `route_request(skills, request, adapter)`pour le filtrage de l'éligibilité avant le classement, la sélection et le refus de pertinence.

- Je vais le faire.

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La démo imprime une matrice et des décisions pour les modèles humains explicites, implicites, agents autonomes, applications, composition de compétences et canaux d'exploitation. Ses résultats de l'adaptateur d'extension montrent qu'un lien lexical supérieur bloqué est supprimé avant que l'alternative admissible ne soit classée. Il inclut également des listes d'allowlistes de noms exacts. Aucune API modèle n'est requise. Le routeur déterministe existe pour rendre les limites de la politique inspectables, et non pour prétendre que le correspondement léxique reproduit le routage du modèle de production.

### Pourquoi les adaptateurs de base et d'extension sont séparés

Si un parseur assigne un sens à chaque champ de frontmatter observé, il favorise silencieusement les conventions de temps d'exécution en un faux standard.

Le `CorePolicyAdapter`Il est possible de modifier la politique de l'application en utilisant uniquement les formulaires de politique fournis par les candidatures.`ExtensionPolicyAdapter`reconnaît un ensemble explicite de champs hôtes et d'enregistrements dans lequel le champ a modifié la décision.

## Utilisez-le

Écrivez un contrat d'invocation avant de publier une compétence:

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

Ce contrat est une documentation de conception pour les adaptateurs et les essais.`SKILL.md`la matière première, à moins qu'une norme ne l'adopte explicitement.

## La faire partir

Cette leçon produit la`skill-invocation-router`Il comprend une référence de modèle d'invocation, une politique d'hôte d'exemple et un CLI non exécutant qui évalue un humain, un modèle, un agent autonome, une application, une composition de compétences ou une demande d'exploitation et renvoie une décision JSON avec canal, adaptateur, score et raison.

La CLI à une seule demande est une enquête de politique, pas une évaluation complète du déclencheur. Utilisez la conception labellée positive et quasi-miss dans la leçon 27 pour calculer le nombre de confusions, la précision, le rappel et la stabilité des exécutions répétées.

## Exercices

1. Créer les quatre rangées de la matrice humaine/modèle et écrire un cas d'utilisation légitime pour chacune.
2. Ajouter à  une activation uniquement pour application`CorePolicyAdapter`Prouver que les appelants humains et modèles restent démenti.
3. Écrivez dix manquements proches pour une compétence de déploiement.
4. Ajoutez une marge d'ambiguïté entre les deux meilleurs scores de routage.`ask`lorsque la marge est trop petite.
5. Ajouter une profondeur maximale de composition aux demandes de compétences et détecter un cycle de deux compétences.
6. Exécutez le même ensemble étiqueté à travers les adaptateurs de base et d'extension.

## Les termes clés

| Term | What people say | What it actually means |
|---|---|---|
| Explicit invocation | "Slash command" | An actor supplies skill identity directly, subject to policy |
| Implicit invocation | "The model chooses" | A router selects from eligible catalog metadata based on task context |
| User-invocable | "Humans can use it" | A host-specific menu or direct-invocation property, not a core field |
| Model-invocable | "The agent can use it" | Eligibility for implicit model selection under host policy |
| Invocation adapter | "Frontmatter parser" | Code that maps a host's fields and APIs into a declared policy model |
| Near miss | "Hard negative" | A non-triggering request that resembles a skill's intended inputs |
| Abstention | "No skill selected" | A deliberate routing result when evidence is absent or ambiguous |

## Pour en savoir plus

- [Optimizing skill descriptions](https://agentskills.io/skill-creation/optimizing-descriptions)pour les déclencheurs positifs, la spécificité et l'évaluation.
- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills)pour la conception de l'évaluation de déclenchement et de sortie.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)pour les contrôles d'invocation explicites et implicites du Codex en vigueur.
- [Claude Code skills](https://code.claude.com/docs/en/skills)pour un hôte `user-invocable`- Je suis là .`disable-model-invocation`Les résultats de la recherche ont été obtenus en fonction des résultats obtenus.
