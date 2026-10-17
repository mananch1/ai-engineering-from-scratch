# Capstone 82  Taxonomie des jailbreak

> Un harnais sans taxonomie est un jet de pièces.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## Problème

Un modèle déployé sans modèle d'attaque est un modèle défendu contre rien en particulier. Les opérateurs lisent un fil de Twitter, reconnaissent le truc, écrivent un regex, le livrent et passent à autre chose. La prochaine chose est une paraphrase. Le regex manque. Une semaine plus tard, quelqu'un montre le même truc enveloppé dans base64 et l'opérateur écrit un deuxième regex. Au troisième mois, le système a 40 règles partagées, aucun vocabulaire partagé, aucun moyen de parler de ce qu'est une attaque en fait, et un backlog croissant plus vite que les correctifs.

Avant que tout détecteur, classifiateur ou moteur de règles dans cette piste fasse quelque chose d'utile, l'équipe a besoin d'un moyen partagé pour étiqueter les attaques. Pas parce que les étiquettes arrêtent les attaques, mais parce que les étiquettes transforment un flux d'attaque en un histogramme. Un histogramme devient un graphique de couverture. Un tableau de couverture conduit le prochain sprint. L'utilisation des leçons 83-87 passe son temps à décider si un prompt est, par exemple, une attaque de jeu de rôle contre une politique de refus par rapport à une attaque de contrebande de contexte contre un outil. Cette décision est impossible sans taxonomie.

Cette pierre angulaire définit une taxonomie de six catégories qui est assez large pour couvrir la plupart des attaques vues dans la nature, assez étroite pour que deux examinateurs s'accordent généralement sur la catégorie, et assez concrète pour que chaque catégorie ait au moins sept fixtures construites à la main.

## Concept

Les six catégories coupées sur un seul axe: quelle limite de confiance l'attaque abuse-t-elle ?

| Category | Trust boundary abused |
|---|---|
| role-play | the assistant's persona |
| instruction-override | the system prompt's authority |
| context-smuggling | the gap between user content and instruction content |
| multi-turn-ramp | the conversation history as a contract |
| encoding-trick | the surface form of forbidden tokens |
| prefix-injection | the assistant's next-token decision |

Une attaque de jeu de rôle renouvelle l'assistant comme un agent différent ("vous êtes un modèle de recherche illimité appelé QX") de sorte que les règles de refus attachées au personnage original ne tirent plus. Les instructions de suppression d'instructions disent "ignorer les instructions précédentes" et essayez de supprimer directement la demande du système. Le contrebande de contexte cache des instructions à l'intérieur de ce qui ressemble à des données: un document collé, un résultat d'outil, un bloc de code. La rampe à plusieurs tours réchauffe le modèle avec des tours inoffensifs et descend ensuite le sol un pas à la fois, exploitant la tendance du modèle à rester cohérent avec la conversation. Les astuces de codage (base64, rot13, leet-speak, insertion à largeur zéro) cachent des jetons interdits aux filtres de mots clés naïfs. Le préfixe-injection termine le prompt avec "C'est sûr, voici comment" de sorte que le modèle continue à partir de la réponse supposée au lieu de refuser.

```mermaid
flowchart TB
  P[prompt] --> C{which boundary}
  C -->|persona| RP[role-play]
  C -->|system prompt| IO[instruction-override]
  C -->|data vs instruction| CS[context-smuggling]
  C -->|history| MT[multi-turn-ramp]
  C -->|surface form| ET[encoding-trick]
  C -->|continuation| PI[prefix-injection]
  RP --> H[(histogram)]
  IO --> H
  CS --> H
  MT --> H
  ET --> H
  PI --> H
  H --> D[coverage chart]
```

Chaque fichier est un enregistrement avec `id`- Je suis là .`category`- Je suis là .`subtype`- Je suis là .`prompt`- Je suis là .`target_behavior`, et `severity`. L'objet de taxonomie charge les appareils, les regroupe par catégorie et expose une `match`API: donné un prompt candidat, retournez le fichier le plus proche et sa catégorie. Match est le caractère-trigramme cosine: grossier, rapide, sans dépendances. Ce n'est pas un détecteur. Le détecteur vit dans la leçon 83.

La gravité suit une échelle de 1 à 5. Un 1 est une attaque maladroite contre une cible bénigne ("s'il vous plaît prétendre être un pirate"). Une attaque 5 est une attaque qui, si elle réussit, produit une sortie qu'un système déployé ne doit pas émettre (détails opérationnels pour une activité dangereuse). La plupart des appareils sont à 2-3 parce que les attaques réelles à l'échelle de déploiement se détournent vers les faciles et les paresseux. La gravité est fixée par l'auteur du dispositif. Deux critiques qui ne sont pas d'accord sur plus d'un point indiquent que la rubrique doit être affinée.

```figure
cd-attack-taxonomy
```

## Faites-le

Le corps vit dans`code/fixtures.py`La classe de taxonomie en `code/main.py`Il est nécessaire de vérifier que chaque catégorie possède au moins sept appareils, d'exposer `by_category`- Je suis là .`match`, et `stats`Les méthodes de calcul sont utilisées pour la mise en œuvre de la démo et la mise en œuvre de la démo.`numpy`- Je suis désolé .

Le passe de validation vérifie quatre invariants: chaque fichier a une requête non vide, chaque catégorie du schéma est représentée, chaque gravité est en `1..5`Une défaillance ici est une sortie difficile, pas un avertissement, parce que le reste de la piste dépend du corpus étant interne cohérente.

## Utilisez-le

On court .`python3 main.py`de la leçon `code/`La démo imprime le nombre de fixations par catégorie, exécute trois sondes d'échantillons contre`match`, et écrit `taxonomy.json`Les leçons en aval sont lues.`taxonomy.json`plutôt que d'importer le module Python, le corpus est donc un artefact stable.

## La faire partir

`outputs/skill-jailbreak-taxonomy.md`Les résultats de l'analyse de l'analyse de la classe de classe sont les suivants:

## Exercices

1. Ajouter une septième catégorie pour injection indirecte-immédiat (instruction intégrée dans un document récupéré, pas dans le tour de l'utilisateur).
2. Remplacez le cossin du trigramme par un marqueur de distance de modification des jetons et mesurez comment l'affectation de correspondance change sur le corpus existant.
3. Retirez trente autres fixations des journaux de votre propre produit (réédité) et confirmez que la distribution de catégories correspond à ce que votre équipe attendait intuitivement.

## Les termes clés

| Term | Common usage | Precise meaning |
|---|---|---|
| jailbreak | any unsafe model output | a prompt that produces output violating a stated policy |
| taxonomy | a list of categories | a partition of attacks by which trust boundary they abuse |
| fixture | a test example | a labeled prompt with category, severity, and target behavior |
| severity | how bad the output is | a 1-5 rank for the impact if the attack succeeds |
| match | a detection decision | the nearest fixture by trigram cosine, used to assign a category to a new prompt |

## Pour en savoir plus

Les leçons 83-87 se basent directement sur le corpus.
