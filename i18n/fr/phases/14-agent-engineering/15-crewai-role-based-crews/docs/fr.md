# Équipes d'agents basées sur le rôle  Roles, tâches, processus

> Quatre primitives: Agent, tâche, équipage, processus. Deux formes de haut niveau: équipages (collaboration autonome, basée sur le rôle) et flux (événement-driven, déterministe). CrewAI est la mise en œuvre de référence 2026 et ses documents sont simples: "pour toute application prête à la production, commencer par un flux".

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Nommer les quatre primitifs de CrewAI (Agent, tâche, équipage, processus) et ce que chacun possède.
- Distinguer le processus de consensus séquentiel, hiérarchique et prévu; choisir un par charge de travail.
- Distinguer les équipes (basées sur des rôles autonomes) des flux (déterministique axée sur les événements) et expliquer la recommandation de production des docteurs.
- Les outils en fil avec la`@tool`décorateur et `BaseTool`sous-classe; raison sur les sorties structurées par rapport au texte libre.
- Nommez les quatre types de mémoire CrewAI et quand chacun se dépense.
- Mettre en œuvre un équipage de trois agents (chercheur, écrivain, rédacteur en chef) qui produit un résumé.
- Détectez les trois modes de défaillance de CrewAI: le gonflement rapide, l'impôt sur le gestionnaire et les transferts fragiles.

## Le problème

Les équipes qui adoptent des cadres multi-agents sont dans le même mur. " Collaboration autonome " sonne bien dans une démo. Ensuite, un client dépose un bug et vous avez besoin de répétition déterministe.

Les équipes de formation libre ne répondent pas à ces questions, mais perdent la forme exploratoire dont un agent de brainstorming a besoin.

La division de CrewAI est honnête sur le commerce. équipes pour le travail collaboratif, axé sur les rôles, exploratoire. flux pour la production axée sur les événements, propriété de code, auditable. même cadre, deux formes, choisir par surface.

## Le concept

### Quatre primitifs

La surface de l'Equipage est petite, mémorisez ceci et le reste est configuré.

- **Agent.** `role + goal + backstory + tools + (optional) llm`L'équipe de recherche a été créée pour la première fois en 1999 et a été créée en 1999 par le groupe de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de l'équipe de recherche de recherche de recherche de l'équipe de recherche de recherche de recherche de recherche de l'équipe de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche de recherche
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`- Unité de travail réutilisable.`expected_output`C'est le contrat.`context`Liste des tâches en amont dont les résultats sont transmis. `output_pydantic`une forme structurée.
- **Crew.**Container, possède la liste des`agents`, la liste des `tasks`, le `process`, et facultatif `memory`+ `verbose`+ `manager_llm`réglages.
- **Process.**Stratégie d'exécution: séquentielle, hiérarchique, consensus (planifié).

Les agents ne se voient pas directement, les tâches sont des agents de référence, l'équipage séquence les tâches, le processus décide qui choisit la prochaine tâche, c'est tout le modèle mental.

> **Validated against**CrewAI 0.86 (2026-05). Les versions plus récentes peuvent renommer ou fusionner les types de processus; vérifiez le [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)avant de s'appuyer sur une forme spécifique.

### Sequence et hiérarchie

- **Sequential.**Les tâches sont exécutées par ordre de déclaration.`context`Le coût le plus bas, le plus prévisible, utilisez-le lorsque l'ordre est fixé.
- **Hierarchical.**Un agent de gestion (appel séparé de LLM) parcourt les itinéraires entre les spécialistes.`manager_llm`le gestionnaire choisit la tâche suivante à chaque tour et peut refuser ou rediriger. Utilisez lorsque vous avez quatre spécialistes ou plus et que la commande dépend vraiment de la sortie préalable.
- **Consensus.**Les docs réservent le nom pour un futur processus basé sur le vote.

La hiérarchie ajoute un appel LLM par round (le gestionnaire) en plus de chaque appel spécialisé. Le coût des jetons peut tripler sur une course en cinq étapes.

### Équipes contre flux

C'est le cadre avec lequel les docteurs vont en 2026.

- **Crew.**Autonomie basée sur le LLM. Le cadre choisit la forme à l'heure de l'exécution. Bon pour: recherche, brainstorming, premiers plans, partout où le chemin fait partie de la réponse. Difficile à reproduire. Difficile à tester. Cheap à prototype.
- **Flow.**Graphique d'événements que vous possédez.`@start`marque l'entrée. `@listen(topic)`Chaque étape est Python simple (peut appeler un équipage en interne).

Les docteurs recommandent de produire en 2026: commencer par un flux.`Crew.kickoff()`Le flux vous donne le sentier d'audit, l'équipage vous donne l'exploration.

### Intégration des outils

Trois façons de donner un outil à un agent.

1. **`@tool` decorator.**Les fonctions pures deviennent des outils. La signature est le schéma; la documentation est la description que le LLM voit.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.**Utilisation lorsque l'outil a un état (client, cache) ou a besoin d'args structurés.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits.**CrewAI expédie des adaptateurs de première partie: `SerperDevTool`- Je suis là .`FileReadTool`- Je suis là .`DirectoryReadTool`- Je suis là .`CodeInterpreterTool`- Je suis là .`RagTool`- Je suis là .`WebsiteSearchTool`- Un câble avec une importation.

Les sorties structurées utilisent Pydantic.`output_pydantic=MyModel`La réponse de la MLL est validée par CrewAI contre le modèle et est soit forcée soit réessayée.`expected_output`Les sorties de texte libre sont bonnes pour les projets; les sorties structurées sont ce que les flux en aval peuvent consommer.

### Les crochets de mémoire

CrewAI envoie quatre types de mémoire hors de la boîte.

> **Validated against**CrewAI 0.86 (2026-05). Les dernières versions tout parcourent un système unifié `Memory`Le modèle conceptuel ci-dessous est toujours valable, mais la surface de la classe publique peut s'effondrer en une seule`Memory`point d'entrée dans les versions plus récentes; vérifier [CrewAI memory docs](https://docs.crewai.com/concepts/memory)pour l'API actuelle.

- **Short-term.**Bouffer de conversation à la fin d'une seule course.
- **Long-term.**Persistant sur les circuits. Stocké dans un vecteur DB (Chroma par défaut, swappable). Retrievé par similitude avec la tâche actuelle.
- **Entity.**"Client X est dans le plan d'entreprise". Clé par entité, pas par similitude.
- **Contextual.**Récupération en temps d'assemblage, extrait la mémoire pertinente au moment où l'agent en a besoin, pas prélèvée.

Activer l' équipage avec `memory=True`La mémoire est l'un des endroits où CrewAI gagne sa place contre les cadres plus fins; LangGraph pur exige que vous câbliez chacun d'entre eux vous-même.

### Lorsque les équipes basées sur les rôles s'adaptent

- Trois à six agents avec des rôles nommés et un flux de travail collaboratif.
- Le routage où le jugement de la LLM sur l'étape suivante fait partie de la valeur (hiérarchique).
- Où que l' équipe soit plus heureuse de lire .`role + goal + backstory`que de lire une définition graphique.

### Quand ils ne le font pas

- Les DAG déterministes avec un ordre strict. Utilisez LangGraph (leçon 13).
- Les budgets de latence sous-seconde. Hiérarchique ajoute aller-retour. Même Sequential sérialise les demandes qui incluent des histoires de fond et des sorties précédentes.
- Les boucles à agent unique. Sautez le cadre; une boucle à agent (leçon 1) plus un registre d'outils est plus court.

La leçon 17 (Agent Framework Tradeoffs) explique cela dans une matrice.

### Forme de dépendance

Indépendante de LangChain. Python 3.10 à 3.13. Utilisations `uv`Le nombre d' étoiles: voir[crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)L'intégration AWS Bedrock est documentée; les benchmarks des fournisseurs rapportent une vitesse significative par rapport à LangGraph sur les charges de travail de QA, mais la méthodologie (ensemble de données, matériel, métrique d'évaluation) n'est pas publiée, alors traitez les numéros du fournisseur-cadre comme directionnels uniquement.

### Où ce modèle va mal

- **Prompt-bloat from backstories.**Un récit de 2000 mots par agent et un équipage de cinq agents brûle le budget contextuel avant le premier appel d'outil. Gardez les récit de 200 mots.
- **Manager-LLM token tax.**Le processus hiérarchique ajoute un appel de gestion LLM avant chaque appel spécialisé. Sur une équipe de cinq tâches qui est six appels de gestion LLM au lieu de cinq, et l'appel de gestion porte la liste complète des tâches plus les sorties précédentes.
- **Brittle handoffs.**La tâche N's `expected_output`est "un schéma". La tâche N+1 la lit comme `context`Le LLM a produit quatre, les agents en aval, les annonces.`output_pydantic`sur la tâche N, la tâche N+1 lit un objet typé, pas du texte libre.
- **Crew-as-prod.**La production de l'équipage libre est effectuée sans enveloppe de flux. La variabilité de sortie est élevée; la répétition est impossible; le coup de feu sur appel ne peut pas différencier une mauvaise course contre une bonne.

```figure
ae-crew-vs-flow
```

## Faites-le

`code/main.py`Il implémentera des versions STDlib des deux formes plus un équipage de trois agents.

La forme:

- `Agent`- Je suis là .`Task`les classes de données correspondant à la surface de l'Equipage d'exploration.
- `SequentialCrew.kickoff(inputs)`Les tâches sont effectuées en ordre de déclaration, en répartissant les sorties en`context`- Je suis désolé .
- `HierarchicalCrew.kickoff(topic)`Un agent de direction choisit le prochain spécialiste à chaque tour, s'arrête à "fin".
- `Flow`avec `@start`et `@listen(topic)`Des décorateurs, une petite boucle d'événements et une trace.
- `tool(name)`Le décorateur reflète les équipes de l'Aéroport de la Croix.`@tool`la forme.
- `Memory`avec `short_term`- Je suis là .`long_term`- Je suis là .`entity`Les magasins; la similitude ridicule utilise des numpy.
- Les réponses de faux LLM sont des chaînes hardcodées déconnectées de rôle plus préfixe d'entrée.

Démo concrète: chercheur, écrivain, équipe d'éditeur produisant un bref sur "ingénierie des agents 2026 ".

- Je vais le faire.

```bash
python3 code/main.py
```

Couvertures de traces: sorties de filetage d' équipage séquentielles à travers `context`, équipe hiérarchique avec choix de gestionnaires (chercheur, écrivain, rédacteur en chef, puis "fait"), flux en cours de cours les mêmes trois étapes avec des sujets explicites (`researched`- Je suis là .`drafted`- Je suis là .`edited`), les appels à l' outil sont parcourus `@tool`, et la mémoire à long terme survivant à deux coups de pied.

La trace de l'équipage est fluide, le gestionnaire pourrait en principe réordonner la trace de flux est fixe.

## Utilisez-le

- **CrewAI Flow**Même si le flux est une étape qui appelle`Crew.kickoff()`Le flux donne la limite de vérification.
- **CrewAI Crew (Sequential)**pour des travaux de collaboration clairs, en particulier les premiers projets et les boucles d'examen.
- **CrewAI Crew (Hierarchical)**lorsque le routage dépend de la sortie et que vous avez quatre spécialistes ou plus.
- **LangGraph**(Létion 13) pour les machines d'état explicites, résumé durable, ordre strict.
- **AutoGen v0.4**(Létion 14) pour la synchronisation du modèle acteur et l'isolement des failles.
- **OpenAI Agents SDK**(Létion 16) pour les produits OpenAI-first avec des roulements et des barreaux.
- **Claude Agent SDK**(Létion 17) pour les produits Claude-first avec sous-boîtiers et magasin de séances.

## La faire partir

`outputs/skill-crew-or-flow.md`Choisissez Crew vs Flow pour une tâche et échauffe la mise en œuvre minimale. Hard rejette sur les sujets Crew-without-backstory, Flow-without-explicite, Hiérarchique avec moins de trois spécialistes.

## Les pièges

- **Backstory as flavor.**Il forme les sorties, teste trois variantes par agent, la variance est réelle, choisissez une, congélez-la.
- **Skipping `expected_output`.**Sans contrat par tâche, les tâches en aval reprennent ce que le LLM a produit.
- **Memory always-on.**Le long terme écrit chaque course. Le vecteur DB augmente. La récupération devient bruyante. La portée écrit aux tâches où le fait est persistant.
- **Manager prompt drift.**Si le routage devient bizarre, jetez-le en mode verbe et lisez.
- **Tool side effects in Crews.**Un équipage peut appeler un outil plus de fois que prévu.

## Exercices

1. Convertir l'équipage de la séquence en flux, compter les points de contact où la variabilité diminue, noter où la lisibilité diminue.
2. Ajouter la mémoire de l'entité à l'équipage: les faits sur un client persistent au cours des coupes.
3. Implémenter un processus hiérarchique où le gestionnaire refuse de se diriger vers l'éditeur jusqu'à ce que la sortie de l'écrivain ait au moins trois paragraphes.
4. - Le câble`BaseTool`Comparer la forme de la trace par rapport à la `@tool`une version décorative.
5. Ajouter `output_pydantic=Brief`à la tâche de rédacteur en chef, où `Brief`Il a`title`- Je suis là .`summary`- Je suis là .`sections`. Faites une fois que la sortie de la tâche de rédacteur a malformé JSON; vérifiez le comportement de réessayer CrewAI dans la trace.
6. Lisez l'introduction des documents de CrewAI.`crewai`Quelles garanties la version stdlib a-t-elle raté ?
7. Tu as oublié quelles traces dans la version stdlib ?

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "Unit of work" | Description + expected output + assignee + optional structured output |
| Crew | "Agent team" | Container for Agents + Tasks + Process |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensus (planned) |
| Flow | "Deterministic workflow" | Event-driven, code-owned, testable |
| Backstory | "Persona prompt" | Tone and judgment shaper for the Agent |
| `@tool` | "Function tool" | Decorator that turns a function into a tool the Agent can call |
| `BaseTool` | "Class tool" | Class-based tool with args schema, retries, async support |
| Entity memory | "Per-entity facts" | Memory scoped to a customer / account / issue |
| Long-term memory | "Cross-run memory" | Vector-backed memory that survives between kickoffs |
| Contextual memory | "Just-in-time retrieval" | Memory pulled at the moment the Agent needs it |
| Manager LLM | "Router agent" | Extra LLM in Hierarchical process that picks the next task |
| `expected_output` | "Task contract" | String that tells the Agent (and audit) what shape to return |

## Pour en savoir plus

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): concepts et voie de production recommandée
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): forme axée sur l'événement, `@start`- Je suis là .`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)Le numéro de la liste:`@tool`- Je suis là .`BaseTool`, kits d'outils intégrés
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): à court terme, à long terme, entité, contextuel
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): quand le multi-agent aide et quand il ne le fait pas
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): l'alternative de la machine d'État
