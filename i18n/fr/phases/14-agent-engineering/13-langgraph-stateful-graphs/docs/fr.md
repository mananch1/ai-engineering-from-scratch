# Orchestration de graphes d'état  Exécution durable et points de contrôle

> L'agent est une machine d'état; les nœuds sont des fonctions; les bords sont des transitions; l'état est placé en checkpoint après chaque nœud. Résumez à partir de toute défaillance au dernier point de contrôle réussi. LangGraph est la référence 2026 pour ce modèle d'orchestration d'état à bas niveau.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Décrivez le modèle de base de LangGraph: machine d'état avec état typé, nœuds de fonction, bords conditionnels et points de contrôle post-nœud.
- Nombre des quatre capacités que les documents mettent en évidence: exécution durable, streaming, humain-in-the-loop, mémoire complète.
- Expliquez les trois topologies d'orchestration que LangGraph prend en charge: superviseur, peer-to-peer (swarm), hiérarchique (subgraphes nidifiés).
- Implémenter un graphique d'état stdlib avec l'état typé, les bords conditionnels et un cycle de contrôle/résumé.

## Le problème

Les agents et les flux de travail partagent un problème: lorsqu'une mise en œuvre de 40 étapes échoue à l'étape 38, vous voulez reprendre à partir de l'étape 38, et non recommencer.

La réponse de conception de LangGraph: l'état est un objet de première classe, les mutations sont explicites et les points de contrôle persistent après chaque nœud.`load_state(session_id)`Je vous appelle.

## Le concept

### Le graphique

Un graphique est défini par:

- **State type.**Un dicté typé (ou modèle Pydantic) que chaque nœud lit et mutent.
- **Nodes.**Des fonctions pures`(state) -> state_update`Les mises à jour sont fusionnées dans l'état après leur retour.
- **Edges.**Transitions conditionnelles ou directes entre les nœuds.
- **Entry and exit.** `START`et `END`Les nœuds de sentinelle marquent la frontière.

Exemple: un agent avec `classify`- Je suis là .`refund`- Je suis là .`bug`- Je suis là .`sales`- Je suis là .`done`les nœuds  un flux de travail de routage en tant que graphique.

### Exécution durable

Après chaque nœud retourne, le runtime sérialise l'état et l'écrit à un point de contrôle (SQLite, Postgres, Redis, personnalisé).`resume(session_id)`et reprendre à partir de l'étape N + 1 avec l'état exact.

Les documents LangGraph mettent explicitement en évidence les utilisateurs de production où cela compte: Klarna, Uber, JP Morgan.

### Retour en continu

Chaque nœud peut produire une sortie partielle. Le graphique transmet des événements par nœud-delta à l'appelant afin que les UI soient mis à jour au fur et à mesure que le graphique fonctionne.

### Les humains dans le cycle

Inspecter et modifier l'état entre les nœuds. Implémentations: pause avant un nœud critique, état de surface à un humain, accepter les modifications, reprendre. Le point de contrôle le rend facile parce que l'état est déjà sérialisé.

### La mémoire

À court terme (dans un cours  histoire de conversation dans l'état) et à long terme (à travers les cours  persistant via le point de contrôle plus un magasin à long terme séparé). LangGraph s'intègre avec des systèmes de mémoire externes (Mem0, personnalisé) via des outils.

### Trois topologies

1. **Supervisor.**Le routeur central LLM envoie des subagents spécialisés. `create_supervisor()`dans `langgraph-supervisor`(bien que l'équipe de LangChain en 2026 recommande de le faire via des appels à l'outil directement pour plus de contrôle de contexte).
2. **Swarm / peer-to-peer.**Les agents se détachent directement via une surface d'outils partagée.
3. **Hierarchical.**Superviseurs gérant des sous-superviseurs, mis en œuvre sous forme de sous-graphes encastrés.

### Où ce modèle va mal

- **Checkpoints too small.**Seule la conversation de point de contrôle tourne laissant l'état de l'outil et la mémoire écrit non récupérable.
- **Non-deterministic nodes.**Le résumé suppose que les entrées de nœud produisent la même mise à jour d'état.
- **Over-use of conditional edges.**Un graphique avec chaque bord conditionnel est une machine d'état qui ne peut être raisonnable.

```figure
langgraph-state
```

## Faites-le

`code/main.py`met en œuvre un graphique d'état stdlib:

- `State` un dicton avec`messages`- Je suis là .`step`- Je suis là .`route`- Je suis là .`output`- Je suis là .`human_approval`- Je suis désolé .
- `Node` appel à l'état de prise et à la restitution d'un dicton de mise à jour.
- `StateGraph` nœuds + bordes + bordes conditionnelles + exécution + CV.
- `SQLiteCheckpointer`(fals en mémoire)  sérialise l' état après chaque nœud; `load(session_id)`- Il est en train de restaurer.
- Un graphique de démonstration: classer -> branche(refund / bug / ventes) -> porte humaine -> envoyer.

- Je vais le faire.

```
python3 code/main.py
```

La trace montre la première course échouant à la porte humaine, persistance, puis reprendre à produire la sortie finale.

## Utilisez-le

- **LangGraph** la référence, prête à la production.`create_react_agent`- Je suis là .`create_supervisor`, ou construire votre propre graphique.
- **AutoGen v0.4**(Létion 14)  Modèle de l'acteur alternative pour les scénarios à forte concurrence.
- **Claude Agent SDK**(Léction 17)  Harness géré avec magasin de session intégré.
- **Custom** lorsque vous avez besoin d'un contrôle exact sur la forme de l'état ou le backend du point de contrôle.

## La faire partir

`outputs/skill-state-graph.md`génère un graphique d'état en forme de LangGraph dans n'importe quel temps d'exécution cible avec le point de contrôle et le CV câblé.

## Exercices

1. Ajouter une limite conditionnelle à partir de `classify`à `end`Lorsque la confiance de la classification est inférieure à un seuil, reprenez la course après un ensemble humain.`route`manuellement.
2. Échangez le faux SQLite contre un vrai point de contrôle SQLite.
3. Implémenter des bords parallèles: deux nœuds fonctionnent simultanément, fusionner par un réducteur personnalisé.
4. Lire `langgraph-supervisor`- Remplacez le jouet à`create_supervisor`- Comparer les traces.
5. Ajouter le flux: chaque nœud produit un état partiel pendant son exécution.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | Serializes state after every node; enables resume |
| Reducer | "State merger" | Function that combines current state with a node's update |
| Conditional edge | "Branch" | Edge chosen by a function of state |
| Subgraph | "Nested graph" | A graph used as a node inside another graph |
| Durable execution | "Resume from failure" | Restart at the last successful node with exact state |
| Supervisor | "Router LLM" | Central dispatcher for specialist subagents |
| Swarm | "P2P agents" | Agents hand off via shared tools; no central router |

## Pour en savoir plus

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) les documents de référence
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) API de modèle de surveillance
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) Modèle alternatif pour les acteurs
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) magasin de séances et sous-bagages
