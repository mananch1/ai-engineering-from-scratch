# Machines de l'État agencées  Graphes, nœuds, points de contrôle

> Une boucle ReAct écrite à la main est une `while True`La même boucle écrite comme un graphique explicite est quelque chose que vous pouvez contrôler, interrompre, brancher, et voyager dans le temps.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## Le problème

Vous envoyez un agent qui appelle à la fonction. Il fonctionne pendant trois tours, puis quelque chose ne va pas: le modèle essaie un outil qui rend 500, l'utilisateur change d'avis au milieu de la tâche, ou l'agent décide de rembourser une commande sans qu'un humain signe.`while True:`Vous ne pouvez pas le faire pauser, vous ne pouvez pas le faire retourner, et vous ne pouvez pas brancher en "et si le modèle avait choisi l'autre outil".

La prochaine étape est évidente une fois que vous la voyez. L'agent est déjà une machine d'état  système prompt plus l'historique des messages plus les appels d'outil en attente plus l'action suivante. Faites expliciter la machine d'état: les nœuds pour "le modèle pense", "un outil fonctionne", "un humain approuve", et les bords pour les transitions conditionnelles entre eux. Une fois le graphique explicite, le harnais obtient quatre choses gratuitement: le point de contrôle (état de sauvegarde entre les étapes), les interruptions (pause pour un humain), le streaming (tokens de flux et événements intermédiaires) et le voyage dans le temps (retour à un état précédent et essayez une branche différente).

La mise en œuvre de référence de cette abstraction est LangGraph. Ce n'est pas un cadre d'agent au sens de LangChain (" ici il y a un AgentExecutor, bonne chance "). C'est un graphe runtime avec état de première classe, persistance de première classe et interruptions de première classe. La boucle d'agent est quelque chose que vous dessinez, pas quelque chose que vous écrivez à la main.

## Le concept

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

Une .`StateGraph`Il y a trois choses.

1. **State.**Un dicté typé (TypedDict ou modèle Pydantic) qui circule à travers le graphique. Chaque nœud reçoit l'état complet et renvoie une mise à jour partielle, que LangGraph fusionne en utilisant un *réducteur* par champ `operator.add`pour les listes qui devraient s'accumuler, écraser par défaut.
2. **Nodes.**Fonctions Python `state -> partial_state`Chacun est une étape discrète: "appeler le modèle", "exécuter des outils", "récapituler".
3. **Edges.**Les tranches entre les nœuds. Les bords statiques vont à un endroit. Les bords conditionnels prennent une fonction de routeur`state -> next_node_name`Donc le graphique peut se brancher sur la sortie du modèle.

Vous compilez le graphique. Compile lie la topologie, attache un point de contrôle (optionnel mais essentiel pour la production), et renvoie un fonctionnable. Vous l'invoquez avec un état initial et un`thread_id`Chaque étape de l' exécution est un point de contrôle à clé .`(thread_id, checkpoint_id)`- Je suis désolé .

### Les quatre superpuissances

**Checkpointing.**Chaque transition de nœud écrit le nouvel état dans un magasin (en mémoire pour les tests, Postgres/Redis/SQLite pour prod).`thread_id`Le graphique reprend son parcours.

**Interrupts.**Marquez un nœud avec `interrupt_before=["human_review"]`L'API répond à l'utilisateur avec "attendant l'approbation". Une demande ultérieure à la même `thread_id`avec `Command(resume=...)`reprend l'exécution.

**Streaming.** `graph.stream(state, mode="updates")`Les zones de delta sont en train de se produire.`mode="messages"`Les jetons LLM sont diffusés à l'intérieur des nœuds du modèle. `mode="values"`Vous choisissez ce qui doit apparaître dans votre interface.

**Time-travel.** `graph.get_state_history(thread_id)`renvoie le journal complet du point de contrôle.`checkpoint_id`à `graph.invoke`C'est un bon moyen de débogage ("et si le modèle avait choisi l'outil B à la place?") et pour les tests de régression qui reproduisent les traces de production.

### Les réducteurs sont le point

Chaque champ d'état a un réducteur. La plupart des paramètres sont bons  une nouvelle valeur surpasse l'ancienne. Mais les listes de messages doivent `operator.add`Les deux nœuds sont mis à jour par la même méthode.`messages`et vous avez oublié le `Annotated[list, add_messages]`Le réducteur est la seule chose subtile dans la bibliothèque; faites-le bien et le reste se compose.

### Le graphique ReAct en quatre nœuds

Un agent ReAct de production est constitué de quatre nœuds et de deux bords:

1. `agent` appelle le LLM avec l'historique de message actuel. Retourne le message assistant (qui peut contenir tool_calls).
2. `tools` exécute tous les appels tool_calls dans le dernier message assistant, ajoute les résultats de l'outil comme messages d'outil.
3. Un bord conditionnel de `agent`qui se dirige vers `tools`si le dernier message contient des appels à outils, sinon `END`- Je suis désolé .
4. Un bord statique de `tools`Retour à `agent`- Je suis désolé .

Vous obtenez la boucle ReAct complète (Pensement → Action → Observation → Pensement → ...) avec point de contrôle, interrompt et streaming, en environ 40 lignes de code.

### StateGraph vs Envoyer (fanout)

`Send(node_name, state)`L'agent décide de consulter trois récupérateurs à la fois.`Send`La méthode de LangGraph est de façon à exprimer le modèle orchestrateur-travailleur sans fil de primitives.

### Les sous-graphes

Un graphique compilé peut être un nœud dans un autre graphique. Le graphique externe voit un seul nœud; le graphique interne a son propre état et ses propres points de contrôle. C'est ainsi que les équipes construisent des agents de travail supervisé: le graphique supervisé rote l'intention de l'utilisateur vers un sous-graphe de travail par domaine.

```figure
l5-state-graph-ledger
```

## Faites-le

### Étape 1: état et nœuds

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages`est le réducteur qui fait accumuler la liste de messages au lieu de la supprimer.

### Étape 2: courir avec un fil

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Chaque mise à jour est un dicton .`{node_name: state_delta}`Votre frontend peut les transmettre à l'interface utilisateur pour que les utilisateurs voient "l'agent pense... appelant search_web... a obtenu le résultat... répondant".

### Étape 3: ajouter une interruption humaine en boucle

Marquez un nœud pour que l'exécution s'arrête avant son exécution.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

L'état, le point de contrôle et le fil persistent tout au long de l'interruption.

### Étape 4: Voyage dans le temps pour débogage

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Passer par là`None`Lorsque l'entrée se reproduit à partir du point de contrôle donné, en passant une valeur, elle est ajoutée comme une mise à jour de l'état de ce point de contrôle avant de reprendre.

### Étape 5: échangez le point de contrôle pour la production

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis et Postgres sont expédiés.`MemorySaver`Tout ce qui persiste à travers les redémarrages veut un vrai magasin.

## La compétence

> Vous construisez des agents comme des graphiques, pas comme `while True`- Les boucles.

Avant de toucher LangGraph, faites une conception de 60 secondes:

1. **Name the nodes.**Chaque décision discrète ou action secondaire est un nœud. " L'agent pense, " " l'outil fonctionne, " " l'examenateur approuve, " " les flux de réponse. " Si vous ne pouvez pas les énumérer, la tâche n'est pas encore en forme d'agent.
2. **Declare the state.**Type minimal avec un réducteur pour chaque champ de liste.`messages`; les champs spécifiques à la tâche de levage (un travail `plan`, une `budget`le compteur, un `retrieved_docs`Liste) au niveau supérieur.
3. **Draw the edges.**La fonction de routeur est statique, sauf si l'étape suivante dépend de la sortie du modèle.
4. **Choose a checkpointer up front.** `MemorySaver`Pour les tests, Postgres/Redis/SQLite pour tout autre.
5. **Decide interrupts before tools run, not after.**Les approbations vont sur le bord dans un nœud d'effet secondaire afin que vous puissiez annuler avant le dommage; la validation va sur le bord hors du modèle afin que vous puissiez rejeter les mauvaises appels à bas prix.
6. **Stream by default.** `mode="updates"`pour l'interface utilisateur, `mode="messages"`pour le streaming au niveau des jetons à l'intérieur des nœuds du modèle, `mode="values"`pour les instantanés complets pendant l'évaluation.

Ne pas envoyer un agent LangGraph qui n'a pas de point de contrôle, ne pas envoyer un agent qui interrompt après l'effet secondaire, ne pas envoyer un agent LangGraph qui ne peut pas être utilisé.`messages`champ sans `add_messages`comme son réducteur.

## Exercices

1. **Easy.**Implémenter le graphique ReAct à quatre nœuds ci-dessus avec un outil de calculateur et un outil de recherche Web.`list(app.get_state_history(config))`retourne au moins quatre points de contrôle pour une conversation à deux tours.
2. **Medium.**Ajouter un `planner`Le nœud qui se déplace avant `agent`et écrit un article structuré `plan: list[str]`Je suis dans l'état.`agent`Marquer les étapes du plan comme faites.`plan`est perdu sur un CV du point de contrôle (réducteur incorrect).
3. **Hard.**Construire un graphique de surveillance qui traverse trois sous-graphes (`researcher`- Je suis là .`writer`- Je suis là .`reviewer`) en utilisant `Send`Chaque sous-graphe a son propre état et son propre point de contrôle.`interrupt_before=["writer"]`Confirmer que le voyage dans le temps depuis un point de contrôle précédent ne refait que la branche fourchée.

## Les termes clés

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| StateGraph | "The LangGraph graph" | The builder object you add nodes and edges to before compile. |
| Reducer | "How the field merges" | A function `(old, new) -> merged` applied when a node returns an update for that field; default is overwrite, `add_messages` appends. |
| Thread | "A conversation ID" | A `thread_id` string that scopes all checkpoints for one session. |
| Checkpoint | "A paused state" | A persisted snapshot of the full graph state after a node transition, keyed on `(thread_id, checkpoint_id)`. |
| Interrupt | "Pause for a human" | `interrupt_before` / `interrupt_after` stop execution at a node boundary; resume with `Command(resume=...)`. |
| Time-travel | "Fork from a prior step" | `graph.invoke(None, config_with_old_checkpoint_id)` replays from that checkpoint forward. |
| Send | "Parallel subgraph dispatch" | A constructor a node can return to spawn N parallel executions of a target node. |
| Subgraph | "A compiled graph as a node" | A compiled StateGraph used as a node in another graph; preserves its own state scope. |

## Pour en savoir plus

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) référence canonique pour StateGraph, réducteurs, points de contrôle et interruptions.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) le modèle mental utilisé dans cette leçon, directement de la source.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) les détails sur les magasins Postgres/SQLite/Redis, les espaces de noms des points de contrôle et les identifiants de filets.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) `interrupt_before`- Je suis là .`interrupt_after`- Je suis là .`Command(resume=...)`, et le modèle de modification-état.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) le modèle que chaque agent LangGraph implémentera; lisez-le pour le raisonnement de la racine.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) quelles formes de graphique (chaîne, routeur, orchestrateur-travailleurs, évaluateur-optimisateur) préférer et quand.
- Phase 11 · 09 (Appel de fonction)  l'outil-appel primitif chaque nœud de l'agent LangGraph réutilise.
- Phase 11 · 14 (Model Context Protocol)  Découverte d'outils externes qui se connectent à un LangGraph `ToolNode`par l'adaptateur MCP.
- Phase 11 · 17 (compromise avec le cadre des agents)  quand choisir LangGraph au lieu de CrewAI, AutoGen ou Agno.
