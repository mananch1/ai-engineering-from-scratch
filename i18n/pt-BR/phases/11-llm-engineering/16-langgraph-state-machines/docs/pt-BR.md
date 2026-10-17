# Máquinas do Estado Agente  Gráficos, Nodos, Pontos de Controle

> Um ciclo ReAct escrito à mão é um `while True`O mesmo ciclo escrito como um gráfico explícito é algo que você pode fazer ponto de controle, interromper, ramificar e viajar no tempo.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## O problema

Enviamos um agente que chama a função. Funciona por três voltas, e depois algo vai mal: o modelo tenta uma ferramenta que retorna 500, o usuário muda de ideias no meio da tarefa, ou o agente decide reembolsar uma encomenda sem uma assinatura humana.`while True:`O loop não tem ganchos. Não pode pausar, não pode voltar a girar, e não pode se ramificar para "e se o modelo tivesse escolhido a outra ferramenta". No momento em que você envia isso para além de uma demonstração, o agente se torna uma caixa negra que ou funcionou ou não.

O próximo passo é óbvio quando o ver. O agente já é um sistema de máquina de estado  prompt mais histórico de mensagem mais pendentes ferramentas chamadas mais a próxima ação. Faça a máquina do estado explícita: nós para "o modelo pensa", "uma ferramenta funciona", "um ser humano aprova", e bordas para as transições condicionais entre elas. Uma vez que o gráfico é explícito, o arnes recebe quatro coisas gratuitamente: checkpointing (salvar estado entre passos), interrupções (pausa para um humano), streaming (tokens de fluxo e eventos intermediários) e viagem no tempo (revoltar para um estado anterior e tentar um ramo diferente).

A implementação de referência desta abstracção é LangGraph. Não é um quadro de agente no sentido de LangChain ("aqui há um AgentExecutor, boa sorte"). É um tempo de execução de gráfico com estado de primeira classe, persistência de primeira classe e interrupções de primeira classe. O loop de agente é algo que você desenha, não algo que você escreve à mão.

## O conceito

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

A.`StateGraph`Tem três coisas.

1. **State.**Um ditado tipado (TypedDict ou modelo Pydantic) que flui através do gráfico. Cada nó recebe o estado completo e retorna uma atualização parcial, que LangGraph combina usando um *redutor* por campo `operator.add`Para as listas que devem acumular-se, substituir-se por padrão.
2. **Nodes.**Funções Python `state -> partial_state`Cada um é um passo discreto: "cham o modelo", "exercem ferramentas", "resumem".
3. **Edges.**Transições entre nós. bordas estáticas vão para um lugar. bordas condicionais tomam uma função de roteador`state -> next_node_name`Então o gráfico pode se ramificar na saída do modelo.

Compile liga a topologia, anexa um checkpointer (opcional, mas essencial para a produção), e retorna um executável.`thread_id`Cada etapa da execução é um ponto de controlo com teclado .`(thread_id, checkpoint_id)`- Não .

### As quatro superpotências

**Checkpointing.**Cada transição de nó escreve o novo estado para um armazém (em memória para testes, Postgres/Redis/SQLite para prod). Resume chamando o gráfico novamente com o mesmo `thread_id`O gráfico retoma onde parou.

**Interrupts.**Marque um nó com `interrupt_before=["human_review"]`A sua API responde ao usuário com "esperando aprovação". Uma solicitação posterior para o mesmo `thread_id`com`Command(resume=...)`retoma a execução.

**Streaming.** `graph.stream(state, mode="updates")`O estado da Delta, quando acontece.`mode="messages"`Transmite os tokens do LLM dentro dos nós do modelo. `mode="values"`E você escolhe o que aparecer na interface.

**Time-travel.** `graph.get_state_history(thread_id)`Retorna o registro completo do posto de controlo.`checkpoint_id`- Não .`graph.invoke`Ótimo para depurar ("e se o modelo tivesse escolhido a ferramenta B em vez disso?") e para testes de regressão que reproduzem traços de produção.

### Os reductores são o ponto

Cada campo de estado tem um redutor. A maioria das definições são boas  um novo valor sobrepõe o antigo. Mas as listas de mensagens precisam `operator.add`As bordas paralelas combinam suas atualizações através do redutor. Se dois nós atualizar ambos`messages`E esqueceste-te do .`Annotated[list, add_messages]`O reductor é a única coisa sutil na biblioteca; faça o certo e o resto compõe.

### O gráfico ReAct em quatro nós

Um agente ReAct de produção é constituído por quatro nós e duas bordas:

1. `agent` chama o Mestrado em Direito com o histórico de mensagem atual. Retorna a mensagem assistente (que pode conter tool_calls).
2. `tools` executa qualquer tool_call na última mensagem assistente, anexa os resultados da ferramenta como mensagens de ferramenta.
3. Uma borda condicional de `agent`que liga-se a `tools`se a última mensagem tem tool_calls, de outra forma `END`- Não .
4. Uma borda estática de `tools`De volta para `agent`- Não .

É isso. Você obtém o ciclo completo ReAct (Pensamento → Ação → Observação → Pensamento → ...) com ponto de verificação, interrupções e streaming, em aproximadamente 40 linhas de código.

### StateGraph vs Enviar (fanout)

`Send(node_name, state)`O agente decide consultar três retrievers ao mesmo tempo.`Send`O LangGraph é um sistema de análise de dados que permite a execução paralela do nó-alvo; suas saídas se fundem através do redutor de estado.

### Subgrafos

Um gráfico compilado pode ser um nó em outro gráfico. O gráfico externo vê um único nó; o gráfico interno tem seu próprio estado e seus próprios pontos de controle. É assim que as equipes construem agentes de supervisor-trabalhador: o gráfico supervisor encaminha a intenção do usuário para um subgrafo de trabalhador por domínio.

```figure
l5-state-graph-ledger
```

## Construí-lo

### Passo 1: estado e nós

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

`add_messages`O redutor é o que faz com que a lista de mensagens se acumula em vez de ser substituída.

### Passo 2: executar com um fio

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Cada atualização é um ditado .`{node_name: state_delta}`O frontend pode transmitir isto para a interface para que os usuários vejam "o agente está a pensar... a ligar para o search_web... obteve resultado... a responder".

### Passo 3: adicionar uma interrupção humana no loop

Marque um nó para que a execução pare antes de ser executada.

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

O estado, o ponto de controlo e o fio persistem durante a interrupção.

### Passo 4: viagem no tempo para depurar

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Passando .`None`como a entrada repete-se a partir do ponto de verificação dado; passar um valor acrescenta-o como uma atualização ao estado do ponto de verificação antes de retomar.

### Passo 5: troca o ponto de controlo para a produção

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis e Postgres estão enviados.`MemorySaver`Qualquer coisa que persista durante o reinicio quer uma loja real.

## A habilidade

> Construem agentes como gráficos, não como`while True`- Os circuitos.

Antes de chegar ao LangGraph, faça um desenho de 60 segundos:

1. **Name the nodes.**Cada decisão discreta ou ação de efeitos colaterais é um nó. "Agente pensa," "outil funciona," "revisor aprova," "resposta fluxos".
2. **Declare the state.**Tipo mínimo com um redutor para cada campo de lista. Não enche tudo em `messages`• campos específicos de tarefa (um trabalho)`plan`, a `budget`Contador, um `retrieved_docs`Lista) até ao nível superior.
3. **Draw the edges.**Estático, a menos que o próximo passo dependa da saída do modelo.
4. **Choose a checkpointer up front.** `MemorySaver`Para testes, Postgres/Redis/SQLite para qualquer outra coisa. Não enviar sem um  nenhum ponto de verificação significa nenhum currículo, nenhuma interrupção, nenhuma viagem no tempo.
5. **Decide interrupts before tools run, not after.**As aprovações vão na borda para um nó de efeitos colaterais para que você possa cancelar antes de causar danos; a validação vai na borda para fora do modelo para que você possa rejeitar chamadas ruins a baixo custo.
6. **Stream by default.** `mode="updates"`para a interfaz de utilização, `mode="messages"`para o streaming de nível de token dentro dos nós do modelo, `mode="values"`para instantâneos completos durante a avaliação.

Recusar-se a enviar um agente LangGraph que não tem ponto de controlo. Recusar-se a enviar um que interrompa *após* o efeito colateral. Recusar-se a enviar um`messages`campo sem `add_messages`como seu reductor.

## Exercícios

1. **Easy.**Implementar o gráfico ReAct de quatro nós acima com uma ferramenta de calculadora e uma ferramenta de pesquisa na web. Verifique se `list(app.get_state_history(config))`Retorna ao menos quatro pontos de controlo para uma conversa de dois turnos.
2. **Medium.**Adicionar um`planner`nó que corre antes `agent`e escreve um estruturado `plan: list[str]`- Não, não.`agent`Marque os passos do plano como feito.`plan`Se o número de dados de verificação for alterado, o número de dados de verificação será alterado.
3. **Hard.**Construir um gráfico de supervisão que percorra entre três subgrafos (`researcher`- Não .`writer`- Não .`reviewer`) utilizando `Send`Cada subgrafo tem o seu próprio estado e ponto de controlo.`interrupt_before=["writer"]`Confirme que a viagem no tempo a partir de um ponto de controlo anterior re-corre apenas o ramo forcado.

## Termos-chave

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

## Mais leitura

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) Referência canónica para StateGraph, reductores, checkpointers e interrupções.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) o modelo mental que esta lição usa, diretamente da fonte.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) os detalhes das lojas Postgres/SQLite/Redis, espaços de nomes dos pontos de verificação e IDs de thread.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)- Não .`interrupt_before`- Não .`interrupt_after`- Não .`Command(resume=...)`, e o padrão de edição-estado.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) o padrão que cada agente LangGraph implementa; leia para o raciocínio racional.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) quais formas de gráfico (cadeia, roteador, orquestrador-trabalhador, avaliador-optimizador) preferir e quando.
- Fase 11 · 09 (Calling Function)  o primitivo de chamada de ferramenta é reutilizado por todos os nós do agente LangGraph.
- Fase 11 · 14 (Modelo de Protocolo de Contexto)  Descoberta de ferramentas externas que se conectam a um LangGraph `ToolNode`através do adaptador MCP.
- Fase 11 · 17 (Compromissos de estrutura de agentes)  quando escolher LangGraph sobre CrewAI, AutoGen ou Agno.
