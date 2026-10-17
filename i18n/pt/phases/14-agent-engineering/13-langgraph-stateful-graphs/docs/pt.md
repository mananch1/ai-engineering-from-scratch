# Orquestração de gráficos estatais  Execução duradoura e pontos de controlo

> O agente é uma máquina de estado; nós são funções; bordas são transições; estado é posto em ponto de checagem após cada nó. Resume de qualquer falha no último ponto de checagem bem sucedido. LangGraph é a referência de 2026 para este modelo de orquestração estadual de baixo nível.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Descreva o modelo central do LangGraph: máquina de estado com estado tipado, nós de função, bordas condicionais e pontos de verificação pós-nodo.
- Nomear as quatro capacidades que os documentos destacam: execução duradoura, streaming, humano-no-loop, memória abrangente.
- Explique as três topologias de orquestração que LangGraph suporta: supervisor, peer-to-peer (swarm), hierárquica (subgrafos aninhados).
- Implementar um gráfico de estado stdlib com estado digitado, bordas condicionais e um ciclo de checkpoint/resume.

## O problema

Agentes e fluxos de trabalho compartilham um problema: quando uma execução de 40 passos falha na etapa 38, você quer retomar a partir da etapa 38, não recomeçar. Modelos de estado de segunda classe deixam os operadores tentando invadir novamente uma biblioteca que assume novas corridas.

A resposta de design do LangGraph: o estado é um objeto de primeira classe, as mutações são explícitas e os pontos de verificação persistem após cada nó.`load_state(session_id)`- Não.

## O conceito

### O gráfico

Um gráfico é definido por:

- **State type.**Um ditado tipado (ou modelo Pydantic) que cada nó lê e muda.
- **Nodes.**Funções puras`(state) -> state_update`As atualizações são fundidas no estado após o retorno.
- **Edges.**Transições condicionais ou diretas entre nós.
- **Entry and exit.** `START`E ...`END`Os nós sentinela marcam a fronteira.

Exemplo: um agente com `classify`- Não .`refund`- Não .`bug`- Não .`sales`- Não .`done`nós  um fluxo de trabalho de roteamento como um gráfico.

### Execução duradoura

Após cada nó retornar, o runtime serializa o estado e o escreve para um checkpointer (SQLite, Postgres, Redis, custom).`resume(session_id)`e retomar a partir do passo N + 1 com estado exato.

Os documentos do LangGraph destacam explicitamente os usuários da produção onde isso importa: Klarna, Uber, JP Morgan. A alegação não é a forma do gráfico; é que a forma do gráfico mais o ponto de verificação torna a recuperação barata.

### Transmissão

Cada nó pode produzir saída parcial. O gráfico transmite eventos por nó-delta para o chamador para que as UI atualizem à medida que o gráfico é executado.

### Homem no circuito

Inspectar e modificar o estado entre nós. Implementações: pausa antes de um nó crítico, estado de superfície para um humano, aceitar modificações, retomar. O checkpointer torna isso fácil porque o estado já está serializado.

### Memória

Curto prazo (dentro de um run  histórico de conversação no estado) e longo prazo (contínuo através do checkpointer e de uma loja separada a longo prazo).

### Três topologias

1. **Supervisor.**O roteador central LLM envia para os subagentes especializados. `create_supervisor()`em `langgraph-supervisor`(embora a equipe da LangChain em 2026 recomende fazer isso através de ferramentas que pedem diretamente mais controle de contexto).
2. **Swarm / peer-to-peer.**Os agentes transmitem directamente através de uma superfície de ferramentas compartilhada.
3. **Hierarchical.**Supervisores que gerenciam sub-supervisores, implementados como subgrafos aninhados.

### Onde este padrão vai mal

- **Checkpoints too small.**Apenas as conversas de ponto de verificação deixam o estado da ferramenta e a memória escreve irrecuperavelmente.
- **Non-deterministic nodes.**Resume assume que as entradas de nós produzem a mesma atualização de estado. Sementes aleatórias, relógio de parede, API externas devem ser capturadas.
- **Over-use of conditional edges.**Um gráfico com cada borda condicional é uma máquina de estado que não pode ser raciocinado.

```figure
langgraph-state
```

## Construí-lo

`code/main.py`Implementa um gráfico de estado stdlib:

- `State` um ditado com `messages`- Não .`step`- Não .`route`- Não .`output`- Não .`human_approval`- Não .
- `Node` Callable tomando estado e devolvendo um ditado de atualização.
- `StateGraph` nós + bordas + bordas condicionais + execução + resumindo.
- `SQLiteCheckpointer`(in-memory fake)  serializa estado após cada nó; `load(session_id)`- Não, não.
- Um gráfico de demonstração: classificar -> ramo(reembolso / bug / vendas) -> portal humano -> enviar.

- É o que é ?

```
python3 code/main.py
```

O rastro mostra a primeira corrida falhando no portal humano, persistência, e depois retomando a produção final.

## Usá-lo

- **LangGraph** o referente, pronto para produção.`create_react_agent`- Não .`create_supervisor`, ou construir o seu próprio gráfico.
- **AutoGen v0.4**(Lessão 14)  Modelo de ator alternativa para cenários de alta concorrência.
- **Claude Agent SDK**(Lessão 17)  Arneses gerenciados com loja de sessões integrada.
- **Custom** quando você precisa de um controlo exato sobre a forma do estado ou o backend do checkpointer.

## Envia-o

`outputs/skill-state-graph.md`gera um gráfico de estado em forma de LangGraph em qualquer tempo de execução de destino com checkpointing e resume conectado.

## Exercícios

1. Adicionar uma borda condicional de `classify`- Não .`end`Quando a confiança da classificação está abaixo de um limiar, retoma a corrida após um conjunto humano.`route`Manualmente.
2. Troca o falso SQLite por um checkpointer SQLite real.
3. Implementar bordas paralelas: dois nós executam simultaneamente, se fundem por um redutor personalizado.
4. Leia `langgraph-supervisor`Referência.`create_supervisor`Comparar as formas das vestígios.
5. Adicionar streaming: cada nó produz estado parcial enquanto corre. Imprima os deltas à sua chegada.

## Termos-chave

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

## Mais leitura

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) os documentos de referência
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) API de padrão de supervisão
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) Modelo alternativo de ator
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) loja de sessões e subagentes
