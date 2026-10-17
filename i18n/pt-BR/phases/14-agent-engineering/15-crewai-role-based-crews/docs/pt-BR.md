# Equipes de agentes baseadas em funções  Funções, tarefas, processos

> Quatro primitivas: Agente, tarefa, tripulação, processo. Duas formas de nível superior: equipes (autônoma, colaboração baseada em papéis) e fluxos (evento-driven, determinista). CrewAI é a implementação de referência de 2026, e seus documentos são contundentes: "para qualquer aplicação pronta para produção, comece com um fluxo".

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Nomear os quatro primitivos da CrewAI (Agente, Tarefa, Equipamento, Processo) e o que cada um possui.
- Distinguir Sequenciais, Hierárquicos e o processo de Consenso planejado; escolher um por carga de trabalho.
- Distinguir as equipas (baseadas em funções autônomas) das fluxos (determinísticas orientadas por eventos) e explicar a recomendação de produção dos docentes.
- Ferramentas de fio com o `@tool`decorador e `BaseTool`Subclasse; razão sobre saídas estruturadas versus texto livre.
- Nomear os quatro tipos de memória CrewAI e quando cada um paga.
- Implementar uma equipe de três agentes (investigador, escritor, editor) que produz um resumo.
- Determine os três modos de falha da CrewAI: "Inflação rápida", "imposto de gerente-LLM", "transmissões frágeis".

## O problema

As equipes que adotam estruturas multi-agentes atingem a mesma parede. "Collaboração autónoma" soa muito bem em uma demonstração. Então um cliente arquivou um bug e você precisa de repetição determinista. Ou financeiro pergunta quanto custa uma equipe de LLM-routed por rodada. Ou em chamada precisa saber qual agente ficou parado às 3 da manhã.

As equipes de forma livre com direção LLM não respondem a nenhuma delas de forma limpa.

A divisão da CrewAI é honesta sobre o comércio. Equipes para trabalho colaborativo, baseado em papéis, exploratório. Fluxos para produção orientada a eventos, de propriedade de código, auditable.

## O conceito

### Quatro primitivos

A superfície da tripulação é pequena, memorizar isto e o resto é configurar.

- **Agent.** `role + goal + backstory + tools + (optional) llm`O conteúdo é carregável, modela o tom, o julgamento, quando o agente se detém, as ferramentas são funções que o agente pode chamar (mais abaixo).
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`Uma unidade de trabalho reutiliável.`expected_output`É o contrato.`context`Lista de tarefas que são transmitidas em linha de frente. `output_pydantic`Força uma forma estruturada.
- **Crew.**O contêiner, é o proprietário da lista de`agents`, a lista de `tasks`, o `process`, e opcionais `memory`+ `verbose`+ `manager_llm`configurações.
- **Process.**Estratégia de execução: sequencial, hierárquica, consensal (planificado).

Os agentes não se veem diretamente, as tarefas são de referência, a tripulação sequencia as tarefas, o processo decide quem escolhe a próxima tarefa, é o modelo mental.

> **Validated against**CrewAI 0.86 (2026-05). As versões mais recentes podem renomear ou fundir os tipos de processo; verifique o [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)antes de depender de uma forma específica.

### Sequenciais vs Hierárquicos vs Consenso

- **Sequential.**As tarefas são executadas na ordem de declaração.`context`O custo mais baixo, mais previsível, usado quando a ordem estiver fixa.
- **Hierarchical.**O agente gerente (chamada de LLM separada) percorre rotas entre especialistas.`manager_llm`Configuração ou padrão. O gerente seleciona a próxima tarefa a cada rodada e pode recusar ou redirecionar.
- **Consensus.**Planeado, não implementado na API pública. Os documentos reservam o nome para um futuro processo baseado em votação. Não confiem nele hoje.

A hierarquica adiciona uma chamada de LLM por rodada (o gerente) em cima de cada chamada especializada. O custo do token pode triplicar em uma corrida de cinco passos. Pague apenas quando você precisa do roteamento.

### Equipes vs Fluxos

É o enquadramento com que os médicos vão liderar em 2026.

- **Crew.**Autonomia orientada pelo LLM. O framework escolhe a forma no tempo de execução. bom para: pesquisa, brainstorming, primeiros rascunhos, onde quer que o caminho seja parte da resposta. Difícil de repetição. Difícil de testar. Barato para protótipo.
- **Flow.**Grafico de eventos que você possui.`@start`Marca a entrada. `@listen(topic)`O que é um passo que dispara quando outro passo emite esse tópico. cada passo é Python simples (pode chamar uma tripulação internamente). bom para: produção. observável. testável. determinista.

Recomendação de produção dos médicos para 2026: comece com um fluxo.`Crew.kickoff()`O fluxo dá-lhe o rastro de auditoria, a tripulação dá-lhe a exploração.

### Integração de ferramentas

Três maneiras de dar uma ferramenta a um agente.

1. **`@tool` decorator.**Funções puras se tornam ferramentas. A assinatura é o esquema; o docstring é a descrição que o LLM vê.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.**Ferramenta baseada em classe com esquema de args explícito, suporte de async, retries. Utilize quando a ferramenta tem estado (um cliente, um cache) ou precisa de args estruturados.

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

3. **Built-in toolkits.**A CrewAI envia adaptadores de primeira parte: `SerperDevTool`- Não .`FileReadTool`- Não .`DirectoryReadTool`- Não .`CodeInterpreterTool`- Não .`RagTool`- Não .`WebsiteSearchTool`- Com um único importador.

As saídas estruturadas usam o Pydantic.`output_pydantic=MyModel`A CrewAI valida a resposta do LLM contra o modelo e quer coacciona ou retrata.`expected_output`As saídas de texto livre são boas para os rascunhos; as saídas estruturadas são o que os fluxos de baixo fluxo podem consumir.

### Anéis de memória

A CrewAI envia quatro tipos de memória para fora da caixa.

> **Validated against**CrewAI 0.86 (2026-05).`Memory`O modelo conceitual abaixo ainda vale, mas a superfície da classe pública pode cair para uma única`Memory`ponto de entrada em versões mais recentes; verificação [CrewAI memory docs](https://docs.crewai.com/concepts/memory)para a API atual.

- **Short-term.**O amortecedor de conversação num único passe, apagado no final.
- **Long-term.**Persistindo em todas as corridas. Armazenado em um vector DB (Chroma por padrão, intercambiável). Retirado por semelhança com a tarefa atual.
- **Entity.**"Cliente X está no plano empresarial", baseado em entidade, não em semelhança.
- **Contextual.**Retira a memória relevante no momento em que o agente precisa, não pré-carregada.

Ativar a tripulação com `memory=True`A memória é um dos lugares onde a CrewAI ganha a sua manutenção contra frameworks mais finos; LangGraph puro exige que você cable cada um deles sozinho.

### Quando as equipes baseadas em papéis se adaptam

- Três a seis agentes com papéis identificados e um fluxo de trabalho colaborativo.
- Roteamento em que o julgamento do MLL sobre o próximo passo faz parte do valor (hierárquico).
- Onde quer que a equipa esteja mais feliz lendo .`role + goal + backstory`- Não. - Não.

### Quando não

- DAG deterministas com ordem rigorosa. Use LangGraph (Lessão 13). A forma do gráfico é a abstracção certa; o enquadramento de papel da CrewAI é o atrito.
- Orçamentos de latência subsegundo. Hierárquico adiciona viagens de ida e volta. Mesmo Sequential serializa instruções que incluem histórias de fundo e saídas anteriores.
- Loops de agente único. Esqueça a estrutura; um loop de agente (Lessão 1) mais um registro de ferramentas é mais curto.

A lição 17 (Agent Framework Tradeoffs) expõe isto numa matriz.

### Forma de dependência

Independente da LangChain. Python 3.10 a 3.13.`uv`Contagem de estrelas: veja[crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)(Shotshot em 2026-05). A integração AWS Bedrock é documentada; benchmarks vendor relatam um aumento substancial de velocidade versus LangGraph em cargas de trabalho de QA, mas a metodologia (dataset, hardware, métrica de avaliação) não é publicada, então trate os números framework-vendor como direcional apenas.

### Onde este padrão vai mal

- **Prompt-bloat from backstories.**Uma história de fundo de 2000 palavras por agente e uma equipe de cinco agentes queima o orçamento contextual antes da primeira chamada de ferramenta. Mantenha as histórias de fundo abaixo de 200 palavras. Reutilize frases em todos os agentes; não repita o estilo da casa cinco vezes.
- **Manager-LLM token tax.**O processo hierárquico adiciona uma chamada de LLM do gerente antes de cada chamada especialista. Em uma equipe de cinco tarefas que é de seis chamadas de LLM em vez de cinco, e a chamada do gerente carrega a lista completa de tarefas mais saídas anteriores.
- **Brittle handoffs.**A tarefa N's `expected_output`A tarefa N+1 diz que `context`O LLM produziu quatro, o agente ad-libs, resolveu com o agente de segurança.`output_pydantic`na tarefa N, então a tarefa N+1 lê um objeto digitado, não texto livre.
- **Crew-as-prod.**A tripulação de forma livre enviada para a produção sem um envolvente de fluxo. A variabilidade de saída é alta; repetição é impossível; a chamada não pode diferenciar uma corrida ruim contra uma boa.

```figure
ae-crew-vs-flow
```

## Construí-lo

`code/main.py`Implementa versões STDlib de ambas as formas mais uma tripulação de três agentes.

Forma:

- `Agent`- Não .`Task`Classe de dados correspondente à superfície da CrewAI.
- `SequentialCrew.kickoff(inputs)`Executa tarefas em ordem de declarações, enfiando as saídas como `context`- Não .
- `HierarchicalCrew.kickoff(topic)`Adiciona um agente gerente escolhendo o próximo especialista a cada rodada, para no "feito".
- `Flow`com`@start`E ...`@listen(topic)`Decoradores, um pequeno ciclo de eventos e um rastro.
- `tool(name)`O decorador reflete o CrewAI.`@tool`- Forma.
- `Memory`com`short_term`- Não .`long_term`- Não .`entity`Lojas; comparação ridicularizada usa numpy.
- As respostas falsas de LLM são cordas codificadas com teclado de papel e prefixo de entrada.

Demo de concreto: pesquisador, escritor, equipe de editor produzindo um resumo sobre "engenharia de agentes 2026". Pesquisador tira (se burla) fontes. Escritor rascunhos. Editor apertando. A mesma equipe corre através de um fluxo para mostrar a forma determinista.

- É o que é ?

```bash
python3 code/main.py
```

Cobertas de rastreamento: sequenciais de tripulantes de threading de saídas através `context`, equipe hierárquica com seleções de gerentes (investigador, escritor, editor, então "feito"), fluxo executando os mesmos três passos com tópicos explícitos (`researched`- Não .`drafted`- Não .`edited`), chamadas de ferramentas encaminhadas através `@tool`, e memória a longo prazo sobreviver a dois golpes.

A traça da tripulação é fluida, o gerente pode reordenar em princípio.

## Usá-lo

- **CrewAI Flow**Mesmo quando o fluxo é um passo que chama`Crew.kickoff()`O fluxo dá o limite da auditoria.
- **CrewAI Crew (Sequential)**para o trabalho colaborativo de ordem clara, especialmente os primeiros projectos e os ciclos de revisão.
- **CrewAI Crew (Hierarchical)**Quando o roteamento depende da saída e você tem quatro ou mais especialistas.
- **LangGraph**(Lessão 13) para máquinas de estado explícito, currículo duradouro, ordem rigorosa.
- **AutoGen v0.4**(Lessão 14) para a concurência do modelo de ator e o isolamento de falhas.
- **OpenAI Agents SDK**(Lessão 16) para os produtos OpenAI-first com manchas e barris.
- **Claude Agent SDK**(Lessão 17) para produtos Claude-first com subagentes e loja de sessões.

## Envia-o

`outputs/skill-crew-or-flow.md`Seleciona Crew vs Flow para uma tarefa e prepara a implementação mínima. Hard rejeita sobre Crew-sem história de fundo, Flow-sem-tópicos explícitos, Hierárquico com menos de três especialistas.

## Encurralagens

- **Backstory as flavor.**Teste três variantes por agente, a variância é real, escolha uma e congela-a.
- **Skipping `expected_output`.**Sem um contrato por tarefa, as tarefas a seguir ao fluxo recolhem o que o LLM produziu.
- **Memory always-on.**O longo prazo escreve cada corrida, o vector DB cresce, a recuperação fica barulhenta, o escopo escreve para tarefas onde o fato é persistente.
- **Manager prompt drift.**Se o roteamento ficar estranho, despeja-o no modo verbose e leia.
- **Tool side effects in Crews.**Uma tripulação pode ligar para uma ferramenta mais vezes do que o esperado.

## Exercícios

1. Converte a tripulação Sequencial em Fluxo, conte os pontos de contacto onde a variabilidade diminui, note onde a legibilidade diminuiu.
2. Adicionar memória da entidade à tripulação: os fatos sobre um cliente persistem durante os lançamentos. Verifique a recuperação atrai a entidade certa.
3. Implementar um processo hierárquico em que o gerente se recusa a encaminhar para o editor até que a saída do escritor tenha pelo menos três parágrafos.
4. - O cabo a .`BaseTool`Subclasse para uma pesquisa na web. Compare a forma de rastreamento com a `@tool`versão decorativa.
5. Adicionar`output_pydantic=Brief`a tarefa do editor, onde `Brief`- Não .`title`- Não .`summary`- Não .`sections`Faça com que a saída da tarefa de escritor tenha JSON mal formado uma vez; verifique o comportamento de retest de CrewAI no rastreamento.
6. Leia a introdução dos documentos da CrewAI.`crewai`Que garantias a versão do STDlib ignorou?
7. Liga o agente de operações ou o Langfuse (Lessão 24) para uma corrida real.

## Termos-chave

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

## Mais leitura

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): conceitos e caminho de produção recomendado
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): forma orientada por eventos, `@start`- Não .`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)- Não .`@tool`- Não .`BaseTool`, kites de ferramentas incorporados
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): curto prazo, longo prazo, entidade, contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)Quando o multi-agente ajuda e quando não
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): a alternativa de máquina de Estado
