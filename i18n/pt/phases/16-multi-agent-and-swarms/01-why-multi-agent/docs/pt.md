# Porquê Multidistribuído?

> Um agente bate numa parede, o movimento inteligente não é um agente maior, é mais agentes.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Objetivos de aprendizagem

- Identificar o limite máximo de um agente único (excesso de conteúdo, experiência mista, garganta de engarrafamento sequencial) e explicar quando dividir em vários agentes é a melhor medida
- Compare padrões de orquestração (pipeline, fan-out paralelo, supervisor, hierárquico) e selecione o certo para uma dada estrutura de tarefa
- Projetar um sistema multi-agente com limites claros de papel, estado compartilhado e contrato de comunicação
- Analisar as compensações da complexidade multi-agente (latencia, custo, dificuldade de depuração) versus a simplicidade de um único agente

## O problema

Você construiu um único agente na Fase 14. Funciona. Ele pode ler arquivos, executar comandos, ligar para APIs e argumentar sobre resultados. Depois você aponta para uma base de código real: 200 arquivos, três idiomas, testes que dependem de infraestrutura e um requisito para pesquisar APIs externas antes de escrever código.

O agente se esmaga. Não porque o LLM seja estúpido, mas porque a tarefa excede o que um agente loop pode lidar. A janela de contexto se enche de conteúdo de arquivo. O agente esquece o que leu 40 chamadas de ferramenta atrás. Ele tenta ser um pesquisador, um codificador e um revisor de uma só vez, e faz mal os três.

Este é o teto de agente único, é atingido sempre que uma tarefa requer:

- **More context than fits in one window**- ler 50 arquivos passa 200 mil tokens
- **Different expertise at different stages**- a investigação exige uma motivação diferente da geração de código
- **Work that can happen in parallel**- Porque ler três arquivos sequencialmente quando você pode lê-los simultaneamente?

## O conceito

### O teto de um só agente

Um único agente é um loop, uma janela de contexto, um sistema de instruções.

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

Três coisas quebram:

1. **Context saturation**A partir da curva 30, o agente já consumiu 150 mil tokens de conteúdo de arquivo, saídas de comando e raciocínio prévio.

2. **Role confusion**- um sistema de instruções que diz "você é um pesquisador, codificador, revisor e testador" produz um agente que metade pesquisa, metade código, e nunca termina a revisão.

3. **Sequential bottleneck**- O agente lê o arquivo A, depois o arquivo B, depois o arquivo C. Três chamadas de LLM em série, três execuções em série de ferramentas.

### A solução multi-agente

Divida o trabalho, dá a cada agente um trabalho, uma janela de contexto e um sistema de resposta ajustada para esse trabalho:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

Cada agente tem:
- Um sistema focado de solicitação ("Você é um revisor de código. Seu único trabalho é encontrar bugs. ")
- O seu próprio quadro de contexto (não poluído pelo trabalho de outros agentes)
- Um contrato de entrada/saída claro (recebe notas de investigação, código de saída)

### Sistemas reais que fazem isso

**Claude Code subagents**- quando Claude Code gera um subagente com`Task`O pai mantém o contexto limpo, o filho trabalha focado e retorna um resumo.

**Devin**- executa um agente de planejamento, um agente de codificação e um agente de navegador. O planejador divide o trabalho em etapas. O codificador escreve código. O navegador pesquisa a documentação. Cada um tem contexto separado.

**Multi-agent coding teams (SWE-bench)**- os sistemas de melhor desempenho no banco SWE utilizam um pesquisador que lê a base de código, um planejador que desenha a correcção e um codificador que a implementa.

**ChatGPT Deep Research**- gerar múltiplos agentes de busca em paralelo, cada um explorando um ângulo diferente, e depois sintetizar os resultados.

### O Espectro

O multi-agente não é binário, é um espectro:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent**- Um loop, um prompt.

**Subagents**- um pai gera filhos para subtarefas focalizadas. o pai mantém o plano. os filhos relatam. é o que Claude Code faz.

**Pipeline**- agentes executam em sequência. A saída do agente A torna-se a entrada do agente B. Bom para fluxos de trabalho em etapas: pesquisa -> código -> revisão -> teste.

**Team**- Agentes funcionam em paralelo com um bus de mensagens compartilhado cada um tem um papel um orquestrador coordena bom quando diferentes habilidades são necessárias simultaneamente

**Swarm**- muitos agentes idênticos ou quase idênticos com estado compartilhado.

### Os quatro padrões multi-agentes

#### Modelo 1: oleoduto

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Cada agente transforma os dados e os transmite, simples de raciocinar, mas o fracasso numa fase bloqueia o resto.

#### Padrão 2: Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Dividir o trabalho em agentes paralelos, depois fundir os resultados.

#### Modelo 3: Orquestra-Trabalhador

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

Um orquestrador inteligente decide o que fazer, delega aos trabalhadores e sintetiza os resultados.

#### Patrão 4: Escolha de pares

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

Não há orquestra central, os agentes comunicam entre pares, as decisões surgem da interação, mais difícil de depurar, mas a escala é de muitos agentes.

### Quando não usar multi-agente

Multi-agente adiciona complexidade. Cada mensagem entre agentes é um ponto de falha potencial. Debug vai de "leia uma conversa" para " rastrear mensagens em cinco agentes".

**Stay single-agent when:**
- A tarefa cabe em uma janela de contexto (menos de ~ 100k tokens de dados de trabalho)
- Não é preciso diferentes instruções do sistema para diferentes etapas
- A execução sequencial é rápida o suficiente.
- A tarefa é simples o suficiente para que a divisão adicione mais gastos gerais do que valor

**The complexity cost:**
- Cada limite de agente é um passo de compressão perdida: o contexto completo do agente A é resumido em uma mensagem para o agente B
- A lógica de coordenação (quem faz o que, quando, em que ordem) é sua própria fonte de bugs
- Aumentam a latência: N agentes significa N chamadas sérias LLM mínimo, mais se eles precisam de falar para frente e para trás
- Multiplicação de custos: cada agente queima tokens de forma independente

Regra geral: se uma tarefa requer menos de 20 chamadas de ferramentas e se encaixa em 100k tokens, mantenha-a monoposto.

```figure
swarm-messages
```

## Construí-lo

### Passo 1: O Agente Solteiro sobrecarregado

Aqui está um único agente tentando fazer tudo. Tem um enorme sistema de resposta e uma janela de contexto contendo pesquisa, código e avaliações:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Problemas com esta abordagem:
- A janela de contexto cresce com cada etapa.
- O sistema de instrução é genérico, não pode ser ajustado para cada etapa.
- Nada corre em paralelo.

### Passo 2: Agentes especializados

Agora, divide-o, cada agente tem um trabalho:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

Cada especialista tem um prompt focado, cada um obtém uma janela de contexto limpa com apenas a entrada que precisa.

### Passo 3: Coordenar através de mensagens

Entregue os especialistas com mensagem explícita:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Cada agente recebe apenas as mensagens que lhe são dirigidas, sem contaminação de contexto, os 50 mil tokens de leitura de documentação do pesquisador nunca entram no contexto do revisor.

### Passo 4: Comparar

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

A versão multi-agente usa mais tokens totais (três agentes, três chamadas separadas de LLM), mas o contexto de cada agente permanece limpo.

## Usá-lo

Esta lição produz uma indicação reutilizables para decidir quando se deve ir para multi-agente.`outputs/prompt-multi-agent-decision.md`- Não .

## Exercícios

1. Adicione um quarto especialista: um agente "tester" que recebe código do codificador e revisar feedback do revisor, em seguida, escreve testes
2. Modificar o pipeline para que o revisor possa enviar feedback de volta ao codificador para um ciclo de revisão (max 2 rodadas)
3. Converte o pipeline sequencial em um fan-out: execute o pesquisador e um agente "analista de requisitos" em paralelo, em seguida, funcione suas saídas antes de passar para o codificador

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | A set of peer agents with shared state and no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and delegates but may not do the actual work. |
| Coordinator | "The traffic cop" | A non-agent component (often just code, not an LLM) that routes messages between agents based on rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns that arise from agent interactions but were not explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting a task across parallel agents (fan-out), then combining their results (fan-in). |
| Message passing | "Agents talk to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## Mais leitura

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)- análise dos padrões de agentes múltiplos
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)- O framework de conversação multi-agente da Microsoft
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code)- como Claude Code delega com a tarefa
- [CrewAI documentation](https://docs.crewai.com/)- quadro multiagente baseado em funções
