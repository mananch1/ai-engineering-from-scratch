# Utilização de Ferramentas e Função

> O Toolformer (Schick et al., 2023) começou a anotação de ferramentas auto-supervisionada. O Berkeley Function Calling Leaderboard V4 (Patil et al., 2025) define a barra de 2026: 40% agente, 30% multi-turn, 10% ao vivo, 10% não ao vivo, 10% alucinação.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## Objetivos de aprendizagem

- Explique o sinal de treinamento auto-supervisionado do Toolformer: mantenha as anotações da ferramenta apenas quando a execução reduz a perda do próximo token.
- Nomear as cinco categorias de avaliação do BFCL V4 e o que cada uma mede.
- Implementar um registro de ferramentas stdlib com validação de esquema, coerção de argumentos e sandboxing de execução.
- Diagnóstico dos três problemas abertos de 2026: cadeia de ferramentas de longo horizonte, tomada de decisões dinâmicas e memória.

## O problema

O uso inicial de ferramentas perguntou: o modelo pode prever uma chamada de função correta? o uso moderno de ferramentas pergunta: o modelo pode ferramentas de cadeia através de 40 passos, com memória, com observabilidade parcial, com recuperação de falhas de ferramentas, sem alucinar ferramentas que não existem?

A ferramentaforum estabeleceu a linha de base: os modelos podem aprender quando chamar ferramentas com auto-supervisão.

## O conceito

### Instrumentformer (Schick et al., NeurIPS 2023)

Ideia: deixe o modelo anote seu próprio corpus de pré-treino com chamadas de API candidato. Para cada candidato, execute-o. Mantenha a anotação somente se incluir o resultado da ferramenta reduz a perda no próximo token.

Ferramentas abrangidas: calculadora, sistema de avaliação de qualidade, motores de busca, tradutor, calendário.

Resultado em escala: o uso de ferramentas surge em escala. Modelos menores prejudicam as anotações de ferramentas; modelos maiores ganham. É por isso que os modelos de fronteira de 2026 têm um forte uso de ferramentas, enquanto a maioria dos modelos 7B precisa de ajustes explícitos no uso de ferramentas para serem confiáveis.

### Berkeley Função chamada Leaderboard V4 (Patil et al., ICML 2025)

BFCL é a avaliação de facto de 2026.

- **Agentic (40%)** Tráetoras de agentes completos: memória, múltiplos turnos, decisões dinâmicas.
- **Multi-Turn (30%)** conversas interativas com cadeias de ferramentas.
- **Live (10%)** Instruções reais enviadas pelo utilizador (distribuição mais difícil).
- **Non-Live (10%)** casos de ensaio sintéticos.
- **Hallucination (10%)** detectar quando não deve ser chamada nenhuma ferramenta.

V3 introduziu avaliação baseada em estado: após uma sequência de ferramentas, verifique o estado real da API (por exemplo, "o arquivo foi criado?") em vez de corresponder à AST das chamadas de ferramentas. V4 adicionou pesquisa web, memória e categorias de sensibilidade ao formato.

Descoberta chave de 2026: a chamada de função de turno único está quase resolvida. As falhas se concentram na memória (carregando contexto através de turnos), na tomada de decisões dinâmicas (escolhendo ferramentas com base em resultados anteriores), em cadeias de longo horizonte (deslocação após 20+ passos) e na detecção de alucinações (recusando-se a ligar quando nenhuma ferramenta se encaixa).

### Esquema de ferramenta

Cada fornecedor tem um esquema.

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Utilizações antropológicas `input_schema`O OpenAI utiliza`function.parameters`As descrições são carregadoras, o modelo as lê para escolher a ferramenta certa. Descrições de ferramentas ruins são a principal causa de falhas escolhidas com ferramentas erradas.

### Validação de argumentos

Não confiem em nenhuma chamada de ferramenta.

1. **Type coercion.**O modelo pode retornar uma cadeia "5" onde o esquema diz int. Forçar se é inequívoco; rejeitar se não.
2. **Enum validation.**Se o esquema diz:`status in {"open", "closed"}`e emissões de modelo `"in_progress"`, rejeitar com um erro descritivo.
3. **Required fields.**Falta campo necessário -> observação de erro imediato de volta ao modelo, não uma queda.
4. **Format validation.**Data, e-mails, URLs validam com parseres de concreto, não regex.

Cada falha de validação deve devolver uma observação estruturada para que o modelo possa tentar novamente com a forma correta.

### Chamadas paralelas de ferramentas

Os provedores modernos suportam chamadas paralelas de ferramentas em um turno de assistente.

1. O modelo emite 3 chamadas de ferramenta com distinção `tool_use_id`S.
2. O runtime executa-as (em paralelo, se independente).
3. Cada resultado retorna como um`tool_result`Bloco correlacionado por `tool_use_id`- Não .

Regra de engenharia: tratar as identidades de correlação como suportes de carga. Trocá-las e você obtém o roteamento de ferramenta para resultado errado.

### Sandboxing

A execução da ferramenta é a fronteira da caixa de areia. Veja a lição 09 para detalhes. Versão curta: cada ferramenta deve especificar a superfície de leitura/escrita, acesso à rede, tempo de saída, limite de memória.`run_shell(cmd)`é uma bandeira vermelha; especificamente `git_status()`É mais seguro.

```figure
tool-routing
```

## Construí-lo

`code/main.py`Implementa um registo de ferramentas de forma de produção:

- JSON Schema subconjunto validador (só stdlib).
- Registro de ferramenta com descrição, esquema de entrada, tempo de encerramento e executor.
- A coerção de argumentos e a validação enum.
- Disposição paralela de ferramentas com identificação de correlação.
- Observações de erro como cadeias estruturadas.

- É o que é ?

```
python3 code/main.py
```

O rastro mostra um mini agente chamando três ferramentas em uma vez, com uma chamada deliberadamente malformada que é rejeitada com um erro descritivo que o modelo pode agir.

## Usá-lo

Cada fornecedor tem seu próprio esquema de ferramentas  Antropic, OpenAI, Gemini, Bedrock. Use uma camada de tradução (OpenAI Agents SDK, Vercel AI SDK, LangChain tool adapter) se você precisar de multi-provedor.

## Envia-o

`outputs/skill-tool-registry.md`gera um catálogo de ferramentas, esquema e registro para um determinado domínio de tarefa. Inclui verificações de qualidade de descrição (a descrição de cada ferramenta diz ao modelo quando usá-lo?).

## Exercícios

1. Adicione uma ferramenta "no-op" que permite que o modelo se recuse explícitamente a usar qualquer outra ferramenta.
2. Implementar a coerção de argumentos para int-as-string e flotar-as-string.
3. Adicione um timeout por ferramenta e um interruptor de circuito (rejeita a ferramenta por 60 anos após 3 falhas consecutivas).
4. Leia a descrição do BFCL V4. Escolha uma categoria (por exemplo, "multi-turn") e execute 10 exemplos de instruções através do seu agente.
5. Portar o validador do STDlib para a Pydantic ou Zod.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Structured-output tool invocation with validated schema |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — keep tool calls whose results reduce next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Function signature for the model" | name, description, JSON Schema of arguments |
| tool_use_id | "Correlation ID" | Ties a tool call to its result; essential for parallel dispatch |
| Hallucination detection | "Know when not to call" | V4 category: refuse to call when no tool fits |
| Argument coercion | "String-to-int repair" | Narrow fixes for predictable schema-mismatch; reject if ambiguous |
| Sandboxing | "Tool execution boundary" | Per-tool read/write surface, network, timeout, memory cap |

## Mais leitura

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) Anotado de ferramentas auto-supervisionadas
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) Valor de referência de avaliação 2026
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) Esquema de ferramentas de produção no SDK Claude Agent
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) Tipo de ferramenta e Guardrails
