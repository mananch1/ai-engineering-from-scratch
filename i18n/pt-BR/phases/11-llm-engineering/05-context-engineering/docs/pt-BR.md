# Engenharia de contexto: Windows, Orçamentos, Memória e Recuperação

> A engenharia de prompt é um subconjunto. A engenharia de contexto é todo o jogo. Um prompt é uma cadeia que você digita. O contexto é tudo o que entra na janela do modelo: instruções do sistema, documentos recuperados, definições de ferramentas, histórico de conversa, alguns exemplos de tiros e o próprio prompt. Os melhores engenheiros de IA em 2026 são engenheiros de contexto. Eles decidem o que entra, o que fica fora e em que ordem.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10 (LLMs from Scratch), Phase 11 Lesson 01-02
**Time:** ~90 minutes
**Related:**Fase 11 · 15 (Cachagem de contato)  o layout amigável ao cache é uma extensão da engenharia de contexto. Fase 5 · 28 (Evaluation de longo contexto) para medir o perdido no meio com NIAH/RULER.

## Objetivos de aprendizagem

- Calcular orçamentos de tokens em todos os componentes da janela de contexto (promete do sistema, ferramentas, histórico, documentos recuperados, espaço de geração)
- Implementar estratégias de gerenciamento de janela de contexto: truncamento, resumo e janela deslizante para histórico de conversação
- Priorizar e ordenar os componentes do contexto para maximizar a atenção do modelo às informações mais relevantes
- Construir um conjunto de contexto que atribui dinâmicamente tokens com base no tipo de consulta e espaço disponível da janela

## O problema

Claude Opus 4.7 tem uma janela de tokens de 200K (1M em beta). GPT-5 tem 400K. Gemini 3 Pro tem 2M. Llama 4 afirma 10M. Estes números soam enormes até que você os preencha.

Aqui está uma divisão real para um assistente de codificação. Promete do sistema: 500 tokens. Definições de ferramentas para 50 ferramentas: 8.000 tokens. Documentação recuperada: 4.000 tokens. História de conversação (10 voltas): 6.000 tokens. Questão atual do usuário: 200 tokens. Orçamento de geração (máxima saída): 4.000 tokens. Total: 22.700 tokens. Isso é apenas 18% de uma janela de 128K.

Mas a atenção não se escala linearmente com o comprimento do contexto.$O(n^2)$A maioria dos modelos de produção utiliza variantes de atenção eficientes). Mais importante, a precisão da recuperação diminui. O teste "Agulha em um Pacote de Hay" mostra que os modelos têm dificuldade em encontrar informações colocadas no meio de contextos longos. Pesquisa de Liu et al. (2023) mostrou que os MLLs recuperam informações no início e no final de contextos longos com precisão quase perfeita, mas a precisão cai de 10-20% para as informações colocadas no meio (posições 40-70% do contexto). Este efeito "perdido no meio" varia de modelo para modelo, mas afeta todas as arquiteturas atuais.

A lição prática: ter 200K tokens disponíveis não significa que usar 200K tokens seja eficaz. Um contexto de token 10K cuidadosamente curado geralmente supera um contexto de token 100K descarregado. A engenharia de contexto é a disciplina de maximizar a relação sinal-ruído dentro da janela de contexto.

Cada token que colocas na janela desplace um token que poderia conter informações mais relevantes. Cada definição de ferramenta irrelevante, cada turno de conversa obsoleta, cada pedaço de texto recuperado que não responde à pergunta - cada um torna o modelo um pouco pior na tarefa.

## O conceito

### A Janela de Contexto é um recurso escasso

Pense na janela de contexto como RAM, não disco. É rápido e diretamente acessível, mas limitado. Você não pode caber em tudo. Você deve escolher.

```mermaid
graph TD
    subgraph Window["Context Window (128K tokens)"]
        direction TB
        S["System Prompt\n~500 tokens"] --> T["Tool Definitions\n~2K-8K tokens"]
        T --> R["Retrieved Context\n~2K-10K tokens"]
        R --> H["Conversation History\n~2K-20K tokens"]
        H --> F["Few-shot Examples\n~1K-3K tokens"]
        F --> Q["User Query\n~100-500 tokens"]
        Q --> G["Generation Budget\n~2K-8K tokens"]
    end

    style S fill:#1a1a2e,stroke:#e94560,color:#fff
    style T fill:#1a1a2e,stroke:#0f3460,color:#fff
    style R fill:#1a1a2e,stroke:#ffa500,color:#fff
    style H fill:#1a1a2e,stroke:#51cf66,color:#fff
    style F fill:#1a1a2e,stroke:#9b59b6,color:#fff
    style Q fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#0f3460,color:#fff
```

Cada componente compete por espaço. Adicionar mais definições de ferramentas significa menos espaço para o histórico de conversação. Adicionar mais contexto recuperado significa menos espaço para alguns exemplos de tiros. Engenharia de contexto é a arte de alocar esse orçamento para maximizar o desempenho da tarefa.

### Perdido no meio

A descoberta empírica mais importante na engenharia de contexto. Os modelos atendem melhor às informações no início e no final do contexto.

Liu et al. (2023) testaram isso sistematicamente. Eles colocaram um documento relevante entre 20 documentos irrelevantes em várias posições e mediram a precisão da resposta. Quando o documento relevante era o primeiro ou o último, a precisão era de 85-90%. Quando estava no meio (posição 10 de 20), a precisão caiu para 60-70%.

Isto tem implicações directas em engenharia:

- Colocar as informações mais importantes em primeiro lugar (instruções de sistema, instruções críticas)
- Colocar a consulta atual e o contexto mais relevante em último lugar (precisão recente ajuda)
- Tratar o meio do contexto como a zona de menor prioridade
- Se você deve incluir informações no meio, duplique o ponto chave no final

```mermaid
graph LR
    subgraph Attention["Attention Distribution Across Context"]
        direction LR
        P1["Position 0-20%\nHIGH attention\n(system prompt)"]
        P2["Position 20-40%\nMODERATE"]
        P3["Position 40-70%\nLOW attention\n(lost in middle)"]
        P4["Position 70-90%\nMODERATE"]
        P5["Position 90-100%\nHIGH attention\n(current query)"]
    end

    style P1 fill:#51cf66,color:#000
    style P2 fill:#ffa500,color:#000
    style P3 fill:#ff6b6b,color:#fff
    style P4 fill:#ffa500,color:#000
    style P5 fill:#51cf66,color:#000
```

### Componentes do contexto

**System prompt**Claude Code usa cerca de 6.000 tokens para seu sistema de instruções, incluindo definições de ferramentas e instruções de comportamento. Mantém-no apertado. Cada palavra no sistema de instruções é repetida em cada chamada de API.

**Tool definitions**Cada ferramenta adiciona 50-200 tokens (nome, descrição, esquema de parâmetros). 50 ferramentas em 150 tokens cada é 7.500 tokens antes de qualquer conversa acontecer. A seleção dinâmica de ferramentas - apenas incluindo ferramentas relevantes para a consulta atual - pode reduzir isso em 60-80%.

**Retrieved context**A qualidade da recuperação determina diretamente a qualidade da resposta. A recuperação ruim é pior do que nenhuma recuperação - enche a janela de ruído e engana ativamente o modelo.

**Conversation history**Uma conversa de 50 voltas a 200 tokens por turno é 10.000 tokens de história. A maioria é irrelevante para a consulta atual.

**Few-shot examples**Os exemplos bem escolhidos, muitas vezes, melhoram a qualidade da saída mais do que milhares de tokens de instruções.

**Generation budget**Se preencher a janela de capacidade, o modelo não tem espaço para responder. Reserve pelo menos 2.000-4.000 tokens para geração.

### Estratégias de compressão de contexto

**History summarization**Em vez de manter todas as voltas anteriores verbatim, resuma a conversa periodicamente. "Discutimos X, decidimos Y, e o usuário quer Z" em 100 tokens substitui 10 voltas que levaram 2.000 tokens.

**Relevance filtering**Se você tiver recuperado 10 pedaços, mas apenas 3 são relevantes, descartar os outros 7. É melhor ter 3 pedaços altamente relevantes do que 10 pedaços mediocres.

**Tool pruning**A questão de código não precisa de ferramentas de calendário. Uma pergunta de programação não precisa de ferramentas de sistema de arquivos. Isso pode reduzir as definições de ferramentas de 8.000 tokens para 1.000.

**Recursive summarization**Para documentos muito longos, resuma em etapas. Primeiro resuma cada seção, depois resuma os resumos. Um documento de 50 páginas se torna um digestão de 500 tokens que capta os pontos-chave.

### Sistemas de memória

A engenharia de contexto abrange três horizontes de tempo.

**Short-term memory**A conversa atual. Armazenada na janela de contexto diretamente. Cresce a cada virada. Gerida por resumo e truncation.

**Long-term memory**"O usuário prefere o TypeScript". "O projeto usa PostgreSQL". Armazenado em um banco de dados, recuperado no início da sessão. Claude Code armazena isso em arquivos CLAUDE.md. ChatGPT armazena-lo em sua função de memória.

**Episodic memory**"Na terça-feira passada, resolvemos um problema similar no módulo auth". Armazenado como embutidos, recuperados quando a conversa atual coincide com um episódio anterior.

```mermaid
graph TD
    subgraph Memory["Memory Architecture"]
        direction TB
        STM["Short-term Memory\n(current conversation)\nDirect in context window"]
        LTM["Long-term Memory\n(facts, preferences)\nDB -> retrieved on session start"]
        EM["Episodic Memory\n(past interactions)\nEmbeddings -> retrieved on similarity"]
    end

    Q["Current Query"] --> STM
    Q --> LTM
    Q --> EM

    STM --> CW["Context Window"]
    LTM --> CW
    EM --> CW

    style STM fill:#1a1a2e,stroke:#51cf66,color:#fff
    style LTM fill:#1a1a2e,stroke:#0f3460,color:#fff
    style EM fill:#1a1a2e,stroke:#e94560,color:#fff
    style CW fill:#1a1a2e,stroke:#ffa500,color:#fff
```

### Assembléia de contexto dinâmico

A principal ideia: diferentes consultas precisam de contextos diferentes. Um sistema estático de prompt + ferramentas estáticas + histórico estático é desperdiçoso. Os melhores sistemas montam dinamicamente contexto por consulta.

1. Classificar a intenção da consulta
2. Selecionar as ferramentas relevantes (não todas)
3. Retirar documentos relevantes (não um conjunto fixo)
4. Incluir as curvas de história relevantes (não todas)
5. Adicionar alguns exemplos de tiros que correspondem ao tipo de tarefa
6. Ordenar tudo por importância: crítico primeiro, importante último, opcional no meio

É isso que separa uma boa aplicação de IA de uma grande. O modelo é o mesmo. O contexto é o diferenciador.

```figure
lost-in-the-middle
```

## Construí-lo

### Passo 1: Contador de Tokens

Não pode orçar o que não pode medir. Construa um contador de tokens simples (aproximação usando divisão de espaço em branco, já que a contagem exata depende do tokenizer).

```python
import json
import numpy as np
from collections import OrderedDict

def count_tokens(text):
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def count_tokens_json(obj):
    return count_tokens(json.dumps(obj))
```

### Passo 2: Gestor de orçamento de contexto

Um gerente de orçamento rastreia quantos tokens cada componente usa e impõe limites.

```python
class ContextBudget:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.max_tokens = max_tokens
        self.generation_reserve = generation_reserve
        self.available = max_tokens - generation_reserve
        self.allocations = OrderedDict()

    def allocate(self, component, content, max_tokens=None):
        tokens = count_tokens(content)
        if max_tokens and tokens > max_tokens:
            words = content.split()
            target_words = int(max_tokens / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        used = sum(self.allocations.values())
        if used + tokens > self.available:
            allowed = self.available - used
            if allowed <= 0:
                return None, 0
            words = content.split()
            target_words = int(allowed / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        self.allocations[component] = tokens
        return content, tokens

    def remaining(self):
        used = sum(self.allocations.values())
        return self.available - used

    def utilization(self):
        used = sum(self.allocations.values())
        return used / self.max_tokens

    def report(self):
        total_used = sum(self.allocations.values())
        lines = []
        lines.append(f"Context Budget Report ({self.max_tokens:,} token window)")
        lines.append("-" * 50)
        for component, tokens in self.allocations.items():
            pct = tokens / self.max_tokens * 100
            bar = "#" * int(pct / 2)
            lines.append(f"  {component:<25} {tokens:>6} tokens ({pct:>5.1f}%) {bar}")
        lines.append("-" * 50)
        lines.append(f"  {'Used':<25} {total_used:>6} tokens ({total_used/self.max_tokens*100:.1f}%)")
        lines.append(f"  {'Generation reserve':<25} {self.generation_reserve:>6} tokens")
        lines.append(f"  {'Remaining':<25} {self.remaining():>6} tokens")
        return "\n".join(lines)
```

### Passo 3: Reordem perdida no meio

Implementar a estratégia de reordenação: os elementos mais importantes são os primeiros e últimos, os menos importantes estão no meio.

```python
def reorder_lost_in_middle(items, scores):
    paired = sorted(zip(scores, items), reverse=True)
    sorted_items = [item for _, item in paired]

    if len(sorted_items) <= 2:
        return sorted_items

    first_half = sorted_items[::2]
    second_half = sorted_items[1::2]
    second_half.reverse()

    return first_half + second_half

def score_relevance(query, documents):
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        if not query_words:
            scores.append(0.0)
            continue
        overlap = len(query_words & doc_words) / len(query_words)
        scores.append(round(overlap, 3))
    return scores
```

### Passo 4: Compressor História da Conversação

Resumir uma conversa antiga volta para recuperar o orçamento de tokens.

```python
class ConversationManager:
    def __init__(self, max_history_tokens=5000):
        self.turns = []
        self.summaries = []
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role, content):
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    def _compress_if_needed(self):
        total = sum(count_tokens(t["content"]) for t in self.turns)
        if total <= self.max_history_tokens:
            return

        while total > self.max_history_tokens and len(self.turns) > 4:
            old_turns = self.turns[:2]
            summary = self._summarize_turns(old_turns)
            self.summaries.append(summary)
            self.turns = self.turns[2:]
            total = sum(count_tokens(t["content"]) for t in self.turns)

    def _summarize_turns(self, turns):
        parts = []
        for t in turns:
            content = t["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"{t['role']}: {content}")
        return "Previous: " + " | ".join(parts)

    def get_context(self):
        parts = []
        if self.summaries:
            parts.append("[Conversation Summary]")
            for s in self.summaries:
                parts.append(s)
        parts.append("[Recent Conversation]")
        for t in self.turns:
            parts.append(f"{t['role']}: {t['content']}")
        return "\n".join(parts)

    def token_count(self):
        return count_tokens(self.get_context())
```

### Passo 5: Selector de ferramentas dinâmicas

Incluir apenas ferramentas relevantes para a consulta atual. Classificar intenção, em seguida, filtrar.

```python
TOOL_REGISTRY = {
    "read_file": {
        "description": "Read contents of a file",
        "tokens": 120,
        "categories": ["code", "files"],
    },
    "write_file": {
        "description": "Write content to a file",
        "tokens": 150,
        "categories": ["code", "files"],
    },
    "search_code": {
        "description": "Search for patterns in codebase",
        "tokens": 130,
        "categories": ["code"],
    },
    "run_command": {
        "description": "Execute a shell command",
        "tokens": 140,
        "categories": ["code", "system"],
    },
    "create_calendar_event": {
        "description": "Create a new calendar event",
        "tokens": 180,
        "categories": ["calendar"],
    },
    "list_emails": {
        "description": "List recent emails",
        "tokens": 160,
        "categories": ["email"],
    },
    "send_email": {
        "description": "Send an email message",
        "tokens": 200,
        "categories": ["email"],
    },
    "web_search": {
        "description": "Search the web for information",
        "tokens": 140,
        "categories": ["research"],
    },
    "query_database": {
        "description": "Run a SQL query on the database",
        "tokens": 170,
        "categories": ["code", "data"],
    },
    "generate_chart": {
        "description": "Generate a chart from data",
        "tokens": 190,
        "categories": ["data", "visualization"],
    },
}

def classify_intent(query):
    query_lower = query.lower()

    intent_keywords = {
        "code": ["code", "function", "bug", "error", "file", "implement", "refactor", "debug", "test"],
        "calendar": ["meeting", "schedule", "calendar", "appointment", "event"],
        "email": ["email", "mail", "send", "inbox", "message"],
        "research": ["search", "find", "what is", "how does", "explain", "look up"],
        "data": ["data", "query", "database", "chart", "graph", "analytics", "sql"],
    }

    scores = {}
    for intent, keywords in intent_keywords.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return ["code"]

    max_score = max(scores.values())
    return [intent for intent, score in scores.items() if score >= max_score * 0.5]

def select_tools(query, token_budget=2000):
    intents = classify_intent(query)
    relevant = {}
    total_tokens = 0

    for name, tool in TOOL_REGISTRY.items():
        if any(cat in intents for cat in tool["categories"]):
            if total_tokens + tool["tokens"] <= token_budget:
                relevant[name] = tool
                total_tokens += tool["tokens"]

    return relevant, total_tokens
```

### Passo 6: Projeto de montagem de contexto completo

Enviar tudo em conjunto, e, em função de uma consulta, montar dinamicamente o contexto ideal.

```python
class ContextEngine:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.budget = ContextBudget(max_tokens, generation_reserve)
        self.conversation = ConversationManager(max_history_tokens=5000)
        self.system_prompt = (
            "You are a helpful AI assistant. You have access to tools for "
            "code editing, file management, web search, and data analysis. "
            "Use the appropriate tools for each task. Be concise and accurate."
        )
        self.knowledge_base = [
            "Python 3.12 introduced type parameter syntax for generic classes using bracket notation.",
            "The project uses PostgreSQL 16 with pgvector for embedding storage.",
            "Authentication is handled by Supabase Auth with JWT tokens.",
            "The frontend is built with Next.js 15 using the App Router.",
            "API rate limits are set to 100 requests per minute per user.",
            "The deployment pipeline uses GitHub Actions with Docker multi-stage builds.",
            "Test coverage must be above 80% for all new modules.",
            "The codebase follows the repository pattern for data access.",
        ]

    def assemble(self, query):
        self.budget = ContextBudget(self.budget.max_tokens, self.budget.generation_reserve)

        system_content, _ = self.budget.allocate("system_prompt", self.system_prompt, max_tokens=1000)

        tools, tool_tokens = select_tools(query, token_budget=2000)
        tool_text = json.dumps(list(tools.keys()))
        tool_content, _ = self.budget.allocate("tools", tool_text, max_tokens=2000)

        relevance = score_relevance(query, self.knowledge_base)
        threshold = 0.1
        relevant_docs = [
            doc for doc, score in zip(self.knowledge_base, relevance)
            if score >= threshold
        ]

        if relevant_docs:
            doc_scores = [s for s in relevance if s >= threshold]
            reordered = reorder_lost_in_middle(relevant_docs, doc_scores)
            doc_text = "\n".join(reordered)
            doc_content, _ = self.budget.allocate("retrieved_context", doc_text, max_tokens=3000)

        history_text = self.conversation.get_context()
        if history_text.strip():
            history_content, _ = self.budget.allocate("conversation_history", history_text, max_tokens=5000)

        query_content, _ = self.budget.allocate("user_query", query, max_tokens=500)

        return self.budget

    def chat(self, query):
        self.conversation.add_turn("user", query)
        budget = self.assemble(query)
        response = f"[Response to: {query[:50]}...]"
        self.conversation.add_turn("assistant", response)
        return budget


def run_demo():
    print("=" * 60)
    print("  Context Engineering Pipeline Demo")
    print("=" * 60)

    engine = ContextEngine(max_tokens=128000, generation_reserve=4000)

    print("\n--- Query 1: Code task ---")
    budget = engine.chat("Fix the bug in the authentication module where JWT tokens expire too early")
    print(budget.report())

    print("\n--- Query 2: Research task ---")
    budget = engine.chat("What is the best approach for implementing vector search in PostgreSQL?")
    print(budget.report())

    print("\n--- Query 3: After conversation history builds up ---")
    for i in range(8):
        engine.conversation.add_turn("user", f"Follow-up question number {i+1} about the implementation details of the system")
        engine.conversation.add_turn("assistant", f"Here is the response to follow-up {i+1} with technical details about the architecture")

    budget = engine.chat("Now implement the changes we discussed")
    print(budget.report())

    print("\n--- Tool Selection Examples ---")
    test_queries = [
        "Fix the bug in auth.py",
        "Schedule a meeting with the team for Tuesday",
        "Show me the database query performance stats",
        "Search for best practices on error handling",
    ]

    for q in test_queries:
        tools, tokens = select_tools(q)
        intents = classify_intent(q)
        print(f"\n  Query: {q}")
        print(f"  Intents: {intents}")
        print(f"  Tools: {list(tools.keys())} ({tokens} tokens)")

    print("\n--- Lost-in-the-Middle Reordering ---")
    docs = ["Doc A (most relevant)", "Doc B (somewhat relevant)", "Doc C (least relevant)",
            "Doc D (relevant)", "Doc E (moderately relevant)"]
    scores = [0.95, 0.60, 0.20, 0.80, 0.50]
    reordered = reorder_lost_in_middle(docs, scores)
    print(f"  Original order: {docs}")
    print(f"  Scores:         {scores}")
    print(f"  Reordered:      {reordered}")
    print(f"  (Most relevant at start and end, least relevant in middle)")
```

## Usá-lo

### Contexto gerenciado por arneses

Claude Code gerencia o contexto com uma abordagem em camadas. O prompt do sistema inclui regras de comportamento e definições de ferramentas (~ 6K tokens). Quando você abre um arquivo, seu conteúdo é injetado como contexto. Quando você pesquisa, os resultados são adicionados. As conversas antigas são resumidas. CLAUDE.md fornece memória de longo prazo que persiste em todas as sessões.

A decisão de engenharia chave: o Claude Code não descarta toda a sua base de código no contexto.

### Carregamento de contexto dinâmico

Cursor indexa toda a sua base de código em embutidos. Quando você escreve uma consulta, ele recupera os arquivos e blocos de código mais relevantes usando semelhança vetorial. Somente essas peças entram na janela de contexto. Uma base de código de 500K-linha é comprimida para os 5-10 blocos de código mais relevantes.

Este é o padrão: incorporar tudo, recuperar à demanda, incluir apenas o que importa.

### Assistente de Memória de Longo Prazo

O ChatGPT armazena as preferências e fatos do usuário como memória de longo prazo. Em cada início de conversa, as memórias relevantes são retiradas e incluídas no prompt do sistema. "O usuário prefere Python" custa 5 tokens, mas salva centenas de tokens de instruções repetidas em todas as conversas.

### RAG como Engenharia de Contexto

A geração recuperada-agendada é a engenharia de contexto formal. Em vez de encher o conhecimento nos pesos do modelo (formação) ou no sistema de instrução (contexto estático), você retira documentos relevantes no momento da consulta e os injeta na janela de contexto. O conjunto do RAG -- fragmentação, inserção, recuperação, re-ranqueamento -- existe para resolver um problema: colocar a informação certa na janela de contexto.

## Envia-o

Esta lição produz`outputs/prompt-context-optimizer.md`-- um prompt reutilizável que verifica uma estratégia de montagem de contexto e recomenda otimizações. Alimenta-o com o seu prompt do sistema, contagem de ferramentas, comprimento médio do histórico e estratégia de recuperação, e identifica desperdício de tokens e sugere melhorias.

Também produz `outputs/skill-context-engineering.md`-- um quadro de decisão para a concepção de canais de montagem de contexto com base no tipo de tarefa, no tamanho da janela de contexto e no orçamento de latência.

## Exercícios

1. Adicionar um "detetor de resíduos de tokens" à classe ContextBudget. Ele deve marcar componentes que utilizam mais de 30% do orçamento e sugerir estratégias de compressão específicas para cada tipo de componente (resumir o histórico, ferramentas de poda, re-ranquear documentos).

2. Implementar deduplicação semântica para contexto recuperado. Se dois documentos recuperados são mais de 80% semelhantes (por sobreposição de palavras ou semelhança cosina de suas incorporações), mantenha apenas o mais alto.

3. Construir uma ferramenta de "replay contextual". Dado uma transcrição de conversa, replay através do ContextEngine e visualizar como a alocação de orçamento muda de vez em quando. Plot o uso de tokens por componente ao longo do tempo. Identificar a vez em que o contexto começa a ser comprimido.

4. Implementar um selector de ferramentas baseado em prioridades. Em vez de incluir/excluir binário, atribuir a cada ferramenta uma pontuação de relevância para a consulta atual. Inclua ferramentas em ordem decrescente de relevância até que o orçamento da ferramenta seja esgotado. Compare o desempenho da tarefa com as ferramentas 5, 10, 20 e 50 incluídas.

5. Construir um compressor de contexto multi-estratégico. Implementar três estratégias de compressão (truncation, summarization, extraction of key sentences) e compará-las em um conjunto de 20 documentos. Medir a compensação entre a relação de compressão e a retenção de informações (a versão comprimida ainda contém a resposta à consulta?).

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Context window | "How much the model can read" | The maximum number of tokens (input + output) the model processes in a single forward pass -- 400K for GPT-5, 200K (1M beta) for Claude Opus 4.7, 2M for Gemini 3 Pro |
| Context engineering | "Advanced prompt engineering" | The discipline of deciding what goes into the context window, in what order, and at what priority -- encompasses retrieval, compression, tool selection, and memory management |
| Lost-in-the-middle | "Models forget stuff in the middle" | Empirical finding that LLMs attend better to the beginning and end of context, with 10-20% accuracy drop for information placed in the middle |
| Token budget | "How many tokens you have left" | An explicit allocation of context window capacity across components (system prompt, tools, history, retrieval, generation) with per-component limits |
| Dynamic context | "Loading stuff on the fly" | Assembling the context window differently for each query based on intent classification, relevant tool selection, and retrieval results |
| History summarization | "Compressing the conversation" | Replacing verbatim old conversation turns with a concise summary, reducing token cost while preserving key information |
| Tool pruning | "Only including relevant tools" | Classifying query intent and only including tool definitions that match, reducing tool token cost by 60-80% |
| Long-term memory | "Remembering across sessions" | Facts and preferences stored in a database and retrieved at session start -- CLAUDE.md, ChatGPT Memory, and similar systems |
| Episodic memory | "Remembering specific past events" | Past interactions stored as embeddings and retrieved when the current query is similar to a past conversation |
| Generation budget | "Room for the answer" | Tokens reserved for the model's output -- if the context fills the window completely, the model has no room to respond |

## Mais leitura

- [Liu et al., 2023 -- "Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172)-- o estudo definitivo sobre a atenção dependente da posição, mostrando que os modelos lutam com a informação no meio de contextos longos
- [Anthropic's Contextual Retrieval blog post](https://www.anthropic.com/news/contextual-retrieval)-- como a Anthropic aborda a recuperação de peças conscientes do contexto, reduzindo a falha da recuperação em 49%
- [Simon Willison's "Context Engineering"](https://simonwillison.net/2025/Jun/27/context-engineering/)- o post no blog que nomeou a disciplina e a distinguiu da engenharia rápida
- [LangChain documentation on RAG](https://python.langchain.com/docs/tutorials/rag/)-- implementação prática da geração aumentada de recuperação como padrão de engenharia contextual
- [Greg Kamradt's Needle in a Haystack test](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)-- o índice de referência que revelou falhas de recuperação dependentes da posição em todos os principais modelos
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102)-- por que o comprimento do contexto leva à memória e à latência, e como o cache KV, MQA e GQA alteram o cálculo do orçamento.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369)-- as duas fases de inferência que fazem as longas indicações caras na TTFT mas baratas na TPOT; a verdade fundamental por trás de compromissos de conteúdo.
- [Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (EMNLP 2023)](https://arxiv.org/abs/2305.13245)-- o papel de atenção de consulta agrupada que corta a memória KV 8x nos decodificadores de produção sem perda de qualidade.
