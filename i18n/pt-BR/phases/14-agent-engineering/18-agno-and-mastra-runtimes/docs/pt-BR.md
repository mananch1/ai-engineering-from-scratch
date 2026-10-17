# Tempo de execução do agente de produção  Instalação rápida e fluxos de trabalho tipografados

> Um agente de produção otimiza o tempo de execução do que os frameworks de prototyping ignoram: custo de instanciamento, superfícies de fluxo de trabalho digitalizadas e um backend pronto para servido. O acoplamento de 2026: Agno (Python) visa instanciamento de agente de microsecondas e backends FastAPI sem estado. Mastra envia agentes, ferramentas, fluxos de trabalho, roteamento de modelo unificado e armazenamento composto no substrato Vercel AI SDK.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Objetivos de aprendizagem

- Identificar os objetivos de desempenho do Agno e quando são importantes.
- Nomear os três primitivos de Mastra  Agentes, Ferramentas, Fluxos de Trabalho  e os adaptadores de servidor suportados.
- Explique por que um backend FastAPI sem estatuto é o caminho recomendado para a produção Agno.
- Escolha Agno vs Mastra para uma determinada pilha (Python-primeiro vs TypeScript-primeiro).

## O problema

LangGraph, AutoGen, CrewAI são pesados em estrutura. As equipes que querem "apenas o loop do agente, rápido, no meu tempo de execução" alcançam Agno (Python) ou Mastra (TypeScript). Ambos trocam alguns dos primitivos de propriedade do framework por velocidade bruta e um ajuste mais próximo à pilha circundante.

## O conceito

### Agno

- Tempo de execução do Python, anteriormente Phi-data.
- "Não há gráficos, cadeias ou padrões complicados, apenas Python puro".
- Objectivos de desempenho de seus documentos: ~ 2μs instantiation agente, ~ 3,75 KiB de memória por agente, ~ 23 fornecedores de modelos.
- Percurso de produção: backend FastAPI sem estado de sessão. Cada solicitação inicia um novo agente; estado de sessão vive em um DB.
- Multimodal nativo (texto, imagem, áudio, vídeo, arquivo) e RAG agente.

Os objetivos de velocidade importam quando você tem milhares de agentes de curta duração por segundo (fantasma de bate-papo, pipelines de avaliação), mas não importam quando um agente corre por 10 minutos.

### Mastra

- TypeScript, construído no SDK Vercel AI.
- Três primitivos:**Agents**- Não .**Tools**(Típico de zona), **Workflows**- Não .
- Roteador Modelo Unificado  3.300+ modelos em 94 provedores (março 2026).
- Armazenamento composto: memória, fluxos de trabalho, observabilidade para diferentes backends; ClickHouse recomendado para observabilidade em escala.
- Apache 2.0 com `ee/`Directórios sob licença empresarial disponível na fonte.
- Adaptadores de servidor para Express, Hono, Fastify, Koa; integração Next.js e Astro de primeira classe.
- Navio Mastra Studio (host local:4111) para depuração.
- 22k+ estrelas do GitHub, 300k+ downloads semanais em 1.0 (Jan 2026).

### Posicionamento

Nem sequer tentam ser LangGraph.

- **Language fit.**Agno para as equipes Python-primeira; Mastra para TypeScript-primeira.
- **Runtime ergonomics.**Agno = quase zero despesas gerais; Mastra = integrado com o ecossistema Vercel.
- **Observability.**Ambos se integram com Langfuse/Phoenix/Opik (Lessão 24) mas o Mastra Studio é de primeira parte.

### Quando escolher cada um

- **Agno**Python backend, muitos agentes de curta duração, fortes requisitos de perf, FastAPI loja.
- **Mastra** Backend TypeScript, Next.js / Vercel deployment, roteamento unificado de modelos multi-provedor, ferramentas Zod-typed.
- **LangGraph**(Lessão 13)  quando o estado durável e o raciocínio gráfico explícito importam mais do que a velocidade bruta.
- **OpenAI / Claude Agent SDK** quando se quer a forma produtizada do fornecedor (Lessões 1617).

### Onde este padrão vai mal

- **Perf-for-perf's-sake.**A escolha do Agno porque "2μs" soa bem quando a carga de trabalho é uma chamada lenta por pedido.
- **Ecosystem lock-in.**A integração com sabor a Vercel de Mastra é um mais em Vercel, um menos em outros lugares.
- **Enterprise license confusion.**O Mastra `ee/`Os diretórios estão disponíveis em fonte, não no Apache 2.0.

```figure
wb-runtime-spawn
```

## Construí-lo

Esta lição é principalmente comparativa  nenhum artefato de código único faria justiça para ambos os quadros. Veja `code/main.py`Para um brinquedo lado a lado: um mínimo de "exercer um agente, fluir a saída, persistir sessão" fluxo implementado duas vezes (uma vez em forma de Agno, uma vez em forma de Mastra).

- É o que é ?

```
python3 code/main.py
```

Duas traças estruturalmente diferentes, mas funcionalmente equivalentes.

## Usá-lo

- **Agno** Python backend que precisa de velocidade e forma FastAPI.
- **Mastra** Backend do TypeScript com muitos provedores e primitivos de fluxo de trabalho.
- Ambos os ganchos de observação de primeira parte da nave, ambos integrados com o Langfuse.

## Envia-o

`outputs/skill-runtime-picker.md`escolhe Agno, Mastra, LangGraph ou um SDK do provedor com base na pilha, no orçamento de latência e na forma operacional.

## Exercícios

1. Leia os documentos do Agno, transporte o loop do STDlib ReAct para Agno.
2. Leia os documentos de Mastra. Portar o mesmo ciclo para Mastra. O que mudou na digitação de ferramentas (Zod vs nada)?
3. Método de referência: medir a latência de instanciamento do agente na sua pilha.
4. Desenhar uma migração: se você está executando CrewAI em Python, o que se rompe se você se mudar para Agno?
5. Leia o Mastra `ee/`Que restrições afetariam um fork de código aberto?

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability each to a different store |
| Mastra Studio | "Local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "Not OSS" | License permits source reading but restricts commercial use |

## Mais leitura

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) Objectivos de desempenho, integração da FastAPI
- [Mastra docs](https://mastra.ai/docs) primitivos, adaptadores de servidores, Router Modelo
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) a alternativa de grafico estadual
- [Comet Opik](https://www.comet.com/site/products/opik/) comparações de observabilidade citadas pelas integrações de Mastra
