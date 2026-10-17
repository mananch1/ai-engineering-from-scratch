# Roteamento de modelo como uma primitiva de redução de custos

> Um corretor dinâmico avalia cada solicitação (tipo de tarefa, comprimento de token, semelhança de inserção, confiança) e envia consultas simples para um modelo barato, aumentando as complexas para um modelo de fronteira. Também chamado de cascata de modelos. Os estudos de caso de produção mostram uma redução de custos de 20-60% na iso-qualidade em todas as implementações dos EUA/Reino Unido/UE; uma melhoria de eficiência de roteamento de 30% em SaaS de grande volume transforma-se em economias anuais de seis dígitos. O contexto de 2026 é que os preços de inferência LLM caíram ~ 10x por ano  um token de classe GPT-4 foi de $20/M to ~$0,40/M, de finais de 2022 a 2026. A maior parte da queda é melhor servindo pilhas (fase 17 · 04-09), não hardware. O roteamento é como você converte essa queda de preço em margem sem regressão do produto. O modo de falha é a deriva do modelo barato: a rota empurra 40% para um modelo mais fraco, a qualidade cai de 3-5% nas tarefas de raciocínio, ninguém percebe por um quarto. Roteiras de portal por métricas de qualidade online, não apenas conjuntos de avaliação offline.

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 minutes

## Objetivos de aprendizagem

- Explique o modelo em cascata: barato primeiro com verificação de confiança, escalada em baixa confiança.
- Enumerar os quatro sinais de roteamento (classificação da tarefa, comprimento da tarefa, incorporação de semelhança com o conjunto de hard-known, autoconfiança a partir da primeira passagem).
- Calcular o custo combinado esperado na divisão de roteamento-alvo e na tolerância à perda de qualidade.
- Nomear a métrica de monitoramento de deriva (gate de qualidade on-line) que pega o modelo barato.

## O problema

O seu serviço custa US$ 80 mil por mês no GPT-5. As suas análises mostram que 70% das perguntas são simples: "qual é a hora em Paris?" "reformula essa frase". Um modelo de classe Haiku lida perfeitamente com 3% do custo. 30% precisa do raciocínio do GPT-5.

Se você encaminhar 70% para barato e 30% para caro, sua conta cai cerca de 65% na mesma qualidade do produto. Isto é encaminhamento. O truque é construir o corretor sem regressar a qualidade.

## O conceito

### Quatro sinais de encaminhamento

1. **Task classification**Pode ser um classificador baseado em regras, um pequeno LLM (Haiku-classe a $ 0,25/M), ou incorporando semelhança com baldes rotulados.

2. **Prompt length**As instruções <500 tokens geralmente não precisam de fronteira para a coerência.

3. **Embedding similarity to known-hard set**Se a consulta estiver próxima (cosin > 0,88) de um balde conhecido de duração, escala directamente para a fronteira.

4. **Self-confidence from first-pass**Se os registos do modelo revelarem baixa confiança OR se ele recusa OR saiba a linguagem de cobertura, tente novamente na fronteira.

### Três padrões

**Pre-route**(classificador de frente): ~ 5-10ms de latência adicionada; mais rápido em geral.

**Cascade**(mais barato primeiro, em baixa confiança): ~ 1,2x de latência média (excurso barato mais verificação), ~ 2x em escalada.

**Ensemble route**(exercício barato e fronteira em paralelo para uma amostra, escolha de modelo de recompensa): mais alta qualidade, maior custo; utilização apenas para A/B crítica.

### Implementação

As gateways de IA (fase 17 · 19) expõem o roteamento.`router`Configurar com fallback e roteamento de custos. Portkey tem guardas + roteamento. Kong AI Gateway tem roteamento baseado em plugins.

O código aberto: RouteLLM (LMSYS), Não Diamante (comercial), Prompt Mule.

### A curva de preços de 2026

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

A maior parte da melhoria é a eficiência de serviço  as lições principais na Fase 17 · 04-09 transformadas em quedas de custos do lado do provedor. O roteamento permite capturar esses ganhos na camada de aplicativo em vez de esperar que todos os seus usuários migrem para o nível barato.

### A deriva é o risco real

A sua rota envia 40% para o modelo barato. Ao longo de seis meses, a distribuição de tarefas muda (os usuários ficam mais sofisticados, fazem perguntas mais longas). O roteador não percebe porque seu classificador foi treinado em dados do Q1. A qualidade cai silenciosamente. Ninguém se queixa suficientemente alto. Você descobre em um benchmark do concorrente que perdeu.

Roteiras de entrada por métricas de qualidade online:

- Os utilizadores colocam os dedos para cima/para baixo por rota.
- Jury de LLM automatizado numa amostra de retenção (5%) por rota.
- Taxa de escalada: se a cascata estiver a subir mais de 30%, o modelo barato está a ser super-routado.
- Taxa de recusa por rota.

### Números que você deve lembrar

- 2026 poupança de roteamento em iso-qualidade: estudos de caso de 20 a 60%.
- Descenso dos preços dos LLM 2022-2026: ~ 10x ao ano agregado.
- Nível GPT-4 2022 vs 2026: ~$20/M → ~$0,40/M.
- Impacto de latência em cascata: ~ 1,2x média, ~ 2x escalada (~ 10% do tráfego).

```figure
model-cascade-router
```

## Usá-lo

`code/main.py`Simula a pré-ruta, a cascata e o conjunto numa carga de trabalho mista.

## Envia-o

Esta lição produz`outputs/skill-router-plan.md`- Tendo em conta a carga de trabalho e o orçamento de qualidade, escolhe um padrão de roteamento e sinais.

## Exercícios

1. Corra .`code/main.py`Em que piso de precisão a cascata bate o pré-caminho?
2. A sua base de usuários é de 30% empresarial (questões complexas), 70% gratuito (simples).
3. Uma rota reduz a qualidade em 2% mas economiza 40%. É um navio?
4. Implementar uma verificação de confiança usando logprobs de OpenAI / APIs antropóficas. Qual é o limiar que você começa com?
5. Ao longo de seis meses, a taxa de escalada sobe de 8% para 22%. Diagnóstico de três causas e a correção para cada uma.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## Mais leitura

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://web.archive.org/web/20260413143335/https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) Porta de entrada multi-modelo com primitivos de roteamento.
