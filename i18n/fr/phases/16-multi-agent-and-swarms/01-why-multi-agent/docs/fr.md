# Pourquoi Multi-Agent ?

> Un agent frappe un mur, le mouvement intelligent n'est pas un agent plus grand, c'est plus d'agents.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Objectifs d'apprentissage

- Identifier le plafond de l'agent unique (survol de contexte, expertise mixte, goulets d'étranglement séquentiels) et expliquer quand la division en plusieurs agents est la bonne décision
- Comparer les modèles d'orchestration (pipeline, ventilateur parallèle, superviseur, hiérarchique) et sélectionner le bon pour une structure de tâche donnée
- Conception d'un système multi-agents avec des limites de rôle claires, un état partagé et un contrat de communication
- Analyse des compromis entre la complexité multi-agent (la latence, le coût, la difficulté de débogage) et la simplicité d'un agent unique

## Le problème

Vous avez construit un seul agent dans la phase 14. Il fonctionne. Il peut lire des fichiers, exécuter des commandes, appeler des API et raisonner sur les résultats. Ensuite, vous le pointez vers une base de code réelle: 200 fichiers, trois langues, des tests qui dépendent de l'infrastructure, et une exigence de rechercher des API externes avant d'écrire du code.

L'agent se noie. Non pas parce que le LLM est stupide, mais parce que la tâche dépasse ce qu'un boucle d'agent peut gérer. La fenêtre contextuelle se remplit de contenu de fichier. L'agent oublie ce qu'il a lu 40 appels d'outils. Il essaie d'être chercheur, un codeur et un critique à la fois, et fait les trois mal.

C'est le plafond à agent unique, on le frappe chaque fois qu'une tâche exige:

- **More context than fits in one window**- lire 50 fichiers passe 200 000 jetons
- **Different expertise at different stages**- la recherche nécessite une incitation différente de la génération de code
- **Work that can happen in parallel**- Pourquoi lire trois fichiers en séquence quand on peut les lire simultanément ?

## Le concept

### Le plafond à agent unique

Un seul agent est une boucle, une fenêtre contextuelle, une requête système.

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

Trois choses se brisent:

1. **Context saturation**Au tour 30, l'agent a consommé 150 000 jetons de contenu de fichiers, de commandes et de raisonnement préalable.

2. **Role confusion**- un prompt système qui dit "vous êtes un chercheur, un codeur, un réviseur et un testeur" produit un agent qui fait la moitié de la recherche, la moitié des codes, et ne termine jamais la révision.

3. **Sequential bottleneck**- l'agent lit le dossier A, puis le dossier B, puis le dossier C. Trois appels en série, trois exécutions en série.

### La solution à plusieurs agents

Donnez à chaque agent un travail, une fenêtre contextuelle et une requête système adaptée à ce travail:

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

Chaque agent a:
- Une requête de système centrée ("Vous êtes un réviseur de code. Votre seul travail est de trouver des bugs. ")
- Sa propre fenêtre de contexte (non polluée par le travail d'autres agents)
- Un contrat de sortie/entrée clair (recevoir des notes de recherche, code de sortie)

### Des systèmes réels qui le font

**Claude Code subagents**- quand Claude Code engendre un subagent avec `Task`Le parent garde son contexte propre, l'enfant travaille avec concentration et renvoie un résumé.

**Devin**- un agent de planification, un agent de programmation et un agent de navigateur. le planificateur découpera le travail en étapes. le codeur écrit le code. le navigateur recherche la documentation. chacune a un contexte distinct.

**Multi-agent coding teams (SWE-bench)**- les systèmes de performance supérieure sur le banc SWE utilisent un chercheur qui lit la base de code, un planificateur qui conçoit la correction et un codeur qui la met en œuvre.

**ChatGPT Deep Research**- génère plusieurs agents de recherche en parallèle, chacun explorant un angle différent, puis synthétise les résultats.

### Le spectre

Le multi-agent n'est pas binaire, c'est un spectre:

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

**Single agent**- Une boucle, une commande.

**Subagents**- un parent donne naissance à des enfants pour des sous-tâches ciblées. le parent maintient le plan. les enfants rapportent. c'est ce que fait Claude Code.

**Pipeline**- les agents fonctionnent en séquence. la sortie de l'agent A devient l'entrée de l'agent B. Bon pour les flux de travail stagnées: recherche -> code -> examen -> test.

**Team**- les agents fonctionnent en parallèle avec un bus de messagerie partagé. chacun a un rôle. un orchestrateur coordonne. bon quand différentes compétences sont nécessaires simultanément.

**Swarm**- beaucoup d'agents identiques ou presque identiques avec un état partagé. pas d'orchestre fixe. les agents prennent le travail à la file d'attente. bon pour des tâches parallèles à haut débit.

### Les quatre modèles multi-agents

#### Modèle 1: Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Chaque agent transforme les données et les transmet, simple à raisonner, mais l'échec d'une étape bloque le reste.

#### Modèle 2: Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Partager le travail entre des agents parallèles, puis fusionner les résultats.

#### Modèle 3: Orchestreur-travailleur

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

Un orchestrateur intelligent décide de ce qu'il doit faire, délègue les résultats aux travailleurs et synthétise les résultats.

#### Modèle 4: Les pairs

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

Aucun orchestrateur central, les agents communiquent entre eux, les décisions émergent de l'interaction, plus difficile à déboguer, mais à échelle de nombreux agents.

### Quand ne pas utiliser de multi-agent

Le débogage passe de "lire une conversation" à "tracer les messages à travers cinq agents".

**Stay single-agent when:**
- La tâche s'inscrit dans une fenêtre contextuelle (moins de 100k tokens de données de travail)
- Vous n'avez pas besoin de différentes instructions système pour les différentes étapes
- L' exécution séquentielle est assez rapide
- La tâche est assez simple pour qu'en la divisant, on ajoute plus de coûts généraux que de valeur.

**The complexity cost:**
- Chaque limite d'agent est une étape de compression perdue: le contexte complet de l'agent A est résumé dans un message pour l'agent B
- La logique de coordination (qui fait quoi, quand, dans quel ordre) est sa propre source de bugs
- L'augmentation de la latence: N agents signifie N séries LLM appels minimum, plus si elles ont besoin de parler en avant et en arrière
- Le coût est multiplié par un: chaque agent brûle des jetons indépendamment

Règle générale: si une tâche prend moins de 20 appels d'outils et s'adapte à 100 000 jetons, gardez-la à un seul agent.

```figure
swarm-messages
```

## Faites-le

### Étape 1: L'agent célibataire surchargé

Il a un énorme système de commande et une fenêtre contextuelle contenant des recherches, du code et des critiques:

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

Problèmes avec cette approche:
- La fenêtre contextuelle grandit à chaque étape.
- Le système est générique, il ne peut pas être réglé pour chaque étape.
- Rien ne fonctionne en parallèle.

### Étape 2: Agents spécialisés

Chaque agent a un travail:

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

Chaque spécialiste a une mise en œuvre centrée, et chacun obtient une fenêtre contextuelle propre avec seulement les informations dont il a besoin.

### Étape 3: Coordonner les messages

Envoyez les spécialistes avec un message explicite:

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

Chaque agent reçoit seulement les messages qui lui sont adressés, sans pollution de contexte, les 50 000 jetons de lecture de la documentation du chercheur n'entrent jamais dans le contexte du réviseur.

### Étape 4: Comparer

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

La version multi-agent utilise plus de jetons totaux (trois agents, trois appels LLM séparés), mais le contexte de chaque agent reste propre.

## Utilisez-le

Cette leçon fournit une information réutilisable pour décider quand se rendre multi-agent.`outputs/prompt-multi-agent-decision.md`- Je suis désolé .

## Exercices

1. Ajouter un quatrième spécialiste: un agent " testeur " qui reçoit du code du codeur et examine les commentaires du réviseur, puis écrit des tests
2. Modifier le pipeline afin que le réviseur puisse envoyer des commentaires au codeur pour une boucle de révision (max 2 tours)
3. Convertir le pipeline séquentiel en un ventilateur: exécuter le chercheur et un agent "analyseur de besoins" en parallèle, puis fusionner leurs sorties avant de passer au codeur

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | A set of peer agents with shared state and no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and delegates but may not do the actual work. |
| Coordinator | "The traffic cop" | A non-agent component (often just code, not an LLM) that routes messages between agents based on rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns that arise from agent interactions but were not explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting a task across parallel agents (fan-out), then combining their results (fan-in). |
| Message passing | "Agents talk to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## Pour en savoir plus

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)- analyse des modèles multi-agents
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)- Le cadre de conversation multi-agents de Microsoft
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code)- comment Claude Code délègue avec la tâche
- [CrewAI documentation](https://docs.crewai.com/)- cadre multi-agents fondé sur les rôles
