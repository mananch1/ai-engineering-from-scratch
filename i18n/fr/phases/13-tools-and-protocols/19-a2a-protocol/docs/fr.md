# A2A  Protocole agent-agent

> MCP est agent-outil. A2A (Agent2Agent) est un protocole ouvert permettant aux agents opaques construits sur différents cadres de collaborer. Sorti par Google en avril 2025, donné à la Fondation Linux en juin 2025, atteint la version 1.0 en avril 2026 avec plus de 150 supporters, notamment AWS, Cisco, Microsoft, Salesforce, SAP et ServiceNow. Il a absorbé l'ACP d'IBM et a ajouté l'extension des paiements AP2. Cette leçon traverse la carte de l'agent, le cycle de vie de la tâche et les trois liens de protocole, en utilisant les noms de fil A2A 1.0.1.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Distinguer les cas d'utilisation de l'agent à l'outil (MCP) des cas d'utilisation de l'agent à l'agent (A2A).
- Publier une carte d' agent à `/.well-known/agent-card.json`avec des compétences et `supportedInterfaces`les métadonnées.
- Suivez le cycle de vie de la tâche: `TASK_STATE_SUBMITTED`- Je suis là .`TASK_STATE_WORKING`- Je suis là .`TASK_STATE_INPUT_REQUIRED`, et les états terminaux `TASK_STATE_COMPLETED`- Je suis là .`TASK_STATE_FAILED`- Je suis là .`TASK_STATE_CANCELED`- Je suis là .`TASK_STATE_REJECTED`- Je suis désolé .
- Utilisez des messages dont chacune des parties contient un de `text`- Je suis là .`raw`- Je suis là .`url`ou `data`, et les objets comme sorties.

## Le problème

Un agent de service client doit déléguer la rédaction de rapports à un agent spécialisé en rédaction.

- L'API REST personnalisée fonctionne, mais chaque couplage est unique.
- Une base de code partagée, exige que les deux agents exécutent le même cadre.
- MCP: pas adapté: MCP est pour appeler des outils, pas pour deux agents collaborant tout en préservant le raisonnement interne opaque de chaque agent.

A2A remplit le vide. Il modélise l'interaction en tant qu'agent envoyer une tâche à un autre, avec un cycle de vie, des messages et des objets. L'état interne de l'agent appelé reste opaque  l'appelant ne voit que les transitions de l'état de tâche et les sorties éventuelles.

A2A est le protocole "laissez les agents à travers les cadres s'entretenir" qui ne remplace pas le MCP, les deux sont complémentaires.

## Le concept

### Agent Card

Chaque agent conforme aux A2A publie une carte à l' adresse `/.well-known/agent-card.json`- Le numéro de la liste:

```json
{
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "version": "1.2.0",
  "supportedInterfaces": [
    {
      "url": "https://research.example.com/a2a",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true},
  "securitySchemes": {
    "bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}
  },
  "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/markdown"],
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "tags": ["research", "summarization"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["text/markdown"]
    }
  ]
}
```

La découverte est basée sur l'URL: ramenez la carte, choisissez la première `supportedInterfaces`entrée dont `protocolBinding`Les modes d'entrée et de sortie sont des types de média.

### Cartes d'agent signées

Une carte peut porter un`signatures`Chaque entrée est un JWS (RFC 7515) calculé sur le JSON canonique RFC 8785 de la carte, avec le `signatures`Les consommateurs canonisent la carte de la même manière et vérifient.

### Cycle de vie des tâches

```text
TASK_STATE_SUBMITTED
  -> TASK_STATE_WORKING
  -> TASK_STATE_COMPLETED | TASK_STATE_FAILED | TASK_STATE_CANCELED | TASK_STATE_REJECTED

TASK_STATE_WORKING
  -> TASK_STATE_INPUT_REQUIRED
  -> TASK_STATE_WORKING (the client sends a message with the same taskId)
```

Les clients commencent par `SendMessage`Le serveur crée la tâche.`GetTask`ou de la rivière sur l' SSE avec `SendStreamingMessage`et `SubscribeToTask`Un ruisseau transporte`statusUpdate`et `artifactUpdate`Les résultats de la recherche sont les suivants:`final`Le drapeau.

### Messages et parties

Un message a une`messageId`, une `role`(le secteur de l'énergie)`ROLE_USER`ou `ROLE_AGENT`), et une ou plusieurs parties. Chaque partie contient exactement un champ de contenu, et ce nom de champ est le type.`kind`le champ.

- `text`: contenu simple.
- `raw`: octets de fichier, base64 en JSON, généralement avec `filename`et `mediaType`- Je suis désolé .
- `url`: un lien vers le contenu du fichier.
- `data`: charge utile JSON structurée (entrée structurée pour l'agent appelé).

Exemple:

```json
{
  "messageId": "msg-001",
  "role": "ROLE_USER",
  "parts": [
    {"text": "Summarize this paper."},
    {"raw": "...", "filename": "paper.pdf", "mediaType": "application/pdf"},
    {"data": {"targetLength": "3 paragraphs"}, "mediaType": "application/json"}
  ]
}
```

### Les objets

Les sorties sont des artifacts, pas des chaînes brutes.

```json
{
  "artifactId": "art-001",
  "name": "summary",
  "parts": [{"text": "...", "mediaType": "text/markdown"}]
}
```

Les objets peuvent être diffusés en morceaux.`artifactUpdate`L' événement porte l' artefact plus `append`et `lastChunk`- L'appelant s'accumule.

### Trois obligations de protocole

1. **JSON-RPC 2.0 over HTTP**(le secteur de l'énergie)`JSONRPC`Les méthodes sont PascalCase: `SendMessage`- Je suis là .`SendStreamingMessage`- Je suis là .`GetTask`- Je suis là .`ListTasks`- Je suis là .`CancelTask`- Je suis là .`SubscribeToTask`- Je suis là .`CreateTaskPushNotificationConfig`- Je suis là .`GetTaskPushNotificationConfig`- Je suis là .`ListTaskPushNotificationConfigs`- Je suis là .`DeleteTaskPushNotificationConfig`, et `GetExtendedAgentCard`- Je suis désolé .
2. **gRPC**(le secteur de l'énergie)`GRPC`Pour les environnements d'entreprise où le gRPC est natif.
3. **HTTP+JSON/REST**(le secteur de l'énergie)`HTTP+JSON`), les URL des ressources telles que `POST /message:send`et `GET /tasks/{id}`- Je suis désolé .

Les trois liaisons portent le même modèle de données.`supportedInterfaces`nom de l'entrée un obligatoire et son `protocolVersion`Les clients envoient l' en-tête .`A2A-Version: 1.0`sur chaque demande, parce qu'un serveur lit une demande sans elle comme version 0.3.

```http
POST /a2a HTTP/1.1
Host: research.example.com
Content-Type: application/json
A2A-Version: 1.0

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{"text": "Summarize this paper."}]
    }
  }
}
```

### Préservation de l'ouverture

Un principe de conception clé: l'état interne de l'agent appelé est opaque. L'appelant voit l'état de la tâche et les artefacts. La chaîne de pensée de l'agent appelé, ses appels à l'outil, sa délégation de sous-agent  sont tous invisibles. Cela est différent de MCP, où les appels à l'outil sont transparents.

Rationale: A2A permet aux concurrents de collaborer sans révéler les informations internes. A2A peut être "appeler cet agent de service client" sans que l'appelant apprenne comment cet agent implemente le service.

### L'année

- **2025-04-09.**Google annonce A2A.
- **2025-06-23.**Donné à la Fondation Linux.
- **2025-08.**Il absorbe l'ACP d'IBM.
- **2025-09.**Les navires de l'extension AP2 (paiements par agent).
- **2026-04.**V1.0 est sorti avec plus de 150 organisations de soutien.

### Relation avec le PCM

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | Agent Card `securitySchemes` + `securityRequirements` |
| Transport | Stdio / Streamable HTTP | JSON-RPC / gRPC / HTTP+JSON |

Utilisez MCP lorsque vous voulez invoquer un outil spécifique. Utilisez A2A lorsque vous voulez déléguer une tâche entière à un autre agent. De nombreux systèmes de production utilisent les deux: un agent utilise MCP pour sa couche d'outils et A2A pour sa couche de collaboration.

```figure
a2a-task-lifecycle
```

## Utilisez-le

`code/main.py`Il implique un harnais A2A minimal: l'agent écrivain publie sa carte, l'agent de recherche l'envoie une `SendMessage`La requête est accompagnée d'une partie PDF et d'une instruction texte, et la tâche passe à travers `TASK_STATE_WORKING`- Je suis là.`TASK_STATE_INPUT_REQUIRED`- Je suis là.`TASK_STATE_WORKING`- Je suis là.`TASK_STATE_COMPLETED`Tout stdlib; utilise un transport en mémoire pour se concentrer sur les formes de message.

À quoi regarder:

- La forme de la carte agent JSON.
- Classification des tâches du côté serveur et transitions d'état.
- Parties typées par le champ de contenu présent.
- `TASK_STATE_INPUT_REQUIRED`branche au milieu de la tâche.
- Le retour de l'artefact à la fin.

## La faire partir

Cette leçon produit `outputs/skill-a2a-agent-spec.md`. Étant donné qu'un nouvel agent doit être appelé par d'autres agents, la compétence produit le JSON de la carte agent, le schéma de compétences et le schéma des points d'extrémité.

## Exercices

1. On court .`code/main.py`- Tracer l'ensemble du cycle de vie de la tâche, y compris la`TASK_STATE_INPUT_REQUIRED`Arrêtez-vous lorsque l'agent appelé demande une clarification.

2. Ajoutez une carte d'agent signée.`signatures`avec `alg`à `HS256`, signer le JSON canonique de la carte sans le `signatures`Écrivez un vérificateur et confirmez qu'il échoue sur une carte mutée.

3. Exécuter la tâche en streaming avec `SendStreamingMessage`: l' agent de rédaction émet le `task`, trois `artifactUpdate`Les pièces et un`statusUpdate`avec `TASK_STATE_COMPLETED`L'appelant accumule les morceaux.

4. Conceptionner un agent A2A qui embrasse un serveur MCP. Mape chaque outil MCP à une compétence A2A. Notez les compromis  quelle opacité est perdue?

5. Lisez l'annonce d'A2A v1.0 et identifiez la seule fonctionnalité qui n'est pas encore mise en œuvre par aucun cadre à partir d'avril 2026.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Open protocol for opaque agent collaboration |
| Agent Card | "`/.well-known/agent-card.json`" | Published metadata describing an agent's skills and `supportedInterfaces` |
| Skill | "A callable unit" | A named operation the agent supports (analog to MCP tool) |
| Task | "Unit of delegation" | A work item with a lifecycle and final artifact |
| Message | "Task input" | Carries Parts (`text`, `raw`, `url`, `data`) |
| Part | "Typed chunk" | Exactly one of `text` / `raw` / `url` / `data`, plus optional `mediaType`; no `kind` field |
| Artifact | "Task output" | Named, typed output returned on completion |
| AP2 | "Agent Payments Protocol" | Payments extension built on A2A; card signing is core A2A (`signatures`) |
| Opacity | "Black-box collaboration" | Called agent's internals are hidden from caller |
| `TASK_STATE_INPUT_REQUIRED` | "Task pause" | Interrupted state when the agent needs more info |

## Pour en savoir plus

- [a2a-protocol.org](https://a2a-protocol.org/latest/) spécification canonique A2A
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) Implémentations de référence et KDD
- [A2A v1.0.1 release](https://github.com/a2aproject/A2A/tree/v1.0.1): les étiquettes `docs/specification.md`et la réglementation `specification/a2a.proto`Cette leçon suit
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) Transfert de gouvernance en juin 2025
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) feuille de route et dynamique des partenaires
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) note de libération v1.0 et orientation rétrocompatible
