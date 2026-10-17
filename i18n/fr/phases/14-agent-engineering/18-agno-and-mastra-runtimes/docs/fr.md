# Temps de fonctionnement de l'agent de production  Installation rapide et flux de travail de type

> Un agent de production optimise le temps d'exécution de ce que les cadres de prototypage ignorent: le coût d'instantation, les surfaces de flux de travail typées et un backend prêt à être servi. L'association 2026: Agno (Python) vise l'instantation d'agent de microseconde et les backends FastAPI sans état. Mastra envoie des agents, des outils, des flux de travail, un routage de modèle unifié et un stockage composite sur le substrat SDK Vercel AI.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Objectifs d'apprentissage

- Identifiez les objectifs de performance d'Agnon et quand ils comptent.
- Nommer les trois primitifs de Mastra  Agents, Tools, Workflows  et les adaptateurs de serveurs pris en charge.
- Expliquez pourquoi un backend FastAPI sans état et scellé par session est la voie de production Agno recommandée.
- Choisissez Agno vs Mastra pour une pile donnée (Python-premier vs TypeScript-premier).

## Le problème

LangGraph, AutoGen, CrewAI sont très lourds en matière de cadres. Les équipes qui veulent " juste la boucle d'agent, rapide, dans mon temps d'exécution " atteignent Agno (Python) ou Mastra (TypeScript).

## Le concept

### Agno

- Python, anciennement Phi-data.
- "Pas de graphiques, de chaînes ou de motifs compliqués  juste python pur".
- Objectifs de performance de leurs documents: ~ 2μs d'instantanation d'agent, ~ 3,75 KiB de mémoire par agent, ~ 23 fournisseurs de modèles.
- Chaque requête démarre un nouvel agent; l'état de session se trouve dans un DB.
- Multimodal natif (texte, image, audio, vidéo, fichier) et RAG agent.

Les objectifs de vitesse comptent quand vous avez des milliers d'agents de courte durée par seconde (chat fan-in, pipelines d'évaluation).

### Mastra

- TypeScript, construit sur le SDK Vercel AI.
- Trois primitifs:**Agents**- Je suis là .**Tools**(type de zone), **Workflows**- Je suis désolé .
- Modèle unifié routeur  3 300+ modèles sur 94 fournisseurs (mars 2026).
- Stockage composé: mémoire, flux de travail, observabilité à différents arrière-plans; ClickHouse recommandé pour l'observabilité à l'échelle.
- Apache 2.0 avec `ee/`les répertoires sous licence d'entreprise disponible sous source.
- Adapters serveurs pour Express, Hono, Fastify, Koa; intégration Next.js et Astro de première classe.
- Il envoie Mastra Studio (host local:4111) pour débogage.
- 22k+ GitHub étoiles, 300k+ téléchargements hebdomadaires npm à 1.0 (janvier 2026).

### Positionnement

Ils ne veulent pas être LangGraph.

- **Language fit.**Agno pour les équipes Python-premières; Mastra pour TypeScript-première.
- **Runtime ergonomics.**Agno = dépenses générales presque nulles; Mastra = intégré à l'écosystème Vercel.
- **Observability.**Les deux sont intégrés à Langfuse/Phoenix/Opik (leçon 24) mais Mastra Studio est une première partie.

### Quand choisir chacun

- **Agno**Python backend, beaucoup d'agents de courte durée, des exigences de perf, FastAPI magasin.
- **Mastra** TypeScript backend, Next.js / Vercel déploiement, routage unifié de modèle multi-provider, outils de type Zod.
- **LangGraph**(Létion 13)  lorsque l'état durable et le raisonnement graphique explicite comptent plus que la vitesse brute.
- **OpenAI / Claude Agent SDK** lorsque vous souhaitez une forme productive du fournisseur (leçons 1617).

### Où ce modèle va mal

- **Perf-for-perf's-sake.**Je choisis Agno parce que "2μs" sonne bien quand la charge de travail est un appel lent par agent par demande.
- **Ecosystem lock-in.**L'intégration à la saveur Vercel de Mastra est un plus sur Vercel, un moins ailleurs.
- **Enterprise license confusion.**Le Mastra `ee/`Les annuaires sont disponibles sous source, pas Apache 2.0.

```figure
wb-runtime-spawn
```

## Faites-le

Cette leçon est principalement comparative  aucun artefact de code unique ne rendrait justice à ces deux cadres. Voir `code/main.py`pour un jouet côte à côte: un flux minimal "exécuter un agent, diffuser la sortie, persister la session" mis en œuvre deux fois (une fois en forme d'Agnon, une fois en forme de Mastra).

- Je vais le faire.

```
python3 code/main.py
```

Deux traces structurellement différentes mais fonctionnellement équivalentes.

## Utilisez-le

- **Agno** Python backend qui a besoin de vitesse et de forme FastAPI.
- **Mastra** TypeScript backend avec de nombreux fournisseurs et des primitifs de flux de travail.
- Les deux navires ont des crochets d'observation de première partie.

## La faire partir

`outputs/skill-runtime-picker.md`choisit Agno, Mastra, LangGraph ou un SDK de fournisseur en fonction de la pile, du budget de latence et de la forme opérationnelle.

## Exercices

1. Lisez les documents d'Agno, portez la boucle de réaction de la stdlib à Agno.
2. Lisez les documents de Mastra. Portez la même boucle à Mastra. Qu'est-ce qui a changé dans la typographie des outils (Zod vs rien)?
3. Mesurer la latence de l'instantanation de l'agent sur votre pile.
4. Conçuez une migration: si vous exécutez CrewAI en Python, qu'est-ce qui se passe si vous déménagez à Agno ?
5. Lisez le livre de Mastra `ee/`Quelles restrictions affecteraient une fourchette open source ?

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability each to a different store |
| Mastra Studio | "Local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "Not OSS" | License permits source reading but restricts commercial use |

## Pour en savoir plus

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) objectifs de performance, intégration de FastAPI
- [Mastra docs](https://mastra.ai/docs) primitifs, adaptateurs serveurs, routeur modèle
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) l'alternative à l'état-graphe
- [Comet Opik](https://www.comet.com/site/products/opik/) comparaisons d'observabilité citées par les intégrations de Mastra
