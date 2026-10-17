# Exit structuré  Schéma JSON, Pydantic, Zod, décoding restreint

> " Demandez au modèle gentiment de retourner JSON " échoue 5 à 15% du temps, même sur les modèles frontaliers. Les sorties structurées comblent ce fossé avec le décoding restreint: le modèle est littéralement empêché d'émettre un jeton qui violerait le schéma.`responseSchema`, l'IA de Pydantic `output_type`, et Zod's `.parse`Cette leçon construit le validateur de schéma et les apprenants de contrat en mode strict utiliseront pour chaque pipeline d'extraction de production.

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Écrivez un schéma JSON 2020-12 pour une cible d'extraction en utilisant les bonnes contraintes (enum, min/max, requis, schéma).
- Expliquez pourquoi le mode strict et le décoding restreint offrent des garanties différentes de "validité après génération".
- Distinguer les trois modes d'échec: erreur de partage, violation du schéma, refus du modèle.
- Envoyer un pipeline d'extraction avec réparation typique et traitement de refus typique.

## Le problème

Un agent qui lit un mail de commande d' achat doit transformer le texte libre en `{customer, line_items, total_usd}`- Trois approches.

**Approach one: prompt for JSON.**"Répondre en JSON avec les champs client, line_items, total_usd. " Fonctionne 85 à 95% du temps sur les modèles frontaliers. Échoue de six façons: brace manquant, virgule arrière, types erronés, champs hallucinés, tronqués à la limite des jetons, prose fuite comme "Voici votre JSON:".

**Approach two: validate after generation.**Générer librement, analyser, valider contre le schéma, réessayer à défaut. fiable mais coûteux  vous payez pour chaque réessayer, et les bugs de tronçage coûtent un tour supplémentaire par événement.

**Approach three: constrained decoding.**Le fournisseur impose le schéma au moment de décoder. Les jetons invalides sont masqués dans la distribution d'échantillonnage. La sortie est garantie d'analyse et garantie de validation. L'échec s'effondre en un mode: refus (le modèle décide que l'entrée ne correspond pas au schéma).

Chaque fournisseur frontalier de 2026 envoie une forme d'approche 3.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}`plus `refusal`dans la réponse si le modèle décline.
- **Anthropic.**L' exécution du régime`tool_use`les entrées; `stop_reason: "refusal"`C'est pas une chose, mais `end_turn`sans appel d'outil est le signal.
- **Gemini.** `responseSchema`au niveau de la demande; en 2026, Gemini expédie des restrictions grammaticales au niveau des tokens pour les types sélectionnés.
- **Pydantic AI.** `output_type=InvoiceModel`émet une structure `RunResult`Typé à `InvoiceModel`- Je suis désolé .
- **Zod (TypeScript).**Parseur de temps d'exécution qui valide la sortie du fournisseur par rapport à un schéma Zod; partage avec OpenAI `beta.chat.completions.parse`- Je suis désolé .

Le fil commun: déclare le schéma une fois, l'exécuter de bout en bout.

## Le concept

### Schéma JSON 2020-12  la langue française

Chaque fournisseur accepte JSON Schema 2020-12. Les constructions que vous utilisez le plus:

- `type`: une des `object`- Je suis là .`array`- Je suis là .`string`- Je suis là .`number`- Je suis là .`integer`- Je suis là .`boolean`- Je suis là .`null`- Je suis désolé .
- `properties`: carte du nom de champ au sous-schéma.
- `required`: liste des noms de champs qui doivent apparaître.
- `enum`: ensemble fermé de valeurs autorisées.
- `minimum`- Je suis là .`maximum`(numéros), `minLength`- Je suis là .`maxLength`- Je suis là .`pattern`Je suis en train de vous dire:
- `items`: sous-schéma appliqué à tous les éléments de tableau.
- `additionalProperties`Le numéro de la liste:`false`interdit les champs supplémentaires (par défaut varie selon le mode).

Le mode strict OpenAI ajoute trois exigences: chaque propriété doit être répertoriée dans `required`- Je suis là .`additionalProperties: false`partout, et pas une seule solution `$ref`Si vous les cassez, l'API vous rend 400 au moment de la demande.

### Pydantic, le lien Python

Pydantic v2 génère des schèmes JSON à partir de modèles en forme de classe de données via `model_json_schema()`L' IA de Pydantic enveloppe ça pour que vous écriviez:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

et le cadre d'agent traduit le schéma en mode strict OpenAI, Anthropic `input_schema`, ou Gémeaux `responseSchema`La sortie du modèle est de retour en tant que type`Invoice`L'erreur de validation augmente`ValidationError`avec des voies d'erreur de typage.

### Zod, la liaison TypeScript

Zod (`z.object({customer: z.string(), ...})`Le SDK Node d'OpenAI expose `zodResponseFormat(Invoice)`qui se traduit par la charge utile du schéma JSON de l'API.

### Réjections

Le mode strict ne peut pas forcer le modèle à répondre. Si l'entrée ne peut pas correspondre au schéma ("le courriel était un poème, pas une facture"), le modèle émet un `refusal`Le code doit traiter cette situation comme un résultat de première classe, et non comme un échec. Le refus est également utile comme signal de sécurité: un modèle demandé à extraire un numéro de carte de crédit d'un courriel contenu protégé renvoie un refus avec la raison de sécurité jointe.

### Décodage restreint à l'ouverture

Les implémentations à poids ouvert utilisent trois techniques.

1. **Grammar-based decoding**(le secteur de l'énergie)`outlines`- Je suis là .`guidance`- Je suis là .`lm-format-enforcer`): construire un automate finit déterministe à partir du schéma; à chaque étape, masquer les logits des jetons qui violeraient le FSM.
2. **Logit masking with a JSON parser**: exécuter un parseur JSON en streaming en phase verrouillée avec le modèle; à chaque étape, calculer le jeu de jetons valid-next.
3. **Speculative decoding with a verifier**: modèle de projet bon marché propose des jetons, vérificateur impose le schéma.

Les fournisseurs commerciaux choisissent l'un de ces produits dans les coulisses.

### Les trois modes d'échec

1. **Parse error.**La sortie n'est pas valide JSON. Ne peut pas se produire en mode strict. peut toujours se produire sur les fournisseurs non stricts.
2. **Schema violation.**La sortie paralyse mais enfreint le schéma.
3. **Refusal.**Le modèle décline, il faut le traiter comme un résultat typé.

### Stratégie de réessayer

Lorsque vous êtes en mode strict (utilisation d'outils anthropiques, non strict OpenAI, plus vieux Gémeaux), le modèle de récupération est:

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

Un retrait est généralement suffisant. Trois retrait capture des flocons de modèle faible. Au-delà de trois est un signe d'un mauvais schéma: le modèle ne peut pas le satisfaire pour certaines entrées, et le prompt ou le schéma doit être corrigé.

### Soutien pour les modèles de petite taille

Le décoding restreint fonctionne sur de petits modèles. Un modèle ouvert de 3B avec un paramètre grammaticalement appliqué surpasse un modèle de 70B avec une promptation brute sur des tâches structurées.

```figure
constrained-decoding
```

## Utilisez-le

`code/main.py`envoie un validateur minimal JSON Schema 2020-12 en stdlib (types, requis, enum, min/max, motif, éléments, propriétés supplémentaires).`Invoice`Il exécute une fausse sortie LLM par le validateur, démontrant des erreurs de partage, des violations de schéma et des voies de refus.

À quoi regarder:

- Le validateur renvoie une tape `[ValidationError]`la liste avec chemin et message. C'est la forme que vous voulez apparaître à la requête de réessayer.
- La branche de refus ne tente PAS de nouveau. Elle enregistre et renvoie un refus typé.
- Le `additionalProperties: false`vérifier les feux sur l'entrée de test adverse, montrant pourquoi le mode strict ferme la porte sur les champs hallucinés.

## La faire partir

Cette leçon produit `outputs/skill-structured-output-designer.md`. Compte tenu d'une cible d'extraction de texte libre (factures, billets de support, CV, etc.), la compétence produit un schéma JSON 2020-12 qui est strictement compatible avec le mode et un modèle Pydantic qui le reflète, avec un refus de type et une nouvelle tentative de manipulation.

## Exercices

1. On court .`code/main.py`- Ajouter un quatrième cas d'essai dont`total_usd`Confirme que le validateur le rejette avec le`minimum`parcours de contrainte.

2. Élargir le validateur à l' appui `oneOf`Le cas commun: `line_item`est soit un produit ou un service, marqué par `kind`. Le mode strict a des règles subtiles ici; consultez le guide de sorties structurées d'OpenAI.

3. Écrire le même schéma de facture qu' un modèle de base Pydantic et comparer `model_json_schema()`Identifiez les ensembles Pydantic d'un champ par défaut que la version roulée manuellement omet.

4. Mesurer les taux de refus. Construire dix entrées qui ne devraient pas être extractibles (une chanson lyrique, une preuve mathématique, un e-mail vide) et les exécuter à travers un fournisseur réel avec mode strict. Comptez les refus par rapport aux sorties hallucinées. Ceci est votre vérité fondamentale pour les retries conscientes de refus.

5. Lisez le guide de sorties structurées d'OpenAI de haut en bas. Identifiez la construction qu'il interdit explicitement en mode strict que le schéma JSON simple permet.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "The schema spec" | IETF-draft schema dialect every modern provider speaks |
| Strict mode | "Guaranteed schema" | OpenAI flag that enforces schema via constrained decoding |
| Constrained decoding | "Logit masking" | Decode-time enforcement that masks invalid next-tokens |
| Refusal | "Model declines" | Typed outcome when input cannot fit the schema |
| Parse error | "Invalid JSON" | Output did not parse as JSON; impossible under strict |
| Schema violation | "Wrong shape" | Parsed but violated types / required / enum / range |
| `additionalProperties: false` | "No extras allowed" | Forbids unknown fields; required in OpenAI strict |
| Pydantic BaseModel | "Typed output" | Python class that emits and validates JSON Schema |
| Zod schema | "TypeScript output type" | TS runtime schema for provider output validation |
| Grammar enforcement | "Open-weights constrained decode" | FSM-based logit masking, as in outlines / guidance |

## Pour en savoir plus

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) Régimes stricts, refus et exigences de schéma
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) Le lancement en août 2024 expliquant la garantie de décoding
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) liens de type output_type typés qui se sérialisent à chaque fournisseur
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) la spécificité canonique
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) notes de déploiement des entreprises et avertissements de mode strict
