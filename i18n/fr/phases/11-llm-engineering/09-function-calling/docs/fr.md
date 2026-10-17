# Appel à fonction et utilisation des outils

> Les LLM ne peuvent rien faire. Ils génèrent du texte. C'est toute la capacité. Ils ne peuvent pas vérifier la météo, consulter une base de données, envoyer un e-mail, exécuter un code ou lire un fichier. Chaque "agent d'IA" que vous avez vu est un LLM générant JSON qui dit quelle fonction appeler -- et ensuite votre code l'appelle réellement. Le modèle est le cerveau. Les outils sont les mains. L'appel à la fonction est le système nerveux qui les relie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 Lesson 03 (Structured Outputs)
**Time:** ~75 minutes
**Related:**Phase 11 · 14 (Model Context Protocol)  lorsque l'outil est partagé entre les hôtes, passez de l'appel à fonction inline à un serveur MCP. Cette leçon couvre le cas inline; MCP couvre le cas protocole.

## Objectifs d'apprentissage

- Implémenter une boucle d'appel de fonction: définir les schémas d'outils, analyser le JSON d'appel d'outils du modèle, exécuter des fonctions et retourner les résultats
- Des schémas d'outils de conception avec des descriptions claires et des paramètres typés que le modèle peut invoquer de manière fiable
- Construire une boucle d'agent multi-tours qui enchaîne plusieurs appels de fonction pour répondre à des requêtes complexes
- Fonction de manipulation appelant les cas de bord: appels parallèles aux outils, propagation d'erreurs et prévention des boucles d'outils infinies

## Le problème

Vous construisez un chatbot. Un utilisateur demande: " Quelle est la météo à Tokyo en ce moment ? "

Le modèle répond: "Je n'ai pas accès à des données météorologiques en temps réel, mais en fonction de la saison, Tokyo est probablement autour de 15 degrés Celsius... "

C'est une hallucination déguisée en délinquant. Le modèle ne connaît pas la météo. Il ne le fera jamais. La météo change toutes les heures.

La bonne réponse consiste à appeler l'API OpenWeatherMap, à obtenir la température actuelle et à retourner le nombre réel. Le modèle ne peut pas appeler les API. Votre code peut. La pièce manquante: un protocole structuré qui permet au modèle de dire "Je dois appeler l'API météo avec ces arguments" et permet à votre code de l'exécuter et de rediriger le résultat.

Le modèle donne des sorties JSON structurées décrivant quelle fonction à invoquer avec quels arguments. Votre application exécute la fonction. Le résultat revient dans la conversation. Le modèle utilise le résultat pour produire sa réponse finale.

Sans appel à la fonction, les LLM sont des encyclopédies.

## Le concept

### La fonction qui appelle la boucle

Chaque interaction entre l'utilisation des outils suit la même boucle de 5 étapes.

```mermaid
sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

Étape 1: l'utilisateur envoie un message. Étape 2: le modèle reçoit le message avec les définitions de l'outil (schéma JSON décrivant les fonctions disponibles). Étape 3: au lieu de répondre avec du texte, le modèle sort un appel d'outil -- un objet JSON structuré avec le nom de la fonction et les arguments. Étape 4: votre code exécute la fonction et capture le résultat. Étape 5: le résultat revient au modèle, qui dispose maintenant de données réelles pour produire sa réponse finale.

Le modèle n'exécute jamais rien, il décide seulement de quoi appeler et avec quels arguments.

### Définitions d'outils: Le contrat de schéma JSON

Chaque outil est défini par un schéma JSON qui indique au modèle ce que fait la fonction, quels arguments il prend, et quels types de ces arguments doivent être.

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'Tokyo' or 'San Francisco'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}
```

Le `description`Les champs sont critiques. Le modèle les lit pour décider quand et comment utiliser l'outil. Une description vague comme " obtient la météo " produit une meilleure sélection d'outils que " Obtient la météo actuelle pour une ville. Retourne la température en Celsius et les conditions. " La description est une invitation à la sélection d'outils.

### Comparaison des fournisseurs

Chaque fournisseur majeur prend en charge l'appel des fonctions, mais la surface de l'API est différente.

| Provider | API Parameter | Tool Call Format | Parallel Calls | Forced Calling |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | Yes (multiple per turn) | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | Yes (multiple blocks) | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | Yes | `function_calling_config` |
| Open-weight (Llama 4, Qwen3, DeepSeek-V3) | Native `tools` on Llama 4; Hermes or ChatML on others | Mixed | Model-dependent | Prompt-based or `tool_choice` if supported |

D'ici 2026, les trois fournisseurs fermés se sont convergés sur des formats basés sur JSON-Schema presque identiques.`tools`Le format Hermes (NousResearch) est le plus courant pour les tweets fin de tiers. Pour les outils partagés entre hôtes, préférez MCP (Phase 11 · 14) à l'appel de fonction en ligne  le serveur est le même pour tous.

### Choix d'outil: automatique, nécessaire, spécifique

Vous contrôlez quand le modèle utilise des outils.

**Auto**(par défaut): le modèle décide d'appeler un outil ou de répondre directement. "Qu'est-ce que 2 + 2?" - répond directement. "Quel est le temps?" - appelle l'outil.

**Required**Le modèle doit appeler au moins un outil.Utilisez ceci lorsque vous savez que l'intention de l'utilisateur exige un outil.Évite le modèle de deviner au lieu de rechercher des données réelles.

**Specific function**: forcer le modèle à appeler une fonction particulière. `tool_choice={"type":"function", "function": {"name": "get_weather"}}`l'outil météo est appelé, quel que soit le requérant. Utilisez cela pour le routage - lorsque la logique en amont détermine déjà quel outil est nécessaire.

### Appel parallèle

GPT-4o et Claude peuvent appeler plusieurs fonctions en un seul tour. Un utilisateur demande: " Quelle est la météo à Tokyo et à New York ? " Le modèle sort deux appels d'outils simultanément:

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

Votre code exécute les deux (idéalement simultanément), renvoie les deux résultats, et le modèle synthétise une seule réponse. Cela réduit les allers-retours de 2 à 1. Pour les agents avec 5-10 appels d'outils par requête, les appels parallèles réduisent la latence de 60-80%.

### Les sorties structurées par rapport aux appels à fonction

Le cours 03 couvrait les sorties structurées.

**Structured outputs**Le produit final est le produit final. Exemple: extraire des informations sur le produit du texte comme`{name, price, in_stock}`- Je suis désolé .

**Function calling**Le modèle déclare une intention d'exécuter une action.`get_weather(city="Tokyo")`-- le modèle demande une action, ne produit pas la réponse finale.

Utilisez des sorties structurées lorsque vous voulez extraire des données. Utilisez des appels de fonction lorsque vous voulez que le modèle interagisse avec des systèmes externes.

### Sécurité: les règles non négociables

L'appel à la fonction est la capacité la plus dangereuse que vous puissiez donner à un LLM. Le modèle choisit ce qu'il doit exécuter. Si votre ensemble d'outils comprend des requêtes de base de données, le modèle construit les requêtes.

**Rule 1: Never pass model-generated SQL directly to a database.**Le modèle peut et générera des tableaux de dépôt, des injections d'union ou des requêtes qui retournent chaque rangée. Paramétriser toujours. Valider toujours. Utiliser toujours une liste d'opérations autorisées.

**Rule 2: Allowlist functions.**Le modèle ne peut appeler que des fonctions que vous définissez explicitement. Ne jamais créer un outil générique "exécuter une fonction par nom". Si vous avez 50 fonctions internes, exposer seulement les 5 dont l'utilisateur a besoin.

**Rule 3: Validate arguments.**Le modèle pourrait passer par le nom de la ville de `"; DROP TABLE users; --"`. Valider tous les arguments contre les types, les gammes et les formats attendus avant l'exécution.

**Rule 4: Sanitize tool results.**Si un outil renvoie des données sensibles (clés API, PII, erreurs internes), filtrez-les avant de les renvoyer au modèle.

**Rule 5: Rate limit tool calls.**Un modèle en boucle peut appeler des outils des centaines de fois.

### Traitement des erreurs

Les outils échouent, les API sont en panne, les bases de données sont en panne, les fichiers n'existent pas, le modèle doit savoir quand un outil échoue et pourquoi.

Retourner les erreurs comme résultat d'outil structuré, pas d'exception:

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

Le modèle lit ceci, ajuste ses arguments et réessaye. Les modèles sont bons pour se corriger à partir de messages d'erreur structurés. Ils sont mauvais pour récupérer des réponses vides ou des erreurs génériques "quelque chose est allé mal".

### MCP: modèle de protocole de contexte

MCP est l'étalon ouvert d'Anthropic pour l'interopérabilité des outils. Au lieu de chaque application définissant ses propres outils, MCP fournit un protocole universel: les outils sont servis par des serveurs MCP, consommés par des clients MCP (comme Claude Code, Cursor ou votre application).

Un serveur MCP peut exposer les outils à n'importe quel client compatible. Un serveur MCP Postgres donne accès à n'importe quelle base de données d'agents compatible avec MCP. Un serveur MCP GitHub donne accès au référentiel d'agents. Les outils sont définis une fois, utilisés partout.

MCP est de fonctionner appelant ce que HTTP est de réseautage. Il normalise la couche de transport de sorte que les outils deviennent portables.

```figure
mx-tool-call-loop
```

## Faites-le

### Étape 1: Définir le répertoire des outils

Construisez un registre qui stocke les définitions des outils et leurs implémentations. Chaque outil a une définition de schéma JSON (ce que le modèle voit) et une fonction Python (ce que votre code exécute).

```python
import ast
import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }
```

### Étape 2: mettre en œuvre 5 outils

Construisez une calculatrice, une recherche météo, un simulateur de recherche sur le Web, un lecteur de fichiers et un code runner.

```python
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": True, "message": f"SyntaxError: {e}", "code": "SYNTAX_ERROR"}
    unsafe_names = {"exec", "eval", "compile", "__import__", "open", "globals", "locals", "vars", "getattr", "setattr", "delattr"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return {"error": True, "message": "Forbidden operation: import is not allowed", "code": "SECURITY_VIOLATION"}
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            return {"error": True, "message": "Forbidden operation: dunder attribute access is not allowed", "code": "SECURITY_VIOLATION"}
        if isinstance(node, ast.Name) and node.id in unsafe_names:
            return {"error": True, "message": f"Forbidden operation: {node.id} is not allowed", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}
```

Une liste de bloc de sous-string lit le code en tant que texte, il manque donc tout ce que la correspondance de chaîne ne signifie pas littéralement.`import`Les données de l'équipe de recherche`__class__`et `__globals__`Les noms de l'interprète sont non seulement des noms de l'interprète, mais aussi des noms de l'interprète. Néanmoins, considérez cela comme un filtre d'enseignement, et non comme une limite réelle. Tout gardien en cours de processus partage l'interprète avec le code qu'il exécute, et un appelant déterminé peut toujours trouver des objets accessibles. Les systèmes de production exécutent du code non fiable dans un processus ou un conteneur séparé (un sous-processus avec des privilèges perdus, gVisor, Firecracker ou un code runner hébergé), où une évasion atterrit l'attaquant dans une boîte à jeter au lieu de votre service.

### Étape 3: Enregistrer tous les outils

```python
def register_all_tools():
    register_tool(
        "calculator", "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"}, "precision": {"type": "integer", "description": "Decimal places in result", "default": 2}}, "required": ["expression"]},
        calculator,
    )
    register_tool(
        "get_weather", "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"}}, "required": ["city"]},
        get_weather,
    )
    register_tool(
        "web_search", "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3}}, "required": ["query"]},
        web_search,
    )
    register_tool(
        "read_file", "Read the contents of a file. Returns the file content, size, and line count.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"}}, "required": ["path"]},
        read_file,
    )
    register_tool(
        "run_code", "Run a small Python snippet behind a static-analysis guard and a restricted interpreter. This is a teaching filter, not real isolation. Set a 'result' variable to return output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "language": {"type": "string", "enum": ["python"], "description": "Programming language"}}, "required": ["code"]},
        run_code,
    )
```

### Étape 4: Construisez la fonction appelant la boucle

C'est le moteur principal. Il simule le modèle, décide quel outil appeler, exécute l'outil et envoie les résultats.

```python
def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}
```

### Étape 5: Valider le raisonnement

Construisez un validateur qui vérifie les arguments d'appel d'outil contre le schéma JSON avant l'exécution.

```python
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors
```

### Étape 6: Exécuter la démo

```python
def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")
```

## Utilisez-le

### Appel à la fonction OpenAI

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)
```

OpenAI renvoie les appels à l' outil comme `response.choices[0].message.tool_calls`Chaque appel a un numéro .`id`le modèle utilise cette ID pour correspondre les résultats aux appels. GPT-4o peut retourner plusieurs appels d'outils en une seule réponse - les iterer et les exécuter tous.

### Utilisation d'outils anthropologiques

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
# )
#
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

L' outil Anthropic renvoie les appels en tant que blocs de contenu avec `type: "tool_use"`. Le résultat de l' outil est envoyé dans un message d' utilisateur avec `type: "tool_result"`. Notez la différence clé: utilisation anthropologique `input_schema`pour les définitions de paramètres des outils, tandis que OpenAI utilise `parameters`- Je suis désolé .

### Intégration des PCM

```python
# MCP servers expose tools over a standardized protocol.
# Any MCP-compatible client can discover and call these tools.
#
# Example: connecting to a Postgres MCP server
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})
```

MCP découple la mise en œuvre des outils de la consommation d'outils. Le serveur Postgres connaît SQL. Le serveur GitHub connaît l'API. Votre agent découvre et appelle simplement les outils - il n'a pas besoin de code spécifique au fournisseur pour chaque intégration.

## La faire partir

Cette leçon produit `outputs/prompt-tool-designer.md`-- un modèle de prompt réutilisable pour concevoir des définitions d'outils. Donnez-lui une description de ce que vous voulez qu'un outil fasse, et il produit la définition complète du schéma JSON avec des descriptions, des types et des contraintes.

Il produit aussi `outputs/skill-function-calling-patterns.md`-- un cadre de décision pour la mise en œuvre des fonctions appelant à la production, couvrant la conception des outils, la gestion des erreurs, la sécurité et les modèles spécifiques au fournisseur.

## Exercices

1. **Add a 6th tool: database query.**Implémenter un outil SQL simulé avec une table en mémoire. L'outil accepte un nom de table et des conditions de filtre (pas SQL brut). Valider que le nom de table est dans une liste d'allowl et que les opérateurs de filtre sont limités à `=`- Je suis là .`>`- Je suis là .`<`- Je suis là .`>=`- Je suis là .`<=`Retourner les lignes correspondantes en JSON.

2. **Implement retry with error feedback.**Lorsqu'un appel d'outil échoue (par exemple, la ville ne se trouve pas), renvoyez le message d'erreur à la fonction de décision du modèle et laissez-le corriger ses arguments. Suivez le nombre de répétitions effectuées par chaque appel.

3. **Build a multi-step agent.**Certaines requêtes nécessitent des appels d'outils de chaîne: "Lisez le fichier de configuration et dites-moi quel modèle est configuré, puis recherchez le prix de ce modèle sur le Web". Implémenter une boucle qui fonctionne jusqu'à ce que le modèle décide qu'il n'y a plus besoin d'outils, en passant les résultats accumulés dans chaque étape de décision. Limitez à 10 itérations pour éviter des boucles infinies.

4. **Measure tool selection accuracy.**Créer 30 requêtes de test avec les noms des outils attendus. Exécuter votre fonction de décision sur les 30 et mesurer le pourcentage de temps qu'il sélectionne l'outil correct. Identifier les requêtes qui causent le plus de confusion entre les outils.

5. **Implement tool call caching.**Si le même outil est appelé avec les mêmes arguments dans les 60 secondes, renvoyez le résultat caché au lieu de le réexécuter.`(tool_name, frozenset(args.items()))`Mesurer les taux de clics dans une conversation avec 20 requêtes.

## Les termes clés

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Function calling | "Tool use" | The model outputs structured JSON describing a function to invoke with specific arguments -- your code executes it, not the model |
| Tool definition | "Function schema" | A JSON Schema object describing a tool's name, purpose, parameters, and types -- the model reads this to decide when and how to use the tool |
| Tool choice | "Calling mode" | Controls whether the model must call a tool (required), may call a tool (auto), or must call a specific tool (named) |
| Parallel calling | "Multi-tool" | The model outputs multiple tool calls in a single turn, reducing round trips -- GPT-4o and Claude both support this |
| Tool result | "Function output" | The return value from executing a tool, sent back to the model as a message so it can use real data in its response |
| Argument validation | "Input checking" | Verifying that model-generated arguments match the expected types, ranges, and constraints before executing the tool |
| MCP | "Tool protocol" | Model Context Protocol -- Anthropic's open standard for exposing tools via servers that any compatible client can discover and call |
| Agent loop | "ReAct loop" | The iterative cycle of model-decides-tool, code-executes-tool, result-feeds-back until the model has enough information to respond |
| Tool poisoning | "Prompt injection via tools" | An attack where tool results contain instructions that manipulate the model's behavior -- sanitize all tool outputs |
| Rate limiting | "Call budget" | Setting a maximum number of tool calls per conversation to prevent infinite loops and runaway API costs |

## Pour en savoir plus

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)-- la référence définitive pour l'utilisation des outils avec GPT-4o, y compris les appels parallèles, les appels forcés et les arguments structurés
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)-- L'outil de Claude utilise la mise en œuvre avec input_schema, réponses multi-outils et configuration de tool_choice
- [Model Context Protocol Specification](https://modelcontextprotocol.io)-- la norme ouverte pour l'interopérabilité des outils entre les applications d'IA, avec une architecture serveur/client
- [Schick et al., 2023 -- "Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761)-- le document fondamental sur la formation des LLM pour décider quand et comment appeler des outils externes
- [Patil et al., 2023 -- "Gorilla: Large Language Model Connected with Massive APIs"](https://arxiv.org/abs/2305.15334)-- réglage des LLM pour des appels API précis sur 1 645 API avec réduction des hallucinations
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)-- référence en temps réel comparant la fonction appelant la précision sur GPT-4o, Claude, Gémeaux, et les modèles ouverts
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629)-- la boucle de pensée-action-observation qui est la boucle d'agent externe autour de chaque appel d'outil; où cette leçon se termine, la phase 14 reprend.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents)-- cinq modèles composables (chaîne de mise en route, routage, parallélisation, orchestrateur-travailleur, évaluateur-optimisateur) construits à partir de l'outil-utilisation unique primitive.
