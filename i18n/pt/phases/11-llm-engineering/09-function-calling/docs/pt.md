# Função chamada e uso de ferramentas

> Os LLM não podem fazer nada. Eles geram texto. É toda a capacidade. Não conseguem verificar o tempo, consultar uma base de dados, enviar um e-mail, executar um código ou ler um arquivo. Cada "agente de IA" que já viste é um LLM que gera JSON que diz a função a ligar e o seu código chama-a. O modelo é o cérebro. As ferramentas são as mãos. A chamada de função é o sistema nervoso que as liga.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 Lesson 03 (Structured Outputs)
**Time:** ~75 minutes
**Related:**Fase 11 · 14 (Modelo Context Protocol)  quando uma ferramenta é compartilhada entre hosts, se graduar da chamada de função inline para um servidor MCP. Esta lição abrange o caso inline; MCP abrange o caso de protocolo.

## Objetivos de aprendizagem

- Implementar um loop de chamada de função: definir esquemas de ferramentas, analisar o JSON de chamada de ferramentas do modelo, executar funções e retornar resultados
- Esquemas de ferramentas de projeto com descrições claras e parâmetros tipografados que o modelo possa invocar de forma confiável
- Construir um loop de agente multi-turn que encadeia várias chamadas de função para responder a consultas complexas
- Função de manuseio chamando casos de borda: chamadas paralelas de ferramentas, propagação de erros e prevenção de loops de ferramentas infinitas

## O problema

Você constrói um chatbot. Um usuário pergunta: "Como está o tempo em Tóquio agora?"

O modelo responde: "Não tenho acesso a dados meteorológicos em tempo real, mas com base na estação, Tóquio provavelmente está em torno de 15 graus Celsius"...

É uma alucinação vestida de um aviso de responsabilidade. O modelo não sabe o tempo. Nunca o vai fazer. O tempo muda a cada hora. Os dados de treinamento do modelo são de meses.

A resposta correta requer ligar para a API OpenWeatherMap, obter a temperatura atual e retornar o número real. O modelo não pode chamar para API. Seu código pode. A peça que falta: um protocolo estruturado que permite que o modelo diga "Eu preciso chamar para a API do tempo com esses argumentos" e permite que seu código execute e reproduzir o resultado.

Este é o chamado de função. O modelo produz JSON estruturado descrevendo qual função invocar com quais argumentos. Sua aplicação executa a função. O resultado volta à conversa. O modelo usa o resultado para produzir sua resposta final.

Sem o chamado de função, os LLM são enciclopédias.

## O conceito

### A função que chama o ciclo

Cada interação entre ferramentas e uso segue o mesmo ciclo de 5 passos.

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

Passo 1: o utilizador envia uma mensagem. Passo 2: o modelo recebe a mensagem juntamente com as definições da ferramenta (Esquema JSON descrevendo as funções disponíveis). Passo 3: Em vez de responder com texto, o modelo expande uma chamada de ferramenta - um objeto JSON estruturado com o nome da função e os argumentos. Passo 4: o seu código executa a função e capta o resultado. Passo 5: o resultado retorna ao modelo, que agora tem dados reais para produzir a sua resposta final.

O modelo nunca executa nada, só decide o que chamar e com que argumentos.

### Definições de ferramentas: O contrato de esquema JSON

Cada ferramenta é definida por um esquema JSON que diz ao modelo o que a função faz, quais argumentos requer e quais tipos esses argumentos devem ser.

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

O `description`O modelo lê-os para decidir quando e como usar a ferramenta. Uma descrição vaga como "obtenha tempo" produz uma seleção de ferramentas pior do que "Obter o tempo atual para uma cidade. Retorna a temperatura em Celsius e condições".

### Comparação entre os fornecedores

Todos os principais provedores suportam chamadas de função, mas a superfície da API é diferente.

| Provider | API Parameter | Tool Call Format | Parallel Calls | Forced Calling |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | Yes (multiple per turn) | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | Yes (multiple blocks) | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | Yes | `function_calling_config` |
| Open-weight (Llama 4, Qwen3, DeepSeek-V3) | Native `tools` on Llama 4; Hermes or ChatML on others | Mixed | Model-dependent | Prompt-based or `tool_choice` if supported |

Em 2026, os três provedores fechados convergem em formatos baseados em JSON-Schema quase idênticos.`tools`campo que corresponde à forma do OpenAI. As tonalidades de peso aberto ainda variam  o formato Hermes (NousResearch) é o mais comum para tonalidades de terceiros. Para ferramentas compartilhadas entre hosts, prefira MCP (Fase 11 · 14) em vez de chamadas de funções inline  o servidor é o mesmo para todos eles.

### Opção de ferramentas: automática, necessária, específica

Você controla quando o modelo usa ferramentas.

**Auto**(default): o modelo decide se chamar uma ferramenta ou responder diretamente. "O que é 2 + 2?" - responde diretamente. "O que é o tempo?" - chama a ferramenta.

**Required**O modelo deve chamar pelo menos uma ferramenta. Use esta quando você sabe que a intenção do usuário requer uma ferramenta. Impede o modelo de adivinhar em vez de procurar dados reais.

**Specific function**: forçar o modelo a chamar uma função específica. `tool_choice={"type":"function", "function": {"name": "get_weather"}}`O que é que é necessário para fazer uma rotação?

### Chamadas para funções paralelas

O GPT-4o e o Claude podem chamar várias funções em uma única vez. Um usuário pergunta: "Qual é o tempo em Tóquio e Nova York?" O modelo emitirá duas chamadas de ferramenta simultaneamente:

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

O seu código executa ambos (idealmente simultaneamente), retorna ambos os resultados e o modelo sintetiza uma única resposta. Isso reduz as viagens de ida e volta de 2 para 1. Para agentes com 5-10 chamadas de ferramentas por consulta, chamadas paralelas reduzem a latência em 60-80%.

### Outputes estruturados vs. Função chamada

A lição 03 abrangeu as saídas estruturadas.

**Structured outputs**O resultado é o produto final. Exemplo: extrair informações do produto do texto como `{name, price, in_stock}`- Não .

**Function calling**O modelo declara a intenção de executar uma ação.`get_weather(city="Tokyo")`- o modelo está a pedir uma ação, não a produzir a resposta final.

Use saídas estruturadas quando quiser extração de dados. Use chamadas de função quando quiser que o modelo interaja com sistemas externos.

### Segurança: Regras não negociaveis

Função chamada é a capacidade mais perigosa que você pode dar a um LLM. O modelo escolhe o que executar. Se o seu conjunto de ferramentas inclui consultas de banco de dados, o modelo constrói as consultas. Se inclui comandos shell, o modelo as escreve.

**Rule 1: Never pass model-generated SQL directly to a database.**O modelo pode e gerará DROP TABLE, injeções UNION ou consultas que retornam cada linha. Sempre parametrize. sempre valida. sempre use uma lista de operações.

**Rule 2: Allowlist functions.**O modelo só pode chamar funções que você define explicitamente. Nunca construa uma ferramenta genérica "executar qualquer função por nome". Se você tem 50 funções internas, expor apenas as 5 necessárias para o usuário.

**Rule 3: Validate arguments.**O modelo pode passar por um nome de cidade de`"; DROP TABLE users; --"`. Validar todos os argumentos contra os tipos, intervalos e formatos esperados antes da execução.

**Rule 4: Sanitize tool results.**Se uma ferramenta retorna dados sensíveis (chaves API, PII, erros internos), filtrá-los antes de enviá-los de volta ao modelo.

**Rule 5: Rate limit tool calls.**Um modelo em um loop pode chamar ferramentas centenas de vezes. Defina um máximo (10-20 chamadas por conversa é razoável).

### Manutenção de erros

As ferramentas falham, as APIs ficam sem tempo, os bancos de dados caem, os arquivos não existem, o modelo precisa saber quando uma ferramenta falha e porquê.

Retorno de erros como resultados de ferramentas estruturadas, não exceções:

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

O modelo lê isso, ajusta seus argumentos e retrata. Os modelos são bons em auto-corrigir mensagens de erro estruturadas.

### MCP: Modelo de protocolo de contexto

MCP é o padrão aberto da Anthropic para a interoperabilidade de ferramentas. Em vez de cada aplicação definir suas próprias ferramentas, MCP fornece um protocolo universal: as ferramentas são servidas por servidores MCP, consumidas por clientes MCP (como Claude Code, Cursor ou seu aplicativo).

Um servidor MCP pode expor ferramentas a qualquer cliente compatível. Um servidor Postgres MCP dá acesso a qualquer banco de dados de agentes compatíveis com MCP. Um servidor GitHub MCP dá acesso ao repositório de qualquer agente. As ferramentas são definidas uma vez, usadas em todos os lugares.

O MCP é para ligar o que HTTP é para rede.

```figure
mx-tool-call-loop
```

## Construí-lo

### Passo 1: Definir o Registro de Ferramentas

Construir um registo que armazene as definições de ferramentas e suas implementações. Cada ferramenta tem uma definição de JSON Schema (o que o modelo vê) e uma função Python (o que seu código executa).

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

### Passo 2: Implementar 5 Ferramentas

Construa uma calculadora, pesquisa do tempo, simulador de pesquisa na web, leitor de arquivos e executor de código.

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

Uma lista de blocos de substrings lê o código como texto, então ele perde qualquer coisa que a correspondência de strings não ortografia literalmente.`import`As informações sobre o acesso ao atributo Dunder (a `__class__`E ...`__globals__`As redes de dados são utilizadas para identificar os nomes de pessoas que estão em situação de risco. Mesmo assim, trate isso como um filtro de ensino, não como um verdadeiro limite. Qualquer guarda em processo compartilha o intérprete com o código que executa, e um chamador determinado ainda pode encontrar objetos acessíveis. Os sistemas de produção executam código não confiável em um processo ou contêiner separado (um subprocesso com privilégios retirados, gVisor, Firecracker ou um executador de código hospedado), onde uma fuga aterra o atacante em uma caixa de descarte em vez do seu serviço.

### Passo 3: Registre todas as ferramentas

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

### Passo 4: Construa a função chamada Loop

Este é o motor central. Simula o modelo, decide qual ferramenta chamar, executa a ferramenta e reencaminha os resultados.

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

### Passo 5: Validação do argumento

Construir um validador que verifica argumentos de chamada de ferramenta contra o JSON Schema antes da execução.

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

### Passo 6: Execute a demonstração

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

## Usá-lo

### Chamadas de função OpenAI

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

O OpenAI retorna as chamadas de ferramenta como `response.choices[0].message.tool_calls`Cada chamada tem um número .`id`O modelo usa este ID para combinar os resultados com chamadas. GPT-4o pode retornar várias chamadas de ferramenta em uma única resposta - iterar e executar todas elas.

### Utilização de Ferramentas Antropicas

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

Antropic retorna as chamadas de ferramenta como blocos de conteúdo com `type: "tool_use"`. O resultado da ferramenta vai em uma mensagem do usuário com `type: "tool_result"`Observe a diferença fundamental: usos antropológicos `input_schema`para definições de parâmetros de ferramentas, enquanto o OpenAI usa `parameters`- Não .

### Integração dos MCP

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

O MCP descopla a implementação de ferramentas do consumo de ferramentas. O servidor Postgres conhece SQL. O servidor GitHub conhece a API. Seu agente apenas descobre e chama as ferramentas - não precisa de código específico do provedor para cada integração.

## Envia-o

Esta lição produz`outputs/prompt-tool-designer.md`-- um modelo de prompt reutilizável para projetar definições de ferramentas. Dê-lhe uma descrição do que você quer que uma ferramenta faça, e ele produz a definição completa do JSON Schema com descrições, tipos e restrições.

Também produz `outputs/skill-function-calling-patterns.md`- um quadro de decisão para a implementação de funções de chamada na produção, abrangendo o design de ferramentas, o tratamento de erros, a segurança e os padrões específicos do fornecedor.

## Exercícios

1. **Add a 6th tool: database query.**Implementar uma ferramenta SQL simulada com uma tabela em memória. A ferramenta aceita um nome de tabela e condições de filtro (não SQL bruto). Validar que o nome da tabela está em uma lista de permisos e que os operadores de filtro são restritos a `=`- Não .`>`- Não .`<`- Não .`>=`- Não .`<=`Retorna as linhas correspondentes como JSON.

2. **Implement retry with error feedback.**Quando uma chamada de ferramenta falhar (por exemplo, cidade não encontrada), envie a mensagem de erro de volta para a função de decisão do modelo e deixe que ele corrija seus argumentos.

3. **Build a multi-step agent.**Algumas consultas exigem chamadas de ferramentas de cadeia: "Leia o arquivo de configuração e diga-me qual o modelo é configurado, e depois procure na web para o preço desse modelo". Implemente um ciclo que corre até que o modelo decida que não são necessários mais ferramentas, passando os resultados acumulados em cada etapa de decisão. Limite a 10 iterações para evitar loops infinitos.

4. **Measure tool selection accuracy.**Crie 30 consultas de teste com nomes de ferramentas esperados. Execute a função de decisão em todas as 30 e mensure qual porcentagem do tempo ela seleciona a ferramenta correta. Identifique quais consultas causam mais confusão entre as ferramentas.

5. **Implement tool call caching.**Se a mesma ferramenta for chamada com argumentos idênticos dentro de 60 segundos, retorne o resultado em cache em vez de re-executar. Use um dicionário com tecla de `(tool_name, frozenset(args.items()))`Meter as taxas de cache em uma conversa com 20 consultas.

## Termos-chave

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

## Mais leitura

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)-- a referência definitiva para o uso de ferramentas com o GPT-4o, incluindo chamadas paralelas, chamadas forçadas e argumentos estruturados
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)-- A ferramenta de Claude usa implementação com input_schema, respostas multi-tool e configuração de tool_choice
- [Model Context Protocol Specification](https://modelcontextprotocol.io)-- o padrão aberto para a interoperabilidade das ferramentas em aplicações de IA, com arquitetura servidor/cliente
- [Schick et al., 2023 -- "Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761)- o documento de base sobre a formação dos MLL para decidir quando e como chamar instrumentos externos
- [Patil et al., 2023 -- "Gorilla: Large Language Model Connected with Massive APIs"](https://arxiv.org/abs/2305.15334)-- sintonização de LLM para chamadas precisas de API em 1.645 API com redução de alucinações
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)-- benchmark em tempo real comparando função chamando precisão em GPT-4o, Claude, Gemini e modelos abertos
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629)-- o ciclo Pensamento-Ação-Observação que é o ciclo de agente externo em torno de cada chamada de ferramenta; onde esta lição termina, a Fase 14 começa.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents)-- cinco padrões compostos (cadeia de urgência, roteamento, paralelação, orquestrador-trabalhador, avaliador-optimizador) construídos a partir da primitiva de uso de ferramentas únicas.
