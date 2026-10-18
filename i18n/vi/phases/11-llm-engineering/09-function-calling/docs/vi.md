# Chọi chức năng & Sử dụng công cụ

> LLM không thể làm gì cả. Chúng tạo ra tin nhắn. Đó là toàn bộ khả năng. Họ không thể kiểm tra thời tiết, truy vấn cơ sở dữ liệu, gửi email, chạy mã hoặc đọc tập tin. Mỗi "hội nhân tạo AI" mà bạn đã từng thấy là một LLM tạo ra JSON cho biết hàm nào để gọi -- và sau đó mã của bạn thực sự gọi nó. Mô hình là não. Công cụ là tay. Vị trí gọi là hệ thần kinh kết nối chúng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 Lesson 03 (Structured Outputs)
**Time:** ~75 minutes
**Related:**Giai đoạn 11 · 14 (Mô hình Context Protocol)  khi một công cụ được chia sẻ giữa các máy chủ, tốt nghiệp từ việc gọi hàm inline đến máy chủ MCP. Bài học này bao gồm trường hợp inline; MCP bao gồm trường hợp giao thức.

## Mục tiêu học tập

- Thực hiện một vòng gọi hàm: xác định các sơ đồ công cụ, phân tích JSON gọi công cụ của mô hình, thực thi các hàm và trả lại kết quả
- Các quy trình thiết kế công cụ với mô tả rõ ràng và các tham số được đánh dấu mà mô hình có thể sử dụng một cách đáng tin cậy
- Xây dựng một vòng lặp đại lý nhiều vòng nối liên kết nhiều hàm gọi để trả lời các truy vấn phức tạp
- Chức năng xử lý gọi các trường hợp cạnh: gọi công cụ song song, lây lan lỗi và ngăn chặn vòng lặp công cụ vô hạn

## Vấn đề

Bạn xây dựng một chatbot. Người dùng hỏi: "Thời tiết ở Tokyo là gì bây giờ?"

Mô hình trả lời: "Tôi không có quyền truy cập vào dữ liệu thời tiết theo thời gian thực, nhưng dựa trên mùa, Tokyo có thể là khoảng 15 độ C... "

Đó là một ảo giác mặc một bản báo cáo không chịu trách nhiệm. mô hình không biết thời tiết. nó sẽ không bao giờ. thời tiết thay đổi mỗi giờ. dữ liệu đào tạo của mô hình là vài tháng tuổi.

Câu trả lời chính xác đòi hỏi phải gọi API OpenWeatherMap, lấy nhiệt độ hiện tại và trả lại số thực. Mô hình không thể gọi API. Mã của bạn có thể. Phần thiếu sót: một giao thức có cấu trúc cho phép mô hình nói "Tôi cần gọi API thời tiết với các lập luận này" và cho phép mã của bạn thực hiện nó và đưa kết quả trở lại.

Đây là gọi hàm. mô hình sẽ đưa ra JSON cấu trúc mô tả hàm nào để gọi với các lập luận nào. Ứng dụng của bạn thực hiện hàm. Kết quả sẽ quay lại cuộc trò chuyện. mô hình sử dụng kết quả để tạo ra câu trả lời cuối cùng của nó.

Không có chức năng gọi, LLM là các ensiklopedia.

## Khái niệm

### Phương pháp gọi vòng lặp

Mỗi tương tác sử dụng công cụ theo cùng vòng lặp 5 bước.

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

Bước 1: người dùng gửi tin nhắn. Bước 2: mô hình nhận được thông điệp cùng với các định nghĩa công cụ (Tình đồ JSON mô tả các chức năng có sẵn). Bước 3: thay vì trả lời bằng văn bản, mô hình sẽ đưa ra một cuộc gọi công cụ -- một đối tượng JSON có cấu trúc với tên và lập luận của hàm. Bước 4: mã của bạn thực hiện chức năng và ghi lại kết quả. Bước 5: kết quả trở lại mô hình, mà bây giờ có dữ liệu thực để tạo ra câu trả lời cuối cùng của nó.

Mô hình không bao giờ thực hiện bất cứ điều gì, nó chỉ quyết định cái gì để gọi và bằng những lập luận nào.

### Các định nghĩa công cụ: Hợp đồng JSON Schema

Mỗi công cụ được định nghĩa bởi một JSON Schema cho mô hình biết chức năng làm gì, nó cần những lập luận nào và những loại lập luận đó phải là gì.

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

- `description`Các trường hợp quan trọng. mô hình đọc chúng để quyết định khi nào và làm thế nào để sử dụng công cụ. Một mô tả mơ hồ như "được thời tiết" tạo ra sự lựa chọn công cụ tồi tệ hơn "Được thời tiết hiện tại cho một thành phố.

### So sánh nhà cung cấp

Mỗi nhà cung cấp chính hỗ trợ gọi chức năng, nhưng bề mặt API khác nhau.

| Provider | API Parameter | Tool Call Format | Parallel Calls | Forced Calling |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | Yes (multiple per turn) | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | Yes (multiple blocks) | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | Yes | `function_calling_config` |
| Open-weight (Llama 4, Qwen3, DeepSeek-V3) | Native `tools` on Llama 4; Hermes or ChatML on others | Mixed | Model-dependent | Prompt-based or `tool_choice` if supported |

Đến năm 2026, ba nhà cung cấp đóng cửa đã hội tụ vào các định dạng dựa trên JSON-Schema gần giống hệt nhau.`tools`trường phù hợp với hình dạng của OpenAI. Open-weight fine-tunes vẫn khác nhau  định dạng Hermes (NousResearch) là phổ biến nhất cho các phần mềm fine-tune của bên thứ ba. Đối với các công cụ chia sẻ trên các máy chủ, hãy ưu tiên MCP (Phase 11 · 14) hơn việc gọi hàm trực tuyến  máy chủ là giống nhau cho tất cả chúng.

### Chọn công cụ: tự động, yêu cầu, cụ thể

Bạn kiểm soát khi mô hình sử dụng công cụ.

**Auto**(đặc định): mô hình quyết định gọi một công cụ hay trả lời trực tiếp. "Điều gì là 2 + 2?" -- trả lời trực tiếp. "Điều gì là thời tiết?" -- gọi công cụ.

**Required**: mô hình phải gọi ít nhất một công cụ. Sử dụng nó khi bạn biết ý định của người dùng đòi hỏi một công cụ. ngăn chặn mô hình đoán thay vì tìm kiếm dữ liệu thực.

**Specific function**: buộc mô hình gọi một hàm cụ thể. `tool_choice={"type":"function", "function": {"name": "get_weather"}}`sử dụng điều này để định tuyến -- khi logic dòng lên đã xác định công cụ cần thiết.

### Hướng gọi hàm song song

GPT-4o và Claude có thể gọi nhiều chức năng trong một lượt. Một người dùng hỏi: "Thời tiết ở Tokyo và New York là gì?" mô hình phát ra hai cuộc gọi công cụ cùng một lúc:

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

Mã của bạn thực hiện cả hai (hiện lý cùng lúc), trả lại cả hai kết quả, và mô hình tổng hợp một phản ứng duy nhất. Điều này cắt giảm đi lại từ 2 đến 1. Đối với các đại lý với 5-10 cuộc gọi công cụ mỗi truy vấn, cuộc gọi song song giảm độ trễ 60-80%.

### Các kết quả được cấu trúc so với việc gọi chức năng

Bài học 03 bao gồm các kết quả cấu trúc. gọi hàm sử dụng cùng một máy tính JSON Schema, nhưng với mục đích khác.

**Structured outputs**: buộc mô hình để tạo ra dữ liệu trong một hình dạng cụ thể.`{name, price, in_stock}`- Tôi không biết.

**Function calling**: mô hình tuyên bố ý định thực hiện một hành động.`get_weather(city="Tokyo")`-- mô hình đang yêu cầu hành động, không tạo ra câu trả lời cuối cùng.

Sử dụng các đầu ra được cấu trúc khi bạn muốn khai thác dữ liệu. Sử dụng gọi hàm khi bạn muốn mô hình tương tác với các hệ thống bên ngoài.

### An ninh: Quy tắc không thể thương lượng

Đơn vị gọi hàm là khả năng nguy hiểm nhất mà bạn có thể cung cấp cho một LLM. Mô hình chọn điều gì để thực hiện. Nếu bộ công cụ của bạn bao gồm các truy vấn cơ sở dữ liệu, mô hình xây dựng các truy vấn. Nếu nó bao gồm các lệnh shell, mô hình viết chúng.

**Rule 1: Never pass model-generated SQL directly to a database.**Mô hình có thể và sẽ tạo ra bảng DROP, tiêm UNION hoặc truy vấn trả lại mỗi hàng. Luôn định nghĩa. Luôn xác nhận. Luôn sử dụng danh sách các hoạt động.

**Rule 2: Allowlist functions.**Mô hình chỉ có thể gọi các chức năng bạn xác định rõ ràng. Không bao giờ xây dựng công cụ chung "hãy thực hiện bất kỳ chức năng nào theo tên". Nếu bạn có 50 chức năng nội bộ, chỉ phơi bày 5 người dùng cần.

**Rule 3: Validate arguments.**Mô hình có thể vượt qua tên thành phố của `"; DROP TABLE users; --"`. Thiết lập tất cả các lập luận chống lại các loại, phạm vi và định dạng dự kiến trước khi thực hiện.

**Rule 4: Sanitize tool results.**Nếu một công cụ trả về dữ liệu nhạy cảm (phím API, PII, lỗi nội bộ), hãy lọc nó trước khi gửi nó trở lại mô hình.

**Rule 5: Rate limit tool calls.**Một mô hình trong vòng lặp có thể gọi các công cụ hàng trăm lần. Đặt tối đa (10-20 cuộc gọi cho mỗi cuộc trò chuyện là hợp lý).

### Việc xử lý lỗi

Các công cụ thất bại, API bị lỗi, cơ sở dữ liệu bị lỗi, các tập tin không tồn tại, mô hình cần biết khi nào một công cụ thất bại và tại sao.

Trả lỗi như kết quả công cụ có cấu trúc, không phải ngoại lệ:

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

Mô hình đọc điều này, điều chỉnh lập luận của nó, và thử lại. Mô hình giỏi tự sửa chữa từ các tin nhắn lỗi cấu trúc. Họ không giỏi phục hồi từ các câu trả lời trống hoặc lỗi chung "một cái gì đó đã sai".

### MCP: Mô hình giao thức ngữ cảnh

MCP là tiêu chuẩn mở của Anthropic cho khả năng tương tác công cụ. Thay vì mỗi ứng dụng xác định các công cụ của riêng mình, MCP cung cấp một giao thức phổ quát: các công cụ được phục vụ bởi các máy chủ MCP, được tiêu thụ bởi các khách hàng MCP (như Claude Code, Cursor hoặc ứng dụng của bạn).

Một máy chủ MCP có thể phơi bày các công cụ cho bất kỳ khách hàng tương thích nào. Một máy chủ MCP Postgres cung cấp quyền truy cập vào cơ sở dữ liệu đại lý tương thích với MCP. Một máy chủ MCP GitHub cung cấp quyền truy cập vào kho lưu trữ đại lý nào. Các công cụ được xác định một lần, được sử dụng ở mọi nơi.

MCP là để gọi chức năng gọi HTTP là mạng lưới. Nó tiêu chuẩn hóa lớp vận chuyển để các công cụ trở nên di động.

```figure
mx-tool-call-loop
```

## Hãy xây dựng nó

### Bước 1: Định nghĩa danh sách công cụ

Xây dựng một sổ đăng ký lưu trữ các định nghĩa công cụ và các thực hiện của chúng. Mỗi công cụ có định nghĩa JSON Schema (những gì mô hình nhìn thấy) và chức năng Python (những gì mã của bạn thực hiện).

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

### Bước 2: Thực hiện 5 công cụ

Xây dựng máy tính, tìm thời tiết, mô phỏng tìm kiếm trên web, đọc tệp và chạy mã.

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

Một danh sách block substring đọc mã như văn bản, vì vậy nó bỏ lỡ bất cứ điều gì mà chuỗi không chính xác đánh vần. Phân tích mã vào một cây tổng hợp và đi bộ nó cho phép người bảo vệ từ chối `import`các tuyên bố, truy cập thuộc tính dunder (the `__class__`và `__globals__`(được gọi là "đường dây" và "đường dây" có thể quay trở lại với người diễn giải thực sự), và tên không an toàn được xây dựng theo cấu trúc thay vì chính tả. Dù vậy, hãy coi điều này như là một bộ lọc giảng dạy, chứ không phải là một ranh giới thực sự. Bất kỳ người bảo vệ nào trong quá trình chia sẻ phiên dịch với mã nó chạy, và một người gọi xác định vẫn có thể tìm thấy các đối tượng có thể tiếp cận. Hệ thống sản xuất chạy mã không đáng tin cậy trong một quy trình hoặc thùng riêng (một quy trình phụ với quyền bị bỏ, gVisor, Firecracker, hoặc một bộ chạy mã được lưu trữ), nơi một kẻ thoát khỏi hạ cánh kẻ tấn công vào hộp ném bỏ thay vì dịch vụ của bạn.

### Bước 3: Đăng tất cả các công cụ

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

### Bước 4: Xây dựng chức năng gọi vòng

Đây là động cơ cốt lõi. Nó mô phỏng mô hình quyết định công cụ nào để gọi, thực thi công cụ, và cung cấp kết quả lại.

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

### Bước 5: Định lý luận

Xây dựng một trình xác thực kiểm tra các lập luận gọi công cụ với Schema JSON trước khi thực hiện.

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

### Bước 6: chạy Demo

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

## Sử dụng nó

### OpenAI Calling Function

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

OpenAI trả lại các cuộc gọi công cụ như `response.choices[0].message.tool_calls`Mỗi cuộc gọi đều có một số`id`bạn phải bao gồm khi trả lại kết quả. mô hình sử dụng ID này để phù hợp kết quả với cuộc gọi. GPT-4o có thể trả lại nhiều tool cuộc gọi trong một phản ứng duy nhất - lặp lại và thực hiện tất cả chúng.

### Sử dụng công cụ nhân loại

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

Anthropic trả lại các cuộc gọi công cụ như các khối nội dung với `type: "tool_use"`Kết quả công cụ đi vào một tin nhắn của người dùng với `type: "tool_result"`Lưu ý sự khác biệt chính: sử dụng nhân văn `input_schema`cho các định nghĩa tham số công cụ, trong khi OpenAI sử dụng `parameters`- Tôi không biết.

### Kết hợp MCP

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

MCP tách ra việc thực hiện công cụ từ việc tiêu thụ công cụ. Server Postgres biết SQL. Server GitHub biết API. Trình của bạn chỉ phát hiện và gọi công cụ - nó không cần mã cụ thể cho mỗi sự tích hợp.

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/prompt-tool-designer.md`-- một mẫu yêu cầu được sử dụng nhiều lần để thiết kế định nghĩa công cụ. Cho nó một mô tả về những gì bạn muốn một công cụ làm, và nó tạo ra định nghĩa JSON Schema đầy đủ với mô tả, loại và hạn chế.

Nó cũng sản xuất `outputs/skill-function-calling-patterns.md`-- một khung quyết định để thực hiện các chức năng gọi trong sản xuất, bao gồm thiết kế công cụ, xử lý lỗi, an ninh và các mô hình cụ thể cho nhà cung cấp.

## Các bài tập

1. **Add a 6th tool: database query.**Thực hiện một công cụ SQL mô phỏng với một bảng trong bộ nhớ. Công cụ này chấp nhận tên bảng và các điều kiện lọc (không phải là SQL thô). Định hành rằng tên bảng nằm trong danh sách quyền và các nhà điều hành lọc bị hạn chế`=`- `>`- `<`- `>=`- `<=`. Trả lại các dòng phù hợp như JSON.

2. **Implement retry with error feedback.**Khi một cuộc gọi công cụ thất bại (ví dụ, thành phố không được tìm thấy), đưa thông điệp lỗi trở lại chức năng quyết định mô hình và để nó sửa chữa lập luận của nó. Theo dõi bao nhiêu lần lặp lại mỗi cuộc gọi. Đặt tối đa 3 lần lặp lại mỗi cuộc gọi công cụ.

3. **Build a multi-step agent.**Một số truy vấn yêu cầu gọi công cụ chuỗi: "Đọc tập tin cấu hình và cho tôi biết mô hình nào được cấu hình, sau đó tìm kiếm trên web về giá của mô hình đó". Thực hiện một vòng lặp chạy cho đến khi mô hình quyết định không cần thêm các công cụ, chuyển kết quả tích lũy vào mỗi bước quyết định. Giới hạn đến 10 lần lặp để ngăn chặn vòng lặp vô hạn.

4. **Measure tool selection accuracy.**Tạo 30 truy vấn thử nghiệm với tên công cụ dự kiến. Động hành chức năng quyết định của bạn trên tất cả 30 và đo tỷ lệ phần trăm thời gian họ chọn công cụ đúng. Xác định các truy vấn gây nhầm lẫn nhiều nhất giữa các công cụ.

5. **Implement tool call caching.**Nếu cùng một công cụ được gọi với các lập luận giống nhau trong vòng 60 giây, trả lại kết quả được lưu trữ trong cache thay vì thực hiện lại. Sử dụng một từ điển được khóa bởi `(tool_name, frozenset(args.items()))`- Đánh giá tỷ lệ truy cập cache trong cuộc trò chuyện với 20 truy vấn.

## Các điều khoản chính

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

## Đọc thêm

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)-- giới thiệu cuối cùng về việc sử dụng công cụ với GPT-4o, bao gồm các cuộc gọi song song, cuộc gọi buộc và các lập luận có cấu trúc
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)-- Claude sử dụng công cụ thực hiện với input_schema, nhiều công cụ phản hồi, và tool_choice cấu hình
- [Model Context Protocol Specification](https://modelcontextprotocol.io)-- tiêu chuẩn mở cho khả năng tương tác giữa các ứng dụng AI, với kiến trúc máy chủ/client
- [Schick et al., 2023 -- "Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761)-- bài báo cơ bản về đào tạo LLM để quyết định khi nào và làm thế nào để gọi các công cụ bên ngoài
- [Patil et al., 2023 -- "Gorilla: Large Language Model Connected with Massive APIs"](https://arxiv.org/abs/2305.15334)-- điều chỉnh LLM cho các cuộc gọi API chính xác trên 1.645 API với giảm ảo giác
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)-- điểm chuẩn thời gian thực so sánh hàm gọi chính xác trên GPT-4o, Claude, Gemini, và các mô hình mở
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629)-- vòng lặp Thought-Action-Observation là vòng lặp bên ngoài của các nhân viên xung quanh mỗi cuộc gọi công cụ; nơi bài học kết thúc, giai đoạn 14 bắt đầu.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents)-- năm mô hình hợp tác (sự chuỗi nhanh, định tuyến, song song, nhạc công-người làm việc, đánh giá-người tối ưu hóa) được xây dựng từ nguyên thủy sử dụng công cụ đơn.
