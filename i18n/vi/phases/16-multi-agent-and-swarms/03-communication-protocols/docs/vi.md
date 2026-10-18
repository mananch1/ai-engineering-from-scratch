# Các giao thức thông tin

> Các đại lý không thể nói cùng một ngôn ngữ không phải là một đội ngũ, họ là những người lạ hét vào không gian.

**Type:** Build
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering), Lesson 16.01 (Why Multi-Agent)
**Time:** ~120 minutes

## Mục tiêu học tập

- Thực hiện công cụ phát hiện và gọi MCP để các đại lý có thể sử dụng các công cụ được tiếp xúc bởi các máy chủ bên ngoài
- Xây dựng thẻ đại lý A2A và điểm cuối nhiệm vụ cho phép một đại lý phân phối công việc cho người khác qua HTTP
- So sánh MCP (tương tự truy cập công cụ), A2A (trong đại lý), ACP (chuyển toán doanh nghiệp) và ANP (trust phân cấp) và giải thích giao thức nào giải quyết vấn đề nào
- Cụm nhiều giao thức cùng nhau trong một hệ thống duy nhất, nơi các đại lý phát hiện công cụ thông qua MCP và ủy thác các nhiệm vụ thông qua A2A

## Vấn đề

Bạn chia hệ thống của mình thành nhiều đại lý, một nhà nghiên cứu, một lập trình viên, một nhà phê bình, họ rất giỏi trong công việc cá nhân của họ, nhưng bây giờ bạn cần họ thực sự nói chuyện với nhau.

Cố gắng đầu tiên của bạn là hiển nhiên: chuyển chuỗi xung quanh. Nhà nghiên cứu trả lại một khối văn bản, người lập trình phân tích nó bất cứ cách nào có thể. Nó hoạt động cho đến khi người lập trình hiểu sai một bản tóm tắt nghiên cứu, hoặc hai đại lý đang chờ nhau, hoặc bạn cần các đại lý được xây dựng bởi các nhóm khác nhau để hợp tác. đột nhiên "chỉ chuyển chuỗi" bị vỡ.

Đây là vấn đề giao thức giao tiếp. Nếu không có hợp đồng chia sẻ về cách mà các đại lý trao đổi thông tin, hệ thống đa đại lý sẽ rất yếu, không thể kiểm tra, và không thể mở rộng vượt ra ngoài một số đại lý mà bạn đã viết.

Hệ sinh thái AI đã phản ứng với bốn giao thức, mỗi giao thức giải quyết một phần khác nhau của vấn đề:

- **MCP**cho việc truy cập công cụ
- **A2A**cho hợp tác giữa các đại lý
- **ACP**đối với khả năng kiểm toán của doanh nghiệp
- **ANP**cho danh tính và niềm tin phi tập trung

Bài học này đi sâu hơn. Bạn sẽ đọc các định dạng dây thực sự từ mỗi thông số, xây dựng các thực hiện hoạt động, và kết nối tất cả bốn thành một hệ thống thống nhất.

## Khái niệm

### Tầm nhìn của giao thức

Hãy nghĩ về bốn giao thức này như là các lớp, mỗi lớp giải quyết một câu hỏi khác nhau:

```mermaid
flowchart TD
  ANP["ANP — How do agents trust strangers?<br/>Decentralized identity (DID), E2EE, meta-protocol"]
  A2A["A2A — How do agents collaborate on goals?<br/>Agent Cards, task lifecycle, streaming, negotiation"]
  ACP["ACP — How do agents talk in auditable systems?<br/>Runs, trajectory metadata, session continuity"]
  MCP["MCP — How does an agent use a tool?<br/>Tool discovery, execution, context sharing"]

  style ANP fill:#f3e8ff,stroke:#7c3aed
  style A2A fill:#dbeafe,stroke:#2563eb
  style ACP fill:#fef3c7,stroke:#d97706
  style MCP fill:#d1fae5,stroke:#059669
```

Họ không phải là đối thủ cạnh tranh, họ giải quyết các vấn đề khác nhau ở các cấp độ khác nhau.

### MCP (Tái khấu)

MCP được bao gồm sâu trong giai đoạn 13. Nghịch lại nhanh chóng: MCP chuẩn hóa cách LLM kết nối với các công cụ bên ngoài và các nguồn dữ liệu.**client-server**giao thức mà người đại lý (thành khách) phát hiện và gọi các công cụ được máy chủ phơi bày.

```mermaid
sequenceDiagram
    participant Agent as Agent (client)
    participant MCP1 as MCP Server<br/>(database, API, files)

    Agent->>MCP1: list tools
    MCP1-->>Agent: tool definitions
    Agent->>MCP1: call tool X
    MCP1-->>Agent: result
```

MCP là **agent-to-tool**liên lạc, không giúp các nhân viên nói chuyện với nhau.

### A2A (Bản ứng viên2Bản ứng viên)

**Created by:**Google (nay dưới Linux Foundation như `lf.a2a.v1`(văn)
**Spec version:**1.0.1
**Problem:**Các đại lý tự trị làm việc với nhau, đàm phán và giao nhiệm vụ với nhau như thế nào?

A2A là giao thức cho **peer-to-peer agent collaboration**. Khi MCP kết nối một đại lý với các công cụ, A2A kết nối một đại lý với các đại lý khác.**Agent Card**tại một URL nổi tiếng, và các đại lý khác phát hiện, đàm phán với và ủy thác nhiệm vụ cho nó.

#### Làm thế nào A2A hoạt động

```mermaid
sequenceDiagram
    participant Client as Client Agent
    participant Remote as Remote Agent

    Client->>Remote: GET /.well-known/agent-card.json
    Remote-->>Client: Agent Card (skills, modes, security)

    Client->>Remote: POST /message:send (returnImmediately)
    Remote-->>Client: Task (TASK_STATE_SUBMITTED or TASK_STATE_WORKING)

    alt Polling
        Client->>Remote: GET /tasks/{id}
        Remote-->>Client: Task status + artifacts
    else Streaming
        Client->>Remote: POST /message:stream
        Remote-->>Client: SSE: statusUpdate
        Remote-->>Client: SSE: artifactUpdate
        Remote-->>Client: SSE: statusUpdate TASK_STATE_COMPLETED, stream closes
    end
```

#### Thẻ đại lý thực sự

Đây là hình ảnh của thẻ đại lý A2A trong tự nhiên.`GET /.well-known/agent-card.json`- Có thể là:

```json
{
  "name": "Research Agent",
  "description": "Searches documentation and summarizes findings",
  "version": "1.0.0",
  "supportedInterfaces": [
    {
      "url": "https://research-agent.example.com/a2a/v1",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    },
    {
      "url": "https://research-agent.example.com/a2a/rest",
      "protocolBinding": "HTTP+JSON",
      "protocolVersion": "1.0"
    }
  ],
  "provider": {
    "organization": "Your Company",
    "url": "https://example.com"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["text/plain", "application/json"],
  "skills": [
    {
      "id": "web-research",
      "name": "Web Research",
      "description": "Searches the web and synthesizes findings",
      "tags": ["research", "search", "summarization"],
      "examples": ["Research the latest changes in React 19"]
    },
    {
      "id": "doc-analysis",
      "name": "Documentation Analysis",
      "description": "Reads and analyzes technical documentation",
      "tags": ["docs", "analysis"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["application/json"]
    }
  ],
  "securitySchemes": {
    "bearer": {
      "httpAuthSecurityScheme": {
        "scheme": "Bearer",
        "bearerFormat": "JWT"
      }
    }
  },
  "securityRequirements": [{ "schemes": { "bearer": { "list": [] } } }]
}
```

Những điều quan trọng cần chú ý:
- **Skills**là những gì một đại lý có thể làm. Mỗi người có một ID, thẻ, và hỗ trợ nhập / ra MIME loại. Đây là cách mà một đại lý khách hàng quyết định liệu đại lý từ xa này có thể xử lý yêu cầu của mình.
- **supportedInterfaces**danh sách liên kết giao thức nhiều. Một đại lý duy nhất có thể nói JSON-RPC, REST và gRPC cùng một lúc.
- **Security**được tích hợp vào thẻ: `securitySchemes`tên của mỗi chương trình và `securityRequirements`Khách hàng biết mình cần gì trước khi đưa ra một yêu cầu.

#### Chuyển đời nhiệm vụ

Các nhiệm vụ là đơn vị cốt lõi của công việc trong A2A. Chúng di chuyển qua các trạng thái được xác định (phần biểu đồ giảm `TASK_STATE_`tiền tố mà mỗi trạng thái mang trên dây):

```mermaid
stateDiagram-v2
    [*] --> SUBMITTED
    SUBMITTED --> WORKING
    WORKING --> INPUT_REQUIRED: needs more info
    INPUT_REQUIRED --> WORKING: client sends data
    WORKING --> COMPLETED: success
    WORKING --> FAILED: error
    WORKING --> CANCELED: client cancels
    SUBMITTED --> REJECTED: agent declines

    COMPLETED --> [*]
    FAILED --> [*]
    CANCELED --> [*]
    REJECTED --> [*]

    note right of COMPLETED
        Terminal states are immutable.
        Follow-ups create new tasks
        within the same contextId.
    end note
```

Tất cả 8 trạng thái (đặc biệt cũng xác định `UNSPECIFIED`như một lính canh, được bỏ qua ở đây):

| State | Terminal? | Meaning |
|---|---|---|
| `TASK_STATE_SUBMITTED` | No | Acknowledged, not yet processing |
| `TASK_STATE_WORKING` | No | Actively being processed |
| `TASK_STATE_INPUT_REQUIRED` | No | Agent needs more info from client |
| `TASK_STATE_AUTH_REQUIRED` | No | Authentication needed |
| `TASK_STATE_COMPLETED` | Yes | Finished successfully |
| `TASK_STATE_FAILED` | Yes | Finished with error |
| `TASK_STATE_CANCELED` | Yes | Canceled before completion |
| `TASK_STATE_REJECTED` | Yes | Agent declined the task |

Khi một nhiệm vụ đạt đến trạng thái cuối, nó không thay đổi. Không tin nhắn nào nữa.`contextId`- Tôi không biết.

#### Phương thức dây

A2A sử dụng JSON-RPC 2.0. Đây là cách trao đổi tin nhắn thực sự trông như thế nào:

**Client sends a message:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{ "text": "Research React 19 compiler features" }]
    },
    "configuration": {
      "acceptedOutputModes": ["text/plain", "application/json"],
      "historyLength": 10
    }
  }
}
```

**Agent responds with a task:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "task": {
      "id": "task-abc-123",
      "contextId": "ctx-xyz-789",
      "status": {
        "state": "TASK_STATE_COMPLETED",
        "timestamp": "2026-03-27T10:30:00Z"
      },
      "artifacts": [
        {
          "artifactId": "art-001",
          "name": "research-results",
          "parts": [{
            "data": {
              "findings": [
                "React 19 compiler auto-memoizes components",
                "No more manual useMemo/useCallback needed",
                "Compiler runs at build time, not runtime"
              ]
            },
            "mediaType": "application/json"
          }]
        }
      ]
    }
  }
}
```

**Streaming via SSE:**
```text
POST /message:stream HTTP/1.1
Content-Type: application/a2a+json
A2A-Version: 1.0

data: {"task":{"id":"task-123","contextId":"ctx-123","status":{"state":"TASK_STATE_WORKING"}}}

data: {"statusUpdate":{"taskId":"task-123","contextId":"ctx-123","status":{"state":"TASK_STATE_WORKING","message":{"messageId":"msg-002","role":"ROLE_AGENT","parts":[{"text":"Searching documentation..."}]}}}}

data: {"artifactUpdate":{"taskId":"task-123","contextId":"ctx-123","artifact":{"artifactId":"art-1","parts":[{"text":"partial findings..."}]},"append":true,"lastChunk":false}}

data: {"statusUpdate":{"taskId":"task-123","contextId":"ctx-123","status":{"state":"TASK_STATE_COMPLETED"}}}
```

### ACP (Phương thức giao tiếp của cơ quan)

**Created by:**IBM / BeeAI
**Spec version:**0.2.0 (OpenAPI 3.1.1)
**Status:**Tham gia vào A2A dưới Linux Foundation
**Problem:**Làm thế nào để các đại lý giao tiếp với khả năng kiểm tra đầy đủ, tính tiếp tục phiên và theo dõi quỹ đạo?

ACP là **enterprise protocol**Không giống như nhiều bản tóm tắt tuyên bố, ACP làm **not**sử dụng JSON-LD. Đó là một API REST / JSON đơn giản được định nghĩa thông qua OpenAPI. Điều làm cho nó đặc biệt là **TrajectoryMetadata**: mỗi phản ứng của đại lý có thể mang theo một nhật ký chi tiết về các bước lý luận và các cuộc gọi công cụ đã tạo ra nó.

```mermaid
sequenceDiagram
    participant Client
    participant ACP as ACP Agent
    participant Audit as Audit Log

    Client->>ACP: POST /runs (mode: sync)
    ACP->>ACP: Process request...
    ACP->>Audit: Log trajectory:<br/>reasoning + tool calls
    ACP-->>Client: Response + TrajectoryMetadata
    Note over Audit: Every step recorded:<br/>tool_name, tool_input,<br/>tool_output, reasoning
```

#### Trưởng lý Discovery ở ACP

ACP xác định bốn phương pháp phát hiện:

```mermaid
graph LR
    A[Agent Discovery] --> B["Runtime<br/>GET /agents"]
    A --> C["Open<br/>.well-known/agent.yml"]
    A --> D["Registry<br/>Centralized catalog"]
    A --> E["Embedded<br/>Container labels"]

    style B fill:#dbeafe,stroke:#2563eb
    style C fill:#d1fae5,stroke:#059669
    style D fill:#fef3c7,stroke:#d97706
    style E fill:#f3e8ff,stroke:#7c3aed
```

- **AgentManifest**đơn giản hơn thẻ đại lý của A2A:

```json
{
  "name": "summarizer",
  "description": "Summarizes documents with source citations",
  "input_content_types": ["text/plain", "application/pdf"],
  "output_content_types": ["text/plain", "application/json"],
  "metadata": {
    "tags": ["summarization", "RAG"],
    "framework": "BeeAI",
    "capabilities": [
      {
        "name": "Document Summarization",
        "description": "Condenses long documents into key points"
      }
    ],
    "recommended_models": ["llama3.3:70b-instruct-fp16"],
    "license": "Apache-2.0",
    "programming_language": "Python"
  }
}
```

#### Tiếp tục chu kỳ đời

ACP sử dụng "Runs" thay vì "Tasks". A Run là một hành động thực hiện bằng đại lý với ba chế độ:

| Mode | Behavior |
|---|---|
| `sync` | Blocking. Response contains the complete result. |
| `async` | Returns 202 immediately. Poll `GET /runs/{id}` for status. |
| `stream` | SSE stream. Events fire as the agent works. |

```mermaid
stateDiagram-v2
    [*] --> created
    created --> in_progress
    in_progress --> completed: success
    in_progress --> failed: error
    in_progress --> awaiting: needs input
    awaiting --> in_progress: client resumes
    in_progress --> cancelling: cancel request
    cancelling --> cancelled

    completed --> [*]
    failed --> [*]
    cancelled --> [*]
```

#### Chế độ Metadata (The Audit Trail)

Đây là điểm phân biệt chính của ACP. Mỗi phần thông điệp có thể bao gồm các metadata cho thấy chính xác những gì đại lý đã làm:

```json
{
  "role": "agent/researcher",
  "parts": [
    {
      "content_type": "text/plain",
      "content": "The weather in San Francisco is 72F and sunny.",
      "metadata": {
        "kind": "trajectory",
        "message": "I need to check the weather for this location",
        "tool_name": "weather_api",
        "tool_input": { "location": "San Francisco, CA" },
        "tool_output": { "temperature": 72, "condition": "sunny" }
      }
    }
  ]
}
```

Đối với các ngành công nghiệp được quy định, đây là vàng. Mỗi câu trả lời đều đi kèm với một chuỗi lý luận có thể chứng minh được: những công cụ nào được gọi, những đầu vào nào được sử dụng, những sản phẩm nào được nhận. Không có hộp đen.

ACP cũng ủng hộ **CitationMetadata**cho việc gán nguồn:

```json
{
  "kind": "citation",
  "start_index": 0,
  "end_index": 47,
  "url": "https://weather.gov/sf",
  "title": "NWS San Francisco Forecast"
}
```

### ANP (Nền tảng mạng đại lý)

**Created by:**Cộng đồng nguồn mở (được thành lập bởi GaoWei Chang)
**Repo:** [github.com/agent-network-protocol/AgentNetworkProtocol](https://github.com/agent-network-protocol/AgentNetworkProtocol)
**Problem:**Làm sao các đại lý từ các tổ chức khác nhau tin tưởng lẫn nhau mà không có một cơ quan trung ương?

ANP là **decentralized identity protocol**Nó xây dựng niềm tin bằng cách sử dụng W3C Decentralized Identifiers (DID) và mã hóa đầu đến cuối. Không giống như A2A, nơi bạn phát hiện các đại lý thông qua các điểm cuối được biết đến, ANP cho phép các đại lý chứng minh danh tính của họ bằng cách mã hóa.

ANP có ba lớp:

```mermaid
graph TB
    subgraph Layer3["Layer 3: Application Protocol"]
        AD[Agent Description Documents]
        DISC[Discovery endpoints]
    end
    subgraph Layer2["Layer 2: Meta-Protocol"]
        NEG[AI-powered protocol negotiation]
        CODE[Dynamic code generation]
    end
    subgraph Layer1["Layer 1: Identity & Secure Communication"]
        DID["did:wba (W3C DID)"]
        HPKE[HPKE E2EE - RFC 9180]
        SIG[Signature verification]
    end

    Layer3 --> Layer2
    Layer2 --> Layer1

    style Layer1 fill:#d1fae5,stroke:#059669
    style Layer2 fill:#dbeafe,stroke:#2563eb
    style Layer3 fill:#f3e8ff,stroke:#7c3aed
```

#### Tài liệu DID (trụ cấu thực tế)

ANP sử dụng một phương pháp DID tùy chỉnh được gọi là `did:wba`(Nhà đại lý dựa trên mạng).`did:wba:example.com:user:alice`quyết định `https://example.com/user/alice/did.json`- Có thể là:

```json
{
  "@context": [
    "https://www.w3.org/ns/did/v1",
    "https://w3id.org/security/suites/jws-2020/v1",
    "https://w3id.org/security/suites/secp256k1-2019/v1"
  ],
  "id": "did:wba:example.com:user:alice",
  "verificationMethod": [
    {
      "id": "did:wba:example.com:user:alice#key-1",
      "type": "EcdsaSecp256k1VerificationKey2019",
      "controller": "did:wba:example.com:user:alice",
      "publicKeyJwk": {
        "crv": "secp256k1",
        "x": "NtngWpJUr-rlNNbs0u-Aa8e16OwSJu6UiFf0Rdo1oJ4",
        "y": "qN1jKupJlFsPFc1UkWinqljv4YE0mq_Ickwnjgasvmo",
        "kty": "EC"
      }
    },
    {
      "id": "did:wba:example.com:user:alice#key-x25519-1",
      "type": "X25519KeyAgreementKey2019",
      "controller": "did:wba:example.com:user:alice",
      "publicKeyMultibase": "z9hFgmPVfmBZwRvFEyniQDBkz9LmV7gDEqytWyGZLmDXE"
    }
  ],
  "authentication": [
    "did:wba:example.com:user:alice#key-1"
  ],
  "keyAgreement": [
    "did:wba:example.com:user:alice#key-x25519-1"
  ],
  "humanAuthorization": [
    "did:wba:example.com:user:alice#key-1"
  ],
  "service": [
    {
      "id": "did:wba:example.com:user:alice#agent-description",
      "type": "AgentDescription",
      "serviceEndpoint": "https://example.com/agents/alice/ad.json"
    }
  ]
}
```

Những điều quan trọng cần chú ý:
- **Key separation**Các khóa ký (secp256k1) được tách biệt với các khóa mã hóa (X25519).
- **`humanAuthorization`**Các khóa này đòi hỏi sự chấp thuận của con người rõ ràng (biometric, mật khẩu, HSM) trước khi sử dụng.
- **`keyAgreement`**Các khóa được sử dụng cho mã hóa HPKE kết thúc đến kết thúc (RFC 9180).
- - **service**Phần liên kết đến tài liệu mô tả đại lý.

#### Làm thế nào sự tin tưởng hoạt động trong ANP

ANP làm **not**sử dụng một biểu đồ web của niềm tin hoặc xác nhận.

```mermaid
sequenceDiagram
    participant A as Agent A
    participant Domain as Agent A's Domain
    participant B as Agent B

    A->>B: HTTP request + DID + signature
    B->>Domain: Fetch DID document (HTTPS)
    Domain-->>B: DID document + public key
    B->>B: Verify signature with public key
    B-->>A: Issue access token
    A->>B: Subsequent requests use token
    Note over A,B: Trust = TLS domain verification<br/>+ DID signature verification<br/>+ Principle of least trust
```

Sự tin tưởng đến từ ba nguồn:
1. **Domain-level TLS**xác minh host tài liệu DID
2. **DID cryptographic signatures**xác minh danh tính của đại lý
3. **Principle of least trust**chỉ cấp phép tối thiểu

Không có sự truyền bá tin tưởng dựa trên những tin đồn hay xếp hạng PageRank.

#### Các cuộc đàm phán về giao thức meta

Đây là tính năng mới nhất của ANP. Khi hai đại lý từ các hệ sinh thái khác nhau gặp nhau, họ không cần định dạng dữ liệu được thỏa thuận trước.

```json
{
  "action": "protocolNegotiation",
  "sequenceId": 0,
  "candidateProtocols": "I can communicate using:\n1. JSON-RPC with hotel booking schema\n2. REST with OpenAPI 3.1 spec\n3. Natural language over HTTP",
  "modificationSummary": "Initial proposal",
  "status": "negotiating"
}
```

```mermaid
sequenceDiagram
    participant A as Agent A
    participant B as Agent B

    A->>B: protocolNegotiation (candidateProtocols)
    B->>A: protocolNegotiation (counter-proposal)
    A->>B: protocolNegotiation (accepted)
    Note over A,B: Agents dynamically generate code<br/>to handle the agreed format.<br/>Max 10 rounds, then timeout.
```

Các đại lý đi lại và trở lại (tối đa 10 lần bắn) cho đến khi họ đồng ý về một định dạng, sau đó tạo ra mã động để xử lý nó.`negotiating`- `rejected`- `accepted`- `timeout`- Tôi không biết.

Điều này có nghĩa là hai đại lý chưa từng gặp nhau trước đây có thể tìm ra cách giao tiếp mà không ai định nghĩa trước một kế hoạch chia sẻ.

### So sánh (được sửa đổi)

| | MCP | A2A | ACP | ANP |
|---|---|---|---|---|
| **Created by** | Anthropic | Google / Linux Foundation | IBM / BeeAI | Community |
| **Spec format** | JSON-RPC | JSON-RPC / REST / gRPC | OpenAPI 3.1 (REST) | JSON-RPC |
| **Primary use** | Agent to Tool | Agent to Agent | Agent to Agent | Agent to Agent |
| **Discovery** | Tool listing | `/.well-known/agent-card.json` | `GET /agents`, `/.well-known/agent.yml` | `/.well-known/agent-descriptions`, DID service endpoints |
| **Identity** | Implicit (local) | Security schemes (OAuth, mTLS) | Server-level | W3C DID (`did:wba`) with E2EE |
| **Audit trail** | N/A | Basic (task history) | TrajectoryMetadata (tool calls, reasoning) | Not formally specified |
| **State machine** | N/A | 9 task states | 7 run states | N/A |
| **Streaming** | N/A | SSE | SSE | Transport-agnostic |
| **Unique feature** | Tool schemas | Agent Cards + Skills | Trajectory audit trail | Meta-protocol negotiation |
| **Best for** | Tools & data | Dynamic collaboration | Regulated industries | Cross-org trust |
| **Status** | Stable | Stable (v1.0) | Merging into A2A | Active development |

### Làm thế nào họ hợp tác

Các giao thức này không loại trừ lẫn nhau.

```mermaid
graph TB
    subgraph org["Your Organization"]
        RA[Research Agent] <-->|A2A| CA[Coding Agent]
        RA -->|MCP| SS[Search Server]
        CA -->|MCP| GS[GitHub Server]
        AUDIT["All agent responses carry<br/>ACP TrajectoryMetadata"]
    end

    subgraph ext["External (DID verified via ANP)"]
        EA[External Agent]
        PA[Partner Agent]
    end

    RA <-->|ANP + A2A| EA
    CA <-->|ANP + A2A| PA

    style org fill:#f8fafc,stroke:#334155
    style ext fill:#fef2f2,stroke:#991b1b
    style AUDIT fill:#fef3c7,stroke:#d97706
```

- **MCP**kết nối mỗi đại lý với công cụ của nó
- **A2A**xử lý sự hợp tác giữa các đại lý (bản bộ và bên ngoài)
- **ACP**kết hợp các phản ứng trong metadata quỹ đạo để kiểm toán
- **ANP**cung cấp xác minh danh tính cho các đại lý mà bạn không kiểm soát

```figure
swarm-message-bus
```

## Hãy xây dựng nó

### Bước 1: Các loại thông điệp chính

Mỗi hệ thống đa đại lý bắt đầu với một định dạng tin nhắn. Chúng tôi xác định các loại mà bản đồ cho những gì các giao thức thực sự sử dụng:

```typescript
import crypto from "node:crypto";

type MessageRole = "ROLE_USER" | "ROLE_AGENT";

type MessagePart =
  | { text: string }
  | { data: unknown; mediaType: string }
  | { url: string; filename: string; mediaType: string };

type TrajectoryEntry = {
  reasoning: string;
  toolName?: string;
  toolInput?: unknown;
  toolOutput?: unknown;
  timestamp: number;
};

type AgentMessage = {
  id: string;
  role: MessageRole;
  parts: MessagePart[];
  trajectory?: TrajectoryEntry[];
  replyTo?: string;
  timestamp: number;
};

function createMessage(
  role: MessageRole,
  parts: MessagePart[],
  replyTo?: string
): AgentMessage {
  return {
    id: crypto.randomUUID(),
    role,
    parts,
    replyTo,
    timestamp: Date.now(),
  };
}

function textMessage(role: MessageRole, text: string): AgentMessage {
  return createMessage(role, [{ text }]);
}
```

Lưu ý: `MessagePart`là đa phương thức (tinh văn, dữ liệu có cấu trúc, tệp) giống như các thông số kỹ thuật A2A và ACP thực tế.`text`- `data`, hoặc`url`) nói phần là gì; không có `kind`Đồ đếm.`TrajectoryEntry`ghi lại chuỗi lý luận, phù hợp với TrajectoryMetadata của ACP.

### Bước 2: Thẻ đại lý A2A và đăng ký

Xây dựng phát hiện đại lý phù hợp với thông số kỹ thuật A2A thực sự:

```typescript
type Skill = {
  id: string;
  name: string;
  description: string;
  tags: string[];
  inputModes: string[];
  outputModes: string[];
};

type AgentInterface = {
  url: string;
  protocolBinding: string;
  protocolVersion: string;
};

type AgentCard = {
  name: string;
  description: string;
  version: string;
  supportedInterfaces: AgentInterface[];
  capabilities: {
    streaming: boolean;
    pushNotifications: boolean;
  };
  defaultInputModes: string[];
  defaultOutputModes: string[];
  skills: Skill[];
};

class AgentRegistry {
  private cards: Map<string, AgentCard> = new Map();

  register(card: AgentCard) {
    this.cards.set(card.name, card);
  }

  discoverBySkillTag(tag: string): AgentCard[] {
    return [...this.cards.values()].filter((card) =>
      card.skills.some((skill) => skill.tags.includes(tag))
    );
  }

  discoverByInputMode(mimeType: string): AgentCard[] {
    return [...this.cards.values()].filter(
      (card) =>
        card.defaultInputModes.includes(mimeType) ||
        card.skills.some((skill) => skill.inputModes.includes(mimeType))
    );
  }

  resolve(name: string): AgentCard | undefined {
    return this.cards.get(name);
  }

  listAll(): AgentCard[] {
    return [...this.cards.values()];
  }
}
```

Đây là một bản đồ có tính năng từ tên đến khả năng đơn giản hơn. Bạn có thể khám phá các đại lý bằng thẻ kỹ năng, bằng các loại MIME nhập, hoặc bằng tên, giống như các đặc điểm A2A thực sự hỗ trợ.

### Bước 3: Chuyển đổi A2A

Xây dựng máy trạng thái nhiệm vụ đầy đủ:

```typescript
type TaskState =
  | "TASK_STATE_SUBMITTED"
  | "TASK_STATE_WORKING"
  | "TASK_STATE_INPUT_REQUIRED"
  | "TASK_STATE_AUTH_REQUIRED"
  | "TASK_STATE_COMPLETED"
  | "TASK_STATE_FAILED"
  | "TASK_STATE_CANCELED"
  | "TASK_STATE_REJECTED";

const TERMINAL_STATES: TaskState[] = [
  "TASK_STATE_COMPLETED",
  "TASK_STATE_FAILED",
  "TASK_STATE_CANCELED",
  "TASK_STATE_REJECTED",
];

type TaskStatus = {
  state: TaskState;
  message?: AgentMessage;
  timestamp: number;
};

type Artifact = {
  id: string;
  name: string;
  parts: MessagePart[];
};

type Task = {
  id: string;
  contextId: string;
  status: TaskStatus;
  artifacts: Artifact[];
  history: AgentMessage[];
};

type TaskEvent =
  | { statusUpdate: { taskId: string; status: TaskStatus } }
  | {
      artifactUpdate: {
        taskId: string;
        artifact: Artifact;
        append: boolean;
        lastChunk: boolean;
      };
    };

type TaskHandler = (
  task: Task,
  message: AgentMessage
) => AsyncGenerator<TaskEvent>;

class TaskManager {
  private tasks: Map<string, Task> = new Map();
  private handlers: Map<string, TaskHandler> = new Map();
  private listeners: Map<string, ((event: TaskEvent) => void)[]> = new Map();

  registerHandler(agentName: string, handler: TaskHandler) {
    this.handlers.set(agentName, handler);
  }

  subscribe(taskId: string, listener: (event: TaskEvent) => void) {
    const existing = this.listeners.get(taskId) ?? [];
    existing.push(listener);
    this.listeners.set(taskId, existing);
  }

  async sendMessage(
    agentName: string,
    message: AgentMessage,
    contextId?: string
  ): Promise<Task> {
    const handler = this.handlers.get(agentName);
    if (!handler) {
      const task = this.createTask(contextId);
      task.status = {
        state: "TASK_STATE_REJECTED",
        timestamp: Date.now(),
        message: textMessage("ROLE_AGENT", `No handler for ${agentName}`),
      };
      return task;
    }

    const task = this.createTask(contextId);
    task.history.push(message);
    task.status = { state: "TASK_STATE_SUBMITTED", timestamp: Date.now() };

    this.processTask(task, handler, message).catch((err) => {
      task.status = {
        state: "TASK_STATE_FAILED",
        timestamp: Date.now(),
        message: textMessage("ROLE_AGENT", String(err)),
      };
    });
    return task;
  }

  getTask(taskId: string): Task | undefined {
    return this.tasks.get(taskId);
  }

  cancelTask(taskId: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task || TERMINAL_STATES.includes(task.status.state)) return false;
    task.status = { state: "TASK_STATE_CANCELED", timestamp: Date.now() };
    this.emit(taskId, {
      statusUpdate: { taskId, status: task.status },
    });
    return true;
  }

  private createTask(contextId?: string): Task {
    const task: Task = {
      id: crypto.randomUUID(),
      contextId: contextId ?? crypto.randomUUID(),
      status: { state: "TASK_STATE_SUBMITTED", timestamp: Date.now() },
      artifacts: [],
      history: [],
    };
    this.tasks.set(task.id, task);
    return task;
  }

  private async processTask(
    task: Task,
    handler: TaskHandler,
    message: AgentMessage
  ) {
    task.status = { state: "TASK_STATE_WORKING", timestamp: Date.now() };
    this.emit(task.id, {
      statusUpdate: { taskId: task.id, status: task.status },
    });

    try {
      for await (const event of handler(task, message)) {
        if (TERMINAL_STATES.includes(task.status.state)) break;

        if ("statusUpdate" in event) {
          task.status = event.statusUpdate.status;
        }
        if ("artifactUpdate" in event) {
          const update = event.artifactUpdate;
          const existing = task.artifacts.find(
            (a) => a.id === update.artifact.id
          );
          if (existing && update.append) {
            existing.parts.push(...update.artifact.parts);
          } else {
            task.artifacts.push(update.artifact);
          }
        }
        this.emit(task.id, event);
      }
    } catch (err) {
      task.status = {
        state: "TASK_STATE_FAILED",
        timestamp: Date.now(),
        message: textMessage("ROLE_AGENT", String(err)),
      };
      this.emit(task.id, {
        statusUpdate: { taskId: task.id, status: task.status },
      });
    }
  }

  private emit(taskId: string, event: TaskEvent) {
    for (const listener of this.listeners.get(taskId) ?? []) {
      listener(event);
    }
  }
}
```

Điều này thực hiện chu kỳ hoạt động thực tế của A2A: `TASK_STATE_SUBMITTED`- `TASK_STATE_WORKING`- `TASK_STATE_INPUT_REQUIRED`Các bộ xử lý là các bộ phát điện đồng bộ`statusUpdate`và `artifactUpdate`các sự kiện, cùng các vòng bao bì dòng SSE mang.

### Bước 4: Đường kiểm toán theo phong cách ACP

Kết nối thông tin liên lạc với việc theo dõi quỹ đạo:

```typescript
type AuditEntry = {
  runId: string;
  agentName: string;
  input: AgentMessage[];
  output: AgentMessage[];
  trajectory: TrajectoryEntry[];
  status: "created" | "in-progress" | "completed" | "failed" | "awaiting";
  startedAt: number;
  completedAt?: number;
  sessionId?: string;
};

class AuditableRunner {
  private log: AuditEntry[] = [];
  private handlers: Map<
    string,
    (input: AgentMessage[]) => Promise<{
      output: AgentMessage[];
      trajectory: TrajectoryEntry[];
    }>
  > = new Map();

  registerAgent(
    name: string,
    handler: (input: AgentMessage[]) => Promise<{
      output: AgentMessage[];
      trajectory: TrajectoryEntry[];
    }>
  ) {
    this.handlers.set(name, handler);
  }

  async run(
    agentName: string,
    input: AgentMessage[],
    sessionId?: string
  ): Promise<AuditEntry> {
    const entry: AuditEntry = {
      runId: crypto.randomUUID(),
      agentName,
      input: structuredClone(input),
      output: [],
      trajectory: [],
      status: "created",
      startedAt: Date.now(),
      sessionId,
    };
    this.log.push(entry);

    const handler = this.handlers.get(agentName);
    if (!handler) {
      entry.status = "failed";
      return entry;
    }

    entry.status = "in-progress";
    try {
      const result = await handler(input);
      entry.output = structuredClone(result.output);
      entry.trajectory = structuredClone(result.trajectory);
      entry.status = "completed";
      entry.completedAt = Date.now();
    } catch (err) {
      entry.status = "failed";
      entry.trajectory.push({
        reasoning: `Error: ${String(err)}`,
        timestamp: Date.now(),
      });
      entry.completedAt = Date.now();
    }
    return entry;
  }

  getFullAuditLog(): AuditEntry[] {
    return structuredClone(this.log);
  }

  getAuditLogForAgent(agentName: string): AuditEntry[] {
    return structuredClone(
      this.log.filter((e) => e.agentName === agentName)
    );
  }

  getAuditLogForSession(sessionId: string): AuditEntry[] {
    return structuredClone(
      this.log.filter((e) => e.sessionId === sessionId)
    );
  }

  getTrajectoryForRun(runId: string): TrajectoryEntry[] {
    const entry = this.log.find((e) => e.runId === runId);
    return entry ? structuredClone(entry.trajectory) : [];
  }
}
```

Mỗi hành động của đại lý tạo ra một mục kiểm toán đầy đủ: những gì đã vào, những gì đã ra, và quỹ đạo đầy đủ của các cuộc gọi công cụ và các bước lý luận giữa hai. Bạn có thể truy vấn theo đại lý, theo phiên hoặc theo chạy riêng lẻ.

### Bước 5: Kiểm tra danh tính theo kiểu ANP

Xây dựng danh tính và xác minh dựa trên DID:

```typescript
type VerificationMethod = {
  id: string;
  type: string;
  controller: string;
  publicKeyDer: string;
};

type DIDDocument = {
  id: string;
  verificationMethod: VerificationMethod[];
  authentication: string[];
  keyAgreement: string[];
  humanAuthorization: string[];
  service: { id: string; type: string; serviceEndpoint: string }[];
};

type AgentIdentity = {
  did: string;
  document: DIDDocument;
  privateKey: crypto.KeyObject;
  publicKey: crypto.KeyObject;
};

class IdentityRegistry {
  private documents: Map<string, DIDDocument> = new Map();

  publish(doc: DIDDocument) {
    this.documents.set(doc.id, doc);
  }

  resolve(did: string): DIDDocument | undefined {
    return this.documents.get(did);
  }

  verify(did: string, signature: string, payload: string): boolean {
    const doc = this.documents.get(did);
    if (!doc) return false;

    const authKeyIds = doc.authentication;
    const authKeys = doc.verificationMethod.filter((vm) =>
      authKeyIds.includes(vm.id)
    );

    for (const key of authKeys) {
      const publicKey = crypto.createPublicKey({
        key: Buffer.from(key.publicKeyDer, "base64"),
        format: "der",
        type: "spki",
      });
      const isValid = crypto.verify(
        null,
        Buffer.from(payload),
        publicKey,
        Buffer.from(signature, "hex")
      );
      if (isValid) return true;
    }
    return false;
  }

  requiresHumanAuth(did: string, operationKeyId: string): boolean {
    const doc = this.documents.get(did);
    if (!doc) return false;
    return doc.humanAuthorization.includes(operationKeyId);
  }
}

function createIdentity(domain: string, agentName: string): AgentIdentity {
  const did = `did:wba:${domain}:agent:${agentName}`;
  const { publicKey, privateKey } = crypto.generateKeyPairSync("ed25519");

  const publicKeyDer = publicKey
    .export({ format: "der", type: "spki" })
    .toString("base64");

  const keyId = `${did}#key-1`;
  const encKeyId = `${did}#key-x25519-1`;

  const document: DIDDocument = {
    id: did,
    verificationMethod: [
      {
        id: keyId,
        type: "Ed25519VerificationKey2020",
        controller: did,
        publicKeyDer,
      },
      {
        id: encKeyId,
        type: "X25519KeyAgreementKey2019",
        controller: did,
        publicKeyDer,
      },
    ],
    authentication: [keyId],
    keyAgreement: [encKeyId],
    humanAuthorization: [],
    service: [
      {
        id: `${did}#agent-description`,
        type: "AgentDescription",
        serviceEndpoint: `https://${domain}/agents/${agentName}/ad.json`,
      },
    ],
  };

  return { did, document, privateKey, publicKey };
}

function signPayload(identity: AgentIdentity, payload: string): string {
  return crypto
    .sign(null, Buffer.from(payload), identity.privateKey)
    .toString("hex");
}
```

Điều này phản ánh mô hình danh tính ANP thực sự: các đại lý có tài liệu DID với xác thực riêng biệt, thỏa thuận chính và khóa ủy quyền của con người.`IdentityRegistry`mô phỏng độ phân giải DID (trong sản xuất đây sẽ là HTTP lấy đến miền của đại lý).

### Bước 6: Protocol Gateway

Kết nối tất cả bốn giao thức vào một hệ thống thống nhất:

```mermaid
graph LR
    REQ[Incoming Request] --> ANP_V{ANP: Verify DID}
    ANP_V -->|Valid| A2A_D{A2A: Discover Agent}
    ANP_V -->|Invalid| REJECT[Reject]
    A2A_D -->|Found| ACP_A[ACP: Audit Run]
    A2A_D -->|Not Found| REJECT
    ACP_A --> A2A_T[A2A: Create Task]
    A2A_T --> RESULT[Task + Audit Entry]

    style ANP_V fill:#d1fae5,stroke:#059669
    style A2A_D fill:#dbeafe,stroke:#2563eb
    style ACP_A fill:#fef3c7,stroke:#d97706
    style A2A_T fill:#dbeafe,stroke:#2563eb
```

```typescript
class ProtocolGateway {
  private registry: AgentRegistry;
  private taskManager: TaskManager;
  private auditRunner: AuditableRunner;
  private identityRegistry: IdentityRegistry;

  constructor(
    registry: AgentRegistry,
    taskManager: TaskManager,
    auditRunner: AuditableRunner,
    identityRegistry: IdentityRegistry
  ) {
    this.registry = registry;
    this.taskManager = taskManager;
    this.auditRunner = auditRunner;
    this.identityRegistry = identityRegistry;
  }

  async delegateTask(
    fromDid: string,
    signature: string,
    targetAgent: string,
    message: AgentMessage,
    sessionId?: string
  ): Promise<{ task: Task; audit: AuditEntry } | { error: string }> {
    if (!this.identityRegistry.verify(fromDid, signature, message.id)) {
      return { error: "Identity verification failed" };
    }

    const card = this.registry.resolve(targetAgent);
    if (!card) {
      return { error: `Agent ${targetAgent} not found in registry` };
    }

    const audit = await this.auditRunner.run(
      targetAgent,
      [message],
      sessionId
    );
    const task = await this.taskManager.sendMessage(targetAgent, message);

    return { task, audit };
  }

  discoverAndDelegate(
    fromDid: string,
    signature: string,
    skillTag: string,
    message: AgentMessage
  ): Promise<{ task: Task; audit: AuditEntry } | { error: string }> {
    const candidates = this.registry.discoverBySkillTag(skillTag);
    if (candidates.length === 0) {
      return Promise.resolve({
        error: `No agents found with skill tag: ${skillTag}`,
      });
    }
    return this.delegateTask(
      fromDid,
      signature,
      candidates[0].name,
      message
    );
  }
}
```

Cổng thông tin làm bốn điều trong một cuộc gọi:
1. **ANP**: Kiểm tra danh tính của người gọi thông qua chữ ký DID
2. **A2A**: Khám phá các tác nhân mục tiêu và kiểm tra khả năng
3. **ACP**: Bị kết thúc trong một đường kiểm toán với quỹ đạo
4. **A2A**: Tạo ra một nhiệm vụ với toàn bộ vòng đời theo dõi

### Bước 7: Hãy kết nối tất cả

```typescript
async function protocolDemo() {
  const registry = new AgentRegistry();
  registry.register({
    name: "researcher",
    description: "Searches and summarizes findings",
    version: "1.0.0",
    supportedInterfaces: [
      {
        url: "https://researcher.local/a2a/v1",
        protocolBinding: "JSONRPC",
        protocolVersion: "1.0",
      },
    ],
    capabilities: { streaming: true, pushNotifications: false },
    defaultInputModes: ["text/plain"],
    defaultOutputModes: ["text/plain", "application/json"],
    skills: [
      {
        id: "web-research",
        name: "Web Research",
        description: "Searches the web",
        tags: ["research", "search", "summarization"],
        inputModes: ["text/plain"],
        outputModes: ["application/json"],
      },
    ],
  });
  registry.register({
    name: "coder",
    description: "Writes code from specs",
    version: "1.0.0",
    supportedInterfaces: [
      {
        url: "https://coder.local/a2a/v1",
        protocolBinding: "JSONRPC",
        protocolVersion: "1.0",
      },
    ],
    capabilities: { streaming: false, pushNotifications: false },
    defaultInputModes: ["text/plain", "application/json"],
    defaultOutputModes: ["text/plain"],
    skills: [
      {
        id: "code-gen",
        name: "Code Generation",
        description: "Generates code",
        tags: ["coding", "generation"],
        inputModes: ["text/plain", "application/json"],
        outputModes: ["text/plain"],
      },
    ],
  });

  const taskManager = new TaskManager();
  const auditRunner = new AuditableRunner();

  const researchTrajectory: TrajectoryEntry[] = [];

  taskManager.registerHandler(
    "researcher",
    async function* (task, message) {
      yield {
        statusUpdate: {
          taskId: task.id,
          status: {
            state: "TASK_STATE_WORKING" as const,
            timestamp: Date.now(),
          },
        },
      };

      researchTrajectory.push({
        reasoning: "Searching for React 19 documentation",
        toolName: "web_search",
        toolInput: { query: "React 19 compiler features" },
        toolOutput: {
          results: ["react.dev/blog/react-19", "github.com/react/react"],
        },
        timestamp: Date.now(),
      });

      researchTrajectory.push({
        reasoning: "Extracting key findings from search results",
        toolName: "doc_analysis",
        toolInput: { url: "react.dev/blog/react-19" },
        toolOutput: {
          summary:
            "React 19 compiler auto-memoizes, no manual useMemo needed",
        },
        timestamp: Date.now(),
      });

      yield {
        artifactUpdate: {
          taskId: task.id,
          artifact: {
            id: crypto.randomUUID(),
            name: "research-results",
            parts: [
              {
                data: {
                  findings: [
                    "React 19 compiler auto-memoizes components",
                    "No more manual useMemo/useCallback needed",
                    "Compiler runs at build time, not runtime",
                  ],
                  sources: ["react.dev/blog/react-19"],
                },
                mediaType: "application/json",
              },
            ],
          },
          append: false,
          lastChunk: true,
        },
      };

      yield {
        statusUpdate: {
          taskId: task.id,
          status: {
            state: "TASK_STATE_COMPLETED" as const,
            timestamp: Date.now(),
          },
        },
      };
    }
  );

  auditRunner.registerAgent("researcher", async () => ({
    output: [
      textMessage("ROLE_AGENT", "React 19 compiler auto-memoizes components"),
    ],
    trajectory: researchTrajectory,
  }));

  const identityRegistry = new IdentityRegistry();

  const coderIdentity = createIdentity("coder.local", "coder");
  const researcherIdentity = createIdentity("researcher.local", "researcher");

  identityRegistry.publish(coderIdentity.document);
  identityRegistry.publish(researcherIdentity.document);

  const gateway = new ProtocolGateway(
    registry,
    taskManager,
    auditRunner,
    identityRegistry
  );

  console.log("=== Protocol Demo ===\n");

  console.log("1. Agent Discovery (A2A)");
  const researchAgents = registry.discoverBySkillTag("research");
  console.log(
    `   Found ${researchAgents.length} agent(s):`,
    researchAgents.map((a) => a.name)
  );

  console.log("\n2. Identity Verification (ANP)");
  const message = textMessage("ROLE_USER", "Research React 19 compiler features");
  const signature = signPayload(coderIdentity, message.id);
  const verified = identityRegistry.verify(
    coderIdentity.did,
    signature,
    message.id
  );
  console.log(`   Coder DID: ${coderIdentity.did}`);
  console.log(`   Signature verified: ${verified}`);

  console.log("\n3. Task Delegation (A2A + ACP + ANP)");
  const result = await gateway.delegateTask(
    coderIdentity.did,
    signature,
    "researcher",
    message,
    "session-001"
  );

  if ("error" in result) {
    console.log(`   Error: ${result.error}`);
    return;
  }

  console.log(`   Task ID: ${result.task.id}`);
  console.log(`   Task state: ${result.task.status.state}`);
  console.log(`   Artifacts: ${result.task.artifacts.length}`);

  console.log("\n4. Audit Trail (ACP)");
  console.log(`   Run ID: ${result.audit.runId}`);
  console.log(`   Status: ${result.audit.status}`);
  console.log(`   Trajectory steps: ${result.audit.trajectory.length}`);
  for (const step of result.audit.trajectory) {
    console.log(`     - ${step.reasoning}`);
    if (step.toolName) {
      console.log(`       Tool: ${step.toolName}`);
    }
  }

  console.log("\n5. Full Audit Log");
  const fullLog = auditRunner.getFullAuditLog();
  console.log(`   Total runs: ${fullLog.length}`);
  for (const entry of fullLog) {
    const duration = entry.completedAt
      ? `${entry.completedAt - entry.startedAt}ms`
      : "in-progress";
    console.log(`   ${entry.agentName}: ${entry.status} (${duration})`);
  }
}

protocolDemo().catch((err) => {
  console.error("Protocol demo failed:", err);
  process.exitCode = 1;
});
```

## Điều gì sai

Các giao thức giải quyết được con đường hạnh phúc.

**Schema drift.**Cảnh sát A xuất bản quảng cáo thẻ Cảnh sát`application/json`nhưng các bản JSON thay đổi giữa các phiên bản. Agent B phân tích các định dạng cũ và nhận được rác. sửa chữa: phiên bản kỹ năng của bạn và các bản phát hành.`version`về Agent Cards vì lý do này.

**State machine violations.**Một người quản lý đại lý sẽ tạo ra một `TASK_STATE_COMPLETED`update status, sau đó cố gắng để tạo ra nhiều artefacts hơn. nhiệm vụ là không thể thay đổi. mã của bạn lặng lẽ bỏ lại các bản cập nhật hoặc ném. sửa chữa: kiểm tra trạng thái cuối trước khi tạo ra.`TaskManager`trên thực thi điều này với `break`sau khi kết thúc.

**Trust resolution failures.**Cảnh sát A cố gắng xác minh DID của Cảnh sát B, nhưng tên miền của Cảnh sát B đã bị mất. Tài liệu DID không thể được lấy. Bạn không mở (tự nhận các đại lý không được xác minh) hoặc không đóng (để từ chối mọi thứ)? ANP khuyến cáo không đóng theo nguyên tắc ít tin tưởng nhất.

**Trajectory bloat.**Việc ghi lại quỹ đạo ACP là mạnh mẽ nhưng tốn kém. Một đại lý phức tạp thực hiện 200 cuộc gọi công cụ mỗi lần chạy tạo ra các mục kiểm toán lớn.

**Discovery thundering herd.**50 nhân viên tất cả truy vấn `GET /agents`sửa chữa: cache Agent Card với TTL, khoảng thời gian phát hiện, hoặc sử dụng đăng ký dựa trên push thay vì thăm dò.

## Sử dụng nó

### Thực hiện thực tế

**A2A**là một trong những người trưởng thành nhất.[official spec](https://github.com/google/A2A)là nguồn mở dưới Linux Foundation. SDK cho Python và TypeScript. Nếu các đại lý của bạn cần phát hiện động và hợp tác, bắt đầu từ đây.

**ACP**đang hợp nhất thành A2A. IBM [BeeAI project](https://github.com/i-am-bee/acp)ACP được tạo ra như một thay thế REST đầu tiên, nhưng khái niệm metadata quỹ đạo đang được hấp thụ vào hệ sinh thái A2A. Sử dụng các mô hình ACP (lập nhật quỹ đạo, vòng đời chạy) ngay cả khi bạn sử dụng A2A như là vận chuyển.

**ANP**là thử nghiệm nhất.[community repo](https://github.com/agent-network-protocol/AgentNetworkProtocol)có một Python SDK (AgentConnect). khái niệm đàm phán meta-protocol thực sự mới mẻ.

**MCP**Nếu bạn muốn các đại lý sử dụng công cụ, MCP là tiêu chuẩn.

### Chọn đúng giao thức

```mermaid
graph TD
    START{Do agents need<br/>to use tools?}
    START -->|Yes| MCP_R[Use MCP]
    START -->|No| TALK{Do agents need to<br/>talk to each other?}
    TALK -->|No| NONE[You don't need<br/>a protocol]
    TALK -->|Yes| AUDIT{Need audit trails<br/>for compliance?}
    AUDIT -->|Yes| ACP_R[A2A + ACP<br/>trajectory patterns]
    AUDIT -->|No| ORG{All agents<br/>within your org?}
    ORG -->|Yes| A2A_R[A2A<br/>Agent Cards + Tasks]
    ORG -->|No| INFRA{Shared<br/>infrastructure?}
    INFRA -->|Yes| BROKER[A2A + message broker]
    INFRA -->|No| ANP_R[ANP + A2A<br/>DID verification]

    style MCP_R fill:#d1fae5,stroke:#059669
    style A2A_R fill:#dbeafe,stroke:#2563eb
    style ACP_R fill:#fef3c7,stroke:#d97706
    style ANP_R fill:#f3e8ff,stroke:#7c3aed
    style BROKER fill:#e0e7ff,stroke:#4338ca
```

## Chuyển nó

Bài học này mang lại:
- `code/main.ts`-- thực hiện hoàn toàn tất cả bốn mô hình giao thức
- `outputs/prompt-protocol-selector.md`-- một lời nhắc giúp bạn chọn các giao thức cho hệ thống của bạn

## Các bài tập

1. **Multi-hop task delegation.**Tăng `TaskManager`Một nhà nghiên cứu nhận được một nhiệm vụ, đại diện "bảo sát" và "chổ chốt" các nhiệm vụ cho hai đại lý chuyên môn, chờ đợi cả hai hoàn thành, sau đó sáp nhập kết quả thành các hiện vật của riêng mình.

2. **Streaming audit trail.**Thay đổi `AuditableRunner`thay vì chờ đợi kết quả đầy đủ, yield `AuditEntry`cập nhật trong thời gian thực khi các mục quỹ đạo được thêm vào. Sử dụng một máy phát điện async tạo ra các ảnh chụp nhanh kiểm toán.

3. **DID rotation.**Thêm vòng quay khóa vào `IdentityRegistry`Một đại lý nên có thể xuất bản một tài liệu DID mới với các khóa cập nhật trong khi duy trì một `previousDid`Các nhà xác minh nên chấp nhận chữ ký từ cả khóa hiện tại và trước đó trong một thời gian miễn phí.

4. **Protocol negotiation.**Thực hiện khái niệm giao thức meta của ANP. Hai đại lý trao đổi `protocolNegotiation`Thông điệp với các định dạng ứng cử viên (ví dụ: "Tôi có thể nói JSON-RPC" so với "Tôi thích REST"). Sau tối đa 3 vòng, họ đồng ý về một định dạng hoặc thời gian nghỉ.`TaskManager`hoặc `AuditableRunner`Họ dùng.

5. **Rate-limited discovery.**Thêm một `RateLimitedRegistry`gói mà cache tìm kiếm thẻ Agent với một TTL cấu hình và giới hạn các truy vấn khám phá mỗi đại lý mỗi giây. mô phỏng một đàn đàn thổi thồng của 100 đại lý phát hiện lẫn nhau khi khởi động và đo sự khác biệt.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| MCP | "The protocol for AI tools" | A client-server protocol for agents to discover and use tools. Agent-to-tool, not agent-to-agent. |
| A2A | "Google's agent protocol" | A peer-to-peer protocol for agent collaboration under the Linux Foundation. Discovery via Agent Cards, 9-state task lifecycle, streaming via SSE. Supports JSON-RPC, REST, and gRPC bindings. |
| ACP | "Enterprise agent messaging" | IBM/BeeAI's REST API for agent runs with TrajectoryMetadata: every response carries the full chain of reasoning and tool calls. Merging into A2A. |
| ANP | "Decentralized agent identity" | A community protocol using `did:wba` (DID) for cryptographic identity, HPKE for E2EE, and AI-powered meta-protocol negotiation for agents that have never seen each other. |
| Agent Card | "An agent's business card" | A JSON document at `/.well-known/agent-card.json` describing skills, supported MIME types, security schemes, and protocol bindings. |
| DID | "Decentralized ID" | W3C standard for cryptographically verifiable identities hosted on the agent's own domain. ANP uses `did:wba` method. |
| TrajectoryMetadata | "The audit receipt" | ACP's mechanism for attaching reasoning steps, tool calls, and their inputs/outputs to every agent response. |
| Meta-protocol | "Agents negotiating how to talk" | ANP's approach where agents use natural language to dynamically agree on data formats, then generate code to handle them. |
| Task | "A unit of work" | A2A's stateful object tracking work from submission through completion. Immutable once terminal. |

## Đọc thêm

- [Google A2A specification](https://github.com/google/A2A)-- thông số kỹ thuật chính thức và SDK (v1.0.1, Linux Foundation)
- [IBM/BeeAI ACP specification](https://github.com/i-am-bee/acp)-- OpenAPI 3.1 spec cho các hoạt động của đại lý và quỹ đạo metadata
- [Agent Network Protocol](https://github.com/agent-network-protocol/AgentNetworkProtocol)-- DID dựa trên danh tính, E2EE, đàm phán giao thức meta
- [Model Context Protocol docs](https://modelcontextprotocol.io/)-- Khóa kỹ thuật MCP của Anthropic (được bao gồm trong giai đoạn 13)
- [W3C Decentralized Identifiers](https://www.w3.org/TR/did-core/)-- tiêu chuẩn danh tính cơ sở cho ANP
- [RFC 9180 (HPKE)](https://www.rfc-editor.org/rfc/rfc9180)-- hệ thống mã hóa ANP sử dụng cho E2EE
- [FIPA Agent Communication Language](http://www.fipa.org/specs/fipa00061/SC00061G.html)-- là tiền thân học thuật cho các giao thức đại lý hiện đại
