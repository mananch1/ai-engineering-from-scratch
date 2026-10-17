#  代理人至代理人协议

> 现在,我们需要一个代理.  A2A (Agent2Agent) 是一个允许不同框架构建的不透明代理合作的开放协议. 谷歌于2025年4月发布,并于2025年6月捐赠给Linux基金会,并在2026年4月获得了150多个支持者,包括AWS,Cisco,微软,Salesforce,SAP和ServiceNow. 它吸收了IBM的ACP,并增加了AP2支付延长. 这一课将经过代理卡,任务生命周期,以及三个协议绑定,使用A2A 1.0.1电线名称.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 学习目标

- 区分代理到工具 (MCP) 与代理到代理 (A2A) 使用情况.
- 在 发行代理卡`/.well-known/agent-card.json`具有技能和`supportedInterfaces`其他数据
- 按照任务生命周期进行: `TASK_STATE_SUBMITTED`现在`TASK_STATE_WORKING`现在`TASK_STATE_INPUT_REQUIRED`终端状态`TASK_STATE_COMPLETED`现在`TASK_STATE_FAILED`现在`TASK_STATE_CANCELED`现在`TASK_STATE_REJECTED`现在,我们要去.
- 使用每个部分包含一个信息`text`现在`raw`现在`url`其他`data`作为产品.

## 问题

客户服务代理需要将报告编写委托给专业的作家代理.

- 定制RESTAPI,但每次配对都是一次性的.
- 需要两个代理运行相同的框架.
- 没有合适:MCP是用来调用工具,而不是两个代理合作,同时保持每个代理的不透明的内部推理.

A2A填补了空白.它模拟了一个代理向另一个代理发送任务的交互,使用生命周期,消息和文物.所谓的代理内部状态保持不透明.

 A2A 是"让跨框架的代理人相互交谈"协议. 它不取代MCP;这两种协议是互补的.

## 概念

### 代理卡

每个符合A2A的代理人都会在`/.well-known/agent-card.json`其他:

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

发现是基于URL的:拿到卡片,选择第一个`supportedInterfaces`列表`protocolBinding`输入和输出模式是媒体类型.

### 签署的代理卡

卡片可以携带一个`signatures`每个输入都是一个JWS (RFC 7515) 通过卡的RFC 8785正规JSON计算,`signatures`消费者可以以同样的方式对卡进行加нони化,并验证.

### 任务生命周期

```text
TASK_STATE_SUBMITTED
  -> TASK_STATE_WORKING
  -> TASK_STATE_COMPLETED | TASK_STATE_FAILED | TASK_STATE_CANCELED | TASK_STATE_REJECTED

TASK_STATE_WORKING
  -> TASK_STATE_INPUT_REQUIRED
  -> TASK_STATE_WORKING (the client sends a message with the same taskId)
```

客户开始使用`SendMessage`调用代理通过状态过渡,客户采访了`GetTask`或通过 SSE 流动`SendStreamingMessage`其他`SubscribeToTask`一条溪流带着`statusUpdate`其他`artifactUpdate`任务达到终端状态时, 任务结束.`final`旗.

### 信息和部分

一个信息有一个`messageId`其他`role`(`ROLE_USER`或`ROLE_AGENT`),以及一个或多个部分.每个部分包含一个内容字段,该字段名称是类型.`kind`其他地方

- `text`简单内容.
- `raw`文件字节,JSON中64基,通常是`filename`其他`mediaType`现在,我们要去.
- `url`:链接到文件内容.
- `data`:结构化JSON有效载荷 (调用代理的结构化输入).

举个例子:

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

### 艺术品

输出是艺术品,而不是原始字符串.

```json
{
  "artifactId": "art-001",
  "name": "summary",
  "parts": [{"text": "...", "mediaType": "text/markdown"}]
}
```

艺术品可以作为块流传.`artifactUpdate`事件带着艺术品加上`append`其他`lastChunk`电话给人积累了.

### 协议的三个约束

1. **JSON-RPC 2.0 over HTTP**(`JSONRPC`对于请求,POST,SSE,流媒体. 方法是PascalCase:`SendMessage`现在`SendStreamingMessage`现在`GetTask`现在`ListTasks`现在`CancelTask`现在`SubscribeToTask`现在`CreateTaskPushNotificationConfig`现在`GetTaskPushNotificationConfig`现在`ListTaskPushNotificationConfigs`现在`DeleteTaskPushNotificationConfig`其他`GetExtendedAgentCard`现在,我们要去.
2. **gRPC**(`GRPC`对于gRPC原生企业环境.
3. **HTTP+JSON/REST**(`HTTP+JSON`资源URL如`POST /message:send`其他`GET /tasks/{id}`现在,我们要去.

所有三个结合都具有相同的数据模型.`supportedInterfaces`条目名称 一个结合性和其`protocolVersion`客户发送标题`A2A-Version: 1.0`在每一个请求上,因为服务器读取一个没有请求的请求是0.3版本.

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

### 保持空位

设计原理:调用代理的内部状态不透明.调用者看到任务状态和文物.调用代理的思想链,其工具调用,其子代理委托都看不见.这与MCP不同,工具调用是透明的.

理由:A2A允许竞争对手在不透露内部信息的情况下协作.A2A可以是"调用这个客户服务代理"而不需要调用者学习该代理如何实现服务.

### 时间线

- **2025-04-09.**谷歌宣布A2A.
- **2025-06-23.**捐给Linux基金会.
- **2025-08.**吸收IBM的ACP.
- **2025-09.**扩展AP2 (代理支付) 船舶.
- **2026-04.**版本 1.0 发布了150多个支持组织.

### 与MCP的关系

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | Agent Card `securitySchemes` + `securityRequirements` |
| Transport | Stdio / Streamable HTTP | JSON-RPC / gRPC / HTTP+JSON |

许多生产系统都使用MCP,用于工具层,A2A用于协作层.

```figure
a2a-task-lifecycle
```

## 用它

`code/main.py`编写代理发布卡,研究代理发送卡`SendMessage`通过 PDF 部分和文本说明,任务通过`TASK_STATE_WORKING`其他`TASK_STATE_INPUT_REQUIRED`其他`TASK_STATE_WORKING`其他`TASK_STATE_COMPLETED`所有的 stdlib; 使用内存运输来集中关注消息形状.

什么要看:

- 机器人卡的JSON形状.
- 服务器侧任务ID分配和状态过渡.
- 输入内容字段的部分.
- `TASK_STATE_INPUT_REQUIRED`分支中任务.
- 工艺品在完成后返回.

## 运送它

这一课产生了`outputs/skill-a2a-agent-spec.md`由于一个新的代理,该技能应该被其他代理调用, 产生的代理卡JSON,技能方案,和终点蓝图.

## 运动

1. 跑步`code/main.py`追踪任务的整个生命周期,包括`TASK_STATE_INPUT_REQUIRED`打电话给代理人要求澄清时停下来.

2. 加入一个签名的代理卡.`signatures`随着`alg`设置为`HS256`没有卡的正规JSON签字`signatures`写一个验证器,确认它在一个突变的卡上失败了.

3. 执行任务流程`SendStreamingMessage`编辑代理发射了`task`现在,`artifactUpdate`子,和一个`statusUpdate`随着`TASK_STATE_COMPLETED`接下来关闭了电流,然后电话给电话的人积累了这些块.

4. 设计一个 A2A 代理,将一个 MCP 服务器包裹起来.将每个 MCP 工具映射到一个 A2A 技能.注意交易.

5. 阅读A2A v1.0公告并确定截至2026年4月,没有任何框架实施的唯一功能. (提示:它涉及多跳任务委托).

## 关键词

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

## 进一步阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/)可нони A2A规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A)参考实施和SDK
- [A2A v1.0.1 release](https://github.com/a2aproject/A2A/tree/v1.0.1)标签:`docs/specification.md`法律和法规`specification/a2a.proto`这一课
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) 2025年6月 管理转让
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade)路线图和合作伙伴势头
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) v1.0 发布说明和后退型紧指导
