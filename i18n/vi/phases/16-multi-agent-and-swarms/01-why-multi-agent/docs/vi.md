# Tại sao lại có nhiều nhân viên?

> Một nhân viên đâm vào một bức tường.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Mục tiêu học tập

- Xác định giới hạn đơn tác nhân (sự lội lội trong bối cảnh, chuyên môn hỗn hợp, nút thắt lỏng theo trình tự) và giải thích khi phân chia thành nhiều tác nhân là động đúng đắn
- So sánh các mô hình dàn nhạc (chuồng ống, fan-out song song, giám sát, cấp bậc) và chọn đúng cho một cấu trúc nhiệm vụ nhất định
- Thiết kế một hệ thống đa đại lý với ranh giới vai trò rõ ràng, trạng thái chia sẻ và hợp đồng giao tiếp
- Phân tích các sự thỏa hiệp về sự phức tạp của nhiều đại lý (sự trễ, chi phí, khó khăn trong việc gỡ lỗi) so với sự đơn giản của một đại lý

## Vấn đề

Bạn đã xây dựng một đại lý duy nhất trong giai đoạn 14. Nó hoạt động. Nó có thể đọc các tệp, chạy lệnh, gọi API và lý luận về kết quả. Sau đó bạn chỉ nó vào một cơ sở mã thực sự: 200 tệp, ba ngôn ngữ, các bài kiểm tra phụ thuộc vào cơ sở hạ tầng, và yêu cầu nghiên cứu các API bên ngoài trước khi viết mã.

Đại lý bị ngạt không phải vì LLM là ngốc, mà vì nhiệm vụ vượt quá những gì một vòng lặp đại lý có thể xử lý. cửa sổ ngữ cảnh chứa đầy nội dung tập tin. Đại lý quên những gì họ đọc 40 cuộc gọi công cụ trước. Nó cố gắng trở thành một nhà nghiên cứu, một lập trình viên, và một nhà phê bình tất cả cùng một lúc, và làm cả ba đều kém.

Đây là trần nhà đơn, bạn phải đánh vào nó mỗi khi một nhiệm vụ đòi hỏi:

- **More context than fits in one window**- đọc 50 tập tin thổi qua 200k token
- **Different expertise at different stages**- nghiên cứu đòi hỏi sự thúc đẩy khác so với việc tạo ra mã
- **Work that can happen in parallel**- Tại sao đọc ba tập tin theo trình tự khi bạn có thể đọc chúng cùng lúc?

## Khái niệm

### Màn giới đơn tác nhân

Một đại lý đơn lẻ là một vòng lặp, một cửa sổ ngữ cảnh, một lệnh hệ thống.

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

Ba thứ bị phá vỡ:

1. **Context saturation**- Kết quả công cụ tích lũy. Đến lượt 30, đại lý đã tiêu thụ 150k token nội dung tập tin, đầu ra lệnh, và lý luận trước.

2. **Role confusion**- một lệnh hệ thống nói "bạn là một nhà nghiên cứu, lập trình, kiểm tra và kiểm tra" tạo ra một đại lý nửa nghiên cứu, nửa mã hóa, và không bao giờ hoàn thành kiểm tra.

3. **Sequential bottleneck**- đại lý đọc tập tin A, rồi tập tin B, rồi tập tin C. Ba cuộc gọi liên tiếp của LLM. Ba vụ xử tử liên tiếp của công cụ.

### Giải pháp đa tác nhân

Chia công việc cho mỗi nhân viên một công việc, một cửa sổ ngữ cảnh, và một hệ thống nhắc nhở được điều chỉnh cho công việc đó:

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

Mỗi đại lý có:
- Một lời nhắc hệ thống tập trung ("Bạn là một nhà kiểm tra mã. Công việc duy nhất của bạn là tìm ra lỗi. ")
- Chiếc cửa sổ ngữ cảnh của riêng nó (không bị ô nhiễm bởi công việc của các nhân viên khác)
- Hợp đồng đầu vào/ ra ra rõ ràng (tận nhận ghi chú nghiên cứu, mã ra ra)

### Hệ thống thực sự làm điều này

**Claude Code subagents**- khi Claude Code sinh ra một con người với `Task`, nó tạo ra một nhân viên trẻ với một nhiệm vụ có mục tiêu. Cha mẹ giữ bối cảnh của nó sạch sẽ. đứa trẻ làm công việc tập trung và trả lại một bản tóm tắt.

**Devin**- chạy một đại lý lập kế hoạch, một đại lý lập trình và một đại lý trình duyệt.

**Multi-agent coding teams (SWE-bench)**- các hệ thống hiệu suất cao nhất trên ghế SWE sử dụng một nhà nghiên cứu đọc cơ sở mã, một lập kế hoạch thiết kế sửa chữa và một lập trình điều chỉnh mã thực hiện nó.

**ChatGPT Deep Research**- tạo ra nhiều nhân viên tìm kiếm song song, mỗi người khám phá một góc độ khác nhau, sau đó tổng hợp kết quả.

### Phân quang phổ

Multi-agent không phải là nhị phân.

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

**Single agent**- Một vòng lặp, một cú thôi thúc.

**Subagents**- một người cha sinh con để tập trung các nhiệm vụ phụ. người cha duy trì kế hoạch. trẻ báo cáo lại. Đây là điều Claude Code làm.

**Pipeline**- các đại lý chạy theo trình tự. đầu ra của đại lý A trở thành đầu vào của đại lý B. Được cho các dòng công việc giai đoạn: nghiên cứu -> mã -> đánh giá -> thử nghiệm.

**Team**- các đại lý chạy song song với một bus thông điệp chia sẻ mỗi người có một vai trò một dàn nhạc phối hợp tốt khi có những kỹ năng khác nhau được yêu cầu cùng lúc

**Swarm**- nhiều đại lý giống hệt hoặc gần giống hệt với trạng thái chung không có nhạc cụ cố định đại lý nhận công việc từ hàng đợi tốt cho các nhiệm vụ song song song có hiệu suất cao

### Bốn mô hình đa tác nhân

#### Mô hình 1: đường ống dẫn

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Mỗi đại lý biến đổi dữ liệu và truyền chúng ra.

#### Mô hình 2: Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Chia công việc qua các đại lý song song, sau đó kết hợp kết quả.

#### Mô hình 3: Người làm nhạc cụ

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

Một nhà dàn nhạc thông minh quyết định những gì phải làm, ủy quyền cho công nhân, và tổng hợp kết quả.

#### Mô hình 4: Nhóm đồng nghiệp

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

Không có người tổ chức trung tâm, các đại lý giao tiếp với nhau, quyết định xuất hiện từ sự tương tác, khó khăn hơn để sửa lỗi, nhưng có thể đạt được nhiều đại lý.

### Khi nào không nên sử dụng nhiều chất

Multi-agent thêm sự phức tạp. Mỗi tin nhắn giữa các đại lý là một điểm thất bại tiềm năng. Debug từ "đọc một cuộc trò chuyện" đến "để theo dõi tin nhắn trên năm đại lý".

**Stay single-agent when:**
- Nhiệm vụ phù hợp trong một cửa sổ ngữ cảnh (dưới ~ 100k token dữ liệu làm việc)
- Bạn không cần các hệ thống khác nhau cho các giai đoạn khác nhau
- Việc xử lý theo trình tự là đủ nhanh
- Nhiệm vụ là đủ đơn giản để chia nó thêm thêm chi phí hơn giá trị

**The complexity cost:**
- Mỗi ranh giới của đại lý là một bước nén thua lỗ: toàn bộ bối cảnh của đại lý A được tóm tắt thành một thông điệp cho đại lý B
- Lễ thuật phối hợp (người làm gì, khi nào, theo thứ tự nào) là nguồn lỗi của riêng nó
- N đại lý có nghĩa là N liên hệ LLM gọi tối thiểu, nhiều hơn nếu họ cần nói chuyện về phía trước và về phía sau
- Chi phí nhân: mỗi đại lý đốt token độc lập

Quy tắc: nếu một nhiệm vụ mất ít hơn 20 cuộc gọi công cụ và phù hợp với 100k token, hãy giữ nó đơn đại lý.

```figure
swarm-messages
```

## Hãy xây dựng nó

### Bước 1: Người đơn bị quá tải

Đây là một đại lý đơn lẻ cố gắng làm mọi thứ. Nó có một hệ thống lớn nhắc và một cửa sổ ngữ cảnh chứa nghiên cứu, mã, và đánh giá:

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

Vấn đề với phương pháp này:
- Chiếc cửa sổ ngữ cảnh tăng lên với mỗi giai đoạn.
- Các hệ thống yêu cầu là chung. Nó không thể được điều chỉnh cho mỗi giai đoạn.
- Không có gì chạy song song.

### Bước 2: Các đại lý chuyên nghiệp

Giờ thì chia nó ra.

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

Mỗi chuyên gia có một lời nhắc tập trung. Mỗi người nhận được một cửa sổ ngữ cảnh sạch sẽ chỉ có đầu vào cần thiết.

### Bước 3: Kết hợp thông qua tin nhắn

Đưa tin nhắn cho các chuyên gia cùng với thông điệp rõ ràng:

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

Mỗi đại lý chỉ nhận được những thông điệp được gửi đến họ. Không có ô nhiễm ngữ cảnh. 50k token của nhà nghiên cứu đọc tài liệu không bao giờ vào ngữ cảnh của nhà phê bình.

### Bước 4: So sánh

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

Phiên bản đa đại lý sử dụng nhiều mã thông báo tổng cộng hơn (ba đại lý, ba cuộc gọi LLM riêng biệt) nhưng bối cảnh của mỗi đại lý vẫn sạch sẽ.

## Sử dụng nó

Bài học này tạo ra một lời nhắc tái sử dụng để quyết định khi nào nên đi đa đại lý.`outputs/prompt-multi-agent-decision.md`- Tôi không biết.

## Các bài tập

1. Thêm một chuyên gia thứ tư: một đại lý "tử nghiệm" nhận mã từ trình lập trình và xem xét phản hồi từ người xem xét, sau đó viết các bài kiểm tra
2. Thay đổi đường ống để người xem có thể gửi phản hồi trở lại cho trình lập trình để một vòng sửa đổi (tối đa 2 vòng)
3. Chuyển đổi đường ống nối theo trình tự thành một fan-out: chạy nhà nghiên cứu và một "đánh phân tích yêu cầu" đại lý song song, sau đó kết hợp các đầu ra của họ trước khi chuyển đến coder

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | A set of peer agents with shared state and no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and delegates but may not do the actual work. |
| Coordinator | "The traffic cop" | A non-agent component (often just code, not an LLM) that routes messages between agents based on rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns that arise from agent interactions but were not explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting a task across parallel agents (fan-out), then combining their results (fan-in). |
| Message passing | "Agents talk to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## Đọc thêm

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)- khảo sát các mô hình đa tác nhân
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)- Microsoft đa đại lý hội thoại framework
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code)- cách Claude Code ủy nhiệm với Task
- [CrewAI documentation](https://docs.crewai.com/)- khung đa tác nhân dựa trên vai trò
