# rea, providers, and Evidence

> rea standardizes the question, the session, and the Evidence record, and leaves import, analysis, and decompilation to the engine it binds, here Ghidra 12.1.4.

You compiled `ledger.c` with `clang -O1`, and you want to know what the machine code of `percent_change` does when `before` is 0. You can open the file in Ghidra and read the listing yourself. You can also ask rea, and rea asks Ghidra for you and returns one record that says what it found and how it knows. This section draws the line between the two programs.

When you finish this section, you can name what rea adds to Ghidra, what it never does, and which adapter to use for one question.

## Three things rea standardizes

**Evidence:** the one record type that every successful analysis call returns. It is a strict object with 16 fields. The fields are `evidence_id`, `subject`, `provider`, `analysis_profile`, `predicate_type`, `operation`, `parameters`, `raw_result`, `normalized_result`, `confidence`, `authority`, `environment`, `limitations`, `relationship_coverage`, `locations`, and `evidence_links`.

The id is `ev_` and a SHA-256 of the semantic content. The same observation of the same bytes gets the same id on any machine. When `relationship_coverage` is `null`, the id leaves it out. `confidence` is one of `observed`, `derived`, or `inferred`, and `authority` says what kind of source backs the fact.

The MCP docs state the rule for tools: "Evidence-producing tools return the complete canonical Evidence record in both text and structured content." {{mcp Tool results}}. [Section 1.2](#s-rea-function-end-to-end) reads one record field by field, and [Part 5](#p-5) covers the record in full.

**Provider:** an analysis engine behind one interface. `GhidraProvider implements AnalysisProviderCandidate` and offers `identity()`, `capabilities()`, `inspectAvailability()`, `inspectTargetSupport(target)`, `resolveAnalysisProfile(...)`, and `createClient(...)`. At 6.5.0 the session holds three deep providers, `hopper`, `ghidra`, and `ida`, plus auxiliary providers that need no engine. The registry sorts candidates by id and selects one without starting it, and "Registration order never affects selection or caller-visible output". Once bound, a provider stays bound: "A selected provider is never replaced automatically after a runtime failure." {{mcp Identity and discovery}}.

**Adapter:** the way a caller reaches the session. The one-shot CLI has 153 commands and one alias, and the MCP server has 207 tools and 6 prompts. The capture's `server_identity.catalog.counts` reports those numbers (capture/out/02-capabilities.json). Both adapters share one `BinarySession`, which "Owns the single active target shared by CLI and MCP adapters". CLI names use hyphens and MCP names use `_` between words. The native family breaks the rule: `rea function` calls `analyze_function`, `rea decompile` calls `procedure_pseudo_code`, and `rea analyze` calls `binary_overview`.

The capture used both adapters on the same file:

```listing
title: four one-shot commands and their exit status
source: capture/out/00-commands.txt
lang: text
note: Cut to the first four commands. The time of each command is cut, because it changes with every recording, and the other commands are cut.
---
$ rea doctor --provider ghidra --format json --filter-output healthy,environment_healthy,scope,scope_checks
[exit 0] (… s) -> 01-doctor-ghidra.json

$ rea capabilities --format json
[exit 0] (… s) -> 02-capabilities.json

$ rea inspect-artifact fixtures/ledger-arm64-O1 --format json
[exit 0] (… s) -> 03-inspect-artifact.json

$ rea inspect-binary-layout fixtures/ledger-x86_64-O1.o --format json
[exit 1] (… s) -> 04-inspect-binary-layout.json
…
```

The first three commands finish in about one second, because none of them starts Ghidra. The fourth fails, and [section 3.2](#s-inspect-binary-layout-and-the-load-image) explains why. [Figure](#fig-1-1) places each name on its side of the line.

```figure
id: fig-1-1
kind: layers
title: what rea standardizes and what it leaves to Ghidra
claim: rea owns the adapters, the session, and the Evidence record, and Ghidra owns import, analysis, and decompilation, with one JSON line per question between them.
caption: Read each stack from the top down. The counts come from capture/out/02-capabilities.json and 05-analyze-arm64-O1.json, the file names from 00-fixtures.txt, and the class names from docs/architecture.mermaid at the tag.
```

## What rea leaves to Ghidra

rea has no disassembler and no decompiler of its own. For a native executable it starts `analyzeHeadless` on a fresh temporary project and imports a private copy of the file. It lets Ghidra's default auto-analysis run, and then it asks questions over a socket. The answers are Ghidra's. The function dossier "combines Ghidra FunctionManager, Listing, ReferenceManager, BasicBlockModel, and decompiler observations", as every `analyze_function` record says in its limitations (capture/out/06-function-percent-change.json). The provider docs draw the same line: "pseudocode is Ghidra decompiler output. Neither is original source" {{ghidra Admitted function-analysis semantics}}.

**Bridge:** `ReaGhidraBridge.java`, a `GhidraScript` that `analyzeHeadless` runs as its last post-script and that then serves requests on a Unix socket. A request is one JSON line, `{ id, token, method, params }`, and a reply is `{ id, ok: true, result }` or `{ id, ok: false, error }`. The bridge answers 16 inventory methods and 13 function methods through one Java `switch`. [Part 4](#p-4) follows each of them.

Ghidra also owns the cost. Each one-shot command that needs Ghidra took 7 to 9 s on the capture Mac (capture/README.md, capture/out/00-commands.txt). The import and the auto-analysis take most of that time. Inside one MCP session the first Ghidra call took 5 to 7 s (capture/README.md, capture/out/10-mcp-session.txt). Every later call on the same file took under 100 ms.

## What rea reads by itself

Before any engine runs, rea reads the file. It detects ELF from `7f 45 4c 46`, `MZ`, and the four Mach-O magic values. It hashes every target with SHA-256 in 64 KiB chunks. It knows eight architectures: `x86`, `x86_64`, `arm`, `arm64`, `mips`, `6502`, `riscv`, and `powerpc`. A raw binary has no header, so the caller names its architecture.

Two commands show the result without Java. `rea inspect-artifact` walks the file with the provider `rea-artifact-graph` and returns an Evidence record in under a second, without Java (capture/out/03-inspect-artifact.json). On macOS, `rea inspect-macho` runs the Xcode tools through `xcrun` and took about three seconds (capture/out/04-inspect-macho.json). [Part 3](#p-3) reads those records.

## Formats beyond native code

Release 6.5.0 adds three readers. `inspect-wasm-artifact` reads a WebAssembly module through the WABT tools. `inspect-go-binary` reads Go build information. The raw-binary format opens a file that has no header, at an architecture that you name. The capture ran `inspect-wasm-artifact` on the fixture module, and the command exited 0 (capture/out/16-inspect-wasm-artifact.json). [Section 6.2](#s-inspect-wasm-artifact) reads that record.

```takeaways
- Ask rea for an Evidence record, and read Ghidra's own listing when you doubt it.
- Use the CLI for one question and the MCP server for a session of many.
- Treat the 16 fields as the contract, and the dossier inside `normalized_result` as Ghidra's answer.
- Expect 7 to 9 s for the first Ghidra question on a file, and milliseconds after it.
```

Sources: mcp Tool results, Identity and discovery (rea docs/mcp-contracts.md); ghidra Admitted function-analysis semantics (rea docs/ghidra-provider.md); rea at 3b79af2: src/domain/evidence.ts, src/ghidra/GhidraProvider.ts, src/composition/binary.ts, src/application/binary/AnalysisProviderRegistry.ts, src/application/binary/BinarySession.ts, src/cliCommandNames.ts, src/ghidra/GhidraClient.ts, src/ghidra/GhidraClientWire.ts, src/ghidra/protocol.ts, src/ghidra/GhidraInventoryValues.ts, src/ghidra/GhidraFunctionValues.ts, bridge/ghidra/ReaGhidraBridge.java, src/domain/binaryTarget.ts, src/application/BinaryTargetResolver.ts, src/domain/binaryTargetTypes.ts, src/catalogIdentity.ts, src/go/GoBinaryProvider.ts, src/wasm/WabtArtifactProvider.ts, docs/architecture.mermaid; capture/README.md; capture/out/00-commands.txt, 00-fixtures.txt, 02-capabilities.json, 03-inspect-artifact.json, 04-inspect-macho.json, 05-analyze-arm64-O1.json, 06-function-percent-change.json, 10-mcp-session.txt, 16-inspect-wasm-artifact.json
