# rea capabilities

> rea capabilities groups every operation with its declared side effects, so you know before a call whether it starts a process, writes files, or edits the session database.

An agent is about to call a tool on a file you gave it. Some rea operations start a Java process, and some write a runtime directory. Three edit the names in the analysis database, and a Hopper operation would open a window. You want that list before the call, not after. `rea capabilities` prints it, and the same record reaches an agent as the MCP tool `binary_session`.

When you finish this section, you can read a capability group, name the seven effect flags, and pick the operations that change nothing.

## The command and its record

`capabilities` and `providers` are registered together, and `capabilities` is described as "List provider capabilities and side effects". Both print the session status, the same object that `binary_session` returns over MCP. With no target open it starts `open: false`:

```listing
title: the head of the status record
source: capture/out/02-capabilities.json
lang: json
note: Cut to the first group of the first provider block and the first fields of the ghidra candidate. The other three groups of that block, the other three target-free blocks, and the other two candidates are cut.
---
{
  "open": false,
  "capabilities": [
    {
      "provider": {
        "id": "native-macos",
        "name": "macOS native inspection utilities",
        "version": null
      },
      "shared_limitations": [
        "Availability and textual formats depend on the installed macOS/Xcode toolchain."
      ],
      "groups": [
        {
          "operations": [
            "capture_native_ui_scenario",
            "observe_native_calls"
          ],
          "effects": [
            "launches_process",
            "may_show_ui",
            "may_access_network",
            "may_write_filesystem"
          ],
          "limitations": [],
          "available": true,
          "reason": null,
          "availability_code": null
        },
        …
  ],
  "analysis_provider_binding": null,
  "analysis_provider_candidates": [
    {
      "provider": {
        "id": "ghidra",
        "name": "Ghidra",
        "version": null
      },
      "availability": {
        "status": "available",
…
```

The record has eight top-level fields in the capture: `open`, `capabilities`, `analysis_provider_binding`, `analysis_provider_candidates`, `analysis_run`, `analysis_activity`, `provider_operation_health`, and `server_identity`. `capabilities` holds the 19 operations that need no bound provider, in four provider blocks: `native-macos`, `rea-artifact-graph`, `rea-dotnet-static`, and `sqlite`. Examples are `inspect_artifact`, `inspect_macho`, and the managed .NET readers. `analysis_provider_candidates` holds the three deep providers with their availability and their own provider blocks. `analysis_provider_binding` is `null` because no target is open, and every candidate reports `target_support.status: "unknown"` with the reason `No target is open.`.

One target-free operation is not available on the capture Mac. `inspect_sqlite_database` has `available: false` and the code `unsupported_version`. Its reason says that it needs Node.js 24.15 or a later 24.x release, or Node.js 26 or later. The capture Mac runs Node 22.22.0.

**Capability group:** a list of operations that share one `available` flag, one list of effects, and one list of limitations. The `effects` list names only the flags that are true. The full limitation list of an operation is the block's `shared_limitations` followed by the group's `limitations`. When a group is not available, `reason` and `availability_code` say why. The session builds the groups from each provider's descriptors, so the field names are the same for every provider. [Figure](#fig-2-3) shows one.

```figure
id: fig-2-3
kind: structure
title: one capability group, analyze_function under ghidra
claim: A capability group answers four questions before any call: is the operation available, what does it start, what does it write, and what does it change.
caption: Read the field, its value for the analyze_function group in the ghidra block, and the note. From capture/out/02-capabilities.json, with the schema from src/contracts/toolOutputSchemaPrimitives.ts at the tag.
```

## The seven effects

The `effects` list of a group names flags from a fixed set of seven. A flag that the list does not name is false for every operation of the group.

| Flag | Meaning | True in the capture for |
|---|---|---|
| `mutates_artifact` | the operation edits the analysis database: names or comments | 3 Ghidra operations, 6 Hopper operations |
| `launches_process` | the operation starts a program, here the Ghidra JVM or a macOS tool | every Ghidra and Hopper operation, 12 target-free ones |
| `may_show_ui` | a window can open | every Hopper operation, 4 IDA operations, 2 target-free ones |
| `may_access_network` | the operation can open a connection | every IDA adapter operation, 2 target-free ones |
| `may_write_filesystem` | the operation writes files, here the runtime root and the project | every Ghidra operation, 6 Hopper operations, 6 target-free ones |
| `changes_permissions` | the operation changes file or process permissions | nothing |
| `requires_root` | the operation needs root | nothing |

Two flags need a second look. `mutates_artifact` sounds like a write to your file, and it is not. A shared limitation of the Ghidra block says: "The imported Program and temporary project are ephemeral and deleted on close. Annotation operations edit session database metadata; original executable bytes are never written." (capture/out/02-capabilities.json). The three operations that set it, `annotate_native_function`, `set_address_name`, and `set_addresses_names`, rename symbols in Ghidra's database, and the edit dies with the session.

`may_write_filesystem` is true for every Ghidra operation, even `list_documents`. The first call writes the runtime root, the private copy, and the project ([section 1.2](#s-rea-function-end-to-end)). [Figure](#fig-2-4) counts the flags per provider.

```figure
id: fig-2-4
kind: comparison
title: declared effects by provider in the capture
claim: Every Ghidra operation declares launches_process and may_write_filesystem, three also declare mutates_artifact, and no operation in the capture declares changes_permissions or requires_root.
caption: Read one column per provider and one row per flag, and each cell counts the operations that declare the flag. Counted from capture/out/02-capabilities.json.
```

## Three candidates, one bound

```listing
title: the two candidates that are not available
source: capture/out/02-capabilities.json
lang: json
note: Cut to the provider and availability of hopper and ida. Their target support and their provider blocks are cut.
---
    {
      "provider": {
        "id": "hopper",
        "name": "Hopper Disassembler",
        "version": null
      },
      "availability": {
        "status": "unavailable",
        "code": "executable_missing",
        "reason": "Hopper launcher is missing or not executable: /Applications/Hopper Disassembler.app/Contents/MacOS/hopper",
…
    {
      "provider": {
        "id": "ida",
        "name": "IDA MCP adapter",
        "version": "1"
      },
      "availability": {
        "status": "unavailable",
        "code": "not_configured",
        "reason": "Set REA_IDA_MCP_CONFIG to an upstream IDA MCP JSON registration; see docs/ida-provider.md.",
…
```

The `ghidra` candidate is `available`, and its `diagnostics` repeat the eight checks of [section 2.2](#s-rea-doctor-and-the-ghidra-provider) with the same paths. `hopper` is unavailable with the code `executable_missing`, and `ida` with `not_configured`. Each unavailable candidate still lists its operations, 37 for Hopper and 13 for IDA. All of them have `available: true` except Hopper's `search_bytes`, which has `available: false` and the reason "Masked byte search has not been verified for Hopper.".

Availability of the engine and availability of an operation are two different facts. The docs put the rule in one sentence: "These commands describe binary-session providers and auxiliary capabilities." {{cli Choose a provider}}. With `--provider ghidra` or `provider_id: "ghidra"`, only the Ghidra block applies.

## Limitations are part of the declaration

Every Ghidra operation carries four shared limitations before its own. The session serves operations only after default auto-analysis completes. The Program and the project are ephemeral. Cancelling a request terminates the database and discards annotations. Addresses are lowercase `0x` hexadecimal in the default space (capture/out/02-capabilities.json).

Then each group adds its own, and these are the sentences to read before you trust an empty list. `list_strings` says: "an empty search does not establish that text is absent from the artifact." `procedure_callers` declares no limitation of its own. Its answer carries `relationship_coverage`, and the Ghidra diagnostic in that field says that only resolved call references are returned and that "unresolved computed or indirect calls remain unknown" (capture/out/06-function-percent-change.json). The design rule behind all of them is "Missing coverage is unknown, not absence." {{design Define the contract}}.

## A second vocabulary in tools/list

The record ends with `server_identity`. Its `catalog` holds two things: the counts, and three SHA-256 digests of the catalog:

```listing
title: the catalog in server_identity
source: capture/out/02-capabilities.json
lang: json
note: Cut to the catalog. The digests are cut to eight hex digits, and the other fields of server_identity are cut.
---
    "catalog": {
      "counts": {
        "cli_commands": 153,
        "mcp_tools": 207,
        "mcp_prompts": 6
      },
      "digests": {
        "tools_sha256": "805433f1…",
        "prompts_sha256": "7152916a…",
        "combined_sha256": "405725e7…"
      }
    },
```

The catalog has no per-tool entries. The side effects of each MCP tool are in the tool annotations that `tools/list` returns:

```listing
title: the tools/list entry for xrefs
source: capture/out/10-mcp-02-tools-list.json
lang: json
note: Cut to the name, the title, and the annotations. The description and the two schemas are cut.
---
        {
          "name": "xrefs",
          "title": "Xrefs",
          …
          "annotations": {
            "readOnlyHint": false,
            "destructiveHint": false,
            "idempotentHint": true,
            "openWorldHint": false
          },
          …
```

In the source, `xrefs` has `mutatesSession: true`, because each answer goes into the Evidence ledger. rea derives the annotations from those effects, so `readOnlyHint` is false. The provider capability of `xrefs` still does not name `mutates_artifact`. The two vocabularies answer two questions. The capability says what the engine does to your machine, and the annotations say what the tool does to the MCP session. Read the one that matches your question, and do not mix them.

```takeaways
- Run `rea capabilities --format json` before the first call on a new machine.
- Treat `launches_process` and `may_write_filesystem` as the cost of every Ghidra operation.
- Avoid `annotate_native_function`, `set_address_name`, and `set_addresses_names` unless you want session renames.
- Read the shared limitations and each group's `limitations` before you read an empty result as absence.
```

Sources: cli Choose a provider (rea docs/cli.md); design Define the contract (rea docs/tool-design.md); rea at 3b79af2: src/cli/utilityCommands.ts, src/application/binary/BinarySessionStatus.ts, src/contracts/toolOutputSchemaPrimitives.ts, src/contracts/toolEffects.ts, src/serverIdentity.ts, src/catalogIdentity.ts, src/domain/relationshipCoverage.ts, bridge/ghidra/ReaGhidraBridge.java; capture/out/02-capabilities.json, 06-function-percent-change.json, 10-mcp-02-tools-list.json
