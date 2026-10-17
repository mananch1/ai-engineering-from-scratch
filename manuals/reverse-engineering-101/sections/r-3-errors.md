# Errors seen in the captures

> Two error envelopes, two unavailability codes, and two refused conclusions appear in the captures, and every rea error has the same five fields.

The capture kit recorded two calls that failed and kept each reply whole. Both came from rea's own error projection, one from the CLI and one from the MCP server. Two more files hold codes that are not errors: the availability of a provider that is not installed, and a comparison that reached no conclusion. This section lists all of them, then the schema they follow.

## The two error envelopes

The CLI printed its envelope to stdout and exited 1 when `inspect-binary-layout` met a Mach-O file on macOS:

```listing
title: capability_unavailable from the CLI
source: capture/out/04-inspect-binary-layout.json
lang: json
note: Nothing is cut. The exit status 1 is in capture/out/00-commands.txt.
---
{
  "code": "capability_unavailable",
  "category": "unsupported_provider",
  "message": "The initial real-verified pwntools artifact profile supports Linux x64 only.",
  "retryable": false,
  "remediation": {
    "action": "The initial real-verified pwntools artifact profile supports Linux x64 only."
  },
  "details": {
    "provider_id": "pwntools-elf",
    "operation": "inspect_binary_layout",
    "reason": "The initial real-verified pwntools artifact profile supports Linux x64 only."
  }
}
```

Over MCP the same projection travels as text. A failed `tools/call` returns `isError: true` and the `{ "error":... }` object as JSON in `content[0].text`, with no `structuredContent`, because the advertised output schema describes success only {{mcp Tool results}}. The capture kit decoded that text into `decoded_text` beside the raw reply:

```listing
title: invalid_request for a procedure that does not exist
source: capture/out/10-mcp-13-analyze-function-missing.json
lang: json
note: The request, the raw response, and details.diagnostics are cut. The raw response holds the same object as one JSON string. The diagnostics hold the Ghidra run id, the bridge endpoint, the launch output, and the remote failure code not_found.
---
  "decoded_text": {
    "error": {
      "code": "invalid_request",
      "category": "invalid_input",
      "message": "Analysis input is invalid. Check the arguments and try again.",
      "retryable": true,
      "remediation": {
        "action": "Correct the listed arguments and retry."
      },
      "details": {
        "operation": "analyze_function",
        "issues": [
          {
            "path": [],
            "reason": "invalid_value",
            "message": "Unknown Ghidra procedure name or address: _nothing"
          }
        ]
      }
    }
  }
```

The bridge raised `not_found` with the message in `issues[0].message`, and rea mapped a remote `not_found` to `AnalysisInputError`, whose code is `invalid_request`. The server log recorded the failure twice, once for the bridge layer and once for the server layer, both at level 40 (capture/out/10-mcp-server.log).

A call to a tool that the server does not have never reaches rea's projection. The MCP SDK answers with a JSON-RPC error `-32602` and the message `Tool <name> not found`, with no tool result. The 6.5.0 capture has no such call.

A misspelt CLI command gets the parser envelope `COMMAND_NOT_FOUND` with a `cta` object that suggests the nearest command. `analysisErrorProjectionSchema` does not define them. A script that reads rea errors needs a branch for a misspelt or missing command before it reads `category`. The 6.5.0 capture has no such call either.

## Codes that are not errors

`rea capabilities` lists every provider with an `availability` object. Two providers were absent on the capture host, and their codes name the cause without failing the command:

```listing
title: two unavailable providers
source: capture/out/02-capabilities.json
lang: json
note: Two of the availability objects. The diagnostics that follow each one are cut.
---
      "availability": {
        "status": "unavailable",
        "code": "executable_missing",
        "reason": "Hopper launcher is missing or not executable: /Applications/Hopper Disassembler.app/Contents/MacOS/hopper",
…
      "availability": {
        "status": "unavailable",
        "code": "not_configured",
        "reason": "Set REA_IDA_MCP_CONFIG to an upstream IDA MCP JSON registration; see docs/ida-provider.md.",
```

Two results in the captures refused a conclusion and exited 0. `rea compare` over the two bundles returned `status: "unknown"` with ten unresolved records (capture/out/14-compare-bundles.json, [Section 5.5](#s-compare-and-compare-functions)). No record is a member of both bundles, and a bundle carries no completeness marker. `verify_reconstruction` returned `status: "fail"` with two failed claims (capture/out/10-mcp-22-verify-reconstruction.json, [Section 7.1](#s-verify-reconstruction)). Neither is an error envelope, and both are results that say the question stays open or the claim is false. A script that treats a non-zero exit as the only failure signal misses both.

## The shape every rea error has

| Field | Meaning |
| --- | --- |
| `code` | the discriminator, one of 20 codes listed below |
| `category` | one of ten: `invalid_input`, `unsupported_provider`, `unsupported_target`, `integrity_mismatch`, `truncated`, `cancelled`, `timeout`, `resource_constraint`, `unavailable`, `execution_failure` |
| `message` | one plain sentence for a person, with provider names and internal reasons removed |
| `retryable` | true for `artifact_changed`, `invalid_request`, `provider_timeout`, `cancelled`, `revision_conflict`, and `provider_unavailable`, for a `capability_unavailable` error that the provider marks retryable, and for a `resource_constraint` error on capacity |
| `remediation.action` | one sentence that says what to do |
| `details` | optional, free JSON for generic codes, and a fixed object with `logical_path`, `declared_sha256`, `calculated_sha256`, and `unpacked` for `artifact_integrity_mismatch` |

The 20 codes are `invalid_request`, `access_denied`, `artifact_changed`, `unreadable_output`, `capability_unavailable`, `unsupported_target`, `provider_unavailable`, `provider_timeout`, `resource_constraint`, `cancelled`, `artifact_integrity_mismatch`, `artifact_operation_failed`, `evidence_integrity_mismatch`, `truncated`, `process_capture_failed`, `cleanup_incomplete`, `revision_conflict`, `configuration_invalid`, `target_unavailable`, and `execution_failure`. Each internal error class maps to one code. `AnalysisInputError` maps to `invalid_request`, `AnalysisCapabilityUnavailableError` to `capability_unavailable`, `AnalysisTimeoutError` to `provider_timeout`, and `ConfigurationError` to `configuration_invalid`. `NoBinaryOpenError` maps to `target_unavailable`, and `ProviderAdapterError` to `execution_failure`. A failure whose cleanup left owned resources behind takes the code `cleanup_incomplete` whatever the underlying code was, and lists them in `details`. The rea docs publish the same union as a JSON Schema, `error-contract.schema.json`.

Two codes deserve a note for the Ghidra provider. `provider_timeout` is what the first Ghidra-backed call returns when the startup deadline of [Section 8.1](#s-the-rea-mcp-server) passes. `target_unavailable` is what every provider tool returns before `open_binary`, with the message "No app is open. Ask the user which app to investigate, then call open_binary with its local path.". Neither appeared in the captures, because the kit set a long client deadline and opened a target first.

Sources: capture/out/04-inspect-binary-layout.json, 10-mcp-13-analyze-function-missing.json, 02-capabilities.json, 14-compare-bundles.json, 10-mcp-22-verify-reconstruction.json, 10-mcp-server.log, 00-commands.txt; rea docs/mcp-contracts.md (Tool results); rea at 3b79af2: src/contracts/errorSchemas.ts, src/domain/analysisErrorProjection.ts, src/domain/analysisErrorPresentation.ts, src/domain/configurationErrors.ts, src/ghidra/GhidraProviderClient.ts, src/ghidra/GhidraDiagnostics.ts, bridge/ghidra/ReaGhidraBridge.java, docs/user-visible-errors.csv, docs/error-contract.schema.json
