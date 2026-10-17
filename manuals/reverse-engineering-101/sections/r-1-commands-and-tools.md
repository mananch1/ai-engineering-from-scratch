# Commands and tools used in this manual

> Every rea command and MCP tool the captures ran, with what it answers, the capture file that holds its output, and the section that explains it.

rea 6.5.0 has 153 CLI commands and 207 MCP tools. `binary_session` reports these counts in its catalog (capture/out/10-mcp-04-binary-session.json). The capture kit ran 16 commands and 13 tools, plus the two MCP methods that every session starts with. CLI names join words with `-` and MCP names with `_`. Three native commands map to tools with other names: `function` calls `analyze_function`, `decompile` calls `procedure_pseudo_code`, and `analyze` calls `binary_overview`. The capture passed `--format json` to every command and `--filter-output` with a key list where a record was too large to keep whole (capture/out/00-commands.txt).

## CLI commands

| Command | Answers | Capture file | Section |
|---|---|---|---|
| `rea doctor --provider ghidra` | whether the provider and its environment are healthy, one check per row | `01-doctor-ghidra.json` | [2.2](#s-rea-doctor-and-the-ghidra-provider) |
| `rea capabilities` | the operations of each provider in groups, with the availability, the effects, and the limitations of each group, and the availability of each provider | `02-capabilities.json` | [2.3](#s-rea-capabilities) |
| `rea inspect-artifact <path>` | the content-addressed artifact graph of a file or package | `03-inspect-artifact.json` | [3.1](#s-inspect-artifact) |
| `rea inspect-binary-layout <path>` | ELF sections, symbols, and relocations through the pwntools profile, Linux x64 only, so it fails on macOS with `capability_unavailable` | `04-inspect-binary-layout.json` | [3.2](#s-inspect-binary-layout-and-the-load-image) |
| `rea inspect-macho <path>` | Mach-O load commands, segments, sections, imports, exports, and build metadata | `04-inspect-macho.json` | [3.2](#s-inspect-binary-layout-and-the-load-image) |
| `rea analyze <path> --provider ghidra` | `binary_overview`: the document, its segments, and the procedure and string counts | `05-analyze-arm64-O1.json`, `05-analyze-x86_64-O1.json` | [3.4](#s-analyze-overview) |
| `rea function <path> <name> --provider ghidra` | the `analyze_function` dossier of one procedure | `06-function-*.json` | [4.6](#s-analyze-function-and-pseudocode) |
| `rea decompile <path> <name> --provider ghidra` | the `procedure_pseudo_code` of one procedure | `07-decompile-*.json` | [4.6](#s-analyze-function-and-pseudocode) |
| `rea address-to-file-offset <path> <address> --provider ghidra` | the file offset of one virtual address | `08-address-to-file-offset.json` | [3.3](#s-address-to-file-offset) |
| `rea inspect-evidence-bundle '{"path":..., "detail": "summary"}'` | one row per record of a bundle, without the payloads | `11-inspect-evidence-bundle-*.json` | [5.4](#s-evidence-bundles-export-and-import) |
| `rea evidence-export <bundle> <out>` | a canonical copy of a bundle, keys sorted | `12-evidence-export.json`, `12-evidence-export.txt` | [5.4](#s-evidence-bundles-export-and-import) |
| `rea evidence-import <bundle>` | validation of every record id and manifest in a bundle | `13-evidence-import.json` | [5.4](#s-evidence-bundles-export-and-import) |
| `rea compare <left> <right>` | the two bundles by record membership and unknown revisions | `14-compare-bundles.json` | [5.5](#s-compare-and-compare-functions) |
| `rea analyze-javascript-application <dir>` | the application graph of a JavaScript or Electron application, without running it | `15-js-app.json`, `15-js-app-graph.json` | [6.1](#s-analyze-javascript-application) |
| `rea inspect-analysis-view <request.json>` | a summary or one page of a saved record | `15-js-app-view-summary.json`, `15-js-app-view-modules.json` | [6.1](#s-analyze-javascript-application) |
| `rea inspect-wasm-artifact <path>` | the validation result, the sections, and the import and export interface of one WebAssembly module, from WABT 1.0.42 | `16-inspect-wasm-artifact.json` | [6.2](#s-inspect-wasm-artifact) |
| `rea mcp` | the MCP server over stdio | `10-mcp-*.json`, `10-mcp-server.log` | [8.1](#s-the-rea-mcp-server) |

The CLI documents its exit status {{cli Output and exit status}}. It is `0` when the operation completed, with partial evidence or open questions allowed. It is `1` when it could not complete, and `128 + N` when signal `N` ended the process. `inspect-binary-layout` is the one command that exited `1` in the capture. An explicit `--provider` wins over `REA_ANALYSIS_PROVIDER`, and with neither set rea uses the single available provider that supports the target {{cli Choose a provider}}.

## MCP methods and tools

| Name | Answers | Hints | CLI command | Capture file | Section |
|---|---|---|---|---|---|
| `initialize` | the negotiated protocol version, the server capabilities, `serverInfo`, and the instruction string | method | | `10-mcp-01-initialize.json` | [8.1](#s-the-rea-mcp-server) |
| `tools/list` | all 207 tools with schemas and annotations | method | | `10-mcp-02-tools-list.json` | [8.1](#s-the-rea-mcp-server) |
| `open_binary` | the opened target: path, format, kind, SHA-256, architecture, with the provider bound and not started | open world | none | `10-mcp-03`, `10-mcp-16` | [4.1](#s-open-binary-and-the-headless-bridge) |
| `binary_session` | the current target, binding, run id, tool availability, client features, and server identity | open world | none | `10-mcp-04-binary-session.json` | [8.1](#s-the-rea-mcp-server) |
| `list_procedures` | every analyzed procedure as address and name, with thunk and external flags | | none | `10-mcp-05-list-procedures.json` | [4.2](#s-list-procedures-and-function-bodies) |
| `procedure_assembly` | the assembly of one procedure as one string | | none | `10-mcp-06-procedure-assembly-grade.json` | [4.3](#s-procedure-assembly) |
| `xrefs` | the addresses that reference one address | | `rea xrefs` | `10-mcp-07-xrefs-grade.json` | [4.4](#s-xrefs-and-the-jump-table) |
| `list_strings` | every defined string with encoding, termination, and byte length | | none | `10-mcp-08-list-strings.json` | [4.5](#s-list-strings) |
| `analyze_function` | the dossier of one procedure | | `rea function` | `10-mcp-09` to `10-mcp-13`, `10-mcp-17`, `10-mcp-18` | [4.6](#s-analyze-function-and-pseudocode) |
| `export_evidence_bundle` | the path, byte count, record count, and unknown count of the bundle it wrote | destructive | none | `10-mcp-14`, `10-mcp-19` | [5.4](#s-evidence-bundles-export-and-import) |
| `import_evidence_bundle` | the records imported and the new total | | none | `10-mcp-23-import-evidence-bundle-O1.json` | [5.4](#s-evidence-bundles-export-and-import) |
| `compare_functions` | two dossiers in eight dimensions | | none | `10-mcp-20`, `10-mcp-21` | [5.5](#s-compare-and-compare-functions) |
| `verify_reconstruction` | pass, fail, or unknown for each declared claim | | none | `10-mcp-22-verify-reconstruction.json` | [7.1](#s-verify-reconstruction) |
| `close_binary` | `null`, after the provider and the ephemeral project are gone | destructive, not idempotent | none | `10-mcp-15`, `10-mcp-25` | [4.1](#s-open-binary-and-the-headless-bridge) |
| `inspect_wasm_artifact` | the WABT validation, sections, and interface of one WebAssembly module, as Evidence | open world | `rea inspect-wasm-artifact` | `10-mcp-24-inspect-wasm-artifact.json` | [6.2](#s-inspect-wasm-artifact) |

The hints column reads the four annotations of each tool in `tools/list`: `readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint`. An empty cell means read-only false, destructive false, idempotent true, and open world false, which is the shape of most analysis tools. "None" in the CLI column means the tool is one of the 65 that have no CLI command.

## Environment variables the capture set

| Variable | Effect at 6.5.0 | Documented in |
|---|---|---|
| `GHIDRA_INSTALL_DIR` | the Ghidra installation, an absolute path | {{install Ghidra}} |
| `JAVA_HOME` | the JDK, an absolute path | {{install Ghidra}} |
| `REA_GHIDRA_STARTUP_TIMEOUT_MS` | the startup deadline, 330,000 ms by default | {{mcp Ghidra first-query deadlines and recovery}} |
| `REA_ANALYSIS_PROVIDER` | the standing provider choice that `--provider` and `provider_id` override | {{cli Choose a provider}} |
| `REA_MCP_INPUT_SCHEMA_PROFILE` | `compact` renders every advertised input schema within 13,312 bytes | {{mcp Compact input schema profile}} |
| `REA_MCP_MAX_RESPONSE_BYTES` | the response budget, at least 10,485,760, read at startup | {{mcp Tool results}} |
| `REA_WABT_BIN_DIRECTORY` | the absolute directory that holds `wasm-validate`, `wasm-objdump`, and `wasm2wat` from WABT 1.0.42 | rea docs/wasm-artifacts.md, Offline WebAssembly artifacts |

The capture kit also removed `PWD` and `JAVA_TOOL_OPTIONS` from the environment of every rea process (capture/README.md). The paths in the records and the Java launch then stayed the same from run to run.

Sources: capture/out/00-commands.txt, 10-mcp-session.txt, 10-mcp-02-tools-list.json, 10-mcp-04-binary-session.json, 04-inspect-binary-layout.json, 16-inspect-wasm-artifact.json, 10-mcp-24-inspect-wasm-artifact.json, 00-versions.txt; capture/README.md; rea docs/cli.md (Native analysis, Choose a provider, Import, export and compare Evidence, Output and exit status), docs/mcp-contracts.md (Compact input schema profile, Tool results), docs/wasm-artifacts.md (Offline WebAssembly artifacts); rea at 3b79af2: src/cliCommandNames.ts, src/config/environment.ts, src/config/providerStartupTimeout.ts, src/contracts/toolEffects.ts, src/contracts/wasm/wasmToolContracts.ts
