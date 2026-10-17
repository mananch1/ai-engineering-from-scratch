# Evidence fields

> The 16 fields of every Evidence record at 6.5.0, with their types, their allowed values, and one value each from the captured dossier of _percent_change.

The Evidence schema is one strict Zod object with 16 fields. The schema rejects a record with an unknown field. `relationship_coverage` is the sixteenth field. It defaults to `null`, and a `null` value does not enter the id hash.

The dossier column takes its values from the `analyze_function` record of `_percent_change` in `ledger-arm64-O1`, `ev_172d39ff` (capture/out/10-mcp-09-analyze-function-percent-change-O1.json). The derived column takes them from the `verify_reconstruction` record `ev_0f0a4142` (capture/out/10-mcp-22-verify-reconstruction.json). [Section 5.1](#s-the-evidence-record) reads the record in prose.

## The 16 fields

| Field | Type | In the dossier of _percent_change | In the verification record |
|---|---|---|---|
| `evidence_id` | `ev_` and 64 hex digits, the SHA-256 of the semantic projection | `ev_172d39ff37ab60d9e0e1dd33ba7486cb2ee789a46b00d600cd083a825edd1350` | `ev_0f0a4142adb15fdaad7601fc4b8a55faa132c8bb44adc9135fcf1cee4c9fb333` |
| `subject` | the analyzed file, `{ name, digest.sha256, format, architecture, local_path }`, or `null` | `ledger-arm64-O1`, `01b5d08b…`, `mach-o`, `arm64`, `$CAPTURE/fixtures/ledger-arm64-O1` | `null` |
| `provider` | `{ id, name, version }`, version nullable | `ghidra`, `Ghidra`, `12.1.4` | `rea-reconstruction-verifier`, `REA reconstruction verifier`, `1` |
| `analysis_profile` | `{ provider, parameters, digest }` or `null`, always present | 25 parameters, digest `966e6016…` | `null` |
| `predicate_type` | a non-empty string | `rea.analysis` | `rea.reconstruction-verification` |
| `operation` | the tool name | `analyze_function` | `verify_reconstruction` |
| `parameters` | the caller's input object | `{ "procedure": "_percent_change" }` | `specification_sha256` and four `claim_ids` |
| `raw_result` | a provider-native value or `null` | the bridge value with `procedure`, `pseudocode`, and the rest | `null` |
| `normalized_result` | the canonical result the reader uses | 16 keys from `procedure` to `limitations` | `status`, `specification_sha256`, `summary`, `claims`, `recommended_probes`, `evidence_links`, `limitations` |
| `confidence` | `observed`, `derived`, or `inferred` | `observed` | `derived` |
| `authority` | one of five values, listed below | `shipped-artifact` | `analyst-inference` |
| `environment` | `{ id, platform, architecture, isolation }` or `null` | `null` | `null` |
| `limitations` | an array of strings | ten strings, from auto-analysis to the macOS decompiler | two strings, artifact identity and the pass rule |
| `relationship_coverage` | `{ status, ... }` with status `complete`, `partial`, or `unknown`, or `null` | `complete`, with one diagnostic about resolved call references | `null` |
| `locations` | an array of five location shapes, listed below | `[]` | `[]` |
| `evidence_links` | `ev_` ids this record was derived from | `[]` | six ids: four dossiers and two comparisons |

The `normalized_result` of a dossier has the keys `procedure`, `pseudocode`, `assembly`, `comments`, `callers`, `callees`, `incoming_references`, `outgoing_references`, `unresolved_calls`, `referenced_strings`, `referenced_names`, `basic_blocks`, `native_api`, `native_value_flow`, `facet_availability`, and `limitations`. `facet_availability` tells which of eight facets the provider supplied. Those inner limitations describe the facets, and the record's own `limitations` describe the provider session.

## The enumerations

| Field | Allowed values |
| --- | --- |
| `confidence` | `observed`, `derived`, `inferred` |
| `authority` | `shipped-artifact`, `controlled-replay`, `historical-reference`, `external-service`, `analyst-inference` |
| `environment.isolation` | `none`, `process`, `container`, `virtual-machine` |
| `locations[].kind` | `address` with `address`, `address-range` with `start` and `end`, `artifact-path` with `path`, `file-offset` with `offset`, `file-offset-range` with `start` and `end` |
| `subject.format` | `hopper`, `analysis-database`, `mach-o`, `elf`, `pe`, `dos-mz`, `dos-com`, `raw-binary`, `zip`, `ipa`, `apk`, `msix`, `appx`, `hap`, `hsp`, `app-pack`, `asar`, `dmg`, `pkg`, `plist`, `javascript`, `source-map`, `directory`, `file`, `unknown`, `mach-o-universal`, `javascript-bundle`, `entitlements` |
| `subject.architecture` | `x86`, `x86_64`, `arm`, `arm64`, `mips`, `riscv`, `powerpc`, or `null`, and `6502` for a `raw-binary` subject only |

The format owns the architecture claim. A `raw-binary` subject can have any architecture, `6502` included. Every other format excludes `6502`.

## The id

The id hashes the semantic projection of the record. That projection keeps the subject's `digest`, `format`, and `architecture` and every other field except `evidence_id`, and leaves out the subject's `name` and `local_path`. The same observation of the same bytes therefore gets the same id in any directory on any machine. The capture check compares ids unmasked (capture/README.md). `parseEvidence` recomputes the hash of every record it reads and rejects a mismatch. A record without a subject gets the limitation "Artifact identity is unavailable for this observation." appended at creation.

The projection leaves out `relationship_coverage` when it is `null`, so records without coverage keep their ids.

Two other id prefixes appear in the captures. `unk_` ids name residual unknowns and hash `{ domain, question, scope_digest }`, and `comparison_conclusion_scope` when the unknown has one ([Section 5.2](#s-residual-unknowns-and-coverage)). The `analysis_profile.digest` is the SHA-256 of the canonical JSON of the provider and its parameters. Two records with different digests are not comparable in `compare_functions`.

Sources: capture/out/10-mcp-09-analyze-function-percent-change-O1.json, 10-mcp-22-verify-reconstruction.json; capture/README.md; rea at 3b79af2: src/domain/evidence.ts, src/domain/relationshipCoverage.ts, src/domain/binaryTargetTypes.ts, src/domain/residualUnknown.ts, src/domain/analysisProfile.ts
