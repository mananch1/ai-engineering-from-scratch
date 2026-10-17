# The Evidence record

> Every successful rea observation is one strict record of sixteen fields, and its id is a SHA-256 over the semantic content with the file path left outside.

An agent tells you that `_percent_change` returns 0 when `before` is 0. You want to know what rea observed, which engine said it, and whether the same claim from another machine would carry the same id. The capture holds the answer in `capture/out/10-mcp-09-analyze-function-percent-change-O1.json`: one Evidence record. When you finish this section, you can read all sixteen fields of an Evidence record and say which three carry the trust decision. You can also recompute an `ev_` id from a captured record.

## One record for every successful observation

**Evidence record:** the one record type that rea returns for a successful observation. The schema is one strict object with sixteen fields. A record with an extra field is rejected. A record that omits `relationship_coverage` gets `null`, so a record written before that field existed still reads. A record that omits any other field is rejected. The MCP server returns the record as `structuredContent` and as a text copy: "Evidence-producing tools return the complete canonical Evidence record in both text and structured content." {{mcp Tool results}}

```listing
title: the sixteen fields of the _percent_change dossier record
source: capture/out/10-mcp-09-analyze-function-percent-change-O1.json
lang: json
note: The profile parameters, the raw result, the normalized result, nine of the ten limitations, and the coverage diagnostics are cut. The kit replaced the text copy of the record with a marker.
---
"evidence_id": "ev_172d39ff37ab60d9e0e1dd33ba7486cb2ee789a46b00d600cd083a825edd1350",
"subject": {
  "name": "ledger-arm64-O1",
  "digest": {
    "sha256": "01b5d08b08fb94678be5919d0e753d44b505a53107c95b94634f0e5d41c9e375"
  },
  "format": "mach-o",
  "architecture": "arm64",
  "local_path": "$CAPTURE/fixtures/ledger-arm64-O1"
},
"provider": {
  "id": "ghidra",
  "name": "Ghidra",
  "version": "12.1.4"
},
"analysis_profile": {
  …
  "digest": "966e6016724421ad282356f583a9d5bf24577db4ac77a75c7099d236b88bcc28"
},
"predicate_type": "rea.analysis",
"operation": "analyze_function",
"parameters": {
  "procedure": "_percent_change"
},
"raw_result": {
  …
},
"normalized_result": {
  …
},
"confidence": "observed",
"authority": "shipped-artifact",
"environment": null,
"limitations": [
  "The session serves operations only after Ghidra reports default auto-analysis complete; incomplete analysis does not expose partial results.",
  …
],
"relationship_coverage": {
  "status": "complete",
  …
},
"locations": [],
"evidence_links": []
```

| Field | In this record | What it holds |
|---|---|---|
| `evidence_id` | `ev_172d39ff37ab` and 52 more hex digits | `ev_` and a SHA-256 |
| `subject` | `ledger-arm64-O1`, sha256 `01b5d08b08fb`, `mach-o`, `arm64`, a path | the analysed file, or `null` |
| `provider` | `ghidra`, Ghidra, `12.1.4` | the engine that produced the fact |
| `analysis_profile` | digest `966e6016724` | the committed engine settings, or `null` |
| `predicate_type` | `rea.analysis` | the kind of claim |
| `operation` | `analyze_function` | the tool that produced the record |
| `parameters` | `{"procedure": "_percent_change"}` | the caller input, as given |
| `raw_result` | the Ghidra report | the provider form, or `null` |
| `normalized_result` | the dossier, 16 keys | the canonical result the agent reads |
| `confidence` | `observed` | `observed`, `derived`, or `inferred` |
| `authority` | `shipped-artifact` | one of five source kinds |
| `environment` | `null` | platform and isolation of a run, or `null` |
| `limitations` | 10 strings | what the fact does not cover |
| `relationship_coverage` | `complete` | whether the caller and callee sets are complete, partial, or unknown, or `null` for other operations |
| `locations` | `[]` | addresses, ranges, paths, or file offsets |
| `evidence_links` | `[]` | the `ev_` ids this record derives from |

The subject `format` is one of 28 container names, from `mach-o` and `elf` to `javascript` and `directory`. The architecture is one of `x86`, `x86_64`, `arm`, `arm64`, `mips`, `6502`, `riscv`, and `powerpc`. Only the `raw-binary` format can have the architecture `6502`. A location has one of five shapes: `address`, `address-range`, `artifact-path`, `file-offset`, or `file-offset-range`.

The MCP handler can run a preflight check before the engine starts. It builds the record after the engine answers. It calls `createEvidence` with the active target, the provider, and the operation. The same call passes the parameters, the result, the profile, the raw result, the limitations, the locations, and the links.

Some tools give a reduced parameter set to the record. The handler records any source records first, then this record, then returns it. Inside a Ghidra session, the binary session makes the same call.

Three defaults fill in what a producer leaves out. When the producer names no confidence or authority, the record gets `observed` and `shipped-artifact`. The subject name is the last segment of the path. When there is no target at all, the record gets the limitation "Artifact identity is unavailable for this observation.".

## Confidence and authority

**Confidence:** one of three words, `observed`, `derived`, or `inferred`. It is a category, never a number. **Authority:** the kind of source behind the fact, one of `shipped-artifact`, `controlled-replay`, `historical-reference`, `external-service`, or `analyst-inference`. The two fields are set by the producer, and the captures show four producers.

| Producer | `confidence` | `authority` | Record |
|---|---|---|---|
| `analyze_function`, provider `ghidra` | `observed` | `shipped-artifact` | `10-mcp-09-analyze-function-percent-change-O1.json` |
| `binary_overview`, provider `rea-workflow` | `derived` | `shipped-artifact` | `05-analyze-arm64-O1.json` |
| `compare_functions`, provider `rea-function-comparison` | `derived` | `analyst-inference` | `10-mcp-20-compare-functions-percent-change.json` |
| `inspect_analysis_view`, provider `rea-analysis-view` | `derived` | `shipped-artifact` | `15-js-app-view-summary.json` |
| `record_unknown`, provider `rea-unknown-registry` | `derived` | `analyst-inference` | none in the capture |

Read `observed` with care. In the dossier it means that rea recorded what Ghidra returned, with the bytes on disk as the authority. It does not mean that the pseudocode is the program's source. The provider documentation says so: "Assembly is complete Ghidra Listing text; pseudocode is Ghidra decompiler output." {{ghidra Admitted function-analysis semantics}}

The `binary_overview` record is `derived`, because a rea workflow computed it from provider calls. Its first limitation says so: "Derived by an REA composed workflow." The other ten are the Ghidra limitations it inherits (`05-analyze-arm64-O1.json`). [Lesson 35](phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns) builds a ledger in plain Python with the same three words and the same five authorities.

## Limitations, locations, and links

`limitations` is an array of plain strings, and a producer appends its own to the ones it inherits. The comparison record shows both kinds. Its first entry is the automatic one for a record without a subject. The next 26 are the dossier limitations of both sides, each prefixed with `Left:` or `Right:`.

```listing
title: the trust fields of the compare_functions record
source: capture/out/10-mcp-20-compare-functions-percent-change.json
lang: json
note: Of the 27 limitations, the first two are shown.
---
"confidence": "derived",
"authority": "analyst-inference",
"environment": null,
"limitations": [
  "Artifact identity is unavailable for this observation.",
  "Left: Cancelling an active Ghidra request terminates its ephemeral database and discards session annotations. The selected target remains open; the next provider query imports the original artifact again.",
  …
],
"locations": [],
"evidence_links": [
  "ev_172d39ff37ab60d9e0e1dd33ba7486cb2ee789a46b00d600cd083a825edd1350",
  "ev_1f3bca22a69485fa2eaf7bdd59847e92fb77517bac6c01e657e143c14870013f"
]
```

`evidence_links` names the records a derived record was computed from. The comparison links the two dossiers it compared, and a view links its parent record. `locations` says where in the artifact the fact was seen. The JavaScript record in `15-js-app.json` carries one `artifact-path` location, `artifact-root`. The Ghidra dossier carries none, because its addresses live inside `normalized_result`.

`analysis_profile` is part of the identity even when it is empty: "`analysis_profile` is always present: a concrete profile object or `null`; either value participates in semantic identity." {{mcp Tool results}}

A profile whose provider differs from the record's `provider` is rejected with "Analysis profile provider does not match Evidence provider". The profile digest is a SHA-256 over the canonical JSON of its provider and parameters. [The snapshot cache](#s-snapshots-and-the-answer-cache) uses the same digest as part of its key.

## The id is a hash of the semantic content

**Semantic projection:** the view of the record that the id hashes. It keeps `subject.digest`, `subject.format`, and `subject.architecture`, and it drops `subject.name` and `subject.local_path`. It also drops `relationship_coverage` when its value is `null`, so a record without coverage keeps the id it had before that field existed. Every other field enters as it is.

The id is `ev_` followed by the SHA-256 of that projection in canonical JSON. Object keys are sorted as RFC 8785 requires. `parseEvidence` recomputes the hash on every record it reads and rejects a mismatch with "Evidence semantic identifier does not match its record".

You can do the same check with a few lines of Node. The script below sorts keys, serializes without spaces, drops the two display fields, drops a `null` coverage field, and hashes.

```javascript
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const canonical = (value) => {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
};

const file = JSON.parse(readFileSync(process.argv[2], "utf8"));
const record = file.response ? file.response.result.structuredContent : file;
const { evidence_id, subject, relationship_coverage, ...rest } = record;
if (relationship_coverage !== null && relationship_coverage !== undefined) rest.relationship_coverage = relationship_coverage;
const projection = {
  subject: subject === null ? null : { digest: subject.digest, format: subject.format, architecture: subject.architecture },
  ...rest,
};
const digest = createHash("sha256").update(canonical(projection)).digest("hex");
console.log(evidence_id, evidence_id === `ev_${digest}` ? "match" : "mismatch");
```

Run on `05-analyze-arm64-O1.json`, it prints `ev_90e65378a5b1d958b02840bda801e5758d39ceb43acd0f4ed2d1a3ad7297588d match`, the id on line 2 of that file. Run on `10-mcp-09-analyze-function-percent-change-O1.json`, it matches `ev_172d39ff37ab60d9e0e1dd33ba7486cb2ee789a46b00d600cd083a825edd1350`. It also matches the comparison record in `10-mcp-20` and the first record inside `11-bundle-O1.json`. The masked `$CAPTURE` path in `subject.local_path` changes nothing, because that field is outside the hash.

Two files in the capture do not match, and both teach the same rule. `15-js-app.json` fails because the kit cut `graph` and `semantic_graph` out of `normalized_result` with `--filter-output`. `15-js-app-view-summary.json` fails because the kit masked the capture directory as `$CAPTURE` inside `normalized_result.summary.input_path` and inside `locations`. A path in `subject` is outside the hash. A path anywhere else is inside it, and a record that carries one gets a different id on every machine.

```figure
id: fig-5-1
kind: structure
title: the sixteen fields of one Evidence record
claim: The id hashes every field of the record except subject.name and subject.local_path, so the same observation of the same bytes gets the same id anywhere.
caption: Read top to bottom. Solid rows enter the hash and dashed rows are display fields. Values are from capture/out/10-mcp-09-analyze-function-percent-change-O1.json, digests shown by their first hex digits.
```

```takeaways
- Read `confidence`, `authority`, and `limitations` before `normalized_result`, and copy all three into your notes.
- Treat `observed` as "rea recorded the engine output", never as "the source was recovered".
- Recompute the `ev_` id of every record you receive from another machine.
- Keep paths out of `normalized_result` and `locations` in your own records, or the id changes per machine.
```

Sources: capture/out/10-mcp-09-analyze-function-percent-change-O1.json, capture/out/10-mcp-20-compare-functions-percent-change.json, capture/out/05-analyze-arm64-O1.json, capture/out/15-js-app.json, capture/out/15-js-app-view-summary.json, capture/out/11-bundle-O1.json; phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns; rea 6.5.0 at 3b79af2: src/domain/evidence.ts, src/domain/relationshipCoverage.ts, src/domain/canonicalDigest.ts, src/domain/analysisProfile.ts, src/domain/binaryTargetTypes.ts, src/server/registerEvidenceTools.ts, src/application/binary/BinarySession.ts, src/application/investigation/UnknownEvidence.ts; rea docs/mcp-contracts.md, rea docs/ghidra-provider.md
