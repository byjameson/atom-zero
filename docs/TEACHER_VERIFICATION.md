# Teacher answer verification

**Status:** deterministic evidence step. It does not train an adapter and does
not change weights.

Atom Zero may ask an opt-in teacher through `TeacherProvider`, and `hungry.py`
stores the answer as an **unverified teacher note**. Before such a note can
become retrieval memory or an adapter candidate, its claims must be checked
against local evidence. `atomzero/teacher_verification.py` performs that check
and returns a small, content-free result.

```text
teacher answer + local hits
  -> sentence-level claims
  -> evidence linking at min_support
  -> numeric/date/identifier/negation conflict check
  -> content-free status: empty | conflicted | unverified | verified
  -> human promotion decision (never automatic)
  -> unverified note becomes reviewed memory / adapter candidate
```

The verifier is a gate, not a promotion. Even a `verified` result only says
that the local evidence supports the answer's claims under the deterministic
lexical rules. A human must still approve storing the note.

## Pipeline

1. **Claim extraction.** `extract_claims` splits the teacher answer into
   sentence-level claims. Empty fragments and fragments shorter than 12
   characters or 2 words are ignored. If nothing remains, the answer cannot be
   verified.
2. **Evidence linking.** Every claim is scored against every local hit with
   `link_evidence(..., min_coverage=min_support)`. The score blends claim-term
   coverage, exact token overlap and, when present, identifier coverage. The
   best hit wins, ties keep the earlier hit. Claims are marked `supported`,
   `weak` or `unsupported` exactly as in `docs/CLAIMS.md`.
3. **Conflict check.** `compare_claims` compares claims that share enough
   terms or entities and reports `numeric_mismatch`, `date_mismatch`,
   `identifier_mismatch` and `negation_mismatch`. This is phrase-level
   normalisation, not natural-language inference.
4. **Status decision.** The rules below are evaluated in order; the first
   match wins.
5. **Content-free result.** The output contains only status, reason, counts
   and fixed flags. It contains no answer text, claim text, evidence snippets
   or source paths.

## Status table

| # | Condition | Status | `reason_code` | `verified` | `requires_human` |
|---|---|---|---|---|---|
| 1 | no claims extracted | `empty` | `no_claims` | false | false |
| 2 | at least one claim conflict | `conflicted` | `contradiction` | false | false |
| 3 | any claim is `weak` or `unsupported` | `unverified` | `insufficient_support` | false | false |
| 4 | sources required and no evidence source exists | `unverified` | `no_sources` | false | false |
| 5 | all claims supported and source gate passed | `verified` | `all_claims_supported` | true | true |

A conflict is checked before support: a fluent answer can look well-supported
and still contradict itself. An empty answer is not treated as verified, even
if hits were supplied.

`requires_human` is true only on a `verified` result, because that is the only
status with a pending promotion. A non-verified result has nothing to promote;
a human may still need to fix the answer or improve the retrieval evidence.

## API

```python
from atomzero.teacher_verification import verify_answer, verification_event

result = verify_answer(
    answer,
    hits,
    min_support=0.60,
    require_sources=True,
)

event = verification_event(result)  # content-free growth event
```

`verify_answer` validates its inputs:

- `answer` must be a non-empty string;
- `hits` must be a list of mappings (normal retrieval hits with a `chunk`
  and optional `source`, `title`, `section` and `page`);
- `min_support` must be a finite number from 0 to 1;
- `require_sources` must be a boolean.

Invalid input raises `ValueError`. The function does not mutate the answer or
the hits and is deterministic for a given input. The output method is
`deterministic_teacher_verification_v1`.

### Content-free result

```json
{
  "status": "verified",
  "reason_code": "all_claims_supported",
  "verified": true,
  "claim_count": 2,
  "supported_count": 2,
  "weak_count": 0,
  "unsupported_count": 0,
  "conflict_count": 0,
  "source_count": 2,
  "min_support": 0.6,
  "requires_human": true,
  "auto_apply": false,
  "weights_changed": false,
  "method": "deterministic_teacher_verification_v1"
}
```

`source_count` counts distinct non-empty evidence source identifiers. The
identifiers themselves are not returned. `auto_apply` and `weights_changed`
are always false.

## Verification event

`verification_event(result, *, outcome="success")` converts a result into a
content-free record suitable for a growth ledger:

```json
{
  "version": 1,
  "status": "verified",
  "reason_code": "all_claims_supported",
  "outcome": "success",
  "verified": true,
  "promotable": true,
  "requires_human": true,
  "auto_apply": false,
  "weights_changed": false,
  "growth_event": "verified_teacher",
  "claim_count": 2,
  "supported_count": 2,
  "weak_count": 0,
  "unsupported_count": 0,
  "conflict_count": 0,
  "source_count": 2,
  "method": "deterministic_teacher_verification_v1"
}
```

`promotable` is true only for a successful, verified result. It means "a human
may now consider promoting this note"; it does not promote anything.

The event is **standalone**, not directly appendable to the escalation growth
ledger. The escalation validator (`docs/ESCALATION.md`) accepts only routing
records with a timestamp and a routing reason, and rejects the `status` and
count fields that make verification auditable. The event deliberately reuses
the `verified` / `verified_teacher` vocabulary so a caller can map a
promotable event to `record_escalation` after the human decision.

## Human promotion step

A verified result is evidence, not a decision. The promotion path is explicit:

```text
atomzero-verify-teacher -> verified result with requires_human=true
  -> human reviews the answer, the local sources and the conflicts
  -> human stores/accepts the reviewed note
  -> a later, separate step may count it as an adapter candidate
  -> offline training and evaluation remain separate jobs
```

There is no automatic call from verification to `teacher_notes.py`,
`hungry.py`, retrieval memory, an adapter export, a trainer or an agent. The
module has no store and no network code; the CLI does not even import the
teacher provider.

## CLI

```powershell
atomzero-verify-teacher --answer-json answer.json --json
atomzero-verify-teacher --answer-json -
```

`answer.json` is a JSON object with exactly `answer` and `hits`:

```json
{
  "answer": "The Skyline battery lasts 8 hours.",
  "hits": [
    {
      "source": "battery.md",
      "title": "Battery",
      "section": "Runtime",
      "chunk": "Under normal load the Skyline battery lasts 8 hours."
    }
  ]
}
```

| Flag | Meaning |
|---|---|
| `--answer-json PATH` | input object; `-` reads stdin |
| `--json` | print the content-free result as JSON |

Exit code `0` means a valid verification result, whether or not it is verified.
Exit code `2` means missing input, an unreadable file, invalid JSON, an invalid
object, or malformed `answer`/`hits`. The CLI never calls a teacher, network,
store or agent, never stores the answer and never changes weights.

## How it fits the snowball ladder

The ladder is described in `docs/GROWTH_LADDER.md`. Verification is layer 2:

| Layer | Role of this module |
|---|---|
| 1. Teacher escalation | the answer arrives; it is still untrusted |
| 2. Verified memory | this module supplies the deterministic evidence gate; a human still approves the note |
| 3. Skills and recipes | verified traces may later inform reviewed procedures |
| 4. Distilled adapter | a verified teacher event may become an **adapter candidate**, never a training run |
| 5. Checkpoint upcycling | unaffected; a separate offline, human-approved job |

This module does not train an adapter, does not create an adapter, does not
download a model, does not write retrieval memory and does not alter any
weights. It only says whether the local evidence supports the teacher's claims
under a transparent lexical rule.

## Limitations

- The score is lexical. A correct paraphrase can be unsupported, and copied
  wording can be supported even when a number is wrong. Use it as an audit
  signal, not a truth verdict.
- Conflicts are detected only between claims that share enough terms or
  entities. Missing context or a single isolated claim will not be flagged.
- Negation is detected only through explicit marker words; sentence splitting
  is regex-based.
- An answer that is `verified` can still be wrong, stale or unlicensed.
  `verified` means "the supplied local evidence supports these claims under
  this method", not "the answer is true".
- The verifier cannot prove that a hit is authoritative, fresh or correctly
  retrieved. Source quality remains a human and retrieval concern.
- Nothing here is a substitute for the human promotion decision, the adapter
  evaluation gate or the release gate.
