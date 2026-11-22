# Hungry mode

Hungry mode is a bounded, opt-in curiosity loop. It grows the **retrieval
index**, not the model weights: unknown answers are stored as unverified
teacher notes and can be cited immediately, then reviewed like any other note.

It is deliberately not an autonomous web agent.

## What one run does

1. Collect candidate questions from:
   - pending knowledge gaps (unanswered user questions);
   - `--topic` seeds;
   - indexed document titles (used as topics for teacher-generated questions).
2. Check the local index for a strong match. A strong local answer marks the gap
   `known`; no teacher call is made.
3. For unknown questions, ask the configured OpenAI-compatible teacher
   (Ollama by default) with a strict short-answer prompt.
4. Store the answer through the normal teacher-note path:
   - `kind=teacher`
   - `verified=false`
   - source `teacher://<label>/<hash>`
5. Mark the gap `learned` and append a JSONL audit record.
6. Stop at the question/provider-call budgets. It does not run in the
   background.

## Safety boundaries

- **No web fetch.** Hungry mode never calls Wikipedia, a search engine or an
  arbitrary URL.
- **No weight changes.** `weights_changed: false` is always reported. A
  reviewed LoRA/full-training run is a separate, explicit project decision.
- **Unverified data.** Teacher answers are notes, not facts. Review them with
  the existing teacher-note verification flow before treating them as
  authoritative.
- **Hard budgets.** `--max-questions` and `--max-provider-calls` bound every
  run. The defaults are small.
- **No daemon.** Run it manually or schedule it explicitly. There is no hidden
  loop inside chat.
- **Local-first provider.** The default teacher URL is localhost Ollama. A
  remote URL is allowed only because the operator passes it explicitly.
- **Teacher uncertainty is skipped.** A bare “I do not know” is recorded as
  skipped instead of being stored as a note.

## Start the teacher

```powershell
ollama serve
ollama pull llama3.2:3b
```

Or let quickstart configure both the UI and the teacher:

```powershell
atomzero-quickstart --teacher
```

The server flags are:

```powershell
atomzero --model <checkpoint> --teacher-url http://127.0.0.1:11434/v1 `
  --teacher-model llama3.2:3b --teacher-label ollama
```

## CLI

```powershell
atomzero-hungry --docs data/atomzero_demo.db --max-questions 5
atomzero-hungry --topic Ankara --max-questions 5
atomzero-hungry --dry-run --json
atomzero-hungry --max-questions 10 --max-provider-calls 15 --expand
atomzero-hungry --ledger data/hungry_ledger.jsonl
```

Useful flags:

| flag | meaning |
|---|---|
| `--topic TEXT` | seed topic; repeatable |
| `--max-questions N` | attempted questions, default 5, max 50 |
| `--max-provider-calls N` | hard provider budget, default 8, max 60 |
| `--question-limit N` | teacher-generated questions per topic, default 3 |
| `--min-confidence F` | local match threshold, default 0.55 |
| `--expand` | ask for follow-up questions after a learned answer |
| `--dry-run` | never call the teacher; only show unknown questions |
| `--json` | machine-readable summary |
| `--ledger PATH` | JSONL audit path; empty string disables |

The default local index is `data/atomzero_demo.db`; point `--docs` at the
database used by your server.

## Example output

```text
Hungry mode: 2 learned, 1 known, 0 skipped, 0 failed
provider calls: 2 | weights changed: False | elapsed: 6.4s
- learned: What is the capital of France? -> teacher://ollama/...
- known: What is the launch code? -> launch_policy.txt
ledger: data/hungry_ledger.jsonl
```

## HTTP and UI

```text
POST /v1/hungry/run
{"topic": "Ankara", "max_questions": 3, "max_provider_calls": 6}
{"dry_run": true, "max_questions": 5}
```

The response contains `learned`, `known`, `skipped`, `failed`,
`provider_calls`, `weights_changed`, `results` and the ledger path.

The local UI has a **Hungry mode** section with a question count, an optional
topic, a dry-run checkbox and a result list. The teacher must be configured with
`--teacher`/`--teacher-url`; otherwise the UI offers dry-run mode only.

## Gap and ledger states

Knowledge gaps move from `pending` to one of:

- `known` — an existing local source answered it;
- `learned` — a teacher note was stored and can be cited;
- `proposed`/`dismissed` — the existing reviewed research path.

The ledger is append-only JSONL. Each record contains the timestamp, question,
origin, status, source, confidence score, gap id and, for learned answers, a
short answer preview. Treat the ledger and notes as private local data.

## Limits

- Hungry mode grows retrieval memory, not the 49.2M weights.
- Teacher answers can be wrong or outdated; verification is manual.
- The confidence gate can classify a weak local match as unknown.
- It does not invent new topics beyond the seeds and teacher follow-ups.
- It does not fetch sources, so answers have no external URL unless the teacher
  provides one in its text (which is not treated as a verified citation).
