# Branch research - beyond AISC

AISC is one branch of the work, not the boundary. The planner scans adjacent
research and local product gaps, then keeps only slices that fit Atom Zero's
constraints: local, deterministic by default, no model calls in the core path,
no silent assurance claims. This note records the scan and the branch map.

## What was reviewed

1. **Claim-level verification.** MedRAGChecker (2026) decomposes an answer into
   atomic claims and combines evidence-grounded NLI with knowledge-graph
   consistency; Claimify (ACL 2025) focuses on high-quality claim extraction.
   Atom Zero's `claims.py` is the deterministic first slice: sentence claims,
   lexical evidence linking and transparent number/date/code/negation checks.
   It is not an NLI model and does not claim entailment.
2. **Source reliability.** RA-RAG (EMNLP 2025) estimates source reliability by
   cross-checking documents; CONFACT (IJCAI 2025) finds automated credibility
   estimation noisy and expert labels stronger. The reliability ledger therefore
   stores content-free observation counts and a local support ratio, not a
   truth score or a web credibility judgment.
3. **Temporal RAG.** 2025 work on temporal layers treats time as a first-class
   retrieval signal: validity intervals, freshness windows, half-life decay and
   as-of queries. The freshness branch is the first slice: extract dates from
   hit text, classify fresh/aging/stale/future/undated, and report the oldest
   and newest evidence. It does not filter retrieval yet.
4. **Consensus and conflicting evidence.** Recent fact-checking work separates
   source-level disagreement from aggregation and warns that agreement is only
   meaningful when sources were checked independently. The consensus branch
   keeps conflicts separate, labels single-source clusters, and never averages
   away a numeric contradiction.
5. **Evidence sufficiency and adaptive retrieval.** AR-Verify (2026) combines
   relevance, completeness, consistency and source reliability to decide whether
   to retrieve more. Atom Zero should expose this as an audit/plan first, not as
   an automatic retriever or an enabled gate.

## Branch map

| Branch | What it is | Status | Honest limit |
|---|---|---|---|
| Tool Plan Algebra | side effects, irreversible steps, confirmation | shipped `0.6.0` | policy audit, not an executor sandbox |
| Context Budget Algebra | bounded evidence and dropped-hit reporting | shipped `0.6.0` | token planning, not semantic compression |
| Proactive Conversation | opt-in local check-ins from tasks/gaps/sources | shipped `0.6.0` | not a background daemon |
| Memory Lifecycle | confirmation + content-free delete ledger | shipped `0.6.1` | local session scope, not remote sync |
| Answerability | measured refusal/coverage experiment | shipped, gate off | targets still missed |
| Claim / Evidence | sentence claims, evidence links, conflicts | shipped `0.6.1` | lexical, not NLI |
| AISC Spec Generator | evidence-to-TOML, drift check | shipped `0.6.1` | trusts the stored profile |
| Cross-source Consensus | lexical clustering, single-source labels, conflicts | local `0.6.2` | agreement is not truth or independence |
| Freshness | date extraction and fresh/aging/stale classification | in progress | dates in text, not file mtime |
| Source Reliability Ledger | hash-only observation counts and support ratio | in progress | small-sample and lexical-audit bias |
| AISC Batch Baseline | regression comparison for batch runs | in progress | compares reports, not behavior |
| Temporal Validity | as-of filtering and validity intervals | candidate | needs dated metadata, not just prose |
| Evidence Sufficiency | relevance/completeness/consistency/reliability audit | candidate | audit only at first, no auto gate |
| Document Version Diff | detect superseded document versions | candidate | local metadata, not web freshness |
| Claim-aware Proactive | start a check-in from unsupported/single-source claims | candidate | opt-in, rate-limited, local only |
| AISC Coverage Report | which critical modules have specs | candidate | coverage is not assurance |

## What we deliberately do not do

- Do not import a large NLI/LLM verifier into the default path. If one is added
  later, it must be optional and measured separately.
- Do not call automated source credibility a truth verdict; it is a local
  observation summary with visible sample size.
- Do not average conflicts into consensus.
- Do not enable answerability/refusal gates that missed their targets.
- Do not turn AISC spec coverage into an assurance claim.

## References

- MedRAGChecker, claim-level verification for biomedical RAG, arXiv 2601.06519.
- Claimify, Microsoft Research / ACL 2025.
- RA-RAG, Retrieval-Augmented Generation with Estimation of Source Reliability,
  EMNLP 2025.
- CONFACT, Resolving Conflicting Evidence in Automated Fact-Checking, IJCAI
  2025.
- Temporal RAG / freshness layers, 2025 (validity filtering, decay, as-of
  retrieval).
- AR-Verify, adaptive retrieval with autonomous evidence verification, 2026.
- Dual-perspective multi-source claim verification, arXiv 2602.18693.
