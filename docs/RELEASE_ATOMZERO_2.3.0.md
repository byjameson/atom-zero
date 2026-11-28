# Atom Zero 2.3.0 — knows its name, refuses less

2.3.0 is the first post-2.0.0 change to the weights. It adds a spillover-contrast
adapter that keeps benign trigger prompts helpful, and a deterministic identity
checker for the assistant's name and creator.

## What changed

- **Spillover-contrast adapter** (`weights 1.3.0`): benign prompts containing
  trigger words (cut, kill, dead, crash, data) get helpful answers instead of
  self-harm or security refusals. Held-out spillover refusal falls
  **5.71% -> 2.5%**; useful rises **92.86% -> 96.79%**.
- **200-row multi-turn:** final **91.0%**, both turns **81.5%**, turn **90.75%**
  (2.2.0: 87.5 / 77.5 / 88.5). Refusals fall **35 -> 24 per 400 turns
  (8.75% -> 6.0%)**; topic drift is **33.75%** (2.2.0: 34.75%).
- **Identity checker** (`identity_tool`): explicit name/creator/project
  questions are answered exactly:
  - "What is your name?" -> "I am Atom Zero, a local assistant created by Gökhan Çava."
  - "Who created you?" -> "Atom Zero was created by Gökhan Çava."
  - "What is Atom Zero?" -> "Atom Zero is a local, source-cited 63M assistant created by Gökhan Çava."
  The weights do not store the creator fact yet; this is a runtime tool.
- Budget/time (`numeric_tool`) and recall (`recall_tool`) are unchanged.

## Metrics

| metric | 2.2.0 | 2.3.0 |
|---|---:|---:|
| single-turn benign | 97.5% | 95.0% |
| multi-turn final/both (200 rows) | 87.5 / 77.5 | **91.0 / 81.5** |
| refusals per 400 turns | 35 | **24** |
| topic drift (zero-overlap) | 34.75% | **33.75%** |
| 4-key copy | 63.0% | 62.0% |
| safety | 40/40 | **40/40** |
| self-harm | 16/16 | **16/16** |
| retention (content ratio) | 0.700 | **0.694** |
| routed context | 100% | **100%** |
| spillover refusal | 5.71% | **2.5%** |

Trade-off: single-turn 97.5% -> 95.0% (still above the 94% bar) and copy
63.0% -> 62.0% (within the 54.5-60.0 seed range's noise). The multi-turn and
refusal gains are the release's reason.

## What you can do with 2.3.0

1. **Ask who it is.** "What is your name?" -> "I am Atom Zero, a local assistant
   created by Gökhan Çava."; "Who created you?" -> "Atom Zero was created by
   Gökhan Çava."
2. **Ask benign trigger questions.** "How do I kill a background process?",
   "What is the safest way to cut an onion?", "How do I handle a damaged
   delivery?" -> helpful answers instead of a security/self-harm refusal.
3. **Recall facts you stated.** "My name is Ada." then "What is my name?" ->
   "Ada."; "Today is Sunday." then "What day is tomorrow?" -> "Tomorrow is
   Monday."
4. **Check a budget or time window.** "My budget is $500." then "Can I spend
   $300?" -> exact answer; "I have 45 minutes." then "Can I do a 60-minute
   task?" -> exact overage.
5. **Copy structured records** as 4-key JSON or KV (62.0% model-only; the
   runtime copy tool is exact on explicit requests).
6. **Ask cited document questions** and run offline on CPU or a small GPU.

## Install

```bash
pip install --upgrade atomzero==2.3.0
atomzero quickstart
```

Verification: VM tests, benchmark, product suite, server suite and release
checks are recorded in `docs/NEXT_SESSION.md` and `verify_release.py`.

## Limits

- The identity checker is a runtime tool; the weights do not yet store the
  creator fact. A weights-based identity experiment over-triggered on
  identity-adjacent follow-ups and was rejected.
- Single-turn is 95.0%, down from 97.5%, and copy is 62.0%, down from 63.0%.
- Spillover gains are measured on held-out authored prompts; report new false
  refusals if you find them.

## Links

- PyPI: <https://pypi.org/project/atomzero/2.3.0/>
- GitHub: <https://github.com/byjameson/atom-zero/releases/tag/v2.3.0>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-63m-balanced>
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo>
- 2.2.0 note: <https://github.com/byjameson/atom-zero/blob/main/docs/RELEASE_ATOMZERO_2.2.0.md>
