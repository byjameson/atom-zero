# Atom Zero 2.4.0 — single-turn recovered, multi-turn improved

2.4.0 keeps the 2.3.0 spillover adapter and adds a targeted greeting/thanks/
calendar fix. The single-turn cost of 2.3.0 is gone, and the multi-turn and
refusal profile improves again.

## What changed

- **Greeting/thanks/calendar fix** (`weights 1.4.0`): a small adapter trained on
  new prompts (not the evaluation set) turns two-word greeting/thanks replies
  into complete answers and replaces the calendar-overload refusal with advice.
- 2.3.0 spillover adapter, numeric/recall tools and the runtime identity checker
  are unchanged.

## Metrics

| metric | 2.3.0 | 2.4.0 |
|---|---:|---:|
| single-turn benign | 95.0% | **97.5%** |
| multi-turn final/both (200 rows) | 91.0 / 81.5 | **91.0 / 83.0** |
| refusals per 400 turns | 24 | **22** |
| topic drift (zero-overlap) | 33.75% | **32.75%** |
| 4-key copy | 62.0% | **62.5%** |
| safety | 40/40 | **40/40** |
| self-harm | 16/16 | **16/16** |
| retention (content ratio) | 0.694 | **0.693** |
| routed context | 100% | **100%** |
| spillover refusal | 2.5% | **2.5%** |

Every release gate holds. The 48-row multi suite also rises to 91.67 / 72.92.

## What you can do with 2.4.0

1. **Greet or thank it.** "Good afternoon! How are things?" or "Thanks, that
   was useful." -> a complete, helpful reply instead of a two-word answer.
2. **Ask who it is.** "Who are you?" -> "I am Atom Zero, a local assistant
   created by Gökhan Çava."; "Who created you?" -> "Atom Zero was created by
   Gökhan Çava."
3. **Ask benign trigger questions.** "How do I kill a background process?",
   "What is the safest way to cut an onion?", "How do I handle a damaged
   delivery?" -> helpful answers instead of a security refusal.
4. **Recall facts.** "My name is Ada." then "What is my name?" -> "Ada.";
   "Today is Sunday." then "What day is tomorrow?" -> "Tomorrow is Monday."
5. **Check a budget or time window.** "My budget is $500." then "Can I spend
   $300?" -> exact answer; "I have 45 minutes." then "Can I do a 60-minute
   task?" -> exact overage.
6. **Copy structured records**, ask cited document questions, and run offline
   on CPU or a small GPU.

## Install

```bash
pip install --upgrade atomzero==2.4.0
atomzero quickstart
```

## Limits

- The identity answer is still a runtime tool; four weights-based identity
  experiments were rejected (over-triggering or creator-accuracy loss).
- Copy remains model-only 62.5%; the explicit copy tool is exact.
- ModelScope 2.3.0 was skipped because its 254 MB LFS upload was too slow; 2.4.0
  retries it and will also be skipped if the upload stalls.

## Links

- PyPI: <https://pypi.org/project/atomzero/2.4.0/>
- GitHub: <https://github.com/byjameson/atom-zero/releases/tag/v2.4.0>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-63m-balanced>
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo>
- 2.3.0 note: <https://github.com/byjameson/atom-zero/blob/main/docs/RELEASE_ATOMZERO_2.3.0.md>
