# Atom Zero 2.2.0 — exact recall for facts you already told it

2.2.0 keeps the 63M `1.2.0` weights and extends the deterministic tool layer
from budget/time checks to conversation facts. Explicit questions about a
name, city, language, preference, weekday or pet are now answered exactly from
the conversation history; the model keeps wording, copy, RAG and safety.

## What changed

- `src/atomzero/recall_router.py`: parses the facts a user stated and answers
  only explicit recall questions ("What is my name?", "Which city do I live
  in?", "What day is tomorrow?"). Everything else returns `handled: False`.
- Wired into `/v1/chat/completions`, `/generate` and the agent chat path with
  `source: recall_tool`.
- 23 new unit tests cover name, city, language correction, preference, day
  successor, pet, new phrasings, no-fact and no-question cases, and the server
  wrapper.
- Budget/time handling from 2.1.0 stays (`numeric_tool`); weights unchanged.

## Metrics

| metric | 2.1.0 | 2.2.0 |
|---|---:|---:|
| routed context (200 held-out pairs) | 82.5% | **100%** |
| natural held-out (276 phrasings) | 91.3% | **100%** |
| model-only context (same 200 pairs) | 65.0% | 65.0% |
| single-turn benign | 97.5% | 97.5% |
| multi-turn final/both | 83.3 / 58.3 | 83.3 / 58.3 |
| 4-key copy (seen) | 63.0% | 63.0% |
| safety | 40/40 | 40/40 |
| self-harm | 16/16 | 16/16 |
| RAG | 100% | 100% |

- All eight context families are **25/25** on the routed path.
- The tools trigger on **0/168** benign prompts and **0/16** self-harm
  prompts.
- The routed number describes the product path. The model alone still scores
  65.0% on the context pairs; the card keeps both numbers separate.

## What you can do with 2.2.0

1. **Recall your name.** "My name is Ada." then "What is my name?" → "Ada."
   Also works with "Hi, I'm Ada." and "Please call me Ada."
2. **Recall your city.** "I live in Izmir." then "Which city do I live in?" →
   "Izmir." "I moved to Bursa last month." then "Where am I based?" → "Bursa."
3. **Correct a language.** "Use Java for this project." then "Actually, use Go.
   Which language are we using?" → "Go."
4. **Keep a preference.** "I prefer coffee over water." then "What do I
   prefer?" → "coffee."
5. **Know the weekday.** "Today is Sunday." then "What day is tomorrow?" →
   "Tomorrow is Monday."
6. **Keep a pet name.** "My dog's name is Lucy." then "What is my dog's
   name?" → "Lucy."
7. **Check a budget or a time window.** The 2.1.0 checker still answers
   "Can I spend $300?" and "Can I do a 60-minute task?" exactly.
8. **Use the rest of the assistant.** Copy 4-key records, ask cited document
   questions, and keep everything local on CPU or a small GPU.

## Install

```bash
pip install --upgrade atomzero==2.2.0
atomzero quickstart
```

Verification: VM **1803 tests passed**; benchmark **20/20**; product suite
**pass**; server suite **21/21**; self-harm **16/16**.

## Limits

- Recall covers facts stated in the current conversation; it is not a
  long-term memory store and does not infer facts.
- Budget/time and recall are runtime tools, not model capabilities in this
  release. The model card says so.
- Weights are unchanged (`1.2.0`); multi-turn coherence is unchanged.

## Links

- PyPI: <https://pypi.org/project/atomzero/2.2.0/>
- GitHub: <https://github.com/byjameson/atom-zero/releases/tag/v2.2.0>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-63m-balanced>
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo>
- 2.1.0 note: <https://github.com/byjameson/atom-zero/blob/main/docs/RELEASE_ATOMZERO_2.1.0.md>
