# Atom Zero 2.1.0 — exact budget and time decisions

2.1.0 keeps the 63M `1.2.0` weights and adds a deterministic numeric decision
checker to the local assistant. The model still handles recall, copy, RAG and
safety; explicit budget and time-fit questions are now answered by code, so
they are exact instead of guessed.

## What changed

- `src/atomzero/numeric_router.py`: recognises a spending limit (budget, cap,
  "set aside") plus a spend amount, or a time window plus a task duration, and
  computes the answer. Everything else returns `handled: False`.
- Wired into `/v1/chat/completions`, `/generate` and the agent chat path with
  `source: numeric_tool`.
- 16 new unit tests cover multi-turn, single-turn, commas, decimals,
  equality, time windows, unrelated amounts and the chat wrapper.
- Weights are unchanged (`1.2.0`); no new training data is needed for a
  correct answer.

## Why a checker instead of a bigger adapter

The 63M model answers its own trained recall rows exactly (name, city,
language, day, preference, time), but it cannot compare two dollar amounts.
The tokenizer splits numbers inconsistently (`500` is one token, `846` is
three, `925` is `9` + `25`), and a LoRA overfit control reached **89.5%** on
the 200 rows it trained on while staying at **45.9%** on 1,250 fresh rows.
That is memorisation, not a comparison rule. A deterministic checker gives
the correct answer today; the model-side arithmetic work continues as
research, not as a public claim.

## Metrics

| metric | 2.0.0 | 2.1.0 |
|---|---:|---:|
| single-turn benign | 97.5% | 97.5% |
| multi-turn final/both | 83.3 / 58.3 | 83.3 / 58.3 |
| 4-key copy (seen) | 63.0% | 63.0% |
| history recall (200 held-out) | 65.0% | **82.5%** |
| budget decisions | 4/25 | **25/25** |
| time decisions | 11/25 | **25/25** |
| safety | 40/40 | 40/40 |
| self-harm | 16/16 | 16/16 |
| RAG | 100% | 100% |

- Routed natural held-out (276 rows, new phrasings): **91.3%** (2.0.0: 48.2%),
  with budget **133/133** and time **97/97**.
- The checker triggers on **0/168** benign prompts and **0/16** self-harm
  prompts, so ordinary chat and crisis handling keep the model path.
- Product suite: **pass** (single 97.5%, multi 83.3 / 58.3, copy exact 63.0%,
  safety 40/40, content ratio 0.700). Server suite: **21/21** (RAG 5/5 with
  citations, copy tool, multi-turn, benign).

## What you can do with 2.1.0

1. **Check a purchase against a budget.** Say "My budget is $500" and then
   "Can I spend $300?" → "Yes, $300 is within your $500 budget." Ask about
   $650 → "No, $650 is over your $500 budget by $150."
2. **Plan a task against a time window.** "I have 45 minutes" then "Can I do a
   60-minute task?" → "No, a 60-minute task is 15 minutes over your
   45-minute window."
3. **Keep the conversation's facts.** Names, cities, languages, days and
   preferences are still recalled from earlier turns, and the answer can be
   combined with the budget check in the same session.
4. **Use it offline.** The checker is local arithmetic; no API, no telemetry,
   no network. It runs on CPU or a small GPU.
5. **Call it from your own client.** The OpenAI-compatible response includes
   `"source": "numeric_tool"` and `"kind": "budget"` or `"time"`, so a UI can
   style or log tool answers separately from model text.

## Install

```bash
pip install --upgrade atomzero==2.1.0
atomzero quickstart
```

Verification: VM **1780 tests passed** (54 s); benchmark **20/20**; product
suite **pass**; server suite **21/21**; self-harm **16/16**.

The Studio demo and the model repositories carry the same 2.1.0 package and
`1.2.0` weights.

## Limits

- The checker covers explicit budget/spend and window/task shapes; it does
  not infer a budget from context or convert currencies.
- Arithmetic is not a model capability in this release. The model card says
  so explicitly.
- Multi-turn coherence is unchanged from 2.0.0.

## Links

- PyPI: <https://pypi.org/project/atomzero/2.1.0/>
- GitHub: <https://github.com/byjameson/atom-zero/releases/tag/v2.1.0>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-63m-balanced>
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo>
- 2.0.0 note: <https://github.com/byjameson/atom-zero/blob/main/docs/RELEASE_ATOMZERO_2.0.0.md>
