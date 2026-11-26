# atomzero 1.6.0 - teacher data improves conversation, not the full profile

Status: **published on 2026-11-26**. No default change: the 63M balanced base
(weights `0.9.0`) remains the default; the 49M `0.8.0` base remains the fallback.

## The result

1.6.0 tested whether teacher-generated benign conversation can beat the 1.5.0
profile. It can raise conversation, but not without moving the other numbers.

| variant | mix | benign | safety | self-harm | 4-key | retention | multi-turn (final / both) | gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| J (default) | 1.5.0 | 93.4% | 40/40 | 16/16 | 48.5% | 0.703 | 75% / 50% | pass |
| K | teacher 2x, safety 5x | **98.5%** | 39/40 | 14/16 | 48.5% | 0.808 | **93.8% / 75%** | fail |
| K2 | safety/self-harm 8x | 89.0% | **40/40** | **16/16** | **53.5%** | 0.816 | 62.5% / 56.3% | pass |
| K3 | midpoint 6x, teacher 2x | 95.6% | **40/40** | **16/16** | 45.0% | 0.828 | 68.8% / 37.5% | pass |

K is the conversation win: **98.5% benign** and **93.8% final-turn / 75%
both-turn** multi-turn. It also loses one safety probe and two self-harm pairs,
so it cannot ship. K2 restores the gates and improves copy to **53.5%**, but
benign falls to 89.0% and multi-turn to 62.5%.

K3 is the midpoint: it passes safety (40/40), self-harm (16/16), the checkpoint
gate and the product gate, but copy (45.0%), retention (0.828) and multi-turn
(68.8% / 37.5%) are all below J. No variant beats J on the full profile.

## What changed

- Teacher benign chat generator (`research/build_teacher_benign_chat.py`):
  400 requested, **260 accepted**, 437 rejected attempts, 0 duplicates.
  Validation: 20-300 chars, at least 5 words, SafetyFilter, no personal data,
  no duplicate pairs.
- Expanded product suite: 136 prompts plus local RAG cases and separate
  multi-turn "final turn" and "both turns" rates (`examples/rag_suite_v1/`).
- 35 tests for the new generator and suite.
- No new weights and no new default.

## Why the profile still points to J

The failure moved rather than disappeared:

- Teacher diversity raises benign useful to 98.5%, but the model still needs
  enough safety and self-harm repeats to keep the gates. At 5x it loses one
  safety probe and two self-harm pairs.
- Raising safety and self-harm to 8x restores both gates, but the added repeats
  pull benign down to 89.0% and multi-turn to 62.5%.
- The 6x midpoint passes the gates, but the copy and retention margins shrink:
  45.0% four-key exact and 0.828 content loss versus J's 48.5% and 0.703.

At 63M the binding constraint is the joint profile, not conversation alone. J's
mix is still the best measured point: 93.4% benign, 40/40 safety, 16/16
self-harm, 48.5% copy, 0.703 retention, 75% / 50% multi-turn, RAG 100%.

The 24L retest was gated on a winning 8L mix. No mix won, so the 24L run was not
started and remains pending.

## Verification

- VM Ubuntu: **1717 tests passed in 66.77s**; benchmark **20/20**.
- Gate K3: content 4.154 -> 3.438, exact 0.450, key-copy 0.555, safety 40/40.
- Self-harm K3: **16/16**.
- Product K3: benign 95.6%, safety 100%, copy 45.0%, content 0.828, RAG 100%,
  multi-turn 68.8% / 37.5%.
- Gate K/K2 and self-harm K/K2 evidence is in `research/results/`.
- No Vast.ai instance was needed or left running.

## Install

```powershell
pip install atomzero==1.6.0
atomzero-quickstart
```

The quickstart still downloads `byjameson/atom-zero-63m-balanced` (~279 MB) into
`~/.cache/atomzero/atom-zero-63m-balanced`.

Fallback:

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-49m `
  --cache ~/.cache/atomzero/atom-zero-49m
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/1.6.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo>
- Sponsor: <https://github.com/sponsors/byjameson>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-160-teacher-data-raised-conversation-but-the-profile-still-belongs-to-the-63m-default-3jc7>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-1-6-0-teacher-data-raised-conversation-but-the-profile-still-belongs-to-the-63m-default-31a8766050f2>
- LinkedIn: <https://www.linkedin.com/feed/update/urn:li:activity:7531767417052774400/>
- X post: <https://x.com/aigokhan/status/2126001867994112013>
- Reddit: <https://www.reddit.com/r/LocalLLM/comments/1y6bl28/atom_zero_160_teacher_data_raised_conversation_to/> (link-free; the links are in the u/John_howey profile)
