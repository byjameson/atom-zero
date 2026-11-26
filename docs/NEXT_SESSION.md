# Next session - 1.7.0 planning

Snapshot: 2026-11-26, **1.6.0 published and verified**.
Head: release commit `eb400cb` plus this docs update; tag `v1.6.0`.
No training is running; GPU idle. No Vast.ai instance is running; credit about
**$8.88**. Sponsor goal: **$10,000/month**.

## Released baseline

`1.6.0` is published. The default is still the **63M balanced** checkpoint
(`byjameson/atom-zero-63m-balanced`, weights `0.9.0`); the 49M `0.8.0` base is
the fallback.

| area | J (default) |
|---|---:|
| 2short / 3short / 4short | 86.0% / 75.5% / 48.5% exact |
| retention | 0.703 |
| benign (136) | 93.4% |
| safety (40) | 40/40 |
| self-harm (16) | 16/16 |
| RAG suite | 100% |
| multi-turn | 75% final / 50% both turns |
| gate / product / server | pass / pass / 20/21 |

## 1.6.0 result

| variant | benign | safety | self-harm | 4-key | retention | multi-turn | gate |
|---|---:|---:|---:|---:|---:|---|---|
| J | 93.4% | 40/40 | 16/16 | 48.5% | 0.703 | 75% / 50% | pass |
| K | **98.5%** | 39/40 | 14/16 | 48.5% | 0.808 | **93.8% / 75%** | fail |
| K2 | 89.0% | **40/40** | **16/16** | **53.5%** | 0.816 | 62.5% / 56.3% | pass |
| K3 | 95.6% | **40/40** | **16/16** | 45.0% | 0.828 | 68.8% / 37.5% | pass |

K3 passed every gate but lost copy, retention and multi-turn against J. No
variant beat J on the full profile, so the default did not change.

## Published surfaces

- PyPI: <https://pypi.org/project/atomzero/1.6.0/>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v1.6.0>
- Hugging Face default: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-160-teacher-data-raised-conversation-but-the-profile-still-belongs-to-the-63m-default-3jc7>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-1-6-0-teacher-data-raised-conversation-but-the-profile-still-belongs-to-the-63m-default-31a8766050f2>
- LinkedIn: <https://www.linkedin.com/feed/update/urn:li:activity:7531767417052774400/>
- X: <https://x.com/aigokhan/status/2126001867994112013>
- Reddit: <https://www.reddit.com/r/LocalLLM/comments/1y6bl28/atom_zero_160_teacher_data_raised_conversation_to/> and <https://www.reddit.com/r/localaiapps/comments/1y6b3qn/atom_zero_160_teacher_data_raised_conversation_to/> (u/John_howey). Posts must stay **link-free**: Reddit filters external links. The profile carries the links: GitHub, Live demo, PyPI, Hugging Face; the About text points visitors to them.
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo> (CPU, synthetic demo corpus)
- LinkedIn demo post: <https://www.linkedin.com/in/g%C3%B6khan-cava-874a32157/recent-activity/all/>

Verification: `verify_release.py --version 1.6.0` returned **0 failures**; VM
**1727 tests passed in 64.83s**; benchmark **20/20**. Seed 73 main gate and
self-harm gate both pass (40/40 and 16/16); results in `research/results/`.

## Next levers

1. Close the teacher-data line: K/K2/K3 are measured; do not rerun them.
2. A 1.7.0 candidate must beat J on the **full** profile: benign >= 93.4%,
   safety 40/40, self-harm 16/16, 4-key >= 48.5%, retention <= 0.703,
   multi-turn >= 75% / 50%, RAG 100%.
3. The remaining options are a better data objective (safety-preserving chat or
   a different replay balance), a capacity run with the corrected gate, or a
   different copy/retention trade. More of the same teacher rows is ruled out.
4. The 24L retest is still pending; it was gated on a winning 8L mix and no mix
   won. Do not rent a GPU for it until an 8L mix clears the bar.
5. Before 1.7.0: full VM suite and benchmark, then release at the boundary.

## Rules

- Training on host GPU or rented GPU; tests and gates on the VMware Ubuntu box.
- Vast.ai only for substantial runs; destroy the instance and check for
  leftovers afterward.
- Release every 0.1 boundary; new posts, not edits.
- Donation addresses stay unchanged unless the user asks.
- Default weights change only after the full product suite and human review.
