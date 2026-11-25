# Capability scorecard

Content-free. No prompt, answer or dataset text is included.

Schema: `atomzero.capability_scorecard.v1`

No aggregate score is reported. The six criteria are not interchangeable; status counts are shown instead.

## Six criteria

| # | Criterion | Status | Score | Evidence |
|---:|---|---|---:|---|
| 1 | Discovers its own knowledge gaps | partial | 0.35 | answerability Brier 0.233 (target <=0.15): missed<br>answerability ECE 0.064 (target <=0.10): met |
| 2 | Decides what, when and how to learn | partial | 0.80 | first attempt: gate=fail; exact 0.370; key copy 0.550; promoted=false; reasons=content_loss_not_improved,content_ppl_regression<br>successful attempt: gate=pass; exact 0.530; key copy 0.600; promoted=true |
| 3 | Retains old knowledge while learning | partial | 0.53 | research/results/weak_stage_rehearsal.json: 1/5 stages kept exact and key copy within 0.02; worst exact delta -0.160 (echo_json)<br>research/results/weak_stage_rehearsal_gentle.json: 2/5 stages kept exact and key copy within 0.02; worst exact delta -0.110 (echo_json) |
| 4 | Transfers learning to different problems | not_met | 0.04 | held-out city_river: base 0.000 / adapter 0.000 exact; n=30; content loss 4.794 -> 3.243 (adapter better)<br>held-out arithmetic: base 0.000 / adapter 0.000 exact; n=30; content loss 4.391 -> 5.142 (adapter worse) |
| 5 | Improves its own learning algorithm | not_met | 0.30 | 4 learning-algorithm change attempt(s) recorded<br>weak-stage rehearsal changed the stage mix; 1/5 stages retained after the change; worst exact delta -0.160 (echo_json) |
| 6 | Does this at low hardware cost | met | 1.00 | cpu 8 (envelope <=8)<br>ram gb 16 (envelope <=16) |

## Criterion details

### 1. Discovers its own knowledge gaps

- **Status:** `partial` (score 0.35)
- **Evidence:**
  - answerability Brier 0.233 (target <=0.15): missed
  - answerability ECE 0.064 (target <=0.10): met
  - refusal recall 0.546 (target >=0.80): missed
  - refusal precision 0.631
  - answerability held-out rows: 200
  - 2/2 answerability specs FAIL
- **Sources:** `benchmarks/results/answerability_hybrid/answerability_profile.json`
- **Next measurable target:** close at least one self-discovered gap end-to-end with a held-out check; reach Brier <=0.15, ECE <=0.10, refusal recall >=0.80 and zero failed answerability specs

### 2. Decides what, when and how to learn

- **Status:** `partial` (score 0.80)
- **Evidence:**
  - first attempt: gate=fail; exact 0.370; key copy 0.550; promoted=false; reasons=content_loss_not_improved,content_ppl_regression
  - successful attempt: gate=pass; exact 0.530; key copy 0.600; promoted=true
  - a recorded candidate was rejected and not promoted
  - a later candidate passed the gate and was promoted
  - selection evidence: 2 rehearsal run(s) ranked two or more stages before choosing a mix
  - selection evidence: multi-key summary compared 3 measured methods
- **Sources:** `research/results/multi_key_0.30_summary.json`, `research/results/self_improve_curriculum_summary.json`, `research/results/weak_stage_rehearsal.json`, `research/results/weak_stage_rehearsal_gentle.json`
- **Next measurable target:** make one autonomous learning decision (no human approval) with a recorded rejected alternative and a passing held-out gate

### 3. Retains old knowledge while learning

- **Status:** `partial` (score 0.53)
- **Evidence:**
  - research/results/weak_stage_rehearsal.json: 1/5 stages kept exact and key copy within 0.02; worst exact delta -0.160 (echo_json)
  - research/results/weak_stage_rehearsal_gentle.json: 2/5 stages kept exact and key copy within 0.02; worst exact delta -0.110 (echo_json)
  - adapter eval: mixed exact 0.480, key copy 0.604, n=250
  - 3-4 key copy remains weak: best exact 0.060
- **Sources:** `research/results/multi_key_0.30_summary.json`, `research/results/multi_recover_summary.json`, `research/results/weak_stage_rehearsal.json`, `research/results/weak_stage_rehearsal_gentle.json`
- **Next measurable target:** after one new learning step, keep >=4/5 stages within 0.02 of their pre-step exact and key-copy score and reach mixed exact >=0.70

### 4. Transfers learning to different problems

- **Status:** `not_met` (score 0.04)
- **Evidence:**
  - held-out city_river: base 0.000 / adapter 0.000 exact; n=30; content loss 4.794 -> 3.243 (adapter better)
  - held-out arithmetic: base 0.000 / adapter 0.000 exact; n=30; content loss 4.391 -> 5.142 (adapter worse)
  - held-out city_yaml: base 0.000 / adapter 0.000 exact; n=30; content loss 3.484 -> 3.632 (adapter worse)
  - 3-4 key copy: best exact 0.060 (n=unknown)
  - copy curriculum (mlp): 0/200 exact across 5 held-out families
  - runtime copy tool 1.000 exact is tool use and is excluded from model transfer
  - city/river transfer target: 0.000 / 0.500 (0.0% of target)
  - arithmetic/YAML transfer target: 0.000 / 0.500 (0.0% of target)
  - 3-4 key transfer target: 0.060 / 0.300 (20.0% of target)
- **Sources:** `research/results/copy_curriculum_summary.json`, `research/results/heldout_task_transfer_summary.json`, `research/results/multi_key_0.30_summary.json`
- **Next measurable target:** reach >=0.50 exact on held-out city/river and arithmetic/YAML with n>=100 per family, and >=0.30 exact on 3-4 arbitrary keys; keep the runtime copy tool out of the model path

### 5. Improves its own learning algorithm

- **Status:** `not_met` (score 0.30)
- **Evidence:**
  - 4 learning-algorithm change attempt(s) recorded
  - weak-stage rehearsal changed the stage mix; 1/5 stages retained after the change; worst exact delta -0.160 (echo_json)
  - weak-stage rehearsal changed the stage mix; 2/5 stages retained after the change; worst exact delta -0.110 (echo_json)
  - key-weighted loss: mixed exact 0.404 vs 0.420 standard (-0.016)
  - copy curriculum compared MLP and attention adapter loci; best exact 0.000, delta +0.000
  - QKV LoRA first result is negative/uncontrolled (6% exact / 18% key copy on multi-short); no positive QKV result is counted
  - no learning-algorithm change improved the aggregate target by >=0.05; process evidence is not counted as improvement
- **Sources:** `docs/BATCH_0.60.0.md`, `research/results/copy_curriculum_summary.json`, `research/results/multi_key_0.30_summary.json`, `research/results/weak_stage_rehearsal.json`, `research/results/weak_stage_rehearsal_gentle.json`
- **Next measurable target:** show one learning-algorithm change that improves mixed exact by >=0.05 over its own baseline, retains >=4/5 stages and reproduces across >=2 seeds

### 6. Does this at low hardware cost

- **Status:** `met` (score 1.00)
- **Evidence:**
  - cpu 8 (envelope <=8)
  - ram gb 16 (envelope <=16)
  - disk gb 40 (envelope <=40)
  - base model 49.2M parameters
  - adapter 405,504 trainable parameters
  - recorded adapter training 149.0s
  - 1,428 tests recorded
- **Sources:** `research/results/weak_stage_rehearsal.json`
- **Next measurable target:** keep the full loop within 8 vCPU / 16 GB RAM / 40 GB disk while adding a held-out transfer eval; report wall-clock and peak RAM per cycle

## Held-out tasks

| Task | n | Base exact | Adapter exact | Model transfer | Status |
|---|---:|---:|---:|---|---|
| city/river | 30 | 0.000 | 0.000 | yes | not_met |
| 3-4 arbitrary keys | — | — | 0.060 | yes | not_met |
| arithmetic | 30 | 0.000 | 0.000 | yes | not_met |
| city YAML | 30 | 0.000 | 0.000 | yes | not_met |
| runtime copy tool | 200 | — | 1.000 | no (tool use) | met |

The runtime copy tool is a deterministic transcription tool. Its accuracy is **not** counted as model transfer.

## Capacity probe (not counted as 49M transfer)

Model: `Qwen/Qwen2.5-0.5B-Instruct`

| Stage | Pairs | Rows | Baseline exact | Flat-prompt exact |
|---|---:|---:|---:|---:|
| 2short | 2 | 20 | 0.650 | 0.850 |
| 3short | 3 | 20 | 0.550 | 0.650 |
| 4short | 4 | 20 | 0.550 | 0.850 |

This is capacity evidence about a larger pretrained model. It is not a from-scratch Atom Zero result and is excluded from the transfer score.

## Hardware

| Resource | Value | Envelope |
|---|---:|---:|
| CPU | 8.0 | <=8 |
| RAM (GB) | 16.0 | <=16 |
| Disk (GB) | 40.0 | <=40 |
| Base model (M params) | 49.2 | — |
| Adapter params | 405,504 | — |
| Recorded training (s) | 148.968 | — |
| Tests recorded | 1,428 | — |

## Input sources

- `benchmarks/results/answerability_hybrid/answerability_profile.json`
- `research/results/copy_curriculum_summary.json`
- `research/results/copy_tool_benchmark_summary.json`
- `research/results/heldout_task_transfer_summary.json`
- `research/results/multi_key_0.30_summary.json`
- `research/results/multi_recover_summary.json`
- `research/results/self_improve_curriculum_summary.json`
- `research/results/weak_stage_rehearsal.json`
- `research/results/weak_stage_rehearsal_gentle.json`

## Status counts

met 1 · partial 3 · not_met 2 · unknown 0

No aggregate score is reported.
