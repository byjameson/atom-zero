# The growth ladder - how Atom Zero snowballs

Atom Zero is deliberately a **small local model**. The product does not pretend
that a 49M model becomes a frontier model by wishing. It grows capability in
layers, and only the last layer changes the parameter count.

## The honest ladder

| Layer | What grows | Parameters | Cost | Gate |
| --- | --- | --- | --- | --- |
| 0. Local model + retrieval | answers, citations | unchanged | CPU | answerability / citation gate |
| 1. Teacher escalation | access to a larger model | unchanged | per call | opt-in, privacy, budgets |
| 2. Verified memory | retrieval index | unchanged | local | claims / consensus / freshness |
| 3. Skills and recipes | procedural memory | unchanged | local | skill tests |
| 4. Distilled adapter | a small LoRA add-on | **grows slightly** | offline GPU | adapter eval + human approval |
| 5. Checkpoint upcycling | a larger base model | **grows** | offline GPU | full eval + human approval |
| 6. Agentic code improvement | the product itself | unchanged | external harness | self-code + gate + human |

The first five layers are capability growth without retraining the base model.
Layer 4 adds parameters; layer 5 replaces the checkpoint. Neither happens
automatically.

## The snowball loop

```text
task
  -> local attempt
  -> confidence / answerability / retrieval evidence
  -> route: local | teacher | agent | refuse
       local success        -> record; no external call
       low confidence       -> teacher (public tasks only, opt-in)
       code change          -> bounded agent packet (human approval)
       secret / private     -> refuse external; stay local
  -> teacher answer is untrusted
  -> claims + consensus + freshness + citation gate
  -> verified note enters retrieval memory
  -> repeated verified traces become skill or adapter candidates
  -> offline adapter/checkpoint training
  -> evaluation gate vs baseline
  -> human approves the new weights
  -> next task starts from a slightly stronger local model
```

The loop is a ratchet, not a runaway: no verified gain, no promotion.

## What the router decides

`atomzero-escalate` is the first product surface of this ladder. It is a
deterministic policy engine, not a model and not an agent:

- local confidence above the threshold -> stay local;
- low confidence or a freshness requirement -> ask the configured teacher;
- code work -> hand a bounded packet to the opt-in agent runner;
- `secret` never leaves the machine;
- private tasks do not go external;
- external and agent budgets are enforced from a content-free history;
- every agent handoff still requires a human.

The router never calls the network, never starts an agent and never changes
weights. It only returns a decision.

## Growth metrics

The scoreboard and growth ledger track:

- local answer rate and escalation rate;
- verified teacher yield (verified notes / teacher calls);
- retrieval index size and citation coverage;
- adapter candidate count and adapter eval delta;
- parameter count (only after an approved adapter/checkpoint);
- cost, latency and refusal rate.

A larger parameter count is not a success by itself. The success condition is a
measured improvement on held-out tasks without a regression in refusal or
citation quality.

## Product shape

- **Local-first:** the default install answers from the local model and local
  documents. No account, no network.
- **Opt-in teacher:** the user configures an OpenAI-compatible teacher (local
  Ollama or a hosted model) and explicit budgets.
- **Verified growth:** teacher answers are never trusted directly; they enter
  memory only after the claim/consensus/freshness checks.
- **Offline distillation:** verified traces become adapter candidates; training
  is a separate, explicit job.
- **Agentic code growth:** `atomzero-self-code` and `atomzero-self-improve`
  handle product code, always through a gate and a human decision.
- **Portable:** the small model, memory, skills and adapters stay local; the
  growth ledger is content-free.

## Roadmap

1. `atomzero-escalate` deterministic router (implemented).
2. Content-free growth ledger and summary (implemented).
3. Adapter-candidate export from verified teacher traces.
4. Adapter training/eval gate with a before/after benchmark.
5. Checkpoint upcycling path, offline and human-approved.
6. Product packaging: one command to inspect the ladder, budgets and current
   capability level.

## Limits

- Asking a larger AI does not automatically make the small model larger.
- Teacher answers can be wrong; verification is mandatory.
- Distillation can regress; every adapter needs a held-out evaluation.
- Parameter growth is expensive and offline; it is never a background side
  effect of chat.
- Agentic code changes remain human-gated and externally executed.
- The answerability gate remains closed until its targets are met; growth work
  does not bypass it.
