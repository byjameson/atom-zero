# Atom Zero 0.22.0: the curriculum loop is now inside self-improve

Atom Zero 0.22.0 is a small release with one important change: the autonomous
self-improvement loop can now run a measured capability-growth job.

## What changed

`atomzero-self-improve` accepts `--adapter-curriculum-command`. The command
runs the staged copy curriculum and returns a content-free ledger. The loop
records:

- return code and elapsed time;
- gate status;
- exact-match and key-copy rates;
- promotion result and active registry path.

It fails closed. A failed runner, missing ledger, non-passing gate or missed
threshold sets `status: curriculum_failed`.

The 0.21.0 curriculum adapter is still the best measured result: **40.0% exact
match / 54.8% key copy** on a 250-row mixed held-out set. It lives in a separate
curriculum registry; base `0.8.0` remains the product default.

## The honest limit

Multi-key copying is still weak: 8-12% exact. Pure multi training improves that
number but destroys one-key JSON (72% -> 10%). Adapter soup was worse than the
rehearsal adapter on every stage. The next target is the multi-key wall, not a
bigger claim.

## Evidence

- 1271 tests passed on the 8 vCPU VM in 46.34s.
- Benchmark 20/20.
- No model weights changed; base remains `0.8.0`.

## Install

```powershell
pip install atomzero==0.22.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/0.22.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

The system can now measure and gate a narrow capability inside its own loop. It
is not ASI, and it should not pretend to be.
