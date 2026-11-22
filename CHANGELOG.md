# Changelog

## atomzero 0.1.9 - support and onboarding polish (2026-11-22)

- Added EVM, Solana and Bitcoin donation addresses to the README, PyPI
  description, HF/ModelScope model card, project overview and articles.
- Added `atomzero-quickstart --reset-demo`.
- Added `atomzero --version` and `atomzero-quickstart --version`.
- Added `docs/TROUBLESHOOTING.md`.
- No model/runtime behavior changed; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.9.md`](docs/RELEASE_ATOMZERO_0.1.9.md).

## atomzero 0.1.8 - install doctor (2026-11-22)

- Added `atomzero-doctor`: Python/package/dependency/torch-CUDA/cache/port/
  write-access/Ollama preflight checks with actionable hints and `--json`.
- Documented the doctor in the README.
- No model/runtime behavior changed; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.8.md`](docs/RELEASE_ATOMZERO_0.1.8.md).

## atomzero 0.1.7 - PyPI docs link cleanup (2026-11-22)

- Republished the package description with all documentation links pointing to
  the `atom-zero` repository.
- Removed the old `stream-finetune` documentation links from the PyPI page.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.7.md`](docs/RELEASE_ATOMZERO_0.1.7.md).

## atomzero 0.1.6 - LinkedIn and ModelScope restored (2026-11-22)

- Restored LinkedIn and ModelScope as canonical Atom Zero release surfaces.
- Added both links back to the package README, PyPI metadata, HF model card and
  project overview.
- Added a job-search-oriented LinkedIn post for Atom Zero 0.1.6.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.6.md`](docs/RELEASE_ATOMZERO_0.1.6.md).

## atomzero 0.1.5 - canonical surfaces cleanup (2026-11-22)

- Removed the remaining ModelScope link from the published package description
  and the LinkedIn project URL from package metadata.
- Documented the five canonical Atom Zero surfaces: GitHub, Hugging Face, PyPI,
  dev.to and Medium.
- Superseded by `0.1.6`: LinkedIn and ModelScope are canonical surfaces again.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.5.md`](docs/RELEASE_ATOMZERO_0.1.5.md).

## atomzero 0.1.4 - standalone repository (2026-11-22)

- Moved the Atom Zero package to its own GitHub repository:
  <https://github.com/byjameson/atom-zero>.
- PyPI project URLs now point to the standalone repository.
- Current Atom Zero links target the `atom-zero` repository, `atomzero` PyPI,
  Hugging Face and the two launch articles. ModelScope/LinkedIn were temporarily
  removed in 0.1.4 and restored as canonical surfaces in 0.1.6.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.4.md`](docs/RELEASE_ATOMZERO_0.1.4.md).

## atomzero 0.1.3 - full package description (2026-11-22)

- Expanded the standalone GitHub/PyPI README to cover the full system: model,
  training, RAG, documents/citations, evidence, tools, research, curiosity,
  learning ladder, provider, safety, privacy, benchmarks and limits.
- Removed stale streamft links from the Atom Zero model card; it now links to
  the atomzero package, source and `atomzero-v0.1.3` release.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.3.md`](docs/RELEASE_ATOMZERO_0.1.3.md).

## atomzero 0.1.2 - standalone quickstart (2026-11-22)

- Added `atomzero-quickstart`: downloads the public checkpoint/tokenizer from
  Hugging Face, seeds a demo corpus and starts the local UI.
- No streamft checkout, trainer install or `git clone` is required.
- Added `--provider`, `--safety-filter`, `--port`, `--no-download` and
  `--no-browser` options.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.2.md`](docs/RELEASE_ATOMZERO_0.1.2.md).
