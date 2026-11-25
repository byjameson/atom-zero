"""Standalone Atom Zero quickstart.

Downloads the public 49.2M checkpoint/tokenizer, seeds a tiny local demo corpus
and starts the local server. No streamft checkout or trainer install is needed.

    atomzero-quickstart
"""
from __future__ import annotations

import argparse
import threading
import urllib.request
import webbrowser
from pathlib import Path

from atomzero import __version__
from atomzero.chat_server import serve
from atomzero.doc_memory import DocMemory

HF_BASE = "https://huggingface.co/byjameson/atom-zero-49m/resolve/main"
FILES = {
    "model.pt": 190_000_000,
    "config.json": 100,
    "chat_en_bpe8k.json": 500_000,
}
DEMO_DOCS = [
    ("launch_policy.txt", "Launch policy",
     "The launch code is L-7781. The primary window opens at 06:30 UTC."),
    ("warranty.txt", "Warranty",
     "The warranty code is W-9900. Claims must be submitted within 30 days."),
    ("travel.txt", "Travel",
     "Book business class for flights longer than four hours. The travel "
     "portal code is TR-42."),
    ("security.txt", "Visitor security",
     "Visitors receive a blue badge at north reception. The badge code is "
     "VB-42."),
]


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {url} -> {destination}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, \
                temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def ensure_assets(cache_dir: Path, download: bool = True,
                  hf_base: str = HF_BASE) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, minimum in FILES.items():
        path = cache_dir / name
        if not path.exists() or path.stat().st_size < minimum:
            if not download:
                raise FileNotFoundError(
                    f"missing {path}; run without --no-download or download "
                    f"{hf_base}/{name}")
            _download(f"{hf_base}/{name}", path)
        paths[name] = path
    return paths


def reset_demo(docs_path: str | Path) -> None:
    Path(docs_path).unlink(missing_ok=True)


def seed_demo(docs_path: str | Path) -> int:
    memory = DocMemory(docs_path)
    try:
        total = 0
        for source, title, text in DEMO_DOCS:
            total += memory.add_text(source, text, title=title)
        return total
    finally:
        memory.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomzero-quickstart",
        description="Download Atom Zero weights, seed a demo corpus and serve.")
    parser.add_argument("--version", action="version", version=f"atomzero {__version__}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--cache", default=str(
        Path.home() / ".cache/atomzero/atom-zero-49m"))
    parser.add_argument("--model-dir", default="",
                        help="use an existing checkpoint directory and never "
                             "download (overrides --cache)")
    parser.add_argument("--model-repo",
                        default="byjameson/atom-zero-49m",
                        help="Hugging Face model repo to download from")
    parser.add_argument("--docs", default="data/atomzero_demo.db")
    parser.add_argument("--reports-index", default="data/weekly_reports.json")
    parser.add_argument("--no-download", action="store_true",
                        help="use already-downloaded files only")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--provider", action="store_true",
                        help="enable a local Ollama/OpenAI-compatible provider")
    parser.add_argument("--provider-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--provider-model", default="llama3.2:3b")
    parser.add_argument("--teacher", action="store_true",
                        help="enable the Ollama teacher for hungry mode and "
                             "teacher notes")
    parser.add_argument("--teacher-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--teacher-model", default="llama3.2:3b")
    parser.add_argument("--teacher-label", default="ollama")
    parser.add_argument("--safety-filter", action="store_true",
                        help="enable the optional rules + ML safety filter")
    parser.add_argument("--reset-demo", action="store_true",
                        help="delete the demo document DB before seeding")
    args = parser.parse_args(argv)

    if args.reset_demo:
        reset_demo(args.docs)
    repo = args.model_repo.strip().strip("/")
    if not repo or repo.count("/") != 1:
        parser.error("--model-repo must look like owner/name")
    hf_base = f"https://huggingface.co/{repo}/resolve/main"
    if args.model_dir:
        cache = Path(args.model_dir)
        assets = ensure_assets(cache, download=False)
    else:
        cache = Path(args.cache)
        assets = ensure_assets(
            cache, download=not args.no_download, hf_base=hf_base)
    chunks = seed_demo(args.docs)
    print(f"demo corpus: {args.docs} ({chunks} chunks)", flush=True)
    url = f"http://{args.host}:{args.port}"
    print(f"Atom Zero quickstart: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    return serve(
        str(cache),
        str(assets["chat_en_bpe8k.json"]),
        host=args.host,
        port=args.port,
        docs_path=args.docs,
        safety_filter=args.safety_filter,
        chat_provider_url=(args.provider_url if args.provider else ""),
        chat_provider_model=(args.provider_model if args.provider else ""),
        teacher_url=(args.teacher_url if args.teacher else ""),
        teacher_model=(args.teacher_model if args.teacher else ""),
        teacher_label=args.teacher_label,
        allow_general_chat=args.provider,
        weekly_reports_path=args.reports_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
