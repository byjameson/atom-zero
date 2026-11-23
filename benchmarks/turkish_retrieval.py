"""Turkish morphology retrieval check.

The queries use the base form of an entity while the target document uses an
inflected form (``kitap`` -> ``kitabın``, ``ağaç`` -> ``ağacın``, ``ev`` ->
``evler``). Distractor documents share the attribute words so lexical FTS has
candidates and reranking has to use the entity term.

Run:

    python benchmarks/turkish_retrieval.py
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from atomzero.doc_memory import DocMemory

DOCS = {
    "kitap.txt": "Kitabın kapağı mavidir. Kitap kütüphanede sergileniyor.",
    "kapak.txt": "Kapağı mavidir. Renk mavidir. Kütüphanede sergileniyor.",
    "agac.txt": "Ağacın gövdesi kalındır. Ağaç bahçede büyüyor.",
    "govde.txt": "Gövdesi kalındır. Bahçede büyüyor.",
    "sokak.txt": "Sokağın sonunda bir park var. Sokak trafiğe kapalı.",
    "park.txt": "Sonunda bir park var. Park trafiğe kapalı.",
    "kanat.txt": "Kanadı kırık kuş uçamıyor. Kuş veterinerde tedavi ediliyor.",
    "kus.txt": "Kırık kuş uçamıyor. Kuş veterinerde tedavi ediliyor.",
    "ev.txt": "Evler çok eski. Evlerin çatısı onarıldı.",
    "bina.txt": "Eski bina. Binaların çatısı onarıldı.",
    "su.txt": "Sular yükseldi. Su seviyesi tehlike sınırında.",
    "deniz.txt": "Deniz yükseldi. Seviye tehlike sınırında.",
    "yol.txt": "Yollar kapalı. Yolların bakımı sürüyor.",
    "kopru.txt": "Köprü kapalı. Köprünün bakımı sürüyor.",
    "sehir.txt": "Şehirde yeni yollar açıldı. Yolların bakımı yapılıyor.",
    "tren.txt": "Tren açıldı mı? Tren istasyonu açıldı.",
    "kitap2.txt": "Kitabı raflarda. Kitabın sayfaları yıpranmış.",
    "defter.txt": "Defter raflarda. Defterin sayfaları yıpranmış.",
}
QUERIES = (
    ("Kitap kapağı ne renk?", "kitap.txt"),
    ("Ağaç gövdesi nasıl?", "agac.txt"),
    ("Sokak sonunda park var mı?", "sokak.txt"),
    ("Kanat kırık mı?", "kanat.txt"),
    ("Ev eski mi?", "ev.txt"),
    ("Su yükseldi mi?", "su.txt"),
    ("Yol kapalı mı?", "yol.txt"),
    ("Yol açıldı mı?", "sehir.txt"),
    ("Kitap raflarda mı?", "kitap2.txt"),
    ("Ağaç gövdesi kalın mı?", "agac.txt"),
    ("Sokak trafiğe kapalı mı?", "sokak.txt"),
    ("Su seviyesi tehlike sınırında mı?", "su.txt"),
)


def run(output: str | Path | None = None) -> dict:
    results = {}
    details = []
    with tempfile.TemporaryDirectory(prefix="atomzero-turkish-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        try:
            for source, text in DOCS.items():
                docs.add_text(source, text, title=source)
            for mode in ("fts", "hybrid"):
                correct = 0
                for query, gold in QUERIES:
                    hits = docs.search(query, k=1, rerank=(mode == "hybrid"))
                    top = hits[0]["source"] if hits else ""
                    ok = top == gold
                    correct += 1 if ok else 0
                    details.append({"mode": mode, "query": query,
                                    "gold": gold, "top1": top, "ok": ok})
                results[mode] = {"queries": len(QUERIES),
                                 "top1": correct / len(QUERIES)}
        finally:
            docs.close()
    payload = {
        "benchmark": "atomzero-turkish-retrieval",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "documents": len(DOCS),
        "results": results,
        "details": details,
    }
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="benchmarks/results/turkish_retrieval.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args.out)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for mode, result in payload["results"].items():
            print(f"{mode}: top-1 {result['top1']:.3f} "
                  f"({result['queries']} queries)")
        for item in payload["details"]:
            status = "OK" if item["ok"] else "MISS"
            print(f"[{status}] {item['mode']}: {item['query']} "
                  f"-> {item['top1']} (gold {item['gold']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
