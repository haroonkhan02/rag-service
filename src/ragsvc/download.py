"""`ragsvc-download`: fetch everything the index and the evaluation need, at pinned versions.

    data/ros2_documentation/   ROS 2 docs sources (CC BY 4.0), ros2/ros2_documentation @ commit
    data/pdfs/                 three open-access ROS 2 papers from arXiv (local use only)
    data/eval/questions.jsonl  66 hand-written questions from robotics-docs-rag @ commit
    data/eval/reference_retrieval.json   that repo's published retrieval results (parity check)
    data/SOURCES.json          URL, size and SHA-256 of every downloaded file

Re-running is a no-op for files that already exist (use --force to refetch).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from ragsvc.config import Settings, load_settings

ROS2_ZIP = "https://codeload.github.com/ros2/ros2_documentation/zip/{commit}"
RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
EVAL_FILES = {
    "questions.jsonl": "eval/questions.jsonl",
    "reference_retrieval.json": "reports/retrieval/results_main.json",
}
# arXiv id -> file stem (same three papers as robotics-docs-rag)
PAPERS = {
    "2211.07752": "ros2-design-architecture-uses",
    "2003.00368": "marathon2-nav2",
    "2307.15236": "nav2-algorithms-survey",
}
PAPER_TITLES = {
    "ros2-design-architecture-uses":
        "Robot Operating System 2: Design, Architecture, and Uses In The Wild",
    "marathon2-nav2": "The Marathon 2: A Navigation System",
    "nav2-algorithms-survey": "From the Desks of ROS Maintainers: A Survey of Modern & Capable "
                              "Mobile Robotics Algorithms in the Robot Operating System 2",
}
UA = {"User-Agent": "rag-service/0.1 (portfolio project)"}


def fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


class Sources:
    """Append-only record of what was downloaded from where (data/SOURCES.json)."""

    def __init__(self, path: Path):
        self.path = path
        self.items: dict = json.loads(path.read_text()) if path.exists() else {}

    def add(self, name: str, url: str, blob: bytes) -> None:
        self.items[name] = {"url": url, "bytes": len(blob), "sha256": sha256(blob)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, indent=1, sort_keys=True) + "\n")


def safe_members(zf: zipfile.ZipFile, suffixes: tuple[str, ...]) -> list[str]:
    """Members to extract: wanted suffixes only, and never a path that escapes the target."""
    out = []
    for name in zf.namelist():
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe path in archive: {name!r}")
        if name.endswith(suffixes):
            out.append(name)
    return out


def extract_ros2_zip(blob: bytes, dest: Path, commit: str) -> int:
    tmp = dest.parent / "_ros2_zip"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        members = safe_members(zf, (".rst", "LICENSE"))  # text only, not 40 MB of images
        zf.extractall(tmp, members)
    roots = [p for p in tmp.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise ValueError(f"expected one top-level folder in the archive, found {len(roots)}")
    shutil.rmtree(dest, ignore_errors=True)
    roots[0].rename(dest)
    shutil.rmtree(tmp)
    (dest / "COMMIT").write_text(commit + "\n")
    return len(members)


def download_ros2_docs(cfg: Settings, sources: Sources, force: bool = False) -> Path:
    dest = cfg.data_dir / "ros2_documentation"
    if (dest / "source").exists() and not force:
        print(f"ROS 2 docs already in {dest}")
        return dest
    commit = cfg.corpus.ros2_commit
    url = ROS2_ZIP.format(commit=commit)
    print(f"downloading ros2_documentation @ {commit[:10]} ...")
    blob = fetch(url)
    sources.add("ros2_documentation.zip", url, blob)
    n = extract_ros2_zip(blob, dest, commit)
    print(f"saved {n} files to {dest}")
    return dest


def download_papers(cfg: Settings, sources: Sources, force: bool = False) -> None:
    pdf_dir = cfg.data_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for arxiv_id, stem in PAPERS.items():
        out = pdf_dir / f"{stem}.pdf"
        if out.exists() and not force:
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"downloading arXiv:{arxiv_id} -> {out.name}")
        blob = fetch(url)
        if not blob.startswith(b"%PDF"):
            raise ValueError(f"{url} did not return a PDF")
        out.write_bytes(blob)
        sources.add(f"pdfs/{out.name}", url, blob)


def download_eval(cfg: Settings, sources: Sources, force: bool = False) -> None:
    out_dir = cfg.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path in EVAL_FILES.items():
        out = out_dir / name
        if out.exists() and not force:
            continue
        url = RAW.format(repo=cfg.eval.source_repo, commit=cfg.eval.source_commit, path=path)
        blob = fetch(url)
        out.write_bytes(blob)
        sources.add(f"eval/{name}", url, blob)
        print(f"saved {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_settings(args.config)
    sources = Sources(cfg.data_dir / "SOURCES.json")
    download_ros2_docs(cfg, sources, args.force)
    if cfg.corpus.pdfs:
        download_papers(cfg, sources, args.force)
    download_eval(cfg, sources, args.force)


if __name__ == "__main__":
    main()
