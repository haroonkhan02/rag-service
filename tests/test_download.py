import io
import json
import zipfile

import pytest

from ragsvc import download
from ragsvc.config import Settings


def make_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def test_extract_keeps_text_sources_only(tmp_path):
    blob = make_zip({
        "ros2_documentation-abc/source/Topics.rst": "Topics\n======\n",
        "ros2_documentation-abc/source/images/a.png": "PNG",
        "ros2_documentation-abc/LICENSE": "CC BY 4.0",
    })
    dest = tmp_path / "ros2_documentation"
    n = download.extract_ros2_zip(blob, dest, "abc")
    assert n == 2
    assert (dest / "source" / "Topics.rst").exists()
    assert not (dest / "source" / "images").exists()
    assert (dest / "COMMIT").read_text().strip() == "abc"
    assert not (tmp_path / "_ros2_zip").exists()


@pytest.mark.parametrize("bad", ["../evil.rst", "/abs/evil.rst", "x/../../evil.rst"])
def test_extract_rejects_path_traversal(tmp_path, bad):
    with pytest.raises(ValueError, match="unsafe"):
        download.extract_ros2_zip(make_zip({bad: "x"}), tmp_path / "d", "abc")


def test_download_eval_uses_pinned_commit_and_records_sources(tmp_path, monkeypatch):
    urls = []

    def fake_fetch(url, timeout=120):
        urls.append(url)
        return b'{"id": "q01"}\n'

    monkeypatch.setattr(download, "fetch", fake_fetch)
    cfg = Settings(data_dir=tmp_path)
    sources = download.Sources(tmp_path / "SOURCES.json")
    download.download_eval(cfg, sources)
    assert all(cfg.eval.source_commit in u for u in urls) and len(urls) == 2
    record = json.loads((tmp_path / "SOURCES.json").read_text())
    assert record["eval/questions.jsonl"]["bytes"] == 14
    assert len(record["eval/questions.jsonl"]["sha256"]) == 64
    # second run is a no-op
    download.download_eval(cfg, sources)
    assert len(urls) == 2


def test_papers_must_be_pdfs(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "fetch", lambda url, timeout=120: b"<html>captcha</html>")
    with pytest.raises(ValueError, match="PDF"):
        download.download_papers(Settings(data_dir=tmp_path),
                                 download.Sources(tmp_path / "S.json"))
