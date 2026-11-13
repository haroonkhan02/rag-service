import json
from dataclasses import replace

import numpy as np
import pytest

from ragsvc.build_index import build_bundle
from ragsvc.bundle import EMBEDDINGS, MANIFEST, BundleError, IndexBundle, load_bundle
from ragsvc.config import ChunkingSettings, EmbeddingSettings, Settings


@pytest.fixture
def bundle(docs, embedder):
    return build_bundle(Settings(), docs, embedder)


def test_round_trip(tmp_path, bundle):
    out = bundle.save(tmp_path / "index")
    loaded = load_bundle(out, Settings())
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in bundle.chunks]
    np.testing.assert_array_equal(loaded.embeddings, bundle.embeddings)
    m = loaded.manifest
    assert m["n_chunks"] == len(bundle.chunks) and m["dim"] == 256
    assert m["n_documents"] == 4
    assert m["build"]["embedding_model"] == Settings().embedding.model
    assert not list(tmp_path.glob(".index-*"))  # temp folder cleaned up


def test_overwrite_replaces_previous_index(tmp_path, bundle, docs, embedder):
    bundle.save(tmp_path / "index")
    smaller = build_bundle(Settings(), docs[:1], embedder)
    smaller.save(tmp_path / "index")
    assert load_bundle(tmp_path / "index").manifest["n_chunks"] == len(smaller.chunks)
    assert not (tmp_path / "index.old").exists()


def test_corrupted_file_is_detected(tmp_path, bundle):
    out = bundle.save(tmp_path / "index")
    emb = np.load(out / EMBEDDINGS)
    emb[0, 0] += 1.0
    np.save(out / EMBEDDINGS, emb)
    with pytest.raises(BundleError, match="checksum"):
        load_bundle(out)
    load_bundle(out, verify=False)  # explicit opt-out still loads


def test_settings_mismatch_is_refused(tmp_path, bundle):
    out = bundle.save(tmp_path / "index")
    other = replace(Settings(), embedding=EmbeddingSettings(model="other/model"),
                    chunking=ChunkingSettings(max_words=100))
    with pytest.raises(BundleError, match="chunking, embedding_model"):
        load_bundle(out, other)


def test_missing_index_and_wrong_format(tmp_path, bundle):
    with pytest.raises(BundleError, match="ragsvc-build-index"):
        load_bundle(tmp_path / "nothing")
    out = bundle.save(tmp_path / "index")
    m = json.loads((out / MANIFEST).read_text())
    m["format_version"] = 99
    (out / MANIFEST).write_text(json.dumps(m))
    with pytest.raises(BundleError, match="format"):
        load_bundle(out)


def test_shape_and_id_checks(bundle):
    with pytest.raises(BundleError, match="shape"):
        IndexBundle(bundle.chunks, bundle.embeddings[:-1], {})
    with pytest.raises(BundleError, match="duplicate"):
        IndexBundle(bundle.chunks[:1] * 2, bundle.embeddings[:2], {})
