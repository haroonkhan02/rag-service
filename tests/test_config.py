from pathlib import Path

import pytest

from ragsvc.config import ConfigError, Settings, load_settings

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "service.yaml"


def test_repo_config_matches_dataclass_defaults():
    # the YAML documents the defaults; they must not drift apart
    assert load_settings(REPO_CONFIG, environ={}) == Settings()


def test_env_overrides_are_typed():
    s = load_settings(REPO_CONFIG, environ={
        "RAGSVC__RETRIEVAL__TOP_K": "8",
        "RAGSVC__RETRIEVAL__MODE": "dense",
        "RAGSVC__CORPUS__PDFS": "false",
        "RAGSVC__CORPUS__EXCLUDE_GLOBS": "a/*,b/*",
        "RAGSVC__INDEX_DIR": "/tmp/idx",
        "UNRELATED": "x",
    })
    assert s.retrieval.top_k == 8 and s.retrieval.mode == "dense"
    assert s.corpus.pdfs is False
    assert s.corpus.exclude_globs == ("a/*", "b/*")
    assert s.index_dir == Path("/tmp/idx")


def test_config_path_from_env(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("retrieval:\n  top_k: 3\n")
    s = load_settings(environ={"RAGSVC_CONFIG": str(cfg)})
    assert s.retrieval.top_k == 3
    assert s.retrieval.candidates == 30  # untouched values keep their defaults


@pytest.mark.parametrize("env, message", [
    ({"RAGSVC__RETRIEVAL__TOPK": "3"}, "unknown setting"),
    ({"RAGSVC__RETRIEVAL__TOP_K": "three"}, "integer"),
    ({"RAGSVC__RETRIEVAL__TOP_K": "2.5"}, "integer"),
    ({"RAGSVC__RETRIEVAL__MODE": "semantic"}, "retrieval.mode"),
    ({"RAGSVC__RETRIEVAL__TOP_K": "50"}, "cannot exceed"),
    ({"RAGSVC__RETRIEVAL__CANDIDATES": "0"}, ">= 1"),
    ({"RAGSVC__CORPUS__PDFS": "maybe"}, "true/false"),
    ({"RAGSVC__EMBEDDING__DEVICE": "tpu"}, "device"),
    ({"RAGSVC__NOPE__X": "1"}, "unknown setting"),
])
def test_bad_values_fail_at_start_up(env, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(REPO_CONFIG, environ=env)


def test_section_must_be_a_mapping(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("retrieval: 5\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(cfg, environ={})


def test_float_settings_and_service_validation():
    s = load_settings(REPO_CONFIG, environ={"RAGSVC__CACHE__SEMANTIC_THRESHOLD": "0.9",
                                            "RAGSVC__RATE_LIMIT__RATE_PER_S": "2"})
    assert s.cache.semantic_threshold == 0.9 and s.rate_limit.rate_per_s == 2.0
    for env, message in [
        ({"RAGSVC__CACHE__SEMANTIC_THRESHOLD": "1.5"}, "semantic_threshold"),
        ({"RAGSVC__CACHE__TTL_S": "soon"}, "number"),
        ({"RAGSVC__GENERATION__BACKEND": "gpt"}, "generation.backend"),
        ({"RAGSVC__RATE_LIMIT__BURST": "0"}, "burst"),
    ]:
        with pytest.raises(ConfigError, match=message):
            load_settings(REPO_CONFIG, environ=env)
