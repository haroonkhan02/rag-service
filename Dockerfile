# syntax=docker/dockerfile:1
# Three stages: install, build the index, run. The index and both models are baked into the
# image, so a container starts without network access and serves the same index every time.

FROM python:3.12-slim AS base
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf
WORKDIR /app
# CPU-only PyTorch first: the default Linux wheel pulls in several GB of CUDA libraries
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[models,serve,otel,anthropic]"

FROM base AS index
# corpus at its pinned commit -> chunks + embeddings + checksummed manifest
RUN ragsvc-download && ragsvc-build-index
# fetch the cross-encoder too, so the runtime stage never downloads anything
RUN python -c "from ragsvc.config import load_settings; from ragsvc.models import STReranker; \
STReranker(load_settings().retrieval.reranker, 'cpu').score('warm up', ['a passage'])"

FROM base AS runtime
COPY --from=index /opt/hf /opt/hf
COPY --from=index /app/artifacts /app/artifacts
RUN useradd --create-home --uid 10001 app && chown -R app:app /app /opt/hf
USER app
# No LLM by default: answers quote the top passage. For Claude, run with
#   -e RAGSVC__GENERATION__BACKEND=anthropic -e ANTHROPIC_API_KEY
ENV HF_HUB_OFFLINE=1 \
    RAGSVC__SERVER__HOST=0.0.0.0 \
    RAGSVC__GENERATION__BACKEND=extractive
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"
CMD ["ragsvc-serve"]
