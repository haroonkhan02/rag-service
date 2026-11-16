#!/usr/bin/env bash
# Start the image, wait until it is ready, and check the API from outside the container.
#   scripts/docker_smoke.sh rag-service:ci
set -euo pipefail

IMAGE="${1:?usage: docker_smoke.sh <image>}"
PORT="${PORT:-8000}"
NAME="rag-service-smoke"
URL="http://127.0.0.1:${PORT}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
# no network at run time: everything the service needs must already be in the image
docker run -d --name "$NAME" --network none -p "${PORT}:8000" "$IMAGE" >/dev/null 2>&1 \
  || { docker rm -f "$NAME" >/dev/null 2>&1 || true
       docker run -d --name "$NAME" -p "${PORT}:8000" "$IMAGE" >/dev/null; OFFLINE=0; }
trap 'docker logs "$NAME" | tail -n 30; docker rm -f "$NAME" >/dev/null' EXIT

# with --network none the port is not published, so talk to the service from inside
call() {
  if [ "${OFFLINE:-1}" = "1" ]; then
    docker exec "$NAME" python -c "
import sys, urllib.request
req = urllib.request.Request('http://127.0.0.1:8000' + sys.argv[1],
                             data=sys.argv[2].encode() if len(sys.argv) > 2 else None,
                             headers={'content-type': 'application/json'})
print(urllib.request.urlopen(req, timeout=60).read().decode())" "$@"
  else
    if [ $# -gt 1 ]; then curl -fsS "${URL}$1" -H 'content-type: application/json' -d "$2"
    else curl -fsS "${URL}$1"; fi
  fi
}

for i in $(seq 1 90); do
  if call /readyz >/dev/null 2>&1; then break; fi
  if [ "$i" = 90 ]; then echo "service did not become ready"; exit 1; fi
  sleep 2
done

READY="$(call /readyz)"
SEARCH="$(call /search '{"query": "How do I record only two topics into a bag?"}')"
AGAIN="$(call /search '{"query": "how do i record only two topics into a bag"}')"
ASK="$(call /ask '{"query": "What replaces latched topics in ROS 2?", "stream": false}')"

READY="$READY" SEARCH="$SEARCH" AGAIN="$AGAIN" ASK="$ASK" python3 - <<'PY'
import json, os

ready, search, again, ask = (json.loads(os.environ[k]) for k in ("READY", "SEARCH", "AGAIN", "ASK"))
assert ready["status"] == "ready" and ready["chunks"] == 3680, ready
assert ready["generator"] == "extractive", ready
assert search["cache"] == "miss" and len(search["passages"]) == 5, search["cache"]
assert "rerank" in search["timings_ms"], search["timings_ms"]
assert any("Recording-And-Playing-Back-Data" in p["doc_id"] for p in search["passages"]), \
    [p["doc_id"] for p in search["passages"]]
assert again["cache"] == "exact", again["cache"]
assert ask["citations"] == [1] and ask["answer"], ask
print("chunks:", ready["chunks"], "| index:", ready["index_sha256"])
print("search timings (ms):", search["timings_ms"])
print("top passage:", search["passages"][0]["doc_id"])
PY

echo "user in container: $(docker exec "$NAME" id -un)"
[ "$(docker exec "$NAME" id -u)" != "0" ] || { echo "container runs as root"; exit 1; }
echo "network: $([ "${OFFLINE:-1}" = "1" ] && echo none || echo bridge)"
echo "image size: $(docker image inspect "$IMAGE" --format '{{.Size}}' | awk '{printf "%.2f GB", $1/1e9}')"
