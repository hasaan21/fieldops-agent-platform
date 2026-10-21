#!/usr/bin/env bash
# Post-deploy smoke test: liveness, readiness, auth, and one real agent turn.
#   scripts/smoke_test.sh https://fieldops-staging-api.<region>.azurecontainerapps.io <api-key>
set -euo pipefail

URL="${1:?base url}"
KEY="${2:?api key}"

retry() {  # new revisions take a moment to pass readiness
  for i in $(seq 1 30); do
    if "$@"; then return 0; fi
    sleep 5
  done
  return 1
}

echo "liveness";  retry curl -fsS "$URL/healthz" > /dev/null
echo "readiness"; retry curl -fsS "$URL/readyz" > /dev/null

echo "auth rejects a bad key"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/v1/agents/job-intake/messages" \
  -H 'x-api-key: not-a-key' -H 'content-type: application/json' \
  -d '{"sender": "smoke", "text": "hi"}')
[ "$code" = "401" ] || { echo "expected 401, got $code"; exit 1; }

echo "agent turn (question path: no side effects)"
body=$(curl -fsS -X POST "$URL/v1/agents/job-intake/messages" \
  -H "x-api-key: $KEY" -H 'content-type: application/json' \
  -d '{"channel": "web", "sender": "smoke-test", "text": "What are your opening hours?"}')
echo "$body" | python3 -c '
import json, sys
r = json.load(sys.stdin)
assert r.get("status") == "done", r
assert r.get("outcome") == "answered", r
assert r.get("reply"), r
print("ok:", r["outcome"], "| latency", r["usage"]["latency_ms"], "ms | cost $", r["usage"]["cost_usd"])
'
