#!/usr/bin/env bash
# End-to-end smoke test for the recsys, one tenant + its Qdrant collection.
#
# Tests each component in isolation, then the wired chain:
#   Qdrant collection -> tenant routing -> ingest -> user model -> recommend
#   -> tag filter -> geo -> reward loop (request_id echo) -> metrics
#
# Usage:
#   HOST=https://recsys-api.dev.memorise.sdu.dk \
#   TENANT=westerbork-ar-ai COLLECTION=westerbork-ar-ai \
#   INGEST_API_KEY=...  \
#   QDRANT_URL=https://vectordb.dev.memorise.sdu.dk QDRANT_API_KEY=... \
#   FILTER_TAG=AiARLocationBarrack3 NEAR_LAT=52.7579 NEAR_LON=9.9048 \
#   bash scripts/e2e_smoke.sh
#
# Only HOST + TENANT are strictly required for the API checks. QDRANT_* enable the
# direct content-store checks; INGEST_API_KEY enables the guarded checks (tenants, ingest).

set -uo pipefail

# Pick a python that actually RUNS (skip the Windows Store 'python3' stub that just errors).
PY=""; for c in python python3 py; do if "$c" -c "" >/dev/null 2>&1; then PY="$c"; break; fi; done
[ -z "$PY" ] && { echo "No working python found (need python/python3)"; exit 2; }
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # Windows: avoid cp1252 decode errors on UTF-8 content

HOST="${HOST:-http://localhost:8000}"
TENANT="${TENANT:-default}"
COLLECTION="${COLLECTION:-}"
INGEST_API_KEY="${INGEST_API_KEY:-}"
QDRANT_URL="${QDRANT_URL:-}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
FILTER_TAG="${FILTER_TAG:-}"
NEAR_LAT="${NEAR_LAT:-}"
NEAR_LON="${NEAR_LON:-}"
USER_ID="${USER_ID:-smoke_$RANDOM}"

H_TEN=(-H "X-Tenant-Id: $TENANT")
H_KEY=(); [ -n "$INGEST_API_KEY" ] && H_KEY=(-H "X-API-Key: $INGEST_API_KEY")
H_JSON=(-H "Content-Type: application/json")

pass=0; fail=0; skip=0
ok()   { echo "  PASS  $1"; pass=$((pass+1)); }
no()   { echo "  FAIL  $1"; fail=$((fail+1)); }
sk()   { echo "  SKIP  $1"; skip=$((skip+1)); }
hdr()  { echo; echo "== $1"; }
# pyq <jq-ish python expr on stdin var d> -> prints result
pyq()  { "$PY" -c "import sys,json; d=json.load(sys.stdin); print($1)" 2>/dev/null; }

# ----------------------------------------------------------------------------
hdr "0. API liveness"
# /api/recommend is allow-listed by the oauth2-proxy gate; /docs,/metrics are NOT (401).
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$HOST/api/recommend?user_id=ping" "${H_TEN[@]}" || echo 000)
[ "$code" = "200" ] && ok "API reachable ($HOST)" || no "API not reachable ($HOST) code=$code"

# ----------------------------------------------------------------------------
hdr "1. Qdrant content store (direct)"
if [ -n "$QDRANT_URL" ] && [ -n "$COLLECTION" ]; then
  body=""; for _try in 1 2 3; do
    body=$(curl -s --max-time 30 "$QDRANT_URL/collections/$COLLECTION" -H "api-key: $QDRANT_API_KEY")
    echo "$body" | grep -q '"result"' && break
  done
  exists=$(echo "$body" | pyq "bool(d.get('result'))")
  pts=$(echo "$body" | pyq "d['result'].get('points_count')")
  idx=$(echo "$body" | pyq "','.join((d['result'].get('payload_schema') or {}).keys())")
  [ "$exists" = "True" ] && ok "collection '$COLLECTION' exists, points=$pts" || no "collection '$COLLECTION' missing"
  echo "$idx" | grep -q "tag_values" && ok "tag_values keyword index present (filter works)" \
                                      || no "NO tag_values index -> tag filter will return empty"
  echo "$idx" | grep -q "locations"  && ok "locations geo index present (geo works)" \
                                      || sk "no locations geo index (geo filter inert)"
else
  sk "QDRANT_URL/COLLECTION not set — skipping direct content-store check"
fi

# ----------------------------------------------------------------------------
hdr "2. Tenant routing (is the engine using YOUR collection?)"
if [ -n "$INGEST_API_KEY" ]; then
  body=$(curl -s --max-time 25 "$HOST/api/tenants" "${H_KEY[@]}")
  match=$(echo "$body" | "$PY" -c "import sys,json;d=json.load(sys.stdin);ts=d.get('result',d if isinstance(d,list) else []);print(next((f\"{t.get('collection')}|{t.get('content_count')}\" for t in ts if t.get('tenant_id')=='$TENANT'),''))" 2>/dev/null)
  if [ -n "$match" ]; then
    col="${match%%|*}"; cnt="${match##*|}"
    [ "$col" = "${COLLECTION:-$col}" ] && ok "tenant '$TENANT' -> collection '$col' (content_count=$cnt)" \
                                       || no "tenant '$TENANT' maps to '$col', expected '$COLLECTION'"
  else
    no "tenant '$TENANT' NOT registered -> falls back to DEFAULT collection. Register: POST /api/tenants {tenant_id,collection}"
  fi
else
  sk "INGEST_API_KEY not set — cannot read /api/tenants (guarded)"
fi

# ----------------------------------------------------------------------------
hdr "3. Recommend (cold user, semantic/cold path)"
body=$(curl -s --max-time 30 "$HOST/api/recommend?user_id=${USER_ID}" "${H_TEN[@]}")
n=$(echo "$body" | pyq "len(d['result']['items'])")
strat=$(echo "$body" | pyq "d['result']['strategy']")
[ "${n:-0}" -gt 0 ] 2>/dev/null && ok "recommend returned $n items (strategy=$strat)" \
                                || no "recommend empty (strategy=$strat) — collection has content? tenant routed?"

# ----------------------------------------------------------------------------
hdr "4. Tag filter"
if [ -n "$FILTER_TAG" ]; then
  body=$(curl -s --max-time 30 "$HOST/api/recommend?user_id=${USER_ID}&filter=${FILTER_TAG}" "${H_TEN[@]}")
  n=$(echo "$body" | pyq "len(d['result']['items'])")
  reason=$(echo "$body" | pyq "d['result']['diagnostics'].get('reason')")
  [ "${n:-0}" -gt 0 ] 2>/dev/null && ok "filter '$FILTER_TAG' -> $n items" \
     || no "filter '$FILTER_TAG' empty (reason=$reason). Wrong collection (tenant fallback) OR tag absent OR no tag_values index."
else
  sk "FILTER_TAG not set"
fi

# ----------------------------------------------------------------------------
hdr "5. Geo proximity"
if [ -n "$NEAR_LAT" ] && [ -n "$NEAR_LON" ]; then
  body=$(curl -s --max-time 30 "$HOST/api/recommend?user_id=${USER_ID}&near_lat=${NEAR_LAT}&near_lon=${NEAR_LON}&geo_radius_m=5000" "${H_TEN[@]}")
  n=$(echo "$body" | pyq "len(d['result']['items'])")
  [ "${n:-0}" -gt 0 ] 2>/dev/null && ok "geo near ($NEAR_LAT,$NEAR_LON) -> $n items" \
                                  || sk "geo empty (few/no geo-tagged points in radius)"
else
  sk "NEAR_LAT/NEAR_LON not set"
fi

# ----------------------------------------------------------------------------
hdr "6. Ingest webhook (events -> Redis user model)"
if [ -n "$INGEST_API_KEY" ]; then
  TS=$("$PY" -c "from datetime import datetime,timezone,timedelta;print((datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat())")
  CID=$(echo "$body" | pyq "d['result']['items'][0]['id']" 2>/dev/null)
  CID="${CID:-A1}"
  payload=$("$PY" -c "import json,sys;
ts='$TS';u='$USER_ID';c='$CID'
print(json.dumps([
 {'event':'CONTENT_VIEW_STARTED','userId':u,'timestamp':ts,'properties':{'content':{'content_id':c}}},
 {'event':'CONTENT_VIEW_ENDED','userId':u,'timestamp':ts,'properties':{'content':{'content_id':c},'details':{'reason':'next_button','dwell_seconds':120}}},
]))")
  resp=$(curl -s --max-time 25 -X POST "$HOST/api/ingest" "${H_TEN[@]}" "${H_KEY[@]}" "${H_JSON[@]}" -d "$payload")
  ing=$(echo "$resp" | pyq "d.get('ingested')")
  [ "${ing:-0}" -ge 2 ] 2>/dev/null && ok "ingest accepted $ing events (content_id=$CID)" \
                                     || no "ingest failed: $resp"
else
  sk "INGEST_API_KEY not set — ingest is guarded"
fi

# ----------------------------------------------------------------------------
hdr "7. User model (events materialized in Redis, per-tenant)"
body=$(curl -s --max-time 25 "$HOST/api/usermodel?user_id=${USER_ID}" "${H_TEN[@]}")
hasmodel=$(echo "$body" | pyq "d.get('result') is not None")
[ "$hasmodel" = "True" ] && ok "user model present for '$USER_ID' (events wired through)" \
                          || sk "no user model yet (ingest skipped, or async lag)"

# ----------------------------------------------------------------------------
hdr "8. Reward loop (request_id echo -> bandit)"
if [ -n "$INGEST_API_KEY" ]; then
  rec=$(curl -s --max-time 30 "$HOST/api/recommend?user_id=${USER_ID}" "${H_TEN[@]}")
  RID=$(echo "$rec" | pyq "d['result']['request_id']")
  RCID=$(echo "$rec" | pyq "d['result']['items'][0]['id']" 2>/dev/null)
  if [ -n "${RID:-}" ] && [ -n "${RCID:-}" ]; then
    TS=$("$PY" -c "from datetime import datetime,timezone;print(datetime.now(timezone.utc).isoformat())")
    rew=$("$PY" -c "import json;print(json.dumps([{'event':'CONTENT_VIEW_ENDED','userId':'$USER_ID','timestamp':'$TS','properties':{'content':{'content_id':'$RCID'},'details':{'reason':'next_button','dwell_seconds':90,'request_id':'$RID'}}}]))")
    resp=$(curl -s --max-time 25 -X POST "$HOST/api/ingest" "${H_TEN[@]}" "${H_KEY[@]}" "${H_JSON[@]}" -d "$rew")
    bu=$(echo "$resp" | pyq "d.get('bandit_updates')")
    [ "${bu:-0}" -ge 1 ] 2>/dev/null && ok "reward joined request_id -> bandit_updates=$bu" \
       || sk "no bandit update (ranking_mode may be static, or request_id not echoed) resp=$resp"
  else
    sk "could not get request_id/content_id from recommend"
  fi
else
  sk "INGEST_API_KEY not set"
fi

# ----------------------------------------------------------------------------
hdr "9. Metrics (durable, per-tenant)"
body=$(curl -s --max-time 25 "$HOST/api/metrics" "${H_TEN[@]}")
src=$(echo "$body" | pyq "d.get('source')")
recs=$(echo "$body" | pyq "(d.get('durable') or d).get('recommends')")
echo "  metrics source=$src recommends=$recs"
[ -n "${src:-}" ] && ok "metrics endpoint up (source=$src)" || sk "metrics shape unexpected"

# ----------------------------------------------------------------------------
echo; echo "================  PASS=$pass  FAIL=$fail  SKIP=$skip  ================"
[ "$fail" -eq 0 ] && echo "All wired components green." || echo "See FAIL lines above."
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
