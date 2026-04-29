#!/usr/bin/env bash
set -euo pipefail

echo "=== NRA E2E Live Server Test ==="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Build
cargo build --release -p nra-cli -p nra-registry-server

# 2. Create test data
TMPDIR=$(mktemp -d)
for i in $(seq 1 20); do
    echo "Test content for file $i — $(date)" > "$TMPDIR/file_$i.txt"
done

# 3. Start server in background
mkdir -p nra-registry-server/data
./target/release/nra-registry-server &
SERVER_PID=$!

# 4. Cleanup on exit (even on failure)
cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    rm -rf "$TMPDIR"
    [ -n "${OUTDIR:-}" ] && rm -rf "$OUTDIR"
    rm -f "$PROJECT_ROOT/e2e_test.nra"
}
trap cleanup EXIT

# Wait for server to start
MAX_WAIT=5
WAIT_COUNT=0
while ! curl -s http://127.0.0.1:3000 >/dev/null 2>&1; do
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT+1))
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "❌ FAIL: Server failed to start within ${MAX_WAIT}s"
        exit 1
    fi
done

# 5. Push via CLI
echo "--- Pushing dataset..."
./target/release/nra-cli push \
    --input "$TMPDIR" \
    --url http://127.0.0.1:3000/api/v1/upload/e2e_test

# 6. Verify file created
if [ ! -f "nra-registry-server/data/e2e_test.nra" ]; then
    echo "❌ FAIL: Archive not created on server"
    exit 1
fi

# 7. Download and extract via CLI
OUTDIR=$(mktemp -d)
echo "--- Streaming dataset back..."
curl -sO http://127.0.0.1:3000/archives/e2e_test.nra
./target/release/nra-cli unpack-beta \
    --input e2e_test.nra \
    --output "$OUTDIR"

# 8. Compare number of files
ORIGINAL_COUNT=$(ls "$TMPDIR" | wc -l | tr -d ' ')
EXTRACTED_COUNT=$(ls "$OUTDIR" | wc -l | tr -d ' ')

if [ "$ORIGINAL_COUNT" -eq "$EXTRACTED_COUNT" ]; then
    echo "✅ PASS: $EXTRACTED_COUNT files roundtripped successfully"
else
    echo "❌ FAIL: Expected $ORIGINAL_COUNT files, got $EXTRACTED_COUNT"
    exit 1
fi

# 9. Byte-for-byte comparison of all files
for f in "$TMPDIR"/*; do
    basename=$(basename "$f")
    if ! diff -q "$f" "$OUTDIR/$basename" > /dev/null 2>&1; then
        echo "❌ FAIL: Content mismatch for $basename"
        exit 1
    fi
done

echo "✅ ALL E2E TESTS PASSED"
