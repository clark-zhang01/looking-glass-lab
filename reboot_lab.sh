#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/clark-zhang/looking_glass"
PORT="8000"
VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"
WEBXR_PATH="$ROOT_DIR/static/libs/webxr.js"
WEBXR_URL="https://unpkg.com/@lookingglass/webxr@0.6.0/dist/webxr.js"
MODEL_PATH="/home/clark-zhang/.cache/huggingface/hub/models--stabilityai--stable-fast-3d/snapshots/f0c9a8ffd62cb1bbc8a7a53c9f87a0be1b6be778/"
LOG_FILE="$ROOT_DIR/server_debug.log"
STARTUP_TIMEOUT_SECONDS="25"
SERVER_IP="${LAB_SERVER_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"

cd "$ROOT_DIR"

echo "[reboot_lab] Cleaning up port $PORT..."
if command -v lsof >/dev/null 2>&1; then
    PORT_PIDS="$(lsof -ti tcp:"$PORT" || true)"
    if [[ -n "$PORT_PIDS" ]]; then
        KILLED_PIDS=()
        SKIPPED_PIDS=()
        for pid in $PORT_PIDS; do
            cmdline="$(ps -p "$pid" -o args= 2>/dev/null || true)"
            if [[ "$cmdline" == *"python app.py"* ]] || [[ "$cmdline" == *"uvicorn"* ]] || [[ "$cmdline" == *"/looking_glass/"* ]]; then
                kill "$pid" >/dev/null 2>&1 || true
                sleep 1
                if kill -0 "$pid" >/dev/null 2>&1; then
                    kill -9 "$pid" >/dev/null 2>&1 || true
                fi
                KILLED_PIDS+=("$pid")
            else
                SKIPPED_PIDS+=("$pid")
            fi
        done

        if [[ ${#KILLED_PIDS[@]} -gt 0 ]]; then
            echo "[reboot_lab] Killed app process(es) on port $PORT: ${KILLED_PIDS[*]}"
        fi
        if [[ ${#SKIPPED_PIDS[@]} -gt 0 ]]; then
            echo "[reboot_lab] Skipped non-app process(es) on port $PORT: ${SKIPPED_PIDS[*]}"
            echo "[reboot_lab] Use FORCE_KILL_ALL_ON_PORT=1 to force-kill all listeners on port $PORT."
        fi
    else
        echo "[reboot_lab] No process found on port $PORT."
    fi
else
    echo "[reboot_lab] lsof not found; skipping explicit port cleanup."
fi

if [[ "${FORCE_KILL_ALL_ON_PORT:-0}" == "1" ]] && command -v fuser >/dev/null 2>&1; then
    echo "[reboot_lab] FORCE_KILL_ALL_ON_PORT=1 set; force-killing all listeners on ${PORT}/tcp"
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
fi

echo "[reboot_lab] Activating virtual environment..."
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "[reboot_lab] ERROR: venv activate script not found at $VENV_ACTIVATE"
    exit 1
fi
source "$VENV_ACTIVATE"

echo "[reboot_lab] Checking Looking Glass WebXR bridge library..."
mkdir -p "$(dirname "$WEBXR_PATH")"
if [[ ! -f "$WEBXR_PATH" ]]; then
    echo "[reboot_lab] webxr.js not found. Downloading..."
    curl -L -o "$WEBXR_PATH" "$WEBXR_URL"
    echo "[reboot_lab] Downloaded $WEBXR_PATH"
else
    echo "[reboot_lab] Found existing $WEBXR_PATH"
fi

echo "[reboot_lab] Exporting model path..."
export STABLE_FAST_3D_MODEL_PATH="$MODEL_PATH"

echo "[reboot_lab] Starting server and logging to $LOG_FILE ..."
nohup python app.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

echo "[reboot_lab] Server started with PID $SERVER_PID"
CHECK_URLS=("http://127.0.0.1:${PORT}" "http://localhost:${PORT}")
if [[ -n "${SERVER_IP}" ]]; then
    CHECK_URLS+=("http://${SERVER_IP}:${PORT}")
fi

echo "[reboot_lab] Verifying startup on: ${CHECK_URLS[*]}"

for ((i=1; i<=STARTUP_TIMEOUT_SECONDS; i++)); do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "[reboot_lab] ERROR: Server process exited during startup."
        tail -n 60 "$LOG_FILE" || true
        exit 1
    fi

    PORT_LISTENING="false"
    if command -v ss >/dev/null 2>&1; then
        if ss -ltn "sport = :${PORT}" | awk 'NR>1 {exit 0} END {exit 1}'; then
            PORT_LISTENING="true"
        fi
    fi

    if command -v curl >/dev/null 2>&1; then
        for check_url in "${CHECK_URLS[@]}"; do
            HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' "$check_url" || true)"
            if [[ "$HTTP_CODE" =~ ^(200|301|302|404)$ ]]; then
                echo "[reboot_lab] Startup verified via $check_url (HTTP $HTTP_CODE)."
                echo "[reboot_lab] Tail logs with: tail -f $LOG_FILE"
                exit 0
            fi
        done
    fi

    if [[ "$PORT_LISTENING" == "true" ]]; then
        echo "[reboot_lab] Port ${PORT} is listening; app is still warming up."
    fi

    sleep 1
done

echo "[reboot_lab] ERROR: Timed out waiting for server on port ${PORT}."
echo "[reboot_lab] Checked URLs: ${CHECK_URLS[*]}"
tail -n 60 "$LOG_FILE" || true
exit 1
