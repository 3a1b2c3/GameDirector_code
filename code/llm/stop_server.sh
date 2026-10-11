#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$(dirname "${HERE}")/output/llm/gemma_server.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "No GameDirector-managed Gemma PID file found."
  exit 0
fi
PID="$(tr -d '[:space:]' < "${PID_FILE}")"
if [[ "${PID}" =~ ^[0-9]+$ ]] && kill -0 "${PID}" 2>/dev/null; then
  kill "${PID}"
  echo "Stopped Gemma server PID ${PID}."
else
  echo "Gemma server PID ${PID} is not running."
fi
rm -f "${PID_FILE}"
