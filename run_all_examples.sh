#!/usr/bin/env bash
# Runs the bundled GameDirector examples one after another (all eight by default) and prints a summary table.
#
# Usage:   bash run_all_examples.sh [--ticks N] [example ...]
#   example   any of nrftw_01 nrftw_02 nrftw_03 hk_01 hk_02 hk_03 hk_04 hk_05 (default: all eight)
#   --ticks N run every selected example for N ticks (default: each example's own, 12, or 6 for nrftw_03)
#
# Each example runs through run_example.sh, so its settings apply (EXAMPLE_LLM_DEVICE, EXAMPLE_LLM_TIMEOUT,
# EXAMPLE_EXTRA_ARGS, EXAMPLE_OUTPUT_ROOT, EXAMPLE_DRY_RUN, GAMEDIRECTOR_LLM_MODEL). A failed example does not stop the others.
# Results: <EXAMPLE_OUTPUT_ROOT>/<game>/<example>/rollout.mp4; the log of each run and summary.txt are in
# <EXAMPLE_OUTPUT_ROOT>/_logs/.
#
# The Gemma planner is one service shared by all runs. A service that is already running is reused and left running; if none is
# running, the first run starts one and this script stops it when the last example has finished.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

all_examples=(nrftw_01 nrftw_02 nrftw_03 hk_01 hk_02 hk_03 hk_04 hk_05)
selected=()
ticks=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ticks) ticks="${2:?--ticks needs a number}"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) selected+=("$1"); shift ;;
  esac
done
if [[ ${#selected[@]} -eq 0 ]]; then
  selected=("${all_examples[@]}")
fi

output_root="${EXAMPLE_OUTPUT_ROOT:-output/local_runs}"
log_dir="${output_root}/_logs"
mkdir -p "${log_dir}"
summary="${log_dir}/summary.txt"
service_url="http://127.0.0.1:${GEMMA_PORT:-8018}/v1/models"

service_up() {
  curl -s -m 3 -o /dev/null "${service_url}"
}

service_was_running=0
if service_up; then
  service_was_running=1
  echo "A Gemma service is already running on ${service_url}: it will be reused and left running."
fi

declare -a names statuses durations finals
failures=0
for name in "${selected[@]}"; do
  log="${log_dir}/${name}.log"
  echo
  echo "################ ${name} ($(date +%H:%M:%S)) ################"
  started=${SECONDS}
  if [[ -n "${ticks}" ]]; then
    bash run_example.sh "${name}" "${ticks}" 2>&1 | tee "${log}"
  else
    bash run_example.sh "${name}" 2>&1 | tee "${log}"
  fi
  code=${PIPESTATUS[0]}
  names+=("${name}")
  durations+=("$((SECONDS - started))")
  finals+=("$(grep -E '^Completed ' "${log}" | tail -1 | sed 's/^Completed //' | cut -c1-110)")
  if [[ ${code} -eq 0 ]]; then
    statuses+=("ok")
  else
    statuses+=("FAILED(${code})")
    failures=$((failures + 1))
  fi
done

# stop the Gemma service only if this script is the reason it exists
if [[ ${service_was_running} -eq 0 && -f output/llm/gemma_server.pid ]]; then
  pid="$(cat output/llm/gemma_server.pid)"
  if [[ -n "${pid}" ]] && ps -p "${pid}" -o cmd= 2>/dev/null | grep -q gemma_server; then
    echo "Stopping the Gemma service this script started (PID ${pid})."
    kill "${pid}" || true
  fi
fi

{
  printf '%-9s %-12s %8s  %s\n' example status seconds "result"
  for i in "${!names[@]}"; do
    printf '%-9s %-12s %8s  %s\n' "${names[$i]}" "${statuses[$i]}" "${durations[$i]}" "${finals[$i]}"
  done
  echo "${#names[@]} examples, ${failures} failed. Logs: ${log_dir}"
} | tee "${summary}"

[[ ${failures} -eq 0 ]]
