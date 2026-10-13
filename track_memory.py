#!/usr/bin/env python3
"""Log GPU and RAM use of a GameDirector run to a CSV every few seconds, so slowdowns from memory pressure are visible.

Columns: time, gpu_used_MiB / gpu_total_MiB / gpu_util_pct / gpu_power_W (nvidia-smi), worldmodel_dedicated_GB and
worldmodel_shared_GB (Windows GPU process counters for the run_closed_loop.py process: "shared" is system RAM the GPU
borrows once its own memory is full, which makes a run crawl), worldmodel_ram_GB, gemma_ram_GB, gemma_cpu_pct (share of
all logical CPUs used by the Gemma service) and free_ram_GB.

Usage:  python track_memory.py [--out output\\memory_track.csv] [--interval 5] [--minutes 120]
Stop it with Ctrl+C. Needs psutil.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

COLUMNS = ["time", "gpu_used_MiB", "gpu_total_MiB", "gpu_util_pct", "gpu_power_W", "worldmodel_dedicated_GB", "worldmodel_shared_GB",
           "worldmodel_ram_GB", "gemma_ram_GB", "gemma_cpu_pct", "free_ram_GB"]
GB = 1024**3


def find_process(marker: str) -> psutil.Process | None:
    """The python.exe whose command line mentions marker and that has the most memory (the real interpreter, not a launcher shim)."""
    best = None
    for process in psutil.process_iter(["name", "cmdline", "memory_info"]):
        command = " ".join(process.info["cmdline"] or [])
        if process.info["name"] == "python.exe" and marker in command:
            if best is None or process.info["memory_info"].rss > best.info["memory_info"].rss:
                best = process
    return best


def gpu_stats() -> list[str]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return [value.strip() for value in result.stdout.strip().split(",")]


def gpu_process_memory(pid: int) -> tuple[str, str]:
    """(dedicated, shared) GPU memory in GB of one process from the Windows GPU Process Memory counters."""
    command = (f"$c = Get-Counter '\\GPU Process Memory(*pid_{pid}_*)\\Dedicated Usage','\\GPU Process Memory(*pid_{pid}_*)\\Shared Usage' "
               "-ErrorAction SilentlyContinue; if ($c) { $d = ($c.CounterSamples | Where-Object { $_.Path -match 'dedicated' } | Measure-Object CookedValue -Sum).Sum; "
               "$s = ($c.CounterSamples | Where-Object { $_.Path -match 'shared' } | Measure-Object CookedValue -Sum).Sum; \"$d $s\" }")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)
    parts = result.stdout.split()
    if len(parts) != 2:
        return "", ""
    return f"{float(parts[0]) / GB:.1f}", f"{float(parts[1]) / GB:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output" / "memory_track.csv")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--minutes", type=float, default=120.0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gemma_cpu_seen: dict[int, float] = {}
    deadline = time.time() + args.minutes * 60
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        while time.time() < deadline:
            started = time.time()
            gpu_used, gpu_total, gpu_util, gpu_power = gpu_stats()
            worldmodel = find_process("run_closed_loop")
            gemma = find_process("gemma_server")
            dedicated, shared = gpu_process_memory(worldmodel.pid) if worldmodel else ("", "")
            worldmodel_ram = f"{worldmodel.memory_info().rss / GB:.1f}" if worldmodel else ""
            gemma_ram = f"{gemma.memory_info().rss / GB:.1f}" if gemma else ""
            gemma_cpu = ""
            if gemma:
                cpu_times = sum(gemma.cpu_times()[:2])
                if gemma.pid in gemma_cpu_seen:
                    gemma_cpu = f"{(cpu_times - gemma_cpu_seen[gemma.pid]) / (time.time() - started + args.interval) * 100 / psutil.cpu_count():.0f}"
                gemma_cpu_seen[gemma.pid] = cpu_times
            free_ram = f"{psutil.virtual_memory().available / GB:.1f}"
            writer.writerow([datetime.now().strftime("%H:%M:%S"), gpu_used, gpu_total, gpu_util, gpu_power, dedicated, shared,
                             worldmodel_ram, gemma_ram, gemma_cpu, free_ram])
            handle.flush()
            time.sleep(max(0.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()
