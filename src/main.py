import json
import csv
import os
import argparse
import datetime
import socket
import sys
import logging
from config import load_config, ValidationConfig
TEMP_LIMIT  = 73
POWER_LIMIT = 300

logger = logging.getLogger(__name__)


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("validation.log")
        ]
    )
    logger.debug("Logging initialized — level: %s", logging.getLevelName(level))


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU validation tool — checks thermal and power metrics"
    )
    parser.add_argument("--input", default=None,
        help="Path to CSV file (omit to read from real GPUs via pynvml)")
    parser.add_argument("--source", choices=["nvml", "csv", "smi"],
        default=None,
        help="Data source: nvml, csv, or smi (nvidia-smi). Auto-detected if omitted.")
    parser.add_argument("--limit", type=int, default=TEMP_LIMIT,
        help=f"Temperature limit in Celsius (default: {TEMP_LIMIT})")
    parser.add_argument("--power-limit", type=int, default=POWER_LIMIT,
        help=f"Power limit in watts (default: {POWER_LIMIT})")
    parser.add_argument("--output", default="results/results.json",
        help="Path to save the results JSON file")
    parser.add_argument("--devices", default=None,
        help="Comma-separated GPU indices, e.g. 0,1,2 (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Enable debug logging")
    parser.add_argument("--config", default="config/validation_config.yaml",
        help="Path to YAML config file")
    parser.add_argument("--profile", default=None,
        choices=["strict", "standard", "burn_in"],
        help="Validation profile to use (overrides config thresholds)")
    return parser.parse_args()


# ─── DATA SOURCE 1: pynvml ──────────────────────────────────────────────────

def read_metrics_from_nvml(device_indices=None):
    try:
        import pynvml
    except ImportError:
        raise RuntimeError(
            "pynvml not installed. Run: pip install pynvml\n"
            "Or use --input to read from a CSV file instead."
        )

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    logger.debug("Found %d GPU(s) via pynvml", device_count)

    if device_indices is None:
        device_indices = list(range(device_count))

    if not device_indices:
        pynvml.nvmlShutdown()
        raise RuntimeError("No GPU devices found on this machine.")

    metrics = []
    for idx in device_indices:
        if idx >= device_count:
            logger.warning("GPU index %d not found — skipping", idx)
            continue

        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle)

        try:
            pcie_gen   = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
            pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
        except pynvml.NVMLError:
            pcie_gen, pcie_width = None, None

        try:
            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
        except pynvml.NVMLError:
            fan_speed = None

        try:
            perf_state = str(pynvml.nvmlDeviceGetPerformanceState(handle))
        except pynvml.NVMLError:
            perf_state = None

        entry = {
            "name":             pynvml.nvmlDeviceGetName(handle),
            "index":            idx,
            "temp":             pynvml.nvmlDeviceGetTemperature(
                                    handle, pynvml.NVML_TEMPERATURE_GPU),
            "power_watts":      round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1),
            "memory_used_gb":   round(mem.used  / 1024**3, 2),
            "memory_total_gb":  round(mem.total / 1024**3, 2),
            "gpu_util_pct":     util.gpu,
            "mem_util_pct":     util.memory,
            "pcie_gen":         pcie_gen,
            "pcie_width":       pcie_width,
            "fan_speed_pct":    fan_speed,
            "perf_state":       perf_state,
            "source":           "nvml",
        }
        logger.debug("GPU-%d %s: temp=%dC power=%.1fW mem=%.1fGB",
                     idx, entry["name"], entry["temp"],
                     entry["power_watts"], entry["memory_used_gb"])
        metrics.append(entry)

    pynvml.nvmlShutdown()
    logger.info("Collected metrics from %d GPU(s) via pynvml", len(metrics))
    return metrics


# ─── DATA SOURCE 2: CSV ─────────────────────────────────────────────────────

def read_metrics_from_csv(filepath):
    logger.debug("Reading metrics from CSV: %s", filepath)
    metrics = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append({
                "name":             row["name"],
                "index":            None,
                "temp":             int(row["temp"]),
                "power_watts":      int(row["power_watts"]),
                "memory_used_gb":   int(row["memory_used_gb"]),
                "memory_total_gb":  None,
                "gpu_util_pct":     None,
                "mem_util_pct":     None,
                "pcie_gen":         None,
                "pcie_width":       None,
                "fan_speed_pct":    None,
                "perf_state":       None,
                "source":           "csv",
            })
    logger.info("Loaded %d GPU entries from %s", len(metrics), filepath)
    return metrics


# ─── DATA SOURCE 3: nvidia-smi via subprocess ───────────────────────────────

def read_metrics_from_smi():
    import subprocess, io

    logger.debug("Running nvidia-smi query")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,"
                             "power.draw,memory.used,memory.total,"
                             "utilization.gpu,pstate",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found. Is NVIDIA driver installed?")
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi timed out after 30 seconds.")

    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")

    metrics = []
    reader = csv.reader(io.StringIO(result.stdout))
    for row in reader:
        if len(row) < 8:
            continue
        entry = {
            "name":             row[1].strip(),
            "index":            int(row[0].strip()),
            "temp":             int(row[2].strip()),
            "power_watts":      float(row[3].strip()),
            "memory_used_gb":   round(int(row[4].strip()) / 1024, 2),
            "memory_total_gb":  round(int(row[5].strip()) / 1024, 2),
            "gpu_util_pct":     int(row[6].strip()),
            "mem_util_pct":     None,
            "pcie_gen":         None,
            "pcie_width":       None,
            "fan_speed_pct":    None,
            "perf_state":       row[7].strip(),
            "source":           "nvidia-smi",
        }
        logger.debug("SMI GPU-%d %s: temp=%dC power=%.1fW",
                     entry["index"], entry["name"],
                     entry["temp"], entry["power_watts"])
        metrics.append(entry)

    logger.info("Collected metrics from %d GPU(s) via nvidia-smi", len(metrics))
    return metrics


# ─── ANALYSIS ───────────────────────────────────────────────────────────────

def analyze_metrics(data, limit, power_limit):
    temps = [d["temp"] for d in data]
    avg   = sum(temps) / len(temps)

    results = []
    for d in data:
        temp_ok  = d["temp"]        <= limit
        power_ok = d["power_watts"] <= power_limit
        status   = "PASS" if temp_ok and power_ok else "FAIL"

        reasons = []
        if not temp_ok:
            reasons.append(f"temp {d['temp']}>{limit}C")
        if not power_ok:
            reasons.append(f"power {d['power_watts']}W>{power_limit}W")

        logger.debug("%s: temp=%dC power=%.1fW → %s",
                     d["name"], d["temp"], d["power_watts"], status)

        results.append({
            "name":            d["name"],
            "index":           d.get("index"),
            "temp":            d["temp"],
            "power_watts":     d["power_watts"],
            "memory_used_gb":  d["memory_used_gb"],
            "memory_total_gb": d.get("memory_total_gb"),
            "gpu_util_pct":    d.get("gpu_util_pct"),
            "pcie_gen":        d.get("pcie_gen"),
            "fan_speed_pct":   d.get("fan_speed_pct"),
            "perf_state":      d.get("perf_state"),
            "status":          status,
            "fail_reason":     ", ".join(reasons) if reasons else "none",
        })

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed

    if failed > 0:
        logger.warning("%d of %d GPU(s) FAILED validation", failed, len(results))
        for r in results:
            if r["status"] == "FAIL":
                logger.warning("  FAIL: %s — %s", r["name"], r["fail_reason"])
    else:
        logger.info("All %d GPU(s) PASSED validation", passed)

    return {
        "avg_temp":    round(avg, 2),
        "max_temp":    max(temps),
        "limit":       limit,
        "power_limit": power_limit,
        "overall":     "PASS" if passed == len(results) else "FAIL",
        "summary": {
            "total":     len(results),
            "passed":    passed,
            "failed":    failed,
            "pass_rate": f"{round((passed / len(results)) * 100, 1)}%",
        },
        "details": results,
    }


# ─── METADATA ───────────────────────────────────────────────────────────────

def build_metadata(source, input_path=None):
    return {
        "timestamp":      datetime.datetime.now().isoformat(),
        "hostname":       socket.gethostname(),
        "python_version": sys.version.split()[0],
        "source":         source,
        "input_file":     input_path,
        "tool_version":   "3.0.0",
    }


# ─── OUTPUT ─────────────────────────────────────────────────────────────────

def save_results(result, output_path, metadata):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {"metadata": metadata, "results": result}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        logger.info("Results saved → %s", output_path)

    # ✅ define it here
    summary_line = f"Overall: {result['overall']} | Pass rate: {result['summary']['pass_rate']}"

    logger.info(summary_line)
    print(summary_line)


# ─── ENTRY POINT ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)

    # Load config — CLI args override config file values
    cfg = load_config(args.config, profile=args.profile)

    # CLI args take priority over config file
    temp_limit  = args.limit  if args.limit  != TEMP_LIMIT  else cfg.thresholds.temp_limit_c
    power_limit = args.power_limit if args.power_limit != POWER_LIMIT else cfg.thresholds.power_limit_w

    device_indices = None
    if args.devices:
        device_indices = [int(x.strip()) for x in args.devices.split(",")]

    source = args.source
    if source is None:
        source = "csv" if args.input else "nvml"

    if source == "csv":
        if not args.input:
            logger.error("--input is required when --source csv is used")
            sys.exit(1)
        metrics  = read_metrics_from_csv(args.input)
        metadata = build_metadata("csv", args.input)
    elif source == "smi":
        metrics  = read_metrics_from_smi()
        metadata = build_metadata("nvidia-smi")
    else:
        metrics  = read_metrics_from_nvml(device_indices)
        metadata = build_metadata("nvml")

    result = analyze_metrics(metrics, temp_limit, power_limit)
    save_results(result, args.output or cfg.tool.output_dir + "/results.json", metadata)