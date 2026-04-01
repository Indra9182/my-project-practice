import json
import csv
import os
import argparse
import datetime
import socket
import sys

TEMP_LIMIT = 73
POWER_LIMIT = 300

def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU validation tool — checks thermal and power metrics"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the CSV file containing GPU metrics"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=TEMP_LIMIT,
        help=f"Temperature limit in Celsius (default: {TEMP_LIMIT})"
    )
    parser.add_argument(
        "--power-limit",
        type=int,
        default=POWER_LIMIT,
        help=f"Power limit in watts (default: {POWER_LIMIT})"
    )
    parser.add_argument(
        "--output",
        default="results/results.json",
        help="Path to save the results JSON file"
    )
    return parser.parse_args()

def read_metrics_from_csv(filepath):
    metrics = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append({
                "name": row["name"],
                "temp": int(row["temp"]),
                "power_watts": int(row["power_watts"]),
                "memory_used_gb": int(row["memory_used_gb"])
            })
    return metrics
def build_metadata(input_path):
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "input_file": input_path,
        "tool_version": "1.0.0"
    }

def analyze_metrics(data, limit, power_limit):
    temps = [d["temp"] for d in data]
    avg = sum(temps) / len(temps)

    results = []
    for d in data:
        temp_ok = d["temp"] <= limit
        power_ok = d["power_watts"] <= power_limit

        status = "PASS" if temp_ok and power_ok else "FAIL"

        reasons = []
        if not temp_ok:
            reasons.append(f"temp {d['temp']}>{limit}")
        if not power_ok:
            reasons.append(f"power {d['power_watts']}>{power_limit}")

        results.append({
            "name": d["name"],
            "temp": d["temp"],
            "power_watts": d["power_watts"],
            "memory_used_gb": d["memory_used_gb"],
            "status": status,
            "fail_reason": ", ".join(reasons) if reasons else "none"
        })

    passed = sum(1 for d in results if d["status"] == "PASS")
    failed = sum(1 for d in results if d["status"] == "FAIL")
    total = len(results)

    return {
        "avg_temp": round(avg, 2),
        "max_temp": max(temps),
        "limit": limit,
        "power_limit": power_limit,
        "overall": "PASS" if max(temps) <= limit and all(
            d["power_watts"] <= power_limit for d in data
        ) else "FAIL",
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{round((passed / total) * 100, 1)}%"
        },
        "details": results
    }

def save_results(result, output_path, metadata):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        "metadata": metadata,
        "results": result
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")
    print(f"Overall: {result['overall']}")
    print(f"Summary: {result['summary']}")
    print(f"Run timestamp: {metadata['timestamp']}")
if __name__ == "__main__":
    args = parse_args()
    metrics = read_metrics_from_csv(args.input)
    result = analyze_metrics(metrics, args.limit, args.power_limit)
    metadata = build_metadata(args.input)
    save_results(result, args.output, metadata)