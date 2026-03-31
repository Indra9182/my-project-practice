import json
import csv
import os
import argparse

TEMP_LIMIT = 73

def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU validation tool — checks thermal metrics against a threshold"
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

def analyze_metrics(data, limit):
    temps = [d["temp"] for d in data]
    avg = sum(temps) / len(temps)

    results = []
    for d in data:
        status = "PASS" if d["temp"] <= limit else "FAIL"
        results.append({
            "name": d["name"],
            "temp": d["temp"],
            "power_watts": d["power_watts"],
            "memory_used_gb": d["memory_used_gb"],
            "status": status
        })

    return {
        "avg_temp": round(avg, 2),
        "max_temp": max(temps),
        "limit": limit,
        "overall": "PASS" if max(temps) <= limit else "FAIL",
        "details": results
    }

def save_results(result, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {output_path}")
    print(f"Overall: {result['overall']}")

args = parse_args()
metrics = read_metrics_from_csv(args.input)
result = analyze_metrics(metrics, args.limit)
save_results(result, args.output)