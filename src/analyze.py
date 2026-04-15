import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import argparse
import logging

logger = logging.getLogger(__name__)

TEMP_LIMIT  = 73
POWER_LIMIT = 300


def load_csv(filepath):
    """Load multi-run GPU metrics from a CSV file."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    logger.info("Loaded %d rows from %s", len(df), filepath)
    logger.info("Columns: %s", list(df.columns))
    return df


def basic_stats(df):
    """Print summary statistics for all numeric columns."""
    print("\n── Basic statistics ──────────────────────")
    print(df[["temp", "power_watts", "memory_used_gb", "gpu_util_pct"]].describe().round(2))


def per_gpu_stats(df):
    """Show mean metrics grouped by GPU name."""
    print("\n── Per-GPU averages ──────────────────────")
    grouped = df.groupby("gpu_name")[
        ["temp", "power_watts", "memory_used_gb", "gpu_util_pct"]
    ].mean().round(2)
    print(grouped)
    return grouped


def flag_failures(df, temp_limit=TEMP_LIMIT, power_limit=POWER_LIMIT):
    """Add a status column — PASS or FAIL — to the dataframe."""
    df = df.copy()
    df["status"] = "PASS"
    df.loc[df["temp"]        > temp_limit,  "status"] = "FAIL"
    df.loc[df["power_watts"] > power_limit, "status"] = "FAIL"

    total  = len(df)
    failed = (df["status"] == "FAIL").sum()
    passed = total - failed

    print(f"\n── Validation results ────────────────────")
    print(f"Total readings : {total}")
    print(f"PASS           : {passed}  ({round(passed/total*100, 1)}%)")
    print(f"FAIL           : {failed}  ({round(failed/total*100, 1)}%)")

    if failed > 0:
        print("\nFailing readings:")
        fails = df[df["status"] == "FAIL"][
            ["run_id", "gpu_name", "temp", "power_watts", "status"]
        ]
        print(fails.to_string(index=False))

    return df


def pass_rate_per_run(df):
    """Calculate pass rate for each run_id."""
    def run_pass_rate(group):
        total  = len(group)
        passed = (group["status"] == "PASS").sum()
        return round(passed / total * 100, 1)

    rates = df.groupby("run_id").apply(run_pass_rate).reset_index()
    rates.columns = ["run_id", "pass_rate_pct"]
    print("\n── Pass rate per run ─────────────────────")
    print(rates.to_string(index=False))
    return rates


def hottest_gpu(df):
    """Find which GPU runs hottest on average."""
    avg_temps = df.groupby("gpu_name")["temp"].mean().round(2)
    hottest   = avg_temps.idxmax()
    print(f"\n── Hottest GPU ───────────────────────────")
    print(f"Hottest GPU: {hottest} (avg {avg_temps[hottest]}C)")
    print(avg_temps.to_string())
    return avg_temps


def plot_dashboard(df, rates, output_path="results/analysis.png"):
    """Generate a 2x2 dashboard of charts."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("GPU Validation Analysis Dashboard", fontsize=16, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"GPU-0": "#4d9eff", "GPU-1": "#00d68f", "GPU-2": "#f87171"}

    # ── Chart 1: Temperature over runs (line chart) ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for gpu, group in df.groupby("gpu_name"):
        ax1.plot(group["run_id"], group["temp"],
                 marker="o", label=gpu,
                 color=colors.get(gpu, "#888"),
                 linewidth=2, markersize=5)
    ax1.axhline(y=TEMP_LIMIT, color="#f87171", linestyle="--",
                linewidth=1.5, label=f"Limit ({TEMP_LIMIT}C)")
    ax1.set_title("Temperature per Run", fontweight="bold")
    ax1.set_xlabel("Run ID")
    ax1.set_ylabel("Temperature (C)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Chart 2: Power usage bar chart ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    avg_power = df.groupby("gpu_name")["power_watts"].mean()
    bars = ax2.bar(avg_power.index, avg_power.values,
                   color=[colors.get(g, "#888") for g in avg_power.index],
                   edgecolor="white", linewidth=0.5)
    ax2.axhline(y=POWER_LIMIT, color="#f87171", linestyle="--",
                linewidth=1.5, label=f"Limit ({POWER_LIMIT}W)")
    for bar, val in zip(bars, avg_power.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.0f}W", ha="center", va="bottom", fontsize=9)
    ax2.set_title("Average Power Usage", fontweight="bold")
    ax2.set_xlabel("GPU")
    ax2.set_ylabel("Power (W)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # ── Chart 3: Pass rate over time (line chart) ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(rates["run_id"], rates["pass_rate_pct"],
             marker="o", color="#76b900", linewidth=2, markersize=6)
    ax3.axhline(y=100, color="#4d9eff", linestyle="--",
                linewidth=1, alpha=0.5, label="100% target")
    ax3.fill_between(rates["run_id"], rates["pass_rate_pct"],
                     alpha=0.15, color="#76b900")
    ax3.set_title("Pass Rate per Run", fontweight="bold")
    ax3.set_xlabel("Run ID")
    ax3.set_ylabel("Pass Rate (%)")
    ax3.set_ylim(0, 110)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Chart 4: Temperature distribution (histogram) ────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for gpu, group in df.groupby("gpu_name"):
        ax4.hist(group["temp"], bins=6, alpha=0.6, label=gpu,
                 color=colors.get(gpu, "#888"), edgecolor="white")
    ax4.axvline(x=TEMP_LIMIT, color="#f87171", linestyle="--",
                linewidth=1.5, label=f"Limit ({TEMP_LIMIT}C)")
    ax4.set_title("Temperature Distribution", fontweight="bold")
    ax4.set_xlabel("Temperature (C)")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    logger.info("Dashboard saved → %s", output_path)
    print(f"\nDashboard saved → {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze GPU validation metrics from CSV data"
    )
    parser.add_argument("--input", default="data/multi_run_metrics.csv",
        help="Path to multi-run metrics CSV")
    parser.add_argument("--output", default="results/analysis.png",
        help="Path to save the dashboard chart")
    parser.add_argument("--temp-limit", type=int, default=TEMP_LIMIT,
        help=f"Temperature threshold (default: {TEMP_LIMIT})")
    parser.add_argument("--power-limit", type=int, default=POWER_LIMIT,
        help=f"Power threshold (default: {POWER_LIMIT})")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    df    = load_csv(args.input)
    basic_stats(df)
    per_gpu_stats(df)
    df    = flag_failures(df, args.temp_limit, args.power_limit)
    rates = pass_rate_per_run(df)
    hottest_gpu(df)
    plot_dashboard(df, rates, args.output)