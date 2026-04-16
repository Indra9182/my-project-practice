"""
ValidationPipeline — orchestrates the complete validation workflow.
This is the top-level entry point for running a full validation run.
"""
import json
import os
import logging
import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Runs a complete validation cycle:
    1. Load config
    2. Read GPU metrics (any source)
    3. Validate against thresholds
    4. Analyze trends if historical data exists
    5. Generate charts
    6. Save structured report
    7. Return exit code (0=pass, 1=fail)
    """

    def __init__(self, config_path: str = "config/validation_config.yaml",
                 profile: str = None):
        from config import load_config
        self.cfg        = load_config(config_path, profile=profile)
        self.profile    = profile
        self.run_id     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.cfg.tool.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Pipeline initialized | run_id=%s | profile=%s",
                    self.run_id, profile or "default")

    def run(self, source: str = "csv", input_path: str = None,
            device_indices: list = None, generate_charts: bool = True) -> int:
        """
        Execute the full validation pipeline.
        Returns 0 if all GPUs pass, 1 if any fail.
        """
        logger.info("=" * 55)
        logger.info("NVIDIA GPU Validation — %s v%s",
                    self.cfg.tool.name, self.cfg.tool.version)
        logger.info("Run ID: %s | Profile: %s",
                    self.run_id, self.profile or "default")
        logger.info("=" * 55)

        # Step 1 — Collect metrics
        metrics = self._collect(source, input_path, device_indices)
        if not metrics:
            logger.error("No metrics collected — aborting")
            return 1

        # Step 2 — Validate
        from main import analyze_metrics, build_metadata, save_results
        result   = analyze_metrics(
            metrics,
            self.cfg.thresholds.temp_limit_c,
            self.cfg.thresholds.power_limit_w
        )
        metadata = build_metadata(source, input_path)
        metadata["run_id"]  = self.run_id
        metadata["profile"] = self.profile or "default"

        # Step 3 — Save JSON report
        report_path = os.path.join(
            self.output_dir, f"results_{self.run_id}.json"
        )
        save_results(result, report_path, metadata)

        # Step 4 — Append to historical CSV for trend analysis
        self._append_to_history(metrics, result)

        # Step 5 — Generate charts if history exists
        if generate_charts:
            self._generate_charts()

        # Step 6 — Print final summary
        self._print_summary(result)

        exit_code = 0 if result["overall"] == "PASS" else 1
        logger.info("Pipeline complete | overall=%s | exit_code=%d",
                    result["overall"], exit_code)
        return exit_code

    def _collect(self, source, input_path, device_indices):
        """Collect GPU metrics from the specified source."""
        import sys
        sys.path.insert(0, os.path.dirname(__file__))

        try:
            if source == "csv":
                from main import read_metrics_from_csv
                logger.info("Source: CSV — %s", input_path)
                return read_metrics_from_csv(input_path)
            elif source == "smi":
                from main import read_metrics_from_smi
                logger.info("Source: nvidia-smi")
                return read_metrics_from_smi()
            elif source == "nvml":
                from main import read_metrics_from_nvml
                logger.info("Source: pynvml")
                return read_metrics_from_nvml(device_indices)
            else:
                logger.error("Unknown source: %s", source)
                return []
        except Exception as e:
            logger.error("Failed to collect metrics: %s", e)
            return []

    def _append_to_history(self, metrics, result):
        """Append this run's metrics to the historical CSV for trend analysis."""
        import csv as csv_mod
        history_path = os.path.join(self.output_dir, "history.csv")
        write_header = not os.path.exists(history_path)

        # Match status back to each GPU
        status_map = {
            d["name"]: d for d in result["details"]
        }

        with open(history_path, "a", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=[
                "run_id", "timestamp", "profile", "gpu_name",
                "temp", "power_watts", "memory_used_gb",
                "gpu_util_pct", "status", "fail_reason"
            ])
            if write_header:
                writer.writeheader()
            for m in metrics:
                detail = status_map.get(m["name"], {})
                writer.writerow({
                    "run_id":          self.run_id,
                    "timestamp":       datetime.datetime.now().isoformat(),
                    "profile":         self.profile or "default",
                    "gpu_name":        m["name"],
                    "temp":            m["temp"],
                    "power_watts":     m["power_watts"],
                    "memory_used_gb":  m["memory_used_gb"],
                    "gpu_util_pct":    m.get("gpu_util_pct", ""),
                    "status":          detail.get("status", ""),
                    "fail_reason":     detail.get("fail_reason", ""),
                })

        logger.debug("History appended → %s", history_path)

    def _generate_charts(self):
        """Generate trend charts from historical data if enough runs exist."""
        history_path = os.path.join(self.output_dir, "history.csv")
        if not os.path.exists(history_path):
            logger.debug("No history yet — skipping charts")
            return

        try:
            import pandas as pd
            df = pd.read_csv(history_path)
            unique_runs = df["run_id"].nunique()

            if unique_runs < 2:
                logger.info("Only %d run(s) in history — need 2+ for trend charts",
                            unique_runs)
                return

            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from analyze import plot_dashboard

            # Calculate pass rates per run for the dashboard
            def pass_rate(grp):
                total  = len(grp)
                passed = (grp["status"] == "PASS").sum()
                return round(passed / total * 100, 1)

            # Add numeric run index for plotting
            run_order = {r: i+1 for i, r in enumerate(df["run_id"].unique())}
            df["run_num"] = df["run_id"].map(run_order)
            df_renamed = df.rename(columns={"run_num": "run_id",
                                             "gpu_name": "gpu_name"})

            rates = df_renamed.groupby("run_id").apply(pass_rate).reset_index()
            rates.columns = ["run_id", "pass_rate_pct"]

            chart_path = os.path.join(self.output_dir, f"trends_{self.run_id}.png")
            plot_dashboard(df_renamed, rates, chart_path)
            logger.info("Trend chart saved → %s", chart_path)

        except Exception as e:
            logger.warning("Chart generation failed: %s", e)

    def _print_summary(self, result):
        """Print a clean human-readable summary to the terminal."""
        s = result["summary"]
        print("\n" + "=" * 55)
        print(f"  VALIDATION SUMMARY")
        print("=" * 55)
        print(f"  Overall:    {result['overall']}")
        print(f"  Total GPUs: {s['total']}")
        print(f"  Passed:     {s['passed']}")
        print(f"  Failed:     {s['failed']}")
        print(f"  Pass rate:  {s['pass_rate']}")
        print(f"  Temp limit: {result['limit']}C")
        print(f"  Power limit:{result['power_limit']}W")
        if s["failed"] > 0:
            print("\n  Failed GPUs:")
            for d in result["details"]:
                if d["status"] == "FAIL":
                    print(f"    ✗ {d['name']} — {d['fail_reason']}")
        print("=" * 55 + "\n")