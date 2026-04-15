"""
Design patterns used in production validation frameworks.
These three patterns appear in almost every NVIDIA tool.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# ─── PATTERN 1: STRATEGY ────────────────────────────────────────────────────
# Define a family of algorithms, make them interchangeable.
# Use when: you want to swap out how something works without changing the caller.
# Real use: different validation strategies for different chip generations.

class ValidationStrategy(ABC):
    """Base class for all validation strategies."""

    @abstractmethod
    def validate(self, gpu: Dict, thresholds: Any) -> Dict:
        """Validate one GPU and return a result dict."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class ThermalValidation(ValidationStrategy):
    """Validate GPU temperature only."""

    @property
    def name(self):
        return "thermal"

    def validate(self, gpu: Dict, thresholds: Any) -> Dict:
        ok = gpu["temp"] <= thresholds.temp_limit_c
        return {
            "strategy":   self.name,
            "status":     "PASS" if ok else "FAIL",
            "fail_reason": f"temp {gpu['temp']}>{thresholds.temp_limit_c}C" if not ok else "none",
        }


class PowerValidation(ValidationStrategy):
    """Validate GPU power only."""

    @property
    def name(self):
        return "power"

    def validate(self, gpu: Dict, thresholds: Any) -> Dict:
        ok = gpu["power_watts"] <= thresholds.power_limit_w
        return {
            "strategy":   self.name,
            "status":     "PASS" if ok else "FAIL",
            "fail_reason": f"power {gpu['power_watts']}W>{thresholds.power_limit_w}W" if not ok else "none",
        }


class FullValidation(ValidationStrategy):
    """Validate temperature AND power — the default strategy."""

    @property
    def name(self):
        return "full"

    def validate(self, gpu: Dict, thresholds: Any) -> Dict:
        temp_ok  = gpu["temp"]        <= thresholds.temp_limit_c
        power_ok = gpu["power_watts"] <= thresholds.power_limit_w

        reasons = []
        if not temp_ok:
            reasons.append(f"temp {gpu['temp']}>{thresholds.temp_limit_c}C")
        if not power_ok:
            reasons.append(f"power {gpu['power_watts']}W>{thresholds.power_limit_w}W")

        return {
            "strategy":    self.name,
            "status":      "PASS" if temp_ok and power_ok else "FAIL",
            "fail_reason": ", ".join(reasons) if reasons else "none",
        }


def get_strategy(name: str) -> ValidationStrategy:
    """Factory function — get a strategy by name."""
    strategies = {
        "thermal": ThermalValidation(),
        "power":   PowerValidation(),
        "full":    FullValidation(),
    }
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Choose from {list(strategies.keys())}")
    return strategies[name]


# ─── PATTERN 2: FACTORY ─────────────────────────────────────────────────────
# Create objects without specifying the exact class.
# Use when: the type of object depends on input (config, CLI args, environment).
# Real use: choosing which data reader to use based on --source flag.

class GPUDataReader(ABC):
    """Abstract base class for all GPU data sources."""

    @abstractmethod
    def read(self, **kwargs) -> List[Dict]:
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass


class CSVReader(GPUDataReader):
    @property
    def source_name(self):
        return "csv"

    def read(self, filepath: str, **kwargs) -> List[Dict]:
        import csv
        metrics = []
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.append({
                    "name":            row["name"],
                    "index":           None,
                    "temp":            int(row["temp"]),
                    "power_watts":     int(row["power_watts"]),
                    "memory_used_gb":  int(row["memory_used_gb"]),
                    "source":          self.source_name,
                })
        logger.info("CSVReader: loaded %d entries", len(metrics))
        return metrics


class NVMLReader(GPUDataReader):
    @property
    def source_name(self):
        return "nvml"

    def read(self, device_indices=None, **kwargs) -> List[Dict]:
        # Delegates to the existing read_metrics_from_nvml function
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from main import read_metrics_from_nvml
        return read_metrics_from_nvml(device_indices)


def create_reader(source: str) -> GPUDataReader:
    """
    Factory function — create the right reader for the given source.
    This is the Factory pattern: caller doesn't need to know which class to use.
    """
    readers = {
        "csv":  CSVReader,
        "nvml": NVMLReader,
    }
    if source not in readers:
        raise ValueError(f"Unknown source: '{source}'. Choose from {list(readers.keys())}")
    reader = readers[source]()
    logger.debug("Factory created reader: %s", reader.source_name)
    return reader


# ─── PATTERN 3: OBSERVER ────────────────────────────────────────────────────
# Notify multiple listeners when something happens.
# Use when: different parts of the system need to react to the same event.
# Real use: when a GPU fails, notify the logger, write to file, send an alert.

class ValidationObserver(ABC):
    """Base class for anything that wants to know about validation events."""

    @abstractmethod
    def on_gpu_fail(self, gpu_name: str, fail_reason: str, run_id: int = None):
        pass

    @abstractmethod
    def on_run_complete(self, overall: str, summary: Dict):
        pass


class LoggingObserver(ValidationObserver):
    """Logs all validation events."""

    def on_gpu_fail(self, gpu_name, fail_reason, run_id=None):
        run_str = f" (run {run_id})" if run_id else ""
        logger.warning("FAIL: %s%s — %s", gpu_name, run_str, fail_reason)

    def on_run_complete(self, overall, summary):
        logger.info("Run complete: %s | pass_rate=%s | failed=%d/%d",
                    overall, summary["pass_rate"],
                    summary["failed"], summary["total"])


class FileObserver(ValidationObserver):
    """Writes failure events to a separate failures log."""

    def __init__(self, filepath: str = "results/failures.log"):
        self.filepath = filepath
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def on_gpu_fail(self, gpu_name, fail_reason, run_id=None):
        import datetime
        with open(self.filepath, "a") as f:
            ts = datetime.datetime.now().isoformat()
            f.write(f"{ts} | FAIL | {gpu_name} | {fail_reason}\n")

    def on_run_complete(self, overall, summary):
        pass  # only care about failures


class ValidationRunner:
    """
    Runs validation and notifies all registered observers.
    This ties together the Strategy and Observer patterns.
    """

    def __init__(self, strategy: ValidationStrategy):
        self.strategy  = strategy
        self.observers: List[ValidationObserver] = []

    def add_observer(self, observer: ValidationObserver):
        self.observers.append(observer)
        return self  # allow chaining: runner.add(a).add(b)

    def run(self, gpu_list: List[Dict], thresholds: Any) -> Dict:
        results = []
        for gpu in gpu_list:
            result = self.strategy.validate(gpu, thresholds)
            result["name"] = gpu["name"]
            results.append(result)

            if result["status"] == "FAIL":
                for obs in self.observers:
                    obs.on_gpu_fail(gpu["name"], result["fail_reason"])

        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = len(results) - passed
        summary = {
            "total":     len(results),
            "passed":    passed,
            "failed":    failed,
            "pass_rate": f"{round(passed/len(results)*100, 1)}%",
        }
        overall = "PASS" if failed == 0 else "FAIL"

        for obs in self.observers:
            obs.on_run_complete(overall, summary)

        return {"overall": overall, "summary": summary, "details": results}