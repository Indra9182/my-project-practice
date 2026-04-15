import pytest
import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from main import analyze_metrics, read_metrics_from_csv, build_metadata, save_results


# ─── FIXTURES ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_metrics():
    """Standard 3-GPU dataset — used across all analysis tests."""
    return [
        {"name": "GPU-0", "temp": 72, "power_watts": 280, "memory_used_gb": 18},
        {"name": "GPU-1", "temp": 68, "power_watts": 265, "memory_used_gb": 16},
        {"name": "GPU-2", "temp": 75, "power_watts": 310, "memory_used_gb": 22},
    ]

@pytest.fixture
def passing_metrics():
    """All GPUs well within limits — should always PASS."""
    return [
        {"name": "GPU-0", "temp": 60, "power_watts": 200, "memory_used_gb": 10},
        {"name": "GPU-1", "temp": 58, "power_watts": 195, "memory_used_gb": 9},
    ]

@pytest.fixture
def nvml_style_metrics():
    """Full metric dict as returned by pynvml — all fields present."""
    return [
        {
            "name": "Tesla H100", "index": 0,
            "temp": 72, "power_watts": 280.0,
            "memory_used_gb": 18.0, "memory_total_gb": 80.0,
            "gpu_util_pct": 85, "mem_util_pct": 60,
            "pcie_gen": 5, "pcie_width": 16,
            "fan_speed_pct": None, "perf_state": "0",
            "source": "nvml",
        }
    ]


# ─── ANALYSIS TESTS ─────────────────────────────────────────────────────────

def test_overall_fail_when_any_gpu_over_limit(sample_metrics):
    result = analyze_metrics(sample_metrics, limit=73, power_limit=350)
    assert result["overall"] == "FAIL"

def test_overall_pass_when_all_gpus_under_limit(sample_metrics):
    result = analyze_metrics(sample_metrics, limit=80, power_limit=350)
    assert result["overall"] == "PASS"

def test_correct_average_temp(sample_metrics):
    result = analyze_metrics(sample_metrics, limit=73, power_limit=350)
    assert result["avg_temp"] == 71.67

def test_correct_max_temp(sample_metrics):
    result = analyze_metrics(sample_metrics, limit=73, power_limit=350)
    assert result["max_temp"] == 75

def test_individual_gpu_statuses(sample_metrics):
    result = analyze_metrics(sample_metrics, limit=73, power_limit=350)
    statuses = {d["name"]: d["status"] for d in result["details"]}
    assert statuses["GPU-0"] == "PASS"
    assert statuses["GPU-1"] == "PASS"
    assert statuses["GPU-2"] == "FAIL"

def test_gpu_exactly_at_limit_is_pass():
    metrics = [{"name": "GPU-0", "temp": 73, "power_watts": 280, "memory_used_gb": 18}]
    result = analyze_metrics(metrics, limit=73, power_limit=350)
    assert result["overall"] == "PASS"

def test_result_contains_expected_keys(sample_metrics):
    result = analyze_metrics(sample_metrics, limit=73, power_limit=350)
    for key in ["avg_temp", "max_temp", "overall", "details", "limit", "summary"]:
        assert key in result

def test_summary_counts_are_correct(sample_metrics):
    result = analyze_metrics(sample_metrics, limit=73, power_limit=350)
    assert result["summary"]["total"]  == 3
    assert result["summary"]["failed"] == 1
    assert result["summary"]["passed"] == 2

def test_fail_when_power_over_limit(sample_metrics):
    result = analyze_metrics(sample_metrics, limit=100, power_limit=260)
    assert result["overall"] == "FAIL"

def test_fail_reason_shows_power_cause(sample_metrics):
    result = analyze_metrics(sample_metrics, limit=100, power_limit=260)
    gpu0 = next(d for d in result["details"] if d["name"] == "GPU-0")
    assert "power" in gpu0["fail_reason"]

def test_fail_reason_is_none_when_passing(passing_metrics):
    result = analyze_metrics(passing_metrics, limit=100, power_limit=350)
    for d in result["details"]:
        assert d["fail_reason"] == "none"

def test_nvml_style_metrics_work(nvml_style_metrics):
    result = analyze_metrics(nvml_style_metrics, limit=80, power_limit=350)
    assert result["overall"] == "PASS"
    assert result["details"][0]["pcie_gen"] == 5

@pytest.mark.parametrize("limit,expected", [
    (80,  "PASS"),
    (73,  "FAIL"),
    (68,  "FAIL"),
    (100, "PASS"),
])
def test_overall_result_at_various_limits(limit, expected, sample_metrics):
    result = analyze_metrics(sample_metrics, limit=limit, power_limit=350)
    assert result["overall"] == expected

def test_analyze_raises_on_empty_data():
    with pytest.raises(ZeroDivisionError):
        analyze_metrics([], limit=73, power_limit=300)


# ─── CSV READER TESTS ────────────────────────────────────────────────────────

def test_read_metrics_from_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "name,temp,power_watts,memory_used_gb\nGPU-0,72,280,18\n"
    )
    metrics = read_metrics_from_csv(str(csv_file))
    assert len(metrics) == 1
    assert metrics[0]["name"] == "GPU-0"
    assert metrics[0]["temp"] == 72
    assert isinstance(metrics[0]["temp"], int)

def test_read_csv_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        read_metrics_from_csv("does_not_exist.csv")


# ─── METADATA TESTS ──────────────────────────────────────────────────────────

def test_metadata_contains_required_keys():
    metadata = build_metadata("csv", "data/gpu_metrics.csv")
    for key in ["timestamp", "hostname", "python_version", "input_file", "tool_version"]:
        assert key in metadata

def test_metadata_input_file_matches():
    metadata = build_metadata("csv", "data/gpu_metrics.csv")
    assert metadata["input_file"] == "data/gpu_metrics.csv"

def test_metadata_timestamp_is_valid_iso_format():
    metadata = build_metadata("csv")
    parsed = datetime.datetime.fromisoformat(metadata["timestamp"])
    assert isinstance(parsed, datetime.datetime)

def test_metadata_hostname_is_string():
    metadata = build_metadata("nvml")
    assert isinstance(metadata["hostname"], str)
    assert len(metadata["hostname"]) > 0

def test_metadata_source_is_stored():
    metadata = build_metadata("nvidia-smi")
    assert metadata["source"] == "nvidia-smi"


# ─── OUTPUT TESTS ────────────────────────────────────────────────────────────

def test_save_results_creates_json_file(tmp_path, sample_metrics):
    result   = analyze_metrics(sample_metrics, limit=73, power_limit=350)
    metadata = build_metadata("csv", "test.csv")
    out_path = str(tmp_path / "results.json")
    save_results(result, out_path, metadata)
    assert os.path.exists(out_path)

def test_save_results_json_is_valid(tmp_path, sample_metrics):
    import json
    result   = analyze_metrics(sample_metrics, limit=73, power_limit=350)
    metadata = build_metadata("csv", "test.csv")
    out_path = str(tmp_path / "results.json")
    save_results(result, out_path, metadata)
    with open(out_path) as f:
        data = json.load(f)
    assert "metadata" in data
    assert "results"  in data
    assert data["results"]["overall"] in ["PASS", "FAIL"]

def test_save_results_prints_summary(tmp_path, sample_metrics, capsys):
    result   = analyze_metrics(sample_metrics, limit=73, power_limit=350)
    metadata = build_metadata("csv", "test.csv")
    save_results(result, str(tmp_path / "out.json"), metadata)
    captured = capsys.readouterr()
    assert "Overall" in captured.out or "FAIL" in captured.out