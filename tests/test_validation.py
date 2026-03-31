import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from main import analyze_metrics, read_metrics_from_csv

# --- sample data used across multiple tests ---
SAMPLE_METRICS = [
    {"name": "GPU-0", "temp": 72, "power_watts": 280, "memory_used_gb": 18},
    {"name": "GPU-1", "temp": 68, "power_watts": 265, "memory_used_gb": 16},
    {"name": "GPU-2", "temp": 75, "power_watts": 310, "memory_used_gb": 22},
]

# --- basic tests ---

def test_overall_fail_when_any_gpu_over_limit():
    result = analyze_metrics(SAMPLE_METRICS, limit=73)
    assert result["overall"] == "FAIL"

def test_overall_pass_when_all_gpus_under_limit():
    result = analyze_metrics(SAMPLE_METRICS, limit=80)
    assert result["overall"] == "PASS"

def test_correct_average_temp():
    result = analyze_metrics(SAMPLE_METRICS, limit=73)
    assert result["avg_temp"] == 71.67

def test_correct_max_temp():
    result = analyze_metrics(SAMPLE_METRICS, limit=73)
    assert result["max_temp"] == 75

def test_individual_gpu_statuses():
    result = analyze_metrics(SAMPLE_METRICS, limit=73)
    statuses = {d["name"]: d["status"] for d in result["details"]}
    assert statuses["GPU-0"] == "PASS"  # 72 <= 73
    assert statuses["GPU-1"] == "PASS"  # 68 <= 73
    assert statuses["GPU-2"] == "FAIL"  # 75 > 73

def test_gpu_exactly_at_limit_is_pass():
    metrics = [{"name": "GPU-0", "temp": 73, "power_watts": 280, "memory_used_gb": 18}]
    result = analyze_metrics(metrics, limit=73)
    assert result["overall"] == "PASS"

def test_result_contains_expected_keys():
    result = analyze_metrics(SAMPLE_METRICS, limit=73)
    assert "avg_temp" in result
    assert "max_temp" in result
    assert "overall" in result
    assert "details" in result
    assert "limit" in result

# --- parametrize: test many limits in one go ---

@pytest.mark.parametrize("limit,expected", [
    (80, "PASS"),   # all GPUs under 80
    (73, "FAIL"),   # GPU-2 at 75 fails
    (68, "FAIL"),   # GPU-1 at 68 fails — exactly at limit passes, but GPU-0 and GPU-2 fail
    (100, "PASS"),  # everything passes
])
def test_overall_result_at_various_limits(limit, expected):
    result = analyze_metrics(SAMPLE_METRICS, limit=limit)
    assert result["overall"] == expected

# --- CSV reader test ---

def test_read_metrics_from_csv(tmp_path):
    csv_file = tmp_path / "test_metrics.csv"
    csv_file.write_text("name,temp,power_watts,memory_used_gb\nGPU-0,72,280,18\n")
    metrics = read_metrics_from_csv(str(csv_file))
    assert len(metrics) == 1
    assert metrics[0]["name"] == "GPU-0"
    assert metrics[0]["temp"] == 72
    assert isinstance(metrics[0]["temp"], int)